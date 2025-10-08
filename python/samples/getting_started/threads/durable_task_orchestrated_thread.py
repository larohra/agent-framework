# Copyright (c) Microsoft. All rights reserved.

"""Durable Task Scheduler orchestration of an Agent Framework thread.

This sample demonstrates how to orchestrate an Agent Framework conversation thread using
Azure Durable Task Scheduler (DTS). Each user turn is scheduled as a durable activity
that executes the agent, persists the serialized thread state, and then checkpoints the
orchestrator. Durable Task Scheduler provides:

* State durability across orchestrator/worker restarts
* Serverless scale-out for activities that execute agent logic
* Automatic recovery from the last successful checkpoint when failures occur

Prerequisites
-------------

1. Deploy the Durable Task Scheduler (preview) and note its gRPC endpoint and task hub.
   See https://learn.microsoft.com/azure/azure-functions/durable/durable-task-scheduler/.
2. Install dependencies in your Python environment:

   ```bash
   pip install durabletask durabletask-azuremanaged azure-identity agent-framework-core --pre
   ```

3. Set the following environment variables so the SDK can connect securely. Managed
   identities are recommended when running in Azure; locally you can use `az login`.

   ```text
   DTS_HOST_ADDRESS    # e.g., "contoso-dts.eastus2-1.ts.svc.azure.com:443"
   DTS_TASK_HUB        # e.g., "chat-agents"
   AZURE_CLIENT_ID     # if you are using a user-assigned managed identity
   ```

   The `DefaultAzureCredential` chain is used, so developer flows such as Azure CLI
   authentication or Visual Studio Code sign-in also work.

4. Configure whichever chat client you choose (OpenAI, Azure OpenAI, etc.). This sample
   uses the OpenAI client for brevity. Refer to
   `python/samples/getting_started/agents/openai/README.md` for environment variables.

Running the sample
------------------

1. Start a Durable Task Scheduler worker sidecar that targets your task hub.
2. Execute this script. It registers an orchestrator plus activity with the DTS worker,
   schedules an orchestration instance, waits for completion, and prints the agent
   conversation history captured in durable state.

Note: The worker runs on a background thread inside this process to keep the sample
self-contained. In production you would normally host workers separately (for example,
inside Azure Container Apps or Functions) and scale them independently of clients.
"""

from __future__ import annotations

from agent_framework._threads import AgentThreadState
import dotenv
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Collection

from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import  DurableTaskSchedulerWorker
from durabletask import task as d_task
from durabletask.task import OrchestrationContext
from durabletask.worker import TaskHubGrpcWorker
from pydantic import BaseModel

from agent_framework import ChatMessage, ChatMessageStoreProtocol
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.observability import get_tracer, setup_observability
from opentelemetry import trace
from opentelemetry.trace.span import format_trace_id

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("durable_task_thread_sample")

dotenv.load_dotenv()

@dataclass
class SchedulerSettings:
    """Runtime configuration of the Durable Task Scheduler backend."""

    host_address: str
    task_hub: str


def load_scheduler_settings() -> SchedulerSettings:
    """Load DTS host metadata from environment variables."""

    host = os.getenv("DTS_HOST_ADDRESS")
    task_hub = os.getenv("DTS_TASK_HUB")
    if not host or not task_hub:
        raise EnvironmentError(
            "Set DTS_HOST_ADDRESS and DTS_TASK_HUB environment variables before running the sample."
        )
    return SchedulerSettings(host_address=host, task_hub=task_hub)


class DurableChatStoreState(BaseModel):
    instance_id: str


class DurableTaskChatMessageStore(ChatMessageStoreProtocol):
    """Chat message store that persists messages via Durable Task Scheduler."""

    def __init__(
        self,
        client: DurableTaskSchedulerClient,
        orchestrator_name: str,
        instance_id: str | None = None,
    ) -> None:
        self._client = client
        self._orchestrator_name = orchestrator_name
        self._instance_id = instance_id or self._start_new_instance()

    def _start_new_instance(self) -> str:
        LOGGER.info("Creating new chat store orchestrator using '%s'", self._orchestrator_name)
        instance_id = self._client.schedule_new_orchestration(
            self._orchestrator_name,
            input={"messages": []},
        )
        return instance_id

    @property
    def instance_id(self) -> str:
        return self._instance_id

    async def add_messages(self, messages: Collection[ChatMessage]) -> None:  # noqa: D401
        payload = {
            "messages": [message.to_json() for message in messages],
        }
        LOGGER.info("Appending %d message(s) to orchestrator %s", len(payload["messages"]), self._instance_id)
        self._client.raise_orchestration_event(
            instance_id=self._instance_id,
            event_name=_EVENT_CHANNEL,
            data=payload,
        )

    async def list_messages(self) -> list[ChatMessage]:
        state = self._client.get_orchestration_state(instance_id=self._instance_id, fetch_payloads=True)
        if not state or not state.serialized_custom_status:
            return []
        data = json.loads(state.serialized_custom_status)
        serialized_messages = data.get("messages", [])
        return [ChatMessage.from_json(msg) for msg in serialized_messages]

    async def deserialize_state(self, serialized_store_state: Any, **kwargs: Any) -> None:
        if serialized_store_state:
            state = DurableChatStoreState.model_validate(serialized_store_state, **kwargs)
            self._instance_id = state.instance_id

    async def serialize_state(self, **kwargs: Any) -> Any:
        return DurableChatStoreState(instance_id=self._instance_id).model_dump(**kwargs)

    async def update_from_state(self, serialized_store_state: Any, **kwargs: Any) -> None:
        await self.deserialize_state(serialized_store_state, **kwargs)


def create_agent():
    """Create the agent used by the activity function.

    The OpenAI chat client is used here; swap in AzureOpenAIChatClient or others as needed.
    """

    chat_client = AzureOpenAIChatClient()
    agent = chat_client.create_agent(
        name="Durable-Orchestrated-Agent",
        instructions=(
            "You are a witty assistant who answers succinctly. "
            "Always acknowledge that responses are orchestrated by Durable Task Scheduler."
        ),
    )
    return agent


async def run_agent_turn_async(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one agent turn and return the updated thread state plus the assistant reply."""

    agent = create_agent()

    serialized_thread = payload.get("thread_state")
    user_message = payload["message"]

    if serialized_thread:
        thread = await agent.deserialize_thread(serialized_thread)
    else:
        thread = agent.get_new_thread()

    reply = await agent.run(user_message, thread=thread)

    # Persist checkpoint so orchestrator can resume from the last processed turn.
    updated_state = AgentThreadState(
            service_thread_id=thread.service_thread_id
        ).model_dump_json()
    # updated_state = await thread.serialize()

    return {
        "assistant_reply": reply.text,
        "thread_state": updated_state,
    }


def agent_turn_activity(context: d_task.ActivityContext, payload: dict[str, Any]) -> dict[str, Any]:
    """Activity function that runs the agent in a durable, retry-safe way."""

    LOGGER.info(
        "Activity %s processing orchestration %s", context.task_id, context.orchestration_id
    )
    return asyncio.run(run_agent_turn_async(payload))


def agent_thread_orchestrator(context: OrchestrationContext, input_: dict[str, Any]):
    """Durable orchestrator that drives an Agent Framework thread.

    Each user message is processed by scheduling the agent activity. After the turn
    completes, the orchestrator checkpoints by continuing as new with the updated state.
    """

    pending: list[str] = list(input_.get("pending_messages", []))
    serialized_thread = input_.get("thread_state")
    transcript: list[dict[str, str]] = list(input_.get("transcript", []))

    if not pending:
        # Nothing to do: return any final state to the caller.
        return {
            "transcript": transcript,
            "thread_state": serialized_thread,
        }

    next_message = pending.pop(0)
    activity_name = d_task.get_name(agent_turn_activity)
    activity_result = yield context.call_activity(
        activity_name,
        input={
            "thread_state": serialized_thread,
            "message": next_message,
        },
    )

    assistant_reply = activity_result["assistant_reply"]
    updated_state = activity_result["thread_state"]

    transcript.append({"user": next_message, "assistant": assistant_reply})

    # Keep the most recent transcript in custom status for live monitoring.
    context.set_custom_status({"transcript": transcript})

    if pending:
        # Continue the orchestration with remaining messages. save_events=True ensures
        # any outstanding external events (e.g., signals) are preserved across rewinds.
        context.continue_as_new(
            {
                "pending_messages": pending,
                "thread_state": updated_state,
                "transcript": transcript,
            },
            save_events=True,
        )
        return None

    # Completed all messages.
    return {
        "transcript": transcript,
        "thread_state": updated_state,
    }


def _ensure_worker_registration(worker: TaskHubGrpcWorker) -> tuple[str, str]:
    """Register orchestrator and activity functions and return orchestrator name."""

    orchestrator_name = worker.add_orchestrator(agent_thread_orchestrator)
    activity_name = worker.add_activity(agent_turn_activity)
    LOGGER.info(
        "Registered orchestrator '%s' and activity '%s'",
        orchestrator_name,
        activity_name,
    )
    return orchestrator_name, activity_name


def demo_conversation() -> Iterable[str]:
    """Static set of prompts demonstrating multi-turn orchestration."""

    return [
        "Tell me a one-sentence fact about the Apollo program.",
        "Summarize it in five words.",
        "Respond with a pirate-style farewell.",
    ]


def main() -> None:
    settings = load_scheduler_settings()
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

    worker = DurableTaskSchedulerWorker(
        host_address=settings.host_address,
        taskhub=settings.task_hub,
        token_credential=credential,
    )

    orchestrator_name, _ = _ensure_worker_registration(worker)

    LOGGER.info("Connecting to Durable Task Scheduler at %s (task hub: %s)", settings.host_address, settings.task_hub)

    try:
        worker.start()

        client = DurableTaskSchedulerClient(
            host_address=settings.host_address,
            taskhub=settings.task_hub,
            token_credential=credential,
        )

        instance_id = client.schedule_new_orchestration(
            orchestrator_name,
            input={
                "pending_messages": list(demo_conversation()),
            },
        )
        LOGGER.info("Started orchestration %s", instance_id)

        state = client.wait_for_orchestration_completion(instance_id, fetch_payloads=True, timeout=300)
        if not state:
            LOGGER.error("Orchestration %s not found", instance_id)
            return

        state.raise_if_failed()

        if state.serialized_output:
            result = json.loads(state.serialized_output)
            LOGGER.info("Final transcript: %s", json.dumps(result["transcript"], indent=2))
        else:
            LOGGER.warning("No output returned from orchestration %s", instance_id)

    finally:
        worker.stop()
        LOGGER.info("Worker stopped.")


if __name__ == "__main__":
    setup_observability(applicationinsights_connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"))
    with get_tracer().start_as_current_span("Durable Orchestrator as Thread", kind=trace.SpanKind.CLIENT) as current_span:
        print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")
        main()
