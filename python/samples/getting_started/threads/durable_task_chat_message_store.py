# Copyright (c) Microsoft. All rights reserved.

"""Durable Task-backed ChatMessageStore implementation.

This sample shows how to back the Agent Framework `ChatMessageStoreProtocol` with an
Azure Durable Task Scheduler orchestration. Each conversation thread owns a dedicated
Durable Task orchestration instance that manages message persistence. The benefits are:

* Distributed, durable storage of chat history managed by the scheduler
* Automatic recovery of the message log after worker restarts or failures
* Elastic scale-out of message ingestion through Durable activities/events

The orchestration keeps the canonical message list in its custom status payload. Agents
append messages by raising events, and readers call `get_orchestration_state` to fetch
materialized history. Because the scheduler checkpoints orchestration state, message
history survives process restarts without custom database plumbing.

Prerequisites are the same as `durable_task_orchestrated_thread.py`:

1. Deploy Azure Durable Task Scheduler and obtain the gRPC endpoint + task hub.
2. Install packages:

   ```bash
   pip install durabletask durabletask-azuremanaged azure-identity agent-framework-core --pre
   ```

3. Set `DTS_HOST_ADDRESS` and `DTS_TASK_HUB` environment variables plus credentials
   compatible with `DefaultAzureCredential` (managed identity or `az login`).
4. Configure your preferred chat client credentials (OpenAI shown). See
   `python/samples/getting_started/agents/openai/README.md` for details.

Running the sample:

* Start a Durable Task Scheduler worker sidecar targeting your task hub.
* Execute this script. It registers a message-log orchestrator, constructs an agent
  whose `ChatMessageStore` pushes messages through the scheduler, performs two turns,
  and prints the persisted message history returned from Durable Task Scheduler.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import dotenv
from dataclasses import dataclass
from typing import Any, Collection

from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import  DurableTaskSchedulerWorker
from durabletask.task import OrchestrationContext
from durabletask.worker import TaskHubGrpcWorker
from pydantic import BaseModel

from agent_framework import ChatMessage, ChatMessageStoreProtocol
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.observability import get_tracer, setup_observability
from opentelemetry import trace
from opentelemetry.trace.span import format_trace_id

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("durable_task_chat_store_sample")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    LOGGER.addHandler(console_handler)
LOGGER.propagate = False

dotenv.load_dotenv()

_EVENT_CHANNEL = "append_messages"


@dataclass
class SchedulerSettings:
    host_address: str
    task_hub: str


def load_scheduler_settings() -> SchedulerSettings:
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


def chat_message_store_orchestrator(context: OrchestrationContext, state: dict[str, Any] | None):
    """Durable orchestrator that maintains chat history in its custom status."""

    state = state or {"messages": []}
    messages: list[dict[str, Any]] = list(state.get("messages", []))
    context.set_custom_status({"messages": messages, "message_count": len(messages)})

    event = yield context.wait_for_external_event(_EVENT_CHANNEL)
    new_messages = event.get("messages", []) if isinstance(event, dict) else []
    if new_messages:
        messages.extend(new_messages)
        context.set_custom_status({"messages": messages, "message_count": len(messages)})

    context.continue_as_new({"messages": messages}, save_events=True)


def _register_worker_components(worker: TaskHubGrpcWorker) -> str:
    orchestrator_name = worker.add_orchestrator(chat_message_store_orchestrator)
    LOGGER.info("Registered message store orchestrator '%s'", orchestrator_name)
    return orchestrator_name


def create_agent_with_store(
    client: DurableTaskSchedulerClient,
    orchestrator_name: str,
):
    """Factory that wires the durable chat store into a new agent instance."""

    def factory() -> DurableTaskChatMessageStore:
        return DurableTaskChatMessageStore(client=client, orchestrator_name=orchestrator_name)

    chat_client = AzureOpenAIChatClient()
    return chat_client.create_agent(
        name="Durable-ChatStore-Agent",
        instructions=(
            "You are a concise assistant. Mention how many messages are currently persisted "
            "each time you reply."
        ),
        chat_message_store_factory=factory,
    )


async def main() -> None:
    settings = load_scheduler_settings()
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

    worker = DurableTaskSchedulerWorker(
        host_address=settings.host_address,
        taskhub=settings.task_hub,
        token_credential=credential,
    )
    orchestrator_name = _register_worker_components(worker)

    try:
        worker.start()

        client = DurableTaskSchedulerClient(
            host_address=settings.host_address,
            taskhub=settings.task_hub,
            token_credential=credential,
        )
        agent = create_agent_with_store(client, orchestrator_name)
        thread = agent.get_new_thread()

        for question in (
            "Log our first note about the project status.",
            "Add another bullet summarizing next steps.",
        ):
            LOGGER.info("User: %s", question)
            reply = await agent.run(question, thread=thread)
            LOGGER.info("Agent: %s", reply.text)

        store = thread.message_store
        if not isinstance(store, DurableTaskChatMessageStore):
            raise RuntimeError("Thread is not using DurableTaskChatMessageStore as expected.")
        messages = await store.list_messages()
        LOGGER.info("Persisted messages (%d):", len(messages))
        for msg in messages:
            LOGGER.info("  - [%s] %s", msg.role, msg.text)

    finally:
        worker.stop()
        LOGGER.info("Worker stopped")


if __name__ == "__main__":
    setup_observability(applicationinsights_connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"))
    with get_tracer().start_as_current_span("Durable Orchestrator as Thread", kind=trace.SpanKind.CLIENT) as current_span:
        print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")        
        asyncio.run(main())
