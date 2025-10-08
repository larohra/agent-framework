# Copyright (c) Microsoft. All rights reserved.

"""Durable Task chat store with orchestrated activity processing.

This sample extends the basic Durable Task chat-message store by routing each append
request through a Durable Task activity function. The orchestrator captures the
message log, timestamps, and activity metadata in its custom status payload so any
worker can reconstruct the conversation state.

Benefits of this pattern:

* Activities can perform additional computation (validation, enrichment, fan-out)
  before the orchestrator persists results.
* Durable Task persistence keeps message history safe across worker restarts.
* Diagnostics improve because orchestrator custom status includes both messages and
  activity outputs.

Prerequisites mirror other Durable Task samples in this folder:

1. Provision Azure Durable Task Scheduler (preview) and note its gRPC endpoint plus
   task hub name.
2. Install dependencies:

   ```bash
   pip install durabletask durabletask-azuremanaged azure-identity agent-framework-core --pre
   ```

3. Configure authentication compatible with ``DefaultAzureCredential`` and export
   ``DTS_HOST_ADDRESS`` / ``DTS_TASK_HUB``. Provide OpenAI credentials for
   ``OpenAIChatClient``.
4. Start a Durable Task Scheduler worker sidecar before running this script.

At runtime the sample:

1. Registers an orchestrator that waits for message append events, calls an activity to
   transform the payload, then continues-as-new with the updated state.
2. Implements ``DurableTaskActivityChatMessageStore`` which raises events to the
   orchestrator and reads back persisted message history.
3. Runs a quick conversation via an Agent Framework agent and prints the recorded
   history, including activity metadata stored by the orchestrator.
"""

from __future__ import annotations
from uuid import uuid4

import dotenv
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection

from agent_framework import ChatMessage, ChatMessageStoreProtocol
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from durabletask.task import ActivityContext, OrchestrationContext
from durabletask.worker import TaskHubGrpcWorker
from pydantic import BaseModel

dotenv.load_dotenv()
LOG_FILE = Path(__file__).with_name("durable_task_activity_chat_store.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(threadName)s %(thread)d]: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w")],
    force=True,
)

LOGGER = logging.getLogger("durable_task_activity_chat_store_sample")
LOGGER.setLevel(logging.INFO)

WORKER_LOGGER = logging.getLogger("durabletask-worker")
WORKER_LOGGER.setLevel(logging.INFO)
WORKER_LOGGER.handlers.clear()
WORKER_LOGGER.propagate = True

CLIENT_LOGGER = logging.getLogger("durabletask-client")
CLIENT_LOGGER.setLevel(logging.INFO)
CLIENT_LOGGER.handlers.clear()
CLIENT_LOGGER.propagate = True

_ORCHESTRATOR_EVENT = "append_via_activity"
_ACTIVITY_NAME = "append_messages_activity"

LOGGER.info("Durable Task activity chat sample writing logs to %s", LOG_FILE)


@dataclass
class SchedulerSettings:
    host_address: str
    task_hub: str


def load_scheduler_settings() -> SchedulerSettings:
    host = os.getenv("DTS_HOST_ADDRESS")
    hub = os.getenv("DTS_TASK_HUB")
    if not host or not hub:
        raise EnvironmentError("Set DTS_HOST_ADDRESS and DTS_TASK_HUB before running the sample.")
    return SchedulerSettings(host_address=host, task_hub=hub)


class DurableActivityChatStoreState(BaseModel):
    instance_id: str


class DurableTaskActivityChatMessageStore(ChatMessageStoreProtocol):
    """Chat message store that routes writes through a Durable Task activity."""

    def __init__(
        self,
        client: DurableTaskSchedulerClient,
        orchestrator_name: str,
        *,
        instance_id: str | None = None,
    ) -> None:
        self._client = client
        self._orchestrator_name = orchestrator_name
        self._instance_id = instance_id or self._start_new_instance()

    def _start_new_instance(self) -> str:
        LOGGER.info("Starting Durable orchestrator '%s' for chat store", self._orchestrator_name)
        instance_id = self._client.schedule_new_orchestration(
            self._orchestrator_name,
            input={"messages": [], "activity_runs": []},
        )
        return instance_id

    @property
    def instance_id(self) -> str:
        return self._instance_id

    async def add_messages(self, messages: Collection[ChatMessage]) -> None:
        if not messages:
            return
        
        payload_id = uuid4().hex
        payload = {
            "messages": [message.to_json() for message in messages],
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "payload_id": payload_id,
        }
        LOGGER.info(
            "Queueing %d message(s) with payload ID %s to orchestrator %s", len(payload["messages"]), payload_id, self._instance_id
        )
        try:
            await asyncio.to_thread(
                self._client.raise_orchestration_event,
                self._instance_id,
                _ORCHESTRATOR_EVENT,
                data=payload,
            )
        except Exception as exc:
            LOGGER.exception("Failed to dispatch Durable event", exc_info=exc)
            raise RuntimeError("Durable Task event dispatch failed") from exc

    async def list_messages(self) -> list[ChatMessage]:
        state = await asyncio.to_thread(
            self._client.get_orchestration_state,
            self._instance_id,
            fetch_payloads=True,
        )
        if not state or not state.serialized_custom_status:
            return []
        payload = json.loads(state.serialized_custom_status)
        entries = payload.get("messages", [])
        LOGGER.info("Retrieved %d messages from orchestrator %s: %s", len(entries), self._instance_id, entries)
        return [ChatMessage.from_dict(msg) if isinstance(msg, dict) else ChatMessage.from_json(msg) for msg in entries]

    async def deserialize_state(self, serialized_store_state: Any, **kwargs: Any) -> None:
        if serialized_store_state:
            restored = DurableActivityChatStoreState.model_validate(serialized_store_state, **kwargs)
            self._instance_id = restored.instance_id

    async def serialize_state(self, **kwargs: Any) -> Any:
        return DurableActivityChatStoreState(instance_id=self._instance_id).model_dump(**kwargs)

    async def update_from_state(self, serialized_store_state: Any, **kwargs: Any) -> None:
        await self.deserialize_state(serialized_store_state, **kwargs)


def chat_store_orchestrator(context: OrchestrationContext, state: dict[str, Any] | None):
    """Durable orchestrator that persists chat messages and activity metadata."""

    LOGGER.info("Orchestrator %s started, state: %s", context.instance_id, state)
    state = state or {"messages": [], "activity_runs": []}
    messages: list[dict[str, Any]] = list(state.get("messages", []))
    activity_runs: list[dict[str, Any]] = list(state.get("activity_runs", []))

    context.set_custom_status({"messages": messages, "activity_runs": activity_runs})

    event = yield context.wait_for_external_event(_ORCHESTRATOR_EVENT)
    LOGGER.info("Orchestrator %s received event: %s", context.instance_id, event)
    if not isinstance(event, dict) or "messages" not in event:
        state["last_error"] = {"received": event}
        context.set_custom_status(state)
        context.continue_as_new(state, save_events=True)
        return

    run_payload = {
        "messages": event["messages"],
        "requested_at": event.get("requested_at"),
    }
    result = yield context.call_activity(_ACTIVITY_NAME, input=run_payload)
    LOGGER.info("Activity %s completed with result: %s", _ACTIVITY_NAME, result)

    new_messages = result.get("messages", []) if isinstance(result, dict) else []
    if new_messages:
        messages.extend(new_messages)
    activity_runs.append(result)

    state = {"messages": messages, "activity_runs": activity_runs}
    context.set_custom_status(state)
    context.continue_as_new(state, save_events=True)


def append_messages_activity(
    context: Any,
    payload: dict[str, Any] | str | None,
) -> dict[str, Any]:
    """Durable activity that enriches message payloads before persistence."""

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Activity payload was not valid JSON") from exc

    payload = payload or {}
    if not isinstance(payload, dict):
        raise TypeError("Activity payload must deserialize to a mapping")

    requested_at = payload.get("requested_at")
    raw_messages = payload.get("messages", [])

    normalized_messages: list[dict[str, Any]] = []
    for entry in raw_messages:
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                raise ValueError("Message payload was not valid JSON") from exc
        if not isinstance(entry, dict):
            raise TypeError("Each message entry must deserialize to a mapping")
        normalized_messages.append(entry)

    # Attach activity metadata for diagnostics.
    enriched = []
    for item in normalized_messages:
        enriched.append({
            **item,
            "metadata": {
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "requested_at": requested_at,
                "activity": _ACTIVITY_NAME,
            },
        })

    return {
        "messages": enriched,
        "activity_id": _ACTIVITY_NAME,
        "requested_at": requested_at,
    }

def _register_worker(worker: TaskHubGrpcWorker) -> str:
    orchestrator_name = worker.add_orchestrator(chat_store_orchestrator)
    worker.add_activity(append_messages_activity)
    LOGGER.info("Registered orchestrator '%s' and activity '%s'", orchestrator_name, _ACTIVITY_NAME)
    return orchestrator_name


def create_agent(orchestrator_name: str, client: DurableTaskSchedulerClient):
    """Create an agent using the durable activity-backed chat store."""

    def factory() -> DurableTaskActivityChatMessageStore:
        return DurableTaskActivityChatMessageStore(client, orchestrator_name)

    chat_client = AzureOpenAIChatClient()
    return chat_client.create_agent(
        name="Durable-Activity-ChatStore-Agent",
        instructions=(
            "You are a short, state-aware assistant. Mention that Durable Task activities handle your "
            "messages and include the total stored message count so far."
            " Keep your replies very concise (less than 15 words each reply)."
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

    orchestrator_name = _register_worker(worker)

    try:
        worker.start()
        client = DurableTaskSchedulerClient(
            host_address=settings.host_address,
            taskhub=settings.task_hub,
            token_credential=credential,
        )

        agent = create_agent(orchestrator_name, client)
        thread = agent.get_new_thread()

        prompts = [
            "Log our first sprint update.",
            "Add a second note about the upcoming release.",
        ]
        for prompt in prompts:
            LOGGER.info("User Prompt: %s", prompt)
            reply = await agent.run(prompt, thread=thread)
            LOGGER.info("Agent Response: %s", getattr(reply, "text", reply))

        store = thread.message_store
        if not isinstance(store, DurableTaskActivityChatMessageStore):
            raise RuntimeError("Thread is not using DurableTaskActivityChatMessageStore as expected.")

        messages = await store.list_messages()
        LOGGER.info("Persisted message count: %d", len(messages))
        for msg in messages:
            LOGGER.info("[%s] %s", msg.role, msg.text)

        state = await asyncio.to_thread(
            client.get_orchestration_state,
            store.instance_id,
            fetch_payloads=True,
        )
        if state and state.serialized_custom_status:
            activity_runs = json.loads(state.serialized_custom_status).get("activity_runs", [])
            LOGGER.info("Activity run metadata: %s", json.dumps(activity_runs, indent=2))

    finally:
        worker.stop()
        LOGGER.info("Durable worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
