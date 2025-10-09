# Copyright (c) Microsoft. All rights reserved.

"""Multi-agent orchestration over a shared Durable Task-backed thread.

This sample builds on ``durable_task_activity_chat_message_store.py`` and showcases how
multiple Agent Framework agents, each with their own toolsets, can collaborate using a
common ``AgentThread`` whose persistence is handled by Azure Durable Task Scheduler.

The scenario simulates a release readiness huddle:

* ``Operations`` agent triages deployment readiness data and assigns follow-up actions.
* ``Research`` agent analyzes qualitative signals to highlight risks and mitigation ideas.
* Both agents call bespoke tools, yet the conversation history lives in a single Durable
  Task orchestration instance so every participant (and observer) has a source of truth.

Key takeaways:

* Multiple agents can share the same ``AgentThread`` (and message store) safely.
* Durable Task Scheduler provides resilience across worker restarts and enriches
  telemetry via activity metadata.
* Tools remain scoped to each agent; the thread only tracks the resulting conversation
  and metadata, making it easy to add or remove specialists.

Prerequisites mirror the other Durable Task samples in this folder:

1. Provision Azure Durable Task Scheduler (preview) and capture the gRPC endpoint plus
   task hub name.
2. Install dependencies:

   ```bash
   pip install durabletask durabletask-azuremanaged azure-identity agent-framework-core --pre
   ```

3. Configure authentication compatible with ``DefaultAzureCredential`` and export
   ``DTS_HOST_ADDRESS`` / ``DTS_TASK_HUB``. Provide OpenAI credentials for
   ``AzureOpenAIChatClient``.
4. Start a Durable Task Scheduler worker sidecar before running this script.

"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Collection
from uuid import uuid4

import dotenv
from agent_framework import ChatMessage, ChatMessageStoreProtocol
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from durabletask.task import OrchestrationContext, when_any
from durabletask.worker import TaskHubGrpcWorker
from pydantic import BaseModel

dotenv.load_dotenv()
LOG_FILE = Path(__file__).with_name("durable_task_multi_agent_thread.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(threadName)s %(thread)d %(funcName)s]: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")],
    force=True,
)

LOGGER = logging.getLogger("durable_task_multi_agent_thread")
LOGGER.setLevel(logging.INFO)
LOGGER.info("Durable Task multi-agent sample writing logs to %s", LOG_FILE)

_ORCHESTRATOR_EVENT = "append_via_activity"
_ACTIVITY_NAME = "append_messages_activity"


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
            "Queueing %d message(s) with payload ID %s to orchestrator %s",
            len(payload["messages"]),
            payload_id,
            self._instance_id,
        )
        try:
            await asyncio.to_thread(
                self._client.raise_orchestration_event,
                self._instance_id,
                event_name=_ORCHESTRATOR_EVENT,
                data=payload,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
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
        LOGGER.info(
            "Retrieved %d messages from orchestrator %s",
            len(entries),
            self._instance_id,
        )
        try:
            return [
                ChatMessage.from_dict(msg) if isinstance(msg, dict) else ChatMessage.from_json(msg)
                for msg in entries
            ]
        except Exception as exc:  # pragma: no cover - defensive logging path
            LOGGER.exception("Failed to deserialize messages", exc_info=exc)
        return []

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

    timeout_deadline = context.current_utc_datetime + timedelta(seconds=10)
    timeout_task = context.create_timer(timeout_deadline)

    event_task = context.wait_for_external_event(_ORCHESTRATOR_EVENT)

    winner_task = yield when_any([event_task, timeout_task])

    result: dict[str, Any] = {}
    if winner_task == event_task:
        event = yield event_task
        LOGGER.info("Orchestrator %s received event: %s", context.instance_id, event)
        if not isinstance(event, dict) or "messages" not in event:
            state["last_error"] = {"received": event}
            context.set_custom_status(state)
            context.continue_as_new(state, save_events=True)
            return

        run_payload = {
            "messages": event["messages"],
            "requested_at": event.get("requested_at"),
            "payload_id": event.get("payload_id"),
        }
        result = yield context.call_activity(_ACTIVITY_NAME, input=run_payload)
        LOGGER.info("Activity %s completed with result: %s", _ACTIVITY_NAME, result)

        new_messages = result.get("messages", []) if isinstance(result, dict) else []
        if new_messages:
            messages.extend(new_messages)
        activity_runs.extend(
            {
                "role": message.get("role"),
                "metadata": message.get("metadata"),
            }
            for message in result.get("messages", [])
            if message.get("role") is not None or message.get("metadata") is not None
        )

        state = {"messages": messages, "activity_runs": activity_runs}
        context.set_custom_status(state)
        context.continue_as_new(state, save_events=True)
    else:
        LOGGER.info("Orchestrator %s timed out waiting for event", context.instance_id)
        result = {
            "status": "Timeout",
            "timed_out_at": context.current_utc_datetime.isoformat(),
        }
        return result


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

    enriched: list[dict[str, Any]] = []
    for item in normalized_messages:
        enriched.append(
            {
                **item,
                "metadata": {
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "requested_at": requested_at,
                    "activity": payload.get("payload_id"),
                },
            }
        )

    return {
        "messages": enriched,
        "activity_name": _ACTIVITY_NAME,
        "requested_at": requested_at,
    }


def _register_worker(worker: TaskHubGrpcWorker) -> str:
    orchestrator_name = worker.add_orchestrator(chat_store_orchestrator)
    worker.add_activity(append_messages_activity)
    LOGGER.info("Registered orchestrator '%s' and activity '%s'", orchestrator_name, _ACTIVITY_NAME)
    return orchestrator_name


# ---------------------------------------------------------------------------
# Tool definitions for the agents
# ---------------------------------------------------------------------------

def fetch_release_metrics() -> str:
    """Return current rollout metrics for tooling demonstrations."""

    snapshot = {
        "automated_tests_passed": 184,
        "automated_tests_total": 186,
        "open_blockers": 1,
        "deploy_windows": ["us-east", "eu-west"],
    }
    LOGGER.info("Tool fetch_release_metrics invoked: %s", snapshot)
    return (
        "Automated suite at 98.9% pass rate, single blocker in EU-West."
        " Windows available: US-East and EU-West tonight."
    )


def list_open_blockers() -> str:
    """Summarize active blocking issues tracked by operations."""

    blockers = [
        "Memory leak in ingestion service under high load (fix validated, awaiting canary)",
        "Documentation for rollback steps needs final signoff",
    ]
    LOGGER.info("Tool list_open_blockers invoked: %s", blockers)
    return " | ".join(blockers)


def synthesize_customer_sentiment(channel: str = "community") -> str:
    """Return a short sentiment snapshot for the requested feedback channel."""

    summaries = {
        "community": "Community threads are cautiously optimistic; primary worry is ingestion stability.",
        "support": "Support tickets trend downward 12%, but upgrades still spike after regional deploys.",
        "social": "Social chatter highlights excitement for analytics refresh; no new risks flagged.",
    }
    result = summaries.get(channel.lower(), summaries["community"])
    LOGGER.info("Tool synthesize_customer_sentiment invoked with channel=%s => %s", channel, result)
    return result


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def create_agent(
    *,
    chat_client: AzureOpenAIChatClient,
    orchestrator_name: str,
    scheduler_client: DurableTaskSchedulerClient,
    name: str,
    instructions: str,
    tools: list[Any],
):
    """Create an agent bound to the Durable Task-backed message store."""

    def factory() -> DurableTaskActivityChatMessageStore:
        return DurableTaskActivityChatMessageStore(scheduler_client, orchestrator_name, instance_id=None)

    return chat_client.create_agent(
        name=name,
        instructions=instructions,
        tools=tools,
        chat_message_store_factory=factory,
    )


async def run_turn(agent, prompt: str, thread) -> Any:
    LOGGER.info("%s turn. Prompt: %s", agent.name, prompt)
    result = await agent.run(prompt, thread=thread)
    LOGGER.info("%s response: %s", agent.name, getattr(result, "text", result))
    return result


async def main() -> None:
    settings = load_scheduler_settings()
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

    worker = DurableTaskSchedulerWorker(
        host_address=settings.host_address,
        taskhub=settings.task_hub,
        token_credential=credential,
    )

    orchestrator_name = _register_worker(worker)

    termination_instance_id: str | None = None
    client: DurableTaskSchedulerClient | None = None

    try:
        worker.start()
        client = DurableTaskSchedulerClient(
            host_address=settings.host_address,
            taskhub=settings.task_hub,
            token_credential=credential,
        )

        chat_client = AzureOpenAIChatClient()

        ops_agent = create_agent(
            chat_client=chat_client,
            orchestrator_name=orchestrator_name,
            scheduler_client=client,
            name="Operations",
            instructions=(
                "You coordinate release readiness."
                " Pull quantitative signals with your tools before answering."
                " Share concise action items and ask Research to verify customer signals."
            ),
            tools=[fetch_release_metrics, list_open_blockers],
        )

        research_agent = create_agent(
            chat_client=chat_client,
            orchestrator_name=orchestrator_name,
            scheduler_client=client,
            name="Research",
            instructions=(
                "You analyze qualitative feedback and risks."
                " Use available tools (especially sentiment) before replying."
                " Coordinate with Operations in the shared thread and recommend mitigations."
            ),
            tools=[synthesize_customer_sentiment],
        )

        thread = ops_agent.get_new_thread()
        store = thread.message_store
        if not isinstance(store, DurableTaskActivityChatMessageStore):
            raise RuntimeError(
                "Thread is not using DurableTaskActivityChatMessageStore as expected."
            )

        termination_instance_id = store.instance_id

        await run_turn(
            ops_agent,
            "Kick off the release checkpoint. Summarize readiness gaps and invite Research to weigh in.",
            thread,
        )

        await run_turn(
            research_agent,
            "Acknowledge Operations update and provide a sentiment-driven risk call."
            " Suggest mitigations referencing any blockers you infer.",
            thread,
        )

        await run_turn(
            ops_agent,
            "Confirm mitigation owners and outline next steps for rollout timing.",
            thread,
        )

        await run_turn(
            research_agent,
            "Close the huddle by confirming customer messaging cadence after deployment.",
            thread,
        )

        try:
            await asyncio.to_thread(
                client.wait_for_orchestration_completion,
                termination_instance_id,
                fetch_payloads=True,
                timeout=20,
            )
            LOGGER.info("Orchestrator instance %s reported completion", termination_instance_id)
        except TimeoutError:
            LOGGER.warning(
                "Timed out waiting for orchestrator instance %s to complete after termination.",
                termination_instance_id,
            )

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
            payload = json.loads(state.serialized_custom_status)
            activity_runs = payload.get("activity_runs", [])
            LOGGER.info("Activity run metadata: %s", json.dumps(activity_runs, indent=2))

    finally:
        if client and termination_instance_id:
            await asyncio.to_thread(
                client.terminate_orchestration,
                termination_instance_id,
                output={"reason": "Multi-agent session complete"},
                recursive=True,
            )
            LOGGER.info(
                "Sent terminate request for orchestrator instance %s",
                termination_instance_id,
            )

        worker.stop()
        LOGGER.info("Durable worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
