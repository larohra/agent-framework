# Copyright (c) Microsoft. All rights reserved.

"""Durable Task-backed checkpoint storage for Agent Framework workflows.

This sample demonstrates how to persist workflow checkpoints using Azure Durable Task
Scheduler instead of the built-in file or in-memory stores. A Durable orchestration
tracks checkpoint metadata in its custom status payload, giving you durable
persistence, geo-redundant storage (depending on your task hub configuration), and the
ability to inspect or replay checkpoints from any worker that can access the Durable
Task Scheduler instance.

Key takeaways:

* Implement a ``CheckpointStorage`` adapter backed by Durable Task Scheduler
* Handle checkpoint save/list/load/delete via Durable orchestration events
* Reuse the adapter with a standard Agent Framework workflow pipeline
* Provide defensive error handling and configuration stubs

Prerequisites:

1. Provision an Azure Durable Task Scheduler (preview) task hub and note the gRPC
   endpoint plus task hub name.
2. Install required packages:

   ```bash
   pip install durabletask durabletask-azuremanaged azure-identity agent-framework-core --pre
   ```

3. Configure authentication that works with ``DefaultAzureCredential`` (Managed
   Identity, workload identity, `az login`, etc.).
4. Export the following environment variables:

   * ``DTS_HOST_ADDRESS`` – gRPC endpoint (e.g. ``my-hub.westus2.eventgrid.azure.net:443``)
   * ``DTS_TASK_HUB`` – Task hub name (e.g. ``sample-hub``)
   * OpenAI credentials as required by ``OpenAIChatClient``

During execution this script will:

1. Start a Durable Task worker that owns the checkpoint orchestration and activities.
2. Build a simple multi-step workflow and run it once with Durable checkpointing.
3. List the stored checkpoints, reload the most recent checkpoint, and print summary
   metadata.

Clean-up: the sample stops the worker on exit but does not delete the Durable
orchestration instance. You can delete it manually via the Durable Task Scheduler
management APIs if desired.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from agent_framework import (
    AgentExecutor,
    AgentExecutorResponse,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    handler,
)
from agent_framework._workflows._checkpoint import CheckpointStorage, WorkflowCheckpoint
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged import DurableTaskSchedulerClient, DurableTaskSchedulerWorker
from durabletask.task import OrchestrationContext
from durabletask.worker import TaskHubGrpcWorker

LOGGER = logging.getLogger("durable_task_checkpoint_sample")
logging.basicConfig(level=logging.INFO)

_EVENT_NAME = "workflow_checkpoint"


@dataclass
class SchedulerSettings:
    """Configuration required to connect to Durable Task Scheduler."""

    host_address: str
    task_hub: str


def load_scheduler_settings() -> SchedulerSettings:
    """Resolve scheduler settings from environment variables with validation."""

    host = os.getenv("DTS_HOST_ADDRESS")
    hub = os.getenv("DTS_TASK_HUB")
    if not host or not hub:
        raise EnvironmentError(
            "Missing Durable Task configuration. Ensure DTS_HOST_ADDRESS and DTS_TASK_HUB are set."
        )
    LOGGER.debug("Loaded Durable Task settings host=%s task_hub=%s", host, hub)
    return SchedulerSettings(host_address=host, task_hub=hub)


class DurableTaskCheckpointStorage(CheckpointStorage):
    """Checkpoint storage adapter that persists data in Durable Task Scheduler."""

    def __init__(
        self,
        client: DurableTaskSchedulerClient,
        orchestrator_name: str,
        *,
        instance_id: str | None = None,
    ) -> None:
        self._client = client
        self._orchestrator_name = orchestrator_name
        self._instance_id = instance_id or self._start_orchestrator()

    def _start_orchestrator(self) -> str:
        LOGGER.info("Starting checkpoint vault orchestrator '%s'", self._orchestrator_name)
        instance_id = self._client.schedule_new_orchestration(
            self._orchestrator_name,
            input={"checkpoints": {}},
        )
        return instance_id

    async def _raise_event(self, payload: dict[str, Any]) -> None:
        try:
            await asyncio.to_thread(
                self._client.raise_orchestration_event,
                self._instance_id,
                _EVENT_NAME,
                payload,
            )
        except Exception as exc:
            LOGGER.exception("Failed to raise checkpoint event %s", payload)
            raise RuntimeError("Durable Task event dispatch failed") from exc

    async def _read_state(self) -> dict[str, Any]:
        try:
            state = await asyncio.to_thread(
                self._client.get_orchestration_state,
                self._instance_id,
                True,
            )
        except Exception as exc:
            raise RuntimeError("Failed to query Durable orchestration state") from exc

        if not state or not state.serialized_custom_status:
            return {"checkpoints": {}}
        return json.loads(state.serialized_custom_status)

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        payload = {
            "action": "save",
            "checkpoint": checkpoint.to_dict(),
        }
        await self._raise_event(payload)
        LOGGER.info("Persisted checkpoint %s to Durable Task", checkpoint.checkpoint_id)
        return checkpoint.checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        state = await self._read_state()
        raw_cp = state.get("checkpoints", {}).get(checkpoint_id)
        if not raw_cp:
            LOGGER.warning("Checkpoint %s not found in Durable Task", checkpoint_id)
            return None
        return WorkflowCheckpoint.from_dict(raw_cp)

    async def list_checkpoint_ids(self, workflow_id: str | None = None) -> list[str]:
        return [
            cp_id
            for cp_id, payload in (await self._read_state()).get("checkpoints", {}).items()
            if workflow_id is None or payload.get("workflow_id") == workflow_id
        ]

    async def list_checkpoints(self, workflow_id: str | None = None) -> list[WorkflowCheckpoint]:
        checkpoints: list[WorkflowCheckpoint] = []
        for payload in (await self._read_state()).get("checkpoints", {}).values():
            if workflow_id is None or payload.get("workflow_id") == workflow_id:
                checkpoints.append(WorkflowCheckpoint.from_dict(payload))
        return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        payload = {"action": "delete", "checkpoint_id": checkpoint_id}
        await self._raise_event(payload)
        LOGGER.info("Deleted checkpoint %s from Durable Task", checkpoint_id)
        return True


def checkpoint_vault_orchestrator(context: OrchestrationContext, state: dict[str, Any] | None):
    """Durable orchestrator that maintains checkpoint JSON in custom status."""

    state = state or {"checkpoints": {}}
    checkpoints = state.setdefault("checkpoints", {})
    context.set_custom_status(state)

    request = yield context.wait_for_external_event(_EVENT_NAME)

    action = request.get("action") if isinstance(request, dict) else None

    if action == "save":
        payload = request.get("checkpoint") or {}
        checkpoint_id = payload.get("checkpoint_id")
        if checkpoint_id:
            checkpoints[checkpoint_id] = payload
    elif action == "delete":
        checkpoint_id = request.get("checkpoint_id")
        if checkpoint_id and checkpoint_id in checkpoints:
            checkpoints.pop(checkpoint_id)
    else:
        # Unsupported or missing action; record last error for debugging.
        state["last_error"] = {
            "action": action,
            "received": request,
        }

    context.set_custom_status(state)
    context.continue_as_new(state, save_events=True)


class RecordInputExecutor(Executor):
    """Stores incoming text in shared state and forwards for processing."""

    def __init__(self, *, id: str = "record-input") -> None:
        super().__init__(id=id)

    @handler
    async def handle(self, message: str, ctx: WorkflowContext[str]) -> None:
        await ctx.set_shared_state("original_message", message)
        await ctx.send_message(message)


class SummarizeAgent(Executor):
    """Wraps an AgentExecutor to produce a concise response."""

    def __init__(self, agent_executor: AgentExecutor, *, id: str = "summarize-agent") -> None:
        super().__init__(id=id)
        self._agent_executor = agent_executor

    @handler
    async def invoke(self, message: str, ctx: WorkflowContext[AgentExecutorResponse, str]) -> None:
        # Forward to the child AgentExecutor and await its response.
        await ctx.send_message(message, target_id=self._agent_executor.id)

    @handler
    async def agent_response(self, response: AgentExecutorResponse, ctx: WorkflowContext[str]) -> None:
        await ctx.yield_output(response.agent_run_response.text or "")


def build_sample_workflow(checkpoint_storage: CheckpointStorage) -> tuple[AgentExecutor, Any]:
    """Construct a small workflow that benefits from checkpoint persistence."""

    chat_client = OpenAIChatClient()
    agent = chat_client.create_agent(
        name="durable-checkpoint-agent",
        instructions=(
            "You are a safety-conscious summarizer. Condense the input into one short sentence and mention "
            "that the workflow state is persisted by Durable Task."
        ),
    )

    agent_executor = AgentExecutor(agent, id="agent")
    record = RecordInputExecutor()
    summarize = SummarizeAgent(agent_executor)

    workflow = (
        WorkflowBuilder(max_iterations=3)
        .add_edge(record, summarize)
        .add_edge(summarize, agent_executor)
        .add_edge(agent_executor, summarize)
        .set_start_executor(record)
        .with_checkpointing(checkpoint_storage=checkpoint_storage)
        .build()
    )

    return agent_executor, workflow


def _register_worker(worker: TaskHubGrpcWorker) -> str:
    orchestrator_name = worker.add_orchestrator(checkpoint_vault_orchestrator)
    LOGGER.info("Registered orchestrator '%s'", orchestrator_name)
    return orchestrator_name


async def run_workflow_with_checkpoints(storage: CheckpointStorage, workflow: Any) -> str | None:
    """Execute the workflow and capture the latest checkpoint ID."""

    workflow_id: str | None = None
    async for event in workflow.run_stream("Document a high-level summary of Durable Task checkpointing." ):
        if isinstance(event, WorkflowOutputEvent):
            LOGGER.info("Workflow output: %s", event.data)
        workflow_id = getattr(event, "workflow_id", workflow_id)

    checkpoints = await storage.list_checkpoints(workflow_id)
    if not checkpoints:
        LOGGER.error("No checkpoints persisted!")
        return None

    latest = max(checkpoints, key=lambda cp: cp.timestamp)
    LOGGER.info("Latest checkpoint %s (workflow_id=%s) contains keys=%s", latest.checkpoint_id, latest.workflow_id, list(latest.shared_state))
    return latest.checkpoint_id


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

        storage = DurableTaskCheckpointStorage(client, orchestrator_name)
        agent_executor, workflow = build_sample_workflow(storage)

        latest_cp_id = await run_workflow_with_checkpoints(storage, workflow)
        if not latest_cp_id:
            return

        loaded = await storage.load_checkpoint(latest_cp_id)
        if loaded:
            LOGGER.info(
                "Loaded checkpoint %s with shared_state=%s",
                latest_cp_id,
                json.dumps(loaded.shared_state, indent=2),
            )

    except Exception as exc:  # pragma: no cover - defensive for sample clarity
        LOGGER.exception("Sample failed: %s", exc)
    finally:
        worker.stop()
        LOGGER.info("Durable worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
