# Copyright (c) Microsoft. All rights reserved.

"""Durable Task orchestrated workflow execution with Agent Framework.

This sample shows how to coordinate Agent Framework workflows using Azure Durable Task
Scheduler. A Durable orchestration receives "run" requests, executes each workflow via
a dedicated activity, and records structured run metadata (status, summary, errors) in
its custom status payload. Durable Task provides reliable fan-out/fan-in semantics,
retries, and checkpointing so your workflow runs can survive process restarts.

What the sample covers:

* Register a Durable orchestrator + activity for workflow execution
* Build a small Agent Framework workflow inside the activity and run it end-to-end
* Send run requests via orchestration events and poll resulting run summaries
* Capture configuration stubs and robust error handling for real deployments

Prerequisites mirror ``durable_task_checkpoint_storage.py``:

1. Azure Durable Task Scheduler endpoint + task hub (`DTS_HOST_ADDRESS`, `DTS_TASK_HUB`).
2. Python packages: ``durabletask``, ``durabletask-azuremanaged``, ``azure-identity``,
   and the Agent Framework core packages.
3. Authentication for ``DefaultAzureCredential`` and credentials for the selected chat
   provider (OpenAI in this example).

The sample is intentionally verbose with logging and comments to serve as a reference
when adapting Durable Task orchestrations to manage complex Agent Framework workflows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from agent_framework import (
    AgentExecutor,
    WorkflowBuilder,
    WorkflowOutputEvent,
    handler,
)
from agent_framework import Executor as AFExecutor
from agent_framework import WorkflowContext
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged import DurableTaskSchedulerClient, DurableTaskSchedulerWorker
from durabletask.task import OrchestrationContext
from durabletask.worker import TaskHubGrpcWorker

LOGGER = logging.getLogger("durable_task_workflow_sample")
logging.basicConfig(level=logging.INFO)

_RUN_EVENT = "workflow_run"
_ACTIVITY_NAME = "execute_agent_workflow"
_POLL_INTERVAL_SECONDS = 2.0


@dataclass
class SchedulerSettings:
    host_address: str
    task_hub: str


def load_scheduler_settings() -> SchedulerSettings:
    host = os.getenv("DTS_HOST_ADDRESS")
    hub = os.getenv("DTS_TASK_HUB")
    if not host or not hub:
        raise EnvironmentError("Set DTS_HOST_ADDRESS and DTS_TASK_HUB before running the sample")
    return SchedulerSettings(host_address=host, task_hub=hub)


@dataclass
class WorkflowRunConfig:
    """Lightweight configuration supplied to the Durable orchestration."""

    prompt: str
    domain: str = "general"
    max_iterations: int = 3


def load_run_configuration() -> WorkflowRunConfig:
    """Collect run configuration from environment or defaults."""

    prompt = os.getenv(
        "DURABLE_WORKFLOW_PROMPT",
        "Create a short deployment checklist for rolling out a new internal AI agent.",
    )
    domain = os.getenv("DURABLE_WORKFLOW_DOMAIN", "operations")
    try:
        max_iterations = int(os.getenv("DURABLE_WORKFLOW_MAX_ITER", "3"))
    except ValueError:
        max_iterations = 3
    return WorkflowRunConfig(prompt=prompt, domain=domain, max_iterations=max_iterations)


def workflow_run_orchestrator(context: OrchestrationContext, state: dict[str, Any] | None):
    """Durable orchestrator that fan-outs to workflow execution activity."""

    state = state or {"runs": []}
    runs: list[dict[str, Any]] = state.setdefault("runs", [])
    context.set_custom_status(state)

    request = yield context.wait_for_external_event(_RUN_EVENT)

    run_id = request.get("run_id") if isinstance(request, dict) else None
    if not run_id:
        run_id = str(uuid.uuid4())
    payload = {"run_id": run_id, "config": request.get("config") if isinstance(request, dict) else None}

    try:
        result = yield context.call_activity(_ACTIVITY_NAME, payload)
        runs.append({
            "run_id": run_id,
            "status": "Succeeded",
            "summary": result.get("summary"),
            "outputs": result.get("outputs"),
        })
    except Exception as exc:  # pragma: no cover - durable runtime already logs failures
        LOGGER.error("Activity failed for run %s: %s", run_id, exc)
        runs.append({
            "run_id": run_id,
            "status": "Failed",
            "error": str(exc),
        })

    context.set_custom_status(state)
    context.continue_as_new(state, save_events=True)


class CollectNotesExecutor(AFExecutor):
    """Simple executor that records workflow messages in shared state."""

    def __init__(self, *, id: str = "collect-notes") -> None:
        super().__init__(id=id)

    @handler
    async def capture(self, text: str, ctx: WorkflowContext[str]) -> None:
        notes = await ctx.get_shared_state("notes") or []
        notes.append(text)
        await ctx.set_shared_state("notes", notes)
        await ctx.send_message(text)


def _build_agent_workflow(config: WorkflowRunConfig) -> tuple[Any, Any]:
    """Create the Agent Framework workflow executed inside the Durable activity."""

    chat_client = OpenAIChatClient()
    checklist_agent = chat_client.create_agent(
        name="durable-workflow-agent",
        instructions=(
            "You produce actionable checklists. Always mention the domain and credit Durable Task for orchestration."
        ),
    )

    agent_executor = AgentExecutor(checklist_agent, id="checklist-agent")
    collector = CollectNotesExecutor()

    workflow = (
        WorkflowBuilder(max_iterations=config.max_iterations)
        .add_edge(collector, agent_executor)
        .add_edge(agent_executor, collector)
        .set_start_executor(collector)
        .build()
    )

    return agent_executor, workflow


async def _run_workflow(config: WorkflowRunConfig) -> dict[str, Any]:
    agent_executor, workflow = _build_agent_workflow(config)
    outputs: list[str] = []
    async for event in workflow.run_stream(config.prompt):
        if isinstance(event, WorkflowOutputEvent):
            outputs.append(str(event.data))
    summary = outputs[-1] if outputs else ""
    return {
        "summary": summary,
        "outputs": outputs,
    }


def execute_agent_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    """Durable Task activity entry point (synchronous wrapper)."""

    config_payload = payload.get("config") or {}
    config = WorkflowRunConfig(
        prompt=config_payload.get("prompt", load_run_configuration().prompt),
        domain=config_payload.get("domain", load_run_configuration().domain),
        max_iterations=int(config_payload.get("max_iterations", load_run_configuration().max_iterations)),
    )

    LOGGER.info("Executing workflow activity run_id=%s domain=%s", payload.get("run_id"), config.domain)
    return asyncio.run(_run_workflow(config))


def _register_worker(worker: TaskHubGrpcWorker) -> tuple[str, str]:
    orchestrator_name = worker.add_orchestrator(workflow_run_orchestrator)
    activity_name = worker.add_activity(execute_agent_workflow, name=_ACTIVITY_NAME)
    LOGGER.info("Registered orchestrator '%s' and activity '%s'", orchestrator_name, activity_name)
    return orchestrator_name, activity_name


async def _send_run_request(
    client: DurableTaskSchedulerClient,
    orchestrator_name: str,
    run_config: WorkflowRunConfig,
) -> str:
    instance_id = client.schedule_new_orchestration(
        orchestrator_name,
        input={"runs": []},
    )
    payload = {
        "run_id": str(uuid.uuid4()),
        "config": run_config.__dict__,
    }
    await asyncio.to_thread(client.raise_orchestration_event, instance_id, _RUN_EVENT, payload)
    LOGGER.info("Raised run request %s", payload["run_id"])
    return instance_id


async def _poll_run_status(client: DurableTaskSchedulerClient, instance_id: str, timeout: float = 60.0) -> list[dict[str, Any]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = await asyncio.to_thread(client.get_orchestration_state, instance_id, True)
        if state and state.serialized_custom_status:
            status = state.runtime_status
            runs = state.serialized_custom_status
            try:
                data = json.loads(runs)
            except Exception:  # pragma: no cover - defensive parsing
                data = {"runs": []}
            if data.get("runs"):
                LOGGER.info("Current orchestration status=%s runs=%s", status, data["runs"])
                return data["runs"]
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
    raise TimeoutError("Timed out waiting for Durable workflow run to complete")


async def main() -> None:
    settings = load_scheduler_settings()
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

    worker = DurableTaskSchedulerWorker(
        host_address=settings.host_address,
        taskhub=settings.task_hub,
        token_credential=credential,
    )

    orchestrator_name, _ = _register_worker(worker)

    try:
        worker.start()
        client = DurableTaskSchedulerClient(
            host_address=settings.host_address,
            taskhub=settings.task_hub,
            token_credential=credential,
        )

        run_config = load_run_configuration()
        instance_id = await _send_run_request(client, orchestrator_name, run_config)
        runs = await _poll_run_status(client, instance_id)
        for run in runs:
            LOGGER.info("Run %s finished with status=%s summary=%s", run.get("run_id"), run.get("status"), run.get("summary"))

    except Exception as exc:  # pragma: no cover - sample robustness
        LOGGER.exception("Durable workflow orchestration failed: %s", exc)
    finally:
        worker.stop()
        LOGGER.info("Durable worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
