"""Azure Functions Durable sample orchestrating Agent Framework conversations.

This sample demonstrates how to trigger Microsoft Agent Framework agents from an
HTTP-triggered Durable Function orchestration. Multiple cooperating agents share a
thread, invoke custom tools, and record their outcomes inside a Durable Task
Entity. One of the tools delegates part of the problem solving to another agent,
illustrating how agents can compose other agents.

Prerequisites:
- Azure Storage connection string configured in ``AzureWebJobsStorage`` (Azurite is fine).
- Azure OpenAI credentials for ``AzureOpenAIChatClient`` (see README for details).
- ``agent-framework`` package available on the Python path (this repo or pip install).
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Annotated
from uuid import uuid4

import azure.durable_functions as df
import azure.functions as func

from opentelemetry import trace
from opentelemetry.trace.span import format_trace_id

# ---------------------------------------------------------------------------
# Ensure repository packages are importable when running the Functions host locally
# ---------------------------------------------------------------------------
_CURRENT_FILE = Path(__file__).resolve()
_PYTHON_ROOT = _CURRENT_FILE.parents[3]  # .../python
for candidate in (_PYTHON_ROOT, _PYTHON_ROOT / "packages", _PYTHON_ROOT / "packages" / "core"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from agent_framework import AgentThread  # noqa: E402
from agent_framework.azure import AzureOpenAIChatClient  # noqa: E402
from agent_framework.observability import get_tracer, setup_observability

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_FILE = _CURRENT_FILE.with_name("function_app.log")
LOG_FILE_HANDLER_NAME = "durable-agents-file-handler"

# Azure Functions often pre-wires console logging. Ensure our file handler is present regardless.
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s [%(process)d %(threadName)s %(funcName)s] %(message)s"
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

file_handler_exists = any(
    isinstance(handler, logging.FileHandler) and getattr(handler, "name", "") == LOG_FILE_HANDLER_NAME
    for handler in root_logger.handlers
)

if not file_handler_exists:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.set_name(LOG_FILE_HANDLER_NAME)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

# Apply consistent formatting to existing console handlers
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(formatter)

LOGGER = logging.getLogger("durable_agents_sample")
LOGGER.info("Durable Functions agent sample logging to %s", LOG_FILE)

# ---------------------------------------------------------------------------
# Durable constants & helpers
# ---------------------------------------------------------------------------
ENTITY_NAME = "AgentStateEntity"
DEFAULT_TOPIC = "release readiness review"
DEFAULT_MAX_TURNS = 3

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@dataclass(slots=True)
class AgentStepRecord:
    agent: str
    prompt: str
    response: str
    tool_calls: list[dict[str, Any]]


class EntityOperationRecorder:
    """Accumulates entity operations for the orchestrator to persist."""

    def __init__(self, conversation_id: str) -> None:
        self._conversation_id = conversation_id
        self._operations: list[dict[str, Any]] = []

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def record_operation(self, operation: str, payload: dict[str, Any] | None = None) -> None:
        data = dict(payload or {})
        data.setdefault("conversation_id", self._conversation_id)
        data.setdefault("timestamp", self._now_iso())
        self._operations.append({"operation": operation, "payload": data})

    def append_message(self, *, role: str, author: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "role": role,
            "author": author,
            "text": text,
        }
        if metadata:
            payload["metadata"] = metadata
        self.record_operation("append_message", payload)

    def record_task(self, *, owner: str, description: str, category: str | None = None) -> None:
        payload = {
            "owner": owner,
            "description": description,
        }
        if category:
            payload["category"] = category
        self.record_operation("record_task", payload)

    def record_tool_call(self, *, agent: str, tool: str, inputs: dict[str, Any], result_preview: str | None = None) -> None:
        payload = {
            "agent": agent,
            "tool": tool,
            "inputs": inputs,
        }
        if result_preview:
            payload["result_preview"] = result_preview
        self.record_operation("record_tool_call", payload)

    def record_summary(self, *, author: str, summary: str) -> None:
        self.record_operation("record_summary", {"author": author, "summary": summary})

    def export(self) -> list[dict[str, Any]]:
        return list(self._operations)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    text = getattr(response, "text", None)
    if text:
        return text
    messages = getattr(response, "messages", None)
    if messages:
        for message in messages:
            msg_text = getattr(message, "text", None)
            if msg_text:
                return msg_text
    try:
        return json.dumps(response)
    except TypeError:
        return str(response)


def _coalesce_topic(payload: dict) -> str:
    topic = payload.get("topic") or payload.get("prompt")
    if not topic:
        return DEFAULT_TOPIC
    return str(topic)


# ---------------------------------------------------------------------------
# HTTP Trigger – starts the durable orchestration
# ---------------------------------------------------------------------------
@app.route(route="agents/durable/start", methods=["POST"])
@app.durable_client_input(client_name="client")
async def http_start_agent_conversation(
    req: func.HttpRequest,
    client: df.DurableOrchestrationClient,
) -> func.HttpResponse:
    """Start the durable orchestration that will coordinate the agents."""

    setup_observability(applicationinsights_connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"))
    with get_tracer().start_as_current_span("Durable Orchestrator as Thread", kind=trace.SpanKind.CLIENT) as current_span:
        print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")

        try:
            body = req.get_json()
        except ValueError:
            body = {}

        conversation_id = body.get("conversationId") or str(uuid4())
        topic = _coalesce_topic(body)
        max_turns = int(body.get("maxTurns", DEFAULT_MAX_TURNS))

        orchestration_payload = {
            "conversation_id": conversation_id,
            "topic": topic,
            "max_turns": max_turns,
            "metadata": body.get("metadata", {}),
        }

        instance_id = await client.start_new(
            "AgentConversationOrchestrator",
            conversation_id,
            orchestration_payload,
        )
        LOGGER.info("Started AgentConversationOrchestrator instance %s for conversation %s", instance_id, conversation_id)

        return client.create_check_status_response(req, instance_id)


# ---------------------------------------------------------------------------
# Orchestrator – coordinates activity + entity updates
# ---------------------------------------------------------------------------
@app.orchestration_trigger(context_name="context")
def AgentConversationOrchestrator(context: df.DurableOrchestrationContext):  # noqa: N802 (Functions naming)
    payload = context.get_input() or {}
    conversation_id = payload.get("conversation_id") or context.instance_id
    topic = payload.get("topic", DEFAULT_TOPIC)
    max_turns = int(payload.get("max_turns", DEFAULT_MAX_TURNS))
    metadata = payload.get("metadata", {})

    entity_id = df.EntityId(ENTITY_NAME, conversation_id)

    context.signal_entity(
        entity_id,
        "initialize",
        {
            "conversation_id": conversation_id,
            "topic": topic,
            "metadata": metadata,
        },
    )
    context.signal_entity(
        entity_id,
        "append_message",
        {
            "role": "user",
            "author": metadata.get("requester", "requester"),
            "text": topic,
        },
    )
    context.signal_entity(entity_id, "set_status", {"status": "running"})

    activity_result = yield context.call_activity(
        "RunAgentCollaboration",
        {
            "conversation_id": conversation_id,
            "topic": topic,
            "max_turns": max_turns,
            "metadata": metadata,
        },
    )

    if activity_result:
        operations = activity_result.get("entity_operations", []) or []
        for op in operations:
            operation = op.get("operation")
            if not operation:
                continue
            context.signal_entity(entity_id, operation, op.get("payload"))

        summary = activity_result.get("final_summary")
        if summary:
            context.signal_entity(
                entity_id,
                "record_summary",
                {"author": activity_result.get("final_author", "Coordinator"), "summary": summary},
            )

        status = activity_result.get("status", "completed")
        context.signal_entity(entity_id, "set_status", {"status": status})

        custom_status = {
            "conversation_id": conversation_id,
            "stage": status,
            "last_turn": activity_result.get("last_turn"),
        }
        context.set_custom_status(custom_status)
    else:
        context.signal_entity(entity_id, "set_status", {"status": "no-result"})
        context.set_custom_status({"conversation_id": conversation_id, "stage": "no-result"})

    final_state = yield context.call_entity(entity_id, "get_state", None)

    summary_status = {
        "conversation_id": conversation_id,
        "stage": "completed",
    }
    if isinstance(final_state, dict):
        summary_status["turn_count"] = len(final_state.get("turns", []))
        summary_status["task_count"] = len(final_state.get("tasks", []))
        summary_status["tool_call_count"] = len(final_state.get("tool_calls", []))
        if final_state.get("summary", {}).get("text"):
            summary_text = final_state["summary"]["text"]
            summary_status["summary_preview"] = summary_text[:200]

    context.set_custom_status(summary_status)
    return {
        "conversation_id": conversation_id,
        "entity_state": final_state,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Activity – runs the multi-agent exchange and collects entity operations
# ---------------------------------------------------------------------------
@app.activity_trigger(input_name="payload")
async def RunAgentCollaboration(payload: dict) -> dict:  # noqa: N802 (Functions naming)
    conversation_id = payload.get("conversation_id", str(uuid4()))
    topic = payload.get("topic", DEFAULT_TOPIC)
    max_turns = int(payload.get("max_turns", DEFAULT_MAX_TURNS))
    metadata = payload.get("metadata", {})

    recorder = EntityOperationRecorder(conversation_id)
    steps: list[AgentStepRecord] = []

    try:
        chat_client = AzureOpenAIChatClient()
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.exception("Failed to initialize AzureOpenAIChatClient", exc_info=exc)
        return {
            "status": "error",
            "error": "Failed to initialize AzureOpenAIChatClient",
            "entity_operations": [
                {
                    "operation": "set_status",
                    "payload": {"status": "error", "details": str(exc)},
                }
            ],
        }

    thread: AgentThread | None = None

    # -------------------------------------------------------------------
    # Define cooperating agents and tools
    # -------------------------------------------------------------------
    recorder.record_operation(
        "register_agents",
        {
            "agents": [
                "Coordinator",
                "Research",
                "Critic",
            ],
        },
    )

    async def _ensure_thread() -> AgentThread:
        nonlocal thread
        if thread is None:
            raise RuntimeError("Agent thread has not been initialized yet.")
        return thread

    async def critic_tool(
        draft: Annotated[str, "Draft plan or insight to critique"],
    ) -> str:
        nonlocal thread
        if thread is None:
            raise RuntimeError("Critic tool invoked before thread initialization.")
        prompt = (
            "You are the critical reviewer in a release readiness huddle."
            " Provide concise risks and questions for the following draft: "
            f"{draft}"
        )
        recorder.record_tool_call(
            agent="Critic",
            tool="critic_review",
            inputs={"draft": draft},
        )
        critic_agent = chat_client.create_agent(
            name="Critic",
            instructions=(
                "You review cross-team plans, highlighting hidden risks."
                " Keep feedback short (<=100 words) and actionable."
            ),
        )
        critique = await critic_agent.run(prompt, thread=thread)
        critique_text = _extract_response_text(critique)
        recorder.append_message(role="assistant", author="Critic", text=critique_text)
        return critique_text

    async def record_research_insight(
        insight: Annotated[str, "Key insight or data point that should be archived"],
    ) -> str:
        recorder.record_tool_call(
            agent="Research",
            tool="record_research_insight",
            inputs={"insight": insight},
            result_preview=insight,
        )
        recorder.record_task(owner="Research", description=insight, category="insight")
        return "Insight stored in durable entity."

    research_agent = chat_client.create_agent(
        name="Research",
        instructions=(
            "You dig into qualitative and quantitative signals."
            " Use the record_research_insight tool for every major finding."
            " Ask the critic tool when the plan might be risky."
        ),
        tools=[record_research_insight, critic_tool],
    )

    async def delegate_research(
        question: Annotated[str, "Research question for the research specialist"],
    ) -> str:
        nonlocal thread
        thread = await _ensure_thread()
        recorder.record_tool_call(
            agent="Coordinator",
            tool="delegate_research",
            inputs={"question": question},
        )
        response = await research_agent.run(question, thread=thread)
        text = _extract_response_text(response)
        recorder.append_message(role="assistant", author="Research", text=text)
        return text

    def record_action_item(
        action_item: Annotated[str, "Actionable task with owner"],
    ) -> str:
        recorder.record_tool_call(
            agent="Coordinator",
            tool="record_action_item",
            inputs={"action_item": action_item},
            result_preview=action_item,
        )
        recorder.record_task(owner="Coordinator", description=action_item, category="action")
        return f"Action item recorded: {action_item}"

    coordinator_agent = chat_client.create_agent(
        name="Coordinator",
        instructions=(
            "You lead release readiness checkpoints."
            " Always delegate research questions using the delegate_research tool before final decisions."
            " Capture every commitment with record_action_item."
            " Close with a brief go/no-go summary referencing stored tasks."
        ),
        tools=[delegate_research, record_action_item],
    )

    thread = coordinator_agent.get_new_thread()

    # -------------------------------------------------------------------
    # Run the conversation across a few guided turns
    # -------------------------------------------------------------------
    prompts = [
        (
            "Coordinator",
            f"Kick off the readiness huddle focused on {topic}."
            " Identify immediate questions for the Research agent and log next steps.",
        ),
        (
            "Coordinator",
            "Summarize the plan based on research findings."
            " Make sure every owner and due date is captured via record_action_item.",
        ),
    ]

    final_summary: str | None = None
    last_turn = None

    for turn_index, (agent_name, prompt_text) in enumerate(prompts, start=1):
        if agent_name != "Coordinator":
            continue  # Reserved for future extension
        recorder.append_message(role="system", author="orchestrator", text=f"Turn {turn_index}: {prompt_text}")
        response = await coordinator_agent.run(prompt_text, thread=thread)
        response_text = _extract_response_text(response)
        recorder.append_message(role="assistant", author="Coordinator", text=response_text)
        steps.append(
            AgentStepRecord(
                agent="Coordinator",
                prompt=prompt_text,
                response=response_text,
                tool_calls=[],
            )
        )
        last_turn = turn_index
        if turn_index >= max_turns:
            break

    # Coordinator closes with final summary referencing tasks recorded along the way
    summary_prompt = (
        "Provide the final release readiness summary."
        " Reference the tasks you've logged and declare go/no-go."
    )
    summary_response = await coordinator_agent.run(summary_prompt, thread=thread)
    final_summary = _extract_response_text(summary_response)
    recorder.append_message(role="assistant", author="Coordinator", text=final_summary)
    recorder.record_summary(author="Coordinator", summary=final_summary)

    return {
        "conversation_id": conversation_id,
        "status": "completed",
        "entity_operations": recorder.export(),
    "agent_steps": [asdict(step) for step in steps],
        "final_summary": final_summary,
        "final_author": "Coordinator",
        "last_turn": last_turn,
    }


# ---------------------------------------------------------------------------
# Durable Entity – persists conversation outcomes
# ---------------------------------------------------------------------------
@app.entity_trigger(context_name="context")
def AgentStateEntity(context: df.DurableEntityContext):  # noqa: N802 (Functions naming)
    state = context.get_state(
        lambda: {
            "conversation_id": None,
            "topic": None,
            "metadata": {},
            "turns": [],
            "tasks": [],
            "tool_calls": [],
            "summary": None,
            "status": "initialized",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
    )

    operation = context.operation_name
    payload = context.get_input() or {}

    def _touch() -> None:
        state["last_updated"] = datetime.now(timezone.utc).isoformat()

    if operation == "initialize":
        state["conversation_id"] = payload.get("conversation_id", state["conversation_id"])
        state["topic"] = payload.get("topic", state["topic"])
        state["metadata"] = payload.get("metadata", state["metadata"])
        _touch()
    elif operation == "append_message":
        state["turns"].append(
            {
                "role": payload.get("role", "assistant"),
                "author": payload.get("author", "unknown"),
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {}),
                "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
            }
        )
        _touch()
    elif operation == "record_task":
        state["tasks"].append(
            {
                "owner": payload.get("owner", "unknown"),
                "description": payload.get("description", ""),
                "category": payload.get("category"),
                "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
            }
        )
        _touch()
    elif operation == "record_tool_call":
        state["tool_calls"].append(
            {
                "agent": payload.get("agent"),
                "tool": payload.get("tool"),
                "inputs": payload.get("inputs", {}),
                "result_preview": payload.get("result_preview"),
                "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
            }
        )
        _touch()
    elif operation == "record_summary":
        state["summary"] = {
            "author": payload.get("author"),
            "text": payload.get("summary"),
            "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
        }
        _touch()
    elif operation == "set_status":
        state["status"] = payload.get("status", state["status"])
        _touch()
    elif operation == "register_agents":
        state["metadata"]["agents"] = payload.get("agents", [])
        _touch()
    elif operation == "get_state":
        context.set_result(state)
    elif operation == "reset":
        state = {
            "conversation_id": None,
            "topic": None,
            "metadata": {},
            "turns": [],
            "tasks": [],
            "tool_calls": [],
            "summary": None,
            "status": "reset",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
    else:  # pragma: no cover - defensive default
        LOGGER.warning("Unknown entity operation received: %s", operation)

    context.set_state(state)
