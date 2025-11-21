# Design: Durable Task Provider for Agent Framework

## Overview

This package, `agent-framework-durabletask`, provides a durability layer for the Microsoft Agent Framework using the `durabletask` Python SDK. It enables stateful, reliable, and distributed agent execution on any platform (Bring Your Own Platform), decoupling the agent's durability from the Azure Functions platform.

## Core Philosophy

*   **Native `DurableEntity` Support**: We will leverage the `DurableEntity` support introduced in `durabletask` v1.0.0.
*   **Wrapper Pattern**: We will provide wrapper classes (`AgentWorker`, `AgentClient`) that accept *any* compatible `durabletask` worker or client.
*   **Separation of Concerns**: We will provide distinct APIs for external client interactions vs. internal orchestration interactions to ensure type safety and clarity.

## Architecture

### 1. Package Structure

```text
packages/durabletask/
├── pyproject.toml
├── README.md
├── agent_framework_durabletask/
│   ├── __init__.py
│   ├── _worker.py      # AgentWorker wrapper
│   ├── _client.py      # AgentClient wrapper (External interactions)
│   ├── _orchestration.py # Orchestration helpers (Internal interactions)
│   ├── _entities.py    # AgentEntity implementation
│   ├── _models.py      # Data models (RunRequest, AgentResponse, etc.)
│   ├── _durable_agent_state.py # State schema (Ported from azurefunctions)
│   └── _utils.py
└── tests/
```

### 2. State Management (`_durable_agent_state.py`)

*   **Goal**: Maintain 100% schema compatibility with `agent-framework-azurefunctions`.
*   **Implementation**: Direct port of `packages/azurefunctions/agent_framework_azurefunctions/_durable_agent_state.py`.

### 3. The Agent Entity (`_entities.py`)

We will implement a class `AgentEntity` that inherits from `durabletask.entities.DurableEntity`.

```python
class AgentEntity(durabletask.entities.DurableEntity):
    def __init__(self, agent: AgentProtocol):
        self.agent = agent
        self.state = DurableAgentState()

    async def run_agent(self, input_data: dict[str, Any]) -> dict[str, Any]:
        # 1. Deserialize Input
        request = RunRequest.from_dict(input_data)
        
        # 2. Update State (User Message)
        state_request = DurableAgentStateRequest.from_run_request(request)
        self.state.data.conversation_history.append(state_request)
        
        # 3. Rehydrate Chat History
        chat_messages = [
            m.to_chat_message() 
            for entry in self.state.data.conversation_history 
            for m in entry.messages
        ]
        
        # 4. Execute Agent
        response = await self.agent.run(messages=chat_messages, ...)
        
        # 5. Update State (Agent Response)
        state_response = DurableAgentStateResponse.from_run_response(
            request.correlation_id, response
        )
        self.state.data.conversation_history.append(state_response)
        
        # 6. Return Result (Serialized AgentRunResponse)
        return response.to_dict() 

    def reset(self) -> None:
        self.state = DurableAgentState()
```

### 4. The Worker Wrapper (`_worker.py`)

The `AgentWorker` wraps an existing `durabletask` worker instance.

```python
class AgentWorker:
    def __init__(self, worker: TaskHubWorker | TaskHubGrpcWorker):
        self._worker = worker

    def add_agent(self, agent: AgentProtocol) -> None:
        """Registers an agent with the worker."""
        entity_name = f"agent:{agent.name}"
        
        def entity_factory(context):
            return AgentEntity(agent)
            
        self._worker.register_entity(entity_name, entity_factory)

    async def start(self):
        await self._worker.start()

    async def stop(self):
        await self._worker.stop()
```

### 5. External Client Interaction (`_client.py`)

The `AgentClient` is strictly for external clients (e.g., FastAPI, CLI) interacting with the backend.

```python
class AgentClient:
    def __init__(self, client: TaskHubClient | TaskHubGrpcClient):
        self._client = client

    async def run_agent(
        self, 
        agent_name: str, 
        message: str, 
        thread_id: str | None = None,
        timeout_seconds: float = 60.0
    ) -> AgentRunResponse:
        """Runs an agent and waits for the result via polling."""
        
        thread_id = thread_id or str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        entity_id = EntityId(f"agent:{agent_name}", thread_id)
        
        request = RunRequest(
            message=message,
            thread_id=thread_id,
            correlation_id=correlation_id
        )
        
        # Signal Entity (Fire-and-forget)
        await self._client.signal_entity(
            entity_id, 
            "run_agent", 
            request.to_dict()
        )
        
        # Poll for Result
        start_time = time.time()
        while (time.time() - start_time) < timeout_seconds:
            state = await self._client.read_entity_state(entity_id)
            if state:
                durable_state = DurableAgentState.from_dict(state)
                response_entry = durable_state.try_get_agent_response_entry(correlation_id)
                if response_entry:
                    return response_entry.to_run_response()
            
            await asyncio.sleep(1.0)
            
        raise TimeoutError("Agent did not respond in time.")
```

### 6. Internal Orchestration Interaction (`_orchestration.py`)

We provide a functional helper `call_agent` (or a wrapper class `AgentOrchestrationContext`) for use *inside* orchestrations. This clearly separates the "yield" behavior from the "await" behavior.

```python
def call_agent(
    context: OrchestrationContext,
    agent_name: str,
    message: str,
    thread_id: str | None = None
) -> Task:
    """
    Helper to call an agent entity from within an orchestration.
    Returns a Task that must be yielded.
    """
    thread_id = thread_id or context.instance_id # Default to orchestration ID
    correlation_id = str(uuid.uuid4())
    entity_id = EntityId(f"agent:{agent_name}", thread_id)
    
    request = RunRequest(
        message=message,
        thread_id=thread_id,
        correlation_id=correlation_id
    )
    
    # Returns the Task directly
    return context.call_entity(entity_id, "run_agent", request.to_dict())
```

## Comparison: `azurefunctions` vs `durabletask`

| Feature | `packages/azurefunctions` | `packages/durabletask` |
| :--- | :--- | :--- |
| **Entity Base Class** | `azure.durable_functions.DurableEntity` | `durabletask.entities.DurableEntity` |
| **Registration** | Function App Decorators | `worker.add_agent(agent)` (via Wrapper) |
| **Client API** | HTTP Requests | `client.run_agent(...)` (via Wrapper) |
| **Orchestration API** | `DurableAIAgent(context).run(...)` | `yield call_agent(context, ...)` |
| **State Schema** | `_durable_agent_state.py` | Same Schema (Shared/Duplicated) |
| **Execution** | In-process (Functions Runtime) | In-process (TaskHubWorker / Managed Worker) |

## Implementation Steps

1.  **Scaffold Package**: Create directory structure and `pyproject.toml`.
2.  **Port State Models**: Copy `_durable_agent_state.py` and `_models.py` (adapting imports).
3.  **Implement `AgentEntity`**: Create `_entities.py`.
4.  **Implement `AgentWorker`**: Create `_worker.py`.
5.  **Implement `AgentClient`**: Create `_client.py` (External only).
6.  **Implement Orchestration Helpers**: Create `_orchestration.py` (Internal only).
7.  **Tests**: Add unit tests and integration tests.
