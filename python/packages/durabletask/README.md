# Agent Framework - Durable Task Integration

Durable Task integration for the Microsoft Agent Framework, enabling stateful, reliable, and distributed agent execution on any platform.

## Overview

This package provides a durability layer for the Microsoft Agent Framework using the [durabletask Python SDK](https://github.com/microsoft/durabletask-python). It enables:

- **Stateful Agents**: Conversation history is automatically persisted and restored
- **Reliable Execution**: Agents survive process restarts and failures
- **Distributed Agents**: Run agents across multiple machines/containers
- **Platform Agnostic**: Works with any Durable Task backend (Dapr, Azure Functions, standalone sidecar)

## Key Features

- ✅ Native `DurableEntity` support for state management
- ✅ Wrapper pattern for easy integration with existing durabletask workers/clients
- ✅ 100% schema compatibility with `agent-framework-azurefunctions`
- ✅ Separation of external client APIs vs internal orchestration APIs
- ✅ Full conversation history preservation across invocations

## Installation

```bash
pip install agent-framework-durabletask
```

## Quick Start

### 1. Create an Agent

```python
from agent_framework import ChatMessage, AgentRunResponse

class MyAgent:
    def __init__(self, name: str):
        self.name = name
    
    async def run(self, messages: list[ChatMessage], **kwargs) -> AgentRunResponse:
        # Your agent logic here
        response = ChatMessage(role="assistant", text="Hello!")
        return AgentRunResponse(messages=[response])
```

### 2. Set Up Worker (Server Side)

```python
from durabletask import TaskHubGrpcWorker
from agent_framework_durabletask import AgentWorker

# Create your agent
agent = MyAgent("my-agent")

# Wrap the durabletask worker
durable_worker = TaskHubGrpcWorker(host_address="localhost:4001")
worker = AgentWorker(durable_worker)

# Register your agent
worker.add_agent(agent)

# Start processing
await worker.start()
```

### 3. Call from Client (External)

```python
from durabletask import TaskHubGrpcClient
from agent_framework_durabletask import AgentClient

# Create client
durable_client = TaskHubGrpcClient(host_address="localhost:4001")
client = AgentClient(durable_client)

# Run the agent
response = await client.run_agent(
    agent_name="my-agent",
    message="Hello, how are you?"
)

print(response.messages[0].text)
```

## Architecture

The package provides three main components:

### 1. `AgentEntity`

A durable entity that wraps your agent and maintains conversation state:

```python
class AgentEntity:
    def __init__(self, agent: AgentProtocol):
        self.agent = agent
        self.state = DurableAgentState()
    
    async def run_agent(self, input_data: dict) -> dict:
        # Manages state, rehydrates history, executes agent
        ...
```

### 2. `AgentWorker`

Wraps a durabletask worker and simplifies agent registration:

```python
worker = AgentWorker(durable_worker)
worker.add_agent(my_agent)
await worker.start()
```

### 3. `AgentClient`

Provides external client interaction with polling for results:

```python
client = AgentClient(durable_client)
response = await client.run_agent(
    agent_name="my-agent",
    message="Hello!"
)
```

## State Management

The package uses the same durable agent state schema as `agent-framework-azurefunctions`, ensuring:

- **Conversation History**: All messages are preserved with timestamps and metadata
- **Token Usage Tracking**: Input/output token counts are recorded
- **Request/Response Correlation**: Each request-response pair is linked via correlation ID
- **Schema Versioning**: Supports migration and backward compatibility

State structure:

```json
{
  "schemaVersion": "1.0.0",
  "data": {
    "conversationHistory": [
      {
        "$type": "request",
        "correlationId": "...",
        "createdAt": "...",
        "messages": [...]
      },
      {
        "$type": "response",
        "correlationId": "...",
        "createdAt": "...",
        "messages": [...],
        "usage": {...}
      }
    ]
  }
}
```

## Testing the Implementation

This package includes a comprehensive simulation for testing and demonstration purposes. See [`simulation/README.md`](simulation/README.md) for details.

### Running the Simulation

```bash
# Make sure a Durable Task sidecar is running on localhost:4001
cd packages/durabletask
python -m simulation.simulate
```

The simulation verifies:
- Agent registration and execution
- State persistence across multiple turns
- Conversation history management
- Turn counting (proving state is maintained)

## Comparison with Azure Functions Package

| Feature | `azurefunctions` | `durabletask` |
|---------|-----------------|---------------|
| **Entity Base** | `azure.durable_functions.DurableEntity` | Manual entity implementation |
| **Registration** | Function App Decorators | `worker.add_agent(agent)` |
| **Client API** | HTTP Requests | `client.run_agent(...)` |
| **State Schema** | `_durable_agent_state.py` | Same Schema (Shared) |
| **Execution** | In-process (Functions Runtime) | In-process (TaskHubWorker) |
| **Platform** | Azure Functions only | Any Durable Task backend |

## Requirements

- Python 3.10+
- `durabletask` - Durable Task SDK for Python
- `agent-framework-core` - Core Agent Framework
- A running Durable Task sidecar or compatible backend

## Documentation

- [Design Document](DESIGN.md) - Detailed architecture and design decisions
- [Simulation Guide](SIMULATION.md) - Simulation design and usage
- [Simulation README](simulation/README.md) - Running the simulation

## Examples

See the [`simulation/`](simulation/) directory for a complete working example that demonstrates:

- Setting up a worker and registering an agent
- Creating a client and calling the agent
- Multi-turn conversations with state persistence
- Proper resource cleanup

## Contributing

Contributions are welcome! Please see the main repository's contributing guidelines.

## License

MIT - See [LICENSE](LICENSE) file for details.