# Simulation Design: Durable Task Agent

## Overview

This document outlines the design for a simulation to validate and demonstrate the `agent-framework-durabletask` package. The simulation serves as an end-to-end integration test and a developer sample, showing how the `AgentWorker`, `AgentClient`, and `AgentEntity` components interact in a local environment.

## Goals

1.  **Verify Registration**: Ensure `AgentWorker` correctly registers the `AgentEntity` with the underlying `durabletask` worker.
2.  **Verify Execution**: Confirm `AgentClient` can trigger the agent and retrieve results via the `durabletask` backend.
3.  **Verify Persistence**: Demonstrate that the agent's state (conversation history) is preserved across multiple method calls (turns).
4.  **Verify Concurrency**: (Optional) Show multiple agent instances (different IDs) running in parallel.

## Prerequisites

*   **Durable Task Sidecar**: The simulation requires a running Durable Task sidecar (e.g., the Dapr sidecar or the standalone Durable Task sidecar) or a compatible backend emulator.
*   **Python Environment**: The `agent-framework-durabletask` package installed (or available in path).

## Components

### 1. The Mock Agent (`MockAgent`)

A simple implementation of `AgentProtocol` designed for testing.

*   **Behavior**:
    *   Accepts a user message.
    *   Appends it to an internal memory (which is re-hydrated by the framework).
    *   Returns a response string formatted as: `Echo: {input} [Turn: {n}]`.
*   **Purpose**: To verify that the framework correctly passes state and inputs to the agent.

### 2. The Simulation Driver (`simulate.py`)

A Python script that orchestrates the simulation.

#### Workflow

1.  **Initialization**:
    *   Instantiate a `TaskHubGrpcWorker` (connected to the local sidecar).
    *   Instantiate a `TaskHubGrpcClient` (connected to the local sidecar).
    *   Instantiate the `MockAgent`.
    *   Wrap the worker: `agent_worker = AgentWorker(grpc_worker)`.
    *   Register the agent: `agent_worker.add_agent(mock_agent)`.

2.  **Startup**:
    *   Start the worker in a background `asyncio.Task`.

3.  **Scenario Execution**:
    *   **Turn 1**:
        *   Client sends "Hello World".
        *   Client polls for result.
        *   **Assert**: Result is "Echo: Hello World [Turn: 1]".
    *   **Turn 2**:
        *   Client sends "How are you?".
        *   Client polls for result.
        *   **Assert**: Result is "Echo: How are you? [Turn: 2]".
        *   *Note*: This proves that the `AgentEntity` correctly reloaded the previous state (Turn 1) before processing Turn 2.

4.  **Teardown**:
    *   Stop the worker.
    *   Cancel background tasks.

## Implementation Plan

### File Structure

```text
packages/durabletask/
└── simulation/
    ├── __init__.py
    ├── simulate.py       # The main driver script
    └── mock_agent.py     # The simple agent implementation
```

### Code Sketches

#### `mock_agent.py`

```python
from agent_framework import Agent, AgentRunResponse
from typing import List

class MockAgent:
    def __init__(self, name: str):
        self.name = name

    async def run(self, messages: List[any], **kwargs) -> AgentRunResponse:
        # The framework passes the full history in 'messages'
        turn_count = len(messages)
        last_message = messages[-1].content
        
        response_text = f"Echo: {last_message} [Turn: {turn_count}]"
        
        return AgentRunResponse(content=response_text, messages=[...])
```

#### `simulate.py`

```python
import asyncio
from durabletask import TaskHubGrpcWorker, TaskHubGrpcClient
from agent_framework_durabletask import AgentWorker, AgentClient
from .mock_agent import MockAgent

async def main():
    # Configuration (Assumes local sidecar on default port)
    host = "localhost:4001"
    
    # Setup
    agent = MockAgent("sim-agent")
    
    # Worker
    durable_worker = TaskHubGrpcWorker(host_address=host)
    worker = AgentWorker(durable_worker)
    worker.add_agent(agent)
    
    # Client
    durable_client = TaskHubGrpcClient(host_address=host)
    client = AgentClient(durable_client)
    
    # Start Worker
    worker_task = asyncio.create_task(worker.start())
    print("Worker started.")
    
    try:
        # Turn 1
        print("--- Turn 1 ---")
        response1 = await client.run_agent("sim-agent", "Hello")
        print(f"Response: {response1.content}")
        
        # Turn 2
        print("--- Turn 2 ---")
        response2 = await client.run_agent("sim-agent", "Again")
        print(f"Response: {response2.content}")
        
    finally:
        await worker.stop()
        await worker_task

if __name__ == "__main__":
    asyncio.run(main())
```

## Success Criteria

The simulation is considered successful if:
1.  The script runs without unhandled exceptions.
2.  The output shows correct responses for both turns.
3.  The "Turn" counter increments, proving state retention.