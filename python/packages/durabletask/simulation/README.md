# Durable Task Agent Framework Simulation

This directory contains a simulation for testing and demonstrating the `agent-framework-durabletask` package.

## Purpose

The simulation serves multiple purposes:

1. **Integration Test**: Validates that all components work together correctly
2. **Developer Sample**: Demonstrates how to use the package
3. **Verification Tool**: Confirms proper state persistence and conversation management

## What It Tests

The simulation verifies:

- ✅ Agent registration with the Durable Task worker
- ✅ Agent execution via the client
- ✅ State persistence across multiple turns
- ✅ Conversation history management
- ✅ Turn counting (proving state is maintained)

## Components

### `mock_agent.py`

A simple agent implementation that:
- Echoes user input
- Tracks turn count based on conversation history
- Returns responses in the format: `Echo: {input} [Turn: {n}]`

This simple behavior makes it easy to verify that the framework correctly maintains and rehydrates state.

### `simulate.py`

The main driver script that:
1. Sets up a mock agent
2. Creates and configures worker and client
3. Starts the worker in the background
4. Executes a multi-turn conversation
5. Verifies responses and state persistence
6. Cleans up resources

## Prerequisites

Before running the simulation, you need:

1. **Durable Task Sidecar**: A running Durable Task sidecar on `localhost:4001`
   - You can use Dapr, the standalone Durable Task sidecar, or a compatible backend

2. **Python Dependencies**:
   ```bash
   pip install durabletask agent-framework-core agent-framework-durabletask
   ```

## Running the Simulation

### Option 1: Direct Execution

```bash
cd packages/durabletask
python -m simulation.simulate
```

### Option 2: As a Module

```bash
python -m agent_framework_durabletask.simulation.simulate
```

## Expected Output

When successful, you should see output like:

```
============================================================
Durable Task Agent Framework Simulation
============================================================
Sidecar address: localhost:4001
Agent name: sim-agent

Setting up components...
✓ Registered agent 'sim-agent' with worker
✓ Created agent client

Starting worker in background...
✓ Worker started

============================================================
Turn 1: Sending 'Hello World'
============================================================
Response: Echo: Hello World [Turn: 1]
✓ Turn 1 PASSED - Response matches expected format

============================================================
Turn 2: Sending 'How are you?'
============================================================
Response: Echo: How are you? [Turn: 2]
✓ Turn 2 PASSED - Response matches expected format
✓ STATE PERSISTENCE VERIFIED - Turn counter incremented correctly

============================================================
SIMULATION COMPLETED SUCCESSFULLY
============================================================
✓ Agent registration verified
✓ Agent execution verified
✓ State persistence verified
✓ Multi-turn conversation verified
```

## Troubleshooting

### "Failed to create worker" Error

This usually means the Durable Task sidecar is not running. Start the sidecar first:

```bash
# Using Dapr
dapr run --app-id myapp --dapr-grpc-port 4001

# Or using standalone sidecar (if available)
durabletask-sidecar --port 4001
```

### Timeout Errors

If you see timeout errors, check:
1. Is the sidecar running and accessible?
2. Is the port (4001) correct and not blocked by firewall?
3. Are the worker and client properly configured?

### Import Errors

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Customization

You can customize the simulation by:

1. **Changing the sidecar address**: Modify the `host` variable in `simulate.py`
2. **Adding more test cases**: Add additional turns or test scenarios
3. **Using a different agent**: Replace `MockAgent` with your own implementation
4. **Adjusting timeouts**: Modify the `timeout_seconds` parameter

## Next Steps

After running the simulation successfully:

1. Explore the implementation files in `agent_framework_durabletask/`
2. Try creating your own agents using the framework
3. Integrate with your production Durable Task backend
4. Build stateful, distributed agent applications!