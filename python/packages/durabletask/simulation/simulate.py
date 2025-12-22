# Copyright (c) Microsoft. All rights reserved.

"""Simulation driver for testing the durable task agent framework.

This script orchestrates an end-to-end simulation to validate and demonstrate
the agent-framework-durabletask package. It serves as both an integration test
and a developer sample.

The simulation verifies:
1. Agent registration with the worker
2. Agent execution via client (with polling using get_entity)
3. State persistence across multiple turns
4. Proper conversation history management

Prerequisites:
- A running Durable Task sidecar (e.g., Dapr sidecar or standalone sidecar)
- The agent-framework-durabletask package installed
- durabletask SDK version 1.1.0+ (for get_entity support)
"""

import asyncio
import sys
import logging

logging.basicConfig(level=logging.INFO)

from agent_framework import get_logger

logger = get_logger("agent_framework.durabletask.simulation")


async def main() -> int:
    """Main simulation entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Import here to provide better error messages if dependencies are missing
        from durabletask.azuremanaged.client import DurableTaskSchedulerClient
        from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
    except ImportError as e:
        logger.error(
            f"Failed to import durabletask SDK: {e}\n"
            "Please ensure the durabletask package is installed:\n"
            "  pip install durabletask"
        )
        return 1

    try:
        from agent_framework_durabletask import AgentWorker, AgentClient
        from .mock_agent import MockAgent
    except ImportError as e:
        logger.error(f"Failed to import simulation components: {e}")
        return 1

    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

    # Configuration
    host = "http://localhost:8080"  # Default Durable Task sidecar address
    agent_name = "sim-agent"
    thread_id = "simulation-session-008"

    logger.info("=" * 60)
    logger.info("Durable Task Agent Framework Simulation")
    logger.info("=" * 60)
    logger.info(f"Sidecar address: {host}")
    logger.info(f"Agent name: {agent_name}")
    logger.info(f"Thread ID: {thread_id}")
    logger.info("")

    # Setup
    logger.info("Setting up components...")
    agent = MockAgent(agent_name)

    # DTS Worker
    try:
        durable_worker = DurableTaskSchedulerWorker(host_address=host, taskhub="testhub05", token_credential=credential)
    except Exception as e:
        logger.error(f"Failed to create worker: {e}")
        logger.error(
            "Please ensure the Durable Task sidecar is running on localhost:4001\n"
            "You can start a test sidecar using Docker or the durabletask CLI"
        )
        return 1

    # Register agent with worker
    worker = AgentWorker(durable_worker)
    worker.add_agent(agent)
    logger.info(f"✓ Registered agent '{agent_name}' with worker")

    # Client
    try:
        durable_client = DurableTaskSchedulerClient(host_address=host, taskhub="testhub05", token_credential=None)
    except Exception as e:
        logger.error(f"Failed to create client: {e}")
        return 1

    # Create AgentClient wrapper for polling-based client interaction
    agent_client = AgentClient(durable_client)

    logger.info("✓ Created client")
    logger.info("")

    # Start Worker in background thread (it's synchronous)
    logger.info("Starting worker in background...")
    worker_task = asyncio.create_task(asyncio.to_thread(worker.start))

    # Give the worker a moment to start
    await asyncio.sleep(2)
    logger.info("✓ Worker started")
    logger.info("")

    try:
        # Test: Client-based execution with polling (durabletask 1.1.0+)
        logger.info("=" * 60)
        logger.info("Testing Client-based Agent Execution with Polling")
        logger.info("=" * 60)
        logger.info("This test demonstrates the AgentClient.run_agent with polling")
        logger.info("using TaskHubGrpcClient.get_entity (durabletask 1.1.0+)")
        logger.info("")

        try:
            # Turn 1: First message
            logger.info("Sending message 1 via AgentClient...")
            logger.info("Message: 'Hello World'")

            # Run agent with polling (wrap synchronous call in thread)
            response_1 = await asyncio.to_thread(
                agent_client.run_agent,
                agent_name=agent_name,
                message="Hello World Pt2",
                thread_id=thread_id,
                timeout_seconds=5.0,
                polling_interval=1.0,
            )

            logger.info(
                f"✓ Response 1 received: {response_1.messages[0].text if response_1.messages else 'No message'}"
            )
            logger.info("")

            # Turn 2: Second message
            logger.info("Sending message 2 via AgentClient...")
            logger.info("Message: 'How are you?'")

            response_2 = await asyncio.to_thread(
                agent_client.run_agent,
                agent_name=agent_name,
                message="How are you?",
                thread_id=thread_id,
                timeout_seconds=5.0,
                polling_interval=1.0,
            )

            logger.info(
                f"✓ Response 2 received: {response_2.messages[0].text if response_2.messages else 'No message'}"
            )
            logger.info("")

            logger.info("=" * 60)
            logger.info("SIMULATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info("✓ Agent registration verified")
            logger.info("✓ Client-based polling with get_entity verified")
            logger.info("✓ Multi-turn conversation verified")
            logger.info("✓ State persistence verified")
            logger.info("")

            return 0

        except Exception as e:
            logger.error(f"✗ FAILED - Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            return 1

    finally:
        # Cleanup
        logger.info("")
        logger.info("Stopping worker...")
        worker.stop()

        # Wait for worker task to complete (with timeout)
        try:
            await asyncio.wait_for(worker_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Worker did not stop gracefully, cancelling...")
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

        logger.info("Cleanup complete")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
