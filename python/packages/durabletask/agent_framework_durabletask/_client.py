# Copyright (c) Microsoft. All rights reserved.

"""Client wrapper for Durable Task agents.

This module provides the AgentClient class for external interactions with
durable agents via the Durable Task backend.
"""

import time
import uuid

from agent_framework import AgentRunResponse, get_logger
from durabletask.client import TaskHubGrpcClient
from durabletask.task import EntityInstanceId

from ._durable_agent_state import DurableAgentState, DurableAgentStateResponse
from ._models import RunRequest

logger = get_logger("agent_framework.durabletask.client")


class AgentClient:
    """Client for interacting with durable agents.

    This class provides a high-level interface for external clients (e.g., FastAPI, CLI)
    to interact with agents running on the Durable Task backend. It polls for agent
    responses using the get_entity API introduced in durabletask 1.1.0.
    """

    def __init__(self, client: TaskHubGrpcClient):
        """Initialize the agent client.

        Args:
            client: A durabletask client instance (TaskHubGrpcClient)
        """
        self._client = client

    def run_agent(
        self,
        agent_name: str,
        message: str,
        thread_id: str | None = None,
        timeout_seconds: float = 60.0,
        polling_interval: float = 1.0,
    ) -> AgentRunResponse:
        """Runs an agent and polls for the result.

        This method sends a message to the agent entity and then polls the entity
        state using get_entity (durabletask 1.1.0+) until the response is available
        or the timeout is reached.

        Args:
            agent_name: The name of the agent to run
            message: The message to send to the agent
            thread_id: Optional thread ID (defaults to a new UUID)
            timeout_seconds: Maximum time to wait for response (default: 60s)
            polling_interval: Time between polling attempts (default: 1s)

        Returns:
            AgentRunResponse: The response from the agent

        Raises:
            TimeoutError: If the agent execution times out
        """
        thread_id = thread_id or str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        entity_id = EntityInstanceId(f"{agent_name}", thread_id)

        request = RunRequest(
            message=message,
            thread_id=thread_id,
            correlation_id=correlation_id,
        )

        logger.info(f"Sending message to agent '{agent_name}' (thread: {thread_id}, correlation: {correlation_id})")

        # 1. Signal Entity
        # Note: signal_entity is synchronous in TaskHubGrpcClient
        self._client.signal_entity(entity_id, "run_agent", request.to_dict())

        logger.info(f"Message sent, polling for response (timeout: {timeout_seconds}s, interval: {polling_interval}s)")

        # 2. Poll for Status
        start_time = time.time()
        while (time.time() - start_time) < timeout_seconds:
            # Fetch entity state using the new get_entity API (durabletask 1.1.0)
            # Note: get_entity is synchronous in TaskHubGrpcClient
            state = self._client.get_entity(entity_id)
            logger.info(f"Polled entity state for '{agent_name}' (thread: {thread_id}): {state}")
            if state:
                # Deserialize state
                try:
                    agent_state = state.get_state()
                    if not agent_state:
                        raise ValueError("Agent state is None")

                    logger.info(f"Raw agent state: {agent_state}")

                    agent_state = DurableAgentState.from_json(agent_state)
                    logger.info(f"Deserialized agent state: {agent_state.to_json()}")

                    # Find response matching correlation_id
                    for entry in reversed(agent_state.data.conversation_history):
                        logger.info(f"Checking conversation history entry: {entry}, type: {type(entry)}")
                        if isinstance(entry, DurableAgentStateResponse) and entry.correlation_id == correlation_id:
                            logger.info(f"Response received for correlation_id: {correlation_id}")
                            return entry.to_run_response()
                    logger.info(f"No response yet for correlation_id: {correlation_id}, continuing to poll...")

                except Exception as e:
                    logger.exception(f"Failed to deserialize entity state: {e}")

            time.sleep(polling_interval)

        raise TimeoutError(f"Agent execution timed out after {timeout_seconds}s")

    def reset_agent(
        self,
        agent_name: str,
        thread_id: str,
    ) -> None:
        """Reset an agent's conversation state.

        Args:
            agent_name: The name of the agent to reset
            thread_id: The thread ID to reset
        """
        entity_id = EntityInstanceId(f"agent:{agent_name}", thread_id)

        logger.info(f"Resetting agent '{agent_name}' (thread: {thread_id})")

        self._client.signal_entity(entity_id, "reset", None)
