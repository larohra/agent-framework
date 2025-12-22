# Copyright (c) Microsoft. All rights reserved.

"""Worker wrapper for Durable Task agents.

This module provides the AgentWorker class that wraps a durabletask worker
and simplifies agent registration.
"""

from agent_framework import get_logger
from durabletask.worker import TaskHubGrpcWorker

from ._entities import AgentProtocol, create_agent_entity

logger = get_logger("agent_framework.durabletask.worker")


class AgentWorker:
    """Wrapper around a Durable Task worker that simplifies agent registration.

    This class provides a high-level interface for registering agents as durable
    entities and managing the worker lifecycle.
    """

    def __init__(self, worker: TaskHubGrpcWorker):
        """Initialize the agent worker.

        Args:
            worker: A durabletask worker instance (TaskHubGrpcWorker)
        """
        self._worker = worker
        self._registered_agents: dict[str, AgentProtocol] = {}

    def add_agent(self, agent: AgentProtocol) -> None:
        """Register an agent with the worker.

        Uses the factory pattern to create an AgentEntity class with the agent
        instance injected, then registers it with the durabletask worker.

        Args:
            agent: The agent to register (must have a 'name' property)
        """
        # Store the agent reference
        self._registered_agents[agent.name] = agent

        # Create a configured entity class using the factory
        entity_class = create_agent_entity(agent)

        # Register the entity class with the worker
        entity_registered: str = self._worker.add_entity(entity_class)  # pyright: ignore[reportUnknownMemberType]

        logger.info(f"Registered agent '{agent.name}' with entity: {entity_registered}")

    def start(self) -> None:
        """Start the worker to begin processing tasks."""
        logger.info("Starting agent worker...")
        self._worker.start()

    def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("Stopping agent worker...")
        if hasattr(self._worker, "stop"):
            self._worker.stop()
