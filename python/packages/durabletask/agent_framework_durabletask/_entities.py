# Copyright (c) Microsoft. All rights reserved.

"""Agent Entity implementation for Durable Task.

This module implements the AgentEntity class that wraps agent execution
in a durable entity, maintaining conversation state across invocations.
"""

import asyncio
from typing import Any, Protocol

from agent_framework import AgentRunResponse, ChatMessage, get_logger
from durabletask.entities import DurableEntity

from ._durable_agent_state import (
    DurableAgentState,
    DurableAgentStateRequest,
    DurableAgentStateResponse,
)
from ._models import RunRequest

logger = get_logger("agent_framework.durabletask.entities")


class AgentProtocol(Protocol):
    """Protocol for agent implementations."""

    @property
    def name(self) -> str:
        """The name of the agent."""
        ...

    async def run(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Run the agent with the given messages."""
        ...


class AgentEntity(DurableEntity):
    """Durable entity that wraps an agent and maintains conversation state.

    This class implements the entity logic for stateful agent execution using
    the Durable Task framework. It persists conversation history and rehydrates
    it on each invocation.

    Note: This class cannot have a custom __init__ due to durabletask SDK constraints.
    Use create_agent_entity() factory function to create instances with injected agents.
    """

    agent: AgentProtocol
    state: DurableAgentState

    def run_agent(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent with the provided input.

        This is the main entity operation that:
        1. Deserializes the input
        2. Updates state with the user message
        3. Rehydrates the full chat history
        4. Executes the agent
        5. Updates state with the response
        6. Returns the serialized response

        Note: This method is synchronous because the durabletask SDK's entity
        framework does not support async entity operations. We use asyncio.run()
        internally to execute the async agent.

        Args:
            input_data: Dictionary containing the RunRequest data

        Returns:
            Dictionary containing the AgentRunResponse data
        """
        # 1. Deserialize Input
        request = RunRequest.from_dict(input_data)
        logger.info(f"AgentEntity '{self.agent.name}' received run request: {request}")

        internal_state = self.get_state()
        logger.info(f"Current state before run: {internal_state}")
        if internal_state:
            self.state = DurableAgentState.from_dict(internal_state)
        else:
            self.state = DurableAgentState()

        # 2. Update State (User Message)
        state_request = DurableAgentStateRequest.from_run_request(request)

        # 3. Rehydrate Chat History - only include messages from history, not the new request yet
        chat_messages: list[ChatMessage] = []
        for entry in self.state.data.conversation_history:
            for msg in entry.messages:
                chat_messages.append(msg.to_chat_message())

        self.state.data.conversation_history.append(state_request)

        logger.info(f"Rehydrated chat history with {len(chat_messages)} messages for agent '{self.agent.name}'")

        logger.info(f"Executing agent '{self.agent.name}' with correlationId: '{state_request.correlation_id}'...")
        # 4. Execute Agent (run async agent synchronously)
        response = asyncio.run(self.agent.run(messages=chat_messages))

        logger.info(f"Agent '{self.agent.name}' execution completed with response: {response}")

        # 5. Update State (Agent Response)
        state_response = DurableAgentStateResponse.from_run_response(request.correlation_id or "", response)
        self.state.data.conversation_history.append(state_response)
        final_state_json = self.state.to_json()
        logger.info(f"Updated state after run: {final_state_json}")

        self.set_state(self.state.to_dict())

        # 6. Return Result (Serialized AgentRunResponse as per design)
        return response.to_dict()

    def reset(self) -> None:
        """Reset the entity state, clearing all conversation history."""
        self.state = DurableAgentState()


def create_agent_entity(agent: AgentProtocol) -> type[AgentEntity]:
    """Factory function to create an AgentEntity class with an injected agent.

    This factory pattern is required because DurableEntity subclasses cannot
    have custom constructors due to durabletask SDK constraints.

    Args:
        agent: The agent instance to inject into the entity

    Returns:
        A new AgentEntity class with the agent pre-configured, named 'dafx-{agent.name}'
    """

    class ConfiguredAgentEntity(AgentEntity):
        def __init__(self):
            self.agent = agent
            self.state = DurableAgentState()

    ConfiguredAgentEntity.__name__ = f"{agent.name}"
    ConfiguredAgentEntity.__qualname__ = f"{agent.name}"

    return ConfiguredAgentEntity
