# Copyright (c) Microsoft. All rights reserved.

"""Mock agent implementation for simulation testing.

This module provides a simple agent implementation designed for testing
the durable task framework. The agent echoes user input and tracks turn count.
"""

from typing import Any

from agent_framework import AgentRunResponse, ChatMessage, TextContent, get_logger

logger = get_logger("agent_framework.durabletask.simulation.mock_agent")


class MockAgent:
    """A simple mock agent for testing that echoes input and tracks turns.

    This agent:
    - Accepts a user message
    - Counts the number of turns based on message history
    - Returns a response formatted as: "Echo: {input} [Turn: {n}]"

    The turn counting verifies that the framework correctly passes state
    and maintains conversation history across invocations.
    """

    def __init__(self, name: str):
        """Initialize the mock agent.

        Args:
            name: The name of the agent
        """
        self.name = name

    async def run(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AgentRunResponse:
        """Run the agent with the provided messages.

        The agent counts the turns (user messages) in the history and
        echoes the last user message with a turn counter.

        Args:
            messages: The full conversation history (re-hydrated by the framework)
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            AgentRunResponse with the echo message and turn count
        """
        # Count user messages to determine turn number
        logger.info(f"MockAgent '{self.name}' received {len(messages)} messages.")
        turn_count = sum(1 for m in messages if m.role.value == "user")

        # Get the last message content
        last_message = "No message"
        if messages:
            last_msg = messages[-1]
            if last_msg.text:
                last_message = last_msg.text
            elif last_msg.contents:
                # Extract text from contents
                for content in last_msg.contents:
                    if isinstance(content, TextContent):
                        last_message = content.text
                        break

        # Format the response
        response_text = f"Echo: {last_message} [Turn: {turn_count}]"

        # Create response message
        response_message = ChatMessage(role="assistant", contents=[TextContent(text=response_text)])

        logger.info(f"MockAgent '{self.name}' sending response: {response_text}")

        return AgentRunResponse(messages=[response_message])
