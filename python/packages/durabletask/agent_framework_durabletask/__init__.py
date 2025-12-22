# Copyright (c) Microsoft. All rights reserved.

"""Durable Task integration for Microsoft Agent Framework.

This package provides a durability layer for the Microsoft Agent Framework using
the durabletask Python SDK. It enables stateful, reliable, and distributed agent
execution on any platform.
"""

from ._client import AgentClient
from ._entities import AgentEntity, AgentProtocol, create_agent_entity
from ._models import RunRequest
from ._worker import AgentWorker

__all__ = [
    "AgentClient",
    "AgentEntity",
    "AgentProtocol",
    "AgentWorker",
    "RunRequest",
    "create_agent_entity",
]
