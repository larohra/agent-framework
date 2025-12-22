# Copyright (c) Microsoft. All rights reserved.

"""Durable agent state management conforming to the durable-agent-entity-state.json schema.

This module provides classes for managing conversation state in Durable Task agents.
It implements the versioned schema that defines how agent conversations are persisted and restored
across invocations, enabling stateful, long-running agent sessions.

Ported from azurefunctions package for use with durabletask.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agent_framework import (
    AgentRunResponse,
    BaseContent,
    ChatMessage,
    DataContent,
    ErrorContent,
    FunctionCallContent,
    FunctionResultContent,
    HostedFileContent,
    HostedVectorStoreContent,
    TextContent,
    TextReasoningContent,
    UriContent,
    UsageContent,
    UsageDetails,
    get_logger,
)
from dateutil import parser as date_parser

from ._models import RunRequest, _serialize_response_format

logger = get_logger("agent_framework.durabletask.durable_agent_state")


def _parse_created_at(value: Any) -> datetime:
    """Normalize created_at values coming from persisted durable state."""
    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            parsed = date_parser.parse(value)
            if isinstance(parsed, datetime):
                return parsed
        except (ValueError, TypeError):
            pass

    return datetime.now(tz=timezone.utc)


class DurableAgentStateContent:
    """Base class for all content types in durable agent state messages."""

    extensionData: dict[str, Any] | None = None
    type: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this content to a dictionary for JSON storage."""
        raise NotImplementedError

    def to_ai_content(self) -> Any:
        """Convert this durable state content back to an agent framework content object."""
        raise NotImplementedError

    @staticmethod
    def from_ai_content(content: Any) -> DurableAgentStateContent:
        """Create a durable state content object from an agent framework content object."""
        if isinstance(content, DataContent):
            return DurableAgentStateDataContent.from_data_content(content)
        if isinstance(content, ErrorContent):
            return DurableAgentStateErrorContent.from_error_content(content)
        if isinstance(content, FunctionCallContent):
            return DurableAgentStateFunctionCallContent.from_function_call_content(content)
        if isinstance(content, FunctionResultContent):
            return DurableAgentStateFunctionResultContent.from_function_result_content(content)
        if isinstance(content, HostedFileContent):
            return DurableAgentStateHostedFileContent.from_hosted_file_content(content)
        if isinstance(content, HostedVectorStoreContent):
            return DurableAgentStateHostedVectorStoreContent.from_hosted_vector_store_content(content)
        if isinstance(content, TextContent):
            return DurableAgentStateTextContent.from_text_content(content)
        if isinstance(content, TextReasoningContent):
            return DurableAgentStateTextReasoningContent.from_text_reasoning_content(content)
        if isinstance(content, UriContent):
            return DurableAgentStateUriContent.from_uri_content(content)
        if isinstance(content, UsageContent):
            return DurableAgentStateUsageContent.from_usage_content(content)
        return DurableAgentStateUnknownContent.from_unknown_content(content)


# Core state classes


class DurableAgentStateData:
    """Container for the core data within durable agent state."""

    conversation_history: list[DurableAgentStateEntry]
    extension_data: dict[str, Any] | None

    def __init__(
        self,
        conversation_history: list[DurableAgentStateEntry] | None = None,
        extension_data: dict[str, Any] | None = None,
    ) -> None:
        self.conversation_history = conversation_history or []
        self.extension_data = extension_data

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "conversationHistory": [entry.to_dict() for entry in self.conversation_history],
        }
        if self.extension_data is not None:
            result["extensionData"] = self.extension_data
        return result

    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> DurableAgentStateData:
        history_data = data_dict.get("conversationHistory", [])
        deserialized_history: list[DurableAgentStateEntry] = []
        for entry_dict in history_data:
            if isinstance(entry_dict, dict):
                entry_type = entry_dict.get("$type") or entry_dict.get("json_type")
                if entry_type == DurableAgentStateEntryJsonType.RESPONSE:
                    deserialized_history.append(DurableAgentStateResponse.from_dict(entry_dict))
                elif entry_type == DurableAgentStateEntryJsonType.REQUEST:
                    deserialized_history.append(DurableAgentStateRequest.from_dict(entry_dict))
                else:
                    deserialized_history.append(DurableAgentStateEntry.from_dict(entry_dict))
            else:
                deserialized_history.append(entry_dict)

        return cls(
            conversation_history=deserialized_history,
            extension_data=data_dict.get("extensionData"),
        )


class DurableAgentState:
    """Manages durable agent state conforming to the durable-agent-entity-state.json schema."""

    data: DurableAgentStateData
    schema_version: str = "1.0.0"

    def __init__(self, schema_version: str = "1.0.0"):
        self.data = DurableAgentStateData()
        self.schema_version = schema_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "data": self.data.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> DurableAgentState:
        schema_version = state.get("schemaVersion")
        if schema_version is None:
            logger.warning("Resetting state as it is incompatible with the current schema, all history will be lost")
            return cls()

        instance = cls(schema_version=state.get("schemaVersion", "1.0.0"))
        instance.data = DurableAgentStateData.from_dict(state.get("data", {}))

        return instance

    @classmethod
    def from_json(cls, json_str: str) -> DurableAgentState:
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError("The durable agent state is not valid JSON.") from e

        logger.info(f"Deserializing DurableAgentState from JSON: {obj}")
        return cls.from_dict(obj)

    @property
    def message_count(self) -> int:
        """Get the count of conversation entries (requests + responses)."""
        return len(self.data.conversation_history)

    def try_get_agent_response_entry(self, correlation_id: str) -> DurableAgentStateResponse | None:
        """Try to get an agent response entry by correlation ID."""
        for entry in self.data.conversation_history:
            if entry.correlation_id == correlation_id and isinstance(entry, DurableAgentStateResponse):
                return entry
        return None


class DurableAgentStateEntryJsonType(str, Enum):
    """Enum for conversation history entry types."""

    REQUEST = "request"
    RESPONSE = "response"


class DurableAgentStateEntry:
    """Base class for conversation history entries (requests and responses)."""

    json_type: DurableAgentStateEntryJsonType
    correlation_id: str | None
    created_at: datetime
    messages: list[DurableAgentStateMessage]
    extension_data: dict[str, Any] | None

    def __init__(
        self,
        json_type: DurableAgentStateEntryJsonType,
        correlation_id: str | None,
        created_at: datetime,
        messages: list[DurableAgentStateMessage],
        extension_data: dict[str, Any] | None = None,
    ) -> None:
        self.json_type = json_type
        self.correlation_id = correlation_id
        self.created_at = created_at
        self.messages = messages
        self.extension_data = extension_data

    def to_dict(self) -> dict[str, Any]:
        created_at_value = self.created_at
        if created_at_value is None:
            created_at_value = datetime.now(tz=timezone.utc)

        return {
            "$type": self.json_type,
            "correlationId": self.correlation_id,
            "createdAt": created_at_value.isoformat() if isinstance(created_at_value, datetime) else created_at_value,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableAgentStateEntry:
        created_at = _parse_created_at(data.get("createdAt"))

        messages = []
        for msg_dict in data.get("messages", []):
            if isinstance(msg_dict, dict):
                messages.append(DurableAgentStateMessage.from_dict(msg_dict))
            else:
                messages.append(msg_dict)

        return cls(
            json_type=DurableAgentStateEntryJsonType(data.get("$type", "entry")),
            correlation_id=data.get("correlationId", ""),
            created_at=created_at,
            messages=messages,
            extension_data=data.get("extensionData"),
        )


class DurableAgentStateRequest(DurableAgentStateEntry):
    """Represents a request entry in the durable agent conversation history."""

    response_type: str | None = None
    response_schema: dict[str, Any] | None = None

    def __init__(
        self,
        correlation_id: str | None,
        created_at: datetime,
        messages: list[DurableAgentStateMessage],
        extension_data: dict[str, Any] | None = None,
        response_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            json_type=DurableAgentStateEntryJsonType.REQUEST,
            correlation_id=correlation_id,
            created_at=created_at,
            messages=messages,
            extension_data=extension_data,
        )
        self.response_type = response_type
        self.response_schema = response_schema

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.response_type is not None:
            data["responseType"] = self.response_type
        if self.response_schema is not None:
            data["responseSchema"] = self.response_schema
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableAgentStateRequest:
        created_at = _parse_created_at(data.get("createdAt"))

        messages = []
        for msg_dict in data.get("messages", []):
            if isinstance(msg_dict, dict):
                messages.append(DurableAgentStateMessage.from_dict(msg_dict))
            else:
                messages.append(msg_dict)

        return cls(
            correlation_id=data.get("correlationId", ""),
            created_at=created_at,
            messages=messages,
            extension_data=data.get("extensionData"),
            response_type=data.get("responseType"),
            response_schema=data.get("responseSchema"),
        )

    @staticmethod
    def from_run_request(request: RunRequest) -> DurableAgentStateRequest:
        return DurableAgentStateRequest(
            correlation_id=request.correlation_id,
            messages=[DurableAgentStateMessage.from_run_request(request)],
            created_at=datetime.now(tz=timezone.utc),
            response_type=request.request_response_format,
            response_schema=_serialize_response_format(request.response_format),
        )


class DurableAgentStateResponse(DurableAgentStateEntry):
    """Represents a response entry in the durable agent conversation history."""

    usage: DurableAgentStateUsage | None = None
    is_error: bool = False

    def __init__(
        self,
        correlation_id: str,
        created_at: datetime,
        messages: list[DurableAgentStateMessage],
        extension_data: dict[str, Any] | None = None,
        usage: DurableAgentStateUsage | None = None,
        is_error: bool = False,
    ) -> None:
        super().__init__(
            json_type=DurableAgentStateEntryJsonType.RESPONSE,
            correlation_id=correlation_id,
            created_at=created_at,
            messages=messages,
            extension_data=extension_data,
        )
        self.usage = usage
        self.is_error = is_error

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.usage is not None:
            data["usage"] = self.usage.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableAgentStateResponse:
        created_at = _parse_created_at(data.get("createdAt"))

        messages = []
        for msg_dict in data.get("messages", []):
            if isinstance(msg_dict, dict):
                messages.append(DurableAgentStateMessage.from_dict(msg_dict))
            else:
                messages.append(msg_dict)

        usage_dict = data.get("usage")
        usage = None
        if usage_dict and isinstance(usage_dict, dict):
            usage = DurableAgentStateUsage.from_dict(usage_dict)
        elif usage_dict:
            usage = usage_dict

        return cls(
            correlation_id=data.get("correlationId", ""),
            created_at=created_at,
            messages=messages,
            extension_data=data.get("extensionData"),
            usage=usage,
        )

    @staticmethod
    def from_run_response(correlation_id: str, response: AgentRunResponse) -> DurableAgentStateResponse:
        """Creates a DurableAgentStateResponse from an AgentRunResponse."""
        return DurableAgentStateResponse(
            correlation_id=correlation_id,
            created_at=_parse_created_at(response.created_at),
            messages=[DurableAgentStateMessage.from_chat_message(m) for m in response.messages],
            usage=DurableAgentStateUsage.from_usage(response.usage_details),
        )

    def to_run_response(self) -> AgentRunResponse:
        """Converts this DurableAgentStateResponse back to an AgentRunResponse."""
        return AgentRunResponse(
            created_at=self.created_at.isoformat() if self.created_at else None,
            messages=[m.to_chat_message() for m in self.messages],
            usage=self.usage.to_usage_details() if self.usage else None,
        )


class DurableAgentStateMessage:
    """Represents a message within a conversation history entry."""

    role: str
    contents: list[DurableAgentStateContent]
    author_name: str | None = None
    created_at: datetime | None = None
    extension_data: dict[str, Any] | None = None

    def __init__(
        self,
        role: str,
        contents: list[DurableAgentStateContent],
        author_name: str | None = None,
        created_at: datetime | None = None,
        extension_data: dict[str, Any] | None = None,
    ) -> None:
        self.role = role
        self.contents = contents
        self.author_name = author_name
        self.created_at = created_at
        self.extension_data = extension_data

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "contents": [
                {"$type": c.to_dict().get("type", "text"), **{k: v for k, v in c.to_dict().items() if k != "type"}}
                for c in self.contents
            ],
        }
        if self.created_at is not None:
            result["createdAt"] = self.created_at.isoformat()
        if self.author_name is not None:
            result["authorName"] = self.author_name
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableAgentStateMessage:
        contents: list[DurableAgentStateContent] = []
        for content_dict in data.get("contents", []):
            if isinstance(content_dict, dict):
                content_type = content_dict.get("$type")
                if content_type == DurableAgentStateTextContent.type:
                    contents.append(DurableAgentStateTextContent(text=content_dict.get("text")))
                elif content_type == DurableAgentStateDataContent.type:
                    contents.append(
                        DurableAgentStateDataContent(
                            uri=content_dict.get("uri", ""), media_type=content_dict.get("mediaType")
                        )
                    )
                elif content_type == DurableAgentStateErrorContent.type:
                    contents.append(
                        DurableAgentStateErrorContent(
                            message=content_dict.get("message"),
                            error_code=content_dict.get("errorCode"),
                            details=content_dict.get("details"),
                        )
                    )
                elif content_type == DurableAgentStateFunctionCallContent.type:
                    contents.append(
                        DurableAgentStateFunctionCallContent(
                            call_id=content_dict.get("callId", ""),
                            name=content_dict.get("name", ""),
                            arguments=content_dict.get("arguments", {}),
                        )
                    )
                elif content_type == DurableAgentStateFunctionResultContent.type:
                    contents.append(
                        DurableAgentStateFunctionResultContent(
                            call_id=content_dict.get("callId", ""), result=content_dict.get("result")
                        )
                    )
                elif content_type == DurableAgentStateHostedFileContent.type:
                    contents.append(DurableAgentStateHostedFileContent(file_id=content_dict.get("fileId", "")))
                elif content_type == DurableAgentStateHostedVectorStoreContent.type:
                    contents.append(
                        DurableAgentStateHostedVectorStoreContent(vector_store_id=content_dict.get("vectorStoreId", ""))
                    )
                elif content_type == DurableAgentStateTextReasoningContent.type:
                    contents.append(DurableAgentStateTextReasoningContent(text=content_dict.get("text")))
                elif content_type == DurableAgentStateUriContent.type:
                    contents.append(
                        DurableAgentStateUriContent(
                            uri=content_dict.get("uri", ""), media_type=content_dict.get("mediaType", "")
                        )
                    )
                elif content_type == DurableAgentStateUsageContent.type:
                    usage_data = content_dict.get("usage")
                    if usage_data and isinstance(usage_data, dict):
                        contents.append(
                            DurableAgentStateUsageContent(usage=DurableAgentStateUsage.from_dict(usage_data))
                        )
                elif content_type == DurableAgentStateUnknownContent.type:
                    contents.append(DurableAgentStateUnknownContent(content=content_dict.get("content", {})))
            else:
                contents.append(content_dict)  # type: ignore

        return cls(
            role=data.get("role", ""),
            contents=contents,
            author_name=data.get("authorName"),
            created_at=_parse_created_at(data.get("createdAt")) if data.get("createdAt") else None,
            extension_data=data.get("extensionData"),
        )

    @property
    def text(self) -> str | None:
        """Extract text from the contents list."""
        text_parts = []
        for content in self.contents:
            if isinstance(content, DurableAgentStateTextContent):
                text_parts.append(content.text or "")
        return "".join(text_parts) if text_parts else None

    @staticmethod
    def from_run_request(request: RunRequest) -> DurableAgentStateMessage:
        """Converts a RunRequest to a DurableAgentStateMessage."""
        return DurableAgentStateMessage(
            role=request.role.value,
            contents=[DurableAgentStateTextContent(text=request.message)],
            created_at=_parse_created_at(request.created_at) if request.created_at else None,
        )

    @staticmethod
    def from_chat_message(chat_message: ChatMessage) -> DurableAgentStateMessage:
        """Converts an Agent Framework chat message to a durable state message."""
        contents_list: list[DurableAgentStateContent] = [
            DurableAgentStateContent.from_ai_content(c) for c in chat_message.contents
        ]

        return DurableAgentStateMessage(
            role=chat_message.role.value,
            contents=contents_list,
            author_name=chat_message.author_name,
            extension_data=dict(chat_message.additional_properties) if chat_message.additional_properties else None,
        )

    def to_chat_message(self) -> ChatMessage:
        """Converts this DurableAgentStateMessage back to an agent framework ChatMessage."""
        ai_contents = [c.to_ai_content() for c in self.contents]

        kwargs: dict[str, Any] = {
            "role": self.role,
            "contents": ai_contents,
        }

        if self.author_name is not None:
            kwargs["author_name"] = self.author_name

        if self.extension_data is not None:
            kwargs["additional_properties"] = self.extension_data

        return ChatMessage(**kwargs)


# Content type classes (abbreviated - include key ones for the simulation)


class DurableAgentStateTextContent(DurableAgentStateContent):
    """Represents plain text content in messages."""

    type: str = "text"

    def __init__(self, text: str | None) -> None:
        self.text = text

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}

    @staticmethod
    def from_text_content(content: TextContent) -> DurableAgentStateTextContent:
        return DurableAgentStateTextContent(text=content.text)

    def to_ai_content(self) -> TextContent:
        return TextContent(text=self.text or "")


class DurableAgentStateDataContent(DurableAgentStateContent):
    """Represents data content with a URI reference."""

    uri: str = ""
    media_type: str | None = None
    type: str = "data"

    def __init__(self, uri: str, media_type: str | None = None) -> None:
        self.uri = uri
        self.media_type = media_type

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "uri": self.uri, "mediaType": self.media_type}

    @staticmethod
    def from_data_content(content: DataContent) -> DurableAgentStateDataContent:
        return DurableAgentStateDataContent(uri=content.uri, media_type=content.media_type)

    def to_ai_content(self) -> DataContent:
        return DataContent(uri=self.uri, media_type=self.media_type)


class DurableAgentStateErrorContent(DurableAgentStateContent):
    """Represents error content in agent responses."""

    message: str | None = None
    error_code: str | None = None
    details: str | None = None
    type: str = "error"

    def __init__(self, message: str | None = None, error_code: str | None = None, details: str | None = None) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "message": self.message, "errorCode": self.error_code, "details": self.details}

    @staticmethod
    def from_error_content(content: ErrorContent) -> DurableAgentStateErrorContent:
        return DurableAgentStateErrorContent(
            message=content.message, error_code=content.error_code, details=content.details
        )

    def to_ai_content(self) -> ErrorContent:
        return ErrorContent(message=self.message, error_code=self.error_code, details=self.details)


class DurableAgentStateFunctionCallContent(DurableAgentStateContent):
    """Represents a function/tool call request from the agent."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    type: str = "functionCall"

    def __init__(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        self.call_id = call_id
        self.name = name
        self.arguments = arguments

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "callId": self.call_id, "name": self.name, "arguments": self.arguments}

    @staticmethod
    def from_function_call_content(content: FunctionCallContent) -> DurableAgentStateFunctionCallContent:
        arguments: dict[str, Any] = {}
        if content.arguments:
            if isinstance(content.arguments, dict):
                arguments = content.arguments
            elif isinstance(content.arguments, str):
                try:
                    arguments = json.loads(content.arguments)
                except json.JSONDecodeError:
                    arguments = {}

        return DurableAgentStateFunctionCallContent(call_id=content.call_id, name=content.name, arguments=arguments)

    def to_ai_content(self) -> FunctionCallContent:
        return FunctionCallContent(call_id=self.call_id, name=self.name, arguments=self.arguments)


class DurableAgentStateFunctionResultContent(DurableAgentStateContent):
    """Represents the result of a function/tool call execution."""

    call_id: str
    result: object | None = None
    type: str = "functionResult"

    def __init__(self, call_id: str, result: Any | None = None) -> None:
        self.call_id = call_id
        self.result = result

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "callId": self.call_id, "result": self.result}

    @staticmethod
    def from_function_result_content(content: FunctionResultContent) -> DurableAgentStateFunctionResultContent:
        return DurableAgentStateFunctionResultContent(call_id=content.call_id, result=content.result)

    def to_ai_content(self) -> FunctionResultContent:
        return FunctionResultContent(call_id=self.call_id, result=self.result)


class DurableAgentStateHostedFileContent(DurableAgentStateContent):
    """Represents a reference to a hosted file resource."""

    file_id: str
    type: str = "hostedFile"

    def __init__(self, file_id: str) -> None:
        self.file_id = file_id

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "fileId": self.file_id}

    @staticmethod
    def from_hosted_file_content(content: HostedFileContent) -> DurableAgentStateHostedFileContent:
        return DurableAgentStateHostedFileContent(file_id=content.file_id)

    def to_ai_content(self) -> HostedFileContent:
        return HostedFileContent(file_id=self.file_id)


class DurableAgentStateHostedVectorStoreContent(DurableAgentStateContent):
    """Represents a reference to a hosted vector store resource."""

    vector_store_id: str
    type: str = "hostedVectorStore"

    def __init__(self, vector_store_id: str) -> None:
        self.vector_store_id = vector_store_id

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "vectorStoreId": self.vector_store_id}

    @staticmethod
    def from_hosted_vector_store_content(
        content: HostedVectorStoreContent,
    ) -> DurableAgentStateHostedVectorStoreContent:
        return DurableAgentStateHostedVectorStoreContent(vector_store_id=content.vector_store_id)

    def to_ai_content(self) -> HostedVectorStoreContent:
        return HostedVectorStoreContent(vector_store_id=self.vector_store_id)


class DurableAgentStateTextReasoningContent(DurableAgentStateContent):
    """Represents reasoning or thought process text from the agent."""

    type: str = "reasoning"

    def __init__(self, text: str | None) -> None:
        self.text = text

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}

    @staticmethod
    def from_text_reasoning_content(content: TextReasoningContent) -> DurableAgentStateTextReasoningContent:
        return DurableAgentStateTextReasoningContent(text=content.text)

    def to_ai_content(self) -> TextReasoningContent:
        return TextReasoningContent(text=self.text or "")


class DurableAgentStateUriContent(DurableAgentStateContent):
    """Represents content referenced by a URI with media type."""

    uri: str
    media_type: str
    type: str = "uri"

    def __init__(self, uri: str, media_type: str) -> None:
        self.uri = uri
        self.media_type = media_type

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "uri": self.uri, "mediaType": self.media_type}

    @staticmethod
    def from_uri_content(content: UriContent) -> DurableAgentStateUriContent:
        return DurableAgentStateUriContent(uri=content.uri, media_type=content.media_type)

    def to_ai_content(self) -> UriContent:
        return UriContent(uri=self.uri, media_type=self.media_type)


class DurableAgentStateUsage:
    """Represents token usage statistics for agent responses."""

    input_token_count: int | None = None
    output_token_count: int | None = None
    total_token_count: int | None = None
    extensionData: dict[str, Any] | None = None

    def __init__(
        self,
        input_token_count: int | None = None,
        output_token_count: int | None = None,
        total_token_count: int | None = None,
        extensionData: dict[str, Any] | None = None,
    ) -> None:
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count
        self.total_token_count = total_token_count
        self.extensionData = extensionData

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "inputTokenCount": self.input_token_count,
            "outputTokenCount": self.output_token_count,
            "totalTokenCount": self.total_token_count,
        }
        if self.extensionData is not None:
            result["extensionData"] = self.extensionData
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableAgentStateUsage:
        return cls(
            input_token_count=data.get("inputTokenCount"),
            output_token_count=data.get("outputTokenCount"),
            total_token_count=data.get("totalTokenCount"),
            extensionData=data.get("extensionData"),
        )

    @staticmethod
    def from_usage(usage: UsageDetails | None) -> DurableAgentStateUsage | None:
        if usage is None:
            return None
        return DurableAgentStateUsage(
            input_token_count=usage.input_token_count,
            output_token_count=usage.output_token_count,
            total_token_count=usage.total_token_count,
        )

    def to_usage_details(self) -> UsageDetails:
        return UsageDetails(
            input_token_count=self.input_token_count,
            output_token_count=self.output_token_count,
            total_token_count=self.total_token_count,
        )


class DurableAgentStateUsageContent(DurableAgentStateContent):
    """Represents token usage information as message content."""

    usage: DurableAgentStateUsage = DurableAgentStateUsage()
    type: str = "usage"

    def __init__(self, usage: DurableAgentStateUsage) -> None:
        self.usage = usage

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "usage": self.usage.to_dict() if hasattr(self.usage, "to_dict") else self.usage}

    @staticmethod
    def from_usage_content(content: UsageContent) -> DurableAgentStateUsageContent:
        return DurableAgentStateUsageContent(usage=DurableAgentStateUsage.from_usage(content.details))  # type: ignore

    def to_ai_content(self) -> UsageContent:
        return UsageContent(details=self.usage.to_usage_details())


class DurableAgentStateUnknownContent(DurableAgentStateContent):
    """Represents unknown or unrecognized content types."""

    content: Any
    type: str = "unknown"

    def __init__(self, content: Any) -> None:
        self.content = content

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "content": self.content}

    @staticmethod
    def from_unknown_content(content: Any) -> DurableAgentStateUnknownContent:
        return DurableAgentStateUnknownContent(content=content)

    def to_ai_content(self) -> BaseContent:
        if not self.content:
            raise Exception("The content is missing and cannot be converted to valid AI content.")
        return BaseContent(content=self.content)
