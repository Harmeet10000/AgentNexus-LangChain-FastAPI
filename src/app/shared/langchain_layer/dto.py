"""
Strongly-typed request/response schemas for agent endpoints.
Use these in FastAPI route handlers for validated, documented APIs.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

OutputT = TypeVar("OutputT")


# ---------------------------------------------------------------------------
# Agent input
# ---------------------------------------------------------------------------


class MessageRole(StrEnum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class AgentInvokeRequest(BaseModel):
    """Single-turn agent invocation request."""

    message: str = Field(..., description="The user's message.")
    thread_id: str = Field(
        ..., description="Session/thread identifier for conversation history."
    )
    user_id: str = Field("default", description="User identifier for long-term memory.")

    # Optional runtime context (serialised; factory will reconstruct the dataclass)
    context: dict[str, Any] | None = Field(
        None,
        description="Runtime context values matching the agent's context_schema.",
    )

    # Streaming preference (used by endpoint, not agent)
    stream: bool = Field(False, description="Whether to stream the response.")


class AgentBatchRequest(BaseModel):
    """Batch invocation request."""

    messages: list[str]
    thread_ids: list[str]
    user_id: str = "default"
    context: dict[str, Any] | None = None
    max_concurrency: int = 5


class MultimodalAgentRequest(AgentInvokeRequest):
    """Invoke with image + text."""

    image_url: str | None = None
    image_b64: str | None = None


# ---------------------------------------------------------------------------
# Agent output
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Standard single-turn response."""

    content: str = Field(..., description="The agent's text response.")
    thread_id: str
    structured_output: dict[str, Any] | None = None
    blocked: bool = False
    block_reason: str | None = None
    tool_calls_made: int = 0
    model_name: str | None = None


class AgentBatchResponse(BaseModel):
    responses: list[AgentResponse]
    total: int
    failed: int = 0


class StreamChunk(BaseModel):
    """Single SSE chunk."""

    type: str  # "token" | "tool_call" | "done" | "error"
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    tool_name: str
    input: dict[str, Any]
    output: Any
    success: bool
    duration_ms: float


# ---------------------------------------------------------------------------
# Embedding schemas
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int


# ---------------------------------------------------------------------------
# Memory schemas
# ---------------------------------------------------------------------------


class MemorySearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 10


class MemoryEntry(BaseModel):
    memory: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResponse(BaseModel):
    results: list[MemoryEntry]
    user_id: str
