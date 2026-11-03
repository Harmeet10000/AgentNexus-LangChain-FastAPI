"""
Strongly-typed request/response schemas for agent endpoints.
Use these in FastAPI route handlers for validated, documented APIs.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any  # noqa: TC003 — runtime build: pydantic resolves field annotations

from pydantic import Field

from app.kernel.models import RequestModel, StrictModel

# ---------------------------------------------------------------------------
# Agent input
# ---------------------------------------------------------------------------


class MessageRole(StrEnum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(RequestModel):
    role: MessageRole
    content: str


class AgentInvokeRequest(RequestModel):
    """Single-turn agent invocation request."""

    message: str = Field(description="The user's message.")
    thread_id: str = Field(description="Session/thread identifier for conversation history.")
    user_id: str = Field("default", description="User identifier for long-term memory.")

    # Optional runtime context (serialised; factory will reconstruct the dataclass)
    context: dict[str, Any] | None = Field(
        None,
        description="Runtime context values matching the agent's context_schema.",
    )

    # Streaming preference (used by endpoint, not agent)
    stream: bool = Field(False, description="Whether to stream the response.")


class AgentBatchRequest(RequestModel):
    """Batch invocation request."""

    messages: list[str]
    thread_ids: list[str]
    user_id: str = "default"
    context: dict[str, Any] | None = None
    max_concurrency: int = Field(default=5, ge=1)


class MultimodalAgentRequest(AgentInvokeRequest):
    """Invoke with image + text."""

    image_url: str | None = None
    image_b64: str | None = None


# ---------------------------------------------------------------------------
# Agent output
# ---------------------------------------------------------------------------


class AgentResponse(StrictModel):
    """Standard single-turn response."""

    content: str = Field(description="The agent's text response.")
    thread_id: str
    structured_output: dict[str, Any] | None = None
    blocked: bool = False
    block_reason: str | None = None
    tool_calls_made: int = 0
    model_name: str | None = None


class AgentBatchResponse(StrictModel):
    responses: list[AgentResponse]
    total: int
    failed: int = 0


class StreamChunk(StrictModel):
    """Single SSE chunk."""

    type: str  # "token" | "tool_call" | "done" | "error"
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


class ToolCallRecord(StrictModel):
    tool_name: str
    input: dict[str, Any]
    output: Any
    success: bool
    duration_ms: float


# ---------------------------------------------------------------------------
# Embedding schemas
# ---------------------------------------------------------------------------


class EmbedRequest(RequestModel):
    texts: list[str] = Field(min_length=1)


class EmbedResponse(StrictModel):
    embeddings: list[list[float]]
    model: str
    dimensions: int


# ---------------------------------------------------------------------------
# Memory schemas
# ---------------------------------------------------------------------------


class MemorySearchRequest(RequestModel):
    query: str
    user_id: str
    limit: int = Field(default=10, ge=1)


class MemoryEntry(StrictModel):
    memory: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResponse(StrictModel):
    results: list[MemoryEntry]
    user_id: str
