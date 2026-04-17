"""
WebSocket protocol for Agent Saul.

Inbound  (client → server): discriminated on "type"
Outbound (server → client): discriminated on "type"

Every WS frame is a Pydantic model serialised with .model_dump().
The client MUST handle all outbound frame types.

HTTP DTOs for the session-creation endpoint are included here so all
request/response contracts live in one module.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from app.shared.langgraph_layer.agent_saul.state import (
    HITLInterruptType,
    PlanStep,
    WorkflowStatus,
)

# ---------------------------------------------------------------------------
# Inbound WS messages (client → server)
# ---------------------------------------------------------------------------


class WSStartMessage(BaseModel, frozen=True):
    """First message on a new WS connection.  Kicks off the graph."""

    type: Literal["start"] = "start"
    doc_id: str = Field(description="ID returned by HTTP upload endpoint")
    user_query: str = Field(min_length=1, max_length=4096)
    thread_id: str | None = Field(
        default=None,
        description="Omit to start a new thread; supply to resume a persisted thread",
    )
    permissions: dict[str, bool] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class WSResumeMessage(BaseModel, frozen=True):
    """Sent by the client to resume a graph paused by interrupt()."""

    type: Literal["resume"] = "resume"
    thread_id: str
    action: Literal["approve", "reject", "modify"]
    feedback: str | None = Field(default=None, max_length=8192)
    modified_plan: list[PlanStep] | None = None
    # Human review fields — populated when action relates to human_review HITL
    overrides: list[dict[str, Any]] | None = None
    reviewer_role: str | None = None

    model_config = {"extra": "forbid"}


class WSPingMessage(BaseModel, frozen=True):
    type: Literal["ping"] = "ping"

    model_config = {"extra": "forbid"}


WSInbound = Annotated[
    WSStartMessage | WSResumeMessage | WSPingMessage,
    Field(discriminator="type"),
]

# Module-level TypeAdapter — built once, reused on every WS frame parse.
ws_inbound_adapter: TypeAdapter[WSInbound] = TypeAdapter(WSInbound)


# ---------------------------------------------------------------------------
# Outbound WS frames (server → client)
# ---------------------------------------------------------------------------


class WSTokenFrame(BaseModel, frozen=True):
    """Streaming LLM token from within an active node."""

    type: Literal["token"] = "token"
    node: str
    token: str


class WSNodeStartFrame(BaseModel, frozen=True):
    """Emitted when a top-level graph node begins execution."""

    type: Literal["node_start"] = "node_start"
    node: str
    step: int
    status: WorkflowStatus


class WSNodeEndFrame(BaseModel, frozen=True):
    """Emitted when a top-level graph node finishes. output_keys tells the
    client which state fields changed — avoids sending full state diffs."""

    type: Literal["node_end"] = "node_end"
    node: str
    step: int
    output_keys: list[str]


class WSHITLInterruptFrame(BaseModel, frozen=True):
    """Graph paused at interrupt(). Client must send WSResumeMessage to continue."""

    type: Literal["hitl_interrupt"] = "hitl_interrupt"
    thread_id: str
    interrupt_type: HITLInterruptType
    payload: dict[str, Any]
    message: str


class WSStateUpdateFrame(BaseModel, frozen=True):
    """Coarse-grained status transition between pipeline stages."""

    type: Literal["state_update"] = "state_update"
    status: WorkflowStatus
    step: int
    total_steps: int


class WSErrorFrame(BaseModel, frozen=True):
    type: Literal["error"] = "error"
    node: str | None
    code: str
    message: str
    retryable: bool


class WSPongFrame(BaseModel, frozen=True):
    type: Literal["pong"] = "pong"


class WSDoneFrame(BaseModel, frozen=True):
    """Terminal frame.  Client should close the WS after receiving this."""

    type: Literal["done"] = "done"
    thread_id: str
    summary: str | None = None


WSOutbound = (
    WSTokenFrame
    | WSNodeStartFrame
    | WSNodeEndFrame
    | WSHITLInterruptFrame
    | WSStateUpdateFrame
    | WSErrorFrame
    | WSPongFrame
    | WSDoneFrame
)


# ---------------------------------------------------------------------------
# HTTP DTOs  — session management
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """POST /agent-saul/sessions — pre-flight before WS connection."""

    doc_id: str
    user_query: str = Field(min_length=1, max_length=4096)
    permissions: dict[str, bool] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class CreateSessionResponse(BaseModel, frozen=True):
    thread_id: str
    ws_url: str = Field(description="Fully qualified WS endpoint the client should connect to")
