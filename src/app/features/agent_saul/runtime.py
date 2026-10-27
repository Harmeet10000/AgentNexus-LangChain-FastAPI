"""Scope-agnostic Saul runtime wiring.

One module owns the session/claims/thread-context construction that the HTTP
dependency chain and the WebSocket handshake used to re-implement in parallel
(commits 7aeba9f / 7f04409 each patched one side of the same semantics). Every
reader here consumes plain values or the ``app.state`` object store — never a
``Request`` or a ``WebSocket`` — and no Starlette types appear in this module:
scope plumbing is provided by the thin FastAPI adapters in ``dependencies.py``
on top of these functions.

Lifespan callers must set (read through ``build_agent_saul_runtime``):
    app.state.saul_graph             → CompiledStateGraph (factory.build_saul_graph)
    app.state.langgraph_checkpointer → AsyncPostgresSaver
    app.state.redis                  → redis.asyncio.Redis
    app.state.db_session_local       → async_sessionmaker[AsyncSession]
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from returns.result import Failure, Success
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.features.auth.tenancy import LegalRole
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from returns.result import Result

    from app.features.auth.repository import SessionData

# ---------------------------------------------------------------------------
# App-state component reads — one implementation, both scopes
# ---------------------------------------------------------------------------

# Exact wire messages of the 503s these readers raise (observed behavior).
_GRAPH_MISSING_MSG = "Saul graph is not wired"
_CHECKPOINTER_MISSING_MSG = "Persistence layer is unavailable"
_REDIS_MISSING_MSG = "Cache layer is unavailable"
_SESSION_LOCAL_MISSING_MSG = "Database layer is unavailable"

# Components degrade to an explicit ``None`` on app.state (lifecycle policy);
# getattr with a None fall-back keeps "never provisioned" on that same path
# instead of Starlette's raising State turning the gap into a 500.
_MISSING = None


def _require_component(state: object, attr: str, message: str) -> Any:
    component = getattr(state, attr, _MISSING)
    if component is None:
        raise ServiceUnavailableException(message)
    return component


def require_saul_graph(state: object) -> CompiledStateGraph[Any]:
    return _require_component(state, "saul_graph", _GRAPH_MISSING_MSG)


def require_saul_checkpointer(state: object) -> AsyncPostgresSaver:
    return _require_component(state, "langgraph_checkpointer", _CHECKPOINTER_MISSING_MSG)


def require_saul_redis(state: object) -> Redis:
    return _require_component(state, "redis", _REDIS_MISSING_MSG)


def require_db_session_local(state: object) -> async_sessionmaker[AsyncSession]:
    return _require_component(state, "db_session_local", _SESSION_LOCAL_MISSING_MSG)


# ---------------------------------------------------------------------------
# Runtime bundle — constructed ONCE, consumed by HTTP and WS adapters
# ---------------------------------------------------------------------------


class AgentSaulDeps(BaseModel):
    """Narrow context object for Agent Saul runtime wiring.

    Typed against infra implementations so nodes remain decoupled from
    concrete client stand-ins in tests. Frozen: wiring never mutates at scope.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    graph: CompiledStateGraph[Any]
    checkpointer: AsyncPostgresSaver
    redis: Redis
    session_factory: async_sessionmaker[AsyncSession]


def build_agent_saul_runtime(state: object) -> AgentSaulDeps:
    """Build the bundle both scope adapters hand to orchestration."""
    return AgentSaulDeps(
        graph=require_saul_graph(state),
        checkpointer=require_saul_checkpointer(state),
        redis=require_saul_redis(state),
        session_factory=require_db_session_local(state),
    )


# ---------------------------------------------------------------------------
# Session reads — fail-closed WS handshake core
# ---------------------------------------------------------------------------

# Observed wire reason of the WS_1008 close on every rejected handshake session.
WS_SESSION_REJECTED_REASON = "Session expired or revoked"


class WsSessionRejection(StrEnum):
    """Fail-closed handshake outcomes; each maps to ``WS_SESSION_REJECTED_REASON``."""

    # Redis read failed — an outage must never authenticate the handshake (7f04409).
    UNAVAILABLE = "session_check_unavailable"
    # Session record is gone: revoked server-side or expired out of Redis.
    REVOKED = "session_revoked"
    # Session belongs to another user — refuse before anything is revealed.
    OWNER_MISMATCH = "session_owner_mismatch"


class SessionStore(Protocol):
    """The session-read surface of auth's ``RefreshTokenRepository``."""

    async def get_session(self, session_id: str) -> Result[SessionData | None, Any]: ...


async def resolve_ws_session(
    session_store: SessionStore,
    calling_user_id: str,
    session_id: str,
) -> Result[SessionData, WsSessionRejection]:
    """Resolve one authenticated user session, fail-closed.

    The single implementation of the handshake session semantics: a Redis
    read failure rejects the connection (never fail-open on outage), an
    absent record is a revocation, and a record describing another user is
    refuse before revealing anything about the thread.
    """
    check = await session_store.get_session(session_id)
    if isinstance(check, Failure):
        return Failure(WsSessionRejection.UNAVAILABLE)
    session = check.unwrap()
    if session is None:
        return Failure(WsSessionRejection.REVOKED)
    if session.user_id != calling_user_id:
        return Failure(WsSessionRejection.OWNER_MISMATCH)
    return Success(session)


# ---------------------------------------------------------------------------
# Thread context — the §9 identity triple, constructed once per session
# ---------------------------------------------------------------------------


class SaulThreadContext(BaseModel):
    """Identity a Saul session runs under (§9 owner gate + correlation)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    thread_id: str
    tenant_id: str = ""
    legal_roles: tuple[LegalRole, ...] = (LegalRole.ANALYST,)
    self_review_allowed: bool = True
    role_revision: int = 0
    # Capability grants are resolved by the authenticated dependency and
    # copied into graph state once; the client never supplies or widens them.
    allowed_doc_ids: tuple[str, ...] = ()
    allowed_operations: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))


def thread_owner_violates(persisted_owner: object, caller_user_id: str) -> bool:
    """True when a persisted thread owner is not the caller (§9 THREAD_NOT_OWNED).

    Threads with no persisted owner yet (clean checkpoint lineage) gate on
    acquisition alone, so ``None`` never violates.
    """
    return persisted_owner is not None and str(persisted_owner) != caller_user_id
