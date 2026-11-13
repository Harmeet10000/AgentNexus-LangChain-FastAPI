"""Task 2.2 — client authority fields are dead at the DTO edge.

`WSStartMessage.permissions` is never copied into graph state and
`WSResumeMessage.reviewer_role` is never trusted: reviewer authority is
re-derived from LIVE tenant membership at the moment the resume is applied.
These tests pin the two fail-closed paths a paused thread can hit after its
authorization changes, plus the server-derivation of the reviewer identity
the interrupted node finally observes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

from langgraph.types import StateSnapshot
from returns.result import Success

from app.features.agent_saul.adapters.document_source import (
    DOCUMENT_SOURCE_CONFIG_KEY,
    DocumentReadiness,
)
from app.features.agent_saul.api.dto import WSResumeMessage
from app.features.agent_saul.application.runtime import SaulThreadContext
from app.features.agent_saul.application.service import AgentSaulService
from app.features.agent_saul.domain.state import STATE_SCHEMA_VERSION
from app.features.auth.domain.tenancy import LegalRole
from app.observability.logging import logger

if TYPE_CHECKING:
    from typing import Any


def _context(*, role_revision: int = 3) -> SaulThreadContext:
    return SaulThreadContext(
        user_id="user-1",
        thread_id="thread-1",
        tenant_id="tenant-1",
        legal_roles=(LegalRole.REVIEWER,),
        role_revision=role_revision,
        allowed_doc_ids=("doc-1",),
        allowed_operations=("evidence.read",),
    )


class _MembershipService:
    """Live authority: what the caller's membership looks like RIGHT NOW.

    ``self_review_allowed`` is part of the LIVE membership fact: the tenant's
    independent-reviewer policy is re-derived on every resume (3.6), so a
    tenant that requires an independent reviewer refuses the owner's own
    approval even when the owner genuinely holds the reviewer role.
    """

    def __init__(
        self,
        roles: tuple[LegalRole, ...],
        revision: int,
        *,
        self_review_allowed: bool = True,
    ) -> None:
        self._roles = roles
        self._revision = revision
        self._self_review_allowed = self_review_allowed

    async def get_live_authorization(self, *, tenant_id: str, user_id: str) -> Any:
        return Success((self._roles, self._self_review_allowed, self._revision))


class _DocumentSource:
    async def ensure_document(self, **_kwargs: Any) -> Any:
        return Success(None)

    async def check_readiness(self, **_kwargs: Any) -> Any:
        return Success(DocumentReadiness(status="completed"))

    async def fetch_text(self, **_kwargs: Any) -> Any:
        message = "paused resume must not read document text"
        raise AssertionError(message)

    async def fetch_segments(self, **_kwargs: Any) -> Any:
        message = "paused resume must not read segments"
        raise AssertionError(message)


class _Recorder:
    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []

    async def record_review_decision(self, **kwargs: Any) -> Any:
        self.decisions.append(kwargs)
        return Success(None)

    async def prepare_resume(self, *, plan_resume: Any = None) -> Any:
        return Success(None)

    async def reconcile(self) -> Any:
        return Success(None)


class _CaptureSecurity:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, _ws: Any, payload: object, _context: Any) -> None:
        self.sent.append(payload)


def _snapshot(
    *,
    interrupt_type: str = "human_review_required",
    created_at: datetime | None = None,
) -> StateSnapshot:
    return StateSnapshot(
        values={
            "schema_version": STATE_SCHEMA_VERSION,
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "doc_id": "doc-1",
            "allowed_doc_ids": ["doc-1"],
        },
        next=("human_review",),
        config={"configurable": {"thread_id": "thread-1", "checkpoint_id": "cp-1"}},
        metadata={},
        created_at=(created_at or datetime.now(tz=UTC)).isoformat(),
        parent_config=None,
        tasks=(SimpleNamespace(interrupts=({"value": {"type": interrupt_type}},)),),
        interrupts=(),
    )


class _StaticGraph:
    def __init__(self, snapshot: StateSnapshot) -> None:
        self.snapshot = snapshot

    async def aget_state(self, _config: Any) -> StateSnapshot:
        return self.snapshot


def _service(
    *,
    context: SaulThreadContext | None = None,
    membership: Any = None,
) -> AgentSaulService:
    service = AgentSaulService.__new__(AgentSaulService)
    service._context = context if context is not None else _context()
    service._tenant_membership_service = membership
    security = _CaptureSecurity()
    service._ws_security = security
    service._ws_context = SimpleNamespace()
    service._correlation_id = "correlation-test"
    service._log = logger.bind(correlation_id="correlation-test")
    return service


def _config() -> dict[str, Any]:
    return {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: _DocumentSource()}}


def _resume(**overrides: Any) -> WSResumeMessage:
    payload: dict[str, Any] = {
        "thread_id": "thread-1",
        "action": "approve",
        # The client claims admin authority in the deprecated field — the
        # server must derive the role from live membership instead.
        "reviewer_role": "admin",
    }
    payload.update(overrides)
    return WSResumeMessage.model_validate(payload)


async def _apply(service: AgentSaulService, **kwargs: Any) -> Any:
    return await service._apply_resume(
        _resume(**kwargs),
        ws=SimpleNamespace(),
        thread_id="thread-1",
        recorder=_Recorder(),
        config=_config(),
    )


async def test_reviewer_revoked_during_pause_fails_closed() -> None:
    """The reviewer membership was removed while the thread was paused: the
    resume is rejected and the client's reviewer_role claim cannot restore it."""
    service = _service(
        membership=_MembershipService((LegalRole.ANALYST,), revision=3),
    )
    service._graph = _StaticGraph(_snapshot())

    command = await _apply(service)

    assert command is None
    sent = service._ws_security.sent
    assert sent, "a refusal frame must reach the wire"
    assert sent[-1]["code"] == "REVIEWER_ROLE_REQUIRED"
    assert sent[-1]["retryable"] is False


async def test_role_revision_change_during_pause_forces_reauth() -> None:
    """Any membership change during the pause invalidates the session's
    authorization snapshot: the resume is refused with AUTHORIZATION_CHANGED."""
    service = _service(
        # Even though the live roles still include reviewer, the revision
        # moved while the thread was paused.
        context=_context(role_revision=3),
        membership=_MembershipService((LegalRole.REVIEWER,), revision=4),
    )
    service._graph = _StaticGraph(_snapshot())

    command = await _apply(service)

    assert command is None
    sent = service._ws_security.sent
    assert sent, "a refusal frame must reach the wire"
    assert sent[-1]["code"] == "AUTHORIZATION_CHANGED"


async def test_reviewer_identity_is_server_derived_never_client_supplied() -> None:
    """A live reviewer resuming is stamped with the server-derived role and
    identity; the deprecated client fields are ignored entirely."""
    recorder = _Recorder()
    service = _service(
        context=_context(role_revision=3),
        membership=_MembershipService((LegalRole.REVIEWER,), revision=3),
    )
    service._graph = _StaticGraph(_snapshot())

    command = await service._apply_resume(
        _resume(),  # client claims reviewer_role="admin"
        ws=SimpleNamespace(),
        thread_id="thread-1",
        recorder=recorder,
        config=_config(),
    )

    assert command is not None
    resume_payload = command.resume
    assert resume_payload["reviewer_role"] == LegalRole.REVIEWER.value
    assert resume_payload["reviewer_id"] == "user-1"
    # The audit fact carries the same server-derived authority. The thread
    # owner and the approving reviewer are the same user here and the tenant
    # allows self-review, so the flag is server-derived — not client-asserted.
    assert recorder.decisions, "an approval must be durably recorded"
    assert recorder.decisions[0]["reviewer_role"] == LegalRole.REVIEWER.value
    assert recorder.decisions[0]["self_reviewed"] is True


async def test_nonreviewer_cannot_borrow_approval_from_the_client_field() -> None:
    """An analyst (never a reviewer) resuming with reviewer_role='reviewer'
    in the payload is still refused — the field is dead weight."""
    service = _service(
        membership=_MembershipService((LegalRole.ANALYST,), revision=3),
    )
    service._graph = _StaticGraph(_snapshot())

    command = await _apply(service, reviewer_role="reviewer")

    assert command is None
    sent = service._ws_security.sent
    assert sent, "a refusal frame must reach the wire"
    assert sent[-1]["code"] == "REVIEWER_ROLE_REQUIRED"


async def test_corpus_admin_cannot_approve_solely_by_being_admin() -> None:
    """A live tenant corpus admin (never a reviewer) resuming an approval is
    refused: admin authority manages documents, not report approval (3.6)."""
    service = _service(
        membership=_MembershipService((LegalRole.CORPUS_ADMIN,), revision=3),
    )
    service._graph = _StaticGraph(_snapshot())

    command = await _apply(service)

    assert command is None
    sent = service._ws_security.sent
    assert sent, "a refusal frame must reach the wire"
    assert sent[-1]["code"] == "REVIEWER_ROLE_REQUIRED"


async def test_tenant_without_self_review_refuses_owner_self_approval() -> None:
    """The thread owner approving their own run is refused when the tenant
    requires an independent reviewer — even though the owner IS a reviewer."""
    # The LIVE tenant policy re-derived at resume is the authority: self-review
    # is disabled for this tenant RIGHT NOW, regardless of the session snapshot.
    membership = _MembershipService((LegalRole.REVIEWER,), revision=3, self_review_allowed=False)
    service = _service(membership=membership)
    service._graph = _StaticGraph(_snapshot())

    command = await _apply(service)

    assert command is None
    sent = service._ws_security.sent
    assert sent, "a refusal frame must reach the wire"
    assert sent[-1]["code"] == "SELF_REVIEW_DISABLED"


async def test_resume_past_24h_pause_window_is_expired_not_resumable() -> None:
    """A paused turn past the 24 h retention window is closed, not resumed —
    even a fully authorized live reviewer cannot revive it (3.13)."""
    from datetime import timedelta

    from app.features.agent_saul.application.transcript.projection import PAUSE_WINDOW

    membership = _MembershipService((LegalRole.REVIEWER,), revision=3)
    service = _service(membership=membership)
    stale_created = datetime.now(tz=UTC) - PAUSE_WINDOW - timedelta(minutes=5)
    service._graph = _StaticGraph(_snapshot(created_at=stale_created))

    command = await _apply(service)

    assert command is None
    sent = service._ws_security.sent
    assert sent, "an expiry refusal frame must reach the wire"
    assert sent[-1]["code"] == "SESSION_PAUSE_EXPIRED"
    assert sent[-1]["retryable"] is False
