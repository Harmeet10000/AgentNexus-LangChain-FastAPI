"""Content-free, append-only audit seam for document and evidence reads."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure

from app.features.audit.models import AuditLog
from app.features.audit.repository import AuditLogRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class DocumentAccessAuditEvent(BaseModel):
    """The minimum accountability record; raw source text is unrepresentable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    principal_id: str
    thread_id: str = ""
    run_id: str = ""
    checkpoint_id: str = ""
    operation: str
    requested_document_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    policy: str
    outcome: str
    reason: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class DocumentAccessAuditRecorder(Protocol):
    async def append(self, event: DocumentAccessAuditEvent) -> None: ...


class InMemoryDocumentAccessAuditRecorder:
    """Deterministic recorder for tests and local development."""

    def __init__(self) -> None:
        self.events: list[DocumentAccessAuditEvent] = []

    async def append(self, event: DocumentAccessAuditEvent) -> None:
        self.events.append(event)


class SqlDocumentAccessAuditRecorder:
    """Persist read events through the existing append-only audit table.

    Each event uses a fresh transaction.  A database failure is propagated to
    the evidence service, which then returns a fail-closed response instead of
    leaking privileged content without an audit trail.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def append(self, event: DocumentAccessAuditEvent) -> None:
        entity_ids = (*event.document_ids, *event.evidence_ids)
        entity_id = entity_ids[0] if entity_ids else "scope"
        entity_type = "evidence" if event.operation.startswith("evidence.") else "document"
        async with self._session_factory() as session, session.begin():
            result = await AuditLogRepository(session).create(
                AuditLog(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    action=event.operation,
                    user_id=event.principal_id,
                    changes={
                        "outcome": event.outcome,
                        "reason": event.reason,
                        "requested_document_ids": list(event.requested_document_ids),
                        "document_ids": list(event.document_ids),
                        "evidence_ids": list(event.evidence_ids),
                    },
                    metadata_={
                        "tenant_id": event.tenant_id,
                        "thread_id": event.thread_id,
                        "run_id": event.run_id,
                        "checkpoint_id": event.checkpoint_id,
                        "policy": event.policy,
                        "timestamp": event.timestamp.isoformat(),
                    },
                )
            )
            if isinstance(result, Failure):
                message = "document access audit append failed"
                raise TypeError(message)


def audit_scope(
    requested_document_ids: Sequence[str],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in requested_document_ids if str(value)))
