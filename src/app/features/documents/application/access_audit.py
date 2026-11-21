"""Content-free, append-only audit seam for document and evidence reads."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence


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


def audit_scope(
    requested_document_ids: Sequence[str],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in requested_document_ids if str(value)))
