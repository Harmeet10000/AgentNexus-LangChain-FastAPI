"""Audit-slice adapter for content-free document access events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure

from app.features.audit.module import build_audit_recorder
from app.features.audit.public import RecordAuditEntryInput

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.features.documents.application.access_audit import DocumentAccessAuditEvent


class PostgresDocumentAccessAuditRecorder:
    """Persist read events through audit's public recorder and one transaction."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def append(self, event: DocumentAccessAuditEvent) -> None:
        entity_ids = (*event.document_ids, *event.evidence_ids)
        entity_id = entity_ids[0] if entity_ids else "scope"
        entity_type = "evidence" if event.operation.startswith("evidence.") else "document"
        async with self._session_factory() as session, session.begin():
            result = await build_audit_recorder(session).record(
                RecordAuditEntryInput(
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
