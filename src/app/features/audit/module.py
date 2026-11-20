"""Audit slice composition descriptor and session-bound recorder binder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.module import Module

from .application.commands.record import RecordAuditEntry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from .application.dto.record import RecordAuditEntryInput, RecordAuditEntryOutput
    from .application.ports.recorder import AuditRecorder
    from .domain.errors import AuditResult


MODULE = Module(name="audit")


def build_audit_recorder(session: AsyncSession) -> AuditRecorder:
    """Bind recording to the caller-owned session without transaction control."""

    from .adapters.outbound.postgres.repository import (  # noqa: PLC0415
        PostgresAuditLogRepository,
    )

    use_case = RecordAuditEntry(PostgresAuditLogRepository(session))
    return _SessionAuditRecorder(use_case)


class _SessionAuditRecorder:
    def __init__(self, use_case: RecordAuditEntry) -> None:
        self._use_case = use_case

    async def record(self, entry: RecordAuditEntryInput) -> AuditResult[RecordAuditEntryOutput]:
        return await self._use_case.execute(entry)
