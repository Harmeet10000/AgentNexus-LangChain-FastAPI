"""Application port exposed to slices that need to append an audit record."""

from typing import Protocol

from app.features.audit.application.dto.record import (
    RecordAuditEntryInput,
    RecordAuditEntryOutput,
)
from app.features.audit.domain.errors import AuditResult


class AuditRecorder(Protocol):
    async def record(self, entry: RecordAuditEntryInput) -> AuditResult[RecordAuditEntryOutput]: ...
