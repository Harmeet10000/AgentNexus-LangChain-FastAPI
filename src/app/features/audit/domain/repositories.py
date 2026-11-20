"""Persistence ports for the audit aggregate."""

from datetime import datetime
from typing import Protocol

from app.features.audit.domain.entities.audit_entry import AuditEntry
from app.features.audit.domain.errors import AuditResult


class AuditLogRepository(Protocol):
    """Append-only persistence contract for audit entries."""

    async def create(self, entry: AuditEntry) -> AuditResult[AuditEntry]: ...

    async def find_by_entity(
        self,
        entity_type: str,
        entity_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditResult[list[AuditEntry]]: ...

    async def query(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        action: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditResult[tuple[list[AuditEntry], int]]: ...
