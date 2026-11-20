"""Small in-memory repository used to prove the repository port contract."""

from returns.result import Success

from app.features.audit.domain.entities.audit_entry import AuditEntry
from app.features.audit.domain.errors import AuditResult


class InMemoryAuditLogRepository:
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    async def create(self, entry: AuditEntry) -> AuditResult[AuditEntry]:
        self.entries.append(entry)
        return Success(entry)

    async def find_by_entity(
        self, entity_type: str, entity_id: str, *, limit: int = 100, offset: int = 0
    ) -> AuditResult[list[AuditEntry]]:
        matches = [
            entry
            for entry in self.entries
            if entry.entity_type == entity_type and entry.entity_id == entity_id
        ]
        return Success(matches[offset : offset + limit])

    async def query(self, **kwargs: object) -> AuditResult[tuple[list[AuditEntry], int]]:
        matches = self.entries
        entity_type = kwargs.get("entity_type")
        entity_id = kwargs.get("entity_id")
        action = kwargs.get("action")
        if entity_type is not None:
            matches = [entry for entry in matches if entry.entity_type == entity_type]
        if entity_id is not None:
            matches = [entry for entry in matches if entry.entity_id == entity_id]
        if action is not None:
            matches = [entry for entry in matches if entry.action == action]
        offset_value = kwargs.get("offset", 0)
        limit_value = kwargs.get("limit", 100)
        offset = offset_value if isinstance(offset_value, int) else 0
        limit = limit_value if isinstance(limit_value, int) else 100
        return Success((matches[offset : offset + limit], len(matches)))
