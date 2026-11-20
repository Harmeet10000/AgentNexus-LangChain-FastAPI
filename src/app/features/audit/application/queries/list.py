"""List-audit-entries application use case."""

from returns.result import Failure, Success

from app.features.audit.application.commands.record import to_output
from app.features.audit.application.dto.list import (
    ListAuditEntriesInput,
    ListAuditEntriesOutput,
)
from app.features.audit.domain.errors import AuditResult
from app.features.audit.domain.repositories import AuditLogRepository


class ListAuditEntries:
    def __init__(self, repository: AuditLogRepository) -> None:
        self._repository = repository

    async def execute(self, query: ListAuditEntriesInput) -> AuditResult[ListAuditEntriesOutput]:
        result = await self._repository.query(
            entity_type=query.entity_type,
            entity_id=query.entity_id,
            action=query.action,
            date_from=query.date_from,
            date_to=query.date_to,
            limit=query.limit,
            offset=query.offset,
        )
        if isinstance(result, Failure):
            return result
        entries, total = result.unwrap()
        return Success(
            ListAuditEntriesOutput(
                entries=tuple(to_output(entry) for entry in entries), total=total
            )
        )
