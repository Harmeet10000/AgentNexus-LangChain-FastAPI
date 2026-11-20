"""Record-audit-entry application use case."""

from returns.result import Failure, Success

from app.features.audit.application.dto.record import (
    RecordAuditEntryInput,
    RecordAuditEntryOutput,
)
from app.features.audit.domain.entities.audit_entry import AuditEntry
from app.features.audit.domain.errors import AuditResult
from app.features.audit.domain.repositories import AuditLogRepository


class RecordAuditEntry:
    """Translate the public command into a domain entry and persist it."""

    def __init__(self, repository: AuditLogRepository) -> None:
        self._repository = repository

    async def execute(self, command: RecordAuditEntryInput) -> AuditResult[RecordAuditEntryOutput]:
        result = await self._repository.create(
            AuditEntry(
                entity_type=command.entity_type,
                entity_id=command.entity_id,
                action=command.action,
                user_id=command.user_id,
                ip_address=command.ip_address,
                user_agent=command.user_agent,
                changes=dict(command.changes),
                metadata_=dict(command.metadata_),
            )
        )
        if isinstance(result, Failure):
            return result
        return Success(to_output(result.unwrap()))


def to_output(entry: AuditEntry) -> RecordAuditEntryOutput:
    return RecordAuditEntryOutput(
        id=entry.id,
        entity_type=entry.entity_type,
        entity_id=entry.entity_id,
        action=entry.action,
        user_id=entry.user_id,
        ip_address=entry.ip_address,
        user_agent=entry.user_agent,
        changes=dict(entry.changes),
        metadata_=dict(entry.metadata_),
        created_at=entry.created_at,
    )
