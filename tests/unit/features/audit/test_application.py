"""Audit application and facade tests."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError
from returns.result import Failure, Success

from app.features.audit.application.commands.record import RecordAuditEntry
from app.features.audit.application.dto.list import (
    ListAuditEntriesInput,
    ListAuditEntriesOutput,
)
from app.features.audit.application.dto.record import (
    RecordAuditEntryInput,
    RecordAuditEntryOutput,
)
from app.features.audit.application.queries.list import ListAuditEntries
from app.features.audit.domain.entities.audit_entry import AuditEntry
from app.features.audit.domain.errors import AuditInfrastructureError
from app.features.audit.public import AuditAction


class _Repository:
    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []
        self.failure: AuditInfrastructureError | None = None

    async def create(self, entry: AuditEntry):
        if self.failure is not None:
            return Failure(self.failure)
        self.entries.append(entry)
        return Success(entry)

    async def find_by_entity(self, entity_type: str, entity_id: str, **kwargs: object):
        del kwargs
        return Success(
            [
                entry
                for entry in self.entries
                if entry.entity_type == entity_type and entry.entity_id == entity_id
            ]
        )

    async def query(self, **kwargs: object):
        entries = self.entries
        if kwargs.get("entity_type") is not None:
            entries = [entry for entry in entries if entry.entity_type == kwargs["entity_type"]]
        return Success((entries, len(entries)))


@pytest.mark.asyncio
async def test_record_use_case_maps_immutable_command_and_result() -> None:
    repository = _Repository()
    command = RecordAuditEntryInput(
        entity_type="plan",
        entity_id="plan-1",
        action=AuditAction.PLAN_CREATED,
        changes={"amount": 100},
    )

    result = await RecordAuditEntry(repository).execute(command)  # type: ignore[arg-type]

    assert isinstance(result, Success)
    output = result.unwrap()
    assert isinstance(output, RecordAuditEntryOutput)
    assert output.action == AuditAction.PLAN_CREATED.value
    assert repository.entries[0].changes == {"amount": 100}
    with pytest.raises(ValidationError):
        command.entity_id = "other"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_record_use_case_preserves_typed_repository_failure() -> None:
    repository = _Repository()
    repository.failure = AuditInfrastructureError(
        message="database unavailable", source="test", operation="create"
    )

    result = await RecordAuditEntry(repository).execute(
        RecordAuditEntryInput(entity_type="plan", entity_id="plan-1", action="plan.created")
    )  # type: ignore[arg-type]

    assert isinstance(result, Failure)
    assert result.failure().operation == "create"


@pytest.mark.asyncio
async def test_list_use_case_returns_immutable_output_tuple() -> None:
    repository = _Repository()
    repository.entries.append(
        AuditEntry(
            entity_type="plan",
            entity_id="plan-1",
            action="plan.created",
            id=uuid4(),
            created_at=datetime.now(tz=UTC),
        )
    )

    result = await ListAuditEntries(repository).execute(ListAuditEntriesInput(entity_type="plan"))  # type: ignore[arg-type]

    assert isinstance(result, Success)
    output = result.unwrap()
    assert isinstance(output, ListAuditEntriesOutput)
    assert isinstance(output.entries, tuple)
    assert output.total == 1
