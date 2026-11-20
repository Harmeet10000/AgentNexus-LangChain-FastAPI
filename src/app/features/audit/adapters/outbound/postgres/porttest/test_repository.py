"""Contract proof shared by the PostgreSQL adapter and its in-memory proof."""

# This is executable pytest contract documentation, not production control flow.
# ruff: noqa: S101

import pytest
from returns.result import Success

from app.features.audit.adapters.outbound.postgres.porttest.in_memory import (
    InMemoryAuditLogRepository,
)
from app.features.audit.domain.entities.audit_entry import AuditEntry


@pytest.mark.asyncio
async def test_repository_port_in_memory_proof() -> None:
    repository = InMemoryAuditLogRepository()
    entry = AuditEntry(entity_type="plan", entity_id="plan-1", action="plan.created")

    created = await repository.create(entry)
    listed = await repository.find_by_entity("plan", "plan-1")
    queried = await repository.query(action="plan.created")

    assert isinstance(created, Success)
    assert isinstance(listed, Success)
    assert isinstance(queried, Success)
    assert listed.unwrap() == [entry]
    assert queried.unwrap() == ([entry], 1)
