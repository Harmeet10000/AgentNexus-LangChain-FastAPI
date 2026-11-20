"""Public-surface and shared-transaction characterization tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from returns.result import Failure, Success
from sqlalchemy.exc import IntegrityError

from app.features.audit.module import MODULE, build_audit_recorder
from app.features.audit.public import (
    AuditAction,
    RecordAuditEntryInput,
    RecordAuditEntryOutput,
)


def test_public_surface_contains_only_facade_symbols() -> None:
    assert MODULE.name == "audit"
    assert MODULE.routes == ()
    assert set(__import__("app.features.audit.public", fromlist=["__all__"]).__all__) == {
        "AuditAction",
        "AuditRecorder",
        "RecordAuditEntryInput",
        "RecordAuditEntryOutput",
    }
    assert AuditAction.PLAN_CREATED.value == "plan.created"
    assert RecordAuditEntryInput.model_config["frozen"] is True
    assert RecordAuditEntryOutput.model_config["frozen"] is True


@pytest.mark.asyncio
async def test_session_bound_recorder_never_commits_or_rolls_back_on_success() -> None:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.rollback = AsyncMock()
    session.commit = AsyncMock()

    recorder = build_audit_recorder(session)
    assert hasattr(recorder, "record")
    result = await recorder.record(
        RecordAuditEntryInput(entity_type="plan", entity_id="plan-1", action="plan.created")
    )

    assert isinstance(result, Success)
    session.flush.assert_awaited_once()
    session.rollback.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_audit_flush_failure_rolls_back_primary_write() -> None:
    writes: list[object] = []
    session = AsyncMock()
    session.add = MagicMock(side_effect=writes.append)

    async def rollback() -> None:
        writes.clear()

    session.rollback = AsyncMock(side_effect=rollback)
    flush_count = 0

    async def flush() -> None:
        nonlocal flush_count
        flush_count += 1
        if flush_count == 2:
            statement = "insert"
            raise IntegrityError(statement, {}, Exception("duplicate"))

    session.flush = AsyncMock(side_effect=flush)
    primary_write = object()
    session.add(primary_write)
    await session.flush()

    result = await build_audit_recorder(session).record(
        RecordAuditEntryInput(entity_type="plan", entity_id="plan-1", action="plan.created")
    )

    assert isinstance(result, Failure)
    assert writes == []
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()
