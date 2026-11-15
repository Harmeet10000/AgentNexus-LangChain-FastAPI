"""G3 regressions: plan lookup by a malformed id must be typed 404, never 500.

Round-2 smoke: ``GET /api/v2/plans/{non-uuid}`` executed the UUID-cast anyway,
asyncpg raised, and the repository serialized the raw SQLAlchemy statement plus
the asyncpg error text into ``error.data`` (500 ``PLAN_DATABASE_ERROR``). A
malformed id cannot address any row, so it is decided as not-found before a
statement runs, and driver error text never enters HTTP-facing details.
"""

from __future__ import annotations

import pytest
from fastapi import Response
from returns.result import Failure
from sqlalchemy.exc import SQLAlchemyError

from app.features.billing.plans.exceptions import (
    PlanCode,
    PlanInfrastructureError,
    PlanNotFoundError,
)
from app.features.billing.plans.repository import PlanRepository
from app.kernel.result import render_result

pytestmark = pytest.mark.unit


class _LeakingDriverError(SQLAlchemyError):
    """Stand-in for a SQLAlchemy error whose text carries the statement (G3 leak)."""


class _ExecuteCountingSession:
    """AsyncSession stand-in that records executes and can raise a driver error."""

    def __init__(self, raise_on_execute: Exception | None = None) -> None:
        self.executes = 0
        self._raise_on_execute = raise_on_execute

    async def execute(self, _statement: object) -> object:
        self.executes += 1
        if self._raise_on_execute is not None:
            raise self._raise_on_execute
        message = "unexpected execute without configured result"
        raise AssertionError(message)

    async def rollback(self) -> None:
        return None


async def test_find_by_id_with_non_uuid_returns_typed_not_found() -> None:
    session = _ExecuteCountingSession()
    repo = PlanRepository(session)  # type: ignore[arg-type]

    result = await repo.find_by_id("this-is-not-a-uuid")

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, PlanNotFoundError)
    assert error.code == PlanCode.NOT_FOUND
    # The malformed id never reaches the database.
    assert session.executes == 0


async def test_archive_with_non_uuid_returns_typed_not_found() -> None:
    session = _ExecuteCountingSession()
    repo = PlanRepository(session)  # type: ignore[arg-type]

    result = await repo.archive("not-even-close")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), PlanNotFoundError)
    assert session.executes == 0


async def test_database_failure_details_never_leak_sql() -> None:
    """A real DB error stays 500-class but the driver/SQL text stays server-side."""
    leak = _LeakingDriverError(
        "SELECT plans.id, plans.parent_plan_id, plans.razorpay_plan_id, plans.name FROM plans "
        "WHERE plans.id = $1::UUID — invalid input for query argument $1: "
        "'this-is-not-a-uuid' (asyncpg.exceptions.DataError)"
    )
    session = _ExecuteCountingSession(raise_on_execute=leak)
    repo = PlanRepository(session)  # type: ignore[arg-type]

    result = await repo.find_by_id("3fa85f64-5717-4562-b3fc-2c963f66afa6")

    assert isinstance(result, Failure)
    error = result.failure()
    assert isinstance(error, PlanInfrastructureError)
    assert "SELECT" not in str(error.details)
    assert "asyncpg" not in str(error.details)

    response = Response()
    rendered = render_result(result, response, message="Plan")
    assert rendered.error is not None
    body_probe = repr(rendered.error.data) + rendered.error.message
    assert "SELECT" not in body_probe
    assert "asyncpg" not in body_probe
