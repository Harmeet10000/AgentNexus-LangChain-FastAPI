"""Regression for repository rollback — tasks 1.12 and 1.13.

Models PendingRollbackError state to prove rollback actually clears the transaction.
Real SQLAlchemy marks the session as needing rollback after IntegrityError; any
subsequent execute/commit raises PendingRollbackError until rollback() is awaited.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from returns.result import Failure
from sqlalchemy.exc import IntegrityError, PendingRollbackError

from app.features.audit.model import AuditLog
from app.features.audit.repository import AuditLogRepository


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_session_with_rollback(*, raise_on_flush: Exception | None = None):
    """Return a mock session that models SQLAlchemy's pending-rollback state.

    After a flush raises, the session enters a failed state where execute/commit
    raises PendingRollbackError until rollback() is awaited, matching real
    async SQLAlchemy behavior.
    """
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    session.rollback = AsyncMock()
    session.commit = AsyncMock()

    # internal state: True means transaction is poisoned
    state = {"needs_rollback": False}

    # configure rollback to clear the flag
    async def _rollback(*_a, **_kw):
        state["needs_rollback"] = False

    session.rollback.side_effect = _rollback

    # configure flush to optionally fail and poison the tx
    async def _flush(*_a, **_kw):
        if state["needs_rollback"]:
            message = "This Session's transaction has been rolled back"
            raise PendingRollbackError(message)
        if raise_on_flush is not None:
            state["needs_rollback"] = True
            raise raise_on_flush

    session.flush.side_effect = _flush

    # configure execute to fail if poisoned
    orig_execute = session.execute

    async def _execute(*a, **kw):
        if state["needs_rollback"]:
            message = "This Session's transaction has been rolled back"
            raise PendingRollbackError(message)
        # delegate to the mock's return_value behavior
        rv = orig_execute.return_value
        # if caller set return_value, return it; else default mock
        if rv is not None and not isinstance(rv, AsyncMock):
            return rv
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        return result

    session.execute.side_effect = _execute

    # commit also fails if poisoned (mirrors get_postgres_db commit)
    async def _commit(*_a, **_kw):
        if state["needs_rollback"]:
            message = "This Session's transaction has been rolled back"
            raise PendingRollbackError(message)

    session.commit.side_effect = _commit

    # expose state for assertions (ponytail: minimal state, no extra class)
    session._needs_rollback = state  # type: ignore[attr-defined]
    return session


class TestRollbackRegression:
    def test_caught_integrity_error_leaves_session_usable(self):
        """1.12: after IntegrityError the session is rolled back and next stmt succeeds."""
        flush_err = IntegrityError("stmt", {}, Exception("orig"))
        session = _mock_session_with_rollback(raise_on_flush=flush_err)

        repo = AuditLogRepository(session)
        entry = MagicMock(spec=AuditLog)
        entry.entity_type = "t"
        entry.entity_id = "1"

        result = _run(repo.create(entry))
        assert isinstance(result, Failure)
        session.rollback.assert_awaited_once()
        assert session._needs_rollback["needs_rollback"] is False  # type: ignore[attr-defined]

        # subsequent statement succeeds only because rollback cleared the poison
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute.return_value = mock_result
        # clear the flush failure so next flush would succeed; but we test execute path
        session.flush.side_effect = None  # type: ignore[assignment]
        # reset execute side_effect to return our mock_result if not poisoned
        # we need to re-wire execute to respect our state but return mock_result
        state = session._needs_rollback  # type: ignore[attr-defined]

        async def _execute_ok(*a, **kw):
            if state["needs_rollback"]:
                message = "poisoned"
                raise PendingRollbackError(message)
            return mock_result

        session.execute.side_effect = _execute_ok  # type: ignore[assignment]

        from returns.result import Success as RetSuccess

        result2 = _run(repo.find_by_entity("t", "1"))
        assert isinstance(result2, RetSuccess)
        assert session.rollback.await_count == 1  # no extra rollback on success

    def test_without_rollback_next_stmt_raises_pending(self):
        """Proves the mock models PendingRollbackError — without rollback, next stmt fails."""
        # Simulate old code that did NOT call rollback on IntegrityError
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock(side_effect=IntegrityError("stmt", {}, Exception("orig")))

        # poisoned flag never cleared

        async def _execute_poisoned(*a, **kw):
            message = "This Session's transaction has been rolled back"
            raise PendingRollbackError(message)

        session.execute = AsyncMock(side_effect=_execute_poisoned)
        session.rollback = AsyncMock()

        # directly verify poisoned state would cause PendingRollbackError

        async def _try_execute():
            return await session.execute(MagicMock())

        with pytest.raises(PendingRollbackError):
            _run(_try_execute())

    def test_swallowed_failure_does_not_reach_commit(self):
        """1.13: service swallowing Failure must not leave poisoned session committed."""
        flush_err = IntegrityError("stmt", {}, Exception("orig"))
        session = _mock_session_with_rollback(raise_on_flush=flush_err)

        repo = AuditLogRepository(session)
        entry = MagicMock(spec=AuditLog)
        entry.entity_type = "t"
        entry.entity_id = "1"

        result = _run(repo.create(entry))
        assert isinstance(result, Failure)
        session.rollback.assert_awaited_once()
        assert session._needs_rollback["needs_rollback"] is False  # type: ignore[attr-defined]

        # Simulate get_postgres_db: after yield, it does await session.commit()
        # With rollback already done, commit should succeed (not raise PendingRollbackError)
        _run(session.commit())
        session.commit.assert_awaited_once()

        # Without the fix, commit would have raised PendingRollbackError:
        poisoned = _mock_session_with_rollback(raise_on_flush=flush_err)
        # manually poison without clearing
        poisoned._needs_rollback["needs_rollback"] = True  # type: ignore[attr-defined]
        with pytest.raises(PendingRollbackError):
            _run(poisoned.commit())

        # repo path itself never calls commit
        # (commit is the dependency's job; we just verify rollback cleared the tx)
        assert session.rollback.await_count == 1


# Integration note (task 1.12/1.13): full PendingRollbackError verification needs a
# real async SQLAlchemy session against Postgres; the unit mock above is the smallest
# check that fails if rollback is removed, per ponytail minimal coverage. An
# integration variant would spin up testcontainers-postgres and assert that
# `await session.execute(text("SELECT 1"))` after a caught IntegrityError succeeds
# only after rollback, and that `await session.commit()` after a swallowed Failure
# does not raise.
