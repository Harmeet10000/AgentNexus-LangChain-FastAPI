"""Regression for repository rollback — tasks 1.12 and 1.13."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from returns.result import Failure
from sqlalchemy.exc import IntegrityError

from app.features.audit.model import AuditLog
from app.features.audit.repository import AuditLogRepository


def _run(coro):
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_session_with_rollback():
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    session.rollback = AsyncMock()
    return session


class TestRollbackRegression:
    def test_caught_integrity_error_leaves_session_usable(self):
        """1.12: after IntegrityError the session is rolled back and next stmt succeeds."""
        session = _mock_session_with_rollback()
        # first flush raises IntegrityError -> repository should rollback and return Failure
        session.flush.side_effect = IntegrityError("stmt", {}, "orig")
        # second execute (next statement) should succeed if rollback happened
        # simulate a second call to find_by_entity that uses execute
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        # after first failure, we reset execute to succeed
        # But first call fails at flush, not execute, so rollback must have been called
        repo = AuditLogRepository(session)
        entry = MagicMock(spec=AuditLog)
        entry.entity_type = "t"
        entry.entity_id = "1"

        result = _run(repo.create(entry))
        assert isinstance(result, Failure)
        session.rollback.assert_awaited_once()

        # subsequent statement succeeds
        session.execute.return_value = mock_result
        # clear side_effect for next call
        session.flush.side_effect = None
        result2 = _run(repo.find_by_entity("t", "1"))
        # Should be Success, not PendingRollbackError
        from returns.result import Success as RetSuccess

        assert isinstance(result2, RetSuccess)
        assert session.rollback.await_count == 1  # no extra rollback on success

    def test_swallowed_failure_does_not_reach_commit(self):
        """1.13: service swallowing Failure must not leave poisoned session committed."""
        session = _mock_session_with_rollback()
        session.flush.side_effect = IntegrityError("stmt", {}, "orig")
        session.commit = AsyncMock()

        repo = AuditLogRepository(session)
        entry = MagicMock(spec=AuditLog)
        entry.entity_type = "t"
        entry.entity_id = "1"

        result = _run(repo.create(entry))
        assert isinstance(result, Failure)
        # Simulate service swallowing: no exception propagates, so get_postgres_db would commit.
        # The repository must have already rolled back, so session is clean.
        session.rollback.assert_awaited_once()
        # If service swallowed and then dependency commits, commit should not carry failed write.
        # Our regression: rollback precedes return, so poisoned tx is cleared.
        # No additional commit was called by repo; swallowing service would call commit at layer above.
        # Verify that commit hasn't been auto-called by repo path.
        session.commit.assert_not_awaited()
