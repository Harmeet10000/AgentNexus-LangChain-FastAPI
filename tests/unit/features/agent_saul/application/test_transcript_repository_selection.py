"""Selection-contract tests for the bounded transcript reconciliation sweep."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from returns.result import Success
from sqlalchemy.dialects import postgresql

from app.features.agent_saul.adapters.persistence.models import ChatMessage, ChatSession
from app.features.agent_saul.application.transcript.repository import (
    ChatTranscriptRepository,
    _schedule_stale_session_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from sqlalchemy import Select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class _ScalarRows:
    def __init__(self, rows: Sequence[ChatSession]) -> None:
        self._rows = rows

    def scalars(self) -> _ScalarRows:
        return self

    def all(self) -> Sequence[ChatSession]:
        return self._rows


class _CapturingSession:
    def __init__(self, batches: Sequence[Sequence[ChatSession]]) -> None:
        self._batches = iter(batches)
        self.statements: list[Select[tuple[ChatSession]]] = []

    async def __aenter__(self) -> _CapturingSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def execute(self, statement: Select[tuple[ChatSession]]) -> _ScalarRows:
        self.statements.append(statement)
        return _ScalarRows(next(self._batches))


class _SessionFactory:
    def __init__(self, session: _CapturingSession) -> None:
        self._session = session

    def __call__(self) -> _CapturingSession:
        return self._session


def _session(session_id: str, checkpoint_ts: datetime) -> ChatSession:
    return ChatSession(
        session_id=session_id,
        user_id="user-1",
        schema_version=1,
        status="active",
        turn_count=0,
        last_checkpoint_id=f"cp-{session_id}",
        last_checkpoint_ts=checkpoint_ts,
    )


def test_pending_plan_decision_schema_has_no_plan_content_or_fingerprint_column() -> None:
    session_columns = ChatSession.__table__.columns
    message_columns = ChatMessage.__table__.columns

    assert "pending_plan_decision_key" in session_columns
    assert "pending_plan_decision_modified" in session_columns
    assert "pending_plan_decision_checkpoint_id" in session_columns
    assert "plan_fingerprint" not in message_columns


async def test_stale_selection_uses_disjoint_query_and_recovery_lanes() -> None:
    checkpoint_ts = datetime(2026, 1, 1, tzinfo=UTC)
    query_backed = _session("query-backed", checkpoint_ts)
    recoverable = _session("first-write-recovery", checkpoint_ts)
    session = _CapturingSession(((query_backed,), (recoverable,)))
    factory = cast("async_sessionmaker[AsyncSession]", _SessionFactory(session))
    repo = ChatTranscriptRepository(factory)

    result = await repo.list_stale_sessions(
        older_than=datetime(2026, 2, 1, tzinfo=UTC),
        terminal_statuses=frozenset({"completed", "failed", "abandoned_expired"}),
        limit=2,
    )

    assert isinstance(result, Success)
    assert [row.session_id for row in result.unwrap()] == [
        "query-backed",
        "first-write-recovery",
    ]
    assert len(session.statements) == 2
    query_sql = str(
        session.statements[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    ambiguous_sql = str(
        session.statements[1].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "EXISTS" in query_sql
    assert "NOT (EXISTS" not in query_sql
    assert "ORDER BY chat_sessions.last_checkpoint_ts, chat_sessions.session_id" in query_sql
    assert "NOT (EXISTS" in ambiguous_sql
    assert (
        "ORDER BY chat_sessions.last_checkpoint_ts DESC, chat_sessions.session_id" in ambiguous_sql
    )
    assert "LIMIT 2" in query_sql
    assert "LIMIT 2" in ambiguous_sql


def test_stale_scheduler_reserves_both_lanes_and_fills_unused_quota() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    query_backed = tuple(_session(f"query-{index}", now) for index in range(4))
    ambiguous = tuple(_session(f"ambiguous-{index}", now) for index in range(4))

    scheduled = _schedule_stale_session_rows(
        query_backed=query_backed,
        ambiguous=ambiguous,
        limit=3,
    )
    assert [row.session_id for row in scheduled] == ["query-0", "query-1", "ambiguous-0"]

    query_fill = _schedule_stale_session_rows(
        query_backed=query_backed,
        ambiguous=(),
        limit=3,
    )
    assert [row.session_id for row in query_fill] == ["query-0", "query-1", "query-2"]

    ambiguous_fill = _schedule_stale_session_rows(
        query_backed=(),
        ambiguous=ambiguous,
        limit=3,
    )
    assert [row.session_id for row in ambiguous_fill] == [
        "ambiguous-0",
        "ambiguous-1",
        "ambiguous-2",
    ]


def test_stale_scheduler_limit_one_is_query_first_with_recovery_fallback() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    query_backed = (_session("query", now),)
    ambiguous = (_session("recovery", now),)

    query_first = _schedule_stale_session_rows(
        query_backed=query_backed,
        ambiguous=ambiguous,
        limit=1,
    )
    assert [row.session_id for row in query_first] == ["query"]

    recovery_fallback = _schedule_stale_session_rows(
        query_backed=(),
        ambiguous=ambiguous,
        limit=1,
    )
    assert [row.session_id for row in recovery_fallback] == ["recovery"]
    assert (
        _schedule_stale_session_rows(
            query_backed=query_backed,
            ambiguous=ambiguous,
            limit=0,
        )
        == ()
    )
