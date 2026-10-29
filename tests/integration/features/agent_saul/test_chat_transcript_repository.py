"""ChatTranscriptRepository behaviour on a live Postgres (doc §13 DoD).

Run explicitly: ``uv run pytest tests/integration/features/agent_saul -m integration``
Requires migrations through 0022 applied (chat tables at the projected shape).
The chain-of-custody properties pinned here: batch no-op replay, fill-once
tokens, turn_seq stability across pause/resume, write-failure healing, the
schema gate, and Graphiti/GC independence of the transcript.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

import pytest
from returns.result import Failure, Success
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.connections.postgres import get_database_url
from app.features.agent_saul.application.transcript.errors import (
    TranscriptInfrastructureError,
    TranscriptNotFoundError,
    TranscriptSchemaError,
)
from app.features.agent_saul.application.transcript.projection import (
    ContentRole,
    ProjectedRow,
    Projection,
    SessionStatus,
    project,
    snapshot_to_view,
)
from app.features.agent_saul.application.transcript.repository import ChatTranscriptRepository
from app.features.agent_saul.domain.state import STATE_SCHEMA_VERSION

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

USER_ID = "integration-user"


class _Db(NamedTuple):
    """One engine's sessionmaker and its repository, for a single test."""

    factory: async_sessionmaker[Any]
    repo: ChatTranscriptRepository
    engine: Any


@pytest.fixture
async def db() -> AsyncIterator[_Db]:
    """A fresh engine per test, disposed at teardown.

    A shared module-level engine cannot work here: asyncpg connections are
    bound to the event loop that created them and pytest-asyncio runs each
    test on its own loop. Leaking one engine per test (no dispose) exhausts
    the TigerData connection cap partway through this file, and the late
    605-row apply then fails as an opaque write failure instead of the
    behaviour under test.
    """
    engine = create_async_engine(get_database_url(flavour="async"), pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield _Db(factory=factory, repo=ChatTranscriptRepository(factory), engine=engine)
    finally:
        await engine.dispose()


async def _apply(
    repo: ChatTranscriptRepository,
    projection: Projection,
    *,
    session_id: str,
    **kwargs: Any,
) -> Any:
    """Apply the projection the way a boundary does (§5.3 guard).

    The transcript watermark is read BEFORE the projection is handed to the
    repository — the monotonicity guard requires that observation, and a test
    helper that skipped it would silently disable the guard.
    """
    peek = await repo.peek_session_checkpoint(session_id=session_id)
    prior = peek.unwrap()[0] if isinstance(peek, Success) else None
    return await repo.apply_projection(
        projection,
        session_id=session_id,
        prior_checkpoint_id=prior,
        **kwargs,
    )


def _projection(
    *,
    query_id: str,
    status: SessionStatus = SessionStatus.ACTIVE,
    rows: tuple[ProjectedRow, ...] = (),
    checkpoint_id: str | None = None,
) -> Projection:
    return Projection(
        session_status=status,
        no_op=None,
        watermark_checkpoint_id=checkpoint_id or f"cp-{query_id}",
        watermark_ts=datetime.now(tz=UTC),
        query_source_key=query_id,
        rows=rows,
    )


def _row(
    ordinal: int, source_key: str, kind: str, *, role: ContentRole | None = None
) -> ProjectedRow:
    return ProjectedRow(
        kind=kind,
        role=role
        or (
            ContentRole.USER
            if kind in {"query", "clarification_reply", "plan_decision"}
            else ContentRole.ASSISTANT
        ),
        ordinal=ordinal,
        source_key=source_key,
        content="integration row",
    )


async def test_replay_is_batch_noop_and_item_deduped(db: _Db) -> None:
    repo = db.repo
    session_id = str(uuid4())
    rows = (
        _row(0, f"query:{session_id}", "query"),
        _row(1, "interrupt:int-1", "clarification_prompt"),
    )
    first = await _apply(
        repo,
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-1"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)
    outcome = first.unwrap()
    assert outcome.applied is True

    # Same checkpoint, no rows → batch no-op.
    replay = await _apply(
        repo,
        _projection(query_id=f"query:{session_id}", checkpoint_id="cp-1"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(replay, Success)
    replayed = replay.unwrap()
    assert replayed.applied is False

    # Same checkpoint, WITH rows (crash-heal form): no duplicate rows land.
    heal = await _apply(
        repo,
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-1"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(heal, Success)
    assert heal.unwrap().rows_inserted == 0
    async with db.factory() as session:
        count = (
            await session.execute(
                sa_text("SELECT count(*) FROM chat_messages WHERE session_id = :sid"),
                {"sid": session_id},
            )
        ).scalar_one()
    assert int(count) == 2


async def test_tokens_fill_once_under_replay(db: _Db) -> None:
    repo = db.repo
    session_id = str(uuid4())
    rows = (_row(0, f"query:{session_id}", "query"),)

    await _apply(
        repo,
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-a"),
        user_id=USER_ID,
        session_id=session_id,
        usage={f"query:{session_id}": 420},
    )
    # A higher usage on replay must NOT overwrite (fill-once COALESCE).
    replay = await _apply(
        repo,
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-a"),
        user_id=USER_ID,
        session_id=session_id,
        usage={f"query:{session_id}": 999},
    )
    assert isinstance(replay, Success)  # batch no-op keeps tokens intact
    async with db.factory() as session:
        tokens = (
            await session.execute(
                sa_text("SELECT tokens_used FROM chat_messages WHERE session_id = :sid"),
                {"sid": session_id},
            )
        ).scalar_one()
    assert int(tokens) == 420


async def test_turn_seq_stable_across_pause_and_resume(db: _Db) -> None:
    repo = db.repo
    session_id = str(uuid4())
    q_key = f"query:{session_id}"

    # First boundary: query lands inside a fresh turn.
    r1 = await _apply(
        repo,
        _projection(query_id=q_key, rows=(_row(0, q_key, "query"),), checkpoint_id="cp-t1"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(r1, Success)
    assert r1.unwrap().turn_seq == 1
    assert r1.unwrap().turn_opened is True

    # Second boundary (same turn, post-resume): turn_seq MUST stay 1.
    r2 = await _apply(
        repo,
        _projection(
            query_id=q_key,
            status=SessionStatus.CLARIFYING,
            rows=(_row(1, "interrupt:int-r", "clarification_prompt"),),
            checkpoint_id="cp-t2",
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(r2, Success)
    assert r2.unwrap().turn_seq == 1
    assert r2.unwrap().turn_opened is False
    async with db.factory() as session:
        session_row = (
            await session.execute(
                sa_text("SELECT turn_count, status FROM chat_sessions WHERE session_id = :sid"),
                {"sid": session_id},
            )
        ).one()
    assert int(session_row[0]) == 1
    assert session_row[1] == "clarifying"


async def test_write_failure_returns_typed_failure_not_raised(db: _Db) -> None:
    repo = db.repo
    # A NUL byte in any text parameter is rejected by Postgres at bind time —
    # a deterministic driver failure to prove the edge sanitizes, not raises.
    failure = await _apply(
        repo,
        _projection(
            query_id="query:bad",
            rows=(_row(0, "query:bad", "query"),),
            checkpoint_id="cp-fail",
        ),
        user_id=USER_ID,
        session_id="bad\x00session",
    )
    assert isinstance(failure, Failure)
    error = failure.failure()
    assert isinstance(error, TranscriptInfrastructureError)
    # Driver detail never leaves the edge (§10).
    assert "asyncpg" not in error.message
    assert "0x00" not in error.message


async def test_schema_gate_fails_closed_on_unknown_version(db: _Db) -> None:
    repo = db.repo
    session_id = str(uuid4())
    await _apply(
        repo,
        _projection(
            query_id=f"query:{session_id}", rows=(_row(0, f"query:{session_id}", "query"),)
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    async with db.factory() as session:
        await session.execute(
            sa_text("UPDATE chat_sessions SET schema_version = 0 WHERE session_id = :sid"),
            {"sid": session_id},
        )
        await session.commit()
    meta = await repo.get_session_meta(session_id=session_id, user_id=USER_ID)
    assert isinstance(meta, Failure)
    assert isinstance(meta.failure(), TranscriptSchemaError)
    # Wrong owner → not found, never someone else's transcript.
    other = await repo.get_session_meta(session_id=session_id, user_id="someone-else")
    assert isinstance(other, Failure)
    assert isinstance(other.failure(), TranscriptNotFoundError)


async def test_record_plan_decision_fact_is_owner_schema_and_prompt_gated(db: _Db) -> None:
    repo = db.repo
    session_id = str(uuid4())
    prompt_key = "interrupt:fact-gated"
    query_key = f"query:{session_id}"
    created = await _apply(
        repo,
        _projection(
            query_id=query_key,
            rows=(
                _row(0, query_key, "query"),
                _row(1, prompt_key, "plan_prompt"),
            ),
            checkpoint_id="cp-fact-gated",
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(created, Success)

    wrong_owner = await repo.record_plan_decision_fact(
        session_id=session_id,
        user_id="another-user",
        decision_key=prompt_key,
        modified=False,
    )
    assert isinstance(wrong_owner, Failure)
    assert isinstance(wrong_owner.failure(), TranscriptNotFoundError)

    invalid_key = await repo.record_plan_decision_fact(
        session_id=session_id,
        user_id=USER_ID,
        decision_key="plaintext-plan-content",
        modified=False,
    )
    assert isinstance(invalid_key, Failure)

    recorded = await repo.record_plan_decision_fact(
        session_id=session_id,
        user_id=USER_ID,
        decision_key=prompt_key,
        modified=True,
    )
    assert isinstance(recorded, Success)
    replay = await repo.record_plan_decision_fact(
        session_id=session_id,
        user_id=USER_ID,
        decision_key=prompt_key,
        modified=True,
    )
    assert isinstance(replay, Success)
    assert replay.unwrap() is True

    changed_answer = await repo.record_plan_decision_fact(
        session_id=session_id,
        user_id=USER_ID,
        decision_key=prompt_key,
        modified=False,
    )
    assert isinstance(changed_answer, Failure)

    async with db.factory() as session:
        pending = (
            await session.execute(
                sa_text(
                    "SELECT pending_plan_decision_key, pending_plan_decision_modified "
                    "FROM chat_sessions WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
        ).one()
    assert pending.pending_plan_decision_key == prompt_key
    assert pending.pending_plan_decision_modified is True

    async with db.factory() as session, session.begin():
        await session.execute(
            sa_text("UPDATE chat_sessions SET schema_version = 0 WHERE session_id = :sid"),
            {"sid": session_id},
        )
    unsupported = await repo.record_plan_decision_fact(
        session_id=session_id,
        user_id=USER_ID,
        decision_key=prompt_key,
        modified=True,
    )
    assert isinstance(unsupported, Failure)
    assert isinstance(unsupported.failure(), TranscriptSchemaError)


async def test_plan_decision_fact_survives_repo_restart_and_hole_fill_is_idempotent(
    db: _Db,
) -> None:
    session_id = str(uuid4())
    query_key = f"query:{session_id}"
    prompt_key = "interrupt:restart-fact"
    decision_key = "decision:restart-fact"
    prompt = _projection(
        query_id=query_key,
        rows=(_row(0, query_key, "query"), _row(1, prompt_key, "plan_prompt")),
        checkpoint_id="cp-restart-prompt",
    )
    first = await _apply(db.repo, prompt, user_id=USER_ID, session_id=session_id)
    assert isinstance(first, Success)
    fact = await db.repo.record_plan_decision_fact(
        session_id=session_id,
        user_id=USER_ID,
        decision_key=prompt_key,
        modified=False,
    )
    assert isinstance(fact, Success)

    # Recreate the repository boundary before the resumed decision arrives.
    await db.engine.dispose()
    restarted_engine = create_async_engine(get_database_url(flavour="async"), pool_pre_ping=True)
    restarted_factory = async_sessionmaker(restarted_engine, expire_on_commit=False)
    restarted = ChatTranscriptRepository(restarted_factory)
    decision = _projection(
        query_id=query_key,
        rows=(_row(2, decision_key, "plan_decision"),),
        checkpoint_id="cp-restart-decision",
        status=SessionStatus.RESUMED,
    )
    applied = await restarted.apply_projection(
        decision,
        user_id=USER_ID,
        session_id=session_id,
        prior_checkpoint_id="cp-restart-prompt",
    )
    assert isinstance(applied, Success)
    replay = await restarted.apply_projection(
        decision,
        user_id=USER_ID,
        session_id=session_id,
        prior_checkpoint_id="cp-restart-decision",
    )
    assert isinstance(replay, Success)

    async with restarted_factory() as session:
        count = (
            await session.execute(
                sa_text(
                    "SELECT count(*) FROM chat_messages "
                    "WHERE session_id = :sid AND kind = 'plan_decision'"
                ),
                {"sid": session_id},
            )
        ).scalar_one()
        pending = (
            await session.execute(
                sa_text(
                    "SELECT pending_plan_decision_key, pending_plan_decision_modified "
                    "FROM chat_sessions WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
        ).one()
    assert count == 1
    assert pending.pending_plan_decision_key is None
    assert pending.pending_plan_decision_modified is None
    await restarted_engine.dispose()


async def test_turn_closed_flow_over_24h_pause_window(db: _Db) -> None:
    repo = db.repo
    session_id = str(uuid4())
    # The D4 chain-of-custody key IS the HumanMessage id; the projector keys
    # the query row by it. Use the raw id everywhere so the seeded turn and
    # the projected turn resolve to the same turn_seq.
    q_key = session_id

    # Seed the paused turn: query + clarification prompt recorded at T1.
    await _apply(
        repo,
        _projection(
            query_id=q_key,
            status=SessionStatus.CLARIFYING,
            rows=(
                _row(0, q_key, "query"),
                _row(1, "interrupt:paused-clr", "clarification_prompt"),
            ),
        ),
        user_id=USER_ID,
        session_id=session_id,
    )

    # 26h later with an interrupt still pending and no heartbeat: the
    # projector (not the repo) derives abandoned_expired and closes the turn.
    now = datetime.now(tz=UTC)
    view = snapshot_to_view(
        _snapshot_with(
            user_id=USER_ID,
            query_id=q_key,
            checkpoint_ts=(now - timedelta(hours=26)).isoformat(),
            with_interrupt=True,
        )
    )
    turn_context = await repo.get_turn_context(session_id=session_id, user_id=USER_ID)
    assert isinstance(turn_context, Success)
    derived = project(
        view,
        turn_context.unwrap().existing_keys,
        turn_context.unwrap().turn_rows,
        now,
        aliveness=False,
    )
    assert derived.session_status == SessionStatus.ABANDONED_EXPIRED
    applied = await _apply(repo, derived, user_id=USER_ID, session_id=session_id)
    assert isinstance(applied, Success)
    assert applied.unwrap().rows_inserted == 1  # the turn_closed row


def _snapshot_with(
    *, user_id: str, query_id: str, checkpoint_ts: str, with_interrupt: bool = False
) -> Any:
    """A minimal StateSnapshot stand-in running through snapshot_to_view.

    The projector reads StateSnapshot duck-typed (attribute access), so a
    SimpleNamespace mirrors exactly what ``aget_state`` yields. The query
    must carry the D4 tag — that IS the chain-of-custody anchor.
    """
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage

    tasks: tuple[Any, ...] = ()
    if with_interrupt:
        tasks = (
            SimpleNamespace(
                interrupts=[
                    SimpleNamespace(
                        id="paused-clr",
                        value={"type": "clarification_needed", "message": "still here"},
                    ),
                ],
            ),
        )
    return SimpleNamespace(
        values={
            "schema_version": STATE_SCHEMA_VERSION,
            "user_id": user_id,
            "messages": [
                HumanMessage(
                    content="integration query",
                    additional_kwargs={"saul_kind": "query"},
                    id=query_id,
                ),
            ],
        },
        config={"configurable": {"thread_id": "t", "checkpoint_id": "cp-dead"}},
        parent_config=None,
        tasks=tasks,
        metadata={},
        created_at=checkpoint_ts,
        next=(),
        step_metadata=None,
    )


async def test_checkpointer_gc_cannot_touch_the_transcript(db: _Db) -> None:
    """Checkpointer lifecycle is namespace-orthogonal to chat rows (D-coverage)."""
    repo = db.repo
    session_id = str(uuid4())
    q_key = session_id
    persisted = await _apply(
        repo,
        _projection(query_id=q_key, rows=(_row(0, q_key, "query"),)),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(persisted, Success)
    # Wipe the checkpointer's namespace for this thread id (same as graph GC),
    # through the production-owned pool factory.
    from app.connections.checkpointer import setup_langgraph_checkpointer

    checkpointer = await setup_langgraph_checkpointer()
    try:
        await checkpointer.adelete_thread(f"gc-{session_id}")  # unknown thread → no-op
    finally:
        # The saver itself exposes no close(); the pool it owns does.
        await checkpointer.conn.close()
    transcript = await repo.read_transcript(session_id=session_id, user_id=USER_ID, limit=10)
    assert isinstance(transcript, Success)
    rows = list(transcript.unwrap())
    assert len(rows) == 1
    assert rows[0].kind == "query"


async def test_owner_mismatch_refused_before_writes(db: _Db) -> None:
    """§9: a wrong-owner projection is a sanitized not-found, no rows, no move."""
    repo = db.repo
    session_id = str(uuid4())
    rows = (
        _row(0, f"query:{session_id}", "query"),
        _row(1, "interrupt:int-owner", "clarification_prompt"),
    )

    first = await _apply(
        repo,
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-owner"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)

    attack = await _apply(
        repo,
        _projection(
            query_id=f"query:{session_id}",
            rows=(_row(2, "interrupt:int-stranger", "clarification_prompt"),),
            checkpoint_id="cp-stranger",
        ),
        user_id="stranger-user",
        session_id=session_id,
    )
    assert isinstance(attack, Failure)
    assert isinstance(attack.failure(), TranscriptNotFoundError)

    # Nothing about the real session moved: no stray rows, watermark intact.
    async with db.factory() as session:
        stranger_rows = await session.execute(
            sa_text("SELECT count(*) FROM chat_messages WHERE user_id = 'stranger-user'"),
        )
        assert int(stranger_rows.scalar_one()) == 0
        cp = await session.execute(
            sa_text("SELECT last_checkpoint_id FROM chat_sessions WHERE session_id = :sid"),
            {"sid": session_id},
        )
        assert (cp.scalar_one()) == "cp-owner"


async def test_sweep_context_returns_owner_with_context(db: _Db) -> None:
    """§8: the sweep helper is (owner_user_id, turn_context) — never bare.

    _reconcile_once unpacks the pair; a bare context here would TypeError
    every sweep pass into its per-thread guard and silently disable
    background reconciliation AND checkpoint GC.
    """
    repo = db.repo
    session_id = str(uuid4())
    first = await _apply(
        repo,
        _projection(
            query_id=f"query:{session_id}",
            rows=(_row(0, f"query:{session_id}", "query"),),
            checkpoint_id="cp-sweep",
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)

    result = await repo.get_sweep_context(session_id=session_id)
    assert isinstance(result, Success)
    owner, context = result.unwrap()
    assert owner == USER_ID
    assert f"query:{session_id}" in context.existing_keys

    unknown = await repo.get_sweep_context(session_id=str(uuid4()))
    assert isinstance(unknown, Failure)
    assert isinstance(unknown.failure(), TranscriptNotFoundError)


async def test_stale_sweep_prioritizes_query_backed_work_without_hiding_recovery(
    db: _Db,
) -> None:
    """A bounded batch must make progress without disabling first-write healing.

    Sessions without a projected query are deliberately treated as ambiguous:
    integration debris and a real checkpoint whose first transcript write
    failed have the same durable SQL shape. Query-backed work gets most of the
    batch while one freshest ambiguous candidate receives reserved service.
    """
    repo = db.repo
    prefix = f"sweep-{uuid4().hex}-"
    query_ids = [f"{prefix}query-{index:02d}" for index in reversed(range(25))]
    debris_ids = [f"{prefix}anchor-{index:02d}" for index in range(25)]
    first_recovery_id = f"{prefix}recovery-1"
    second_recovery_id = f"{prefix}recovery-2"
    terminal = frozenset(
        {
            str(SessionStatus.COMPLETED.value),
            str(SessionStatus.FAILED.value),
            str(SessionStatus.ABANDONED_EXPIRED.value),
        }
    )
    query_ts = datetime(2000, 1, 1, tzinfo=UTC)
    debris_ts = datetime(1990, 1, 1, tzinfo=UTC)
    first_recovery_ts = datetime(2005, 1, 1, tzinfo=UTC)
    second_recovery_ts = datetime(2006, 1, 1, tzinfo=UTC)

    try:
        async with db.factory() as session, session.begin():
            await session.execute(
                sa_text(
                    "INSERT INTO chat_sessions "
                    "(session_id, user_id, status, last_checkpoint_id, last_checkpoint_ts) "
                    "VALUES (:session_id, :user_id, 'active', :checkpoint_id, :checkpoint_ts)"
                ),
                [
                    {
                        "session_id": session_id,
                        "user_id": USER_ID,
                        "checkpoint_id": f"cp-{session_id}",
                        "checkpoint_ts": query_ts,
                    }
                    for session_id in query_ids
                ]
                + [
                    {
                        "session_id": session_id,
                        "user_id": USER_ID,
                        "checkpoint_id": "cp-unprojected",
                        "checkpoint_ts": debris_ts,
                    }
                    for session_id in debris_ids
                ]
                + [
                    {
                        "session_id": first_recovery_id,
                        "user_id": USER_ID,
                        "checkpoint_id": "cp-recovery-1",
                        "checkpoint_ts": first_recovery_ts,
                    }
                ],
            )
            await session.execute(
                sa_text(
                    "INSERT INTO chat_messages "
                    "(session_id, user_id, role, content, turn_seq, ordinal, kind, "
                    "source_key, source_checkpoint_id) VALUES "
                    "(:session_id, :user_id, 'user', 'query', 1, 0, 'query', "
                    ":source_key, :checkpoint_id)"
                ),
                [
                    {
                        "session_id": session_id,
                        "user_id": USER_ID,
                        "source_key": session_id,
                        "checkpoint_id": f"cp-{session_id}",
                    }
                    for session_id in query_ids
                ],
            )

        bounded = await repo.list_stale_sessions(
            older_than=datetime(2010, 1, 1, tzinfo=UTC),
            terminal_statuses=terminal,
            limit=20,
        )
        assert isinstance(bounded, Success)
        selected = [meta.session_id for meta in bounded.unwrap()]
        assert len(selected) == 20
        assert selected[:19] == sorted(query_ids)[:19]
        assert selected[19] == first_recovery_id
        assert not set(selected).intersection(debris_ids)

        # After the selected recovery candidate leaves the non-terminal lane,
        # a newer failed-first-write candidate receives the reserved slot on
        # the next bounded pass instead of the same old debris monopolizing it.
        async with db.factory() as session, session.begin():
            await session.execute(
                sa_text("UPDATE chat_sessions SET status = 'completed' WHERE session_id = :sid"),
                {"sid": first_recovery_id},
            )
            await session.execute(
                sa_text(
                    "INSERT INTO chat_sessions "
                    "(session_id, user_id, status, last_checkpoint_id, last_checkpoint_ts) "
                    "VALUES (:session_id, :user_id, 'active', :checkpoint_id, :checkpoint_ts)"
                ),
                {
                    "session_id": second_recovery_id,
                    "user_id": USER_ID,
                    "checkpoint_id": "cp-recovery-2",
                    "checkpoint_ts": second_recovery_ts,
                },
            )
        next_bounded = await repo.list_stale_sessions(
            older_than=datetime(2010, 1, 1, tzinfo=UTC),
            terminal_statuses=terminal,
            limit=20,
        )
        assert isinstance(next_bounded, Success)
        next_selected = [meta.session_id for meta in next_bounded.unwrap()]
        assert len(next_selected) == 20
        assert next_selected[:19] == sorted(query_ids)[:19]
        assert next_selected[19] == second_recovery_id
        assert not set(next_selected).intersection(debris_ids)

        # With the query lane empty under this cutoff, its unused quota is
        # filled from the ambiguous lane while preserving the hard bound.
        debris_only = await repo.list_stale_sessions(
            older_than=datetime(1995, 1, 1, tzinfo=UTC),
            terminal_statuses=terminal,
            limit=20,
        )
        assert isinstance(debris_only, Success)
        debris_selected = [meta.session_id for meta in debris_only.unwrap()]
        assert len(debris_selected) == 20
        assert debris_selected == sorted(debris_ids)[:20]
    finally:
        async with db.factory() as session, session.begin():
            await session.execute(
                sa_text("DELETE FROM chat_messages WHERE session_id LIKE :prefix"),
                {"prefix": f"{prefix}%"},
            )
            await session.execute(
                sa_text("DELETE FROM chat_sessions WHERE session_id LIKE :prefix"),
                {"prefix": f"{prefix}%"},
            )


async def test_gc_stamp_excludes_freshly_audited_session(db: _Db) -> None:
    """§8: exclusion is stamp-generation AND pending-aware, not one-shot.

    A terminal session past retention IS a candidate with no stamp; a PENDING
    pre-delete stamp excludes it only while fresh (another sweep's critical
    section), re-enters once stale, and a CONFIRMED stamp excludes while it
    covers the checkpoint generation — a later NEWER checkpoint must
    qualify again, which the SQL comparison — not a boolean flag — guarantees.
    """
    repo = db.repo
    session_id = str(uuid4())
    first = await _apply(
        repo,
        _projection(
            query_id=f"query:{session_id}",
            rows=(_row(0, f"query:{session_id}", "query"),),
            checkpoint_id="cp-gc",
            status=SessionStatus.COMPLETED,
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)
    older_than = datetime.now(tz=UTC) + timedelta(days=9)  # far horizon: fresh rows qualify
    fresh = await repo.list_gc_candidates(older_than=older_than, limit=1000)
    assert isinstance(fresh, Success)
    assert session_id in {m.session_id for m in fresh.unwrap()}

    # The PENDING pre-delete stamp excludes the candidate while fresh (§8) —
    # another sweep may legitimately be inside its critical section.
    stamped = await repo.stamp_gc_audit(
        session_id=session_id,
        deleted_checkpoint_id="cp-gc",
        swept_at=datetime.now(tz=UTC),
        pending=True,
    )
    assert isinstance(stamped, Success)
    after_pending = await repo.list_gc_candidates(older_than=older_than, limit=1000)
    assert isinstance(after_pending, Success)
    assert session_id not in {m.session_id for m in after_pending.unwrap()}

    # A worker death between stamp and delete surfaces the candidate again
    # once the pending record goes STALE (§8 pending-retry).
    stale_pending = await repo.stamp_gc_audit(
        session_id=session_id,
        deleted_checkpoint_id="cp-gc",
        swept_at=datetime.now(tz=UTC) - timedelta(minutes=30),
        pending=True,
    )
    assert isinstance(stale_pending, Success)
    retry = await repo.list_gc_candidates(older_than=older_than, limit=1000)
    assert isinstance(retry, Success)
    assert session_id in {m.session_id for m in retry.unwrap()}

    # A CONFIRMED stamp covers the (old) checkpoint generation — excluded.
    confirmed = await repo.stamp_gc_audit(
        session_id=session_id,
        deleted_checkpoint_id="cp-gc",
        swept_at=datetime.now(tz=UTC),
        pending=False,
    )
    assert isinstance(confirmed, Success)
    after = await repo.list_gc_candidates(older_than=older_than, limit=1000)
    assert isinstance(after, Success)
    assert session_id not in {m.session_id for m in after.unwrap()}

    # A NEWER checkpoint generation re-qualifies: its watermark postdates the
    # stamp, so the stamp no longer covers the checkpoint (§8).
    newer = await _apply(
        repo,
        _projection(
            query_id=f"query:{session_id}",
            rows=(_row(1, "interrupt:gc2", "clarification_prompt"),),
            checkpoint_id="cp-gc-2",
            status=SessionStatus.COMPLETED,
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(newer, Success)
    reopened = await repo.list_gc_candidates(older_than=older_than, limit=1000)
    assert isinstance(reopened, Success)
    assert session_id in {m.session_id for m in reopened.unwrap()}


async def test_apply_rejects_a_stale_projection_under_the_row_lock(db: _Db) -> None:
    """§5.3: a caller that read an OLD watermark must not move the cache.

    Row locking only ORDERS writers — without the comparison, a sweep that
    read checkpoint A and applies late would overwrite a newer writer's
    status and watermark with A's (older) values. The comparison is against
    the watermark the caller OBSERVED, so a legitimately newer projection
    still lands while a stale one answers a no-op.
    """
    repo = db.repo
    session_id = str(uuid4())
    q = f"query:{session_id}"

    first = await _apply(
        repo,
        _projection(
            query_id=q,
            rows=(_row(0, q, "query"),),
            checkpoint_id="cp-a",
            status=SessionStatus.ACTIVE,
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)

    # A boundary that observed "cp-a" and projects the NEXT checkpoint
    # "cp-b": observed-prior is older by one step — must still land.
    second = await _apply(
        repo,
        _projection(query_id=q, checkpoint_id="cp-b", status=SessionStatus.COMPLETED),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(second, Success)
    assert second.unwrap().applied is True

    # A stale sweep that read "cp-a" before that writer landed now applies a
    # FAILED verdict derived from the OLD checkpoint → refused, nothing moves.
    stale = await repo.apply_projection(
        _projection(query_id=q, checkpoint_id="cp-a", status=SessionStatus.FAILED),
        user_id=USER_ID,
        session_id=session_id,
        prior_checkpoint_id="cp-a",
    )
    assert isinstance(stale, Success)
    assert stale.unwrap().applied is False
    assert stale.unwrap().reason == "stale_projection"

    async with db.factory() as session:
        row = (
            await session.execute(
                sa_text(
                    "SELECT status, last_checkpoint_id FROM chat_sessions WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
        ).one()
        assert row.status == str(SessionStatus.COMPLETED.value)
        assert row.last_checkpoint_id == "cp-b"


async def test_usage_only_fill_lands_on_existing_row(db: _Db) -> None:
    """§7: segment totals land once even when the projector deduped the row."""
    repo = db.repo
    session_id = str(uuid4())
    rows = (
        _row(0, f"query:{session_id}", "query"),
        _row(1, "interrupt:fill-target", "clarification_prompt"),
    )

    await _apply(
        repo,
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-fill-1"),
        user_id=USER_ID,
        session_id=session_id,
    )

    # Follow-up boundary (new checkpoint) whose target row already exists: the
    # projector drops it, so the usage rides as a usage-only fill. The
    # content-free metadata breakdown merges into the same row (§7), fill-once.
    fill = await _apply(
        repo,
        _projection(
            query_id=f"query:{session_id}",
            rows=(_row(2, "interrupt:fill-other", "clarification_prompt"),),
            checkpoint_id="cp-fill-2",
        ),
        user_id=USER_ID,
        session_id=session_id,
        usage={"interrupt:fill-target": 77, "interrupt:never-inserted": 5},
        usage_metadata={
            "interrupt:fill-target": {"usage": {"input": 40, "output": 37}, "tool_calls": ()}
        },
    )
    assert isinstance(fill, Success)
    async with db.factory() as session:
        landed = (
            await session.execute(
                sa_text(
                    "SELECT tokens_used, extra_metadata FROM chat_messages "
                    "WHERE session_id = :sid AND source_key = 'interrupt:fill-target'"
                ),
                {"sid": session_id},
            )
        ).one()
        assert landed.tokens_used == 77
        assert landed.extra_metadata["usage"] == {"input": 40, "output": 37}
        # Fill-once for the merge gate as well: a replayed boundary cannot
        # rewrite it.
        replay = await _apply(
            repo,
            _projection(
                query_id=f"query:{session_id}",
                rows=(_row(2, "interrupt:fill-other", "clarification_prompt"),),
                checkpoint_id="cp-fill-2",
            ),
            user_id=USER_ID,
            session_id=session_id,
            usage={"interrupt:fill-target": 77},
            usage_metadata={"interrupt:fill-target": {"usage": {"input": 999, "output": 999}}},
        )
        assert isinstance(replay, Success)
        unchanged = (
            await session.execute(
                sa_text(
                    "SELECT tokens_used, extra_metadata FROM chat_messages "
                    "WHERE session_id = :sid AND source_key = 'interrupt:fill-target'"
                ),
                {"sid": session_id},
            )
        ).one()
        assert unchanged.tokens_used == 77
        assert unchanged.extra_metadata["usage"] == {"input": 40, "output": 37}


async def test_read_transcript_paging_completes_large_transcripts(db: _Db) -> None:
    """§8 export: offset paging returns every row in display order (D3)."""
    repo = db.repo
    session_id = str(uuid4())
    big = tuple(
        _row(
            i, f"query:{session_id}" if i == 0 else f"row-{i}", "query" if i == 0 else "plan_prompt"
        )
        for i in range(605)
    )
    first = await _apply(
        repo,
        _projection(
            query_id=f"query:{session_id}",
            rows=big,
            checkpoint_id="cp-paging",
            status=SessionStatus.COMPLETED,
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)

    seen: list[int] = []
    page_guard = 0
    cursor_seq: int | None = None
    cursor_ord: int | None = None
    cursor_id: int | None = None
    while True:
        result = await repo.read_transcript(
            session_id=session_id,
            user_id=USER_ID,
            limit=500,
            after_turn_seq=cursor_seq,
            after_ordinal=cursor_ord,
            after_id=cursor_id,
        )
        assert isinstance(result, Success)
        page_rows = list(result.unwrap())
        if not page_rows:
            break
        seen.extend(r.ordinal for r in page_rows)
        assert all(r.turn_seq == 1 for r in page_rows)
        if len(page_rows) < 500:
            break
        last = page_rows[-1]
        cursor_seq, cursor_ord, cursor_id = last.turn_seq, last.ordinal, last.row_id
        page_guard += 1
        assert page_guard < 10  # safety bound — 605 rows need exactly 2 pages
    assert len(seen) == 605
    assert len(seen) == len(set(seen))


async def test_turn_recovery_reserves_the_slot(db: _Db) -> None:
    """§6: a recovered query row OPENS its turn — turn_count advances in the
    same transaction, so the next real turn cannot reuse its turn_seq."""
    repo = db.repo
    session_id = str(uuid4())

    # Established turn 1, then a crash-like situation for a NEW turn: the
    # recovery path replays the historical projection carrying its query.
    first = await _apply(
        repo,
        _projection(
            query_id=f"query:{session_id}", rows=(_row(0, f"query:{session_id}", "query"),)
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)

    recovered_key = f"query:{session_id}-2"
    recovery = await repo.insert_rows_only(
        _projection(query_id=recovered_key, rows=(_row(0, recovered_key, "query"),)),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(recovery, Success)
    outcome = recovery.unwrap()
    assert outcome.turn_opened is True
    assert outcome.turn_seq == 2

    # The slot was RESERVED: a later full projection of turn 3 must not
    # collide with the recovered one.
    third_key = f"query:{session_id}-3"
    third = await _apply(
        repo,
        _projection(
            query_id=third_key,
            rows=(_row(0, third_key, "query"),),
            checkpoint_id="cp-third",
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(third, Success)
    third_outcome = third.unwrap()
    assert third_outcome.turn_seq == 3

    async with db.factory() as session:
        counts = (
            await session.execute(
                sa_text("SELECT turn_count FROM chat_sessions WHERE session_id = :sid"),
                {"sid": session_id},
            )
        ).scalar_one()
        assert counts == 3
        seqs = [
            int(t)
            for t in (
                await session.execute(
                    sa_text(
                        "SELECT turn_seq FROM chat_messages WHERE session_id = :sid "
                        "AND source_key LIKE 'query:%' ORDER BY turn_seq"
                    ),
                    {"sid": session_id},
                )
            ).scalars()
        ]
        assert sorted(set(seqs)) == seqs  # no reused turn numbers


async def test_ensure_session_creates_discoverable_identity_without_messages(db: _Db) -> None:
    session_id = str(uuid4())

    ensured = await db.repo.ensure_session(session_id=session_id, user_id=USER_ID)

    assert isinstance(ensured, Success)
    assert ensured.unwrap() is True
    meta = await db.repo.get_session_meta(session_id=session_id, user_id=USER_ID)
    assert isinstance(meta, Success)
    assert meta.unwrap().status == SessionStatus.ACTIVE.value
    assert meta.unwrap().turn_count == 0
    assert meta.unwrap().last_checkpoint_id is None
    rows = await db.repo.read_transcript(session_id=session_id, user_id=USER_ID)
    assert isinstance(rows, Success)
    assert rows.unwrap() == ()


async def test_ensure_session_replay_never_resets_existing_metadata(db: _Db) -> None:
    session_id = str(uuid4())
    first = await db.repo.ensure_session(session_id=session_id, user_id=USER_ID)
    assert isinstance(first, Success)
    async with db.factory() as session, session.begin():
        await session.execute(
            sa_text(
                "UPDATE chat_sessions SET title = 'kept', status = 'completed', "
                "turn_count = 7, last_checkpoint_id = 'cp-kept' WHERE session_id = :sid"
            ),
            {"sid": session_id},
        )

    replay = await db.repo.ensure_session(session_id=session_id, user_id=USER_ID)

    assert isinstance(replay, Success)
    assert replay.unwrap() is False
    meta = await db.repo.get_session_meta(session_id=session_id, user_id=USER_ID)
    assert isinstance(meta, Success)
    saved = meta.unwrap()
    assert (saved.title, saved.status, saved.turn_count, saved.last_checkpoint_id) == (
        "kept",
        SessionStatus.COMPLETED.value,
        7,
        "cp-kept",
    )


async def test_ensure_session_wrong_owner_is_not_found_and_does_not_mutate(db: _Db) -> None:
    session_id = str(uuid4())
    created = await db.repo.ensure_session(session_id=session_id, user_id=USER_ID)
    assert isinstance(created, Success)

    refused = await db.repo.ensure_session(session_id=session_id, user_id="other-user")

    assert isinstance(refused, Failure)
    assert isinstance(refused.failure(), TranscriptNotFoundError)
    original = await db.repo.get_session_meta(session_id=session_id, user_id=USER_ID)
    assert isinstance(original, Success)
    assert original.unwrap().user_id == USER_ID


async def test_ensure_session_unknown_schema_fails_closed_without_rewrite(db: _Db) -> None:
    session_id = str(uuid4())
    async with db.factory() as session, session.begin():
        await session.execute(
            sa_text(
                "INSERT INTO chat_sessions (session_id, user_id, schema_version) "
                "VALUES (:sid, :uid, 99)"
            ),
            {"sid": session_id, "uid": USER_ID},
        )

    refused = await db.repo.ensure_session(session_id=session_id, user_id=USER_ID)

    assert isinstance(refused, Failure)
    error = refused.failure()
    assert isinstance(error, TranscriptSchemaError)
    assert error.operation == "ensure_session"
    async with db.factory() as session:
        found = (
            await session.execute(
                sa_text("SELECT schema_version FROM chat_sessions WHERE session_id = :sid"),
                {"sid": session_id},
            )
        ).scalar_one()
    assert found == 99


async def test_same_checkpoint_replay_fills_late_usage_without_session_mutation(db: _Db) -> None:
    """An empty replay still applies late usage to the existing assistant row."""
    repo = db.repo
    session_id = str(uuid4())
    query_key = f"query:{session_id}"
    assistant_key = "prompt:late-usage"
    first = await _apply(
        repo,
        _projection(
            query_id=query_key,
            checkpoint_id="cp-late-usage",
            rows=(_row(0, query_key, "query"), _row(1, assistant_key, "plan_prompt")),
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)

    late = await _apply(
        repo,
        _projection(query_id=query_key, checkpoint_id="cp-late-usage"),
        user_id=USER_ID,
        session_id=session_id,
        usage={assistant_key: 23},
        usage_metadata={assistant_key: {"usage": {"input": 11, "output": 12}}},
    )
    assert isinstance(late, Success)
    assert late.unwrap().applied is True
    assert late.unwrap().reason == "usage_filled"

    async with db.factory() as session:
        saved_session = (
            await session.execute(
                sa_text(
                    "SELECT status, last_checkpoint_id, turn_count FROM chat_sessions "
                    "WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
        ).one()
        saved_row = (
            await session.execute(
                sa_text(
                    "SELECT tokens_used, extra_metadata FROM chat_messages "
                    "WHERE session_id = :sid AND source_key = :source_key"
                ),
                {"sid": session_id, "source_key": assistant_key},
            )
        ).one()
    assert saved_session.status == SessionStatus.ACTIVE.value
    assert saved_session.last_checkpoint_id == "cp-late-usage"
    assert saved_session.turn_count == 1
    assert saved_row.tokens_used == 23
    assert saved_row.extra_metadata["usage"] == {"input": 11, "output": 12}


async def test_zero_token_metadata_only_replay_is_retained_and_idempotent(db: _Db) -> None:
    """Tool calls are useful usage facts even when the segment has zero tokens."""
    repo = db.repo
    session_id = str(uuid4())
    query_key = f"query:{session_id}"
    assistant_key = "prompt:zero-tool"
    await _apply(
        repo,
        _projection(
            query_id=query_key,
            checkpoint_id="cp-zero-tool",
            rows=(_row(0, query_key, "query"), _row(1, assistant_key, "plan_prompt")),
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    metadata = {
        "usage": {"input": 0, "output": 0},
        "tool_calls": [{"name": "search", "ok": True, "ms": None}],
    }
    first_fill = await _apply(
        repo,
        _projection(query_id=query_key, checkpoint_id="cp-zero-tool"),
        user_id=USER_ID,
        session_id=session_id,
        usage={assistant_key: 0},
        usage_metadata={assistant_key: metadata},
    )
    assert isinstance(first_fill, Success)
    assert first_fill.unwrap().applied is True

    async with db.factory() as session:
        session_before = (
            await session.execute(
                sa_text(
                    "SELECT status, last_checkpoint_id, last_checkpoint_ts, turn_count "
                    "FROM chat_sessions WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
        ).one()

    replay = await _apply(
        repo,
        _projection(query_id=query_key, checkpoint_id="cp-zero-tool"),
        user_id=USER_ID,
        session_id=session_id,
        usage={assistant_key: 99},
        usage_metadata={
            assistant_key: {
                "usage": {"input": 99, "output": 0},
                "tool_calls": [{"name": "different", "ok": False, "ms": 1.0}],
            }
        },
    )
    assert isinstance(replay, Success)

    async with db.factory() as session:
        saved = (
            await session.execute(
                sa_text(
                    "SELECT tokens_used, extra_metadata FROM chat_messages "
                    "WHERE session_id = :sid AND source_key = :source_key"
                ),
                {"sid": session_id, "source_key": assistant_key},
            )
        ).one()
        saved_session = (
            await session.execute(
                sa_text(
                    "SELECT status, last_checkpoint_id, last_checkpoint_ts, turn_count "
                    "FROM chat_sessions WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
        ).one()
    assert saved.tokens_used == 0
    assert saved.extra_metadata == metadata
    assert saved_session == session_before


async def test_same_checkpoint_usage_cannot_fill_non_assistant_row(db: _Db) -> None:
    """Usage side-input targets assistant rows only, even for a shared key."""
    repo = db.repo
    session_id = str(uuid4())
    query_key = f"query:{session_id}"
    first = await _apply(
        repo,
        _projection(
            query_id=query_key,
            checkpoint_id="cp-non-assistant-target",
            rows=(_row(0, query_key, "query"),),
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)

    replay = await _apply(
        repo,
        _projection(query_id=query_key, checkpoint_id="cp-non-assistant-target"),
        user_id=USER_ID,
        session_id=session_id,
        usage={query_key: 42},
        usage_metadata={query_key: {"tool_calls": [{"name": "search", "ok": True}]}},
    )
    assert isinstance(replay, Success)
    assert replay.unwrap().applied is False
    assert replay.unwrap().reason == "same_checkpoint"

    async with db.factory() as session:
        saved = (
            await session.execute(
                sa_text(
                    "SELECT tokens_used, extra_metadata FROM chat_messages "
                    "WHERE session_id = :sid AND source_key = :source_key"
                ),
                {"sid": session_id, "source_key": query_key},
            )
        ).one()
    assert saved.tokens_used is None
    assert saved.extra_metadata is None
