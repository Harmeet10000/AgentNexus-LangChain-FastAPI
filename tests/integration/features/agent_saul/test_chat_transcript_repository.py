"""ChatTranscriptRepository behaviour on a live Postgres (doc §13 DoD).

Run explicitly: ``uv run pytest tests/integration/features/agent_saul -m integration``
Requires migrations through 0022 applied (chat tables at the projected shape).
The chain-of-custody properties pinned here: batch no-op replay, fill-once
tokens, turn_seq stability across pause/resume, write-failure healing, the
schema gate, and Graphiti/GC independence of the transcript.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from returns.result import Failure, Success
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.connections.postgres import get_database_url
from app.features.agent_saul.transcript.errors import (
    TranscriptInfrastructureError,
    TranscriptNotFoundError,
    TranscriptSchemaError,
)
from app.features.agent_saul.transcript.projection import (
    ContentRole,
    ProjectedRow,
    Projection,
    SessionStatus,
    project,
    snapshot_to_view,
)
from app.features.agent_saul.transcript.repository import ChatTranscriptRepository

if TYPE_CHECKING:
    from typing import Any

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

USER_ID = "integration-user"


async def _session_factory() -> async_sessionmaker[Any]:
    engine = create_async_engine(get_database_url(flavour="async"), pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False)


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


async def test_replay_is_batch_noop_and_item_deduped() -> None:
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())
    rows = (
        _row(0, f"query:{session_id}", "query"),
        _row(1, "interrupt:int-1", "clarification_prompt"),
    )
    first = await repo.apply_projection(
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-1"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)
    outcome = first.unwrap()
    assert outcome.applied is True

    # Same checkpoint, no rows → batch no-op.
    replay = await repo.apply_projection(
        _projection(query_id=f"query:{session_id}", checkpoint_id="cp-1"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(replay, Success)
    replayed = replay.unwrap()
    assert replayed.applied is False

    # Same checkpoint, WITH rows (crash-heal form): no duplicate rows land.
    heal = await repo.apply_projection(
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-1"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(heal, Success)
    assert heal.unwrap().rows_inserted == 0
    async with factory() as session:
        count = (
            await session.execute(
                sa_text("SELECT count(*) FROM chat_messages WHERE session_id = :sid"),
                {"sid": session_id},
            )
        ).scalar_one()
    assert int(count) == 2


async def test_tokens_fill_once_under_replay() -> None:
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())
    rows = (_row(0, f"query:{session_id}", "query"),)

    await repo.apply_projection(
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-a"),
        user_id=USER_ID,
        session_id=session_id,
        usage={f"query:{session_id}": 420},
    )
    # A higher usage on replay must NOT overwrite (fill-once COALESCE).
    replay = await repo.apply_projection(
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-a"),
        user_id=USER_ID,
        session_id=session_id,
        usage={f"query:{session_id}": 999},
    )
    assert isinstance(replay, Success)  # batch no-op keeps tokens intact
    async with factory() as session:
        tokens = (
            await session.execute(
                sa_text("SELECT tokens_used FROM chat_messages WHERE session_id = :sid"),
                {"sid": session_id},
            )
        ).scalar_one()
    assert int(tokens) == 420


async def test_turn_seq_stable_across_pause_and_resume() -> None:
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())
    q_key = f"query:{session_id}"

    # First boundary: query lands inside a fresh turn.
    r1 = await repo.apply_projection(
        _projection(query_id=q_key, rows=(_row(0, q_key, "query"),), checkpoint_id="cp-t1"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(r1, Success)
    assert r1.unwrap().turn_seq == 1
    assert r1.unwrap().turn_opened is True

    # Second boundary (same turn, post-resume): turn_seq MUST stay 1.
    r2 = await repo.apply_projection(
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
    async with factory() as session:
        session_row = (
            await session.execute(
                sa_text("SELECT turn_count, status FROM chat_sessions WHERE session_id = :sid"),
                {"sid": session_id},
            )
        ).one()
    assert int(session_row[0]) == 1
    assert session_row[1] == "clarifying"


async def test_write_failure_returns_typed_failure_not_raised() -> None:
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    # A NUL byte in any text parameter is rejected by Postgres at bind time —
    # a deterministic driver failure to prove the edge sanitizes, not raises.
    failure = await repo.apply_projection(
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


async def test_schema_gate_fails_closed_on_unknown_version() -> None:
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())
    await repo.apply_projection(
        _projection(
            query_id=f"query:{session_id}", rows=(_row(0, f"query:{session_id}", "query"),)
        ),
        user_id=USER_ID,
        session_id=session_id,
    )
    async with factory() as session:
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


async def test_turn_closed_flow_over_24h_pause_window() -> None:
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())
    # The D4 chain-of-custody key IS the HumanMessage id; the projector keys
    # the query row by it. Use the raw id everywhere so the seeded turn and
    # the projected turn resolve to the same turn_seq.
    q_key = session_id

    # Seed the paused turn: query + clarification prompt recorded at T1.
    await repo.apply_projection(
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
    applied = await repo.apply_projection(derived, user_id=USER_ID, session_id=session_id)
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
            "schema_version": 1,
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


async def test_checkpointer_gc_cannot_touch_the_transcript() -> None:
    """Checkpointer lifecycle is namespace-orthogonal to chat rows (D-coverage)."""
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())
    q_key = session_id
    persisted = await repo.apply_projection(
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


async def test_owner_mismatch_refused_before_writes() -> None:
    """§9: a wrong-owner projection is a sanitized not-found, no rows, no move."""
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())
    rows = (
        _row(0, f"query:{session_id}", "query"),
        _row(1, "interrupt:int-owner", "clarification_prompt"),
    )

    first = await repo.apply_projection(
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-owner"),
        user_id=USER_ID,
        session_id=session_id,
    )
    assert isinstance(first, Success)

    attack = await repo.apply_projection(
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
    async with factory() as session:
        stranger_rows = await session.execute(
            sa_text("SELECT count(*) FROM chat_messages WHERE user_id = 'stranger-user'"),
        )
        assert int(stranger_rows.scalar_one()) == 0
        cp = await session.execute(
            sa_text("SELECT last_checkpoint_id FROM chat_sessions WHERE session_id = :sid"),
            {"sid": session_id},
        )
        assert (cp.scalar_one()) == "cp-owner"


async def test_usage_only_fill_lands_on_existing_row() -> None:
    """§7: segment totals land once even when the projector deduped the row."""
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())
    rows = (
        _row(0, f"query:{session_id}", "query"),
        _row(1, "interrupt:fill-target", "clarification_prompt"),
    )

    await repo.apply_projection(
        _projection(query_id=f"query:{session_id}", rows=rows, checkpoint_id="cp-fill-1"),
        user_id=USER_ID,
        session_id=session_id,
    )

    # Follow-up boundary (new checkpoint) whose target row already exists: the
    # projector drops it, so the usage rides as a usage-only fill.
    fill = await repo.apply_projection(
        _projection(
            query_id=f"query:{session_id}",
            rows=(_row(2, "interrupt:fill-other", "clarification_prompt"),),
            checkpoint_id="cp-fill-2",
        ),
        user_id=USER_ID,
        session_id=session_id,
        usage={"interrupt:fill-target": 77, "interrupt:never-inserted": 5},
    )
    assert isinstance(fill, Success)
    async with factory() as session:
        tokens = (
            await session.execute(
                sa_text(
                    "SELECT tokens_used FROM chat_messages "
                    "WHERE session_id = :sid AND source_key = 'interrupt:fill-target'"
                ),
                {"sid": session_id},
            )
        ).scalar_one()
        assert tokens == 77


async def test_read_transcript_paging_completes_large_transcripts() -> None:
    """§8 export: offset paging returns every row in display order (D3)."""
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())
    big = tuple(
        _row(
            i, f"query:{session_id}" if i == 0 else f"row-{i}", "query" if i == 0 else "plan_prompt"
        )
        for i in range(605)
    )
    first = await repo.apply_projection(
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

    seen: list[str] = []
    page = 0
    while True:
        result = await repo.read_transcript(
            session_id=session_id,
            user_id=USER_ID,
            limit=500,
            offset=page * 500,
        )
        assert isinstance(result, Success)
        page_rows = list(result.unwrap())
        if not page_rows:
            break
        seen.extend(r.ordinal for r in page_rows)
        assert all(r.turn_seq == 1 for r in page_rows)
        page += 1
        assert page < 10  # safety bound — 605 rows need exactly 2 pages
    assert len(seen) == 605
    assert len(seen) == len(set(seen))


async def test_turn_recovery_reserves_the_slot() -> None:
    """§6: a recovered query row OPENS its turn — turn_count advances in the
    same transaction, so the next real turn cannot reuse its turn_seq."""
    factory = await _session_factory()
    repo = ChatTranscriptRepository(factory)
    session_id = str(uuid4())

    # Established turn 1, then a crash-like situation for a NEW turn: the
    # recovery path replays the historical projection carrying its query.
    first = await repo.apply_projection(
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
    third = await repo.apply_projection(
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

    async with factory() as session:
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
