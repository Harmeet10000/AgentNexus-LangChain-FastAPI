"""Transcript sweep + checkpoint GC (doc §6 trigger 3, §8 GC).

Reconciliation triggers list a **beat sweep** over non-terminal sessions as
the backstop for lost reconcile hints and missed boundaries: bounded batches
of stale sessions are re-projected so an orphaned run reaches its derived
fate without a WS connection. The sweep never invents facts — it runs the
same pure projector over fresh ``aget_state`` reads, with:

- **aliveness = False unless the thread-lock heartbeat key is fresh** (§4.1).
  A stale-but-live turn derives nothing (the §5.3 NoOp row keeps the sweep
  safe on live equipment); only *conclusively dead longer than grace* closes.
- **[v3] Redis-outage guard**: when Redis is unavailable the sweep cannot
  judge aliveness, so it derives NOTHING — no verdicts, no closes, statuses
  freeze and unfreeze on the next beat after Redis returns.

Checkpoint GC (§8, daily beat ``chat_transcript_gc``) deletes a thread only
after a verified transcript catch-up reconcile, then stamps the audit
(``extra_metadata.gc_swept_at`` + the deleted checkpoint id) — the only
proof that the catch-up verification happened.

Checkpoint reads go through public LangGraph APIs only (``aget_state``,
``adelete_thread``) on a state-compatible read graph — never raw
checkpointer tables. The read graph declares the same ``LegalAgentState``
schema as the production graph, so persisted channel names resolve; a
single self-ending stub node makes it compilable because state reads
require no graph execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from redis.exceptions import RedisError
from returns.result import Failure

from app.features.agent_saul.graph.graph import wire_saul_topology
from app.features.agent_saul.thread_lock import ThreadLock
from app.features.agent_saul.transcript.projection import (
    ORPHAN_GRACE,
    SessionStatus,
    project,
    snapshot_to_view,
)
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.graph.state import CompiledStateGraph
    from redis.asyncio import Redis

    from app.features.agent_saul.transcript.repository import ChatTranscriptRepository

SAUL_RECONCILE_HINT_KEY = "saul:reconcile"
TRANSCRIPT_RETENTION = timedelta(days=7)  # §8 (D5): proposed retention period
DEFAULT_SWEEP_BATCH = 20
_HINT_BOUND = 16

_GC_STAMP_KEY = "gc_swept_at"
_LIVE_LOCK_KEY_PREFIX = "saul:thread_lock:"
_TERMINAL_STATUS_VALUES: frozenset[str] = frozenset(
    (
        SessionStatus.COMPLETED.value,
        SessionStatus.FAILED.value,
        SessionStatus.ABANDONED_EXPIRED.value,
    )
)


def build_saul_read_graph(checkpointer: AsyncPostgresSaver) -> CompiledStateGraph[Any]:
    """Compile a state-compatible read graph for sweep-only checkpoint reads.

    Same schema ⇒ same channels; the SAME edge topology as the production
    graph (``wire_saul_topology``) over self-ending stub nodes. The topology
    matters even for reads: ``Pregel.aget_state`` resolves pending tasks by
    node NAME against the graph's process map, so a node that is absent here
    makes its pending task (a production interrupt incl.) silently vanish
    and the sweep mis-derive the thread's status. The stubs are never
    executed by the sweep — the graph exists to interpret checkpoints.
    """
    from langgraph.graph import StateGraph  # noqa: PLC0415 — cold path

    from app.features.agent_saul.graph.state import (  # noqa: PLC0415 — cold path
        LegalAgentInputState,
        LegalAgentOutputState,
        LegalAgentState,
    )

    def _stub_factory() -> Callable[[str], Any]:
        stubs: dict[str, Any] = {}

        def node_for(name: str) -> Any:
            if name not in stubs:

                async def stub(_state: dict[str, Any], _name: str = name) -> None:
                    """Self-ending stub: interprets nothing, writes nothing."""

                stub.__name__ = name
                stubs[name] = stub
            return stubs[name]

        return node_for

    # Same cast idiom as graph.build_saul_graph: langgraph's generics reject
    # the plain TypedDict at type-check time though it compiles fine.
    state_graph_factory = cast("Any", StateGraph)
    state_graph: Any = state_graph_factory(
        LegalAgentState,
        input_schema=LegalAgentInputState,
        output_schema=LegalAgentOutputState,
    )
    wire_saul_topology(state_graph, _stub_factory())
    return state_graph.compile(checkpointer=checkpointer)


@dataclass(frozen=True)
class SweepReport:
    """Machine-parseable outcome of one sweep invocation (structured logs, §10)."""

    skipped: str | None
    hints_reconciled: int = 0
    stale_scanned: int = 0
    closed: int = 0
    gc_deleted: int = 0
    failures: int = 0


async def sweep_transcripts(
    *,
    repo: ChatTranscriptRepository,
    graph: CompiledStateGraph[Any],
    redis: Redis,
    checkpointer: AsyncPostgresSaver,
    now: datetime | None = None,
    batch_size: int = DEFAULT_SWEEP_BATCH,
) -> SweepReport:
    """One bounded sweep pass: consume hints, heal stale sessions, then GC.

    The status cache only moves THROUGH a projection (§3.1 [v2]) — every
    diagonal of this sweep is the projector, never a direct UPDATE.
    """
    current = now or datetime.now(tz=UTC)
    # [v3] Redis-outage guard: no aliveness ⇒ no verdicts, no closes.
    try:
        await redis.ping()
    except RedisError:
        logger.warning("transcript_sweep_redis_unavailable_skip_derivation")
        return SweepReport(skipped="redis_unavailable")

    hints_reconciled = await _reconcile_hinted_threads(
        repo=repo, graph=graph, redis=redis, now=current, bound=_HINT_BOUND
    )
    stale_scanned, closed = await _reconcile_stale_sessions(
        repo=repo, graph=graph, redis=redis, now=current, batch_size=batch_size
    )
    gc_deleted = await _gc_checkpointed_threads(
        repo=repo,
        graph=graph,
        checkpointer=checkpointer,
        redis=redis,
        now=current,
        older_than=current - TRANSCRIPT_RETENTION,
        batch_size=batch_size,
    )
    logger.bind(
        skipped=None,
        hints_reconciled=hints_reconciled,
        stale_scanned=stale_scanned,
        closed=closed,
        gc_deleted=gc_deleted,
    ).info("transcript_sweep_finished")
    return SweepReport(
        skipped=None,
        hints_reconciled=hints_reconciled,
        stale_scanned=stale_scanned,
        closed=closed,
        gc_deleted=gc_deleted,
    )


@dataclass(frozen=True)
class _ThreadReconcileOutcome:
    """What one pass of the projector did to a thread (never a euphemism).

    ``status``      — the freshly derived status value when a projection with
                      a derivable status ran to completion (§3.1), whether or
                      not it landed new rows (an already-persisted replay is
                      still a fresh derivation of the CURRENT checkpoint).
    ``clean``       — the projector derived nothing (NoOp). NOT proof of
                      verified catch-up: a NoOp may equally mean the
                      projector saw nothing derivable (pre-first-step,
                      missing opener, stale heartbeat) — a checkpoint under
                      such a verdict stays put.
    ``failed``      — any repository/graph/aliveness error; nothing may be
                      concluded.
    """

    status: str | None = None
    checkpoint_id: str | None = None
    clean: bool = False
    failed: bool = True


async def _lock_alive(redis: Redis, thread_id: str) -> bool | None:
    """Aliveness input for sweep projections: heartbeat fresh, dead, or UNKNOWN.

    ``None`` means the probe itself failed (§6 [v3]): aliveness is unknowable,
    and the projector must derive NOTHING from a verdict it cannot see.
    """
    try:
        exists = await redis.exists(f"{_LIVE_LOCK_KEY_PREFIX}{thread_id}")
    except RedisError:
        return None
    return bool(exists)


async def _reconcile_once(
    *,
    repo: ChatTranscriptRepository,
    graph: CompiledStateGraph[Any],
    redis: Redis,
    session_id: str,
    now: datetime,
    aliveness: bool | None = None,
) -> _ThreadReconcileOutcome:
    """Reconcile one thread through the projector; the outcome is explicit.

    The sweep context is the projector's only extra privilege: the CALLER's
    turn context arrives owner-checked inside the repository, and the write
    itself re-runs every §9 gate (unknown kinds, fail-closed versions).
    ``aliveness`` is normally probed against the live lock key; a caller
    holding the thread lock passes ``False`` outright — inside its exclusive
    section the thread is un-owned by FACT, not by probe (§8 GC).
    """
    context_result = await repo.get_sweep_context(session_id=session_id)
    if isinstance(context_result, Failure):
        return _ThreadReconcileOutcome()
    owner_user_id, turn_context = context_result.unwrap()
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    if aliveness is None:
        aliveness = await _lock_alive(redis, session_id)
        if aliveness is None:
            # §6 [v3]: aliveness unknowable → no verdict, no rows, no close.
            return _ThreadReconcileOutcome()
    snapshot = await graph.aget_state(config)  # public LangGraph API (§1)
    view = snapshot_to_view(snapshot)
    derived = project(
        view,
        turn_context.existing_keys,
        turn_context.turn_rows,
        now,
        aliveness=aliveness,
    )
    if derived.is_no_op:
        # Explicit "derive nothing" (§5.3 NoOp row). Neither clean nor a
        # verdict — the reason field says which observation gap it was.
        return _ThreadReconcileOutcome(clean=True, failed=False)
    if derived.session_status is None:
        return _ThreadReconcileOutcome(failed=True)
    applied = await repo.apply_projection(
        derived,
        user_id=owner_user_id,
        session_id=session_id,
    )
    if isinstance(applied, Failure):
        return _ThreadReconcileOutcome()
    return _ThreadReconcileOutcome(
        status=derived.session_status.value,
        checkpoint_id=derived.watermark_checkpoint_id,
        clean=False,
        failed=False,
    )


async def _reconcile_hinted_threads(
    *,
    repo: ChatTranscriptRepository,
    graph: CompiledStateGraph[Any],
    redis: Redis,
    now: datetime,
    bound: int,
) -> int:
    """Consume up to ``bound`` best-effort reconcile hints (§6 trigger 4)."""
    try:
        popped = await redis.spop(SAUL_RECONCILE_HINT_KEY, bound)
    except RedisError:
        logger.warning("transcript_sweep_hint_consume_unavailable")
        return 0
    if not popped or not isinstance(popped, (list, tuple, set)):
        return 0
    hint_ids = [str(item) for item in popped]
    for thread_id in hint_ids:
        try:
            outcome = await _reconcile_once(
                repo=repo, graph=graph, redis=redis, session_id=thread_id, now=now
            )
            logger.bind(thread_id=thread_id, status=outcome.status, clean=outcome.clean).debug(
                "transcript_sweep_hint_reconciled"
            )
        except Exception as exc:  # noqa: BLE001 — one hinted thread never stops the sweep
            exc.add_note(f"operation=transcript_sweep, thread_id={thread_id}")
            logger.bind(thread_id=thread_id, error=str(exc)).warning(
                "transcript_sweep_hint_reconcile_failed"
            )
    return len(hint_ids)


async def _reconcile_stale_sessions(
    *,
    repo: ChatTranscriptRepository,
    graph: CompiledStateGraph[Any],
    redis: Redis,
    now: datetime,
    batch_size: int,
) -> tuple[int, int]:
    """Trigger 3 (§6): bounded batch of non-terminal sessions past orphan grace."""
    stale = await repo.list_stale_sessions(
        older_than=now - ORPHAN_GRACE,
        terminal_statuses=_TERMINAL_STATUS_VALUES,
        limit=batch_size,
    )
    if isinstance(stale, Failure):
        return 0, 0
    scanned = closed = 0
    for session_meta in stale.unwrap():
        scanned += 1
        try:
            outcome = await _reconcile_once(
                repo=repo,
                graph=graph,
                redis=redis,
                session_id=session_meta.session_id,
                now=now,
            )
        except Exception as exc:  # noqa: BLE001 — one session failing never stops the sweep
            exc.add_note(f"operation=transcript_sweep, session_id={session_meta.session_id}")
            logger.bind(session_id=session_meta.session_id, error=str(exc)).warning(
                "transcript_sweep_reconcile_failed"
            )
            continue
        if outcome.status in _TERMINAL_STATUS_VALUES:
            closed += 1
    return scanned, closed


async def _gc_checkpointed_threads(
    *,
    repo: ChatTranscriptRepository,
    graph: CompiledStateGraph[Any],
    checkpointer: AsyncPostgresSaver,
    redis: Redis,
    now: datetime,
    older_than: datetime,
    batch_size: int,
) -> int:
    """§8 checkpoint GC: terminal sessions past retention.

    GC always follows a verified catch-up reconcile of the thread (doc §1/§8);
    the audit stamp lands on the session row before the checkpoint disappears.
    A session whose catch-up reconcile fails keeps its checkpoint (lossy GC is
    never chosen over losing the last write, §8 "the only proof...").
    """
    candidates = await repo.list_gc_candidates(older_than=older_than, limit=batch_size)
    if isinstance(candidates, Failure):
        return 0
    deleted = 0
    for session_meta in candidates.unwrap():
        session_id = session_meta.session_id
        # §5.5/§8: the catch-up and the deletion run UNDER the same thread
        # exclusion a connection uses. While it is held, a reconnect can
        # neither start a new turn on the thread nor lose checkpoints it is
        # writing; conversely, holding it makes "no live connection" a FACT
        # inside the section, so the verdict runs with aliveness=False
        # instead of probing a key this very section holds.
        lock = ThreadLock(redis, session_id)
        acquired = await lock.acquire()
        if not acquired:
            continue  # owned by a live connection — never GC under it (§5.5)
        try:
            outcome = await _reconcile_once(
                repo=repo,
                graph=graph,
                redis=redis,
                session_id=session_id,
                now=now,
                aliveness=False,
            )
            # §8 verified catch-up: GC acts ONLY when the catch-up just
            # derived a FRESH terminal verdict from the CURRENT checkpoint.
            # A no-op (outcome.clean) is not proof — it may equally be an
            # observation gap — and a failure (outcome.failed) releases
            # nothing.
            if not (not outcome.failed and outcome.status in _TERMINAL_STATUS_VALUES):
                continue  # catch-up did not complete — keep the checkpoint (§8)
            # Generation + retention recheck UNDER the lock (§8): the
            # checkpoint the verdict came from must STILL be the session
            # row's latest, and past the retention deadline as read now.
            peek = await repo.peek_session_checkpoint(session_id=session_id)
            if isinstance(peek, Failure):
                continue
            current_id, current_ts = peek.unwrap()
            if current_id != outcome.checkpoint_id:
                logger.bind(
                    session_id=session_id,
                    verdict_checkpoint=outcome.checkpoint_id,
                    current_checkpoint=current_id,
                ).debug("transcript_gc_generation_moved_skip")
                continue
            if current_ts is None or current_ts > older_than:
                continue  # a new turn (+its checkpoint) arrived — survive (§8)
            # The audit stamp lands BEFORE the checkpoint disappears (§8) so
            # a crash mid-GC cannot strand a candidate; a failed delete
            # clears the stamp so the next sweep may retry.
            stamped = await repo.stamp_gc_audit(
                session_id=session_id,
                deleted_checkpoint_id=session_meta.last_checkpoint_id,
                swept_at=now,
            )
            if isinstance(stamped, Failure):
                logger.bind(session_id=session_id, error=str(stamped.failure())).warning(
                    "transcript_gc_stamp_failed"
                )
                continue
            await checkpointer.adelete_thread(session_id)  # public API only (§1)
            deleted += 1
        except Exception as exc:  # noqa: BLE001 — a GC refusal never stops the sweep
            exc.add_note(f"operation=transcript_gc, session_id={session_meta.session_id}")
            logger.bind(session_id=session_meta.session_id, error=str(exc)).warning(
                "transcript_gc_refused"
            )
            unstamp = await repo.clear_gc_audit(session_id=session_id)
            if isinstance(unstamp, Failure):
                logger.bind(session_id=session_id).warning("transcript_gc_unstamp_failed")
        finally:
            await lock.release()
    return deleted
