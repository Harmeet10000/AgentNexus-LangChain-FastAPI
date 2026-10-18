"""TurnRecorder — per-turn projection driver (doc §10).

One recorder per connection. Each boundary method (on_first_checkpoint /
on_pause / on_end / on_abort / reconcile / prepare_resume) does exactly one
thing: fetch a fresh ``aget_state`` → ``snapshot_to_view`` → ``project`` →
``repo.apply_projection`` under a hard ``asyncio.timeout`` — the WS loop is
never waited on (§5.2). An exception is never evidence: no row exists because
an exception occurred (§1); T3 projects a fresh snapshot and nothing else.

Usage is a live side-input (§7): a segment-scoped collector aggregates
``usage_metadata`` from every LLM call and attaches per-segment totals to that
segment's last assistant row through the fill-once upsert. Loss of usage is
fail-open (under-bills only, never double-bills).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, override

from langchain_core.callbacks import BaseCallbackHandler
from returns.result import Failure

from app.features.agent_saul.transcript.errors import (
    TranscriptInfrastructureError,
)
from app.features.agent_saul.transcript.projection import (
    VERDICT_ROW_KINDS,
    TurnRows,
    project,
    snapshot_to_view,
)
from app.observability.logging import logger

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import StateSnapshot
    from redis.asyncio import Redis

    from app.features.agent_saul.thread_lock import ThreadLock
    from app.features.agent_saul.transcript.errors import (
        TranscriptResult,
    )
    from app.features.agent_saul.transcript.projection import (
        SnapshotView,
    )
    from app.features.agent_saul.transcript.repository import (
        ApplyOutcome,
        ChatTranscriptRepository,
        TranscriptTurnContext,
    )
DEFAULT_PROJECTION_TIMEOUT_MS = 750  # §5.2 hard timeout, configurable
_RECONCILE_HINT_KEY = "saul:reconcile"


@dataclass
class _SegmentUsage:
    """Tokens + tool calls gathered for ONE segment (start→pause / resume→X)."""

    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[dict[str, object]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_metadata(self) -> dict[str, object]:
        # Content-free facts only (§3.4): numbers and call names, no arguments.
        return {
            "usage": {"input": self.input_tokens, "output": self.output_tokens},
            "tool_calls": list(self.tool_calls),
        }


class SegmentUsageCollector(BaseCallbackHandler):
    """Aggregates usage from EVERY LLM call — never the model's final message (§7)."""

    def __init__(self) -> None:
        self.usage = _SegmentUsage()

    def reset(self) -> None:
        self.usage = _SegmentUsage()

    @override
    def on_llm_end(self, response: object, **kwargs: object) -> None:
        generations = getattr(response, "generations", None) or []
        for generation in generations:
            for gen in generation:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if isinstance(usage, dict):
                    self.usage.input_tokens += int(usage.get("input_tokens") or 0)
                    self.usage.output_tokens += int(usage.get("output_tokens") or 0)

    @override
    def on_tool_end(self, output: object, **kwargs: object) -> None:
        started_at = kwargs.get("started_at")
        name = kwargs.get("name")
        self.usage.tool_calls.append(
            {
                "name": str(name or "unknown"),
                "ok": True,
                "ms": (
                    round((perf_counter() - started_at) * 1000, 1)
                    if isinstance(started_at, float)
                    else None
                ),
            }
        )

    @override
    def on_tool_error(self, error: object, **kwargs: object) -> None:
        started_at = kwargs.get("started_at")
        name = kwargs.get("name")
        self.usage.tool_calls.append(
            {
                "name": str(name or "unknown"),
                "ok": False,
                "ms": (
                    round((perf_counter() - started_at) * 1000, 1)  # ty: checked below
                    if isinstance(started_at, float)
                    else None
                ),
            }
        )


class TurnRecorder:
    """Runs the projector at the four named boundaries; the loop ignores Results
    except the project-before-resume retry (§10). Results, never exceptions."""

    def __init__(
        self,
        *,
        repo: ChatTranscriptRepository,
        graph: CompiledStateGraph[Any],
        config: RunnableConfig,
        thread_id: str,
        user_id: str,
        redis: Redis,
        lock: ThreadLock,
        timeout_ms: float = DEFAULT_PROJECTION_TIMEOUT_MS,
    ) -> None:
        self._repo = repo
        self._graph = graph
        self._config = config
        self._thread_id = thread_id
        self._user_id = user_id
        self._redis = redis
        self._lock = lock
        self._timeout = timeout_ms / 1000
        self._usage = SegmentUsageCollector()
        self._t0_task: asyncio.Task[TranscriptResult[ApplyOutcome]] | None = None
        self._paused_plan: tuple[dict[str, object], ...] = ()
        self.log = logger.bind(session_id=thread_id, component="turn_recorder")

    @property
    def usage_handler(self) -> SegmentUsageCollector:
        """Attach into the run config: ``config["callbacks"] = [recorder.usage_handler]``."""
        return self._usage

    # ------------------------------------------------------------------
    # Boundaries (§5.1)
    # ------------------------------------------------------------------

    def on_first_checkpoint(self) -> None:
        """T0 — tracked background task; awaited at the next boundary and T3 (§5.2)."""
        if self._t0_task is None or self._t0_task.done():
            self._t0_task = asyncio.create_task(
                self._project(stage="t0_first_checkpoint"),
                name=f"saul-recorder-t0-{self._thread_id}",
            )

    async def on_pause(self) -> TranscriptResult[ApplyOutcome]:
        """T1 — paused on an interrupt; capture the plan the user was shown."""
        await self._await_t0()
        # Project FIRST, then read the plan the user was shown: the pause this
        # projection emits may also carry the previous turn's plan decision,
        # which needs the PRE-resume plan preserved — the new interrupt's
        # (usually missing) plan must not overwrite it before it is derived.
        result = await self._project(stage="t1_pause", attach_usage=True)
        await self._capture_paused_plan()
        self._usage.reset()
        return result

    async def on_end(self) -> TranscriptResult[ApplyOutcome]:
        """T2 — run reached END; segment totals attach to the final row."""
        await self._await_t0()
        result = await self._project(stage="t2_end", attach_usage=True)
        self._usage.reset()
        return result

    async def on_abort(self) -> TranscriptResult[ApplyOutcome]:
        """T3 — WS drop or exception: await T0, then project fresh, nothing else."""
        await self._await_t0()
        result = await self._project(stage="t3_abort")
        self._usage.reset()
        return result

    async def reconcile(self) -> TranscriptResult[ApplyOutcome]:
        """§6: reconcile = the projector over a fresh ``aget_state`` snapshot,
        preceded by the §6 hole-fill (missing prompt row ⇒ recover from
        ``aget_state_history``). The recovery replay rides the same hard
        deadline as a projection (§5.2) — recovery must never wedge a turn."""
        await self._await_t0()
        try:
            await asyncio.wait_for(self._fill_prompt_holes(), self._timeout)
        except TimeoutError:
            self.log.warning("transcript_hole_fill_timeout")
        return await self._project(stage="reconcile")

    async def _fill_prompt_holes(
        self,
        *,
        max_states: int = 24,
    ) -> None:
        """§6 recovery: a missing prompt row is filled from checkpoint history.

        A crash between the client-visible prompt and its T1 write leaves a
        hole: the LATEST snapshot no longer contains the interrupt at all, so
        a flat ``reconcile`` cannot see the prompt — and a later plan decision
        row would lose its key entirely. Replay the bounded checkpoint
        history oldest-first; the dedupe keys make the replay idempotent, and
        only keyed row FACTS recover (§9): verdict rows (turn_closed /
        final_summary), the watermark and the cached status all stay the
        primary latest projection's job, so an old pause can never plant a
        false terminal outcome under today's clock.
        """
        now = datetime.now(tz=UTC)
        try:
            history = [
                state
                async for state in self._graph.aget_state_history(self._config, limit=max_states)
            ]
        except Exception:  # noqa: BLE001 — history is a recovery path, never a gate (§10)
            self.log.warning("transcript_history_unavailable")
            return
        # One turn context for the whole replay: keys refresh ONLY after rows
        # actually land, so each state dedupes against everything recovered
        # before it (§6).
        turn_context = await self._turn_context()
        if turn_context is None:
            return
        paused_plan = self._paused_plan  # the plan this connection knows about
        for state in reversed(history):  # oldest → newest; watermark stays latest-first
            try:
                view = snapshot_to_view(state)
                projection = project(
                    view,
                    existing_keys=turn_context.existing_keys,
                    turn_rows=turn_context.turn_rows.model_copy(
                        update={"paused_plan": paused_plan}
                    ),
                    # Historical time: a historical boundary is judged in its
                    # own moment — today's clock would plant verdicts (§6).
                    now=view.checkpoint_ts or now,
                    aliveness=self._lock.is_alive(),
                )
            except Exception as exc:  # noqa: BLE001 — §9 boundary guard
                self.log.bind(error=str(exc)).debug("transcript_history_state_skipped")
                continue
            if projection.is_no_op or not projection.rows:
                continue
            if not any(row.kind not in VERDICT_ROW_KINDS for row in projection.rows):
                continue
            applied = await self._repo.insert_rows_only(
                projection, user_id=self._user_id, session_id=self._thread_id
            )
            if isinstance(applied, Failure):
                self.log.bind(operation="insert_rows_only").warning("transcript_hole_fill_failed")
            elif applied.unwrap().rows_inserted > 0:
                turn_context = await self._turn_context() or turn_context
            plan_seen = _paused_plan_of(state)
            if plan_seen is not None:
                paused_plan = plan_seen  # later decisions read the shown plan

    async def _turn_context(self) -> TranscriptTurnContext | None:
        """Fresh turn context, or None when the repository cannot serve it."""
        context_result = await self._repo.get_turn_context(
            session_id=self._thread_id, user_id=self._user_id
        )
        if isinstance(context_result, Failure):
            return None
        return context_result.unwrap()

    async def _existing_keys(self) -> frozenset[str]:
        context_result = await self._repo.get_turn_context(
            session_id=self._thread_id, user_id=self._user_id
        )
        if isinstance(context_result, Failure):
            return frozenset()
        return context_result.unwrap().existing_keys

    async def _turn_rows(self) -> TurnRows:
        context_result = await self._repo.get_turn_context(
            session_id=self._thread_id, user_id=self._user_id
        )
        if isinstance(context_result, Failure):
            return TurnRows()
        return context_result.unwrap().turn_rows

    async def prepare_resume(self) -> TranscriptResult[ApplyOutcome]:
        """Project-before-resume (§5.2): retry once on failure; then proceed
        fail-open and register the reconcile hint."""
        await self._await_t0()
        first = await self._project(stage="t1p_before_resume")
        failure = _as_infra_failure(first)
        if failure is None:
            return first
        retry = await self._project(stage="t1p_before_resume_retry")
        if _as_infra_failure(retry) is not None:
            await self._register_reconcile_hint()
        return retry

    # ------------------------------------------------------------------
    # One projection cycle — shared by every boundary
    # ------------------------------------------------------------------

    async def _await_t0(self) -> None:
        if self._t0_task is not None and not self._t0_task.done():
            self.log.debug("t0_awaited_at_later_boundary")
            await asyncio.wait({self._t0_task})

    async def _capture_paused_plan(self) -> None:
        """Snapshot the plan shown at the CURRENT interrupt, after projecting.

        Guarded: a failing checkpoint read must never escape the recorder and
        turn a delivered prompt into an INTERNAL_ERROR disconnect (§10). The
        previous plan stays until this capture succeeds, so a decision row
        still reads it correctly.
        """
        try:
            snapshot = await self._system_snapshot() if self._graph else None
        except Exception:  # noqa: BLE001 — §9 boundary guard
            self.log.warning("paused_plan_capture_failed")
            return
        self._paused_plan = _paused_plan_of(snapshot)

    async def _project(
        self,
        *,
        stage: str,
        attach_usage: bool = False,
    ) -> TranscriptResult[ApplyOutcome]:
        """Hard-bounded boundary (§5.2): turn context, checkpoint read, and the
        apply ALL run inside one deadline — the WS loop never waits longer."""
        try:
            return await asyncio.wait_for(
                self._project_core(stage=stage, attach_usage=attach_usage), self._timeout
            )
        except TimeoutError:
            self.log.bind(stage=stage).warning("transcript_projection_timeout")
            await self._register_reconcile_hint()
            return _infra_failure_for(stage, "transcript projection timed out")

    async def _project_core(
        self,
        *,
        stage: str,
        attach_usage: bool = False,
    ) -> TranscriptResult[ApplyOutcome]:
        # 1. Turn context — the only snapshot-independent knowledge door (§4.1).
        context_result = await self._repo.get_turn_context(
            session_id=self._thread_id, user_id=self._user_id
        )
        turn_context = _as_infra_failure(context_result)
        if turn_context is not None:
            await self._register_reconcile_hint()
            return Failure(turn_context)
        try:
            context: TranscriptTurnContext = context_result.unwrap()
        except Exception:  # noqa: BLE001 — defensive: unwrap only after isinstance
            return _infra_failure_for(stage, "turn context unreachable")

        # 2. Fresh snapshot + view (§4.1: the single LangGraph evidence boundary).
        try:
            snapshot = await self._system_snapshot()
            view: SnapshotView = snapshot_to_view(snapshot)
        except Exception:  # noqa: BLE001 — §9 'projector bug' stays inside the boundary
            self.log.bind(stage=stage).warning("transcript_projection_refused")
            return _infra_failure_for(stage, "snapshot could not be viewed")

        # 3. Pure projection with injected time and aliveness (§4.1 [v2]).
        try:
            projection = project(
                view,
                existing_keys=context.existing_keys,
                turn_rows=context.turn_rows.model_copy(update={"paused_plan": self._paused_plan}),
                now=datetime.now(tz=UTC),
                aliveness=self._lock.is_alive(),
            )
        except Exception:  # noqa: BLE001 — see above
            self.log.bind(stage=stage).warning("transcript_projection_refused")
            return _infra_failure_for(stage, "projection could not run")
        if projection.is_no_op and projection.no_op is not None:
            noop = projection.no_op
            self.log.bind(stage=stage, reason=noop.reason).info("transcript_projection_noop")

        # 4. Apply (sub-step of the bounded boundary; the WS loop never waits
        # longer than the boundary deadline, §5.2).
        usage_map: dict[str, int] = {}
        if attach_usage and self._usage.usage.total:
            # §7 target: this projection's last assistant row; when the
            # projector deduped it away (row already exists), the saved turn
            # rows resolve the target and the repository fills usage-only.
            last_key = _segment_row_key(projection, context.turn_rows)
            if last_key is not None:
                usage_map[last_key] = self._usage.usage.total
        applied = await self._repo.apply_projection(
            projection,
            user_id=self._user_id,
            session_id=self._thread_id,
            usage=usage_map,
        )

        if isinstance(applied, Failure):
            await self._register_reconcile_hint()
        return applied

    async def _system_snapshot(self) -> StateSnapshot:
        return await self._graph.aget_state(self._config)

    async def _register_reconcile_hint(self) -> None:
        """Best-effort Redis hint; sweep is the backstop (§6 trigger 4)."""
        try:
            await self._redis.sadd(_RECONCILE_HINT_KEY, self._thread_id)
        except Exception:  # noqa: BLE001 — a hint fail is fail-open (§9)
            self.log.debug("reconcile_hint_unwritten")


def _as_infra_failure(
    value: object,
) -> TranscriptInfrastructureError | None:
    """isinstance narrowing (P1) on a TranscriptResult variant → payload or None."""
    if isinstance(value, Failure):
        inner = value.failure()
        if isinstance(inner, TranscriptInfrastructureError):
            return inner
    return None


def _infra_failure_for(stage: str, what: str) -> TranscriptResult[ApplyOutcome]:
    return Failure(
        TranscriptInfrastructureError(message=what, source="turn_recorder", operation=stage)
    )


_ASSISTANT_PROMPT_KINDS = frozenset({"plan_prompt", "clarification_prompt", "final_summary"})


def _segment_row_key(projection: object, saved_turn_rows: object | None = None) -> str | None:
    """Segment totals attach to that segment's LAST assistant row (§7).

    The projector drops rows keyed in the saved turn context, so when this
    projection carries no assistant row the saved rows resolve the target
    instead — the repository then fills usage on the existing row, once.
    """
    emitted = [r for r in getattr(projection, "rows", ()) if r.kind in _ASSISTANT_PROMPT_KINDS]
    if emitted:
        return str(emitted[-1].source_key)
    saved = [
        r
        for r in getattr(saved_turn_rows, "rows", ()) or ()
        if getattr(r, "kind", None) in _ASSISTANT_PROMPT_KINDS
    ]
    return str(saved[-1].source_key) if saved else None


def _paused_plan_of(snapshot: StateSnapshot | None) -> tuple[dict[str, object], ...]:
    """The plan the user was shown — lives in the pending interrupt's payload.

    Captured at T1 because at T1' the interrupt is gone from the fresh snapshot.
    """
    plan_steps: list[dict[str, object]] = []
    for task in snapshot.tasks if snapshot is not None else ():
        for interrupt in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None)
            raw_plan = value.get("plan") if isinstance(value, dict) else None
            if not isinstance(raw_plan, list):
                continue
            for step in raw_plan:
                if isinstance(step, dict):
                    plan_steps.append(step)
                elif hasattr(step, "model_dump"):
                    plan_steps.append(dict(step.model_dump()))
    return tuple(plan_steps)
