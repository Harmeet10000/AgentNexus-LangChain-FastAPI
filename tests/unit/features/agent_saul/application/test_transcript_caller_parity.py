"""Parity contracts for the live recorder and recovery sweep callers.

These tests deliberately enter through the real caller paths.  The repository
double records the projection transaction each caller requests, so a shared
helper that is deterministic in isolation cannot make this suite pass when a
caller supplies different evidence or interprets ``ApplyOutcome`` differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast, override

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from returns.result import Success

from app.features.agent_saul.application.reconcile import _reconcile_once, _ThreadReconcileOutcome
from app.features.agent_saul.application.transcript.projection import (
    SessionStatus,
    TurnRow,
    TurnRows,
)
from app.features.agent_saul.application.transcript.recorder import TurnRecorder
from app.features.agent_saul.application.transcript.repository import (
    ApplyOutcome,
    ChatTranscriptRepository,
    TranscriptTurnContext,
)
from app.features.agent_saul.domain.state import STATE_SCHEMA_VERSION

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from langgraph.graph.state import CompiledStateGraph
    from redis.asyncio import Redis

    from app.features.agent_saul.application.thread_lock import ThreadLock
    from app.features.agent_saul.application.transcript.errors import TranscriptResult
    from app.features.agent_saul.application.transcript.projection import Projection

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
OLD_CHECKPOINT = datetime(2020, 1, 1, tzinfo=UTC)
THREAD_ID = "thread-caller-parity"
USER_ID = "user-caller-parity"
CHECKPOINT_ID = "checkpoint-caller-parity"
PRIOR_CHECKPOINT_ID = "checkpoint-before-parity"


@dataclass(frozen=True)
class _ProjectionCall:
    projection: Projection
    user_id: str
    session_id: str
    usage: Mapping[str, int] | None
    usage_metadata: Mapping[str, Mapping[str, Any]] | None
    prior_checkpoint_id: str | None


class _CaptureRepository(ChatTranscriptRepository):
    """Concrete repository subtype with only the caller-facing seams replaced."""

    def __init__(self, context: TranscriptTurnContext, outcome: ApplyOutcome) -> None:
        self.context = context
        self.outcome = outcome
        self.calls: list[_ProjectionCall] = []

    @override
    async def get_turn_context(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> TranscriptResult[TranscriptTurnContext]:
        assert (session_id, user_id) == (THREAD_ID, USER_ID)
        return Success(self.context)

    @override
    async def get_sweep_context(
        self,
        *,
        session_id: str,
    ) -> TranscriptResult[tuple[str, TranscriptTurnContext]]:
        assert session_id == THREAD_ID
        return Success((USER_ID, self.context))

    @override
    async def apply_projection(
        self,
        projection: Projection,
        *,
        user_id: str,
        session_id: str,
        usage: Mapping[str, int] | None = None,
        usage_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        prior_checkpoint_id: str | None,
    ) -> TranscriptResult[ApplyOutcome]:
        self.calls.append(
            _ProjectionCall(
                projection=projection,
                user_id=user_id,
                session_id=session_id,
                usage=usage,
                usage_metadata=usage_metadata,
                prior_checkpoint_id=prior_checkpoint_id,
            )
        )
        if projection.is_no_op:
            reason = projection.no_op.reason if projection.no_op is not None else "unspecified"
            return Success(ApplyOutcome(applied=False, reason=f"noop:{reason}"))
        return Success(self.outcome)


class _SnapshotGraph:
    def __init__(self, state: StateSnapshot) -> None:
        self.state = state
        self.configs: list[RunnableConfig] = []

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        self.configs.append(config)
        return self.state


class _Lock:
    def __init__(self, alive: bool) -> None:
        self.alive = alive

    def is_alive(self) -> bool:
        return self.alive


def _tagged_query() -> HumanMessage:
    return HumanMessage(
        content="Is this NDA mutual?",
        additional_kwargs={"saul_kind": "query"},
        id="query-caller-parity",
    )


def _snapshot(
    *,
    messages: list[HumanMessage] | None = None,
    next_nodes: tuple[str, ...] = ("planner",),
    final_report: dict[str, str] | None = None,
    created_at: datetime = NOW,
    plan: list[dict[str, object]] | None = None,
    plan_approved: bool = False,
) -> StateSnapshot:
    return StateSnapshot(
        values={
            "user_id": USER_ID,
            "thread_id": THREAD_ID,
            "schema_version": STATE_SCHEMA_VERSION,
            "messages": messages if messages is not None else [_tagged_query()],
            "plan": plan or [],
            "plan_approved": plan_approved,
            "errors": [],
            "final_report": final_report,
        },
        next=next_nodes,
        config=RunnableConfig(
            configurable={"thread_id": THREAD_ID, "checkpoint_id": CHECKPOINT_ID}
        ),
        metadata={"step": 5},
        created_at=created_at.isoformat(),
        parent_config=None,
        tasks=(),
        interrupts=(),
    )


async def _exercise_callers(
    state: StateSnapshot,
    *,
    outcome: ApplyOutcome,
    aliveness: bool,
    attach_usage: bool = False,
    turn_rows: TurnRows | None = None,
) -> tuple[
    ApplyOutcome,
    _ThreadReconcileOutcome,
    _CaptureRepository,
    _CaptureRepository,
]:
    context = TranscriptTurnContext(
        last_checkpoint_id=PRIOR_CHECKPOINT_ID,
        turn_rows=turn_rows or TurnRows(),
    )
    recorder_repo = _CaptureRepository(context, outcome)
    sweep_repo = _CaptureRepository(context, outcome)

    recorder_graph = _SnapshotGraph(state)
    recorder = TurnRecorder(
        repo=recorder_repo,
        graph=cast("CompiledStateGraph[Any]", recorder_graph),
        config=RunnableConfig(configurable={"thread_id": THREAD_ID}),
        thread_id=THREAD_ID,
        user_id=USER_ID,
        redis=cast("Redis", object()),
        lock=cast("ThreadLock", _Lock(aliveness)),
    )
    if attach_usage:
        recorder.usage_handler.usage.input_tokens = 7
        recorder.usage_handler.usage.output_tokens = 5
        recorder.usage_handler.on_tool_end(None, name="search")

    recorder_result = await recorder._project_core(
        stage="caller_parity",
        attach_usage=attach_usage,
    )
    assert isinstance(recorder_result, Success)

    sweep_graph = _SnapshotGraph(state)
    sweep_result = await _reconcile_once(
        repo=sweep_repo,
        graph=cast("CompiledStateGraph[Any]", sweep_graph),
        redis=cast("Redis", object()),
        session_id=THREAD_ID,
        now=NOW,
        aliveness=aliveness,
    )

    assert recorder_graph.configs == [RunnableConfig(configurable={"thread_id": THREAD_ID})]
    assert sweep_graph.configs == [RunnableConfig(configurable={"thread_id": THREAD_ID})]
    return recorder_result.unwrap(), sweep_result, recorder_repo, sweep_repo


async def test_recorder_and_sweep_apply_identical_active_projection() -> None:
    applied = ApplyOutcome(
        applied=True,
        rows_inserted=1,
        turn_seq=3,
        turn_opened=True,
    )

    recorder_outcome, sweep_outcome, recorder_repo, sweep_repo = await _exercise_callers(
        _snapshot(),
        outcome=applied,
        aliveness=True,
    )

    assert recorder_outcome == applied
    assert sweep_outcome == _ThreadReconcileOutcome(
        status=SessionStatus.ACTIVE.value,
        checkpoint_id=CHECKPOINT_ID,
        clean=False,
        failed=False,
    )
    assert recorder_repo.calls == sweep_repo.calls
    call = recorder_repo.calls[0]
    assert call.projection.session_status == SessionStatus.ACTIVE
    assert [row.kind for row in call.projection.rows] == ["query"]
    assert call.prior_checkpoint_id == PRIOR_CHECKPOINT_ID


@pytest.mark.parametrize("modified", [False, True])
async def test_restart_recovery_preserves_plan_decision_without_plan_leakage(
    modified: bool,
) -> None:
    original = {"step": "confidential original plan", "ordinal": 1}
    resumed = {"step": "confidential modified plan", "ordinal": 1} if modified else original
    context_rows = TurnRows(
        rows=(
            TurnRow(kind="query", source_key="query-caller-parity", ordinal=0),
            TurnRow(
                kind="plan_prompt",
                source_key="interrupt:plan-caller-parity",
                ordinal=1,
            ),
        ),
        pending_plan_decision_key="interrupt:plan-caller-parity",
        pending_plan_decision_modified=modified,
        pending_plan_decision_checkpoint_id=CHECKPOINT_ID,
    )
    _, sweep_outcome, recorder_repo, sweep_repo = await _exercise_callers(
        _snapshot(
            plan=[resumed],
            plan_approved=True,
        ),
        outcome=ApplyOutcome(applied=True, rows_inserted=1, turn_seq=1),
        aliveness=True,
        turn_rows=context_rows,
    )

    assert sweep_outcome.status == SessionStatus.RESUMED.value
    assert recorder_repo.calls == sweep_repo.calls
    decision = next(
        row for row in recorder_repo.calls[0].projection.rows if row.kind == "plan_decision"
    )
    assert decision.content == ("modified" if modified else "approved")
    assert original["step"] not in str(decision.model_dump())
    assert resumed["step"] not in str(decision.model_dump())


@pytest.mark.parametrize(
    ("state", "aliveness", "expected_status", "terminal_kind"),
    [
        (
            _snapshot(next_nodes=(), final_report={"summary": "Report is ready."}),
            True,
            SessionStatus.COMPLETED,
            "final_summary",
        ),
        (
            _snapshot(next_nodes=(), created_at=OLD_CHECKPOINT),
            True,
            SessionStatus.FAILED,
            "turn_closed",
        ),
    ],
)
async def test_recorder_and_sweep_preserve_terminal_row_semantics(
    state: StateSnapshot,
    aliveness: bool,
    expected_status: SessionStatus,
    terminal_kind: str,
) -> None:
    applied = ApplyOutcome(applied=True, rows_inserted=2, turn_seq=4, turn_opened=True)

    recorder_outcome, sweep_outcome, recorder_repo, sweep_repo = await _exercise_callers(
        state,
        outcome=applied,
        aliveness=aliveness,
    )

    assert recorder_outcome == applied
    assert sweep_outcome.status == expected_status.value
    assert sweep_outcome.failed is False
    assert recorder_repo.calls == sweep_repo.calls
    rows = recorder_repo.calls[0].projection.rows
    assert terminal_kind in {row.kind for row in rows}
    if terminal_kind == "final_summary":
        assert not any(row.kind == "turn_closed" for row in rows)
    else:
        assert not any(row.kind == "final_summary" for row in rows)


async def test_recorder_and_sweep_preserve_detailed_noop_reason() -> None:
    untagged = HumanMessage(content="not a turn opener", id="untagged-message")

    recorder_outcome, sweep_outcome, recorder_repo, sweep_repo = await _exercise_callers(
        _snapshot(messages=[untagged]),
        outcome=ApplyOutcome(applied=True),
        aliveness=True,
    )

    assert recorder_outcome == ApplyOutcome(
        applied=False,
        reason="noop:missing_tagged_query",
    )
    assert sweep_outcome == _ThreadReconcileOutcome(
        reason="noop:missing_tagged_query",
        clean=True,
        failed=False,
    )
    # A no-op is an explicit shared-orchestration result, not a write request.
    assert recorder_repo.calls == sweep_repo.calls == []


@pytest.mark.parametrize(
    ("applied", "expected_sweep"),
    [
        (
            ApplyOutcome(applied=False, reason="stale_projection"),
            _ThreadReconcileOutcome(),
        ),
        (
            ApplyOutcome(applied=False, reason="same_checkpoint"),
            _ThreadReconcileOutcome(
                status=SessionStatus.ACTIVE.value,
                checkpoint_id=CHECKPOINT_ID,
                clean=False,
                failed=False,
            ),
        ),
    ],
)
async def test_callers_preserve_repository_no_apply_outcomes(
    applied: ApplyOutcome,
    expected_sweep: _ThreadReconcileOutcome,
) -> None:
    recorder_outcome, sweep_outcome, recorder_repo, sweep_repo = await _exercise_callers(
        _snapshot(),
        outcome=applied,
        aliveness=True,
    )

    assert recorder_outcome == applied
    assert sweep_outcome == expected_sweep
    assert recorder_repo.calls == sweep_repo.calls


async def test_live_usage_side_input_does_not_change_caller_projection_parity() -> None:
    applied = ApplyOutcome(applied=True, rows_inserted=2, turn_seq=5, turn_opened=True)

    recorder_outcome, sweep_outcome, recorder_repo, sweep_repo = await _exercise_callers(
        _snapshot(next_nodes=(), final_report={"summary": "Report is ready."}),
        outcome=applied,
        aliveness=True,
        attach_usage=True,
    )

    assert recorder_outcome == applied
    assert sweep_outcome.status == SessionStatus.COMPLETED.value
    recorder_call = recorder_repo.calls[0]
    sweep_call = sweep_repo.calls[0]
    assert recorder_call.projection == sweep_call.projection
    assert recorder_call.usage == {"final:query-caller-parity": 12}
    assert recorder_call.usage_metadata == {
        "final:query-caller-parity": {
            "usage": {"input": 7, "output": 5},
            "tool_calls": [{"name": "search", "ok": True, "ms": None}],
        }
    }
    assert sweep_call.usage is None
    assert sweep_call.usage_metadata is None
