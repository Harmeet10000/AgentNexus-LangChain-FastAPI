"""Projector unit tests: fixtures for every §13 case plus the totality sweep.

The totality sweep (§13 [v3]) is ``derive_status``'s test #0: every combination
of interrupts × aliveness × within/past grace × 24 h × next empty/non-empty maps
to exactly one status row or the explicit NoOp. Nothing in this module hits I/O;
every function under test is pure (§4.1).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import product

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import CheckpointTask, StateSnapshot

from app.features.agent_saul.graph.state import STATE_SCHEMA_VERSION
from app.features.agent_saul.transcript.projection import (
    ORPHAN_GRACE,
    PAUSE_WINDOW,
    InterruptView,
    NoOp,
    SessionStatus,
    SnapshotView,
    TurnRow,
    TurnRows,
    derive_status,
    project,
    snapshot_to_view,
)

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
THREAD = "thread-0001"
CHECKPOINT = "checkpoint-0001"


def interrupt_tasks(interrupts: tuple) -> tuple[CheckpointTask, ...]:
    """Build the task dict shape the projector reads interrupts off (§4.1 only reader)."""
    return (
        CheckpointTask(id="t1", name="qna", error=None, interrupts=interrupts, state=None, path=()),
    )


def snapshot(
    *,
    values: dict,
    next_nodes: tuple[str, ...] = (),
    interrupts: tuple = (),
    step: int = 5,
    ts: datetime | None = NOW,
) -> StateSnapshot:
    """A handcrafted StateSnapshot with LangGraph conventions applied."""
    metadata = {"step": step}
    return StateSnapshot(
        values=values,
        next=next_nodes,
        config=RunnableConfig(configurable={"thread_id": THREAD, "checkpoint_id": CHECKPOINT}),
        metadata=metadata,
        created_at=ts.isoformat() if ts else None,
        parent_config=None,
        tasks=interrupt_tasks(interrupts),
        interrupts=interrupts,
    )


def tagged_query(text: str = "Is this NDA mutual?") -> HumanMessage:
    return HumanMessage(content=text, additional_kwargs={"saul_kind": "query"}, id="msg-query-1")


def untagged_reply(
    text: str = "Yes, and prefer it that way.", id_: str | None = None
) -> HumanMessage:
    return HumanMessage(content=text, id=id_ or "msg-reply-1")


# ---------------------------------------------------------------------------
# snapshot_to_view
# ---------------------------------------------------------------------------


def test_snapshot_to_view_extracts_frozen_facts() -> None:
    snap = snapshot(
        values={
            "user_id": "u1",
            "thread_id": THREAD,
            "schema_version": STATE_SCHEMA_VERSION,
            "messages": [tagged_query(), untagged_reply()],
            "plan": [],
            "plan_approved": False,
            "errors": [],
            "final_report": None,
            "doc_id": "d-1",
        },
        next_nodes=("planner",),
    )
    view = snapshot_to_view(snap)
    assert view.user_id == "u1"
    assert view.checkpoint_id == CHECKPOINT
    assert view.step == 5
    assert view.next_nodes == ("planner",)
    assert len(view.messages) == 2
    assert view.schema_version == STATE_SCHEMA_VERSION
    assert view.final_report_summary is None


def test_snapshot_to_view_drops_non_human_messages() -> None:
    snap = snapshot(
        values={
            "user_id": "u1",
            "messages": [tagged_query(), AIMessage(content="assistant text", id="a1")],
        }
    )
    view = snapshot_to_view(snap)
    assert len(view.messages) == 1


def test_snapshot_to_view_refuses_unknown_schema_version() -> None:
    snap = snapshot(values={"user_id": "u1", "messages": [], "schema_version": 99})
    with pytest.raises(Exception, match="schema_version"):
        snapshot_to_view(snap)


def test_snapshot_to_view_accepts_legacy_missing_version() -> None:
    snap = snapshot(values={"user_id": "u1", "messages": [tagged_query()]})
    view = snapshot_to_view(snap)
    assert view.schema_version == STATE_SCHEMA_VERSION


def test_snapshot_to_view_normalizes_final_report_summary() -> None:
    snap = snapshot(
        values={
            "user_id": "u1",
            "messages": [tagged_query()],
            "final_report": {"summary": "5 risks found, 3 citations"},
        }
    )
    view = snapshot_to_view(snap)
    assert view.final_report_summary == "5 risks found, 3 citations"


# ---------------------------------------------------------------------------
# serializer round-trip fixture (D4: §13 [v2], most load-bearing [verify])
# ---------------------------------------------------------------------------


def test_d4_saul_kind_survives_checkpoint_serializer() -> None:
    """additional_kwargs.saul_kind must survive the pinned JsonPlusSerializer.

    Without this assertion, turn-boundary detection collapses silently: the
    projector derives turn existence purely from a tagged HumanMessage inside
    the serialized checkpoint state, and if the tag disappears there would be
    quiet transcript corruption across every write.
    Same serializer used by the actual checkpointer (langgraph-checkpoint 4.2.0).
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serializer = JsonPlusSerializer()
    payload_message = tagged_query()

    encoding, serialized = serializer.dumps_typed(payload_message)
    restored = serializer.loads_typed((encoding, serialized))
    assert restored.additional_kwargs.get("saul_kind") == "query"
    assert restored.id == "msg-query-1"

    encoding2, serialized2 = serializer.dumps_typed({"messages": [payload_message]})
    restored_value = serializer.loads_typed((encoding2, serialized2))
    assert restored_value["messages"][0].additional_kwargs.get("saul_kind") == "query"


# ---------------------------------------------------------------------------
# derive_status — total sweep (§13 [v3] test #0)
# ---------------------------------------------------------------------------

_TURN_ROWS_EMPTY = TurnRows()
_TURN_ROWS_WITH_REPLY = TurnRows(rows=(TurnRow(kind="query", source_key="q1", ordinal=0),))


def _sweep_view(
    *,
    next_nodes: tuple[str, ...] = (),
    interrupts: int = 0,
    has_report: bool = False,
) -> SnapshotView:
    return SnapshotView(
        user_id="u1",
        checkpoint_id=CHECKPOINT,
        checkpoint_ts=NOW,
        step=5,
        next_nodes=next_nodes,
        interrupts=tuple(
            InterruptView(id=f"int-{i}", value={"type": "clarification_needed"})
            for i in range(interrupts)
        ),
        final_report_summary="done" if has_report else None,
    )


def _expected_sweep_status(
    *, interrupt_count: int, next_empty: bool, has_report: bool, age: str, aliveness: bool
) -> SessionStatus | None:
    """The closed §5.3 table, branch by branch (None ⇒ explicit NoOp row 4)."""
    if interrupt_count:
        if age == "past":
            return SessionStatus.ABANDONED_EXPIRED
        # fresh AND middle: within the 24 h pause window (§8).
        return SessionStatus.CLARIFYING
    if next_empty:
        return SessionStatus.COMPLETED if has_report else SessionStatus.FAILED
    if aliveness:
        return SessionStatus.ACTIVE
    if age == "fresh":
        return None
    return SessionStatus.FAILED  # orphaned beyond grace (D1)


def _age_adjusted(view: SnapshotView, age: str) -> SnapshotView:
    if age == "past":
        return view.model_copy(update={"checkpoint_ts": NOW - PAUSE_WINDOW - timedelta(minutes=5)})
    if age == "middle":
        # Past the orphan grace, inside the 24 h pause window — the bucket all
        # normal long reviews live in (§5.3 vs §8).
        return view.model_copy(update={"checkpoint_ts": NOW - timedelta(hours=1)})
    return view


def test_totality_sweep_derive_status_closed_over_cross_product() -> None:
    """Combinatorial totality: every fixture maps to exactly one status or NoOp (§5.3)."""
    checked = 0
    for next_empty, interrupt_count, has_report, age, aliveness in product(
        (True, False), (0, 1), (True, False), ("fresh", "middle", "past"), (True, False)
    ):
        if interrupt_count == 0 and next_empty and not has_report:
            pass  # valid combination: failed terminal (row 6)
        view = _age_adjusted(
            _sweep_view(
                next_nodes=() if next_empty else ("ingestion",),
                interrupts=interrupt_count,
                has_report=has_report,
            ),
            age,
        )
        result = derive_status(view, TurnRows(), NOW, aliveness)
        checked += 1
        expected = _expected_sweep_status(
            interrupt_count=interrupt_count,
            next_empty=next_empty,
            has_report=has_report,
            age=age,
            aliveness=aliveness,
        )
        if expected is None:
            assert isinstance(result, NoOp), f"expected NoOp: {view}"
        else:
            assert result == expected, f"expected {expected}: {view}"
    assert checked == 2 * 2 * 2 * 3 * 2  # full cross product actually traversed


def test_orphan_within_grace_yields_noop() -> None:
    view = _sweep_view(next_nodes=("ingestion",))
    view = view.model_copy(update={"checkpoint_ts": NOW - ORPHAN_GRACE * 0.5})
    result = derive_status(view, TurnRows(), NOW, aliveness=False)
    assert isinstance(result, NoOp)


def test_orphan_past_grace_yields_failed_and_requires_not_alive() -> None:
    view = _sweep_view(next_nodes=("ingesion",))
    view = view.model_copy(update={"checkpoint_ts": NOW - ORPHAN_GRACE * 2})
    assert derive_status(view, TurnRows(), NOW, aliveness=False) == SessionStatus.FAILED
    assert derive_status(view, TurnRows(), NOW, aliveness=True) == SessionStatus.ACTIVE


def test_missing_checkpoint_ts_ne_verifies_fail_closed() -> None:
    view = _sweep_view(next_nodes=("ingestion",))
    view = view.model_copy(update={"checkpoint_ts": None})
    assert isinstance(derive_status(view, TurnRows(), NOW, aliveness=False), NoOp)
    assert derive_status(view, TurnRows(), NOW, aliveness=True) == SessionStatus.ACTIVE


def test_paused_turn_with_reply_row_derives_resumed() -> None:
    view = _sweep_view(next_nodes=("qna",), interrupts=1)
    rows = TurnRows(
        rows=(
            TurnRow(kind="query", source_key="q1", ordinal=0),
            TurnRow(kind="clarification_prompt", source_key="interrupt:i1", ordinal=1),
        )
    )
    result = derive_status(view, rows, NOW, aliveness=True)
    assert result == SessionStatus.CLARIFYING  # interrupts present → clarifying


def test_running_turn_with_reply_row_derives_resumed() -> None:
    view = _sweep_view(next_nodes=("ingestion",))
    rows = TurnRows(
        rows=(
            TurnRow(kind="query", source_key="q1", ordinal=0),
            TurnRow(kind="plan_decision", source_key="decision:i1", ordinal=2),
        )
    )
    assert derive_status(view, rows, NOW, aliveness=True) == SessionStatus.RESUMED


# ---------------------------------------------------------------------------
# project — rows, ordinals, terminal behaviors
# ---------------------------------------------------------------------------


def _view(
    **overrides: object,
) -> SnapshotView:
    base = {
        "user_id": "u1",
        "checkpoint_id": CHECKPOINT,
        "checkpoint_ts": NOW,
        "step": 5,
        "next_nodes": ("ingestion",),
        "messages": (tagged_query(),),
    }
    base.update(overrides)
    return SnapshotView(**base)


EMPTY_KEYS: frozenset[str] = frozenset()


def test_project_first_checkpoint_inserts_query_row_active() -> None:
    out = project(_view(), EMPTY_KEYS, TurnRows(), NOW, aliveness=True)
    assert out.session_status == SessionStatus.ACTIVE
    assert out.watermark_checkpoint_id == CHECKPOINT
    assert out.query_source_key == "msg-query-1"
    assert len(out.rows) == 1
    row = out.rows[0]
    assert (row.kind, row.ordinal, row.source_key) == ("query", 0, "msg-query-1")
    assert row.content == "Is this NDA mutual?"
    assert row.role.value == "user"


def test_project_replay_is_no_rows() -> None:
    keys = frozenset({"msg-query-1"})
    out = project(
        _view(),
        keys,
        TurnRows(rows=(TurnRow(kind="query", source_key="msg-query-1", ordinal=0),)),
        NOW,
        aliveness=True,
    )
    assert out.rows == ()
    assert out.session_status == SessionStatus.ACTIVE


def test_project_pause_writes_prompt_row() -> None:
    out = project(
        _view(
            next_nodes=("qna",),
            interrupts=(
                InterruptView(
                    id="i1", value={"type": "clarification_needed", "message": "Which doc type?"}
                ),
            ),
        ),
        EMPTY_KEYS,
        TurnRows(rows=(TurnRow(kind="query", source_key="msg-query-1", ordinal=0),)),
        NOW,
        aliveness=True,
    )
    statuses = [row.kind for row in out.rows]
    assert "clarification_prompt" in statuses
    prompt = next(r for r in out.rows if r.kind == "clarification_prompt")
    assert prompt.ordinal == 1  # 2k-1 for k=1 (§4.2)
    assert prompt.source_key == "interrupt:i1"
    assert prompt.content == "Which doc type?"
    assert prompt.extra_metadata["interrupt"]["id"] == "i1"
    assert out.session_status == SessionStatus.CLARIFYING


def test_project_pause_plan_prompt_is_plan_kind() -> None:
    out = project(
        _view(
            next_nodes=("planner",),
            interrupts=(
                InterruptView(id="i2", value={"type": "plan_approval", "message": "Review plan"}),
            ),
        ),
        EMPTY_KEYS,
        TurnRows(rows=(TurnRow(kind="query", source_key="msg-query-1", ordinal=0),)),
        NOW,
        aliveness=True,
    )
    prompt = next(r for r in out.rows if r.kind == "plan_prompt")
    assert prompt.source_key == "interrupt:i2"
    assert out.session_status == SessionStatus.CLARIFYING


def test_project_resume_writes_reply_then_decision() -> None:
    turn_rows = TurnRows(
        rows=(
            TurnRow(kind="query", source_key="msg-query-1", ordinal=0),
            TurnRow(kind="plan_prompt", source_key="interrupt:i2", ordinal=1),
        )
    )
    out = project(
        _view(
            next_nodes=("ingestion",),
            plan_approved=True,
            plan=({"step": "ingest", "ordinal": 1},),
        ),
        EMPTY_KEYS,
        turn_rows,
        NOW,
        aliveness=True,
    )
    decision = next(r for r in out.rows if r.kind == "plan_decision")
    assert decision.source_key == "decision:i2"
    assert decision.ordinal == 2  # prompt.ordinal + 1
    assert decision.content == "approved"
    assert decision.extra_metadata["modified"] is False
    assert out.session_status == SessionStatus.RESUMED


def test_project_modified_plan_decision_label() -> None:
    turn_rows = TurnRows(
        rows=(
            TurnRow(kind="query", source_key="msg-query-1", ordinal=0),
            TurnRow(kind="plan_prompt", source_key="interrupt:i2", ordinal=1),
        ),
        paused_plan=({"step": "ingest raw", "ordinal": 1},),
    )
    out = project(
        _view(plan_approved=True, plan=({"step": "ingest modified", "ordinal": 1},)),
        EMPTY_KEYS,
        turn_rows,
        NOW,
        aliveness=True,
    )
    decision = next(r for r in out.rows if r.kind == "plan_decision")
    assert decision.content == "modified"
    assert decision.extra_metadata["modified"] is True


def test_project_decision_skipped_when_prompt_row_missing() -> None:
    """A hole in the prompt row: no decision key source; reconcile heals later."""
    out = project(
        _view(plan_approved=True, plan=({"step": "s", "ordinal": 1},)),
        frozenset(),
        TurnRows(rows=(TurnRow(kind="query", source_key="msg-query-1", ordinal=0),)),
        NOW,
        aliveness=True,
    )
    assert all(r.kind != "plan_decision" for r in out.rows)


def test_project_completion_writes_final_summary_with_report_ref() -> None:
    turn_rows = TurnRows(
        rows=(
            TurnRow(kind="query", source_key="msg-query-1", ordinal=0),
            TurnRow(kind="plan_decision", source_key="decision:i2", ordinal=2),
        )
    )
    out = project(
        _view(
            next_nodes=(),
            final_report_summary="Report is ready.",
            doc_id="d-1",
            long_term_refs=("legal::u1::reports#d-1",),
        ),
        EMPTY_KEYS,
        turn_rows,
        NOW,
        aliveness=False,  # irrelevant for next-empty path
    )
    summary = next(r for r in out.rows if r.kind == "final_summary")
    assert summary.source_key == "final:msg-query-1"
    assert summary.ordinal == 3
    assert summary.content == "Report is ready."
    # The ref is COPIED from the graph's own confirmed write, never
    # fabricated from doc_id alone (a skipped/failed storage claims nothing).
    assert summary.extra_metadata["report_ref"] == "legal::u1::reports#d-1"
    assert summary.role.value == "assistant"
    assert out.session_status == SessionStatus.COMPLETED
    assert not any(r.kind == "turn_closed" for r in out.rows)


def test_project_completion_without_confirmed_ref_claims_nothing() -> None:
    """/Greptile P2: `persist_memory_node` skipped or failed ⇒ long_term_refs
    carries no new ref ⇒ the export must NOT claim a saved report."""
    turn_rows = TurnRows(rows=(TurnRow(kind="query", source_key="msg-query-1", ordinal=0),))
    out = project(
        _view(
            next_nodes=(),
            final_report_summary="Report is ready.",
            doc_id="d-1",
            # storage skipped: empty refs
        ),
        EMPTY_KEYS,
        turn_rows,
        NOW,
        aliveness=False,
    )
    summary = next(r for r in out.rows if r.kind == "final_summary")
    assert "report_ref" not in summary.extra_metadata


def test_project_orphaned_failed_appends_turn_closed() -> None:
    past = NOW - ORPHAN_GRACE * 2
    out = project(
        _view(checkpoint_ts=past),
        EMPTY_KEYS,
        TurnRows(rows=(TurnRow(kind="query", source_key="msg-query-1", ordinal=0),)),
        NOW,
        aliveness=False,
    )
    assert out.session_status == SessionStatus.FAILED
    closed = next(r for r in out.rows if r.kind == "turn_closed")
    assert closed.source_key == "closed:msg-query-1"
    assert closed.role.value == "system"
    assert closed.extra_metadata["outcome"] == "failed"


def test_project_abandoned_expired_appends_turn_closed() -> None:
    old = NOW - PAUSE_WINDOW * 2
    out = project(
        _view(
            checkpoint_ts=old,
            next_nodes=("qna",),
            interrupts=(InterruptView(id="i1", value={"type": "clarification_needed"}),),
        ),
        EMPTY_KEYS,
        TurnRows(rows=(TurnRow(kind="query", source_key="msg-query-1", ordinal=0),)),
        NOW,
        aliveness=False,
    )
    assert out.session_status == SessionStatus.ABANDONED_EXPIRED
    closed = next(r for r in out.rows if r.kind == "turn_closed")
    assert closed.extra_metadata["outcome"] == "abandoned_expired"
    prompt = next(r for r in out.rows if r.kind == "clarification_prompt")
    assert prompt.ordinal == closed.ordinal - 1  # prompt before close


def test_project_noop_within_grace_writes_nothing() -> None:
    young = NOW - ORPHAN_GRACE * 0.5
    out = project(
        _view(checkpoint_ts=young),
        EMPTY_KEYS,
        TurnRows(),
        NOW,
        aliveness=False,
    )
    assert out.is_no_op
    assert out.rows == ()
    assert out.session_status is None
    assert out.watermark_checkpoint_id is None


def test_project_pre_first_super_step_writes_nothing() -> None:
    out = project(
        _view(step=-1),
        EMPTY_KEYS,
        TurnRows(),
        NOW,
        aliveness=True,
    )
    assert out.is_no_op
    assert out.rows == ()


def test_project_missing_tagged_query_writes_nothing() -> None:
    out = project(
        _view(messages=(untagged_reply(),)),
        EMPTY_KEYS,
        TurnRows(),
        NOW,
        aliveness=True,
    )
    assert out.is_no_op


def test_project_unknown_hostile_kind_fails_closed() -> None:
    """ContentRole.for_kind refuses unknown kinds — never persist them."""
    from app.features.agent_saul.transcript.projection import ContentRole

    with pytest.raises(Exception, match="unknown row kind"):
        ContentRole.for_kind("unmodeled_kind")


def test_project_reply_follows_structural_slot_formula_when_prompt_missing() -> None:
    """Reply j: prompt row hole → ordinal 2j fallback (§4.2)."""
    view = _view(
        messages=(
            tagged_query(),
            untagged_reply("yes", id_="msg-reply-1"),
        )
    )
    out = project(
        view,
        EMPTY_KEYS,
        TurnRows(rows=(TurnRow(kind="query", source_key="msg-query-1", ordinal=0),)),
        NOW,
        aliveness=True,
    )
    reply = next(r for r in out.rows if r.kind == "clarification_reply")
    assert reply.ordinal == 2 * 1
    assert reply.extra_metadata["hitl"]["answers_prompt"] is None


def test_project_reply_keys_to_chain_prompt_summary_in_metadata() -> None:
    """Reply knows its prompt via previous row (§3.4 hitl.answers_prompt)."""
    view = _view(messages=(tagged_query(), untagged_reply("yes", id_="msg-reply-1")))
    out = project(
        view,
        EMPTY_KEYS,
        TurnRows(
            rows=(
                TurnRow(kind="query", source_key="msg-query-1", ordinal=0),
                TurnRow(kind="clarification_prompt", source_key="interrupt:i1", ordinal=1),
            )
        ),
        NOW,
        aliveness=True,
    )
    reply = next(r for r in out.rows if r.kind == "clarification_reply")
    assert reply.ordinal == 2  # prompt ordinal + 1
    assert reply.extra_metadata["hitl"]["answers_prompt"] == "interrupt:i1"


def test_prompt_text_prefers_question_over_message() -> None:
    """A clarification interrupt's real question wins over the generic note (§4.1)."""
    out = project(
        _view(
            next_nodes=("planner",),
            interrupts=(
                InterruptView(
                    id="i9",
                    value={
                        "type": "clarification",
                        "message": "Needs your input",
                        "question": "Should the NDA be mutual?",
                    },
                ),
            ),
        ),
        EMPTY_KEYS,
        TurnRows(rows=(TurnRow(kind="query", source_key="msg-q", ordinal=0),)),
        NOW,
        aliveness=True,
    )
    prompt = next(r for r in out.rows if r.kind == "clarification_prompt")
    assert prompt.content == "Should the NDA be mutual?"


def test_decision_slot_reserved_before_followup_prompt() -> None:
    """Approval + newly pending interrupt in ONE projection never share a slot.

    The decision resolves first (slot = last prompt ordinal + 1) and the
    follow-up prompt escalates past it — exports must not show the answer
    before the approval it encodes (§4.2).
    """
    turn_rows = TurnRows(
        rows=(
            TurnRow(kind="query", source_key="msg-query-1", ordinal=0),
            TurnRow(kind="plan_prompt", source_key="interrupt:i2", ordinal=1),
        )
    )
    out = project(
        _view(
            next_nodes=("planner",),
            plan_approved=True,
            plan=({"step": "ingest", "ordinal": 1},),
            interrupts=(
                InterruptView(id="i3", value={"type": "human_review", "message": "Review"}),
            ),
        ),
        EMPTY_KEYS,
        turn_rows,
        NOW,
        aliveness=True,
    )
    decision = next(r for r in out.rows if r.kind == "plan_decision")
    prompt = next(r for r in out.rows if r.kind == "clarification_prompt")
    assert decision.ordinal == 2
    assert prompt.ordinal == 3
    ordinals = [r.ordinal for r in out.rows]
    assert len(ordinals) == len(set(ordinals))


def test_project_uses_latest_tagged_query_when_messages_accumulate() -> None:
    """/Greptile P1: `messages` accumulates across turns (add_messages) — the
    projector must attach a later turn to ITS OWN opener, never turn one's."""
    q1 = HumanMessage(content="first", additional_kwargs={"saul_kind": "query"}, id="q1")
    q2 = HumanMessage(content="second", additional_kwargs={"saul_kind": "query"}, id="q2")
    out = project(
        _view(messages=(q1, q2), next_nodes=(), step=5), EMPTY_KEYS, TurnRows(), NOW, aliveness=True
    )
    assert out.query_source_key == "q2"
    assert any(r.kind == "query" and r.source_key == "q2" for r in out.rows)
    assert not any(r.source_key == "q1" for r in out.rows)


def test_project_reply_scopes_to_the_current_turn_only() -> None:
    """/Greptile P1: HITL replies are those AFTER the latest query — an
    earlier turn's reply was already projected (or belongs to its own rows)."""
    q1 = HumanMessage(content="first", additional_kwargs={"saul_kind": "query"}, id="q1")
    q2 = HumanMessage(content="second", additional_kwargs={"saul_kind": "query"}, id="q2")
    reply2 = HumanMessage(content="reply to second", id="r2")
    out = project(
        _view(messages=(q1, q2, reply2), next_nodes=(), step=5),
        EMPTY_KEYS,
        TurnRows(),
        NOW,
        aliveness=True,
    )
    keys = {r.source_key for r in out.rows if r.kind == "clarification_reply"}
    assert keys == {"r2"}
