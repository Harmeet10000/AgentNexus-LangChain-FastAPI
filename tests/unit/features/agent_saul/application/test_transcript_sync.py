"""Usage side-input contracts for the shared transcript synchronization seam."""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.messages import HumanMessage

from app.features.agent_saul.application.transcript.projection import (
    SnapshotView,
    TurnRow,
    TurnRows,
)
from app.features.agent_saul.application.transcript.sync import prepare_snapshot

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


def _view() -> SnapshotView:
    return SnapshotView(
        user_id="user-usage",
        checkpoint_id="checkpoint-usage",
        checkpoint_ts=NOW,
        step=3,
        next_nodes=("planner",),
        messages=(
            HumanMessage(
                content="question",
                additional_kwargs={"saul_kind": "query"},
                id="query-usage",
            ),
        ),
    )


def _turn_rows() -> TurnRows:
    return TurnRows(
        rows=(
            TurnRow(kind="query", source_key="query-usage", ordinal=0),
            TurnRow(kind="plan_prompt", source_key="prompt-usage", ordinal=1),
        )
    )


def test_zero_token_tool_metadata_survives_empty_projection() -> None:
    prepared = prepare_snapshot(
        view=_view(),
        existing_keys=frozenset({"query-usage", "prompt-usage"}),
        turn_rows=_turn_rows(),
        now=NOW,
        aliveness=True,
        usage_total=0,
        usage_metadata={
            "usage": {"input": 0, "output": 0},
            "tool_calls": [{"name": "search", "ok": True, "ms": None}],
        },
    )

    assert prepared.projection.rows == ()
    assert prepared.usage == {"prompt-usage": 0}
    assert prepared.usage_metadata == {
        "prompt-usage": {
            "usage": {"input": 0, "output": 0},
            "tool_calls": [{"name": "search", "ok": True, "ms": None}],
        }
    }


def test_usage_side_input_is_absent_when_no_usage_was_collected() -> None:
    prepared = prepare_snapshot(
        view=_view(),
        existing_keys=frozenset({"query-usage", "prompt-usage"}),
        turn_rows=_turn_rows(),
        now=NOW,
        aliveness=True,
    )

    assert prepared.projection.rows == ()
    assert prepared.usage is None
    assert prepared.usage_metadata is None
