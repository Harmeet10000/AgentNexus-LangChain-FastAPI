"""Property and unit tests for deep research process metrics.

Feature: evaluation-infrastructure-extension, Property 4: Ratio Metrics
Non-Negative and Finite, Property 5: Node Transition Tracking Completeness,
Property 6: Metric Computation Determinism.
"""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.ai.evaluation.process_metrics import (
    NodeTransition,
    ProcessMetricsError,
    ProcessMetricsTracker,
)

pytestmark = pytest.mark.property

_NAIVE_MIN = datetime(2020, 1, 1, tzinfo=UTC).replace(tzinfo=None)
_NAIVE_MAX = datetime(2030, 1, 1, tzinfo=UTC).replace(tzinfo=None)

_TIMESTAMPS = st.datetimes(
    min_value=_NAIVE_MIN,
    max_value=_NAIVE_MAX,
    timezones=st.just(UTC),
)

_OUTPUT_STATES = st.fixed_dictionaries(
    {},
    optional={
        "retrieved_document_ids": st.lists(
            st.text(min_size=1, max_size=8), max_size=6, unique=True
        ),
        "follow_up_queries": st.lists(st.text(min_size=1, max_size=8), max_size=6, unique=True),
        "primary_query_count": st.integers(min_value=0, max_value=20),
        "report_word_count": st.integers(min_value=0, max_value=10_000),
    },
)

_TRANSITIONS = st.builds(
    NodeTransition,
    node_name=st.sampled_from(["supervisor", "researcher", "reporter"]),
    timestamp=_TIMESTAMPS,
    input_state=st.just({"query": "initial"}),
    output_state=_OUTPUT_STATES,
    tool_calls=st.lists(st.text(min_size=1, max_size=8), max_size=6),
)


def _tracked(transitions: list[NodeTransition]) -> ProcessMetricsTracker:
    tracker = ProcessMetricsTracker()
    tracker.start_tracking(initial_query="initial", node_limit=len(transitions))
    for transition in transitions:
        tracker.record_node_transition(transition)
    return tracker


# ---------------------------------------------------------------------------
# Property 4: Ratio Metrics Non-Negative and Finite
# ---------------------------------------------------------------------------


@given(transitions=st.lists(_TRANSITIONS, max_size=20))
@settings(max_examples=100, deadline=None)
def test_ratio_metrics_non_negative_and_finite(
    transitions: list[NodeTransition],
) -> None:
    """Property 4: every computed ratio is finite and non-negative."""
    metrics = _tracked(transitions).compute_metrics()

    for value in (
        metrics.brief_to_report_coverage_ratio,
        metrics.follow_up_depth,
        metrics.breadth_vs_depth_ratio,
        metrics.total_duration_seconds,
    ):
        assert isfinite(value)
        assert value >= 0


# ---------------------------------------------------------------------------
# Property 5: Node Transition Tracking Completeness
# ---------------------------------------------------------------------------


@given(transitions=st.lists(_TRANSITIONS, max_size=20))
@settings(max_examples=100, deadline=None)
def test_node_transition_tracking_completeness(
    transitions: list[NodeTransition],
) -> None:
    """Property 5: recorded transitions match the input exactly and in order."""
    tracker = _tracked(transitions)
    metrics = tracker.compute_metrics()

    assert tracker.node_count == len(transitions)
    assert list(tracker.transitions) == transitions
    assert metrics.total_node_transitions == len(transitions)
    assert metrics.total_tool_calls == sum(len(item.tool_calls) for item in transitions)


# ---------------------------------------------------------------------------
# Property 6: Metric Computation Determinism
# ---------------------------------------------------------------------------


@given(transitions=st.lists(_TRANSITIONS, max_size=20))
@settings(max_examples=100, deadline=None)
def test_metric_computation_determinism(
    transitions: list[NodeTransition],
) -> None:
    """Property 6: identical sequences yield identical metrics on every call."""
    first = _tracked(transitions).compute_metrics()
    second = _tracked(transitions).compute_metrics()

    assert first == second

    tracker = _tracked(transitions)
    assert tracker.compute_metrics() == tracker.compute_metrics()


# ---------------------------------------------------------------------------
# Unit tests: formulas and zero-division edges
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_computes_documented_formulas() -> None:
    transitions = [
        NodeTransition(
            node_name="researcher",
            timestamp=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
            input_state={},
            output_state={
                "retrieved_document_ids": ["doc-1", "doc-2"],
                "follow_up_queries": ["q-follow-1", "q-follow-2"],
                "primary_query_count": 2,
            },
            tool_calls=["search", "retrieve"],
        ),
        NodeTransition(
            node_name="reporter",
            timestamp=datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC),
            input_state={},
            output_state={
                "retrieved_document_ids": ["doc-2", "doc-3"],
                "follow_up_queries": ["q-follow-2", "q-follow-3"],
                "report_word_count": 100,
            },
            tool_calls=["write"],
        ),
    ]

    metrics = _tracked(transitions).compute_metrics()

    assert metrics.brief_to_report_coverage_ratio == pytest.approx(3 / 100)
    assert metrics.follow_up_depth == pytest.approx(3 / 2)
    assert metrics.breadth_vs_depth_ratio == pytest.approx((2 + 2) / 2)
    assert metrics.total_node_transitions == 2
    assert metrics.total_tool_calls == 3
    assert metrics.total_duration_seconds == pytest.approx(30.0)


@pytest.mark.unit
def test_zero_division_edges_return_zero() -> None:
    transition = NodeTransition(
        node_name="researcher",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        input_state={},
        output_state={"report_word_count": 0, "primary_query_count": 0},
        tool_calls=[],
    )

    metrics = _tracked([transition]).compute_metrics()

    assert metrics.brief_to_report_coverage_ratio == pytest.approx(0.0)
    assert metrics.follow_up_depth == pytest.approx(0.0)
    assert metrics.breadth_vs_depth_ratio == pytest.approx(0.0)


@pytest.mark.unit
def test_empty_transition_set_returns_zero_metrics() -> None:
    metrics = _tracked([]).compute_metrics()

    assert metrics.brief_to_report_coverage_ratio == pytest.approx(0.0)
    assert metrics.follow_up_depth == pytest.approx(0.0)
    assert metrics.breadth_vs_depth_ratio == pytest.approx(0.0)
    assert metrics.total_node_transitions == 0
    assert metrics.total_tool_calls == 0
    assert metrics.total_duration_seconds == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Unit tests: start/record ordering
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_before_start_raises() -> None:
    with pytest.raises(ProcessMetricsError, match="compute_metrics"):
        ProcessMetricsTracker().compute_metrics()


@pytest.mark.unit
def test_record_before_start_raises() -> None:
    transition = NodeTransition("supervisor", datetime(2026, 1, 1, tzinfo=UTC), {}, {}, [])
    with pytest.raises(ProcessMetricsError, match="record_node_transition"):
        ProcessMetricsTracker().record_node_transition(transition)


@pytest.mark.unit
def test_node_limit_exceeded_raises() -> None:
    tracker = ProcessMetricsTracker()
    tracker.start_tracking(initial_query="initial", node_limit=1)
    tracker.record_node_transition(
        NodeTransition("supervisor", datetime(2026, 1, 1, tzinfo=UTC), {}, {}, [])
    )

    with pytest.raises(ProcessMetricsError, match="node_limit exceeded"):
        tracker.record_node_transition(
            NodeTransition("researcher", datetime(2026, 1, 1, tzinfo=UTC), {}, {}, [])
        )


@pytest.mark.unit
def test_negative_node_limit_raises() -> None:
    with pytest.raises(ProcessMetricsError, match="non-negative"):
        ProcessMetricsTracker().start_tracking(initial_query="initial", node_limit=-1)


@pytest.mark.unit
def test_start_tracking_resets_state() -> None:
    tracker = _tracked([NodeTransition("supervisor", datetime(2026, 1, 1, tzinfo=UTC), {}, {}, [])])
    tracker.start_tracking(initial_query="fresh", node_limit=5)

    assert tracker.node_count == 0
    assert tracker.compute_metrics().total_node_transitions == 0


@pytest.mark.unit
def test_build_snapshot_uses_recorded_window() -> None:
    tracker = _tracked(
        [
            NodeTransition("supervisor", datetime(2026, 1, 1, 0, 0, tzinfo=UTC), {}, {}, []),
            NodeTransition(
                "reporter",
                datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                {},
                {"report_word_count": 10},
                ["write"],
            ),
        ]
    )

    snapshot = tracker.build_snapshot(operation_id="op-1")

    assert snapshot.operation_id == "op-1"
    assert snapshot.initial_query == "initial"
    assert snapshot.final_report_word_count == 10
    assert snapshot.total_iterations == 2
    assert snapshot.unique_documents_per_iteration == [0, 0]
    assert snapshot.end_time == datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
