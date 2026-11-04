"""Tests for taxonomy candidate generation from ticket clusters."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.ai.evaluation.signal_ingestion import (
    CandidateGenerationError,
    SupportTicket,
    TaxonomyCandidateGenerator,
)

pytestmark = pytest.mark.unit

_FIXTURE_DIR = Path(__file__).resolve().parents[4] / "evals" / "fixtures" / "tickets"


def _load_tickets(name: str) -> list[SupportTicket]:
    payload = json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return cast("list[SupportTicket]", payload)


def _iso(offset_days: int) -> str:
    return (
        (datetime(2024, 3, 1, tzinfo=UTC) + timedelta(days=offset_days))
        .isoformat()
        .replace("+00:00", "Z")
    )


def _ticket(
    ticket_id: str,
    created_at: str,
    *,
    symptom: str = "Cannot locate indemnification provisions in MSA",
    category: str = "retrieval_failure",
) -> SupportTicket:
    return SupportTicket(
        ticket_id=ticket_id,
        customer_id=f"customer-{ticket_id}",
        issue_category=category,
        symptom_description=symptom,
        resolution_note=None,
        created_at=created_at,
        resolved_at=None,
    )


def _window_holds(
    tickets: list[SupportTicket],
    *,
    threshold: int,
    window_days: int,
) -> bool:
    stamps = sorted(datetime.fromisoformat(ticket["created_at"]) for ticket in tickets)
    for index, start in enumerate(stamps):
        count = sum(1 for stamp in stamps[index:] if (stamp - start).days <= window_days)
        if count >= threshold:
            return True
    return False


@pytest.mark.property
@given(
    day_offsets=st.lists(
        st.integers(min_value=0, max_value=60),
        min_size=1,
        max_size=10,
    ),
    threshold=st.integers(min_value=1, max_value=6),
    window_days=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=100)
def test_property_8_cluster_threshold_validation(
    day_offsets: list[int],
    threshold: int,
    window_days: int,
) -> None:
    """Feature: evaluation-infrastructure-extension, Property 8."""
    tickets = [_ticket(f"T{i}", _iso(offset)) for i, offset in enumerate(day_offsets)]
    if not tickets:
        return

    candidates = TaxonomyCandidateGenerator().generate_candidates(
        {"cluster-1": tickets},
        threshold=threshold,
        window_days=window_days,
    )

    assert bool(candidates) is _window_holds(
        tickets,
        threshold=threshold,
        window_days=window_days,
    )
    if candidates:
        assert len(candidates[0].source_ticket_ids) == len(tickets)


def test_sliding_window_accepts_five_within_thirty_days() -> None:
    tickets = [_ticket(f"T{i}", _iso(offset)) for i, offset in enumerate([0, 7, 14, 21, 29])]

    candidates = TaxonomyCandidateGenerator().generate_candidates(
        {"cluster-1": tickets}, threshold=5, window_days=30
    )

    assert len(candidates) == 1


def test_sliding_window_rejects_five_across_thirty_one_days() -> None:
    tickets = [_ticket(f"T{i}", _iso(offset)) for i, offset in enumerate([0, 7, 14, 21, 31])]

    candidates = TaxonomyCandidateGenerator().generate_candidates(
        {"cluster-1": tickets}, threshold=5, window_days=30
    )

    assert candidates == []


def test_threshold_boundary_is_inclusive() -> None:
    tickets = [_ticket(f"T{i}", _iso(i)) for i in range(4)]
    generator = TaxonomyCandidateGenerator()

    assert generator.generate_candidates({"cluster-1": tickets}, threshold=5) == []
    assert len(generator.generate_candidates({"cluster-1": tickets}, threshold=4)) == 1


def test_recommended_case_type_is_deterministic() -> None:
    tickets = _load_tickets("clustered_failures.json")
    generator = TaxonomyCandidateGenerator()

    first = generator.generate_candidates({"cluster-1": tickets})
    second = generator.generate_candidates({"cluster-1": tickets})

    assert first == second
    assert first[0].recommended_case_type == "unanswerable"
    assert first[0].query_template.endswith("?")
    assert first[0].source_ticket_ids == [ticket["ticket_id"] for ticket in tickets]


def test_ingestion_failure_recommends_ingestion_case_type() -> None:
    tickets = [
        _ticket(
            f"T{i}",
            _iso(i),
            category="ingestion_failure",
            symptom="Document was not ingested after upload",
        )
        for i in range(5)
    ]

    candidates = TaxonomyCandidateGenerator().generate_candidates({"cluster-1": tickets})

    assert candidates[0].recommended_case_type == "no_evidence_vs_ingestion_failure"


def test_record_starts_pending() -> None:
    tickets = _load_tickets("clustered_failures.json")
    generator = TaxonomyCandidateGenerator()
    candidate = generator.generate_candidates({"cluster-1": tickets})[0]
    record = generator.to_record(
        candidate,
        cluster_id="cluster-1",
        created_at="2024-04-01T00:00:00Z",
    )

    assert record.approval_status == "pending"
    assert record.approved_at is None
    assert record.source_cluster_id == "cluster-1"
    assert record.query_template == candidate.query_template


@pytest.mark.parametrize(
    ("threshold", "window_days"),
    [(0, 30), (-1, 30), (5, 0), (5, -1)],
)
def test_invalid_threshold_and_window_raise(threshold: int, window_days: int) -> None:
    tickets = _load_tickets("clustered_failures.json")

    with pytest.raises(CandidateGenerationError):
        TaxonomyCandidateGenerator().generate_candidates(
            {"cluster-1": tickets},
            threshold=threshold,
            window_days=window_days,
        )
