"""Tests for support-ticket similarity scoring and clustering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.ai.evaluation.signal_ingestion import (
    ClusteringError,
    SignalIngestionError,
    SupportTicket,
    TicketCluster,
    TicketClusterer,
)

pytestmark = pytest.mark.unit

_FIXTURE_DIR = Path(__file__).resolve().parents[4] / "evals" / "fixtures" / "tickets"


def _load_tickets(name: str) -> list[SupportTicket]:
    payload = json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return cast("list[SupportTicket]", payload)


@st.composite
def _ticket_pair(draw: st.DrawFn) -> tuple[SupportTicket, SupportTicket]:
    def build() -> SupportTicket:
        return SupportTicket(
            ticket_id=draw(st.text(min_size=1, max_size=8)),
            customer_id=draw(st.text(min_size=1, max_size=8)),
            issue_category=draw(
                st.sampled_from(["retrieval_failure", "ingestion_failure", "other", ""])
            ),
            symptom_description=draw(st.text(max_size=60)),
            resolution_note=None,
            created_at="2024-01-01T00:00:00Z",
            resolved_at=None,
        )

    return build(), build()


@pytest.mark.property
@given(_ticket_pair())
@settings(max_examples=100)
def test_property_7_dedup_at_eighty_percent_similarity(
    pair: tuple[SupportTicket, SupportTicket],
) -> None:
    """Feature: evaluation-infrastructure-extension, Property 7: Deduplication.

    Tickets at or above 0.8 similarity share a cluster; below 0.8 they do not.
    """
    clusterer = TicketClusterer()
    left, right = pair

    similarity = clusterer.similarity(left, right)
    clusters = clusterer.cluster([left, right])

    assert 0.0 <= similarity <= 1.0
    if similarity >= 0.8:
        assert len(clusters) == 1
    else:
        assert len(clusters) == 2


def test_cluster_is_deterministic() -> None:
    tickets = _load_tickets("clustered_failures.json")
    clusterer = TicketClusterer()

    first = clusterer.cluster(tickets)
    second = clusterer.cluster(tickets)

    assert first == second
    assert list(first) == list(second)


def test_clustered_failures_form_one_cluster() -> None:
    tickets = _load_tickets("clustered_failures.json")

    clusters = TicketClusterer().cluster(tickets)

    assert list(clusters) == ["cluster-1"]
    assert [ticket["ticket_id"] for ticket in clusters["cluster-1"]] == [
        "TICKET-101",
        "TICKET-102",
        "TICKET-103",
        "TICKET-104",
        "TICKET-105",
        "TICKET-106",
    ]


def test_retrieval_failures_form_multiple_clusters() -> None:
    tickets = _load_tickets("retrieval_failures.json")

    clusters = TicketClusterer().cluster(tickets)

    assert len(clusters) > 1
    clustered_ids = [ticket["ticket_id"] for group in clusters.values() for ticket in group]
    assert sorted(clustered_ids) == sorted(ticket["ticket_id"] for ticket in tickets)


def test_compute_cluster_centroid_fields_and_date_ordering() -> None:
    tickets = _load_tickets("clustered_failures.json")

    centroid = TicketClusterer().compute_cluster_centroid(tickets)

    assert centroid["cluster_id"] == "cluster-1"
    assert centroid["ticket_ids"] == [ticket["ticket_id"] for ticket in tickets]
    assert centroid["failure_count"] == 6
    assert centroid["symptom_variations"] == [ticket["symptom_description"] for ticket in tickets]
    score = cast("float", centroid["centroid_similarity_score"])
    assert 0.8 <= score <= 1.0
    assert centroid["first_seen"] == "2024-03-01T09:00:00Z"
    assert centroid["last_seen"] == "2024-03-15T09:30:00Z"


def test_singleton_centroid_similarity_is_one() -> None:
    tickets = _load_tickets("clustered_failures.json")

    centroid = TicketClusterer().compute_cluster_centroid([tickets[0]])

    assert centroid["failure_count"] == 1
    assert centroid["centroid_similarity_score"] == pytest.approx(1.0)


def test_cluster_to_models_returns_validated_clusters() -> None:
    tickets = _load_tickets("clustered_failures.json")

    models = TicketClusterer().cluster_to_models(tickets)

    assert len(models) == 1
    model = models[0]
    assert isinstance(model, TicketCluster)
    assert model.cluster_id == "cluster-1"
    assert model.failure_count == 6
    assert model.first_seen == "2024-03-01T09:00:00Z"
    assert model.last_seen == "2024-03-15T09:30:00Z"


def test_empty_input_raises_clustering_error() -> None:
    with pytest.raises(ClusteringError) as exc_info:
        TicketClusterer().cluster([])

    assert isinstance(exc_info.value, SignalIngestionError)


def test_malformed_created_at_raises_clustering_error() -> None:
    tickets = _load_tickets("clustered_failures.json")
    original = tickets[0]
    malformed = SupportTicket(
        ticket_id=original["ticket_id"],
        customer_id=original["customer_id"],
        issue_category=original["issue_category"],
        symptom_description=original["symptom_description"],
        resolution_note=original["resolution_note"],
        created_at="not-a-date",
        resolved_at=original["resolved_at"],
    )

    with pytest.raises(ClusteringError, match="malformed created_at"):
        TicketClusterer().compute_cluster_centroid([malformed])
