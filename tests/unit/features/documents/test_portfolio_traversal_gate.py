"""Unit tests for the portfolio traversal scaffolding (task 1.1 / 1.2).

Covers the gate probe verdicts, the binding-time gate failure naming the
gate, envelope bounds, honest statuses, and the server-owned-scope shape of
the envelopes (no model-supplied scope path exists on the models at all).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from app.features.documents.application.tools.portfolio import (
    GATE_PROBE_QUERY,
    MAX_CONFLICTING_CLAUSE_MATCHES,
    MAX_CROSS_DOCUMENT_OBLIGATION_MATCHES,
    MAX_PARTY_OBLIGATION_MAP_MATCHES,
    TRAVERSAL_GATE_NAME,
    ConflictingClausesResult,
    CrossDocumentObligationsResult,
    PartyObligationMapResult,
    RelationshipEdgeProbe,
    RelationshipEdgeVerdict,
    TraversalCoverage,
    TraversalGateError,
    TraversalMatch,
    TraversalStatus,
    relationship_edges_available,
    require_traversal_gate,
)
from app.retrieval.edge_expansion import LegalEdgeRecord

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def _edge(edge_id: str = "edge-1") -> LegalEdgeRecord:
    return LegalEdgeRecord(
        edge_id=edge_id,
        document_id="doc-a",
        from_entity="Acme Corp",
        relationship="obliges",
        to_entity="GlobalTech Ltd",
        clause_id="2.1",
        char_start=1042,
        char_end=1110,
    )


def _match(index: int) -> TraversalMatch:
    return TraversalMatch(
        edge_id=f"edge-{index}",
        clause_id=f"C-{index:03d}",
        from_party="Acme Corp",
        to_party="GlobalTech Ltd",
        relationship="obliges",
        document_id="doc-a",
        evidence_handle=f"ev-{index}",
    )


class _StubEdgeService:
    """Test double for the read-only edge fetch the probe reuses."""

    def __init__(self, edges: tuple[LegalEdgeRecord, ...]) -> None:
        self.edges = edges
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def fetch_relationship_edges(
        self,
        query: str,
        *,
        allowed_doc_ids: tuple[str, ...] = (),
        tenant_id: str | None = None,
        contract_index: int | None = None,
        document_versions: Sequence[tuple[str, int, str | None]] = (),
        document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
        num_results: int = 32,
    ) -> tuple[LegalEdgeRecord, ...]:
        self.calls.append(
            (
                query,
                {
                    "allowed_doc_ids": allowed_doc_ids,
                    "tenant_id": tenant_id,
                    "num_results": num_results,
                },
            )
        )
        return self.edges


async def test_probe_reports_edges_present() -> None:
    service = _StubEdgeService((_edge(),))
    probe = await relationship_edges_available(
        service,  # type: ignore[arg-type]
        allowed_doc_ids=("doc-a", "doc-b"),
        tenant_id="tenant-1",
    )
    assert probe.verdict is RelationshipEdgeVerdict.EDGES_PRESENT
    assert probe.edges_observed == 1
    assert probe.documents_probed == 2
    # Existence probe: the read is identity-scoped, so one edge suffices and
    # the probe query names its purpose.
    assert service.calls[0][0] == GATE_PROBE_QUERY
    assert service.calls[0][1]["allowed_doc_ids"] == ("doc-a", "doc-b")
    assert service.calls[0][1]["num_results"] == 1


async def test_probe_reports_no_edges_when_read_comes_back_empty() -> None:
    service = _StubEdgeService(())
    probe = await relationship_edges_available(
        service,  # type: ignore[arg-type]
        allowed_doc_ids=("doc-a",),
    )
    assert probe.verdict is RelationshipEdgeVerdict.NO_EDGES
    assert probe.edges_observed == 0
    assert probe.documents_probed == 1


async def test_probe_refuses_to_probe_an_empty_scope() -> None:
    service = _StubEdgeService((_edge(),))
    probe = await relationship_edges_available(
        service,  # type: ignore[arg-type]
        allowed_doc_ids=(),
    )
    # Fail-closed boundary: an empty allowlist is not a corpus verdict.
    assert probe.verdict is RelationshipEdgeVerdict.NO_SCOPE
    assert service.calls == []


def test_binding_gate_passes_only_on_edges_present() -> None:
    probe = RelationshipEdgeProbe(verdict=RelationshipEdgeVerdict.EDGES_PRESENT)
    assert require_traversal_gate(probe) is None


def test_binding_gate_failure_names_the_gate() -> None:
    probe = RelationshipEdgeProbe(
        verdict=RelationshipEdgeVerdict.NO_EDGES,
        edges_observed=0,
        documents_probed=2,
    )
    with pytest.raises(TraversalGateError) as exc_info:
        require_traversal_gate(probe)
    # Exact message text is pinned: the factory integration (task 2.x) and
    # the projection tests match on it.
    assert str(exc_info.value) == (
        "binding refused by the portfolio traversal gate 'relationship-edges': "
        "verdict no_edges (0 typed relationship edge(s) observed across 2 in-scope "
        "document(s)); portfolio traversal tools must not be bound until the corpus "
        "has typed relationship edges"
    )
    assert TRAVERSAL_GATE_NAME in str(exc_info.value)
    assert exc_info.value.verdict is RelationshipEdgeVerdict.NO_EDGES


def test_binding_gate_fails_closed_on_scopeless_probe() -> None:
    probe = RelationshipEdgeProbe(verdict=RelationshipEdgeVerdict.NO_SCOPE)
    with pytest.raises(TraversalGateError) as exc_info:
        require_traversal_gate(probe)
    assert "portfolio traversal gate 'relationship-edges'" in str(exc_info.value)
    assert "verdict no_scope" in str(exc_info.value)


@pytest.mark.parametrize(
    ("envelope_cls", "bound"),
    [
        (CrossDocumentObligationsResult, MAX_CROSS_DOCUMENT_OBLIGATION_MATCHES),
        (ConflictingClausesResult, MAX_CONFLICTING_CLAUSE_MATCHES),
        (PartyObligationMapResult, MAX_PARTY_OBLIGATION_MAP_MATCHES),
    ],
)
def test_envelope_rejects_matches_over_the_bound(
    envelope_cls: type[CrossDocumentObligationsResult]
    | type[ConflictingClausesResult]
    | type[PartyObligationMapResult],
    bound: int,
) -> None:
    # Pinned design: over-bound construction is *rejected*, never silently
    # truncated — the traversal implementation truncates upstream and
    # discloses the gap in coverage.
    with pytest.raises(ValidationError, match="accepts at most"):
        envelope_cls(
            status=TraversalStatus.COMPLETED,
            matches=tuple(_match(i) for i in range(bound + 1)),
            coverage=TraversalCoverage(matches_found=bound + 1, matches_returned=bound + 1),
        )
    at_bound = envelope_cls(
        status=TraversalStatus.COMPLETED,
        matches=tuple(_match(i) for i in range(bound)),
        coverage=TraversalCoverage(matches_found=bound, matches_returned=bound),
    )
    assert len(at_bound.matches) == bound


@pytest.mark.parametrize(
    "envelope_cls",
    [CrossDocumentObligationsResult, ConflictingClausesResult, PartyObligationMapResult],
)
def test_envelope_statuses_are_honest(envelope_cls: Any) -> None:
    # An empty success is the defect the gate exists to prevent: completed
    # with zero matches and no_results carrying matches are both invalid.
    with pytest.raises(ValidationError, match="completed requires at least one match"):
        envelope_cls(status=TraversalStatus.COMPLETED)
    with pytest.raises(ValidationError, match="no_results must not carry matches"):
        envelope_cls(
            status=TraversalStatus.NO_RESULTS,
            matches=(_match(1),),
        )
    honest_empty = envelope_cls(status=TraversalStatus.NO_RESULTS)
    assert honest_empty.matches == ()
    assert honest_empty.status is TraversalStatus.NO_RESULTS
    # Budget exhaustion may carry partial matches.
    partial = envelope_cls(
        status=TraversalStatus.BUDGET_EXHAUSTED,
        matches=(_match(1),),
        coverage=TraversalCoverage(matches_found=50, matches_returned=1),
    )
    assert partial.status is TraversalStatus.BUDGET_EXHAUSTED


@pytest.mark.parametrize(
    "envelope_cls",
    [CrossDocumentObligationsResult, ConflictingClausesResult, PartyObligationMapResult],
)
def test_envelopes_carry_no_model_supplied_scope_fields(envelope_cls: Any) -> None:
    # Scope is server-owned: the envelopes define no scope-like input fields
    # at all, and extra="forbid" makes injecting one a schema violation.
    scope_like_names = {
        "allowed_doc_ids",
        "tenant_id",
        "user_id",
        "run_id",
        "thread_id",
        "doc_ids",
        "document_ids",
        "document_scope",
        "scope",
    }
    field_names = set(envelope_cls.model_fields)
    assert not field_names & scope_like_names
    for field_name in ("status", "matches", "coverage", "message"):
        assert field_name in field_names
    match_fields = set(TraversalMatch.model_fields)
    coverage_fields = set(TraversalCoverage.model_fields)
    assert not match_fields & scope_like_names
    assert not coverage_fields & scope_like_names
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        envelope_cls(
            status=TraversalStatus.NO_RESULTS,
            allowed_doc_ids=["doc-outside-scope"],
            tenant_id="tenant-1",
        )
