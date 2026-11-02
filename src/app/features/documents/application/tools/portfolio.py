"""Portfolio traversal contracts and the relationship-edge binding gate (D2).

Scaffolding for the gated openspec change ``agent-portfolio-traversal-tools``:
the three traversal envelope models, the gate probe over the Graphiti
relationship-edge read path, and the binding-time gate check a future graph
factory integration calls before binding any portfolio tool.

Design decisions (openspec design.md):

1. Traversal and content are different tools' jobs — envelopes carry
   identifiers, provenance, and evidence handles only; ``read_evidence_by_id``
   stays the sole content materializer. No clause text exists on any match.
2. The gate is a binding-time check, not a runtime condition — binding a
   portfolio tool while the corpus lacks typed relationship edges fails the
   graph build with an error naming the gate, instead of shipping tools that
   return empty successes.
3. Scope is server-owned — the envelopes have no scope input fields at all
   (tenant, document ids, run ids). Model-visible query arguments and the
   injected runtime context are decided in the gated factory work (task 2.x),
   never here.

This module deliberately defines **no** ``@tool`` callables: scaffolding must
not make anything bindable. The integration is the gated work.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.retrieval.graphiti.client import BoundGraphitiService

#: Existence probe: the edge read is identity-scoped, not query-ranked, so any
#: stable query string works. It exists so the probe log names its purpose.
GATE_PROBE_QUERY = "portfolio traversal gate: typed relationship edge existence probe"

#: The gate name embedded in the binding-failure error. The factory
#: integration (task 2.x) and the projection-consistency tests match on this.
TRAVERSAL_GATE_NAME = "relationship-edges"

#: Per-envelope match bounds. The envelope *rejects* construction over the
#: bound (pinned choice, enforced by ``_TraversalEnvelope``); the traversal
#: implementation (task 2.x) truncates upstream and discloses the gap through
#: coverage (``matches_found`` vs ``matches_returned``).
MAX_CROSS_DOCUMENT_OBLIGATION_MATCHES = 20
MAX_CONFLICTING_CLAUSE_MATCHES = 20
MAX_PARTY_OBLIGATION_MAP_MATCHES = 25


class TraversalStatus(StrEnum):
    """Terminal envelope statuses — honest states, never empty successes.

    ``completed`` requires at least one match, ``no_results`` forbids any;
    ``budget_exhausted`` may carry partial matches.
    """

    COMPLETED = "completed"
    NO_RESULTS = "no_results"
    BUDGET_EXHAUSTED = "budget_exhausted"


class RelationshipEdgeVerdict(StrEnum):
    """Typed verdict of the gate probe over the corpus.

    ``EDGES_PRESENT`` is the only verdict under which binding is allowed;
    both other verdicts are fail-closed.
    """

    EDGES_PRESENT = "edges_present"
    NO_EDGES = "no_edges"
    NO_SCOPE = "no_scope"


class RelationshipEdgeProbe(BaseModel):
    """One gate probe's typed result: verdict plus what was actually read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: RelationshipEdgeVerdict
    edges_observed: int = Field(default=0, ge=0)
    documents_probed: int = Field(default=0, ge=0)


class TraversalMatch(BaseModel):
    """One structured traversal hit — identifiers and provenance, no content.

    ``evidence_handle`` is the reference resolvable through the run-scoped
    ``read_evidence_by_id`` contract; raw clause text never rides the
    envelope.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_id: str
    clause_id: str
    from_party: str
    to_party: str
    relationship: str
    document_id: str
    contract_index: int | None = Field(default=None, ge=0)
    document_version: int | None = Field(default=None, ge=1)
    evidence_handle: str


class TraversalCoverage(BaseModel):
    """Coverage disclosure, including a truncated-run gap when one exists."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents_in_scope: int = Field(default=0, ge=0)
    edges_scanned: int = Field(default=0, ge=0)
    matches_found: int = Field(default=0, ge=0)
    matches_returned: int = Field(default=0, ge=0)


class _TraversalEnvelope(BaseModel):
    """Shared envelope shape for the three traversal tools.

    Bounded by construction (``max_matches``, enforced as a rejection) and
    honest by construction (status/match consistency is a model invariant,
    so a query the graph cannot answer cannot be projected as an empty
    success).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_matches: ClassVar[int]

    status: TraversalStatus
    matches: tuple[TraversalMatch, ...] = ()
    coverage: TraversalCoverage = Field(default_factory=TraversalCoverage)
    message: str | None = None

    @field_validator("matches")
    @classmethod
    def _enforce_match_bound(cls, value: tuple[TraversalMatch, ...]) -> tuple[TraversalMatch, ...]:
        if len(value) > cls.max_matches:
            bound_message = (
                f"{cls.__name__} accepts at most {cls.max_matches} matches "
                f"(got {len(value)}); truncate upstream and disclose the gap in coverage"
            )
            raise ValueError(bound_message)
        return value

    @model_validator(mode="after")
    def _enforce_honest_status(self) -> _TraversalEnvelope:
        if self.status is TraversalStatus.COMPLETED and not self.matches:
            empty_success_message = (
                "completed requires at least one match; use no_results for an honest empty state"
            )
            raise ValueError(empty_success_message)
        if self.status is TraversalStatus.NO_RESULTS and self.matches:
            dishonest_empty_message = "no_results must not carry matches"
            raise ValueError(dishonest_empty_message)
        return self


class CrossDocumentObligationsResult(_TraversalEnvelope):
    """Envelope for ``cross_document_obligations`` (task 2.1)."""

    max_matches: ClassVar[int] = MAX_CROSS_DOCUMENT_OBLIGATION_MATCHES


class ConflictingClausesResult(_TraversalEnvelope):
    """Envelope for ``find_conflicting_clauses`` (task 2.2)."""

    max_matches: ClassVar[int] = MAX_CONFLICTING_CLAUSE_MATCHES


class PartyObligationMapResult(_TraversalEnvelope):
    """Envelope for ``party_obligation_map`` (task 2.3)."""

    max_matches: ClassVar[int] = MAX_PARTY_OBLIGATION_MAP_MATCHES


async def relationship_edges_available(
    graphiti_service: BoundGraphitiService,
    *,
    allowed_doc_ids: tuple[str, ...],
    tenant_id: str | None = None,
    contract_index: int | None = None,
    document_versions: Sequence[tuple[str, int, str | None]] = (),
    document_contracts: Sequence[tuple[str, int, Sequence[int | None], bool]] = (),
) -> RelationshipEdgeProbe:
    """Probe whether the authorized corpus has typed relationship edges.

    Read-only reuse of the Graphiti edge read (``fetch_relationship_edges``)
    at existence granularity — one edge is enough to open the gate. The probe
    reports the corpus state it observed; it never binds or enables anything.

    Verdicts: ``edges_present`` when at least one typed relationship edge was
    observed, ``no_scope`` when the fail-closed empty allowlist leaves nothing
    to probe, ``no_edges`` when the read completed and observed none (the
    underlying read degrades transport failures to an empty result, so this
    verdict means "no edges observable", reported as such).
    """
    if not allowed_doc_ids:
        return RelationshipEdgeProbe(verdict=RelationshipEdgeVerdict.NO_SCOPE)

    edges = await graphiti_service.fetch_relationship_edges(
        GATE_PROBE_QUERY,
        allowed_doc_ids=allowed_doc_ids,
        tenant_id=tenant_id,
        contract_index=contract_index,
        document_versions=document_versions,
        document_contracts=document_contracts,
        num_results=1,
    )
    if edges:
        return RelationshipEdgeProbe(
            verdict=RelationshipEdgeVerdict.EDGES_PRESENT,
            edges_observed=len(edges),
            documents_probed=len(allowed_doc_ids),
        )
    return RelationshipEdgeProbe(
        verdict=RelationshipEdgeVerdict.NO_EDGES,
        documents_probed=len(allowed_doc_ids),
    )


class TraversalGateError(RuntimeError):
    """Binding a portfolio traversal tool while the gate is closed.

    The message always names the gate (``TRAVERSAL_GATE_NAME``) so the
    factory failure is greppable and the projection tests can pin it.
    """

    def __init__(self, message: str, *, verdict: RelationshipEdgeVerdict) -> None:
        super().__init__(message)
        self.verdict = verdict


def require_traversal_gate(probe: RelationshipEdgeProbe) -> None:
    """Binding-time gate check: refuse to bind unless the corpus has edges.

    Called by the future graph factory before binding any portfolio tool.
    Only ``edges_present`` opens the gate; every other verdict fails loudly
    with an error naming the gate (fail-closed on ``no_scope`` too — an
    unscoped build cannot honestly claim the gate is open).
    """
    if probe.verdict is RelationshipEdgeVerdict.EDGES_PRESENT:
        return
    gate_message = (
        f"binding refused by the portfolio traversal gate '{TRAVERSAL_GATE_NAME}': "
        f"verdict {probe.verdict.value} ({probe.edges_observed} typed relationship "
        f"edge(s) observed across {probe.documents_probed} in-scope document(s)); "
        "portfolio traversal tools must not be bound until the corpus has "
        "typed relationship edges"
    )
    raise TraversalGateError(gate_message, verdict=probe.verdict)
