"""Portfolio traversal contracts, the relationship-edge binding gate (D2), and
the three traversal tools (tasks 2.1-2.4 of openspec change
``agent-portfolio-traversal-tools``).

Design decisions (openspec design.md):

1. Traversal and content are different tools' jobs — envelopes carry
   identifiers, provenance, and evidence handles only; ``read_evidence_by_id``
   stays the sole content materializer. No clause text exists on any match.
2. The gate is a binding-time check, not a runtime condition — binding a
   portfolio tool while the corpus lacks typed relationship edges fails the
   graph build with an error naming the gate, instead of shipping tools that
   return empty successes. The tools still probe the gate at runtime as
   defense in depth (a corpus that lost its edges is refused, never answered
   with an empty success).
3. Scope is server-owned — the model-visible argument schemas carry only
   query arguments (party names, clause class, conflict notion); tenant and
   document scope come exclusively from the injected ``ToolRuntimeContext``
   via :func:`trusted_tool_context`, and a traversal never crosses the
   authorized allowlist no matter what the arguments name.
4. Honest empty states — a query the graph cannot answer returns the typed
   ``no_results`` status, budget exhaustion (the per-tool match bound) returns
   ``budget_exhausted`` with the partial matches, and an authorization or gate
   failure returns a ``refused`` envelope. Nothing returns an empty success.
   A read that reached the upstream cap is never reported as ``no_results``:
   the envelope discloses the truncation through coverage (``truncated``) so
   a capped prefix of the scope is never presented as a whole-scope answer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, ClassVar

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.retrieval.graphiti.schemas import RelationshipType

from .authorization import trusted_tool_context

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.tools.base import BaseTool

    from app.retrieval.edge_expansion import LegalEdgeRecord
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

    ``evidence_locator`` is a deterministic server-authored clause locator —
    deliberately NOT an ``EvidenceService`` handle token (handles are issued
    only from fused retrieval; see the change design addendum). It names the
    cited clause for the run's scoped evidence read path; raw clause text
    never rides the envelope.
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
    evidence_locator: str


class TraversalCoverage(BaseModel):
    """Coverage disclosure, including a truncated-run gap when one exists.

    ``truncated`` is True when the upstream read reached
    ``_TRAVERSAL_READ_CAP``: the traversal then filtered a capped prefix of
    the scope, so matches_found is a floor and a ``no_results`` verdict was
    never earned.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents_in_scope: int = Field(default=0, ge=0)
    edges_scanned: int = Field(default=0, ge=0)
    matches_found: int = Field(default=0, ge=0)
    matches_returned: int = Field(default=0, ge=0)
    truncated: bool = False


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
    strict: bool = False,
) -> RelationshipEdgeProbe:
    """Probe whether the authorized corpus has typed relationship edges.

    Read-only reuse of the Graphiti edge read (``fetch_relationship_edges``)
    at existence granularity — one edge is enough to open the gate. The probe
    reports the corpus state it observed; it never binds or enables anything.

    Verdicts: ``edges_present`` when at least one typed relationship edge was
    observed, ``no_scope`` when the fail-closed empty allowlist leaves nothing
    to probe, ``no_edges`` when the read completed and observed none.

    ``strict`` controls backend failures: the underlying read degrades
    transport errors and timeouts to an empty result (the expansion path's
    never-fail-an-ask rule). Leniently (the default, used by the per-request
    runtime probe) a degraded read reports ``no_edges`` — honest as a
    refusal, because a closed gate refuses. Strictly (used by the
    binding-time probe) a backend failure is re-raised so the caller can
    distinguish an unevaluable probe from a genuinely completed empty
    corpus; a binding decision must never be made on a read that never ran.
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
        strict=strict,
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


# ---------------------------------------------------------------------------
# Traversal tools (tasks 2.1-2.4)
# ---------------------------------------------------------------------------

#: Pre-envelope refusal status (same shape convention as deep research): a
#: refusal is not a traversal outcome, so it never reuses TraversalStatus.
TRAVERSAL_REFUSED_STATUS = "refused"

#: Relationship vocabulary that expresses a duty binding a party. The conflict
#: finder and the party web instead filter on model-chosen / full vocabulary.
_OBLIGATION_RELATIONSHIPS = frozenset(
    {
        RelationshipType.OBLIGES.value,
        RelationshipType.RESTRICTS.value,
        RelationshipType.INDEMNIFIES.value,
    }
)

#: Upstream read cap for one traversal. The envelope bound (20/20/25) is the
#: model-visible budget; the read is capped above it so truncation is detected
#: honestly and disclosed through coverage instead of silently clipping.
_TRAVERSAL_READ_CAP = 200

#: Evidence-locator prefix. Traversal never returns clause text: the handle is
#: a deterministic server-authored locator naming the cited clause; content is
#: materialized only through the evidence seam (design Decision 1).
_EVIDENCE_LOCATOR_PREFIX = "portfolio"


def _evidence_locator(edge: LegalEdgeRecord) -> str:
    """Deterministic evidence locator for one traversal match.

    Format ``portfolio:<document_id>:<clause_id>:<edge_id>`` — enough to
    resolve the cited clause through the evidence seam's scoped read path
    without ever carrying clause text on the envelope. Deliberately not an
    ``EvidenceService`` handle token: handles are issued only from fused
    retrieval (chunk identity), and a traversal match carries clause
    identity — see the change design addendum's recorded deviation.
    """
    return f"{_EVIDENCE_LOCATOR_PREFIX}:{edge.document_id}:{edge.clause_id}:{edge.edge_id}"


def _refusal(message: str) -> dict[str, object]:
    """Typed refusal envelope: same keys as a traversal envelope dump."""
    return {
        "status": TRAVERSAL_REFUSED_STATUS,
        "matches": [],
        "coverage": {},
        "message": message,
    }


def _involves_party(edge: LegalEdgeRecord, party: str) -> bool:
    """Case-insensitive exact endpoint match (parties are named, not searched)."""
    wanted = party.casefold()
    return wanted in {edge.from_entity.casefold(), edge.to_entity.casefold()}


class CrossDocumentObligationsInput(BaseModel):
    """Model-visible arguments for ``cross_document_obligations``.

    Deliberately ``extra="ignore"`` (same precedent as ``WebSearchInput``):
    ToolNode injects the ``InjectedToolArg``-marked ``runtime`` kwarg into
    the call args before validating them against this schema, so forbidding
    extras rejects every ToolNode invocation. Tenant and document scope are
    server-injected properties, never model arguments — a scope-like extra
    key the model supplies is ignored, never honored (spec: Scope is
    server-owned).
    """

    model_config = ConfigDict(extra="ignore")

    party: str = Field(
        min_length=2,
        description=(
            "Exact party name as written in the agreements (case-insensitive), e.g. 'Acme Corp'."
        ),
    )
    clause_class: str | None = Field(
        default=None,
        description=(
            "Optional clause-class filter, matched case-insensitively inside "
            "the clause identifier (e.g. 'indemn' for indemnity clauses)."
        ),
    )


class FindConflictingClausesInput(BaseModel):
    """Model-visible arguments for ``find_conflicting_clauses``.

    ``extra="ignore"`` for the ToolNode-injected ``runtime`` (see
    ``CrossDocumentObligationsInput``); scope-like extras are ignored,
    never honored.
    """

    model_config = ConfigDict(extra="ignore")

    conflict_notion: str = Field(
        description=(
            "Relationship type expressing the conflict, from the graph's closed "
            f"vocabulary ({', '.join(t.value for t in RelationshipType)}); "
            "e.g. 'overridden_by' or 'supersedes'."
        ),
    )


class PartyObligationMapInput(BaseModel):
    """Model-visible arguments for ``party_obligation_map``.

    ``extra="ignore"`` for the ToolNode-injected ``runtime`` (see
    ``CrossDocumentObligationsInput``); scope-like extras are ignored,
    never honored.
    """

    model_config = ConfigDict(extra="ignore")

    parties: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
        description=(
            "One to eight exact party names as written in the agreements "
            "(case-insensitive); the map covers the web of typed edges "
            "touching any of them."
        ),
    )


async def _traverse(
    graphiti_service: BoundGraphitiService | None,
    runtime: ToolRuntime | None,
    *,
    probe_query: str,
    filter_fn: Callable[[LegalEdgeRecord], bool],
    envelope_cls: type[_TraversalEnvelope],
    empty_description: str,
) -> dict[str, object]:
    """Shared traversal runner: authorize, probe the gate, read, bound, envelope.

    Honest states (task 2.4): a query the graph cannot answer is
    ``no_results``; more matches than the envelope bound is
    ``budget_exhausted`` carrying the truncated prefix with the gap disclosed
    in coverage; authorization, scope, backend, and gate failures are
    ``refused`` envelopes. Nothing returns an empty success.
    """
    try:
        authz = trusted_tool_context(runtime)
        authz.require_operation("document.read")
    except PermissionError as exc:
        return _refusal(str(exc))
    if not authz.allowed_doc_ids:
        return _refusal("portfolio traversal has no authorized document scope in the tool runtime")
    if graphiti_service is None:
        return _refusal(
            "portfolio traversal requires a provisioned graph backend; "
            "none is bound in this process"
        )

    # Runtime gate probe (defense in depth behind the binding-time check): a
    # corpus that lost its typed edges is refused, never answered as empty.
    probe = await relationship_edges_available(
        graphiti_service,
        allowed_doc_ids=authz.allowed_doc_ids,
        tenant_id=authz.tenant_id,
        document_versions=authz.document_versions,
        document_contracts=authz.document_contracts,
    )
    if probe.verdict is not RelationshipEdgeVerdict.EDGES_PRESENT:
        return _refusal(
            f"portfolio traversal gate '{TRAVERSAL_GATE_NAME}' is closed for this "
            f"corpus (verdict {probe.verdict.value}); traversal refused"
        )

    edges = await graphiti_service.fetch_relationship_edges(
        probe_query,
        allowed_doc_ids=authz.allowed_doc_ids,
        tenant_id=authz.tenant_id,
        document_versions=authz.document_versions,
        document_contracts=authz.document_contracts,
        num_results=_TRAVERSAL_READ_CAP,
    )
    # Honest truncation (review finding C): the read is identity-scoped, not
    # query-ranked — it returns a capped prefix of the scope and offers no
    # server-side party/clause filter or paging, so the traversal filters
    # that prefix client-side. When the read reached the cap the whole scope
    # was NOT searched, and the envelope must say so: coverage.truncated
    # marks it, and a capped empty read is budget_exhausted (unevaluable),
    # never a no_results that only an uncapped empty read earns.
    read_hit_cap = len(edges) >= _TRAVERSAL_READ_CAP
    # Defense in depth over the read's own scoping (same idiom as the evidence
    # service): a match naming a document outside the allowlist never ships.
    hits = sorted(
        (edge for edge in edges if edge.document_id in authz.allowed_doc_ids and filter_fn(edge)),
        key=lambda edge: (edge.document_id, edge.clause_id, edge.edge_id),
    )
    bound = envelope_cls.max_matches
    status = TraversalStatus.COMPLETED
    truncated = read_hit_cap
    message: str | None = None
    if not hits:
        if read_hit_cap:
            status = TraversalStatus.BUDGET_EXHAUSTED
            message = (
                f"The traversal read reached its cap ({_TRAVERSAL_READ_CAP} edge(s)) "
                f"across {len(authz.allowed_doc_ids)} in-scope document(s) before any "
                f"match for {empty_description} was found; the scope was not fully "
                "searched, so this is an unevaluable partial result, not a no-result "
                "state. Narrow the query and retry."
            )
        else:
            status = TraversalStatus.NO_RESULTS
            message = (
                f"No typed relationship edges matched {empty_description} across "
                f"{len(authz.allowed_doc_ids)} in-scope document(s) "
                f"({len(edges)} edge(s) scanned); this is a no-result state, "
                "not a partial answer."
            )
    elif len(hits) > bound:
        status = TraversalStatus.BUDGET_EXHAUSTED
        truncated = True
        message = (
            f"Traversal budget exhausted: {len(hits)} matching edge(s) found, "
            f"returning the first {bound}; coverage discloses the gap and "
            "narrowing the query will reach the remainder."
        )
        if read_hit_cap:
            message += (
                f" The read also reached its cap ({_TRAVERSAL_READ_CAP} edge(s)), "
                "so the scope was not fully searched."
            )
    elif read_hit_cap:
        # Under the bound but the read was capped: completed, with the
        # prefix-of-scope disclosure carried in the message and coverage.
        message = (
            f"The traversal read reached its cap ({_TRAVERSAL_READ_CAP} edge(s)); "
            "the matches below cover only the scanned prefix of the scope — "
            "later matches may exist. Narrow the query for complete coverage."
        )
    matches = tuple(
        TraversalMatch(
            edge_id=edge.edge_id,
            clause_id=edge.clause_id,
            from_party=edge.from_entity,
            to_party=edge.to_entity,
            relationship=edge.relationship,
            document_id=edge.document_id,
            contract_index=edge.contract_index,
            document_version=edge.document_version,
            evidence_locator=_evidence_locator(edge),
        )
        for edge in hits[:bound]
    )
    envelope = envelope_cls(
        status=status,
        matches=matches,
        coverage=TraversalCoverage(
            documents_in_scope=len(authz.allowed_doc_ids),
            edges_scanned=len(edges),
            matches_found=len(hits),
            matches_returned=len(matches),
            truncated=truncated,
        ),
        message=message,
    )
    return envelope.model_dump(mode="json")


def make_cross_document_obligations_tool(
    graphiti_service: BoundGraphitiService | None,
) -> BaseTool:
    """Task 2.1: one party's obligations across every in-scope agreement."""

    @tool(args_schema=CrossDocumentObligationsInput)
    async def cross_document_obligations(
        party: str,
        clause_class: str | None = None,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> dict[str, object]:
        """Find one party's obligations across every agreement in the authorized scope.

        Returns obligation-class relationships (obliges, restricts,
        indemnifies) touching the named party, with document provenance and
        evidence locators — never clause text. Read a locator's content
        through ``read_evidence_by_id``.
        """

        def _matches(edge: LegalEdgeRecord) -> bool:
            if edge.relationship.casefold() not in _OBLIGATION_RELATIONSHIPS:
                return False
            if not _involves_party(edge, party):
                return False
            return clause_class is None or clause_class.casefold() in edge.clause_id.casefold()

        return await _traverse(
            graphiti_service,
            runtime,
            probe_query=f"cross-document obligations for {party}",
            filter_fn=_matches,
            envelope_cls=CrossDocumentObligationsResult,
            empty_description=f"party '{party}'",
        )

    return cross_document_obligations  # type: ignore[no-any-return]


def make_find_conflicting_clauses_tool(
    graphiti_service: BoundGraphitiService | None,
) -> BaseTool:
    """Task 2.2: clauses joined by a conflicting relationship type, in scope."""

    @tool(args_schema=FindConflictingClausesInput)
    async def find_conflicting_clauses(
        conflict_notion: str,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> dict[str, object]:
        """Find clauses related by a conflicting relationship type across the authorized scope.

        ``conflict_notion`` is a relationship type from the graph's closed
        vocabulary (e.g. ``overridden_by`` when one clause overrides another,
        ``supersedes`` for amendment chains). Returns both sides of every
        matching edge with provenance and evidence locators — never clause
        text.
        """
        notion = conflict_notion.strip().casefold()
        try:
            relationship = RelationshipType(notion).value
        except ValueError:
            return _refusal(
                f"unknown conflict notion {conflict_notion!r}; the graph's closed "
                f"relationship vocabulary is: "
                f"{', '.join(t.value for t in RelationshipType)}"
            )
        return await _traverse(
            graphiti_service,
            runtime,
            probe_query=f"conflicting clauses via {relationship}",
            filter_fn=lambda edge: edge.relationship.casefold() == relationship,
            envelope_cls=ConflictingClausesResult,
            empty_description=f"conflict notion '{relationship}'",
        )

    return find_conflicting_clauses  # type: ignore[no-any-return]


def make_party_obligation_map_tool(
    graphiti_service: BoundGraphitiService | None,
) -> BaseTool:
    """Task 2.3: the party→obligation web for a named set of parties."""

    @tool(args_schema=PartyObligationMapInput)
    async def party_obligation_map(
        parties: tuple[str, ...],
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> dict[str, object]:
        """Map the relationship web connecting named parties across the authorized scope.

        Unlike ``cross_document_obligations`` (one party, obligation-class
        edges), this returns every typed edge touching any of the named
        parties — the full web, with provenance and evidence locators, never
        clause text.
        """
        wanted = {party.strip().casefold() for party in parties}

        def _matches(edge: LegalEdgeRecord) -> bool:
            return bool(wanted) and bool(
                wanted & {edge.from_entity.casefold(), edge.to_entity.casefold()}
            )

        return await _traverse(
            graphiti_service,
            runtime,
            probe_query=f"party obligation web for {', '.join(sorted(wanted))}",
            filter_fn=_matches,
            envelope_cls=PartyObligationMapResult,
            empty_description=f"parties {sorted(wanted)}",
        )

    return party_obligation_map  # type: ignore[no-any-return]


def make_portfolio_traversal_tools(
    graphiti_service: BoundGraphitiService | None,
) -> tuple[BaseTool, BaseTool, BaseTool]:
    """Build the three traversal tools against one (possibly unprovisioned) backend.

    ``None`` builds runtime-refusing tools so the catalog can be populated
    before a graph backend exists; the graph factory re-registers
    backend-bound instances when the binding-time gate passes. Order is
    fixed: cross-document obligations, conflicting clauses, party map.
    """
    return (
        make_cross_document_obligations_tool(graphiti_service),
        make_find_conflicting_clauses_tool(graphiti_service),
        make_party_obligation_map_tool(graphiti_service),
    )
