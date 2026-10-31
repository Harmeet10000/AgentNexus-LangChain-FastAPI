"""Second-pass typed relationship extraction over pass-1 clause extractions.

Pass 1 (``service.AsyncExtractionService`` with the ingestion knowledge
prompt) yields grounded clause extractions.  This module runs a *different
prompt over different input*: the pass-1 clause identities are fed back as a
server-derived inventory, and the model is constrained to emit typed
relationships grounded in character spans of the actual contract boundary text
(the two-pass strategy sketched in ``langextract_to_graph``).

Security contract (mirrors ``ingestion.enrichment.legal_metadata``):

- Document content is untrusted data.  It travels only inside the shared
  ``UNTRUSTED_DOCUMENT_BEGIN``/``UNTRUSTED_DOCUMENT_END`` region, after
  delimiter-literal neutralization, and the prompt carries the shared
  never-follow-document-instructions directive.
- The prompt description carries only server-derived data (clause ids,
  clause types, char offsets) — never raw document text.  Pass-1 clause
  texts are used exclusively for server-side endpoint validation.
- Every candidate edge is validated before it can be persisted: the char
  interval must resolve inside the boundary text, the relationship value must
  be in the closed ``RelationshipType`` vocabulary, the clause reference must
  resolve to a pass-1 clause identity, and both endpoints must occur in the
  pass-1 entity corpus.  Anything else is dropped and counted — never written.
- Server-owned identity: edges never carry tenant/document/group identity
  from content.  Identity is attached at the writer boundary from job
  identity only.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import TYPE_CHECKING

import langextract as lx
from pydantic import BaseModel, ConfigDict, Field

from app.features.agent_saul.domain.state import RelationshipType
from app.features.documents.ingestion.enrichment.legal_metadata import (
    UNTRUSTED_CONTENT_DIRECTIVE,
    UNTRUSTED_DOCUMENT_BEGIN,
    UNTRUSTED_DOCUMENT_END,
    neutralize_untrusted_delimiters,
)

from .service import (
    ExtractionBudget,
    ExtractionFailureCode,
    ExtractionRequest,
    ExtractionSucceeded,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

    from .service import AsyncExtractionService

RELATIONSHIP_EXTRACTION_CLASS: Final[str] = "relationship"
RELATIONSHIP_CONFIG_DISABLED_REASON: Final[str] = "relationship_pass_disabled_by_config"
RELATIONSHIP_BUDGET_EXHAUSTED_REASON: Final[str] = "relationship_budget_exhausted"
# The untrusted region wraps the document with one-line delimiters; pass-2
# char intervals are relative to the wrapped text and are shifted by this
# prefix before span validation against the boundary body.
_RELATIONSHIP_TEXT_PREFIX: Final[str] = f"{UNTRUSTED_DOCUMENT_BEGIN}\n"
_RELATIONSHIP_TEXT_SUFFIX: Final[str] = f"\n{UNTRUSTED_DOCUMENT_END}"
_MAX_ENTITY_CHARS: Final[int] = 256
_DEFAULT_CONFIDENCE: Final[float] = 0.5

_RELATIONSHIP_PROMPT_INSTRUCTION: Final[str] = (
    "Extract typed legal relationships between the parties, obligations, and "
    "clauses of the document. Emit one extraction per relationship with "
    f"extraction class '{RELATIONSHIP_EXTRACTION_CLASS}'. The extraction text "
    "must be the exact span of document text that states the relationship. "
    "Set these attributes: from_entity, relationship, to_entity, clause_id, "
    "confidence. The 'relationship' attribute must be exactly one of: "
    f"{', '.join(member.value for member in RelationshipType)}. The "
    "'clause_id' attribute must reference a clause_id from the server-provided "
    "clause inventory below. Both entities must be exact names as written in "
    "the document. " + UNTRUSTED_CONTENT_DIRECTIVE
)


class RelationshipPassConfig(BaseModel):
    """Configuration for the second-pass relationship extraction."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_model_calls: int = Field(default=8, ge=0)
    max_input_tokens: int = Field(default=50_000, ge=0)
    max_output_tokens: int = Field(default=25_000, ge=0)
    max_cost_usd: float = Field(default=2.5, ge=0)
    max_edges_per_boundary: int = Field(default=32, ge=1)


def relationship_pass_config_from_settings(settings: object) -> RelationshipPassConfig:
    """Read the relationship-pass knobs, falling back to typed defaults."""
    return RelationshipPassConfig(
        enabled=bool(getattr(settings, "LANGEXTRACT_RELATIONSHIP_PASS_ENABLED", True)),
        max_model_calls=int(getattr(settings, "LANGEXTRACT_RELATIONSHIP_MAX_MODEL_CALLS", 8)),
        max_input_tokens=int(
            getattr(settings, "LANGEXTRACT_RELATIONSHIP_MAX_INPUT_TOKENS", 50_000)
        ),
        max_output_tokens=int(
            getattr(settings, "LANGEXTRACT_RELATIONSHIP_MAX_OUTPUT_TOKENS", 25_000)
        ),
        max_cost_usd=float(getattr(settings, "LANGEXTRACT_RELATIONSHIP_MAX_COST_USD", 2.5)),
        max_edges_per_boundary=int(
            getattr(settings, "LANGEXTRACT_RELATIONSHIP_MAX_EDGES_PER_BOUNDARY", 32)
        ),
    )


class ClauseIdentity(BaseModel):
    """Server-derived identity of one pass-1 clause (never raw clause text)."""

    model_config = ConfigDict(frozen=True)

    clause_id: str
    clause_type: str
    section_ref: str
    char_start: int
    char_end: int


class GroundedRelationship(BaseModel):
    """One validated, span-grounded typed edge candidate.

    Carries everything the downstream writer needs except server identity:
    from/to entity text, the typed relationship, the pass-1 clause identity it
    came from, and the char interval grounding it in the boundary text.
    """

    model_config = ConfigDict(frozen=True)

    from_entity: str
    relationship: RelationshipType
    to_entity: str
    clause_id: str
    char_start: int
    char_end: int
    quote: str
    confidence: float = Field(ge=0.0, le=1.0)

    def dedup_key(self) -> tuple[str, str, str, str, int, int]:
        """Stable identity used to collapse duplicate candidates."""
        return (
            self.from_entity.casefold(),
            self.relationship.value,
            self.to_entity.casefold(),
            self.clause_id,
            self.char_start,
            self.char_end,
        )


def relationship_edge_id(
    *,
    document_id: str,
    contract_index: int,
    edge: GroundedRelationship,
) -> str:
    """Deterministic edge id: identical extractions must collide on retry.

    The idempotency guard keys relationship writes on this id, so a re-run of
    the same boundary produces the same key and skips already-written edges
    instead of duplicating them.
    """
    identity = (
        f"{document_id}\0{contract_index}\0{edge.clause_id}\0"
        f"{edge.from_entity}\0{edge.relationship.value}\0{edge.to_entity}\0"
        f"{edge.char_start}\0{edge.char_end}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


class RelationshipDropReason(StrEnum):
    """Closed vocabulary for why a candidate edge was dropped."""

    INVALID_SPAN = "invalid_span"
    INVALID_TYPE = "invalid_type"
    UNKNOWN_CLAUSE = "unknown_clause"
    INVALID_ENDPOINT = "invalid_endpoint"
    DUPLICATE = "duplicate"
    OVER_CAP = "over_cap"


class RelationshipDropCounts(BaseModel):
    """Typed per-reason drop tally; every drop is visible, never silent."""

    model_config = ConfigDict(frozen=True)

    invalid_span: int = Field(default=0, ge=0)
    invalid_type: int = Field(default=0, ge=0)
    unknown_clause: int = Field(default=0, ge=0)
    invalid_endpoint: int = Field(default=0, ge=0)
    duplicate: int = Field(default=0, ge=0)
    over_cap: int = Field(default=0, ge=0)

    def bump(self, reason: RelationshipDropReason) -> RelationshipDropCounts:
        updates = {reason.value: getattr(self, reason.value) + 1}
        return self.model_copy(update=updates)

    @property
    def total(self) -> int:
        return (
            self.invalid_span
            + self.invalid_type
            + self.unknown_clause
            + self.invalid_endpoint
            + self.duplicate
            + self.over_cap
        )


class RelationshipPassResult(BaseModel):
    """Outcome of one second-pass relationship extraction over a boundary.

    ``ran`` is True only when the provider was actually invoked.  A pass that
    degraded before the provider call carries a ``skip_reason``; a share
    exhaustion additionally sets ``budget_exhausted`` so the caller can emit
    the explicit ``not_attempted``/``relationship_budget_exhausted`` state.
    """

    model_config = ConfigDict(frozen=True)

    ran: bool = False
    skip_reason: str | None = None
    budget_exhausted: bool = False
    provider_error: str | None = None
    edges: tuple[GroundedRelationship, ...] = ()
    drops: RelationshipDropCounts = Field(default_factory=RelationshipDropCounts)
    budget: ExtractionBudget


def wrap_boundary_text(boundary_text: str) -> tuple[str, str, int]:
    """Neutralize delimiter literals and wrap the body in the untrusted region.

    Returns ``(body, wrapped_text, prefix_len)`` where ``body`` is the
    neutralized document text the pass-2 char intervals resolve against
    (they are relative to ``wrapped_text`` and shift by ``prefix_len``).
    """
    body, _neutralized_count = neutralize_untrusted_delimiters(boundary_text)
    wrapped = f"{_RELATIONSHIP_TEXT_PREFIX}{body}{_RELATIONSHIP_TEXT_SUFFIX}"
    return body, wrapped, len(_RELATIONSHIP_TEXT_PREFIX)


def build_relationship_request(
    *,
    boundary_text: str,
    inventory: Sequence[ClauseIdentity],
    budget: ExtractionBudget,
    document_id: str,
    tenant_id: str = "",
    document_version: int = 1,
    contract_index: int = 0,
    max_char_buffer: int,
    max_workers: int,
) -> ExtractionRequest:
    """Build the pass-2 request: delimited text channel, server-only prompt.

    The prompt description contains the closed vocabulary, the anti-instruction
    directive, and the server-derived clause inventory — never document text.
    The document itself travels only inside the delimited untrusted region of
    the text channel, so no document content exists outside the region in the
    rendered prompt.
    """
    _body, wrapped, _prefix_len = wrap_boundary_text(boundary_text)
    inventory_lines = "\n".join(
        f"- clause_id={entry.clause_id} type={entry.clause_type} "
        f"section_ref={entry.section_ref} chars={entry.char_start}-{entry.char_end}"
        for entry in inventory
    )
    prompt_description = (
        f"{_RELATIONSHIP_PROMPT_INSTRUCTION}\n\n"
        "Server-provided clause inventory (trusted server-derived data):\n"
        f"{inventory_lines}"
    )
    return ExtractionRequest(
        text=wrapped,
        prompt_description=prompt_description,
        examples=_relationship_examples(),
        # One pass: this is a different prompt over different input, not a
        # recall re-run of the same prompt (that is what extraction_passes
        # does for pass 1).
        extraction_passes=1,
        max_workers=max_workers,
        max_char_buffer=max_char_buffer,
        document_id=document_id,
        tenant_id=tenant_id,
        document_version=document_version,
        contract_index=contract_index,
        budget=budget,
    )


def _relationship_examples() -> tuple[lx.data.ExampleData, ...]:
    """Static server-authored few-shot examples (no user document content)."""
    example_text = (
        "The Employer shall pay the monthly charge of GBP 12,500 within "
        "fourteen (14) days of invoice."
    )
    return (
        lx.data.ExampleData(
            text=example_text,
            extractions=[
                lx.data.Extraction(
                    extraction_class=RELATIONSHIP_EXTRACTION_CLASS,
                    extraction_text=example_text,
                    attributes={
                        "from_entity": "the Employer",
                        "relationship": RelationshipType.OBLIGES.value,
                        "to_entity": "the monthly charge of GBP 12,500",
                        "clause_id": "2.1",
                        "confidence": "0.9",
                    },
                )
            ],
        ),
    )


def _relationship_share_exhausted(budget: ExtractionBudget) -> str | None:
    """Name the first exhausted relationship-share dimension, if any."""
    if budget.relationship_model_calls >= budget.relationship_max_model_calls:
        return "model_calls"
    if budget.relationship_input_tokens >= budget.relationship_max_input_tokens:
        return "input_tokens"
    if budget.relationship_output_tokens >= budget.relationship_max_output_tokens:
        return "output_tokens"
    if budget.relationship_cost_estimate_usd >= budget.relationship_max_cost_usd:
        return "cost"
    return None


def _relationship_share_budget(budget: ExtractionBudget) -> ExtractionBudget:
    """Shared ledger with maxima tightened to the relationship pool boundary.

    The relationship pass gets its own bounded share of model calls, tokens,
    and cost.  Tightening the call-scoped maxima means the shared budget
    cannot be driven past the pool boundary by this pass; the counters stay
    cumulative so the checkpoint ledger remains crash-truthful.
    """
    return budget.model_copy(
        update={
            "max_model_calls": budget.model_calls
            + max(0, budget.relationship_max_model_calls - budget.relationship_model_calls),
            "max_input_tokens": budget.input_tokens
            + max(0, budget.relationship_max_input_tokens - budget.relationship_input_tokens),
            "max_output_tokens": budget.output_tokens
            + max(0, budget.relationship_max_output_tokens - budget.relationship_output_tokens),
            "max_cost_usd": budget.cost_estimate_usd
            + max(0, budget.relationship_max_cost_usd - budget.relationship_cost_estimate_usd),
        }
    )


def _merge_relationship_charge(
    *, original: ExtractionBudget, returned: ExtractionBudget
) -> ExtractionBudget:
    """Restore shared maxima and accumulate the pass-2 charge into the pool.

    ``original`` is the pre-call shared ledger: its maxima were only ever
    tightened on the call-scoped copy, so they are restored here, and the
    counter deltas against it are exactly what this pass spent.
    """
    return returned.model_copy(
        update={
            "max_model_calls": original.max_model_calls,
            "max_input_tokens": original.max_input_tokens,
            "max_output_tokens": original.max_output_tokens,
            "max_cost_usd": original.max_cost_usd,
            "relationship_model_calls": original.relationship_model_calls
            + max(0, returned.model_calls - original.model_calls),
            "relationship_input_tokens": original.relationship_input_tokens
            + max(0, returned.input_tokens - original.input_tokens),
            "relationship_output_tokens": original.relationship_output_tokens
            + max(0, returned.output_tokens - original.output_tokens),
            "relationship_cost_estimate_usd": original.relationship_cost_estimate_usd
            + max(0.0, returned.cost_estimate_usd - original.cost_estimate_usd),
        }
    )


async def run_relationship_pass(
    *,
    extraction: AsyncExtractionService | None,
    boundary_text: str,
    inventory: Sequence[ClauseIdentity],
    entity_corpus: Sequence[str],
    budget: ExtractionBudget,
    config: RelationshipPassConfig,
    document_id: str,
    tenant_id: str = "",
    document_version: int = 1,
    contract_index: int = 0,
    max_char_buffer: int,
    max_workers: int,
) -> RelationshipPassResult:
    """Run the typed relationship pass for one contract boundary.

    Fails closed: every candidate edge is validated against the actual
    boundary text and the pass-1 identities; dropped candidates are counted
    per reason.  Budget share exhaustion degrades explicitly (``skip_reason``
    ``relationship_budget_exhausted``) instead of silently skipping.
    """
    if not config.enabled:
        return RelationshipPassResult(
            ran=False,
            skip_reason=RELATIONSHIP_CONFIG_DISABLED_REASON,
            budget=budget,
        )
    if not inventory:
        return RelationshipPassResult(ran=False, skip_reason="no_clause_inventory", budget=budget)
    if extraction is None:
        return RelationshipPassResult(
            ran=False, skip_reason="extraction_unconfigured", budget=budget
        )
    exhausted = _relationship_share_exhausted(budget)
    if exhausted is not None:
        return RelationshipPassResult(
            ran=False,
            skip_reason=RELATIONSHIP_BUDGET_EXHAUSTED_REASON,
            budget_exhausted=True,
            budget=budget,
        )
    share_budget = _relationship_share_budget(budget)
    request = build_relationship_request(
        boundary_text=boundary_text,
        inventory=inventory,
        budget=share_budget,
        document_id=document_id,
        tenant_id=tenant_id,
        document_version=document_version,
        contract_index=contract_index,
        max_char_buffer=max_char_buffer,
        max_workers=max_workers,
    )
    outcome = await extraction.extract(request)
    merged = _merge_relationship_charge(original=budget, returned=outcome.budget)
    if isinstance(outcome, ExtractionSucceeded):
        body, _wrapped, prefix_len = wrap_boundary_text(boundary_text)
        edges, drops = _validate_relationship_extractions(
            outcome.documents,
            boundary_body=body,
            prefix_len=prefix_len,
            inventory=inventory,
            entity_corpus=entity_corpus,
            max_edges=config.max_edges_per_boundary,
        )
        return RelationshipPassResult(ran=True, edges=edges, drops=drops, budget=merged)
    if outcome.code is ExtractionFailureCode.BUDGET_EXHAUSTED:
        return RelationshipPassResult(
            ran=False,
            skip_reason=RELATIONSHIP_BUDGET_EXHAUSTED_REASON,
            budget_exhausted=True,
            budget=merged,
        )
    return RelationshipPassResult(
        ran=True,
        provider_error=outcome.code.value,
        budget=merged,
    )


def _validate_relationship_extractions(
    documents: Sequence[lx.data.AnnotatedDocument],
    *,
    boundary_body: str,
    prefix_len: int,
    inventory: Sequence[ClauseIdentity],
    entity_corpus: Sequence[str],
    max_edges: int,
) -> tuple[tuple[GroundedRelationship, ...], RelationshipDropCounts]:
    """Fail-closed validation of pass-2 candidates; drops are counted."""
    drops = RelationshipDropCounts()
    edges: list[GroundedRelationship] = []
    seen: set[tuple[str, str, str, str, int, int]] = set()
    known_clause_ids = {entry.clause_id for entry in inventory}
    corpus = " \n".join(text.casefold() for text in entity_corpus if text)

    for document in documents:
        for extraction in document.extractions:
            if extraction.extraction_class.casefold() != RELATIONSHIP_EXTRACTION_CLASS:
                continue
            validated = _validated_edge(
                extraction,
                boundary_body=boundary_body,
                prefix_len=prefix_len,
                known_clause_ids=known_clause_ids,
                corpus=corpus,
            )
            if isinstance(validated, RelationshipDropReason):
                drops = drops.bump(validated)
                continue
            key = validated.dedup_key()
            if key in seen:
                drops = drops.bump(RelationshipDropReason.DUPLICATE)
                continue
            if len(edges) >= max_edges:
                drops = drops.bump(RelationshipDropReason.OVER_CAP)
                continue
            seen.add(key)
            edges.append(validated)
    return tuple(edges), drops


def _validated_edge(
    extraction: lx.data.Extraction,
    *,
    boundary_body: str,
    prefix_len: int,
    known_clause_ids: set[str],
    corpus: str,
) -> GroundedRelationship | RelationshipDropReason:
    """Validate one candidate against the boundary text and pass-1 identities."""
    interval = extraction.char_interval
    if interval is None or interval.start_pos is None or interval.end_pos is None:
        # Skip ungrounded extractions exactly as langextract_to_graph
        # does (char_interval is None) — and count the skip.
        return RelationshipDropReason.INVALID_SPAN
    start = interval.start_pos - prefix_len
    end = interval.end_pos - prefix_len
    if not 0 <= start < end <= len(boundary_body):
        return RelationshipDropReason.INVALID_SPAN
    attributes = extraction.attributes or {}
    try:
        relationship = RelationshipType(str(attributes.get("relationship", "")).strip().casefold())
    except ValueError:
        return RelationshipDropReason.INVALID_TYPE
    clause_id = str(attributes.get("clause_id", "")).strip()
    if clause_id not in known_clause_ids:
        return RelationshipDropReason.UNKNOWN_CLAUSE
    from_entity = str(attributes.get("from_entity", "")).strip()
    to_entity = str(attributes.get("to_entity", "")).strip()
    if not (_entity_grounded(from_entity, corpus) and _entity_grounded(to_entity, corpus)):
        return RelationshipDropReason.INVALID_ENDPOINT
    return GroundedRelationship(
        from_entity=from_entity,
        relationship=relationship,
        to_entity=to_entity,
        clause_id=clause_id,
        char_start=start,
        char_end=end,
        quote=boundary_body[start:end],
        confidence=_parse_confidence(attributes.get("confidence")),
    )


def _entity_grounded(entity: str, corpus: str) -> bool:
    """An endpoint must be non-empty, bounded, and occur in pass-1 entity text."""
    folded = entity.casefold()
    return bool(folded) and len(entity) <= _MAX_ENTITY_CHARS and folded in corpus


def _parse_confidence(raw: object) -> float:
    """Parse a model-supplied confidence, defaulting when absent/invalid."""
    try:
        value = float(str(raw))
    except (TypeError, ValueError):
        return _DEFAULT_CONFIDENCE
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
