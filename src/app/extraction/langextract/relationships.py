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
  clause types, char offsets) — never raw document text.  Clause identities
  are issued server-side from boundary coordinates; pass-1 model-derived
  attribution may enter the trusted section only as a short section-number-
  ish token (see ``prompt_safe_attribution``), and inventory entries that
  fail the check are dropped and counted.  Pass-1 clause texts are used
  exclusively for server-side endpoint validation.
- The document travels in a SINGLE model prompt: the request's
  ``max_char_buffer`` is never smaller than the wrapped text, so langextract
  cannot split the untrusted region across per-chunk prompts (a middle chunk
  would carry raw document text with no delimiters).
- Every candidate edge is validated before it can be persisted: the char
  interval must resolve inside the boundary text, the relationship value must
  be in the closed ``RelationshipType`` vocabulary, the clause reference must
  resolve to a pass-1 clause identity whose inventory range contains the
  validated span, and both endpoints must occur in the pass-1 entity corpus.
  Anything else is dropped and counted — never written.
- Server-owned identity: edges never carry tenant/document/group identity
  from content.  Identity is attached at the writer boundary from job
  identity only.
"""

from __future__ import annotations

import hashlib
import math
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple

import langextract as lx
from pydantic import BaseModel, ConfigDict, Field

from app.ai.validation import (
    EvaluationContext,
    MechanicalEvaluator,
    SemanticFinding,
    ValidationPipeline,
)
from app.features.agent_saul.domain.state import RelationshipType
from app.features.documents.ingestion.enrichment.legal_metadata import (
    UNTRUSTED_CONTENT_DIRECTIVE,
    UNTRUSTED_DOCUMENT_BEGIN,
    UNTRUSTED_DOCUMENT_END,
    neutralize_untrusted_delimiters_with_map,
)

from .service import (
    ExtractionBudget,
    ExtractionFailureCode,
    ExtractionRequest,
    ExtractionSucceeded,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Final

    from app.ai.validation import RetryFeedback

    from .service import AsyncExtractionService, ExtractionFailed

RELATIONSHIP_EXTRACTION_CLASS: Final[str] = "relationship"
RELATIONSHIP_CONFIG_DISABLED_REASON: Final[str] = "relationship_pass_disabled_by_config"
RELATIONSHIP_BUDGET_EXHAUSTED_REASON: Final[str] = "relationship_budget_exhausted"
# The untrusted region wraps the document with one-line delimiters; pass-2
# char intervals are relative to the wrapped text and are shifted by this
# prefix before span validation against the boundary body.
_RELATIONSHIP_TEXT_PREFIX: Final[str] = f"{UNTRUSTED_DOCUMENT_BEGIN}\n"
_RELATIONSHIP_TEXT_SUFFIX: Final[str] = f"\n{UNTRUSTED_DOCUMENT_END}"
_MAX_ENTITY_CHARS: Final[int] = 256
# A model-derived attribution string may enter the prompt's trusted
# server-derived section only as a short section-number-ish token: letters,
# digits, and the punctuation section labels use.  Anything else (prose,
# spaces, semicolons, control characters) is rejected, so a document cannot
# steer instructions into the trusted section even in bounded form.
_ATTRIBUTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_.()§\-]{1,32}")
_DEFAULT_CONFIDENCE: Final[float] = 0.5
# A wrapped boundary larger than this degrades the pass explicitly rather
# than building one unbounded model prompt.
_MAX_SINGLE_CALL_CHARS: Final[int] = 200_000

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
        return relationship_candidate_key(
            from_entity=self.from_entity,
            relationship=self.relationship,
            to_entity=self.to_entity,
            clause_id=self.clause_id,
            start=self.char_start,
            end=self.char_end,
        )


# Identity of one pass-2 candidate, validated or dropped: the dedup-key shape
# of :class:`GroundedRelationship`, constructible even for a candidate whose
# span and attributes parsed cleanly but failed a retryable-class check.
_RelationshipKey = tuple[str, str, str, str, int, int]


def relationship_candidate_key(
    *,
    from_entity: str,
    relationship: RelationshipType,
    to_entity: str,
    clause_id: str,
    start: int,
    end: int,
) -> _RelationshipKey:
    """Stable candidate identity shared by validated edges and dropped candidates."""
    return (
        from_entity.casefold(),
        relationship.value,
        to_entity.casefold(),
        clause_id,
        start,
        end,
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
    CLAUSE_SPAN_MISMATCH = "clause_span_mismatch"
    INVALID_ENDPOINT = "invalid_endpoint"
    DUPLICATE = "duplicate"
    OVER_CAP = "over_cap"
    INVALID_CLAUSE_IDENTITY = "invalid_clause_identity"


class RelationshipDropCounts(BaseModel):
    """Typed per-reason drop tally; every drop is visible, never silent."""

    model_config = ConfigDict(frozen=True)

    invalid_span: int = Field(default=0, ge=0)
    invalid_type: int = Field(default=0, ge=0)
    unknown_clause: int = Field(default=0, ge=0)
    clause_span_mismatch: int = Field(default=0, ge=0)
    invalid_endpoint: int = Field(default=0, ge=0)
    duplicate: int = Field(default=0, ge=0)
    over_cap: int = Field(default=0, ge=0)
    invalid_clause_identity: int = Field(default=0, ge=0)

    def bump(self, reason: RelationshipDropReason) -> RelationshipDropCounts:
        updates = {reason.value: getattr(self, reason.value) + 1}
        return self.model_copy(update=updates)

    @property
    def total(self) -> int:
        return (
            self.invalid_span
            + self.invalid_type
            + self.unknown_clause
            + self.clause_span_mismatch
            + self.invalid_endpoint
            + self.duplicate
            + self.over_cap
            + self.invalid_clause_identity
        )


# Shared-validation adoption (validation-seam-adoption): the pass-2 batch is
# produced and gated through the shared ValidationPipeline. The existing
# per-candidate validators are unchanged — they run inside the mechanical
# predicate — and only the declared retryable finding classes are worth a
# feedback-carrying retry, inside the existing relationship budget share.
RELATIONSHIP_VALIDATION_SEAM: Final[str] = "relationship_pass"
# First attempt + one feedback-carrying retry, mirroring the metadata seam's
# one-retry bound.
_RELATIONSHIP_MAX_VALIDATION_ATTEMPTS: Final[int] = 2
_RETRYABLE_DROP_REASONS: Final[tuple[RelationshipDropReason, ...]] = (
    RelationshipDropReason.UNKNOWN_CLAUSE,
    RelationshipDropReason.CLAUSE_SPAN_MISMATCH,
)
_RETRY_FINDING_CODES: Final[tuple[str, ...]] = tuple(
    reason.value for reason in _RETRYABLE_DROP_REASONS
)
_RETRY_FEEDBACK_GUIDANCE: Final[dict[str, str]] = {
    RelationshipDropReason.UNKNOWN_CLAUSE.value: (
        "set 'clause_id' to a clause_id exactly as listed in the server-provided clause inventory"
    ),
    RelationshipDropReason.CLAUSE_SPAN_MISMATCH.value: (
        "emit the extraction text as an exact span of text inside the char range "
        "of the clause named by 'clause_id'"
    ),
}


class RelationshipCandidateBatch(BaseModel):
    """Schema-gate wrapper around one pass-2 provider outcome.

    The langextract documents are held opaquely: the gate exists so the
    pipeline owns the produce→validate loop, not to re-model langextract's
    shapes (the per-candidate validators below remain the ground truth).
    """

    model_config = ConfigDict(extra="forbid")

    documents: tuple[Any, ...] = ()


class _ProviderCallError(Exception):
    """The provider outcome carried no candidates to validate.

    Raised out of ``produce`` so the typed failure propagates to the caller,
    which maps it onto the existing pass result exactly as the pre-pipeline
    code did — unless an earlier attempt already validated, whose result
    stands (a retry failure must not discard validated edges).
    """

    def __init__(self, outcome: ExtractionFailed) -> None:
        super().__init__(outcome.message)
        self.outcome = outcome


class _ValidatedAttempt(NamedTuple):
    """One validation attempt's retained result, plus its retryable drops.

    ``retryable_drops`` records each candidate dropped for a retryable reason
    as ``(candidate key, reason)`` so a later attempt can decide per candidate
    whether the drop was actually fixed — counts alone cannot.
    """

    edges: tuple[GroundedRelationship, ...]
    drops: RelationshipDropCounts
    retryable_drops: tuple[tuple[_RelationshipKey, RelationshipDropReason], ...]


class _RelationshipRunState:
    """Mutable per-run cell shared by the pipeline's produce and evaluator.

    The pipeline callables are stateless between themselves, so the merged
    budget, the last completed attempt's validation result, and a provider
    failure ride here; the caller maps the outcome onto
    :class:`RelationshipPassResult` from this cell.
    """

    __slots__ = ("last_validation", "merged_budget", "provider_failure")

    def __init__(self) -> None:
        self.merged_budget: ExtractionBudget | None = None
        self.provider_failure: ExtractionFailed | None = None
        self.last_validation: _ValidatedAttempt | None = None


class _RelationshipShareLedger:
    """:class:`BudgetLedger` port over the relationship share of the ledger.

    A retry re-invokes the provider, so a charge is granted only while the
    relationship share still has headroom on every dimension. The real
    counters move when the provider's returned ledger is merged back; this
    adapter only answers whether one more call fits.
    """

    __slots__ = ("_state",)

    def __init__(self, state: _RelationshipRunState) -> None:
        self._state = state

    def charge(self) -> bool:
        budget = self._state.merged_budget
        if budget is None:
            return False
        return _relationship_share_exhausted(budget) is None


def _retryable_drop_resolved(
    candidate: _RelationshipKey,
    validated_keys: Sequence[_RelationshipKey],
) -> bool:
    """Whether the retry demonstrably fixed one dropped candidate.

    A fix is a validated retry edge that re-states the same relationship claim
    (same endpoints) with the flawed part corrected: the same span under a
    clause that now resolves (an ``unknown_clause`` fix), or the same clause
    under a span that now fits (a ``clause_span_mismatch`` fix).  A retry that
    merely emits unrelated edges never resolves anything.
    """
    from_entity, relationship, to_entity, clause_id, start, end = candidate
    return any(
        key[0] == from_entity
        and key[1] == relationship
        and key[2] == to_entity
        and ((key[4], key[5]) == (start, end) or key[3] == clause_id)
        for key in validated_keys
    )


def _merge_validated_attempts(
    previous: _ValidatedAttempt | None,
    edges: tuple[GroundedRelationship, ...],
    drops: RelationshipDropCounts,
    retryable_drops: tuple[tuple[_RelationshipKey, RelationshipDropReason], ...],
    *,
    max_edges: int,
) -> _ValidatedAttempt:
    """Fold one validation attempt into the run's retained result.

    A validated attempt is never discarded by the retry: the retained edges
    are the union of the attempts' validated edges, deduplicated by
    :meth:`GroundedRelationship.dedup_key` and still capped at ``max_edges``.
    A re-emitted relationship the earlier attempt already validated is not a
    drop — the retry feedback explicitly asks for that unchanged re-emission —
    but union members beyond the cap are counted (``OVER_CAP``), never
    silent. Non-retryable drop tallies accumulate (attempt one's stand as the
    parity suites pin them).

    Retryable drops persist per candidate: attempt one's retryable drops
    survive the merge unless the retry demonstrably fixed them (the candidate
    re-appears validated, per :func:`_retryable_drop_resolved`), and the
    retry's own new candidate drops join them.  A retry that simply omits a
    dropped candidate therefore keeps its drop visible — only a demonstrated
    fix removes one — and a candidate the retry re-drops under its own key is
    counted once, not stacked.
    """
    if previous is None:
        return _ValidatedAttempt(edges, drops, retryable_drops)
    prev_edges, prev_drops, prev_retryable = previous
    retryable = {reason.value for reason in _RETRYABLE_DROP_REASONS}
    validated_keys = [edge.dedup_key() for edge in edges]
    surviving = [
        (candidate, reason)
        for candidate, reason in prev_retryable
        if not _retryable_drop_resolved(candidate, validated_keys)
    ]
    surviving_keys = {candidate for candidate, _ in surviving}
    merged_retryable = (
        *surviving,
        *(item for item in retryable_drops if item[0] not in surviving_keys),
    )
    retryable_tally = RelationshipDropCounts()
    for _candidate, reason in merged_retryable:
        retryable_tally = retryable_tally.bump(reason)
    merged_drops = RelationshipDropCounts(
        **{
            field: getattr(retryable_tally, field)
            if field in retryable
            else getattr(prev_drops, field) + getattr(drops, field)
            for field in RelationshipDropCounts.model_fields
        }
    )
    merged_edges = list(prev_edges)
    seen = {edge.dedup_key() for edge in prev_edges}
    for edge in edges:
        key = edge.dedup_key()
        if key in seen:
            # The unchanged re-emission the retry feedback asks for.
            continue
        if len(merged_edges) >= max_edges:
            merged_drops = merged_drops.bump(RelationshipDropReason.OVER_CAP)
            continue
        seen.add(key)
        merged_edges.append(edge)
    return _ValidatedAttempt(tuple(merged_edges), merged_drops, merged_retryable)


def _retry_feedback_section(feedback: RetryFeedback) -> str:
    """Render the pipeline's structured feedback as a trusted prompt addendum.

    The one allowed prompt-content change (appending structured retry
    feedback): server-authored finding codes and correction guidance — never
    document content, so the trusted section stays document-free.
    """
    lines = [
        "PREVIOUS ATTEMPT CORRECTION (server validation): only the candidate "
        "relationships counted below were dropped; the other relationships "
        "from the previous attempt were validated and stand. Re-emit the "
        "validated relationships unchanged, and re-emit replacements for the "
        "dropped candidates only, so that the dropped candidates satisfy:"
    ]
    lines.extend(f"- [{finding.code}] {finding.message}" for finding in feedback.semantic_findings)
    lines.extend(f"- [{error.code}] {error.message}" for error in feedback.schema_errors)
    return "\n".join(lines)


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


class BoundaryFrame(NamedTuple):
    """The wrapped pass-2 text plus its mapping back to original coordinates.

    ``body`` is the neutralized document text the provider's char intervals
    resolve against, ``wrapped`` the delimited text actually sent, and
    ``prefix_len`` the shift between wrapped and body coordinates.
    ``to_original`` maps a body offset back to the ORIGINAL boundary text:
    delimiter-literal neutralization replaces markers with a different-length
    string, so offsets after a marker drift — validation, quotes, and saved
    spans must always cite the document as written, never its neutralized
    form.
    """

    body: str
    wrapped: str
    prefix_len: int
    to_original: Callable[[int], int]


def wrap_boundary_text(boundary_text: str) -> BoundaryFrame:
    """Neutralize delimiter literals and wrap the body in the untrusted region.

    Returns the :class:`BoundaryFrame` for the boundary: the neutralized
    body, the wrapped text, the wrapped-to-body prefix shift, and the
    sanitized-to-original offset map.
    """
    body, _count, to_original = neutralize_untrusted_delimiters_with_map(boundary_text)
    wrapped = f"{_RELATIONSHIP_TEXT_PREFIX}{body}{_RELATIONSHIP_TEXT_SUFFIX}"
    return BoundaryFrame(
        body=body,
        wrapped=wrapped,
        prefix_len=len(_RELATIONSHIP_TEXT_PREFIX),
        to_original=to_original,
    )


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

    SINGLE-CALL GUARANTEE: langextract's annotator chunks the input text into
    ``max_char_buffer``-sized pieces and renders ONE model prompt per chunk,
    each carrying only its own slice of the text.  A mid-document chunk would
    then hold raw document text with no delimiters at all, breaking the
    untrusted-region contract at the model-prompt level.  The request
    therefore raises ``max_char_buffer`` to at least the wrapped text length,
    pinning the boundary to exactly one chunk (and one charged model call,
    which the budget accounting in ``AsyncExtractionService`` computes from
    the same buffer).
    """
    frame = wrap_boundary_text(boundary_text)
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
        text=frame.wrapped,
        prompt_description=prompt_description,
        examples=_relationship_examples(),
        # One pass: this is a different prompt over different input, not a
        # recall re-run of the same prompt (that is what extraction_passes
        # does for pass 1).
        extraction_passes=1,
        max_workers=max_workers,
        # Never smaller than the wrapped text: see the single-call guarantee
        # above.  A smaller buffer would split the untrusted region across
        # per-chunk prompts.
        max_char_buffer=max(max_char_buffer, len(frame.wrapped)),
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

    Every tightened ceiling is additionally clamped to the document's
    ORIGINAL limit: a pass-1 that ran at its cap must not be allowed to
    spend its relationship share past the configured document ceiling
    (``current_usage + remaining_share`` can otherwise exceed it).
    """
    return budget.model_copy(
        update={
            "max_model_calls": min(
                budget.max_model_calls,
                budget.model_calls
                + max(0, budget.relationship_max_model_calls - budget.relationship_model_calls),
            ),
            "max_input_tokens": min(
                budget.max_input_tokens,
                budget.input_tokens
                + max(0, budget.relationship_max_input_tokens - budget.relationship_input_tokens),
            ),
            "max_output_tokens": min(
                budget.max_output_tokens,
                budget.output_tokens
                + max(0, budget.relationship_max_output_tokens - budget.relationship_output_tokens),
            ),
            "max_cost_usd": min(
                budget.max_cost_usd,
                budget.cost_estimate_usd
                + max(
                    0.0, budget.relationship_max_cost_usd - budget.relationship_cost_estimate_usd
                ),
            ),
        }
    )


def _merge_relationship_charge(
    *, original: ExtractionBudget, returned: ExtractionBudget | None
) -> ExtractionBudget:
    """Restore shared maxima and accumulate the pass-2 charge into the pool.

    ``original`` is the pre-call shared ledger: its maxima were only ever
    tightened on the call-scoped copy, so they are restored here, and the
    counter deltas against it are exactly what this pass spent.

    ``returned`` may be None: failure outcomes may carry no budget, and the
    pass-level failure handling already records the failure — there is
    nothing to merge.
    """
    if returned is None:
        return original
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


def prompt_safe_attribution(raw: object) -> str | None:
    """Validate model-derived clause attribution for the trusted prompt section.

    Pass-1 clause attribution is model output over untrusted document
    content: a document can steer it to carry instructions.  Only a short
    section-number-ish token (``_ATTRIBUTION_PATTERN``: letters, digits,
    ``_.()§-``) can enter the prompt's trusted server-derived section;
    prose, separators, and control characters are all rejected, and the
    caller falls back to the server-issued identity.
    """
    text = str(raw).strip() if raw is not None else ""
    if not text or _ATTRIBUTION_PATTERN.fullmatch(text) is None:
        return None
    return text


def _is_prompt_safe(value: str) -> bool:
    """Whether a string may enter the prompt's trusted section at all."""
    return prompt_safe_attribution(value) is not None


def _sanitize_inventory(
    inventory: Sequence[ClauseIdentity],
) -> tuple[tuple[ClauseIdentity, ...], RelationshipDropCounts]:
    """Drop inventory entries whose identity strings fail the trusted-form check."""
    sanitized: list[ClauseIdentity] = []
    drops = RelationshipDropCounts()
    for entry in inventory:
        if all(
            _is_prompt_safe(value)
            for value in (entry.clause_id, entry.clause_type, entry.section_ref)
        ):
            sanitized.append(entry)
        else:
            drops = drops.bump(RelationshipDropReason.INVALID_CLAUSE_IDENTITY)
    return tuple(sanitized), drops


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

    Validation runs through the shared :class:`ValidationPipeline`: the
    per-candidate validators are unchanged (they execute inside the mechanical
    predicate), a ``unknown_clause``/``clause_span_mismatch`` finding earns
    one feedback-carrying retry inside the relationship budget share, and
    every other drop reason behaves exactly as before the adoption.
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
    sanitized_inventory, inventory_drops = _sanitize_inventory(inventory)
    if not sanitized_inventory:
        return RelationshipPassResult(
            ran=False,
            skip_reason="no_valid_clause_inventory",
            drops=inventory_drops,
            budget=budget,
        )
    # The single-call design raises max_char_buffer to the wrapped length; a
    # pathological boundary (a whole book as one "contract") must degrade the
    # pass explicitly instead of building an unbounded prompt.
    if len(wrap_boundary_text(boundary_text).wrapped) > _MAX_SINGLE_CALL_CHARS:
        return RelationshipPassResult(
            ran=False,
            skip_reason="boundary_too_large_for_single_call",
            drops=inventory_drops,
            budget=budget,
        )
    frame = wrap_boundary_text(boundary_text)
    state = _RelationshipRunState()

    async def _produce(feedback: RetryFeedback | None) -> object:
        """One provider call: the raw pass-2 candidate documents, unparsed.

        The retry re-derives the tightened share from the merged ledger so
        the second call spends only the share's remaining headroom, and it
        appends the pipeline's structured feedback to the prompt's trusted
        section (the one allowed prompt-content change).
        """
        base_budget = budget if state.merged_budget is None else state.merged_budget
        request = build_relationship_request(
            boundary_text=boundary_text,
            inventory=sanitized_inventory,
            budget=_relationship_share_budget(base_budget),
            document_id=document_id,
            tenant_id=tenant_id,
            document_version=document_version,
            contract_index=contract_index,
            max_char_buffer=max_char_buffer,
            max_workers=max_workers,
        )
        if feedback is not None:
            request = request.model_copy(
                update={
                    "prompt_description": (
                        f"{request.prompt_description}\n\n{_retry_feedback_section(feedback)}"
                    )
                }
            )
        outcome = await extraction.extract(request)
        state.merged_budget = _merge_relationship_charge(original=budget, returned=outcome.budget)
        if isinstance(outcome, ExtractionSucceeded):
            return {"documents": outcome.documents}
        state.provider_failure = outcome
        raise _ProviderCallError(outcome)

    def _batch_predicate(
        value: RelationshipCandidateBatch, _context: EvaluationContext
    ) -> SemanticFinding | None:
        """Mechanical gate: the unchanged per-candidate validators, verbatim.

        Non-retryable flaws stay plain per-candidate drops (recorded in the
        run state); a retryable-class drop surfaces as the finding that earns
        the feedback-carrying retry.
        """
        # A retry starts from a clean candidate tally: the inventory drops and
        # the earlier attempt's non-retryable drops are already retained in
        # the run state and must not be counted twice.
        edges, drops, retryable_drops = _validate_relationship_extractions(
            value.documents,
            boundary_text=boundary_text,
            frame=frame,
            inventory=sanitized_inventory,
            entity_corpus=entity_corpus,
            max_edges=config.max_edges_per_boundary,
            drops=inventory_drops if state.last_validation is None else RelationshipDropCounts(),
        )
        state.last_validation = _merge_validated_attempts(
            state.last_validation,
            edges,
            drops,
            retryable_drops,
            max_edges=config.max_edges_per_boundary,
        )
        for reason in _RETRYABLE_DROP_REASONS:
            count = getattr(drops, reason.value)
            if count:
                return SemanticFinding(
                    code=reason.value,
                    message=(
                        f"{count} candidate relationship(s) were dropped as "
                        f"{reason.value}: {_RETRY_FEEDBACK_GUIDANCE[reason.value]}."
                    ),
                )
        return None

    pipeline = ValidationPipeline(
        schema=RelationshipCandidateBatch,
        evaluator=MechanicalEvaluator((_batch_predicate,)),
        max_attempts=_RELATIONSHIP_MAX_VALIDATION_ATTEMPTS,
        retryable_findings=_RETRY_FINDING_CODES,
        budget=_RelationshipShareLedger(state),
    )
    context = EvaluationContext.model_validate(
        {
            "seam": RELATIONSHIP_VALIDATION_SEAM,
            "document_id": document_id,
            "contract_index": contract_index,
        }
    )
    try:
        await pipeline.validate(_produce, context)
    except _ProviderCallError as failure:
        merged = budget if state.merged_budget is None else state.merged_budget
        if state.last_validation is not None:
            # A validated attempt stands regardless of why the retry failed:
            # the edges it validated are returned with the merged budget, and
            # only the pre-pipeline skip/error shapes below apply when no
            # attempt ever validated.
            attempt = state.last_validation
            return RelationshipPassResult(
                ran=True, edges=attempt.edges, drops=attempt.drops, budget=merged
            )
        if failure.outcome.code is ExtractionFailureCode.BUDGET_EXHAUSTED:
            return RelationshipPassResult(
                ran=False,
                skip_reason=RELATIONSHIP_BUDGET_EXHAUSTED_REASON,
                budget_exhausted=True,
                budget=merged,
            )
        return RelationshipPassResult(
            ran=True,
            provider_error=failure.outcome.code.value,
            budget=merged,
        )
    merged = budget if state.merged_budget is None else state.merged_budget
    if state.last_validation is None:
        # Defensive: a terminal outcome without any completed validation run
        # (e.g. a schema-gate defect). Fail closed with no edges, as before.
        return RelationshipPassResult(ran=True, drops=inventory_drops, budget=merged)
    attempt = state.last_validation
    return RelationshipPassResult(ran=True, edges=attempt.edges, drops=attempt.drops, budget=merged)


class _EdgeRejection(NamedTuple):
    """One dropped candidate: the reason plus its identity when reconstructable.

    ``candidate_key`` is set only for the retryable-class drops — those
    candidates' spans and attributes parsed cleanly, so the candidate can be
    tracked across attempts and its drop removed only by a demonstrated fix.
    """

    reason: RelationshipDropReason
    candidate_key: _RelationshipKey | None


def _validate_relationship_extractions(
    documents: Sequence[lx.data.AnnotatedDocument],
    *,
    boundary_text: str,
    frame: BoundaryFrame,
    inventory: Sequence[ClauseIdentity],
    entity_corpus: Sequence[str],
    max_edges: int,
    drops: RelationshipDropCounts,
) -> tuple[
    tuple[GroundedRelationship, ...],
    RelationshipDropCounts,
    tuple[tuple[_RelationshipKey, RelationshipDropReason], ...],
]:
    """Fail-closed validation of pass-2 candidates; drops are counted.

    Returns the validated edges, the per-reason drop tally, and the retryable
    drops as per-candidate ``(candidate key, reason)`` records.
    """
    edges: list[GroundedRelationship] = []
    retryable_drops: list[tuple[_RelationshipKey, RelationshipDropReason]] = []
    seen: set[tuple[str, str, str, str, int, int]] = set()
    # Retained inventory char ranges: a validated edge span must lie inside
    # the range of the clause it names, or it is dropped (CLAUSE_SPAN_MISMATCH).
    clause_ranges = {entry.clause_id: (entry.char_start, entry.char_end) for entry in inventory}
    corpus = " \n".join(text.casefold() for text in entity_corpus if text)

    for document in documents:
        for extraction in document.extractions:
            if extraction.extraction_class.casefold() != RELATIONSHIP_EXTRACTION_CLASS:
                continue
            validated = _validated_edge(
                extraction,
                boundary_text=boundary_text,
                frame=frame,
                clause_ranges=clause_ranges,
                corpus=corpus,
            )
            if isinstance(validated, _EdgeRejection):
                drops = drops.bump(validated.reason)
                if validated.candidate_key is not None:
                    retryable_drops.append((validated.candidate_key, validated.reason))
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
    return tuple(edges), drops, tuple(retryable_drops)


def _validated_edge(
    extraction: lx.data.Extraction,
    *,
    boundary_text: str,
    frame: BoundaryFrame,
    clause_ranges: dict[str, tuple[int, int]],
    corpus: str,
) -> GroundedRelationship | _EdgeRejection:
    """Validate one candidate against the original boundary text and identities.

    The provider's char interval is relative to the wrapped, delimiter-
    neutralized text; the validated span is mapped back to ORIGINAL boundary
    coordinates before range checks, so quotes and saved spans always cite
    the document as written.
    """
    interval = extraction.char_interval
    if interval is None or interval.start_pos is None or interval.end_pos is None:
        # Skip ungrounded extractions exactly as langextract_to_graph
        # does (char_interval is None) — and count the skip.
        return _EdgeRejection(RelationshipDropReason.INVALID_SPAN, None)
    body_start = interval.start_pos - frame.prefix_len
    body_end = interval.end_pos - frame.prefix_len
    if not 0 <= body_start < body_end <= len(frame.body):
        return _EdgeRejection(RelationshipDropReason.INVALID_SPAN, None)
    start = frame.to_original(body_start)
    end = frame.to_original(body_end)
    if not 0 <= start < end <= len(boundary_text):
        return _EdgeRejection(RelationshipDropReason.INVALID_SPAN, None)
    attributes = extraction.attributes or {}
    try:
        relationship = RelationshipType(str(attributes.get("relationship", "")).strip().casefold())
    except ValueError:
        return _EdgeRejection(RelationshipDropReason.INVALID_TYPE, None)
    clause_id = str(attributes.get("clause_id", "")).strip()
    from_entity = str(attributes.get("from_entity", "")).strip()
    to_entity = str(attributes.get("to_entity", "")).strip()
    # A retryable-class drop's candidate is fully identified (its span and
    # attributes parsed cleanly), so the drop can be tracked across attempts.
    candidate_key = relationship_candidate_key(
        from_entity=from_entity,
        relationship=relationship,
        to_entity=to_entity,
        clause_id=clause_id,
        start=start,
        end=end,
    )
    clause_range = clause_ranges.get(clause_id)
    if clause_range is None:
        return _EdgeRejection(RelationshipDropReason.UNKNOWN_CLAUSE, candidate_key)
    if not (clause_range[0] <= start and end <= clause_range[1]):
        return _EdgeRejection(RelationshipDropReason.CLAUSE_SPAN_MISMATCH, candidate_key)
    if not (_entity_grounded(from_entity, corpus) and _entity_grounded(to_entity, corpus)):
        return _EdgeRejection(RelationshipDropReason.INVALID_ENDPOINT, None)
    return GroundedRelationship(
        from_entity=from_entity,
        relationship=relationship,
        to_entity=to_entity,
        clause_id=clause_id,
        char_start=start,
        char_end=end,
        quote=boundary_text[start:end],
        confidence=_parse_confidence(attributes.get("confidence")),
    )


def _entity_grounded(entity: str, corpus: str) -> bool:
    """An endpoint must be non-empty, bounded, and occur in pass-1 entity text."""
    folded = entity.casefold()
    return bool(folded) and len(entity) <= _MAX_ENTITY_CHARS and folded in corpus


def _parse_confidence(raw: object) -> float:
    """Parse a model-supplied confidence, defaulting when absent/invalid.

    Non-finite floats (``NaN``, ``inf``) slip past the range comparisons —
    both are false for NaN — so they are rejected explicitly; a non-finite
    confidence would otherwise raise inside ``GroundedRelationship`` and kill
    the whole ingestion task instead of dropping one candidate.
    """
    try:
        value = float(str(raw))
    except (TypeError, ValueError):
        return _DEFAULT_CONFIDENCE
    if not math.isfinite(value):
        return _DEFAULT_CONFIDENCE
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
