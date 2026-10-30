"""Typed ingestion-truth models: graph-write phase state and fidelity metrics.

The ingestion pipeline must never represent skipped work as success.  The models
in this module are the typed vocabulary for the two truths that were previously
collapsed into success-shaped aggregation:

1. ``GraphRelationshipsState`` — whether relationship-edge writes were
   attempted, skipped (not implemented), succeeded, or failed.  ``all([])`` over
   an empty relationship result list is vacuously true; this state makes
   "not attempted", "succeeded with zero edges", and "failed" distinct and
   persisted on the document record.
2. ``IngestionFidelityMetrics`` — measured source fidelity (coverage, ordering,
   OCR confidence/coverage, table/section preservation, chunk-count
   plausibility).  A metric that cannot be computed is persisted as an explicit
   unavailable state (``None`` plus an entry in ``unavailable_metrics``); the
   pipeline never fabricates a number.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from typing import Final

FIDELITY_SCHEMA_VERSION: Literal["fidelity_v1"] = "fidelity_v1"
FIDELITY_THRESHOLDS_VERSION: Literal["fidelity_thresholds_v1"] = "fidelity_thresholds_v1"

RELATIONSHIP_SKIP_REASON: Final[str] = "relationship_extraction_not_implemented"


class GraphRelationshipsStatus(StrEnum):
    """Distinct lifecycle states for the graph relationship-write phase."""

    NOT_ATTEMPTED = "not_attempted"
    SKIPPED_NOT_IMPLEMENTED = "skipped_not_implemented"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class GraphRelationshipsState(BaseModel):
    """Persisted truth about the relationship-extraction/write phase.

    ``SKIPPED_NOT_IMPLEMENTED`` is the state the live pipeline reports while the
    grounded second extraction pass does not exist (the writer accepts
    relationships, but ingestion passes an empty list).  It is deliberately a
    different state from ``NOT_ATTEMPTED`` (no clause segments were extracted at
    all) and from ``SUCCEEDED`` (extraction ran and produced edges).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GraphRelationshipsStatus
    reason: str | None = None
    relationships_attempted: int = Field(default=0, ge=0)
    relationships_succeeded: int = Field(default=0, ge=0)

    def model_dump_for_metadata(self) -> dict[str, object]:
        """Stable JSON shape persisted under ``documents.metadata_.graph_relationships``."""
        return {
            "status": str(self.status.value),
            "reason": self.reason,
            "relationships_attempted": self.relationships_attempted,
            "relationships_succeeded": self.relationships_succeeded,
        }


def aggregate_relationship_states(
    states: list[GraphRelationshipsState],
) -> GraphRelationshipsState:
    """Collapse per-boundary relationship states into one document-level truth.

    Failure dominates (a boundary that attempted and failed is a failure even
    when other boundaries were skipped).  When no boundary ever attempted a
    relationship write, the aggregate is skipped-not-implemented if any clause
    write phase ran, otherwise not-attempted.
    """
    if not states:
        return GraphRelationshipsState(
            status=GraphRelationshipsStatus.NOT_ATTEMPTED,
            reason="graph_write_phase_never_ran",
        )
    failed = [state for state in states if state.status is GraphRelationshipsStatus.FAILED]
    if failed:
        return GraphRelationshipsState(
            status=GraphRelationshipsStatus.FAILED,
            reason=failed[0].reason,
            relationships_attempted=sum(state.relationships_attempted for state in states),
            relationships_succeeded=sum(state.relationships_succeeded for state in states),
        )
    succeeded = [state for state in states if state.status is GraphRelationshipsStatus.SUCCEEDED]
    if succeeded:
        return GraphRelationshipsState(
            status=GraphRelationshipsStatus.SUCCEEDED,
            reason=succeeded[0].reason,
            relationships_attempted=sum(state.relationships_attempted for state in states),
            relationships_succeeded=sum(state.relationships_succeeded for state in states),
        )
    ran_clause_writes = any(
        state.status is GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED for state in states
    )
    aggregate_status = (
        GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED
        if ran_clause_writes
        else GraphRelationshipsStatus.NOT_ATTEMPTED
    )
    # The surfaced reason must come from a state that shares the aggregate's
    # status — a failure's reason must never describe a skipped aggregate.
    return GraphRelationshipsState(
        status=aggregate_status,
        reason=next(
            (
                state.reason
                for state in states
                if state.status is aggregate_status and state.reason is not None
            ),
            RELATIONSHIP_SKIP_REASON if ran_clause_writes else None,
        ),
        relationships_attempted=sum(state.relationships_attempted for state in states),
        relationships_succeeded=sum(state.relationships_succeeded for state in states),
    )


class FidelityThresholds(BaseModel):
    """Versioned degradation thresholds applied to fidelity metrics.

    Defaults follow the plan's verification-before-enforcement discipline:
    structural loss (dropped tables) and chunk-count implausibility are enforced
    immediately because their correct value is unambiguous, while source
    coverage and OCR confidence stay report-only (0.0 = disabled) until the
    queued-ingestion baseline establishes real variance on production-shaped
    documents.  The threshold set actually used is persisted alongside the
    metrics so any gate result can be replayed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["fidelity_thresholds_v1"] = FIDELITY_THRESHOLDS_VERSION
    source_coverage_min: float = Field(default=0.0, ge=0.0, le=1.0)
    table_loss_max: float = Field(default=0.0, ge=0.0, le=1.0)
    section_loss_max: float = Field(default=0.5, ge=0.0, le=1.0)
    ocr_confidence_min: float = Field(default=0.0, ge=0.0, le=1.0)


class ChunkPlausibilityVerdict(StrEnum):
    PLAUSIBLE = "plausible"
    IMPLAUSIBLE = "implausible"
    UNAVAILABLE = "unavailable"


class TokenCountBasis(StrEnum):
    """How the chunk-plausibility token total was obtained, or that it wasn't."""

    TOKENIZER = "tokenizer"
    UNAVAILABLE = "unavailable"


class ChunkPlausibilityState(BaseModel):
    """Chunk-count plausibility: zero-chunk and out-of-bounds detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: ChunkPlausibilityVerdict
    reason: str | None = None
    chunk_count: int = Field(ge=0)
    expected_chunk_count_lower: int | None = Field(default=None, ge=0)
    expected_chunk_count_upper: int | None = Field(default=None, ge=0)
    token_count_basis: TokenCountBasis = TokenCountBasis.UNAVAILABLE


class IngestionFidelityMetrics(BaseModel):
    """Measured source fidelity for one ingested document.

    ``None`` always means "could not be computed" and is mirrored in
    ``unavailable_metrics`` — never a fabricated value.  Fields are document
    level; per-boundary measurements are aggregated before this model is built.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fidelity_schema_version: Literal["fidelity_v1"] = FIDELITY_SCHEMA_VERSION
    thresholds: FidelityThresholds = Field(default_factory=FidelityThresholds)

    # Source coverage / ordering (from chunk source spans vs source markdown).
    source_coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    source_order_preserved: bool | None = None
    chunks_total: int = Field(default=0, ge=0)
    chunks_with_source_span: int = Field(default=0, ge=0)
    chunks_out_of_order: int = Field(default=0, ge=0)

    # OCR (from the docling conversion result; unavailable for formats the
    # pipeline cannot measure, which is the honest state, not a zero).
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ocr_source_coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    ocr_status: Literal["measured", "unavailable"] = "unavailable"

    # Table / section preservation.
    tables_total: int = Field(default=0, ge=0)
    tables_preserved: int = Field(default=0, ge=0)
    table_loss_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    sections_total: int = Field(default=0, ge=0)
    sections_preserved: int = Field(default=0, ge=0)
    section_loss_ratio: float | None = Field(default=None, ge=0.0, le=1.0)

    # Chunk-count plausibility.
    chunk_plausibility: ChunkPlausibilityState = Field(
        default_factory=lambda: ChunkPlausibilityState(
            verdict=ChunkPlausibilityVerdict.UNAVAILABLE, chunk_count=0
        )
    )

    unavailable_metrics: tuple[str, ...] = Field(default_factory=tuple)


class GraphWritePhaseResult(BaseModel):
    """Outcome of one graph-write phase (clause episodes + relationships).

    Replaces the previous ``bool`` whose ``all([])`` aggregation made a
    relationship phase that never ran indistinguishable from one that fully
    succeeded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    clause_writes_attempted: int = Field(default=0, ge=0)
    clause_writes_succeeded: int = Field(default=0, ge=0)
    relationships: GraphRelationshipsState

    @classmethod
    def skipped_relationships(cls) -> GraphRelationshipsState:
        """The canonical typed state for the not-yet-implemented relationship pass."""
        return GraphRelationshipsState(
            status=GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED,
            reason=RELATIONSHIP_SKIP_REASON,
        )
