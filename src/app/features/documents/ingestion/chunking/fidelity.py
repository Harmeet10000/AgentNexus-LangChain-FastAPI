"""Pure fidelity measurement for the ingestion pipeline.

Every function here is honest by construction: a metric that cannot be measured
is reported as unavailable (``None`` / ``UNAVAILABLE``), never as a zero and
never as a fabricated number.  The measurements are computed where the
information actually exists:

- source coverage and ordering: from the source spans
  (``source_start_char``/``source_end_char``) annotated onto chunks against the
  parsed source markdown;
- table/section preservation: from parsed tables and structured sections vs the
  text that survived into chunk content;
- chunk-count plausibility: token-count bounds from the process tokenizer and
  the resolved chunk policy;
- OCR confidence/coverage: extracted by the parser from the docling conversion
  result (``ParsedDocument.ocr_confidence`` / ``ParsedDocument.ocr_source_coverage_ratio``).
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.extraction.docling.chunker import get_tokenizer
from app.features.documents.domain.fidelity import (
    ChunkPlausibilityState,
    ChunkPlausibilityVerdict,
    FidelityThresholds,
    IngestionFidelityMetrics,
    TokenCountBasis,
)
from app.features.documents.ingestion.chunking.classification import (
    PreparedChunk,
    QualityWarning,
    _structure_markdown,
)
from app.retrieval.chunk_policy import resolve_chunk_policy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase

    from app.features.documents.ingestion.chunking.classification import ParsedDocument

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_cell_space(text: str) -> str:
    """Normalize text into a pipe-free, whitespace-collapsed cell space.

    Both table rows and chunk text are projected into this space, so row
    matching is robust to column-width re-serialization and to chunk text that
    still carries pipe delimiters.
    """
    return _WHITESPACE_RE.sub(" ", text.replace("|", " ")).strip()


class SourceCoverageMeasurement(BaseModel):
    """Coverage of the source markdown by located chunk spans."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    coverage_ratio: float | None = None
    order_preserved: bool | None = None
    chunks_total: int = 0
    chunks_with_source_span: int = 0
    chunks_out_of_order: int = 0


def measure_source_coverage(
    *, source_markdown: str, chunks: Sequence[PreparedChunk]
) -> SourceCoverageMeasurement:
    """Measure how much of the source markdown the located chunk spans cover.

    Spans are absolute character offsets into ``ParsedDocument.markdown``.  A
    chunk whose content could not be located in the source contributes no span
    (visible in ``chunks_with_source_span``) and reduces coverage.  Ordering is
    evaluated only over located spans; if no span exists at all, ordering is
    unavailable rather than vacuously preserved.
    """
    total_chars = len(source_markdown)
    if total_chars == 0 or not chunks:
        return SourceCoverageMeasurement(
            coverage_ratio=None if total_chars == 0 else 0.0,
            order_preserved=None,
            chunks_total=len(chunks),
        )
    intervals: list[tuple[int, int]] = []
    out_of_order = 0
    previous_start: int | None = None
    for chunk in chunks:
        start = chunk.source_start_char
        end = chunk.source_end_char
        if start is None or end is None or end < start:
            continue
        if previous_start is not None and start < previous_start:
            out_of_order += 1
        previous_start = start
        intervals.append((start, min(end, total_chars)))
    covered = 0
    cursor: int | None = None
    for start, end in sorted(intervals):
        if cursor is None:
            covered += end - start
            cursor = end
        elif end > cursor:
            covered += end - max(start, cursor)
            cursor = end
    return SourceCoverageMeasurement(
        coverage_ratio=min(covered / total_chars, 1.0),
        order_preserved=(out_of_order == 0) if intervals else None,
        chunks_total=len(chunks),
        chunks_with_source_span=len(intervals),
        chunks_out_of_order=out_of_order,
    )


def _pipe_table_rows(table: str) -> list[list[str]] | None:
    """Parse a pipe-markdown table into semantic cell rows, or None.

    Separator rows (``---``) carry no content and are dropped.  Returns None
    for text that is not pipe-table shaped so callers can fall back to plain
    containment.
    """
    lines = [line.strip() for line in table.strip().splitlines() if line.strip()]
    if len(lines) < 1 or not all("|" in line for line in lines):
        return None
    rows: list[list[str]] = []
    for line in lines:
        cells = [_normalize(cell) for cell in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r"[-: ]*", cell) for cell in cells):
            continue  # header/body separator row — formatting, not content
        content = [cell for cell in cells if cell]
        if content:
            rows.append(content)
    return rows or None


def measure_table_preservation(*, tables: Sequence[str], chunk_text: str) -> tuple[int, int]:
    """Count parsed tables whose content survives into chunk text.

    Comparison is semantic, not byte-exact: a pipe table is preserved when
    every content row's normalized cell sequence appears in the normalized
    chunk text, so rows split across adjacent chunks (each row still intact
    within its chunk) and re-serialized column alignment both count as
    preserved.  A genuinely dropped row makes the table lost.
    """
    normalized_chunks = _normalize_cell_space(chunk_text)
    preserved = 0
    for table in tables:
        if not table.strip():
            continue
        rows = _pipe_table_rows(table)
        if rows is None:
            # Not pipe-table shaped: fall back to normalized containment.
            if _normalize_cell_space(table) in normalized_chunks:
                preserved += 1
            continue
        if all(" ".join(cells) in normalized_chunks for cells in rows):
            preserved += 1
    return len(tables), preserved


def measure_section_preservation(*, source_markdown: str, chunk_text: str) -> tuple[int, int]:
    """Count structured source sections whose text survives into chunk content."""
    normalized_chunks = _normalize(chunk_text)
    sections = [item.text for item in _structure_markdown(source_markdown) if item.kind == "text"]
    preserved = sum(1 for section in sections if _normalize(section) in normalized_chunks)
    return len(sections), preserved


def _plausibility_bounds(
    *,
    markdown: str,
    document_kind: str,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[int, int, int]:
    """Plausible chunk-count bounds for one boundary under its own policy.

    Returns ``(lower, upper, total_tokens)``.  The hybrid chunker never emits a
    chunk larger than the policy's token bound, so the count is bounded below
    by ``ceil(total_tokens / max_tokens)``.  Every structure item (headings and
    text sections alike) can emit its own chunk, and any item exceeding the
    token bound is split into ``ceil(item_tokens / max_tokens)`` chunks, so the
    count is bounded above by ``len(items)`` plus those split extras.  An empty
    boundary contributes ``(0, 0, 0)``.
    """
    items = _structure_markdown(markdown)
    total_tokens = len(tokenizer.encode(markdown))
    item_tokens = [len(tokenizer.encode(item.text)) for item in items]
    max_tokens = resolve_chunk_policy(document_kind).max_tokens
    lower = math.ceil(total_tokens / max_tokens) if total_tokens else 0
    split_extras = sum(max(0, math.ceil(tokens / max_tokens) - 1) for tokens in item_tokens)
    upper = len(items) + split_extras
    return lower, upper, total_tokens


def _combined_plausibility(
    *,
    markdown: str,
    boundaries: Sequence[BoundaryFidelityInputs],
    chunk_count: int,
    document_kind: str,
) -> ChunkPlausibilityState:
    """Judge the persisted chunk count against per-boundary policy bounds.

    A mixed document chunks each contract boundary under that boundary's own
    policy, so a document-wide bound derived from the combined classification
    can reject fully-retained content (e.g. a statute boundary chunked at 384
    tokens inside a document classified as a 512-token contract).  Bounds are
    therefore computed per boundary and summed.  Whole-document truthfulness
    checks (empty source, zero output) stay document-wide.
    """
    if not markdown.strip():
        return ChunkPlausibilityState(
            verdict=(
                ChunkPlausibilityVerdict.PLAUSIBLE
                if chunk_count == 0
                else ChunkPlausibilityVerdict.IMPLAUSIBLE
            ),
            reason=None if chunk_count == 0 else "Empty document produced chunks.",
            chunk_count=chunk_count,
        )
    if chunk_count == 0:
        return ChunkPlausibilityState(
            verdict=ChunkPlausibilityVerdict.IMPLAUSIBLE,
            reason="Non-empty document produced zero chunks (empty output).",
            chunk_count=0,
        )
    try:
        tokenizer = get_tokenizer()
    except Exception as exc:  # noqa: BLE001 — plausibility degrades to unavailable, never fails ingestion
        exc.add_note("operation=fidelity_chunk_plausibility_token_count")
        # Carry the failure identity (missing model vs corrupted download vs
        # offline runner) into the persisted reason so operators can tell the
        # unavailability causes apart.
        return ChunkPlausibilityState(
            verdict=ChunkPlausibilityVerdict.UNAVAILABLE,
            reason=(
                f"Tokenizer unavailable ({type(exc).__name__}: {str(exc)[:200]}); "
                "chunk-count bounds could not be computed."
            ),
            chunk_count=chunk_count,
        )
    if not boundaries:
        lower, upper, total_tokens = _plausibility_bounds(
            markdown=markdown, document_kind=document_kind, tokenizer=tokenizer
        )
    else:
        lower = upper = total_tokens = 0
        for boundary in boundaries:
            boundary_lower, boundary_upper, boundary_tokens = _plausibility_bounds(
                markdown=boundary.markdown,
                document_kind=boundary.document_kind,
                tokenizer=tokenizer,
            )
            lower += boundary_lower
            upper += boundary_upper
            total_tokens += boundary_tokens
    lower = max(1, lower)
    upper = max(upper, lower, 1)
    if chunk_count < lower or chunk_count > upper:
        return ChunkPlausibilityState(
            verdict=ChunkPlausibilityVerdict.IMPLAUSIBLE,
            reason=(
                f"Chunk count {chunk_count} is outside the expected bounds "
                f"[{lower}, {upper}] for {total_tokens} tokens."
            ),
            chunk_count=chunk_count,
            expected_chunk_count_lower=lower,
            expected_chunk_count_upper=upper,
            token_count_basis=TokenCountBasis.TOKENIZER,
        )
    return ChunkPlausibilityState(
        verdict=ChunkPlausibilityVerdict.PLAUSIBLE,
        chunk_count=chunk_count,
        expected_chunk_count_lower=lower,
        expected_chunk_count_upper=upper,
        token_count_basis=TokenCountBasis.TOKENIZER,
    )


def measure_chunk_plausibility(
    *,
    source_markdown: str,
    chunk_count: int,
    document_kind: str,
) -> ChunkPlausibilityState:
    """Single-policy plausibility check for one document or boundary.

    Detection of zero-chunk output for a non-empty document and out-of-bounds
    counts under the policy resolved for ``document_kind``.  The queued
    pipeline uses :func:`_combined_plausibility` instead, because each contract
    boundary is chunked under its own policy.
    """
    return _combined_plausibility(
        markdown=source_markdown,
        boundaries=(),
        chunk_count=chunk_count,
        document_kind=document_kind,
    )


class BoundaryFidelityInputs(BaseModel):
    """One contract boundary's contribution to document fidelity measurement.

    ``document_kind`` is the boundary's own classification, not the document's:
    a mixed document chunks each boundary under its own policy, so plausibility
    bounds must be derived per boundary and combined, never document-wide.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    markdown: str
    chunks: tuple[PreparedChunk, ...]
    document_kind: str


class StructuralPreservation(BaseModel):
    """Table and section preservation totals across all boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tables_total: int = 0
    tables_preserved: int = 0
    sections_total: int = 0
    sections_preserved: int = 0


def _measure_structural_preservation(
    *,
    parsed: ParsedDocument,
    boundaries: Sequence[BoundaryFidelityInputs],
    chunk_text: str,
) -> StructuralPreservation:
    tables_total, tables_preserved = measure_table_preservation(
        tables=parsed.tables, chunk_text=chunk_text
    )
    sections_total = 0
    sections_preserved = 0
    for boundary in boundaries:
        boundary_total, boundary_preserved = measure_section_preservation(
            source_markdown=boundary.markdown, chunk_text=chunk_text
        )
        sections_total += boundary_total
        sections_preserved += boundary_preserved
    return StructuralPreservation(
        tables_total=tables_total,
        tables_preserved=tables_preserved,
        sections_total=sections_total,
        sections_preserved=sections_preserved,
    )


def build_fidelity_metrics(
    *,
    parsed: ParsedDocument,
    boundaries: Sequence[BoundaryFidelityInputs],
    chunk_count: int,
    document_kind: str,
    thresholds: FidelityThresholds | None = None,
) -> IngestionFidelityMetrics:
    """Aggregate per-boundary measurements into document-level fidelity metrics.

    ``chunk_count`` is the persisted chunk-row count, which is the number
    plausibility must judge.
    """
    thresholds = thresholds or FidelityThresholds()
    markdown = parsed.markdown
    all_chunks = [chunk for boundary in boundaries for chunk in boundary.chunks]
    coverage = measure_source_coverage(source_markdown=markdown, chunks=all_chunks)
    chunk_text = "\n".join(chunk.content for chunk in all_chunks)
    structural = _measure_structural_preservation(
        parsed=parsed, boundaries=boundaries, chunk_text=chunk_text
    )
    plausibility = _combined_plausibility(
        markdown=markdown,
        boundaries=boundaries,
        chunk_count=chunk_count,
        document_kind=document_kind,
    )
    unavailable: list[str] = []
    if coverage.coverage_ratio is None:
        unavailable.append("source_coverage_ratio")
    if coverage.order_preserved is None:
        unavailable.append("source_order_preserved")
    ocr_confidence = parsed.ocr_confidence
    ocr_coverage = parsed.ocr_source_coverage_ratio
    if ocr_confidence is None:
        unavailable.append("ocr_confidence")
    if ocr_coverage is None:
        unavailable.append("ocr_source_coverage_ratio")
    # ``ocr_status`` reflects whether OCR *confidence* was measured — the
    # quality signal.  Page text coverage alone does not make OCR measured;
    # coverage with unavailable confidence stays an explicit OCR-unavailable
    # record rather than an implied-healthy one.
    ocr_measured = ocr_confidence is not None and math.isfinite(ocr_confidence)
    table_loss = (
        (structural.tables_total - structural.tables_preserved) / structural.tables_total
        if structural.tables_total > 0
        else 0.0
    )
    section_loss = (
        (structural.sections_total - structural.sections_preserved) / structural.sections_total
        if structural.sections_total > 0
        else 0.0
    )
    return IngestionFidelityMetrics(
        thresholds=thresholds,
        source_coverage_ratio=coverage.coverage_ratio,
        source_order_preserved=coverage.order_preserved,
        chunks_total=chunk_count,
        chunks_with_source_span=coverage.chunks_with_source_span,
        chunks_out_of_order=coverage.chunks_out_of_order,
        ocr_confidence=ocr_confidence,
        ocr_source_coverage_ratio=ocr_coverage,
        ocr_status="measured" if ocr_measured else "unavailable",
        tables_total=structural.tables_total,
        tables_preserved=structural.tables_preserved,
        table_loss_ratio=table_loss,
        sections_total=structural.sections_total,
        sections_preserved=structural.sections_preserved,
        section_loss_ratio=section_loss,
        chunk_plausibility=plausibility,
        unavailable_metrics=tuple(unavailable),
    )


def fidelity_degradation_warnings(metrics: IngestionFidelityMetrics) -> list[QualityWarning]:
    """Typed warnings for every threshold breach; empty means fully healthy."""
    warnings: list[QualityWarning] = []
    thresholds = metrics.thresholds
    if (
        metrics.source_coverage_ratio is not None
        and metrics.source_coverage_ratio < thresholds.source_coverage_min
    ):
        warnings.append(
            QualityWarning(
                stage="fidelity",
                code="FIDELITY_SOURCE_COVERAGE_LOW",
                message=(
                    f"Source coverage {metrics.source_coverage_ratio:.3f} is below the "
                    f"configured minimum {thresholds.source_coverage_min:.3f}."
                ),
                severity="warning",
            )
        )
    if metrics.source_order_preserved is False:
        warnings.append(
            QualityWarning(
                stage="fidelity",
                code="FIDELITY_SOURCE_ORDER_VIOLATED",
                message="Located chunk spans are not in source order.",
                severity="warning",
            )
        )
    if (
        metrics.ocr_confidence is not None
        and math.isfinite(metrics.ocr_confidence)
        and thresholds.ocr_confidence_min > 0.0
        and metrics.ocr_confidence < thresholds.ocr_confidence_min
    ):
        warnings.append(
            QualityWarning(
                stage="fidelity",
                code="FIDELITY_OCR_CONFIDENCE_LOW",
                message=(
                    f"OCR confidence {metrics.ocr_confidence:.3f} is below the enabled "
                    f"minimum {thresholds.ocr_confidence_min:.3f}."
                ),
                severity="warning",
            )
        )
    if (
        metrics.table_loss_ratio is not None
        and metrics.table_loss_ratio > thresholds.table_loss_max
    ):
        warnings.append(
            QualityWarning(
                stage="fidelity",
                code="FIDELITY_TABLE_LOSS",
                message=(
                    f"{metrics.tables_total - metrics.tables_preserved} of {metrics.tables_total} "
                    "parsed tables did not survive into chunks."
                ),
                severity="warning",
            )
        )
    if (
        metrics.section_loss_ratio is not None
        and metrics.section_loss_ratio > thresholds.section_loss_max
    ):
        warnings.append(
            QualityWarning(
                stage="fidelity",
                code="FIDELITY_SECTION_LOSS",
                message=(
                    f"{metrics.sections_total - metrics.sections_preserved} of "
                    f"{metrics.sections_total} source sections did not survive into chunks."
                ),
                severity="warning",
            )
        )
    if metrics.chunk_plausibility.verdict is ChunkPlausibilityVerdict.IMPLAUSIBLE:
        warnings.append(
            QualityWarning(
                stage="fidelity",
                code="FIDELITY_CHUNK_COUNT_IMPLAUSIBLE",
                message=metrics.chunk_plausibility.reason
                or "Chunk count is outside the expected bounds for this document.",
                severity="warning",
            )
        )
    return warnings
