"""Offline deterministic ingestion-fidelity measurement over the recorded corpus (I4).

Task 4.7 of ``harden-agent-saul-evidence-pipeline``: run the queued pipeline's
deterministic stages against the I4 corpus and publish a machine-readable
fidelity report.  The live queued run (real docling parse, embeddings, Graphiti,
LangExtract) is a separate, quota-gated activity recorded in
``scripts/live_measurement/RUNBOOK.md``; when it is blocked, this module runs
the stages that are fully offline and deterministic:

- contract boundaries: the real ``establish_contract_boundaries`` over the
  reconstructed source markdown;
- chunks: the recorded corpus convention (``legal_retrieval_v2.CHUNKING.md`` —
  normalize rules R1-R7, split on blank-line runs, 1-based ordinal per page), so
  gold chunk ids are the same convention the retrieval gold sets cite;
- fidelity signals: the real production measurements
  (``measure_source_coverage``, ``measure_section_preservation``,
  ``measure_table_preservation``, ``fidelity_degradation_warnings``);
- clause identity: the real locus recovery and Phase-3-style clause-kind
  inference over the convention chunks.

Honest unavailability is part of the contract: the docling parse (parsed-table
totals, OCR confidence), the production tokenizer (chunk-count plausibility)
and every model-backed stage do not run offline, and the report says so
explicitly instead of fabricating numbers.  The report's pilot gate evaluates
exactly what it covers — the labeled coverage/order and warning outcomes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure

from app.ai.evaluation.deterministic import chunks_for_page
from app.ai.evaluation.schema import load_ingestion_fidelity_set
from app.features.documents.domain.fidelity import (
    ChunkPlausibilityState,
    ChunkPlausibilityVerdict,
    FidelityThresholds,
    IngestionFidelityMetrics,
    TokenCountBasis,
)
from app.features.documents.ingestion.chunking.classification import (
    ParsedDocument,
    PreparedChunk,
    _infer_clause_type,
    classify_document,
    establish_contract_boundaries,
    recover_locus_from_text,
)
from app.features.documents.ingestion.chunking.fidelity import (
    fidelity_degradation_warnings,
    measure_section_preservation,
    measure_source_coverage,
    measure_table_preservation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

    from app.ai.evaluation.schema import IngestionFidelityExpectation, IngestionFidelityRow

REPORT_VERSION: Final[str] = "ingestion_fidelity_report_v1"

# What the offline path deliberately does not measure (part of every report so
# no consumer mistakes absence for a passing measurement).
OFFLINE_LIMITATIONS: Final[tuple[str, ...]] = (
    "docling parse, parsed-table totals and OCR confidence require the original "
    "binaries and are not measured offline; table preservation is labeled and "
    "CI-enforced on the hermetic evals/ingestion fixtures instead",
    "chunk-count plausibility needs the production tokenizer; offline it is "
    "recorded as explicitly unavailable, never as a fabricated bound",
    "embeddings, Graphiti verification, LangExtract extraction and the "
    "completed/completed_with_warnings terminal states are live-queued-pipeline "
    "outcomes; the offline report covers parse->boundaries->chunk->fidelity "
    "signals only",
)

# The live queued run state at report time (scripts/live_measurement/RUNBOOK.md,
# 2026-11-10/11): 12/15 corpus ingestions failed on Gemini free-tier daily
# embedding quota exhaustion; the rerun is queued behind the quota reset.
LIVE_RUN_PENDING: Final[dict[str, str]] = {
    "status": "pending",
    "reason": (
        "live queued run blocked by Gemini free-tier daily embedding quota "
        "exhaustion (12/15 ingestions failed, 2026-11-10); resume per "
        "scripts/live_measurement/RUNBOOK.md"
    ),
}

_ANNEXURE_LOCUS_PREFIXES: Final[tuple[str, ...]] = (
    "Schedule",
    "Annexure",
    "Annex",
    "Appendix",
    "Exhibit",
)


class OfflineArtifactFidelity(BaseModel):
    """The offline deterministic fidelity measurement for one corpus artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    pages: int
    source_chars: int
    chunk_count: int
    first_chunk_id: str | None = None
    last_chunk_id: str | None = None
    # Coverage/order (real measurement over convention-chunk source spans).
    source_coverage_ratio: float | None = None
    source_order_preserved: bool | None = None
    chunks_with_source_span: int = 0
    chunks_out_of_order: int = 0
    # Structural preservation.
    tables_total: int = 0
    tables_preserved: int = 0
    sections_total: int = 0
    sections_preserved: int = 0
    section_loss_ratio: float | None = None
    # OCR is not measurable offline — an explicit unavailable state, never a zero.
    ocr_status: str = "unavailable"
    ocr_confidence: float | None = None
    recorded_ocr_status: str | None = None
    # Chunk-count plausibility needs the production tokenizer.
    chunk_plausibility_verdict: str = "unavailable"
    chunk_plausibility_reason: str | None = None
    # Contract-boundary identity (real boundary function + per-slice classification).
    boundary_count: int = 0
    boundary_ambiguous: bool = False
    boundary_kinds: list[str] = Field(default_factory=list)
    # Clause identity.
    loci_recovered: int = 0
    annexure_loci: list[str] = Field(default_factory=list)
    clause_kind_counts: dict[str, int] = Field(default_factory=dict)
    # Typed fidelity warnings measured under the versioned default thresholds.
    warning_codes: list[str] = Field(default_factory=list)
    unavailable_metrics: list[str] = Field(default_factory=list)


class RowOutcome(BaseModel):
    """One gold row evaluated against the offline measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    document_id: str
    label: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


class ReportFinding(BaseModel):
    """A fidelity finding the report records without failing the coverage gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    finding: str


class LiveRunState(BaseModel):
    """The live queued-pipeline run state this report was published under."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    reason: str


class GateOutcome(BaseModel):
    """The pilot-blocking verdict over the labeled coverage/order/warning rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: str  # "pass" | "fail"
    basis: str
    rows_passed: int
    rows_failed: int
    failed_case_ids: list[str] = Field(default_factory=list)


class IngestionFidelityReport(BaseModel):
    """Machine-readable ingestion-fidelity report (task 4.7 output shape)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_version: str = REPORT_VERSION
    generated_at_utc: str
    golden_set_version: str
    pipeline_note: str
    limitations: list[str] = Field(default_factory=list)
    live_run: LiveRunState
    artifacts: list[OfflineArtifactFidelity] = Field(default_factory=list)
    rows: list[RowOutcome] = Field(default_factory=list)
    findings: list[ReportFinding] = Field(default_factory=list)
    gate: GateOutcome


class IngestionFidelityEvaluationError(RuntimeError):
    """The offline evaluation could not run over the requested inputs."""


def reconstruct_markdown(pages: list[dict[str, object]]) -> tuple[str, list[int]]:
    """Join recorded page texts into one source markdown with page offsets.

    Mirrors what the live pipeline sees as ``ParsedDocument.markdown``: the
    parsed page texts in order.  Returns the markdown and each page's start
    offset so convention chunks can carry absolute source spans.
    """
    page_texts = [str(page.get("text", "")) for page in pages]
    offsets: list[int] = []
    parts: list[str] = []
    offset = 0
    for text in page_texts:
        offsets.append(offset)
        parts.append(text)
        offset += len(text) + 2  # pages are joined with a blank line
    return "\n\n".join(parts), offsets


def convention_chunks(
    *,
    source_id: str,
    pages: list[dict[str, object]],
    page_offsets: Sequence[int],
) -> list[tuple[str, PreparedChunk]]:
    """Derive the corpus-convention chunks with ids and absolute source spans.

    Chunk ids follow ``legal_retrieval_v2.CHUNKING.md``: ``<SRC-ID>:p<page>:c<ordinal>``
    with a 1-based ordinal within the page.  Spans are located inside the page
    text (never fabricated): a chunk that cannot be located keeps no span,
    which the coverage measurement then honestly reports.
    """
    prepared: list[tuple[str, PreparedChunk]] = []
    for page, page_offset in zip(pages, page_offsets, strict=True):
        raw_page_no = page.get("page_no", 0) or 0
        page_no = int(str(raw_page_no)) if raw_page_no else 0
        page_text = str(page.get("text", ""))
        cursor = 0
        for ordinal, content in enumerate(chunks_for_page(page_text), start=1):
            start = page_text.find(content, cursor)
            if start < 0:
                start = page_text.find(content)
            absolute_start = page_offset + start if start >= 0 else None
            absolute_end = page_offset + start + len(content) if start >= 0 else None
            if start >= 0:
                cursor = start + len(content)
            prepared.append(
                (
                    f"{source_id}:p{page_no}:c{ordinal:02d}",
                    PreparedChunk(
                        chunk_index=len(prepared),
                        chunk_kind="corpus_convention",
                        content=content,
                        source_start_char=absolute_start,
                        source_end_char=absolute_end,
                    ),
                )
            )
    return prepared


def _pipe_tables_in_markdown(markdown: str) -> list[str]:
    """Extract pipe-markdown table blocks (the offline parse-table analogue).

    The recorded corpus text layer carries no pipe tables (pdftotext output is
    plain text), so real artifacts measure zero here — an honest zero for the
    offline path, with the docling-table signal labeled on the hermetic
    fixtures instead.  Synthetic/fixture markdown with real tables measures
    correctly.
    """
    tables: list[str] = []
    for block in re.split(r"\n{2,}", markdown):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) >= 2 and all("|" in line for line in lines):
            tables.append(block)
    return tables


def measure_artifact_fidelity(  # noqa: PLR0914 — one measurement pass keeps every fidelity signal together
    *,
    source_id: str,
    artifact_dir: Path,
    recorded_ocr_status: str | None = None,
) -> OfflineArtifactFidelity:
    """Run the offline deterministic stages over one recorded artifact."""
    pages_path = artifact_dir / "pages.json"
    if not pages_path.is_file():
        message = f"missing pages.json for {source_id} at {pages_path}"
        raise IngestionFidelityEvaluationError(message)
    payload = json.loads(pages_path.read_text(encoding="utf-8"))
    pages: list[dict[str, object]] = list(payload.get("pages", []))
    markdown, page_offsets = reconstruct_markdown(pages)
    convention = convention_chunks(source_id=source_id, pages=pages, page_offsets=page_offsets)
    chunks = [chunk for _chunk_id, chunk in convention]
    ids = [chunk_id for chunk_id, _chunk in convention]

    boundary_result = establish_contract_boundaries(markdown=markdown)
    boundary_kinds = [
        classify_document(markdown=boundary.markdown, filename=f"{source_id}.pdf").document_kind
        for boundary in boundary_result.boundaries
    ]

    parsed = ParsedDocument(
        title=source_id,
        markdown=markdown,
        page_count=len(pages),
        tables=_pipe_tables_in_markdown(markdown),
    )
    chunk_text = "\n".join(chunk.content for chunk in chunks)
    # The boundaries partition the markdown and the convention chunks cover the
    # text they were derived from, so section preservation measured document-wide
    # is equivalent to the pipeline's per-boundary aggregation — without
    # re-normalizing the full chunk text once per boundary (hundreds of
    # boundaries on the large corpus artifacts).
    sections_total, sections_preserved = measure_section_preservation(
        source_markdown=markdown, chunk_text=chunk_text
    )
    tables_total, tables_preserved = measure_table_preservation(
        tables=parsed.tables, chunk_text=chunk_text
    )
    coverage = measure_source_coverage(source_markdown=markdown, chunks=chunks)

    metrics = IngestionFidelityMetrics(
        thresholds=FidelityThresholds(),
        source_coverage_ratio=coverage.coverage_ratio,
        source_order_preserved=coverage.order_preserved,
        chunks_total=len(chunks),
        chunks_with_source_span=coverage.chunks_with_source_span,
        chunks_out_of_order=coverage.chunks_out_of_order,
        ocr_confidence=None,
        ocr_source_coverage_ratio=None,
        ocr_status="unavailable",
        tables_total=tables_total,
        tables_preserved=tables_preserved,
        table_loss_ratio=(
            (tables_total - tables_preserved) / tables_total if tables_total else 0.0
        ),
        sections_total=sections_total,
        sections_preserved=sections_preserved,
        section_loss_ratio=(
            (sections_total - sections_preserved) / sections_total if sections_total else 0.0
        ),
        chunk_plausibility=ChunkPlausibilityState(
            verdict=ChunkPlausibilityVerdict.UNAVAILABLE,
            reason=(
                "Offline measurement does not run the production tokenizer; "
                "chunk-count bounds are not computed here."
            ),
            chunk_count=len(chunks),
            token_count_basis=TokenCountBasis.UNAVAILABLE,
        ),
        unavailable_metrics=(
            "ocr_confidence",
            "ocr_source_coverage_ratio",
            "chunk_plausibility_tokenizer_bounds",
        ),
    )
    warnings = fidelity_degradation_warnings(metrics)

    loci = [locus for chunk in chunks if (locus := recover_locus_from_text(chunk.content))]
    clause_kinds: dict[str, int] = {}
    for chunk in chunks:
        kind = _infer_clause_type(chunk.content)
        clause_kinds[kind] = clause_kinds.get(kind, 0) + 1

    return OfflineArtifactFidelity(
        source_id=source_id,
        pages=len(pages),
        source_chars=len(markdown),
        chunk_count=len(chunks),
        first_chunk_id=ids[0] if ids else None,
        last_chunk_id=ids[-1] if ids else None,
        source_coverage_ratio=coverage.coverage_ratio,
        source_order_preserved=coverage.order_preserved,
        chunks_with_source_span=coverage.chunks_with_source_span,
        chunks_out_of_order=coverage.chunks_out_of_order,
        tables_total=tables_total,
        tables_preserved=tables_preserved,
        sections_total=sections_total,
        sections_preserved=sections_preserved,
        section_loss_ratio=metrics.section_loss_ratio,
        ocr_status="unavailable",
        ocr_confidence=None,
        recorded_ocr_status=recorded_ocr_status,
        chunk_plausibility_verdict=str(ChunkPlausibilityVerdict.UNAVAILABLE.value),
        chunk_plausibility_reason=metrics.chunk_plausibility.reason,
        boundary_count=len(boundary_result.boundaries),
        boundary_ambiguous=boundary_result.ambiguous,
        boundary_kinds=boundary_kinds,
        loci_recovered=len(loci),
        annexure_loci=sorted(
            {locus for locus in loci if locus.startswith(_ANNEXURE_LOCUS_PREFIXES)}
        ),
        clause_kind_counts=dict(sorted(clause_kinds.items())),
        warning_codes=sorted(warning.code for warning in warnings),
        unavailable_metrics=list(metrics.unavailable_metrics),
    )


def evaluate_row(  # noqa: PLR0912 — one branch per labeled expectation dimension
    row: IngestionFidelityRow,
    measurement: OfflineArtifactFidelity,
) -> RowOutcome:
    """Evaluate one gold row's expectation against the offline measurement.

    Dimensions the offline path cannot evaluate (OCR confidence, live parsed
    tables) are recorded as skipped with the reason — never silently passed.
    """
    expectation: IngestionFidelityExpectation = row.expectation
    failures: list[str] = []
    skipped: list[str] = []

    def _check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    if expectation.min_source_coverage is not None:
        if measurement.source_coverage_ratio is None:
            failures.append("source coverage was not measurable")
        else:
            _check(
                measurement.source_coverage_ratio >= expectation.min_source_coverage,
                f"source coverage {measurement.source_coverage_ratio:.4f} < "
                f"{expectation.min_source_coverage}",
            )
    if expectation.order_preserved is not None:
        if measurement.source_order_preserved is None:
            failures.append("source order was not measurable")
        else:
            _check(
                measurement.source_order_preserved is expectation.order_preserved,
                f"source order preserved={measurement.source_order_preserved} != "
                f"{expectation.order_preserved}",
            )
    if expectation.exact_chunk_count is not None:
        _check(
            measurement.chunk_count == expectation.exact_chunk_count,
            f"chunk count {measurement.chunk_count} != {expectation.exact_chunk_count}",
        )
    if expectation.min_chunk_count is not None:
        _check(
            measurement.chunk_count >= expectation.min_chunk_count,
            f"chunk count {measurement.chunk_count} < {expectation.min_chunk_count}",
        )
    if expectation.first_chunk_id is not None:
        _check(
            measurement.first_chunk_id == expectation.first_chunk_id,
            f"first chunk id {measurement.first_chunk_id!r} != {expectation.first_chunk_id!r}",
        )
    if expectation.last_chunk_id is not None:
        _check(
            measurement.last_chunk_id == expectation.last_chunk_id,
            f"last chunk id {measurement.last_chunk_id!r} != {expectation.last_chunk_id!r}",
        )
    if expectation.min_sections_preserved_ratio is not None:
        if measurement.sections_total == 0:
            failures.append("no sections measured to satisfy the preservation floor")
        else:
            ratio = measurement.sections_preserved / measurement.sections_total
            _check(
                ratio >= expectation.min_sections_preserved_ratio,
                f"sections preserved ratio {ratio:.4f} < "
                f"{expectation.min_sections_preserved_ratio}",
            )
    if expectation.min_tables_total is not None:
        skipped.append(
            "parsed-table totals require the docling parse (offline measures the "
            "text-layer pipe tables only)"
        )
    if expectation.min_tables_preserved is not None:
        skipped.append("table preservation requires the docling parse (see evals/ingestion gate)")
    for code in expectation.expected_warning_codes:
        _check(code in measurement.warning_codes, f"expected warning {code} did not fire")
    for code in expectation.forbidden_warning_codes:
        _check(code not in measurement.warning_codes, f"forbidden warning {code} fired")
    if expectation.expected_ocr_status is not None and expectation.expected_ocr_status != (
        measurement.ocr_status
    ):
        skipped.append(
            "live OCR status comes from the docling conversion; offline it is "
            "always explicitly unavailable"
        )
    if expectation.max_ocr_confidence is not None:
        skipped.append("OCR confidence is only measurable by the live docling parse")
    if expectation.exact_contract_boundaries is not None:
        _check(
            measurement.boundary_count == expectation.exact_contract_boundaries,
            f"contract boundaries {measurement.boundary_count} != "
            f"{expectation.exact_contract_boundaries}",
        )
    if expectation.min_contract_boundaries is not None:
        _check(
            measurement.boundary_count >= expectation.min_contract_boundaries,
            f"contract boundaries {measurement.boundary_count} < "
            f"{expectation.min_contract_boundaries}",
        )
    if expectation.min_loci_recovered is not None:
        _check(
            measurement.loci_recovered >= expectation.min_loci_recovered,
            f"loci recovered {measurement.loci_recovered} < {expectation.min_loci_recovered}",
        )
    for prefix in expectation.expected_locus_prefixes:
        _check(
            any(locus.startswith(prefix) for locus in measurement.annexure_loci),
            f"no recovered locus starts with {prefix!r} "
            f"(measured: {measurement.annexure_loci[:5]})",
        )
    for kind, minimum in expectation.expected_clause_kinds.items():
        _check(
            measurement.clause_kind_counts.get(kind, 0) >= minimum,
            f"clause kind {kind!r} count {measurement.clause_kind_counts.get(kind, 0)} < {minimum}",
        )
    passed = not failures and not skipped and bool(expectation.model_dump(exclude_none=True))
    return RowOutcome(
        case_id=row.case_id,
        document_id=row.document_id,
        label=str(row.label),
        passed=passed,
        failures=failures,
        skipped=skipped,
    )


def _recorded_ocr_status_by_id(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(entry.get("source_id")): str(entry.get("ocr_status") or "")
        for entry in manifest.get("per_artifact", [])
        if entry.get("source_id")
    }


def evaluate_gold_set(
    *,
    gold_set_path: Path,
    corpus_root: Path = Path("evals/corpus"),
    live_run: LiveRunState | None = None,
) -> IngestionFidelityReport:
    """Load the validated gold set, measure the corpus, and evaluate every row."""
    load_result = asyncio.run(load_ingestion_fidelity_set(gold_set_path))
    if isinstance(load_result, Failure):
        raise IngestionFidelityEvaluationError(load_result.failure().message)
    gold_set = load_result.unwrap()
    manifest_path = gold_set_path.with_name(f"{gold_set_path.stem}.manifest.json")
    recorded_ocr = _recorded_ocr_status_by_id(manifest_path)

    document_ids = sorted({row.document_id for row in gold_set.rows})
    artifacts = [
        measure_artifact_fidelity(
            source_id=document_id,
            artifact_dir=corpus_root / "artifacts" / document_id,
            recorded_ocr_status=recorded_ocr.get(document_id),
        )
        for document_id in document_ids
    ]
    by_id = {artifact.source_id: artifact for artifact in artifacts}
    rows = [evaluate_row(row, by_id[row.document_id]) for row in gold_set.rows]

    findings = _corpus_findings(artifacts)
    failed = [outcome for outcome in rows if not outcome.passed]
    gate = GateOutcome(
        verdict="fail" if failed else "pass",
        basis=(
            "labeled source-to-chunk coverage/order and typed warning outcomes "
            "over the offline deterministic stages (boundaries, convention "
            "chunks, fidelity signals)"
        ),
        rows_passed=len(rows) - len(failed),
        rows_failed=len(failed),
        failed_case_ids=[outcome.case_id for outcome in failed],
    )
    return IngestionFidelityReport(
        generated_at_utc=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        golden_set_version=gold_set.version,
        pipeline_note=(
            "Offline deterministic path: boundaries (real establish_contract_boundaries), "
            "chunks (recorded corpus convention per legal_retrieval_v2.CHUNKING.md), "
            "fidelity signals (real production measurements). The live queued run is "
            "recorded separately under live_run."
        ),
        limitations=list(OFFLINE_LIMITATIONS),
        live_run=live_run or LiveRunState(**LIVE_RUN_PENDING),
        artifacts=artifacts,
        rows=rows,
        findings=findings,
        gate=gate,
    )


def _corpus_findings(artifacts: list[OfflineArtifactFidelity]) -> list[ReportFinding]:
    """Record cross-checkable fidelity findings the coverage gate does not gate on.

    Boundary detection against the recorded human-verified manifest boundaries
    is a multi-contract-identity finding (measured again by 8.4/8.5), not a
    coverage/order/warning outcome, so it is reported here.
    """
    findings: list[ReportFinding] = []
    manifest_boundaries = _manifest_boundary_counts()
    for artifact in artifacts:
        recorded = manifest_boundaries.get(artifact.source_id)
        if recorded is not None and recorded != artifact.boundary_count:
            findings.append(
                ReportFinding(
                    document_id=artifact.source_id,
                    finding=(
                        f"recorded contract_boundaries={recorded} in the corpus manifest but "
                        f"establish_contract_boundaries measures {artifact.boundary_count}; "
                        "multi-contract identity for this artifact is re-measured by tasks "
                        "8.4/8.5 before Saul reads it"
                    ),
                )
            )
        if artifact.boundary_count > 8:
            findings.append(
                ReportFinding(
                    document_id=artifact.source_id,
                    finding=(
                        f"boundary detection produced {artifact.boundary_count} slices "
                        "(short lines naming an agreement/contract split the source); "
                        "per-slice metadata stays correct but boundary identity is "
                        "fragmented — flagged for the conservative-marker follow-up"
                    ),
                )
            )
    return findings


def _manifest_boundary_counts() -> dict[str, int]:
    manifest_path = Path("evals/golden/legal_retrieval_v2.manifest.json")
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(entry.get("source_id")): len(entry.get("contract_boundaries") or [])
        for entry in manifest.get("per_artifact", [])
        if entry.get("source_id")
    }


def main() -> None:
    """CLI: measure the corpus and write the machine-readable fidelity report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold_set", type=Path, help="Path to the ingestion gold-set JSONL")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("evals/corpus"),
        help="Corpus root (default: evals/corpus)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Report JSON output path")
    args = parser.parse_args()
    report = evaluate_gold_set(gold_set_path=args.gold_set, corpus_root=args.corpus_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(
        f"{args.output}: gate={report.gate.verdict} "
        f"({report.gate.rows_passed}/{report.gate.rows_passed + report.gate.rows_failed} rows, "
        f"{len(report.artifacts)} artifacts, {len(report.findings)} findings)\n"
    )


if __name__ == "__main__":
    main()
