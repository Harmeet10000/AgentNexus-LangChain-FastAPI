"""Fidelity-visibility fixtures (I2, task 4.4).

Scanned-PDF, table-heavy, and clause-numbered documents are ingested through
the real measurement chain (``build_fidelity_metrics`` +
``fidelity_degradation_warnings``, the same seam the queued pipeline runs
between chunk annotation and embedding) and the assertions cover exactly what
Agent Saul can see BEFORE any analysis starts:

- the document record's typed ``completed_with_warnings`` reasons (the
  ``FIDELITY_*`` codes riding the persisted chunk warnings that
  ``fetch_status`` rolls up into ``DocumentStatusResponse.warnings``);
- the source-fidelity result in the record's ``fidelity`` metrics (source
  coverage and order, OCR status, structural preservation) — decoded exactly
  the way the status endpoint decodes the persisted metadata;
- the pre-dispatch ``get_ingestion_state`` tool result (the ingestion truth
  Saul reads before planning), whose state degrades to
  ``completed_with_warnings`` and whose coverage counts the warned document.

Real scanned binaries are unavailable to the unit suite, so the scanned
profile is synthesized (a re-rendered scanned page in the SYN-05 style: the
page text layer is recovered, docling measured a low OCR confidence). The
OCR-confidence warning is report-only under the production default thresholds
(``ocr_confidence_min=0.0``), so the scanned fixture asserts both halves of
the contract: the measured-low metric is visible in the record, and the typed
``FIDELITY_OCR_CONFIDENCE_LOW`` reason fires and degrades the terminal state
as soon as the versioned threshold set enables it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from returns.result import Success

from app.features.documents.api.http.dto import DocumentStatusResponse
from app.features.documents.application.commands.upload_document import (
    _fidelity_metrics_from_metadata,
)
from app.features.documents.application.evidence.service import EvidenceService
from app.features.documents.application.ingestion.chunking.classification import (
    ParsedDocument,
    PreparedChunk,
)
from app.features.documents.application.ingestion.chunking.fidelity import (
    build_fidelity_metrics,
    fidelity_degradation_warnings,
)
from app.features.documents.application.queries.documents import _flatten_warnings
from app.features.documents.domain.fidelity import FidelityThresholds

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

# ---------------------------------------------------------------------------
# Synthesized fixtures (real scanned binaries are unavailable here)
# ---------------------------------------------------------------------------
#
# SYN-05 analogue: a re-rendered scanned page. The text layer was recovered
# (docling's ocr_source_coverage_ratio is high) but the OCR engine measured a
# low confidence — exactly the profile the V9 spike recorded for scanned scans.
SCANNED_PAGE_MARKDOWN = (
    "PAGE 1 — RECORDED DEED OF VARIATION (SCANNED)\n\n"
    "TH1S DEED 0F VAR1AT1ON 1s made on 4 September 2026 between GREENF1ELD "
    "H0LDINGS PTY LTD (the 0wner) and MERID1AN C0NSTRUCT1ON LTD (the "
    "C0ntractor).\n\n"
    "1. The part1es agree that clause 4 0f the pr1nc1pal contract 1s amended "
    "t0 extend the complet10n date t0 31 March 2027.\n\n"
    "2. 4ll 0ther terms 0f the pr1nc1pal contract rema1n 1n full f0rce."
)

# Table-heavy profile: a schedule whose obligations are tabular. The chunk
# seam below drops the second table entirely — the canonical table-loss shape.
TABLE_HEAVY_MARKDOWN = (
    "# SCHEDULE 2 — PAYMENT MILESTONES\n\n"
    "The following milestones govern payment under the principal agreement.\n\n"
    "| Milestone | Amount (AUD) | Due |\n"
    "|---|---|---|\n"
    "| Design complete | 120,000 | Day 30 |\n"
    "| Foundations poured | 250,000 | Day 90 |\n\n"
    "| Retention | Percentage | Release |\n"
    "|---|---|---|\n"
    "| Practical completion | 5% | Day 180 |\n"
    "| Defects liability | 5% | Day 365 |\n\n"
    "Payments are made within fourteen days of a valid tax invoice."
)

# Clause-numbered profile: three numbered clauses, of which the chunk seam
# below keeps only the first — the canonical section-loss shape (loss ratio
# 2/3 exceeds the 0.5 default threshold).
CLAUSE_NUMBERED_MARKDOWN = (
    "# SERVICE AGREEMENT — CONDITIONS\n\n"
    "## 1. Definitions\n\n"
    "Agreement means this service agreement as varied from time to time.\n\n"
    "## 2. Termination\n\n"
    "Either party may terminate for material breach not remedied within "
    "fourteen days of written notice.\n\n"
    "## 3. Governing Law\n\n"
    "This agreement is governed by the laws of New South Wales."
)

# The healthy control: everything survives, no OCR signal, nothing degraded.
HEALTHY_MARKDOWN = (
    "# CONSULTANCY AGREEMENT\n\n"
    "This consultancy agreement is made on 2 February 2026 between the "
    "Client and the Consultant.\n\n"
    "The Consultant shall provide the services described in Schedule 1."
)


def _chunk(content: str, index: int, markdown: str, cursor: int) -> tuple[PreparedChunk, int]:
    start = markdown.find(content, cursor)
    assert start >= 0, f"fixture part not located in source: {content[:40]!r}"
    end = start + len(content)
    chunk = PreparedChunk(
        chunk_index=index,
        chunk_kind="generic",
        content=content,
        source_start_char=start,
        source_end_char=end,
    )
    return chunk, end


def _spanned_chunks(markdown: str, parts: Sequence[str]) -> list[PreparedChunk]:
    """Chunks with real, in-order source spans located inside the markdown."""
    chunks: list[PreparedChunk] = []
    cursor = 0
    for index, part in enumerate(parts):
        chunk, cursor = _chunk(part, index, markdown, cursor)
        chunks.append(chunk)
    return chunks


class _WordCountTokenizer:
    """Deterministic ~1-token-per-word counter so plausibility is exact."""

    def encode(self, text: str) -> list[int]:
        return list(range(1, len(str(text).split()) + 1))


@pytest.fixture(autouse=True)
def _word_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.features.documents.application.ingestion.chunking.fidelity as fidelity_module

    monkeypatch.setattr(fidelity_module, "get_tokenizer", _WordCountTokenizer)


def measure_fixture_fidelity(
    *,
    markdown: str,
    chunks: Sequence[PreparedChunk],
    document_kind: str = "generic",
    ocr_confidence: float | None = None,
    ocr_source_coverage_ratio: float | None = None,
    thresholds: FidelityThresholds | None = None,
):
    """Real measurement seam: one boundary carrying this fixture's chunks."""
    from app.features.documents.application.ingestion.chunking.fidelity import (
        BoundaryFidelityInputs,
    )

    parsed = ParsedDocument(
        title="fixture",
        markdown=markdown,
        page_count=1,
        ocr_confidence=ocr_confidence,
        ocr_source_coverage_ratio=ocr_source_coverage_ratio,
        tables=[block for block in markdown.split("\n\n") if block.lstrip().startswith("|")],
    )
    return build_fidelity_metrics(
        parsed=parsed,
        boundaries=(
            BoundaryFidelityInputs(
                markdown=markdown, chunks=tuple(chunks), document_kind=document_kind
            ),
        ),
        chunk_count=len(chunks),
        document_kind=document_kind,
        thresholds=thresholds,
    )


def _persisted_record(
    *,
    document_id: str,
    metrics,
    warnings: Sequence[Any],
) -> dict[str, object]:
    """The fetch_status record shape for a document whose chunks carry warnings.

    Mirrors the queued pipeline's persistence: fidelity warnings ride the chunk
    ``quality_warnings`` (jsonb-agg'd by ``fetch_status``) and the measured
    fidelity metrics land in the document metadata under ``fidelity``.
    """
    chunk_warnings = [warning.model_dump() for warning in warnings]
    return {
        "document_id": document_id,
        "status": "completed_with_warnings" if warnings else "completed",
        "object_uri": f"s3://fixtures/{document_id}.md",
        "title": document_id,
        "document_kind": "generic",
        "chunk_count": 3,
        "verified_chunk_count": 0,
        "document_version": 1,
        "extraction_incomplete": False,
        "warnings": [chunk_warnings],
        "document_metadata": {"fidelity": metrics.model_dump(mode="json")},
    }


class _StatusRepo:
    """EvidenceService repo stub exposing only the ingestion-state read."""

    def __init__(self, records: dict[str, dict[str, object]]) -> None:
        self._records = records

    async def fetch_status(
        self, *, user_id: str, document_id: str, tenant_id: str | None = None
    ) -> Success[dict[str, object] | None]:
        return Success(self._records.get(document_id))


async def _get_ingestion_state(records: dict[str, dict[str, object]], document_id: str):
    service = EvidenceService(repo=_StatusRepo(records))
    return await service.get_ingestion_state(
        principal_id="user-fidelity",
        tenant_id="personal:user-fidelity",
        allowed_doc_ids=[document_id],
        document_ids=[document_id],
    )


# ---------------------------------------------------------------------------
# Scanned-PDF profile: measured-low OCR is visible; the typed reason degrades


def test_scanned_profile_records_measured_low_ocr_before_analysis() -> None:
    chunks = _spanned_chunks(SCANNED_PAGE_MARKDOWN, [SCANNED_PAGE_MARKDOWN])
    metrics = measure_fixture_fidelity(
        markdown=SCANNED_PAGE_MARKDOWN,
        chunks=chunks,
        ocr_confidence=0.31,
        ocr_source_coverage_ratio=0.9,
    )
    assert metrics.ocr_status == "measured"
    assert metrics.ocr_confidence == pytest.approx(0.31)
    assert metrics.source_coverage_ratio == pytest.approx(1.0)
    assert metrics.source_order_preserved is True
    # Production defaults keep OCR report-only: no warning fires, but the
    # measured-low metric is already part of the document record.
    assert fidelity_degradation_warnings(metrics) == []

    record = _persisted_record(document_id="SYN-05-scanned", metrics=metrics, warnings=[])
    fidelity = _fidelity_metrics_from_metadata(record["document_metadata"])
    assert fidelity is not None
    assert fidelity.ocr_confidence == pytest.approx(0.31)
    assert fidelity.ocr_status == "measured"


async def test_scanned_profile_fires_typed_ocr_warning_when_threshold_enabled() -> None:
    chunks = _spanned_chunks(SCANNED_PAGE_MARKDOWN, [SCANNED_PAGE_MARKDOWN])
    thresholds = FidelityThresholds(ocr_confidence_min=0.5)
    metrics = measure_fixture_fidelity(
        markdown=SCANNED_PAGE_MARKDOWN,
        chunks=chunks,
        ocr_confidence=0.31,
        ocr_source_coverage_ratio=0.9,
        thresholds=thresholds,
    )
    warnings = fidelity_degradation_warnings(metrics)
    codes = {warning.code for warning in warnings}
    assert "FIDELITY_OCR_CONFIDENCE_LOW" in codes
    message = next(w.message for w in warnings if w.code == "FIDELITY_OCR_CONFIDENCE_LOW")
    assert "0.310" in message
    assert "0.500" in message

    record = _persisted_record(document_id="SYN-05-scanned", metrics=metrics, warnings=warnings)
    visible = _flatten_warnings(record["warnings"])
    assert {warning.code for warning in visible} >= {"FIDELITY_OCR_CONFIDENCE_LOW"}
    response = DocumentStatusResponse(
        doc_id="SYN-05-scanned",
        status=str(record["status"]),
        object_uri=str(record["object_uri"]),
        title=str(record["title"]),
        document_kind=str(record["document_kind"]),
        chunk_count=3,
        verified_chunk_count=0,
        warning_count=len(visible),
        warnings=visible,
        fidelity=_fidelity_metrics_from_metadata(record["document_metadata"]),
    )
    assert response.status == "completed_with_warnings"
    assert response.fidelity is not None
    assert response.fidelity.ocr_confidence == pytest.approx(0.31)
    ingestion = await _get_ingestion_state({"SYN-05-scanned": record}, "SYN-05-scanned")
    assert ingestion.documents[0].state == "completed_with_warnings"
    assert ingestion.documents[0].warning_count >= 1
    assert ingestion.coverage.documents_with_warnings == 1


# ---------------------------------------------------------------------------
# Table-heavy profile: parsed-table loss fires under production defaults


async def test_table_heavy_profile_degrades_with_typed_table_loss_reason() -> None:
    # Keep the heading, the intro and the milestones table; drop the retention
    # table entirely (the canonical table-loss shape).
    kept_text = TABLE_HEAVY_MARKDOWN.split("\n\n| Retention", maxsplit=1)[0]
    kept, _cursor = _chunk(kept_text, 0, TABLE_HEAVY_MARKDOWN, 0)
    invoice, _cursor = _chunk(
        "Payments are made within fourteen days of a valid tax invoice.",
        1,
        TABLE_HEAVY_MARKDOWN,
        kept.source_end_char,
    )
    chunks = [kept, invoice]
    metrics = measure_fixture_fidelity(
        markdown=TABLE_HEAVY_MARKDOWN, chunks=chunks, document_kind="legal_contract"
    )
    assert metrics.tables_total == 2
    assert metrics.tables_preserved == 1
    codes = {warning.code for warning in fidelity_degradation_warnings(metrics)}
    assert "FIDELITY_TABLE_LOSS" in codes
    # The dropped table's text is genuinely uncovered (coverage < 1.0), but
    # source coverage is report-only under the production defaults — the
    # enforced degradation here is the table loss itself.
    assert metrics.source_coverage_ratio < 1.0
    assert "FIDELITY_SOURCE_COVERAGE_LOW" not in codes
    assert metrics.source_order_preserved is True

    warnings = fidelity_degradation_warnings(metrics)
    record = _persisted_record(document_id="SRC-table-heavy", metrics=metrics, warnings=warnings)
    visible = _flatten_warnings(record["warnings"])
    # Dropping the whole retention table loses both its parsed-table content
    # and its section text, so both typed reasons are visible.
    assert {"FIDELITY_TABLE_LOSS", "FIDELITY_SECTION_LOSS"} <= {warning.code for warning in visible}
    table_message = next(
        warning.message for warning in visible if warning.code == "FIDELITY_TABLE_LOSS"
    )
    assert table_message == "1 of 2 parsed tables did not survive into chunks."
    ingestion = await _get_ingestion_state({"SRC-table-heavy": record}, "SRC-table-heavy")
    assert ingestion.status == "ok"
    assert ingestion.documents[0].state == "completed_with_warnings"
    assert ingestion.coverage.documents_with_warnings == 1
    assert ingestion.coverage.documents_ready == 0


# ---------------------------------------------------------------------------
# Clause-numbered profile: section loss fires under production defaults


async def test_clause_numbered_profile_degrades_with_typed_section_loss_reason() -> None:
    kept = CLAUSE_NUMBERED_MARKDOWN.split("\n\n## 2.", maxsplit=1)[0]
    chunks = _spanned_chunks(CLAUSE_NUMBERED_MARKDOWN, [kept])
    metrics = measure_fixture_fidelity(
        markdown=CLAUSE_NUMBERED_MARKDOWN, chunks=chunks, document_kind="legal_contract"
    )
    assert metrics.sections_total == 3
    assert metrics.sections_preserved == 1
    codes = {warning.code for warning in fidelity_degradation_warnings(metrics)}
    assert "FIDELITY_SECTION_LOSS" in codes

    warnings = fidelity_degradation_warnings(metrics)
    record = _persisted_record(
        document_id="SRC-clause-numbered", metrics=metrics, warnings=warnings
    )
    visible = _flatten_warnings(record["warnings"])
    assert "FIDELITY_SECTION_LOSS" in {warning.code for warning in visible}
    ingestion = await _get_ingestion_state({"SRC-clause-numbered": record}, "SRC-clause-numbered")
    assert ingestion.documents[0].state == "completed_with_warnings"
    # The source-fidelity result rides the same record: coverage and order are
    # measured (not unavailable) even when a structural warning degraded state.
    assert ingestion.coverage.documents_with_warnings == 1
    fidelity = _fidelity_metrics_from_metadata(record["document_metadata"])
    assert fidelity is not None
    assert fidelity.source_coverage_ratio is not None
    assert fidelity.source_order_preserved is True


# ---------------------------------------------------------------------------
# Healthy control: no fidelity signal, the record stays ready


async def test_healthy_document_stays_ready_with_zero_fidelity_warnings() -> None:
    chunks = _spanned_chunks(HEALTHY_MARKDOWN, [HEALTHY_MARKDOWN])
    metrics = measure_fixture_fidelity(markdown=HEALTHY_MARKDOWN, chunks=chunks)
    assert fidelity_degradation_warnings(metrics) == []
    assert metrics.source_coverage_ratio == pytest.approx(1.0)
    assert metrics.source_order_preserved is True
    assert metrics.section_loss_ratio == pytest.approx(0.0)

    record = _persisted_record(document_id="SRC-healthy", metrics=metrics, warnings=[])
    visible = _flatten_warnings(record["warnings"])
    assert visible == []
    assert record["status"] == "completed"
    ingestion = await _get_ingestion_state({"SRC-healthy": record}, "SRC-healthy")
    assert ingestion.documents[0].state == "ready"
    assert ingestion.coverage.documents_with_warnings == 0
    assert ingestion.coverage.documents_ready == 1
