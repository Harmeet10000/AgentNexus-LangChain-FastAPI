"""Ingestion truthfulness tests: relationship-phase states and fidelity metrics.

Regression net for the ``all([])`` vacuity defect: a relationship phase that
never ran must surface as ``skipped_not_implemented``/``not_attempted`` — never
as success.  Also covers the persisted fidelity metrics, their explicit
unavailable states, and the zero-chunk untruthful-success detection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import langextract as lx
import pytest
from returns.result import Success

from app.extraction.langextract.service import AsyncExtractionService, ExtractionSucceeded
from app.features.documents.api.http.dto import IngestionJob
from app.features.documents.application.commands import upload_document as document_ingest
from app.features.documents.application.commands.upload_document import (
    _write_extracted_clause_episodes,
    process_document_ingestion,
)
from app.features.documents.domain.fidelity import (
    ChunkPlausibilityVerdict,
    FidelityThresholds,
    GraphRelationshipsState,
    GraphRelationshipsStatus,
    TokenCountBasis,
    aggregate_relationship_states,
)
from app.features.documents.ingestion.chunking.classification import (
    ClassifiedDocument,
    ParsedDocument,
    PreparedChunk,
)
from app.features.documents.ingestion.chunking.fidelity import (
    BoundaryFidelityInputs,
    build_fidelity_metrics,
    fidelity_degradation_warnings,
    measure_chunk_plausibility,
    measure_source_coverage,
    measure_table_preservation,
)

if TYPE_CHECKING:
    from typing import Any

    from app.extraction.langextract.service import ExtractionRequest


# ---------------------------------------------------------------------------
# Fakes


class _ObjectStore:
    async def get_object(self, *, key: str) -> Success[bytes]:
        del key
        return Success(b"fixture")


class _Repo:
    def __init__(self) -> None:
        self.status_calls: list[dict[str, Any]] = []
        self.chunk_rows: list[dict[str, Any]] = []

    async def update_document_status(self, **kwargs: Any) -> Success[None]:
        self.status_calls.append(kwargs)
        return Success(None)

    async def upsert_chunks(self, rows: object) -> Success[None]:
        self.chunk_rows = list(cast("Any", rows))
        return Success(None)


def _job() -> IngestionJob:
    return IngestionJob(
        document_id="00000000-0000-0000-0000-000000000001",
        user_id="tenant-1",
        tenant_id="personal:tenant-1",
        filename="fixture.md",
        content_type="text/markdown",
        object_uri="s3://bucket/fixture.md",
    )


def _markdown() -> str:
    return "# Terms\n\nPayment is due in thirty days.\n\nTermination requires notice."


def _chunk(content: str, index: int = 0) -> PreparedChunk:
    return PreparedChunk(chunk_index=index, chunk_kind="generic", content=content)


async def _run_ingestion(
    monkeypatch: Any,
    provider: object,
    *,
    graph_writer: object | None = None,
    idempotency: object | None = None,
    chunks: list[PreparedChunk] | None = None,
) -> tuple[_Repo, Any]:
    repo = _Repo()

    async def parse_document(**_kwargs: object) -> ParsedDocument:
        return ParsedDocument(title="Fixture", markdown=_markdown(), page_count=1)

    async def segment_chunks(
        **_kwargs: object,
    ) -> tuple[list[PreparedChunk], list[object]]:
        return list(
            chunks if chunks is not None else [_chunk("Payment is due in thirty days.")]
        ), []

    async def embed_chunks(**kwargs: object) -> list[dict[str, object]]:
        outcome = cast("Any", kwargs["extraction_outcome"])
        return [
            {
                "id": f"10000000-0000-0000-0000-00000000000{index}",
                "chunk_index": index,
                "custom_metadata": document_ingest._extraction_metadata(outcome),
            }
            for index in range(len(cast("Any", kwargs["chunks"])))
        ]

    class RecordingProvider:
        def extract(self, request: ExtractionRequest) -> tuple[lx.data.AnnotatedDocument, ...]:
            del request
            return cast("Any", provider)()

    monkeypatch.setattr(document_ingest, "parse_document", parse_document)
    monkeypatch.setattr(
        document_ingest,
        "classify_document",
        lambda **_kwargs: ClassifiedDocument(document_kind="generic"),
    )
    monkeypatch.setattr(document_ingest, "segment_chunks", segment_chunks)
    monkeypatch.setattr(document_ingest, "_embed_chunks", embed_chunks)

    runtime = SimpleNamespace(
        object_store=_ObjectStore(),
        repo=repo,
        graphiti=None,
        llm=object(),
        extraction=AsyncExtractionService(RecordingProvider()),
        graph_writer=graph_writer,
        idempotency=idempotency,
    )
    result = await process_document_ingestion(job=_job(), runtime=cast("Any", runtime))
    return repo, result


class _SucceedingWriter:
    async def write_clause_episode(self, clause_text: str, metadata: object) -> str:
        del clause_text, metadata
        return "episode-1"


class _Idempotency:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self.values.get(key)

    async def set(self, key: str, result: object, **_kwargs: object) -> None:
        self.values[key] = result


# ---------------------------------------------------------------------------
# Relationship-phase truth through the pipeline


async def test_relationship_phase_persists_skipped_not_implemented(
    monkeypatch: Any,
) -> None:
    """Regression: the empty relationship list must never collapse into success."""
    extraction = lx.data.Extraction(
        extraction_class="payment",
        extraction_text="Payment is due in thirty days.",
        char_interval=lx.data.CharInterval(start_pos=14, end_pos=44),
        attributes={"clause_id": "4.1"},
    )

    def extract() -> tuple[lx.data.AnnotatedDocument, ...]:
        return (lx.data.AnnotatedDocument(text=_markdown(), extractions=[extraction]),)

    repo, result = await _run_ingestion(
        monkeypatch,
        extract,
        graph_writer=_SucceedingWriter(),
        idempotency=_Idempotency(),
    )

    assert result.unwrap()["graph_relationships_status"] == "skipped_not_implemented"
    for status_call in repo.status_calls:
        if "metadata_" not in status_call:
            continue
        persisted = status_call["metadata_"]["graph_relationships"]
        assert persisted["status"] == "skipped_not_implemented"
        assert persisted["reason"] == "relationship_extraction_not_implemented"
        assert persisted["relationships_attempted"] == 0


async def test_relationship_phase_not_attempted_without_segments(
    monkeypatch: Any,
) -> None:
    repo, result = await _run_ingestion(monkeypatch, lambda: ())

    assert result.unwrap()["graph_relationships_status"] == "not_attempted"
    persisted = repo.status_calls[-1]["metadata_"]["graph_relationships"]
    assert persisted["status"] == "not_attempted"


async def test_zero_chunks_for_non_empty_document_is_not_a_success(
    monkeypatch: Any,
) -> None:
    """Untruthful-success fixture: empty chunk output must degrade, not complete."""
    repo, result = await _run_ingestion(monkeypatch, lambda: (), chunks=[])

    payload = result.unwrap()
    assert payload["status"] == "completed_with_warnings"
    fidelity = repo.status_calls[-1]["metadata_"]["fidelity"]
    assert fidelity["chunk_plausibility"]["verdict"] == "implausible"
    assert "zero chunks" in fidelity["chunk_plausibility"]["reason"]
    # Fidelity arrives only after chunking; the pre-chunking checkpoint must not
    # pretend it exists yet.
    assert "fidelity" not in repo.status_calls[0]["metadata_"]


async def test_write_phase_reports_relationships_skipped_with_clause_counts() -> None:
    extraction = lx.data.Extraction(
        extraction_class="payment",
        extraction_text="Payment is due.",
        char_interval=lx.data.CharInterval(start_pos=0, end_pos=15),
    )
    outcome = ExtractionSucceeded(
        documents=(lx.data.AnnotatedDocument(text="Payment is due.", extractions=[extraction]),)
    )
    result = await _write_extracted_clause_episodes(
        outcome=outcome,
        graph_writer=_SucceedingWriter(),
        idempotency=_Idempotency(),
        document_id="doc-1",
        user_id="tenant-1",
        jurisdiction="India",
        document_type="contract",
    )
    assert result.success is True
    assert result.clause_writes_attempted == 1
    assert result.clause_writes_succeeded == 1
    assert result.relationships.status is GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED


# ---------------------------------------------------------------------------
# Aggregate relationship states


def test_aggregate_relationship_states_prefers_failure() -> None:
    failed = GraphRelationshipsState(
        status=GraphRelationshipsStatus.FAILED,
        reason="graph_write_failed",
        relationships_attempted=2,
    )
    skipped = GraphRelationshipsState(
        status=GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED,
        reason="relationship_extraction_not_implemented",
    )
    aggregate = aggregate_relationship_states([skipped, failed])
    assert aggregate.status is GraphRelationshipsStatus.FAILED
    assert aggregate.relationships_attempted == 2


def test_aggregate_relationship_states_distinct_empty_cases() -> None:
    skipped = GraphRelationshipsState(
        status=GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED,
        reason="relationship_extraction_not_implemented",
    )
    not_attempted = GraphRelationshipsState(
        status=GraphRelationshipsStatus.NOT_ATTEMPTED,
        reason="no_clause_segments_extracted",
    )
    assert aggregate_relationship_states([]).status is GraphRelationshipsStatus.NOT_ATTEMPTED
    assert (
        aggregate_relationship_states([not_attempted]).status
        is GraphRelationshipsStatus.NOT_ATTEMPTED
    )
    assert (
        aggregate_relationship_states([skipped]).status
        is GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED
    )


# ---------------------------------------------------------------------------
# Checkpoint reuse and partial-failure aggregation


async def test_checkpoint_reuse_restores_persisted_relationship_state() -> None:
    """A retry rehydrating the checkpoint must not overwrite the truth it carries."""
    from types import SimpleNamespace

    from returns.result import Success as _Success

    from app.features.documents.application.commands.upload_document import (
        _extract_document_knowledge,
    )
    from app.features.documents.ingestion.chunking.classification import ContractBoundary

    class _CheckpointRepo:
        async def fetch_status(self, **_kwargs: object) -> _Success[dict[str, Any]]:
            return _Success(
                {
                    "document_metadata": {
                        "knowledge_extraction": "complete",
                        "knowledge_extraction_document_version": 1,
                        "knowledge_extraction_classes": ["payment"],
                        "graph_relationships": {
                            "status": "skipped_not_implemented",
                            "reason": "relationship_extraction_not_implemented",
                            "relationships_attempted": 0,
                            "relationships_succeeded": 0,
                        },
                    }
                }
            )

    runtime = SimpleNamespace(repo=_CheckpointRepo(), extraction=object())
    _outcome, state = await _extract_document_knowledge(
        runtime=cast("Any", runtime),
        job=_job(),
        boundaries=[
            ContractBoundary(
                contract_index=0,
                markdown="clause",
                start_char=0,
                end_char=6,
            )
        ],
        jurisdiction=None,
        document_type="generic",
    )
    assert state.status is GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED
    assert state.reason == "relationship_extraction_not_implemented"


async def test_checkpoint_reuse_falls_back_to_not_attempted_without_state() -> None:
    from types import SimpleNamespace

    from returns.result import Success as _Success

    from app.features.documents.application.commands.upload_document import (
        _extract_document_knowledge,
    )
    from app.features.documents.ingestion.chunking.classification import ContractBoundary

    class _LegacyCheckpointRepo:
        async def fetch_status(self, **_kwargs: object) -> _Success[dict[str, Any]]:
            return _Success(
                {
                    "document_metadata": {
                        "knowledge_extraction": "complete",
                        "knowledge_extraction_document_version": 1,
                    }
                }
            )

    runtime = SimpleNamespace(repo=_LegacyCheckpointRepo(), extraction=object())
    _outcome, state = await _extract_document_knowledge(
        runtime=cast("Any", runtime),
        job=_job(),
        boundaries=[
            ContractBoundary(contract_index=0, markdown="clause", start_char=0, end_char=6)
        ],
        jurisdiction=None,
        document_type="generic",
    )
    assert state.status is GraphRelationshipsStatus.NOT_ATTEMPTED
    assert state.reason == "knowledge_extraction_checkpoint_reused"


async def test_later_boundary_extraction_failure_keeps_earlier_states() -> None:
    from types import SimpleNamespace

    import langextract as lx

    from app.extraction.langextract.service import ExtractionFailed
    from app.features.documents.application.commands.upload_document import (
        _extract_document_knowledge,
    )
    from app.features.documents.ingestion.chunking.classification import ContractBoundary

    class _SwitchProvider:
        """Succeeds for boundary 0, raises (provider error) for boundary 1."""

        def extract(self, request: Any) -> tuple[Any, ...]:
            if request.contract_index == 0:
                return (
                    lx.data.AnnotatedDocument(
                        text="clause one",
                        extractions=[
                            lx.data.Extraction(
                                extraction_class="payment",
                                extraction_text="clause one",
                                char_interval=lx.data.CharInterval(start_pos=0, end_pos=10),
                            )
                        ],
                    ),
                )
            message = "provider down"
            raise RuntimeError(message)

    class _Writer:
        async def write_clause_episode(self, clause_text: str, metadata: object) -> str:
            del clause_text, metadata
            return "episode-1"

    runtime = SimpleNamespace(
        repo=_Repo(),
        extraction=AsyncExtractionService(_SwitchProvider()),
        graph_writer=_Writer(),
        idempotency=_Idempotency(),
    )
    boundaries = [
        ContractBoundary(contract_index=0, markdown="clause one", start_char=0, end_char=10),
        ContractBoundary(contract_index=1, markdown="clause two", start_char=0, end_char=10),
    ]
    outcome, state = await _extract_document_knowledge(
        runtime=cast("Any", runtime),
        job=_job(),
        boundaries=boundaries,
        jurisdiction=None,
        document_type="generic",
    )
    assert isinstance(outcome, ExtractionFailed)
    # Boundary 0's skipped-not-implemented truth must survive boundary 1's
    # extraction failure — the aggregate never collapses back to not_attempted.
    assert state.status is GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED
    assert state.reason == "relationship_extraction_not_implemented"


# ---------------------------------------------------------------------------
# Fidelity measurements


def test_source_coverage_measures_union_and_order() -> None:
    markdown = "abcdef"
    chunks = [
        _chunk("ab").model_copy(update={"source_start_char": 0, "source_end_char": 2}),
        _chunk("cd").model_copy(update={"source_start_char": 2, "source_end_char": 4}),
    ]
    coverage = measure_source_coverage(source_markdown=markdown, chunks=chunks)
    assert coverage.coverage_ratio == pytest.approx(4 / 6)
    assert coverage.order_preserved is True
    assert coverage.chunks_out_of_order == 0


def test_source_order_violation_is_detected_not_vacuous() -> None:
    chunks = [
        _chunk("cd").model_copy(update={"source_start_char": 2, "source_end_char": 4}),
        _chunk("ab").model_copy(update={"source_start_char": 0, "source_end_char": 2}),
    ]
    coverage = measure_source_coverage(source_markdown="abcd", chunks=chunks)
    assert coverage.order_preserved is False
    assert coverage.chunks_out_of_order == 1


def test_unlocated_chunks_reduce_coverage() -> None:
    chunks = [_chunk("xx")]  # no source spans
    coverage = measure_source_coverage(source_markdown="abcdef", chunks=chunks)
    assert coverage.coverage_ratio <= 0.0
    assert coverage.order_preserved is None


def test_table_preservation_counts_survivors() -> None:
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    total, preserved = measure_table_preservation(
        tables=[table, "| gone | |"],
        chunk_text=f"preamble {table} suffix",
    )
    assert total == 2
    assert preserved == 1


def test_chunk_plausibility_zero_chunks_for_non_empty_document() -> None:
    state = measure_chunk_plausibility(
        source_markdown="Some real content.", chunk_count=0, document_kind="generic"
    )
    assert state.verdict is ChunkPlausibilityVerdict.IMPLAUSIBLE
    assert state.reason is not None
    assert "zero chunks" in state.reason


class _FixedTokenizer:
    """Deterministic token counter so bounds tests never load the real model."""

    def __init__(self, token_count: int) -> None:
        self._token_count = token_count

    def encode(self, _text: str) -> list[int]:
        return [0] * self._token_count


def test_chunk_plausibility_out_of_bounds_count(monkeypatch: Any) -> None:
    import app.features.documents.ingestion.chunking.fidelity as fidelity_module

    monkeypatch.setattr(fidelity_module, "get_tokenizer", lambda: _FixedTokenizer(10_000))
    state = fidelity_module.measure_chunk_plausibility(
        source_markdown="Word " * 10_000, chunk_count=2, document_kind="generic"
    )
    assert state.token_count_basis is TokenCountBasis.TOKENIZER
    assert state.expected_chunk_count_lower is not None
    assert state.expected_chunk_count_lower > 2
    assert state.verdict is ChunkPlausibilityVerdict.IMPLAUSIBLE


def test_chunk_plausibility_degrades_to_unavailable_without_tokenizer(
    monkeypatch: Any,
) -> None:
    def _boom() -> object:
        message = "offline"
        raise RuntimeError(message)

    import app.features.documents.ingestion.chunking.fidelity as fidelity_module

    monkeypatch.setattr(fidelity_module, "get_tokenizer", _boom)
    state = fidelity_module.measure_chunk_plausibility(
        source_markdown="Content.", chunk_count=1, document_kind="generic"
    )
    assert state.verdict is ChunkPlausibilityVerdict.UNAVAILABLE
    assert state.token_count_basis is TokenCountBasis.UNAVAILABLE


def test_degradation_warnings_fire_on_threshold_breaches() -> None:
    metrics = build_fidelity_metrics(
        parsed=ParsedDocument(
            title="Doc",
            markdown="# H\n\nAlpha body.\n\nBeta body.",
            page_count=1,
            tables=["| A | B |"],
        ),
        boundaries=(
            BoundaryFidelityInputs(markdown="# H\n\nAlpha body.\n\nBeta body.", chunks=()),
        ),
        chunk_count=0,
        document_kind="generic",
        thresholds=FidelityThresholds(source_coverage_min=0.8),
    )
    codes = {warning.code for warning in fidelity_degradation_warnings(metrics)}
    assert "FIDELITY_TABLE_LOSS" in codes
    assert "FIDELITY_SECTION_LOSS" in codes
    assert "FIDELITY_CHUNK_COUNT_IMPLAUSIBLE" in codes
    assert "FIDELITY_SOURCE_COVERAGE_LOW" in codes
    assert metrics.unavailable_metrics == (
        "source_order_preserved",
        "ocr_confidence",
        "ocr_source_coverage_ratio",
    )
    assert metrics.ocr_status == "unavailable"


def test_degradation_warnings_silent_when_healthy() -> None:
    markdown = "Alpha body."
    chunks = [
        _chunk("Alpha body.").model_copy(
            update={"source_start_char": 0, "source_end_char": len(markdown)}
        )
    ]
    metrics = build_fidelity_metrics(
        parsed=ParsedDocument(title="Doc", markdown=markdown, page_count=1),
        boundaries=(BoundaryFidelityInputs(markdown=markdown, chunks=tuple(chunks)),),
        chunk_count=1,
        document_kind="generic",
    )
    assert fidelity_degradation_warnings(metrics) == []
    assert metrics.table_loss_ratio <= 0.0


# ---------------------------------------------------------------------------
# OCR warning semantics (F2) and plausibility upper bound (F3)


def test_ocr_warning_suppressed_for_coverage_without_confidence() -> None:
    metrics = build_fidelity_metrics(
        parsed=ParsedDocument(
            title="Doc",
            markdown="Alpha body.",
            page_count=2,
            ocr_source_coverage_ratio=0.5,
        ),
        boundaries=(
            BoundaryFidelityInputs(
                markdown="Alpha body.",
                chunks=(
                    _chunk("Alpha body.").model_copy(
                        update={"source_start_char": 0, "source_end_char": 11}
                    ),
                ),
            ),
        ),
        chunk_count=1,
        document_kind="generic",
        thresholds=FidelityThresholds(ocr_confidence_min=0.5),
    )
    codes = {warning.code for warning in fidelity_degradation_warnings(metrics)}
    assert "FIDELITY_OCR_CONFIDENCE_LOW" not in codes
    # Page coverage alone does not make OCR measured — the confidence signal is
    # unavailable and stays an explicit unavailable state.
    assert metrics.ocr_status == "unavailable"
    assert metrics.ocr_confidence is None
    assert "ocr_confidence" in metrics.unavailable_metrics


def test_ocr_warning_suppressed_when_threshold_disabled() -> None:
    metrics = build_fidelity_metrics(
        parsed=ParsedDocument(
            title="Doc",
            markdown="Alpha body.",
            page_count=1,
            ocr_confidence=0.2,
            ocr_source_coverage_ratio=1.0,
        ),
        boundaries=(
            BoundaryFidelityInputs(
                markdown="Alpha body.",
                chunks=(
                    _chunk("Alpha body.").model_copy(
                        update={"source_start_char": 0, "source_end_char": 11}
                    ),
                ),
            ),
        ),
        chunk_count=1,
        document_kind="generic",
        thresholds=FidelityThresholds(ocr_confidence_min=0.0),
    )
    codes = {warning.code for warning in fidelity_degradation_warnings(metrics)}
    assert "FIDELITY_OCR_CONFIDENCE_LOW" not in codes


def test_ocr_warning_fires_for_finite_confidence_below_enabled_threshold() -> None:
    metrics = build_fidelity_metrics(
        parsed=ParsedDocument(
            title="Doc",
            markdown="Alpha body.",
            page_count=1,
            ocr_confidence=0.2,
            ocr_source_coverage_ratio=1.0,
        ),
        boundaries=(
            BoundaryFidelityInputs(
                markdown="Alpha body.",
                chunks=(
                    _chunk("Alpha body.").model_copy(
                        update={"source_start_char": 0, "source_end_char": 11}
                    ),
                ),
            ),
        ),
        chunk_count=1,
        document_kind="generic",
        thresholds=FidelityThresholds(ocr_confidence_min=0.5),
    )
    codes = {warning.code for warning in fidelity_degradation_warnings(metrics)}
    assert "FIDELITY_OCR_CONFIDENCE_LOW" in codes


def test_plausibility_upper_bound_counts_every_chunk_emitting_item() -> None:
    """Heading items can each emit a chunk; the old text-only bound false-flagged.

    Six headings + six short sections with eight chunks is structurally normal,
    but the previous upper bound (text sections only) called it implausible.
    """
    markdown = "\n\n".join(
        f"## Heading {index}\n\nBody paragraph {index} with text." for index in range(6)
    )
    state = measure_chunk_plausibility(
        source_markdown=markdown, chunk_count=8, document_kind="generic"
    )
    assert state.verdict is ChunkPlausibilityVerdict.PLAUSIBLE
    assert state.expected_chunk_count_upper is not None
    assert state.expected_chunk_count_upper >= 12


def test_plausibility_upper_bound_accounts_for_split_sections(monkeypatch: Any) -> None:
    """A section exceeding max_tokens contributes ceil(tokens/max_tokens) chunks."""
    import app.features.documents.ingestion.chunking.fidelity as fidelity_module

    monkeypatch.setattr(fidelity_module, "get_tokenizer", lambda: _FixedTokenizer(2_000))
    markdown = "One long section body. " * 50
    # One text item of 2000 tokens splits into ceil(2000/512)=4 chunks.
    state = fidelity_module.measure_chunk_plausibility(
        source_markdown=markdown, chunk_count=4, document_kind="generic"
    )
    assert state.verdict is ChunkPlausibilityVerdict.PLAUSIBLE
    assert state.expected_chunk_count_upper == 4


# ---------------------------------------------------------------------------
# Status surfacing


def test_status_response_decodes_persisted_truth() -> None:
    from app.features.documents.application.commands.upload_document import (
        _fidelity_metrics_from_metadata,
        _relationship_state_from_metadata,
    )

    metadata = {
        "graph_relationships": {
            "status": "skipped_not_implemented",
            "reason": "relationship_extraction_not_implemented",
            "relationships_attempted": 0,
            "relationships_succeeded": 0,
        },
        "fidelity": {
            "fidelity_schema_version": "fidelity_v1",
            "source_coverage_ratio": 0.9,
            "chunk_plausibility": {"verdict": "plausible", "chunk_count": 3},
        },
    }
    state = _relationship_state_from_metadata(metadata)
    assert state is not None
    assert state.status is GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED
    fidelity = _fidelity_metrics_from_metadata(metadata)
    assert fidelity is not None
    assert fidelity.source_coverage_ratio == pytest.approx(0.9)
    assert fidelity.chunk_plausibility.verdict is ChunkPlausibilityVerdict.PLAUSIBLE


def test_status_response_returns_none_for_absent_or_invalid_metadata() -> None:
    from app.features.documents.application.commands.upload_document import (
        _fidelity_metrics_from_metadata,
        _relationship_state_from_metadata,
    )

    assert _relationship_state_from_metadata({}) is None
    assert _fidelity_metrics_from_metadata({"fidelity": "garbage"}) is None
    assert _relationship_state_from_metadata({"graph_relationships": {"status": 1}}) is None
