"""Task 4.6/4.7 regression protection for the ingestion-fidelity gold set.

- The versioned ingestion gold set (``evals/golden/ingestion_fidelity_v1.jsonl``)
  validates through the provenance validator and covers every required
  I4 profile — scanned PDFs, tables, clause numbering, annexures,
  multi-contract files, and Phase-3-style clause-kind documents — with rows
  that label source-to-chunk coverage/order, never retrieval relevance.
- The published machine-readable fidelity report
  (``evals/reports/ingestion_fidelity_i4_offline.json``) passes the
  pilot-blocking coverage/order/warning gate over exactly what the offline
  deterministic stages cover, and honestly records the live-run pending state
  and the explicitly-unavailable metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from returns.result import Success

from app.ai.evaluation.ingestion_fidelity import (
    GateOutcome,
    OfflineArtifactFidelity,
    evaluate_row,
)
from app.ai.evaluation.schema import (
    INGESTION_FIDELITY_SET_VERSION,
    IngestionFidelityExpectation,
    IngestionFidelityRow,
    load_ingestion_fidelity_set,
)

GOLD_SET_PATH = Path("evals/golden/ingestion_fidelity_v1.jsonl")
REPORT_PATH = Path("evals/reports/ingestion_fidelity_i4_offline.json")


def _raw_gold_rows() -> list[dict[str, object]]:
    """The gold set's data rows (metadata line dropped), read outside async fns."""
    return [
        json.loads(line)
        for line in GOLD_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][1:]


@pytest.mark.asyncio
async def test_ingestion_fidelity_gold_set_is_valid_and_covers_i4_profiles() -> None:
    result = await load_ingestion_fidelity_set(GOLD_SET_PATH)

    assert isinstance(result, Success)
    gold_set = result.unwrap()
    assert gold_set.version == INGESTION_FIDELITY_SET_VERSION
    assert len(gold_set.rows) == 25

    profiles = {profile for row in gold_set.rows for profile in row.profiles}
    # Scanned PDFs, tables, clause numbering, multi-contract files.
    assert {"scan_heavy", "table_schedule_heavy", "irregular_numbering", "multi_contract"} <= (
        profiles
    )
    labels = {str(row.label) for row in gold_set.rows}
    assert labels == {"coverage_order", "warning_reason", "boundary_identity", "clause_kind"}
    # Annexure coverage: at least one row requires Schedule/Annex loci.
    annexure_rows = [
        row
        for row in gold_set.rows
        if any(
            prefix.startswith(("Schedule", "Annex"))
            for prefix in row.expectation.expected_locus_prefixes
        )
    ]
    assert annexure_rows, "annexure coverage must be labeled explicitly"
    # Phase-3-style clause-kind documents.
    assert any(row.label == "clause_kind" for row in gold_set.rows)

    raw_rows = _raw_gold_rows()
    forbidden = {
        "query",
        "queries",
        "retrieval_relevance",
        "claim_entailment",
        "expected_chunk_ids_relevant",
    }
    for raw in raw_rows:
        blob = json.dumps(raw)
        assert not forbidden & set(raw), (
            f"row {raw.get('case_id')} carries a retrieval-relevance field"
        )
        assert "retrieval_relevance" not in blob
        assert "claim_entailment" not in blob


def test_published_fidelity_report_passes_gate_and_records_live_pending() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["report_version"] == "ingestion_fidelity_report_v1"
    assert report["golden_set_version"] == INGESTION_FIDELITY_SET_VERSION
    gate = report["gate"]
    assert gate["verdict"] == "pass"
    assert gate["rows_failed"] == 0
    assert gate["failed_case_ids"] == []
    assert gate["rows_passed"] == 25
    # The live queued run is quota-blocked and recorded as pending, not faked.
    assert report["live_run"]["status"] == "pending"
    assert "quota" in report["live_run"]["reason"]
    assert report["limitations"]

    artifacts = report["artifacts"]
    assert len(artifacts) == 17
    for artifact in artifacts:
        # Coverage is the measured, gated dimension; order is None only when
        # no chunk could be located at all (the recorded scanned profiles with
        # an empty text layer), which the gold rows then skip honestly.
        assert artifact["source_coverage_ratio"] is not None
        if artifact["chunk_count"] == 0:
            assert artifact["source_order_preserved"] is None
        else:
            assert artifact["source_order_preserved"] is not None
        assert artifact["boundary_count"] >= 1
        # Offline honesty: OCR and chunk-count plausibility are explicitly
        # unavailable, never fabricated.
        assert artifact["ocr_status"] == "unavailable"
        assert artifact["chunk_plausibility_verdict"] == "unavailable"
        assert set(artifact["unavailable_metrics"]) >= {
            "ocr_confidence",
            "chunk_plausibility_tokenizer_bounds",
        }
    # Every gold row was evaluated against its artifact's measurement.
    evaluated = {row["case_id"] for row in report["rows"]}
    assert evaluated == {row["case_id"] for row in _raw_gold_rows()}


def _row(expectation_kwargs: dict[str, object]) -> IngestionFidelityRow:
    return IngestionFidelityRow(
        case_id="coverage_order-unit-row",
        label="coverage_order",
        document_id="UNIT",
        profiles=["clean_control"],
        expectation=IngestionFidelityExpectation(**expectation_kwargs),
        provenance={
            "source_type": "synthetic",
            "source_reference": "unit test",
            "permission_status": "synthetic_owned",
            "license_terms_status": "not_required",
            "is_public": False,
            "is_synthetic": True,
            "human_review_status": "engineer_reviewed",
            "annotators": {
                "engineer_relevance": "unit-test-labeler",
                "legal_sme_entailment": "not-required-ingestion-fidelity",
            },
        },
        notes="unit fixture",
    )


def _measurement(**overrides: object) -> OfflineArtifactFidelity:
    defaults: dict[str, object] = {
        "source_id": "UNIT",
        "pages": 1,
        "source_chars": 100,
        "chunk_count": 2,
        "first_chunk_id": "UNIT:p1:c01",
        "last_chunk_id": "UNIT:p1:c02",
        "source_coverage_ratio": 0.97,
        "source_order_preserved": True,
        "sections_total": 4,
        "sections_preserved": 4,
        "warning_codes": [],
    }
    return OfflineArtifactFidelity.model_validate({**defaults, **overrides})


def test_gate_row_evaluation_blocks_on_failed_coverage_expectation() -> None:
    passed = evaluate_row(
        _row({"min_source_coverage": 0.95, "order_preserved": True}), _measurement()
    )
    assert passed.passed

    blocked = evaluate_row(
        _row({"min_source_coverage": 0.99, "order_preserved": True}), _measurement()
    )
    assert not blocked.passed
    assert blocked.failures
    assert "source coverage" in blocked.failures[0]

    order_violation = evaluate_row(
        _row({"min_source_coverage": 0.95, "order_preserved": True}),
        _measurement(source_order_preserved=False),
    )
    assert not order_violation.passed
    assert any("order" in failure for failure in order_violation.failures)


def test_gate_row_evaluation_blocks_on_unexpected_and_missing_warnings() -> None:
    missing = evaluate_row(
        _row({"expected_warning_codes": ["FIDELITY_OCR_CONFIDENCE_LOW"]}), _measurement()
    )
    assert not missing.passed
    assert any("did not fire" in failure for failure in missing.failures)

    fired = evaluate_row(
        _row({"forbidden_warning_codes": ["FIDELITY_TABLE_LOSS"]}),
        _measurement(warning_codes=["FIDELITY_TABLE_LOSS"]),
    )
    assert not fired.passed
    assert any("fired" in failure for failure in fired.failures)

    # A row whose offline-unevaluable dimensions are the only expectations is
    # skipped — recorded, never silently passed.
    skipped = evaluate_row(_row({"max_ocr_confidence": 0.5}), _measurement())
    assert not skipped.passed
    assert skipped.skipped


def test_gate_outcome_reflects_failed_rows() -> None:
    gate = GateOutcome(
        verdict="fail",
        basis="labeled source-to-chunk coverage/order and typed warning outcomes",
        rows_passed=24,
        rows_failed=1,
        failed_case_ids=["coverage_order-SRC-01"],
    )
    assert gate.verdict == "fail"
    assert gate.failed_case_ids == ["coverage_order-SRC-01"]


# ---------------------------------------------------------------------------
# Regressions: source normalization, empty expectations, sibling manifest, CLI.
# ---------------------------------------------------------------------------


def test_normalized_source_locates_chunks_the_raw_text_hides() -> None:
    """Chunk contents are derived from the normalized page (R1-R7), so the
    span search must run over the normalized source too: a recorded
    ``payment\\u00a0terms`` is the intact ``payment terms`` chunk, not missing
    content."""
    from app.ai.evaluation.ingestion_fidelity import convention_chunks, reconstruct_markdown

    page = {
        "page_no": 1,
        "text": "Intro paragraph.\n\npayment\u00a0terms are net thirty days from invoice.",
    }
    _markdown, offsets = reconstruct_markdown([page])
    prepared = convention_chunks(source_id="UNIT", pages=[page], page_offsets=offsets)

    assert any("payment terms" in chunk.content for _id, chunk in prepared)
    located = [chunk for _id, chunk in prepared if chunk.source_start_char is not None]
    assert located, "intact normalized content must keep its source span"


def test_an_empty_expectation_cannot_pass_vacuously() -> None:
    """A row that labels no evaluable dimension is a labeling mistake: the
    previous ``bool(model_dump(exclude_none=True))`` guard was vacuous because
    the default empty containers survive ``exclude_none``."""
    vacuous = evaluate_row(_row({}), _measurement())
    assert not vacuous.passed
    assert vacuous.failures == []
    assert vacuous.skipped == []


def test_boundary_findings_read_the_supplied_sibling_manifest(tmp_path: Path) -> None:
    """The boundary cross-check reads the gold set's own sibling manifest — a
    custom gold set or ``--corpus-root`` run is never silently checked against
    the default corpus manifest."""
    from app.ai.evaluation.ingestion_fidelity import _corpus_findings

    manifest = tmp_path / "custom.manifest.json"
    manifest.write_text(
        json.dumps({"per_artifact": [{"source_id": "UNIT", "contract_boundaries": [1, 2, 3]}]}),
        encoding="utf-8",
    )

    measured = _measurement(boundary_count=1)
    mismatched = _corpus_findings([measured], manifest_path=manifest)
    assert any("contract_boundaries=3" in finding.finding for finding in mismatched)

    missing = _corpus_findings([measured], manifest_path=tmp_path / "absent.manifest.json")
    assert missing == []


def test_cli_exits_nonzero_on_a_failed_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed corpus gate is a failed evaluation run: the report is still
    written, but the exit code must not read it as success."""
    import sys as _sys

    from app.ai.evaluation import ingestion_fidelity as fidelity_module
    from app.ai.evaluation.ingestion_fidelity import GateOutcome, IngestionFidelityReport

    report = IngestionFidelityReport(
        generated_at_utc="2026-01-01T00:00:00+00:00",
        golden_set_version="ingestion_fidelity_v1",
        pipeline_note="unit fixture",
        live_run={"status": "pending", "reason": "unit fixture"},
        gate=GateOutcome(
            verdict="fail",
            basis="unit fixture",
            rows_passed=0,
            rows_failed=1,
            failed_case_ids=["coverage_order-UNIT"],
        ),
    )
    monkeypatch.setattr(fidelity_module, "evaluate_gold_set", lambda **_kwargs: report)
    monkeypatch.setattr(
        _sys, "argv", ["ingestion_fidelity", "gold.jsonl", "--output", str(tmp_path / "r.json")]
    )

    with pytest.raises(SystemExit) as caught:
        fidelity_module.main()

    assert caught.value.code == 1
    assert (
        json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["gate"]["verdict"] == "fail"
    )


def _fixture_gold_set(tmp_path: Path, *, row_overrides: dict[str, object] | None = None) -> Path:
    """A minimal loadable ingestion-fidelity set + sibling manifest in tmp_path.

    Clones the committed set's first row (guaranteed-valid provenance) and
    pairs it with a manifest whose per_artifact entry declares the row's
    document; the artifact file itself is absent, which the manifest audit
    treats as a checksum-skip warning, not an issue.
    """
    base_row = dict(_raw_gold_rows()[0])
    row: dict[str, object] = {
        "case_id": "coverage_order-FIX-01",
        "label": "coverage_order",
        "document_id": "FIX-01",
        "profiles": [],
        "expectation": {"min_source_coverage": 0.9},
        "provenance": base_row["provenance"],
    }
    if row_overrides:
        row.update(row_overrides)
    metadata = {
        "version": INGESTION_FIDELITY_SET_VERSION,
        "provenance_schema_version": "golden_provenance_v1",
    }
    gold_path = tmp_path / "ingestion_fidelity_fixture.jsonl"
    gold_path.write_text(
        json.dumps(metadata) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )
    manifest = {
        "manifest_version": "1",
        "golden_set_version": INGESTION_FIDELITY_SET_VERSION,
        "artifact": "evals/golden/ingestion_fidelity_fixture.jsonl",
        "per_artifact": [
            {
                "source_id": "FIX-01",
                "sha256": "a" * 64,
                "source_url": "https://example.org/fix-01",
                "classification": "public",
                "permission_status": "public_terms_verified",
                "redistribution_status": "allowed",
            }
        ],
    }
    (tmp_path / "ingestion_fidelity_fixture.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return gold_path


@pytest.mark.asyncio
async def test_ingestion_loader_rejects_missing_provenance_schema_version(tmp_path: Path) -> None:
    """A set with no provenance_schema_version fails closed (wave-3 finding)."""
    gold_path = _fixture_gold_set(tmp_path)
    lines = gold_path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0])
    del metadata["provenance_schema_version"]
    gold_path.write_text(
        json.dumps(metadata) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8"
    )

    result = await load_ingestion_fidelity_set(gold_path)

    assert not isinstance(result, Success)
    assert "provenance_schema_version" in result.failure().message


@pytest.mark.asyncio
async def test_ingestion_loader_rejects_unknown_provenance_schema_version(tmp_path: Path) -> None:
    gold_path = _fixture_gold_set(tmp_path)
    lines = gold_path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0])
    metadata["provenance_schema_version"] = "golden_provenance_v99"
    gold_path.write_text(
        json.dumps(metadata) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8"
    )

    result = await load_ingestion_fidelity_set(gold_path)

    assert not isinstance(result, Success)
    assert "provenance_schema_version" in result.failure().message


@pytest.mark.asyncio
async def test_ingestion_loader_rejects_release_tier_without_completed_review(
    tmp_path: Path,
) -> None:
    """A release-tier row with pending review and no adjudication must fail.

    The loader used to skip the shared release-tier accountability checks the
    retrieval validator enforces, so unreviewed labels validated here.
    """
    gold_path = _fixture_gold_set(tmp_path)
    lines = gold_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["provenance"]["quality_tier"] = "release"
    row["provenance"]["human_review_status"] = "pending"
    row["provenance"]["adjudication"] = None
    gold_path.write_text(lines[0] + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    result = await load_ingestion_fidelity_set(gold_path)

    assert not isinstance(result, Success)
    failure = result.failure()
    assert "release-quality case requires" in failure.message
    # The loader reports every row issue, not only the first one.
    reported = " ".join(str(i) for i in (failure.details or {}).get("issues", []))
    assert "requires completed human review" in reported


def test_fresh_evaluate_gold_set_passes_on_the_current_code() -> None:
    """Run the evaluator NOW, not only against the committed report.

    Greptile wave-3 finding: the published-report test checks a committed JSON
    file, so chunking/threshold/gold-expectation changes could leave the suite
    green while the actual corpus regressed. The fresh run is cheap (pure CPU,
    ~3s) and asserts the live verdict; the saved-report checks above stay
    separate (they pin what was published).
    """
    from app.ai.evaluation.ingestion_fidelity import evaluate_gold_set

    # Sync entry point: returns the report, or raises when the set fails to load.
    report = evaluate_gold_set(
        gold_set_path=GOLD_SET_PATH,
        corpus_root=Path("evals/corpus"),
    )
    gate = report.gate
    assert gate.verdict == "pass", (
        f"fresh corpus evaluation failed: {gate.rows_failed} rows "
        f"({gate.failed_case_ids})"
    )
    assert gate.rows_failed == 0
    assert gate.rows_passed == 25
