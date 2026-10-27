import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from returns.result import Failure, Success

from app.ai.evaluation.schema import (
    AdjudicationRecord,
    GoldenProvenanceManifest,
    GoldenQuery,
    GoldenSet,
    audit_golden_manifest,
    load_golden_set,
    load_validated_golden_set,
)


def _provenance(**overrides: object) -> dict[str, object]:
    provenance: dict[str, object] = {
        "source_type": "synthetic",
        "source_reference": "eval-fixture://synthetic-1",
        "permission_status": "synthetic_owned",
        "license_terms_status": "not_required",
        "license_terms_reference": None,
        "is_public": False,
        "is_synthetic": True,
        "human_review_status": "engineer_reviewed",
        "annotators": {
            "engineer_relevance": "engineer-1",
            "legal_sme_entailment": "pending-sme",
        },
        "quality_tier": "candidate",
    }
    provenance.update(overrides)
    return provenance


def _v2_row(**overrides: object) -> dict[str, object]:
    """A fully compliant v2 direct-answer row used as the fixture base."""
    row: dict[str, object] = {
        "query": "Which clause applies?",
        "expected_chunk_ids": ["chunk-1"],
        "expected_document_ids": ["doc-1"],
        "case_id": "direct_answer_fixture_001",
        "measurement_family": "retrieval",
        "grader": "deterministic",
        "ocr_profile": "clean",
        "expected_spans": [
            {
                "document_id": "doc-1",
                "contract_index": None,
                "page_no": 1,
                "quote": "Governing clause.",
            }
        ],
        "expected_claims": [
            {
                "claim_id": "claim-1",
                "text": "Clause one governs.",
                "expected_support": "entailed",
                "evidence_chunk_ids": ["chunk-1"],
                "gradability": "deterministic",
            }
        ],
        "expected_ingestion_states": [{"document_id": "doc-1", "state": "ready"}],
        "jurisdiction": "India",
        "document_kind": "contracts",
        "difficulty": "easy",
        "notes": "fixture",
        "awaiting_sme_expansion": False,
        "case_type": "direct_answer",
        "retrieval_relevance": "relevant",
        "claim_entailment": "entailed",
        "provenance": _provenance(),
    }
    row.update(overrides)
    return row


def _write_set(path: Path, *rows: dict[str, object]) -> None:
    metadata = {
        "version": "legal_retrieval_v2",
        "provenance_schema_version": "golden_provenance_v1",
    }
    lines = [json.dumps(metadata), *(json.dumps(row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(
    path: Path,
    per_artifact: list[dict[str, object]],
    corpus_root: Path,
) -> None:
    manifest = {
        "manifest_version": "golden_provenance_v1",
        "golden_set_version": "legal_retrieval_v2",
        "artifact": path.name.removesuffix(".manifest.json") + ".jsonl",
        "corpus_manifest_path": str(corpus_root),
        "source_policy": {},
        "annotation_policy": {},
        "per_artifact": per_artifact,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_artifact(corpus_root: Path, source_id: str, content: bytes = b"corpus-bytes") -> str:
    artifact_dir = corpus_root / "artifacts" / source_id
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "original.txt").write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _artifact_entry(source_id: str, sha256: str, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "source_id": source_id,
        "sha256": sha256,
        "source_url": f"https://example.test/{source_id}",
        "original_filename": "original.txt",
    }
    entry.update(overrides)
    return entry


@pytest.mark.asyncio
async def test_corrupted_manifest_checksum_fails_validation(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    _write_artifact(corpus_root, "doc-1")
    _write_manifest(
        tmp_path / "set.manifest.json",
        [_artifact_entry("doc-1", "0" * 64)],
        corpus_root,
    )
    path = tmp_path / "set.jsonl"
    _write_set(path, _v2_row())

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "sha256 mismatch" in result.failure().message


@pytest.mark.asyncio
async def test_manifest_entry_without_checksum_and_source_url_fails(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "set.manifest.json", [{"source_id": "doc-1"}], tmp_path)
    path = tmp_path / "set.jsonl"
    _write_set(path, _v2_row())
    loaded = await load_golden_set(path)
    assert isinstance(loaded, Success)
    manifest = GoldenProvenanceManifest.model_validate(
        json.loads((tmp_path / "set.manifest.json").read_text(encoding="utf-8"))
    )

    issues = await audit_golden_manifest(manifest, loaded.unwrap())

    assert any("sha256 checksum is missing" in issue for issue in issues)
    assert any("source_url is missing" in issue for issue in issues)


@pytest.mark.asyncio
async def test_release_row_with_pending_sme_adjudication_fails(tmp_path: Path) -> None:
    row = _v2_row(
        provenance=_provenance(quality_tier="release", adjudication={"status": "pending_sme"})
    )
    path = tmp_path / "release.jsonl"
    _write_set(path, row)

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "release-quality case requires adjudication status" in result.failure().message


@pytest.mark.asyncio
async def test_multi_contract_case_without_contract_indexes_fails(tmp_path: Path) -> None:
    row = _v2_row(
        case_type="multi_contract_boundary",
        case_id="multi_contract_boundary_fixture_002",
        measurement_family="contract_identity",
    )
    path = tmp_path / "multi.jsonl"
    _write_set(path, row)

    result = await load_golden_set(path)

    assert isinstance(result, Failure)
    assert result.failure().row_index == 2
    assert "expected_contract_indexes" in result.failure().message


@pytest.mark.asyncio
async def test_v2_row_referencing_undeclared_document_fails(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    sha256 = _write_artifact(corpus_root, "doc-1")
    _write_manifest(
        tmp_path / "set.manifest.json",
        [_artifact_entry("doc-1", sha256)],
        corpus_root,
    )
    row = _v2_row(
        expected_document_ids=["doc-1", "doc-unknown"],
        expected_ingestion_states=[
            {"document_id": "doc-1", "state": "ready"},
            {"document_id": "doc-unknown", "state": "ready"},
        ],
    )
    path = tmp_path / "set.jsonl"
    _write_set(path, row)

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "not present in manifest per_artifact" in result.failure().message


@pytest.mark.asyncio
async def test_disallowed_redistribution_with_stored_original_fails(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    sha256 = _write_artifact(corpus_root, "doc-1")
    _write_manifest(
        tmp_path / "set.manifest.json",
        [_artifact_entry("doc-1", sha256, redistribution_status="not_permitted")],
        corpus_root,
    )
    path = tmp_path / "set.jsonl"
    _write_set(path, _v2_row())

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "disallows" in result.failure().message


@pytest.mark.asyncio
async def test_paired_rows_must_form_a_control_treatment_pair(tmp_path: Path) -> None:
    rows = [
        _v2_row(
            case_id="direct_answer_fixture_010",
            pair_id="pair-1",
            pair_variable="ocr_profile",
            pair_role="control",
        ),
        _v2_row(
            case_id="direct_answer_fixture_011",
            pair_id="pair-1",
            pair_variable="ocr_profile",
            pair_role="control",
        ),
    ]
    path = tmp_path / "paired.jsonl"
    _write_set(path, *rows)

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "paired rows" in result.failure().message


def test_case_type_aliases_are_normalized() -> None:
    legacy_standard = GoldenQuery(**_v2_row(case_type="standard"))  # type: ignore[arg-type]
    legacy_near_miss = GoldenQuery(
        **_v2_row(case_type="near_miss", case_id="near_miss_clause_fixture_003")  # type: ignore[arg-type]
    )

    assert legacy_standard.case_type == "direct_answer"
    assert legacy_near_miss.case_type == "near_miss_clause"


def test_adjudicator_resembling_a_model_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="human reviewer"):
        AdjudicationRecord(status="adjudicated", adjudicator="gpt-5-adjudicator")

    record = AdjudicationRecord(
        status="adjudicated",
        adjudicator="senior-sme-1",
        method="independent_sme",
    )
    assert record.adjudicator == "senior-sme-1"


def _manifest_for(golden_set_version: str, per_artifact: list[dict[str, object]]) -> GoldenProvenanceManifest:
    return GoldenProvenanceManifest.model_validate(
        {
            "manifest_version": "golden_provenance_v1",
            "golden_set_version": golden_set_version,
            "artifact": "set.jsonl",
            "per_artifact": per_artifact,
        }
    )


def _v1_set() -> GoldenSet:
    """A minimal v1-shaped set: no provenance, no v2 measurement fields."""
    row = GoldenQuery(
        query="legacy query",
        expected_chunk_ids=["chunk-1"],
        document_kind="contracts",
        difficulty="easy",
        notes="legacy fixture",
        awaiting_sme_expansion=False,
    )
    return GoldenSet(version="legal_retrieval_v1", queries=[row])


def _v2_set() -> GoldenSet:
    return GoldenSet(version="legal_retrieval_v2", queries=[GoldenQuery(**_v2_row())])


@pytest.mark.asyncio
async def test_v2_manifest_with_empty_per_artifact_is_an_issue() -> None:
    manifest = _manifest_for("legal_retrieval_v2", [])

    issues = await audit_golden_manifest(manifest, _v2_set())

    assert any("v2 manifest has empty per_artifact; provenance audit disabled" in i for i in issues)


@pytest.mark.asyncio
async def test_v2_manifest_with_empty_per_artifact_fails_load(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "set.manifest.json", [], tmp_path)
    path = tmp_path / "set.jsonl"
    _write_set(path, _v2_row())

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "v2 manifest has empty per_artifact" in result.failure().message


@pytest.mark.asyncio
async def test_v1_manifest_without_per_artifact_has_no_new_issues() -> None:
    manifest = _manifest_for("legal_retrieval_v1", [])

    issues = await audit_golden_manifest(manifest, _v1_set())

    assert issues == []


@pytest.mark.asyncio
async def test_version_mismatch_is_reported_even_with_empty_per_artifact() -> None:
    manifest = _manifest_for("legal_retrieval_v1", [])

    issues = await audit_golden_manifest(manifest, _v2_set())

    assert any(
        "manifest golden_set_version 'legal_retrieval_v1' does not match golden-set version "
        "'legal_retrieval_v2'" in issue
        for issue in issues
    )


@pytest.mark.asyncio
async def test_v2_exemplar_set_passes_validation() -> None:
    result = await load_validated_golden_set(Path("evals/golden/legal_retrieval_v2.jsonl"))

    assert isinstance(result, Success)
    golden_set = result.unwrap()
    assert golden_set.version == "legal_retrieval_v2"
    assert golden_set.provenance_schema_version == "golden_provenance_v1"
