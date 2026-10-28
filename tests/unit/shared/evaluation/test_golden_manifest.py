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
    SmeRequirements,
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
    sme_requirements: dict[str, object] | None = None,
) -> None:
    manifest: dict[str, object] = {
        "manifest_version": "golden_provenance_v1",
        "golden_set_version": "legal_retrieval_v2",
        "artifact": path.name.removesuffix(".manifest.json") + ".jsonl",
        "corpus_manifest_path": str(corpus_root),
        "source_policy": {},
        "annotation_policy": {},
        "per_artifact": per_artifact,
    }
    if sme_requirements is not None:
        manifest["sme_requirements"] = sme_requirements
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

    issues = await audit_golden_manifest(
        manifest,
        loaded.unwrap(),
        manifest_path=tmp_path / "set.manifest.json",
    )

    assert any("sha256 checksum is missing" in issue for issue in issues)
    assert any("source_url is missing" in issue for issue in issues)


@pytest.mark.asyncio
async def test_release_row_with_pending_sme_adjudication_fails(tmp_path: Path) -> None:
    # v2 loads fail closed without a manifest, so ship one to exercise the
    # provenance rule this test targets.
    corpus_root = tmp_path / "corpus"
    sha256 = _write_artifact(corpus_root, "doc-1")
    _write_manifest(
        tmp_path / "release.manifest.json",
        [_artifact_entry("doc-1", sha256)],
        corpus_root,
    )
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
async def test_ingestion_state_for_undeclared_document_fails(tmp_path: Path) -> None:
    """A doc named only in expected_ingestion_states must also be declared."""
    corpus_root = tmp_path / "corpus"
    sha256 = _write_artifact(corpus_root, "doc-1")
    _write_manifest(
        tmp_path / "set.manifest.json",
        [_artifact_entry("doc-1", sha256)],
        corpus_root,
    )
    row = _v2_row(
        expected_ingestion_states=[
            {"document_id": "doc-1", "state": "ready"},
            {"document_id": "doc-undeclared", "state": "failed"},
        ],
    )
    path = tmp_path / "set.jsonl"
    _write_set(path, row)

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "not present in manifest per_artifact" in result.failure().message


@pytest.mark.asyncio
async def test_ingestion_state_document_declared_in_per_artifact_passes(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    sha256 = _write_artifact(corpus_root, "doc-1")
    undeclared_sha256 = _write_artifact(corpus_root, "doc-undeclared")
    _write_manifest(
        tmp_path / "set.manifest.json",
        [
            _artifact_entry("doc-1", sha256),
            _artifact_entry("doc-undeclared", undeclared_sha256),
        ],
        corpus_root,
    )
    row = _v2_row(
        expected_ingestion_states=[
            {"document_id": "doc-1", "state": "ready"},
            {"document_id": "doc-undeclared", "state": "failed"},
        ],
    )
    path = tmp_path / "set.jsonl"
    _write_set(path, row)

    result = await load_validated_golden_set(path)

    assert isinstance(result, Success)


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
    # v2 loads fail closed without a manifest, so ship one to exercise the
    # pairing rule this test targets.
    corpus_root = tmp_path / "corpus"
    sha256 = _write_artifact(corpus_root, "doc-1")
    _write_manifest(
        tmp_path / "paired.manifest.json",
        [_artifact_entry("doc-1", sha256)],
        corpus_root,
    )
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


def test_waiver_requires_rationale_and_human_adjudicator() -> None:
    rationale = "SME lead waived this row: duplicate of an adjudicated case in the same wave"
    with pytest.raises(ValidationError, match="rationale"):
        AdjudicationRecord(status="waived_with_reason")
    with pytest.raises(ValidationError, match="rationale"):
        AdjudicationRecord(
            status="waived_with_reason", rationale="too short", adjudicator="sme-lead-7"
        )
    with pytest.raises(ValidationError, match="adjudicator"):
        AdjudicationRecord(status="waived_with_reason", rationale=rationale)
    with pytest.raises(ValidationError, match="human reviewer"):
        AdjudicationRecord(status="waived_with_reason", rationale=rationale, adjudicator="gpt-5")

    record = AdjudicationRecord(
        status="waived_with_reason", rationale=rationale, adjudicator="sme-lead-7"
    )
    assert record.status == "waived_with_reason"
    assert record.adjudicator == "sme-lead-7"


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


@pytest.mark.asyncio
async def test_v2_set_without_sibling_manifest_fails_load(tmp_path: Path) -> None:
    """v2 gold data is unauditable without its manifest, so every load refuses it."""
    path = tmp_path / "set.jsonl"
    _write_set(path, _v2_row())  # provenance-clean rows: the manifest is the only problem

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "requires a sibling set.manifest.json provenance manifest" in result.failure().message


@pytest.mark.asyncio
async def test_v2_manifest_audit_issue_fails_load(tmp_path: Path) -> None:
    """An existing manifest with an audit defect still fails the normal load path."""
    corpus_root = tmp_path / "corpus"
    _write_artifact(corpus_root, "doc-1")
    _write_manifest(
        tmp_path / "set.manifest.json",
        [
            {
                "source_id": "doc-1",
                "source_url": "https://example.test/doc-1",
                "original_filename": "original.txt",
            }
        ],
        corpus_root,
    )
    path = tmp_path / "set.jsonl"
    _write_set(path, _v2_row())

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "sha256 checksum is missing" in result.failure().message


@pytest.mark.asyncio
async def test_v1_set_without_manifest_still_loads(tmp_path: Path) -> None:
    """v1 keeps loading with no sibling manifest and no audit."""
    metadata = {
        "version": "legal_retrieval_v1",
        "provenance_schema_version": "golden_provenance_v1",
    }
    v1_row: dict[str, object] = {
        "query": "Which clause applies?",
        "expected_chunk_ids": ["chunk-1"],
        "expected_document_ids": ["doc-1"],
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
    path = tmp_path / "v1.jsonl"
    path.write_text(json.dumps(metadata) + "\n" + json.dumps(v1_row) + "\n", encoding="utf-8")

    result = await load_validated_golden_set(path)

    assert isinstance(result, Success)
    assert result.unwrap().version == "legal_retrieval_v1"


# --- C12: release-tier named adjudicator must be a present human name --------


def test_sme_requirements_rejects_model_like_named_adjudicator() -> None:
    with pytest.raises(ValidationError, match="human reviewer"):
        SmeRequirements(
            role="practising lawyer",
            domain="contract law",
            independence="independent of whoever produced the labels",
            commitment="sign-off on disputed cases",
            named_adjudicator="claude-sme-lead",
        )

    record = SmeRequirements(
        role="practising lawyer",
        domain="contract law",
        independence="independent of whoever produced the labels",
        commitment="sign-off on disputed cases",
        named_adjudicator="sme-lead-7",
    )
    assert record.named_adjudicator == "sme-lead-7"


def _adjudicated_release_row() -> dict[str, object]:
    return _v2_row(
        provenance=_provenance(
            quality_tier="release",
            human_review_status="adjudicated",
            adjudication={
                "status": "adjudicated",
                "adjudicator": "sme-lead-7",
                "method": "independent_sme",
                "rationale": "SME confirmed the disputed relevance",
                "checklist": {},
                "decided_at": "2026-10-28T00:00:00+00:00",
            },
        )
    )


@pytest.mark.asyncio
async def test_release_adjudicated_row_with_blank_manifest_adjudicator_fails(
    tmp_path: Path,
) -> None:
    """A whitespace-only manifest named_adjudicator counts as missing."""
    corpus_root = tmp_path / "corpus"
    sha256 = _write_artifact(corpus_root, "doc-1")
    _write_manifest(
        tmp_path / "set.manifest.json",
        [_artifact_entry("doc-1", sha256)],
        corpus_root,
        sme_requirements={
            "role": "practising lawyer",
            "domain": "contract law",
            "independence": "independent of whoever produced the labels",
            "commitment": "sign-off on disputed cases",
            "named_adjudicator": "   ",
            "adjudication_status": "complete",
        },
    )
    path = tmp_path / "set.jsonl"
    _write_set(path, _adjudicated_release_row())

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "release-tier adjudication requires a named adjudicator" in result.failure().message


@pytest.mark.asyncio
async def test_release_adjudicated_row_with_model_manifest_adjudicator_fails(
    tmp_path: Path,
) -> None:
    """A model-like manifest named_adjudicator is rejected at parse time."""
    corpus_root = tmp_path / "corpus"
    sha256 = _write_artifact(corpus_root, "doc-1")
    _write_manifest(
        tmp_path / "set.manifest.json",
        [_artifact_entry("doc-1", sha256)],
        corpus_root,
        sme_requirements={
            "role": "practising lawyer",
            "domain": "contract law",
            "independence": "independent of whoever produced the labels",
            "commitment": "sign-off on disputed cases",
            "named_adjudicator": "gpt-5-sme-lead",
            "adjudication_status": "complete",
        },
    )
    path = tmp_path / "set.jsonl"
    _write_set(path, _adjudicated_release_row())

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    assert "Unable to load golden-set manifest" in result.failure().message


# --- C13: relative corpus_manifest_path anchors to the manifest directory ----


@pytest.mark.asyncio
async def test_audit_resolves_relative_corpus_path_from_manifest_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A relative corpus_manifest_path must not depend on the process cwd."""
    corpus_root = tmp_path / "corpus"
    _write_artifact(corpus_root, "doc-1")
    manifest = {
        "manifest_version": "golden_provenance_v1",
        "golden_set_version": "legal_retrieval_v2",
        "artifact": "set.jsonl",
        "corpus_manifest_path": "corpus",  # relative to the manifest's directory
        "source_policy": {},
        "annotation_policy": {},
        "per_artifact": [_artifact_entry("doc-1", "0" * 64)],  # corrupt checksum
    }
    (tmp_path / "set.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    path = tmp_path / "set.jsonl"
    _write_set(path, _v2_row())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # cwd holds no evals/corpus

    result = await load_validated_golden_set(path)

    # a sha256 mismatch proves the artifact was located relative to the
    # manifest directory and actually hashed (an unresolved path would only
    # produce the checksum-skipped warning and pass)
    assert isinstance(result, Failure)
    assert "sha256 mismatch" in result.failure().message
    assert "checksum skipped" not in capsys.readouterr().err


# --- C14: manifest load failures retain the underlying cause -----------------


@pytest.mark.asyncio
async def test_corrupted_manifest_failure_carries_cause_details(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    _write_set(path, _v2_row())
    (tmp_path / "set.manifest.json").write_text("{not json", encoding="utf-8")

    result = await load_validated_golden_set(path)

    assert isinstance(result, Failure)
    failure = result.failure()
    # the API-facing message is unchanged; the underlying cause is retained
    # in the structured details (MalformedGoldenRowError is a pydantic model,
    # so e.add_note() is not available)
    assert failure.message.startswith("Unable to load golden-set manifest")
    details = failure.details or {}
    assert "JSONDecodeError" in str(details.get("manifest_cause"))
    assert "set.manifest.json" in str(details.get("manifest"))
