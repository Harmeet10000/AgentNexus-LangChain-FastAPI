import json
from pathlib import Path

import pytest
from returns.result import Failure, Success

from app.ai.evaluation.schema import (
    DoubleLabel,
    GoldenItemProvenance,
    GoldenQuery,
    GoldenSet,
    GoldenSetProvenanceError,
    load_golden_set,
    load_validated_golden_set,
    validate_golden_set,
)


def _row(*, provenance: dict[str, object], **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "query": "Which clause applies?",
        "expected_chunk_ids": ["chunk-1"],
        "expected_document_ids": ["document-1"],
        "jurisdiction": "India",
        "document_kind": "contracts",
        "difficulty": "easy",
        "notes": "fixture",
        "awaiting_sme_expansion": False,
        "case_type": "standard",
        "retrieval_relevance": "relevant",
        "claim_entailment": "entailed",
        "provenance": provenance,
    }
    row.update(overrides)
    return row


def _public_provenance() -> dict[str, object]:
    return {
        "source_type": "public",
        "source_reference": "https://example.test/public-contract",
        "permission_status": "public_terms_verified",
        "license_terms_status": "verified",
        "license_terms_reference": "https://example.test/terms",
        "is_public": True,
        "is_synthetic": False,
        "human_review_status": "sme_reviewed",
        "annotators": {
            "engineer_relevance": "engineer-1",
            "legal_sme_entailment": "sme-1",
        },
    }


def _synthetic_provenance() -> dict[str, object]:
    return {
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
    }


def _write_set(path: Path, row: dict[str, object]) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "version": "test_v1",
                        "provenance_schema_version": "golden_provenance_v1",
                    }
                ),
                json.dumps(row),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_public_and_synthetic_items_are_valid(tmp_path: Path) -> None:
    for name, provenance in (
        ("public", _public_provenance()),
        ("synthetic", _synthetic_provenance()),
    ):
        path = tmp_path / f"{name}.jsonl"
        _write_set(path, _row(provenance=provenance))

        result = await load_validated_golden_set(path)

        assert isinstance(result, Success)


@pytest.mark.asyncio
async def test_unknown_permission_status_fails_schema_validation(tmp_path: Path) -> None:
    provenance = _synthetic_provenance() | {"permission_status": "unknown"}
    path = tmp_path / "unknown.jsonl"
    _write_set(path, _row(provenance=provenance))

    result = await load_golden_set(path)

    assert isinstance(result, Failure)
    assert result.failure().row_index == 2


@pytest.mark.asyncio
async def test_client_material_without_written_permission_fails(tmp_path: Path) -> None:
    provenance = _public_provenance() | {
        "source_type": "client",
        "is_public": False,
        "is_synthetic": False,
        "permission_status": "public_terms_verified",
    }
    path = tmp_path / "client.jsonl"
    _write_set(path, _row(provenance=provenance))

    result = await load_golden_set(path)

    assert isinstance(result, Failure)
    assert result.failure().row_index == 2


@pytest.mark.asyncio
async def test_public_item_with_missing_terms_review_fails(tmp_path: Path) -> None:
    provenance = _public_provenance() | {
        "license_terms_status": "pending",
        "license_terms_reference": None,
    }
    path = tmp_path / "pending-terms.jsonl"
    _write_set(path, _row(provenance=provenance))

    result = await load_golden_set(path)

    assert isinstance(result, Failure)
    assert result.failure().row_index == 2


def test_double_label_agreement_and_adjudication_are_preserved() -> None:
    double_label = DoubleLabel(
        retrieval_relevance="relevant",
        claim_entailment="partial",
        agreement="disagree",
        adjudication_status="complete",
        adjudicator="senior-sme",
    )
    provenance = GoldenItemProvenance(
        **_synthetic_provenance(),
        double_label=double_label,
    )

    assert provenance.double_label is not None
    assert provenance.double_label.agreement == "disagree"
    assert provenance.double_label.adjudicator == "senior-sme"


def test_no_evidence_cases_have_separate_labels() -> None:
    query = GoldenQuery(
        query="Which missing statute applies?",
        expected_chunk_ids=[],
        expected_document_ids=[],
        jurisdiction="India",
        document_kind="statutes",
        difficulty="hard",
        notes="No statute is present in the selected corpus.",
        awaiting_sme_expansion=False,
        case_type="statute_not_in_corpus",
        retrieval_relevance="not_applicable",
        claim_entailment="not_applicable",
    )

    assert query.case_type == "statute_not_in_corpus"


def test_legacy_rows_are_not_silently_ci_valid() -> None:
    query = GoldenQuery(
        query="Legacy row",
        expected_chunk_ids=["chunk"],
        expected_document_ids=["document"],
        jurisdiction=None,
        document_kind="contracts",
        difficulty="easy",
        notes="Legacy fixture",
        awaiting_sme_expansion=True,
    )
    golden_set = GoldenSet(version="legacy", queries=[query])

    with pytest.raises(GoldenSetProvenanceError, match="provenance is missing"):
        validate_golden_set(golden_set)


@pytest.mark.asyncio
async def test_current_golden_set_passes_strict_provenance_validation() -> None:
    result = await load_validated_golden_set(Path("evals/golden/legal_retrieval_v1.jsonl"))

    assert isinstance(result, Success)
    golden_set = result.unwrap()
    assert golden_set.provenance_schema_version == "golden_provenance_v1"
    assert all(query.provenance is not None for query in golden_set.queries)
