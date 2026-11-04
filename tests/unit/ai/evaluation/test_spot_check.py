from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ai.evaluation.spot_check import (
    Claim,
    ReportSection,
    RetrievedChunk,
    compute_claim_consistency,
    run_spot_checks,
)

pytestmark = pytest.mark.unit

FIXTURES = Path("evals/fixtures/research_outputs")


def _fixture_sections(name: str) -> tuple[list[ReportSection], list[RetrievedChunk]]:
    payload = json.loads((FIXTURES / name).read_text())
    sections = [ReportSection.model_validate(section) for section in payload["report_sections"]]
    chunk_ids = {citation.chunk_id for section in sections for citation in section.citations}
    content = " ".join(section.title + " " + section.content for section in sections)
    chunks = [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=chunk_id.split("-chunk-")[0],
            content="" if name == "low_quality_report.json" else content,
        )
        for chunk_id in chunk_ids
    ]
    return sections, chunks


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("complete_report.json", {"source_attribution": True, "topic_coverage": True}),
        (
            "missing_citations_report.json",
            {"source_attribution": False, "missing_citations": False},
        ),
        ("low_quality_report.json", {"source_attribution": False, "topic_coverage": False}),
        ("contradictory_claims_report.json", {"claim_consistency": False}),
    ],
)
async def test_fixture_quality_profiles(filename: str, expected: dict[str, bool]) -> None:
    sections, chunks = _fixture_sections(filename)
    checks = await run_spot_checks(report_sections=sections, retrieved_chunks=chunks)
    outcomes = {check.check_name: check.passed for check in checks}
    assert len(checks) == 5
    for check_name, passed in expected.items():
        assert outcomes[check_name] is passed


async def test_source_attribution_requires_retrieved_citation() -> None:
    section = ReportSection(section_id="s", title="Topic", content="A claim.", citations=[])
    checks = await run_spot_checks(report_sections=[section], retrieved_chunks=[])
    assert checks[0].check_name == "source_attribution"
    assert checks[0].passed is False
    assert checks[0].severity == "error"


async def test_topic_coverage_reports_uncovered_tokens() -> None:
    section = ReportSection(section_id="s", title="Alien topic", content="Zorblax.")
    chunks = [RetrievedChunk(chunk_id="c", document_id="d", content="ordinary text")]
    checks = await run_spot_checks(report_sections=[section], retrieved_chunks=chunks)
    topic_check = next(check for check in checks if check.check_name == "topic_coverage")
    assert topic_check.passed is False
    assert "zorblax" in topic_check.details["uncovered_topics"]


async def test_missing_citations_flags_section() -> None:
    section = ReportSection(section_id="s", title="Topic", content="Claim.")
    checks = await run_spot_checks(report_sections=[section], retrieved_chunks=[])
    missing = next(check for check in checks if check.check_name == "missing_citations")
    assert missing.passed is False
    assert missing.severity == "warning"


async def test_citation_density_triggers_after_four_uncited_paragraphs() -> None:
    section = ReportSection(
        section_id="s", title="Topic", content="One.\n\nTwo.\n\nThree.\n\nFour."
    )
    checks = await run_spot_checks(report_sections=[section], retrieved_chunks=[])
    density = next(check for check in checks if check.check_name == "citation_density")
    assert density.passed is False
    assert density.severity == "error"


async def test_citation_density_accepts_inline_marker() -> None:
    section = ReportSection(
        section_id="s", title="Topic", content="One.\n\nTwo.\n\nThree [chunk-1].\n\nFour."
    )
    checks = await run_spot_checks(report_sections=[section], retrieved_chunks=[])
    density = next(check for check in checks if check.check_name == "citation_density")
    assert density.passed is True


async def test_claim_consistency_shared_evidence() -> None:
    claims = [
        Claim(claim_id="a", text="a", evidence_chunk_ids=["c1"]),
        Claim(claim_id="b", text="b", evidence_chunk_ids=["c1"]),
    ]
    result = await compute_claim_consistency(claims=claims, retrieved_chunks=[])
    assert result.score == pytest.approx(1.0)
    assert result.claim_pairs[0].shared_evidence is True
    assert result.claim_pairs[0].evidence_chunk_ids == ["c1"]


async def test_claim_consistency_without_shared_evidence() -> None:
    claims = [
        Claim(claim_id="a", text="a", evidence_chunk_ids=["c1"]),
        Claim(claim_id="b", text="b", evidence_chunk_ids=["c2"]),
    ]
    result = await compute_claim_consistency(claims=claims, retrieved_chunks=[])
    assert result.score == pytest.approx(0.0)
    assert result.claim_pairs[0].shared_evidence is False


async def test_claim_consistency_embedding_similarity() -> None:
    claims = [
        Claim(claim_id="a", text="a", evidence_chunk_ids=["c1"]),
        Claim(claim_id="b", text="b", evidence_chunk_ids=["c2"]),
    ]
    chunks = [
        RetrievedChunk(chunk_id="c1", document_id="d", content="", embedding=[1.0, 0.0]),
        RetrievedChunk(chunk_id="c2", document_id="d", content="", embedding=[0.9, 0.1]),
    ]
    result = await compute_claim_consistency(claims=claims, retrieved_chunks=chunks)
    assert result.score == pytest.approx(1.0)
    assert result.claim_pairs[0].shared_evidence is True


@pytest.mark.parametrize(
    "claims",
    [[], [Claim(claim_id="a", text="a")]],
)
async def test_claim_consistency_zero_or_one_claim(claims: list[Claim]) -> None:
    result = await compute_claim_consistency(claims=claims, retrieved_chunks=[])
    assert result.score == pytest.approx(1.0)
    assert result.claim_pairs == []
