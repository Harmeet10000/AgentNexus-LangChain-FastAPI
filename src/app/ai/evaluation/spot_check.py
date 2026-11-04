"""Lightweight deterministic checks for deep research reports."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "under",
        "was",
        "were",
        "with",
    }
)


class SpotCheckError(Exception):
    """Raised when spot-check input cannot be analyzed."""


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    relevance: str | None = None


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str
    title: str
    content: str
    citations: list[Citation] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    document_id: str
    content: str
    embedding: list[float] | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    text: str
    section_id: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class SpotCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    check_name: str
    passed: bool
    issue_description: str | None = None
    severity: Literal["warning", "error"]
    details: dict[str, object] | None = None


class ClaimPairAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim1_text: str
    claim2_text: str
    shared_evidence: bool
    evidence_chunk_ids: list[str]


class ClaimConsistencyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float
    claim_pairs: list[ClaimPairAnalysis]


class AsyncIndexClient(Protocol):
    async def get_chunks(self, chunk_ids: list[str]) -> list[RetrievedChunk]: ...


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if token not in _STOPWORDS}


def _derived_claims(report_sections: list[ReportSection]) -> list[Claim]:
    return [
        Claim(
            claim_id=f"{section.section_id}-{index}",
            text=sentence.strip(),
            section_id=section.section_id,
            evidence_chunk_ids=[citation.chunk_id for citation in section.citations],
        )
        for section in report_sections
        for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", section.content), start=1)
        if sentence.strip()
    ]


def _cosine_similarity(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or not first:
        return None
    first_norm = sqrt(sum(value * value for value in first))
    second_norm = sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return None
    return sum(left * right for left, right in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )


def _shared_evidence_ids(
    first: Claim, second: Claim, embeddings: dict[str, list[float] | None]
) -> list[str]:
    shared_ids = sorted(set(first.evidence_chunk_ids) & set(second.evidence_chunk_ids))
    if shared_ids:
        return shared_ids
    for first_id in first.evidence_chunk_ids:
        first_embedding = embeddings.get(first_id)
        if first_embedding is None:
            continue
        for second_id in second.evidence_chunk_ids:
            second_embedding = embeddings.get(second_id)
            if second_embedding is None:
                continue
            similarity = _cosine_similarity(first_embedding, second_embedding)
            if similarity is not None and similarity >= 0.85:
                return sorted({first_id, second_id})
    return []


async def compute_claim_consistency(
    *, claims: list[Claim], retrieved_chunks: list[RetrievedChunk]
) -> ClaimConsistencyResult:
    """Score evidence overlap; zero or one claim yields 1.0 and no pairs."""
    embeddings = {chunk.chunk_id: chunk.embedding for chunk in retrieved_chunks}
    pairs = [
        ClaimPairAnalysis(
            claim1_text=first.text,
            claim2_text=second.text,
            shared_evidence=bool(shared_ids := _shared_evidence_ids(first, second, embeddings)),
            evidence_chunk_ids=shared_ids,
        )
        for index, first in enumerate(claims)
        for second in claims[index + 1 :]
    ]
    score = sum(pair.shared_evidence for pair in pairs) / len(pairs) if pairs else 1.0
    return ClaimConsistencyResult(score=score, claim_pairs=pairs)


async def run_spot_checks(
    *,
    report_sections: list[ReportSection],
    retrieved_chunks: list[RetrievedChunk],
    index_client: AsyncIndexClient | None = None,
) -> list[SpotCheckResult]:
    """Run the five deterministic spot-check rules without requiring an index client."""
    del index_client
    chunk_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    source_issues = [
        section.section_id
        for section in report_sections
        if not any(citation.chunk_id in chunk_by_id for citation in section.citations)
    ]
    results = [
        SpotCheckResult(
            check_name="source_attribution",
            passed=not source_issues,
            issue_description=(
                "Sections lack a citation to a retrieved chunk: " + ", ".join(source_issues)
                if source_issues
                else None
            ),
            severity="error",
            details={"unattributed_sections": source_issues},
        )
    ]

    chunk_tokens = {token for chunk in retrieved_chunks for token in _tokens(chunk.content)}
    uncovered = sorted(
        {
            token
            for section in report_sections
            for token in _tokens(f"{section.title} {section.content}")
            if token not in chunk_tokens
        }
    )
    results.append(
        SpotCheckResult(
            check_name="topic_coverage",
            passed=not uncovered,
            issue_description=(
                "Topics are absent from retrieved chunks: " + ", ".join(uncovered)
                if uncovered
                else None
            ),
            severity="warning",
            details={"uncovered_topics": uncovered},
        )
    )

    missing_sections = [section.section_id for section in report_sections if not section.citations]
    missing_claims = [
        claim.claim_id for claim in _derived_claims(report_sections) if not claim.evidence_chunk_ids
    ]
    results.append(
        SpotCheckResult(
            check_name="missing_citations",
            passed=not missing_sections and not missing_claims,
            issue_description=(
                "Sections or claims lack citations/evidence"
                if missing_sections or missing_claims
                else None
            ),
            severity="warning",
            details={"sections": missing_sections, "claims": missing_claims},
        )
    )

    citation_density_issues: list[str] = []
    for section in report_sections:
        if section.citations:
            continue
        consecutive = 0
        for paragraph in re.split(r"\n\s*\n", section.content.strip()):
            if re.search(r"\[[^\]]+\]", paragraph):
                consecutive = 0
            else:
                consecutive += 1
                if consecutive > 3:
                    citation_density_issues.append(section.section_id)
                    break
    results.append(
        SpotCheckResult(
            check_name="citation_density",
            passed=not citation_density_issues,
            issue_description=(
                "More than three consecutive paragraphs lack citations"
                if citation_density_issues
                else None
            ),
            severity="error",
            details={"sections": citation_density_issues},
        )
    )

    consistency = await compute_claim_consistency(
        claims=_derived_claims(report_sections), retrieved_chunks=retrieved_chunks
    )
    results.append(
        SpotCheckResult(
            check_name="claim_consistency",
            passed=consistency.score >= 0.5,
            issue_description=(
                f"Claim consistency score is {consistency.score:.3f}, below 0.5"
                if consistency.score < 0.5
                else None
            ),
            severity="warning",
            details={
                "score": consistency.score,
                "claim_pairs": len(consistency.claim_pairs),
            },
        )
    )
    return results


def _fixture_output(path: Path) -> tuple[list[ReportSection], list[RetrievedChunk], list[Claim]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"Fixture {path} must contain a JSON object"
        raise SpotCheckError(message)

    raw_sections = payload.get("report_sections", [])
    raw_documents = payload.get("retrieved_documents", [])
    if not isinstance(raw_sections, list) or not isinstance(raw_documents, list):
        message = f"Fixture {path} has invalid report_sections or retrieved_documents"
        raise SpotCheckError(message)

    sections = [ReportSection.model_validate(section) for section in raw_sections]
    document_ids: dict[str, str] = {}
    for document in raw_documents:
        if isinstance(document, dict):
            document_id = document.get("document_id")
            if isinstance(document_id, str):
                document_ids[document_id] = document_id

    chunk_contents: dict[str, list[str]] = {}
    claims: list[Claim] = []
    for section in sections:
        cited_ids = [citation.chunk_id for citation in section.citations]
        for chunk_id in cited_ids:
            chunk_contents.setdefault(chunk_id, []).append(section.title + " " + section.content)
        claims.extend(
            Claim(
                claim_id=f"{section.section_id}-{index}",
                text=sentence.strip(),
                section_id=section.section_id,
                evidence_chunk_ids=cited_ids,
            )
            for index, sentence in enumerate(re.split(r"(?<=[.!?])\\s+", section.content), start=1)
            if sentence.strip()
        )

    chunks = [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=document_ids.get(
                chunk_id.split("-chunk-")[0], chunk_id.split("-chunk-")[0]
            ),
            content=" ".join(contents),
        )
        for chunk_id, contents in sorted(chunk_contents.items())
    ]
    return sections, chunks, claims


def main(argv: list[str] | None = None) -> int:
    """Run fixture spot checks and write a machine-readable JSON report.

    The gate fails when errors are greater than or equal to ``max_errors``;
    with the default of 3, fewer than three errors pass, matching the CI spec.
    """
    parser = argparse.ArgumentParser(description="Run lightweight research-output spot checks")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument(
        "--fixtures-dir", type=Path, default=Path("evals/fixtures/research_outputs")
    )
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--max-warnings", type=int, default=None)
    args = parser.parse_args(argv)

    if args.max_errors < 0 or (args.max_warnings is not None and args.max_warnings < 0):
        parser.error("error and warning thresholds must be non-negative")
    fixture_paths = sorted(args.fixtures_dir.glob("*.json"))
    outputs: list[dict[str, object]] = []
    for path in fixture_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sections, chunks, claims = _fixture_output(path)
        checks = asyncio.run(run_spot_checks(report_sections=sections, retrieved_chunks=chunks))
        # Recompute the consistency check against explicitly adapted claims.
        consistency = asyncio.run(compute_claim_consistency(claims=claims, retrieved_chunks=chunks))
        checks[-1] = checks[-1].model_copy(
            update={
                "passed": consistency.score >= 0.5,
                "issue_description": (
                    f"Claim consistency score is {consistency.score:.3f}, below 0.5"
                    if consistency.score < 0.5
                    else None
                ),
                "details": {
                    "score": consistency.score,
                    "claim_pairs": len(consistency.claim_pairs),
                },
            }
        )
        results = [check.model_dump(mode="json") for check in checks]
        outputs.append(
            {
                "file": path.name,
                "operation_id": payload.get("operation_id") if isinstance(payload, dict) else None,
                "process_metrics": payload.get("process_metrics", {})
                if isinstance(payload, dict)
                else {},
                "error_count": sum(
                    not check.passed and check.severity == "error" for check in checks
                ),
                "warning_count": sum(
                    not check.passed and check.severity == "warning" for check in checks
                ),
                "results": results,
            }
        )

    error_count = sum(output["error_count"] for output in outputs)
    warning_count = sum(output["warning_count"] for output in outputs)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_outputs": len(outputs),
        "error_count": error_count,
        "warning_count": warning_count,
        "outputs": outputs,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(
        error_count >= args.max_errors
        or (args.max_warnings is not None and warning_count >= args.max_warnings)
    )


if __name__ == "__main__":
    raise SystemExit(main())
