"""Ingestion state validation for evaluation golden sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure, Success

from .schema import (
    CoverageAuditReport,
    CoverageGap,
    CoverageMetric,
    GoldenIngestionState,
    IngestionValidationReport,
    IngestionValidationResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from returns.result import Result

    from .schema import (
        GoldenEvidencePolicy,
        GoldenProvenanceManifest,
        GoldenQuery,
        GoldenSet,
    )


class IngestionValidationError(Exception):
    """Base class for ingestion validation errors."""


class MissingProvenanceError(IngestionValidationError):
    """Golden set row missing provenance information."""


class StateMismatchError(IngestionValidationError):
    """Expected ingestion state does not match actual state."""

    def __init__(
        self,
        document_id: str,
        expected: GoldenIngestionState,
        actual: GoldenIngestionState | None,
    ) -> None:
        self.document_id = document_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"Document {document_id}: expected {expected}, got {actual}")


class PolicyValidationError(IngestionValidationError):
    """Ingestion validation failed under specified policy."""


class GoldenSetValidationError(Exception):
    """Golden set validation failed during pre-commit checks."""


def _build_coverage_metric(
    policy: GoldenEvidencePolicy,
    total_documents: dict[GoldenEvidencePolicy, set[str]],
    state_counts: dict[GoldenEvidencePolicy, dict[GoldenIngestionState, int]],
) -> CoverageMetric:
    """Compute a CoverageMetric from accumulated tracking dicts."""
    policy_docs = total_documents[policy]
    total = len(policy_docs) if policy_docs else 1
    counts = state_counts[policy]
    ready = counts[GoldenIngestionState.READY]
    return CoverageMetric(
        total_documents=total,
        ready_documents=ready,
        completed_with_warnings_documents=counts[GoldenIngestionState.COMPLETED_WITH_WARNINGS],
        index_not_ready_documents=counts[GoldenIngestionState.INDEX_NOT_READY],
        failed_documents=counts[GoldenIngestionState.FAILED],
        unauthorized_documents=counts[GoldenIngestionState.UNAUTHORIZED],
        coverage_ratio=ready / total,
    )


async def validate_ingestion_states(
    *,
    golden_rows: list[GoldenQuery],
    ingestion_state_provider: Callable[[str], Awaitable[GoldenIngestionState]],
    policy: GoldenEvidencePolicy = "mechanically_verified",
) -> IngestionValidationReport:
    """Validate each row's declared ingestion states against actual state.

    Args:
        golden_rows: List of golden query rows to validate
        ingestion_state_provider: Async function that takes a document_id
            and returns the actual ingestion state for that document
        policy: Evidence policy to use for validation

    Returns:
        IngestionValidationReport with validation results and metrics
    """
    results: list[IngestionValidationResult] = []
    total_documents: dict[GoldenEvidencePolicy, set[str]] = {
        "mechanically_verified": set(),
        "graph_indexed_only": set(),
    }
    state_counts: dict[GoldenEvidencePolicy, dict[GoldenIngestionState, int]] = {
        "mechanically_verified": dict.fromkeys(GoldenIngestionState, 0),
        "graph_indexed_only": dict.fromkeys(GoldenIngestionState, 0),
    }

    for row_index, row in enumerate(golden_rows, start=1):
        if row.case_type not in {"no_evidence_vs_ingestion_failure", "index_not_ready"}:
            continue
        for state_info in row.expected_ingestion_states:
            total_documents[policy].add(state_info.document_id)
        for state_info in row.expected_ingestion_states:
            actual_state = await ingestion_state_provider(state_info.document_id)
            state_counts[policy][
                actual_state if actual_state is not None else GoldenIngestionState.FAILED
            ] += 1
            results.append(
                IngestionValidationResult(
                    row_index=row_index,
                    case_id=row.case_id,
                    document_id=state_info.document_id,
                    expected_state=state_info.state,
                    actual_state=actual_state,
                    match=actual_state == state_info.state,
                    policy=policy,
                    details=None,
                )
            )

    return IngestionValidationReport(
        golden_set_version="ingestion_v1",
        validation_timestamp="2024-01-01T00:00:00Z",  # Placeholder
        total_rows=len(results),
        passed_rows=sum(1 for r in results if r.match),
        failed_rows=sum(1 for r in results if not r.match),
        results=results,
        coverage_metrics={policy: _build_coverage_metric(policy, total_documents, state_counts)},
    )


def validate_ingestion_provenance(
    *,
    golden_set: GoldenSet,
    manifest: GoldenProvenanceManifest,
) -> Result[None, list[str]]:
    """Validate provenance schema and manifest presence.

    This function serves as a CI gate to ensure golden sets have proper
    provenance validation before being used in evaluation or deployment.

    Args:
        golden_set: The golden set to validate
        manifest: The provenance manifest for the golden set

    Returns:
        Success(None) if provenance is valid, or Failure with list of issues
    """
    issues: list[str] = []

    # Check provenance schema version
    if golden_set.provenance_schema_version != "golden_provenance_v1":
        issues.append(
            f"golden set is missing provenance_schema_version=golden_provenance_v1, "
            f"found: {golden_set.provenance_schema_version!r}"
        )

    # Check manifest has required fields
    if not manifest.manifest_version:
        issues.append("manifest is missing manifest_version")

    if not manifest.golden_set_version:
        issues.append("manifest is missing golden_set_version")

    if not manifest.artifact:
        issues.append("manifest is missing artifact")

    # Check manifest per_artifact is not empty for v2 sets
    if golden_set.version == "legal_retrieval_v2" and not manifest.per_artifact:
        issues.append("v2 manifest has empty per_artifact; provenance audit cannot be performed")

    if issues:
        return Failure(issues)
    return Success(None)


# Query templates per ingestion-relevant case type for candidate generation.
_INGESTION_QUERY_TEMPLATES: dict[str, list[str]] = {
    "no_evidence_vs_ingestion_failure": [
        "What are the indemnification obligations under the agreement?",
        "What are the termination conditions in the contract?",
        "What liability caps apply under this agreement?",
        "What are the payment terms and late fees?",
        "What warranties are provided by the vendor?",
    ],
    "index_not_ready": [
        "What are the governing law provisions?",
        "What dispute resolution mechanisms are specified?",
        "What are the confidentiality obligations?",
        "What intellectual property rights are assigned?",
        "What are the force majeure conditions?",
    ],
    "partial_coverage_answer": [
        "Summarize the key obligations across all schedules.",
        "What are the aggregate liability limits across all documents?",
        "List all renewal and termination rights across the agreement set.",
        "What data protection obligations apply across all parties?",
        "What are the audit rights specified in the agreement?",
    ],
}


def generate_candidate_queries(
    gaps: list[CoverageGap],
) -> list[dict[str, object]]:
    """Generate templated candidate queries for case types with coverage gaps.

    Args:
        gaps: Coverage gaps from a CoverageAuditReport

    Returns:
        List of candidate query dicts with ``query``, ``case_type``, and ``notes`` keys.
    """
    candidates: list[dict[str, object]] = []
    for gap in gaps:
        templates = _INGESTION_QUERY_TEMPLATES.get(gap.case_type, [])
        needed = gap.required_minimum - gap.current_count
        candidates.extend(
            {
                "query": template,
                "case_type": gap.case_type,
                "notes": (
                    f"Auto-generated candidate for {gap.case_type}; "
                    "requires SME review before promotion to release tier."
                ),
            }
            for template in templates[:needed]
        )
    return candidates


async def audit_ingestion_coverage(
    *,
    golden_rows: list[GoldenQuery],
    minimum_per_type: int = 5,
) -> CoverageAuditReport:
    """Map rows to case types and identify coverage gaps.

    Args:
        golden_rows: List of golden query rows from the golden set
        minimum_per_type: Minimum number of rows required per case type

    Returns:
        CoverageAuditReport with coverage analysis and recommendations
    """
    case_type_counts: dict[str, int] = {}
    for row in golden_rows:
        case_type_counts[row.case_type] = case_type_counts.get(row.case_type, 0) + 1

    gaps: list[CoverageGap] = []
    for case_type, count in case_type_counts.items():
        if count < minimum_per_type:
            templates = _INGESTION_QUERY_TEMPLATES.get(case_type, [])
            needed = minimum_per_type - count
            gaps.append(
                CoverageGap(
                    case_type=case_type,
                    current_count=count,
                    required_minimum=minimum_per_type,
                    suggested_queries=templates[:needed]
                    or [
                        f"Add query for {case_type} case type",
                        f"Increase coverage for {case_type} to {minimum_per_type} rows",
                    ],
                )
            )

    missing_ingestion_states: list[str] = [
        row.case_id or row.query[:50]
        for row in golden_rows
        if row.case_type in {"no_evidence_vs_ingestion_failure", "index_not_ready"}
        and not row.expected_ingestion_states
    ]

    recommendations: list[str] = []
    if gaps:
        recommendations.append(f"Add {minimum_per_type} or more rows for each case type with gaps")
    if missing_ingestion_states:
        recommendations.append(
            "Ensure ingestion-failure cases have expected_ingestion_states defined"
        )
    if not gaps and not missing_ingestion_states:
        recommendations.append("Ingestion coverage meets minimum requirements")

    return CoverageAuditReport(
        total_rows=len(golden_rows),
        case_type_coverage=case_type_counts,
        gaps=gaps,
        missing_ingestion_states=missing_ingestion_states,
        recommendations=recommendations,
    )
