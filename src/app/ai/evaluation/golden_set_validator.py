"""Golden set validation before commit for taxonomy candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure, Success

from .schema import load_validated_golden_set, manifest_path_for

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from returns.result import Result

    from .schema import GoldenSet


class GoldenSetValidationError(Exception):
    """Golden set validation failed during pre-commit checks."""


class GoldenSetValidator:
    """Validator for golden set rows before committing changes."""

    def __init__(
        self,
        *,
        min_rows: int = 20,
        min_per_case_type: int = 5,
    ) -> None:
        """Initialize validator with threshold configuration.

        Args:
            min_rows: Minimum total rows required for ingestion_v1 golden set
            min_per_case_type: Minimum rows required per major case type
        """
        self.min_rows = min_rows
        self.min_per_case_type = min_per_case_type

    @staticmethod
    def validate_manifest_presence(
        golden_set_path: Path,
    ) -> Result[None, str]:
        """Validate that v2 golden sets have a sibling manifest.

        Args:
            golden_set_path: Path to the golden set JSONL file

        Returns:
            Success(None) if manifest exists or v1 set, Failure with error message otherwise
        """
        manifest_path = manifest_path_for(golden_set_path)

        if golden_set_path.name.startswith("legal_retrieval_v2") and not manifest_path.is_file():
            return Failure(
                f"legal_retrieval_v2 golden set requires sibling manifest: {manifest_path.name}"
            )

        return Success(None)

    def validate_coverage_thresholds(
        self,
        golden_set: GoldenSet,
    ) -> Result[None, Sequence[str]]:
        """Validate that golden set meets minimum coverage thresholds.

        Args:
            golden_set: The golden set to validate

        Returns:
            Success(None) if thresholds met, Failure with list of issues otherwise
        """
        issues: list[str] = []

        # Check minimum total rows
        if len(golden_set.queries) < self.min_rows:
            issues.append(
                f"Golden set has {len(golden_set.queries)} rows; minimum required is {self.min_rows}"
            )

        # Count rows per case type
        case_type_counts: dict[str, int] = {}
        for query in golden_set.queries:
            case_type = query.case_type
            case_type_counts[case_type] = case_type_counts.get(case_type, 0) + 1

        # Check minimum per case type
        ingestion_case_types = {
            "no_evidence_vs_ingestion_failure",
            "index_not_ready",
            "partial_coverage_answer",
        }

        for case_type in ingestion_case_types:
            count = case_type_counts.get(case_type, 0)
            if count < self.min_per_case_type:
                issues.append(
                    f"Case type '{case_type}' has {count} rows; minimum required is {self.min_per_case_type}"
                )

        if issues:
            return Failure(issues)

        return Success(None)

    async def validate_for_commit(
        self,
        golden_set_path: Path,
    ) -> Result[None, Sequence[str]]:
        """Run full validation suite before committing golden set changes.

        Args:
            golden_set_path: Path to the golden set JSONL file

        Returns:
            Success(None) if all validations pass, Failure with list of issues otherwise
        """
        # Validate manifest presence
        manifest_result = self.validate_manifest_presence(golden_set_path)
        if isinstance(manifest_result, Failure):
            return Failure([manifest_result.failure()])

        # Load and validate golden set
        load_result = await load_validated_golden_set(golden_set_path)

        if isinstance(load_result, Failure):
            error = load_result.failure()
            return Failure([error.message])

        golden_set = load_result.unwrap()

        # Validate coverage thresholds
        coverage_result = self.validate_coverage_thresholds(golden_set)

        if isinstance(coverage_result, Failure):
            return Failure(coverage_result.failure())

        return Success(None)


async def validate_golden_set_before_commit(
    golden_set_path: Path,
    *,
    min_rows: int = 20,
    min_per_case_type: int = 5,
) -> Result[None, Sequence[str]]:
    """Convenience function to validate a golden set before commit.

    Args:
        golden_set_path: Path to the golden set JSONL file
        min_rows: Minimum total rows required
        min_per_case_type: Minimum rows required per major case type

    Returns:
        Success(None) if validation passes, Failure with list of issues otherwise
    """
    validator = GoldenSetValidator(
        min_rows=min_rows,
        min_per_case_type=min_per_case_type,
    )
    return await validator.validate_for_commit(golden_set_path)
