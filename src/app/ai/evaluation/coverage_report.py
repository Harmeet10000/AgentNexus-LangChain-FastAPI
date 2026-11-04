"""Coverage audit report generator for ingestion quality analysis."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import aiofiles
from returns.result import Failure

from .ingestion import audit_ingestion_coverage
from .schema import load_validated_golden_set

if TYPE_CHECKING:
    from pathlib import Path

    from .schema import CoverageAuditReport

_MISSING_STATES_DISPLAY_LIMIT = 5


async def generate_coverage_audit_report(
    golden_set_path: Path,
    output_path: Path,
    *,
    minimum_per_type: int = 5,
) -> None:
    """Generate a coverage audit report for an ingestion golden set.

    Args:
        golden_set_path: Path to the golden set JSONL file
        output_path: Path where the report JSON should be written
        minimum_per_type: Minimum number of rows required per case type
    """
    load_result = await load_validated_golden_set(golden_set_path)

    if isinstance(load_result, Failure):
        error_msg = f"Failed to load golden set: {load_result.failure().message}"
        raise RuntimeError(error_msg)  # noqa: TRY004

    golden_set = load_result.unwrap()

    manifest_path = golden_set_path.with_name(f"{golden_set_path.stem}.manifest.json")
    if manifest_path.is_file():
        async with aiofiles.open(manifest_path) as f:
            _manifest_raw = json.loads(await f.read())  # reserved for future use

    report = await audit_ingestion_coverage(
        golden_rows=golden_set.queries,
        minimum_per_type=minimum_per_type,
    )

    comprehensive_report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "golden_set": str(golden_set_path),
        "version": golden_set.version,
        "summary": {
            "total_rows": report.total_rows,
            "total_case_types": len(report.case_type_coverage),
            "coverage_gaps": len(report.gaps),
        },
        "case_type_coverage": report.case_type_coverage,
        "coverage_gaps": [
            {
                "case_type": gap.case_type,
                "current_count": gap.current_count,
                "required_minimum": gap.required_minimum,
                "suggested_queries": gap.suggested_queries,
            }
            for gap in report.gaps
        ],
        "missing_ingestion_states": report.missing_ingestion_states,
        "recommendations": report.recommendations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(comprehensive_report, indent=2))
        await f.write("\n")


def generate_coverage_summary(
    report: CoverageAuditReport,
    *,
    minimum_per_type: int = 5,
) -> str:
    """Generate a human-readable summary of the coverage audit report.

    Args:
        report: The coverage audit report to summarize
        minimum_per_type: Minimum number of rows required per case type

    Returns:
        A formatted string summary of the coverage audit.
    """
    lines: list[str] = [
        "=" * 60,
        "INGESTION COVERAGE AUDIT SUMMARY",
        "=" * 60,
        "",
        f"Total rows analyzed: {report.total_rows}",
        f"Case types covered: {len(report.case_type_coverage)}",
        f"Coverage gaps found: {len(report.gaps)}",
        "",
        "CASE TYPE BREAKDOWN",
        "-" * 40,
    ]

    lines.extend(
        f"  {'✓' if count >= minimum_per_type else '✗'} {case_type}: {count} rows"
        for case_type, count in sorted(report.case_type_coverage.items())
    )
    lines.append("")

    if report.gaps:
        lines.extend(("COVERAGE GAPS", "-" * 40))
        for gap in report.gaps:
            lines.extend(
                (
                    f"  ✗ {gap.case_type}: {gap.current_count}/{gap.required_minimum}",
                    "    Suggestions:",
                )
            )
            lines.extend(f"      - {s}" for s in gap.suggested_queries)
        lines.append("")

    if report.missing_ingestion_states:
        lines.extend(("MISSING INGESTION STATES", "-" * 40))
        lines.extend(
            f"  - {m}" for m in report.missing_ingestion_states[:_MISSING_STATES_DISPLAY_LIMIT]
        )
        if len(report.missing_ingestion_states) > _MISSING_STATES_DISPLAY_LIMIT:
            lines.append(
                f"  ... and {len(report.missing_ingestion_states) - _MISSING_STATES_DISPLAY_LIMIT} more"
            )
        lines.append("")

    if report.recommendations:
        lines.extend(("RECOMMENDATIONS", "-" * 40))
        lines.extend(f"  • {rec}" for rec in report.recommendations)
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


async def generate_coverage_report_from_audit(
    audit_report: CoverageAuditReport,
    output_path: Path,
) -> None:
    """Generate a coverage report from an existing audit report.

    Args:
        audit_report: The coverage audit report to serialize
        output_path: Path where the report JSON should be written
    """
    comprehensive_report = {
        "total_rows": audit_report.total_rows,
        "case_type_coverage": audit_report.case_type_coverage,
        "coverage_gaps": [
            {
                "case_type": gap.case_type,
                "current_count": gap.current_count,
                "required_minimum": gap.required_minimum,
                "suggested_queries": gap.suggested_queries,
            }
            for gap in audit_report.gaps
        ],
        "missing_ingestion_states": audit_report.missing_ingestion_states,
        "recommendations": audit_report.recommendations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(comprehensive_report, indent=2))
        await f.write("\n")
