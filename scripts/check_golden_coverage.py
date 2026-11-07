"""Coverage-matrix check and row-map generator for legal_retrieval_v2.

The matrix documented in ``evals/golden/legal_retrieval_v2.COVERAGE.md`` is the
contract: every golden row must map to a declared matrix cell, and the corpus
composition must cover the documented families, the multi-contract documents
and the scanned-OCR profile. This script is both the generator of the
auto-maintained row-map section of that file (``--write``) and the check run in
CI (default): any row whose (document_kind, jurisdiction, difficulty) cell is
not declared in the matrix, or any composition gap, fails.

The query-style dimension is a documented coarse classification of the query
text (temporal / permission / cross_reference / obligation_lookup); it is a
planning aid, not a golden field.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

GOLDEN_PATH = Path("evals/golden/legal_retrieval_v2.jsonl")
COVERAGE_PATH = Path("evals/golden/legal_retrieval_v2.COVERAGE.md")

MATRIX_TARGET_MIN = 150
MATRIX_TARGET_MAX = 300

# Documents the corpus records with multi-contract boundaries (README section 4).
MULTI_CONTRACT_DOCS = ("SRC-04", "SRC-14", "SYN-02")
# Documents carrying the scanned-OCR profile: pure scans (heavy noise) and the
# noisy scanned prefix / scanned-style render (light noise).
SCANNED_OCR_DOCS = ("SRC-05", "SRC-06", "SRC-07", "SYN-05")
SCANNED_OCR_PROFILES = ("light_noise", "heavy_noise")

QUERY_KINDS = ("contracts", "statutes", "judgments", "filings")
JURISDICTIONS = ("India", "GB")
DIFFICULTIES = ("easy", "medium", "hard")
QUERY_STYLES = ("obligation_lookup", "cross_reference", "permission", "temporal")

_TEMPORAL_PATTERN = re.compile(
    r"how many days|within how many|on what date|what date|by how many days"
    r"|within what (?:period|cure)|how soon|within what time|no later than"
    r"|how long|what period|within what",
    re.IGNORECASE,
)
_PERMISSION_PATTERN = re.compile(
    r"^(?:does|can|may|must|is|are)\b.*\?|allow|prevent|required to prove|excused",
    re.IGNORECASE,
)
_CROSS_REFERENCE_PATTERN = re.compile(
    r"\b(?:clause|article|section|schedule|annexure|appendix|annex|amendment|addendum"
    r"|tender number|framework|edition)\b|which (?:contract|part|appendix|schedule)",
    re.IGNORECASE,
)
_CROSS_REFERENCE_CASE_TYPES = frozenset(
    {
        "multi_document",
        "multi_hop",
        "multi_contract_boundary",
        "cross_tenant_isolation",
        "stale_version",
        "amendment_overrides",
        "annexure_as_main",
        "wrong_jurisdiction",
        "wrong_date_version_authority",
    }
)


def classify_style(query: str, case_type: str) -> str:
    """Classify a query into one documented style bucket (coarse, documented)."""
    if _TEMPORAL_PATTERN.search(query):
        return "temporal"
    if _PERMISSION_PATTERN.search(query):
        return "permission"
    if case_type in _CROSS_REFERENCE_CASE_TYPES or _CROSS_REFERENCE_PATTERN.search(query):
        return "cross_reference"
    return "obligation_lookup"


def load_rows() -> list[dict]:
    lines = GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[1:]]


def cell_of(row: dict) -> str:
    return f"{row['document_kind']} / {row['jurisdiction']} / {row['difficulty']}"


def all_declared_cells() -> list[str]:
    return [
        f"{kind} / {jurisdiction} / {difficulty}"
        for kind in QUERY_KINDS
        for jurisdiction in JURISDICTIONS
        for difficulty in DIFFICULTIES
    ]


def coverage_stats(rows: list[dict]) -> dict[str, object]:
    cells = Counter(cell_of(row) for row in rows)
    styles = Counter(classify_style(row["query"], row["case_type"]) for row in rows)
    referenced: set[str] = set()
    for row in rows:
        referenced.update(row.get("expected_document_ids", []))
        referenced.update(span["document_id"] for span in row.get("expected_spans", []))
        referenced.update(
            state["document_id"] for state in row.get("expected_ingestion_states", [])
        )
    return {
        "count": len(rows),
        "cells": cells,
        "styles": styles,
        "referenced_documents": referenced,
        "multi_contract_rows": sum(
            1 for row in rows if row.get("case_type") == "multi_contract_boundary"
        ),
        "scanned_ocr_rows": sum(
            1 for row in rows if row.get("ocr_profile") in SCANNED_OCR_PROFILES
        ),
    }


def check(rows: list[dict]) -> list[str]:
    issues: list[str] = []
    declared = set(all_declared_cells())
    stats = coverage_stats(rows)
    if not MATRIX_TARGET_MIN <= len(rows) <= MATRIX_TARGET_MAX:
        issues.append(
            f"corpus has {len(rows)} rows; the matrix target is "
            f"{MATRIX_TARGET_MIN}-{MATRIX_TARGET_MAX}"
        )
    for index, row in enumerate(rows, start=2):
        if cell_of(row) not in declared:
            issues.append(
                f"line {index} ({row.get('case_id')}): undeclared matrix cell {cell_of(row)}"
            )
    missing_styles = [style for style in QUERY_STYLES if stats["styles"].get(style, 0) == 0]  # type: ignore[literal-required]
    issues.extend(f"query style {style} has no rows" for style in missing_styles)
    missing_kinds = [
        kind for kind in QUERY_KINDS if not any(row["document_kind"] == kind for row in rows)
    ]
    issues.extend(f"document kind {kind} has no rows" for kind in missing_kinds)
    missing_multi = [doc for doc in MULTI_CONTRACT_DOCS if doc not in stats["referenced_documents"]]
    if missing_multi:
        issues.append(f"multi-contract documents not referenced by any row: {missing_multi}")
    if not stats["scanned_ocr_rows"]:
        issues.append("scanned-OCR profile has no rows")
    return issues


def row_map_markdown(rows: list[dict]) -> str:
    stats = coverage_stats(rows)
    lines = [
        f"Auto-generated by `scripts/check_golden_coverage.py --write`; {len(rows)} rows.",
        "",
        "| case_id | kind / jurisdiction / difficulty | style | ocr | case_type |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {row['case_id']} | {cell_of(row)} | {classify_style(row['query'], row['case_type'])} "
        f"| {row.get('ocr_profile', 'clean')} | {row['case_type']} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            f"Cell occupancy: {dict(sorted(stats['cells'].items()))}",  # type: ignore[arg-type]
            f"Style counts: {dict(sorted(stats['styles'].items()))}",  # type: ignore[arg-type]
        ]
    )
    return "\n".join(lines)


def write_row_map(rows: list[dict]) -> None:
    text = COVERAGE_PATH.read_text(encoding="utf-8")
    begin = "<!-- BEGIN ROW MAP -->"
    end = "<!-- END ROW MAP -->"
    pre, _, rest = text.partition(begin)
    _, _, post = rest.partition(end)
    updated = f"{pre}{begin}\n{row_map_markdown(rows)}\n{end}{post}"
    COVERAGE_PATH.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the row-map section")
    args = parser.parse_args()
    rows = load_rows()
    if args.write:
        write_row_map(rows)
    issues = check(rows)
    if issues:
        for issue in issues:
            sys.stderr.write(issue + "\n")
        raise SystemExit(1)
    sys.stdout.write(f"coverage check passed: {len(rows)} rows map to declared matrix cells\n")


if __name__ == "__main__":
    main()
