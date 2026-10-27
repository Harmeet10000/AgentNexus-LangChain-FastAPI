"""Fold agreed wave-3 engineer annotations into the v2 gold set.

Runs AFTER ``compute_iaa.py`` reports a passing gate.  For every gold row it
attaches both ``AnnotationRecord``-shaped sidecar entries (minus their
``case_id`` binding key, which the schema forbids on attached records) to
``provenance.annotations``, escalates ``human_review_status`` from ``pending``
to ``engineer_reviewed``, and applies the tier rules:

- agreement on ``retrieval_relevance`` + ``grader == "deterministic"`` (read
  from the gold row) + no unresolved disagreement  =>  ``quality_tier=release``
  and ``adjudication={status: not_required, method: independent_sme,
  adjudicator: null, rationale, checklist: {}, decided_at: <utc>}``;
- ``grader == "sme"`` (any non-deterministic grader) or any disagreement  =>
  ``quality_tier=candidate`` and ``adjudication.status=pending_sme``.  An SME
  will adjudicate later; a model is never recorded as adjudicator.

With ``--apply`` the merged JSONL is written to a temp file, re-validated with
``python -m app.ai.evaluation.validate`` (the exact CI gate), and only moved
into place when validation passes; otherwise the original gold file is left
untouched and the script exits 1.  Without ``--apply`` it is a dry run that
prints what would change.  The metadata line 1 and row order are preserved.

Usage:
    uv run python scripts/merge_annotations.py \
        --gold evals/golden/legal_retrieval_v2.jsonl \
        --a evals/annotations/wave3-annotator-a.jsonl \
        --b evals/annotations/wave3-annotator-b.jsonl \
        --iaa evals/reports/iaa_wave3.json [--apply]

Exit codes: 0 success (or clean dry run), 1 structural error (including a
failed post-merge validation), 2 the IAA gate has not passed.

Dependency-free: standard library only, mirroring ``compute_iaa.py``.  The
sidecar validation helpers are imported from ``compute_iaa`` so both tools
enforce one identical contract; the fallback import keeps the script runnable
when executed directly as a file (``python scripts/merge_annotations.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # noqa: S404 — invokes the project's own validate module only
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Any

try:
    from scripts.compute_iaa import load_jsonl_objects, relevance_of, validate_sidecar
except ImportError:  # pragma: no cover - direct script execution puts scripts/ on sys.path
    from compute_iaa import load_jsonl_objects, relevance_of, validate_sidecar


_EXIT_OK = 0
_EXIT_STRUCTURAL = 1
_EXIT_GATE_FAILED = 2

_RELEASE_RATIONALE = (
    "dual engineer agreement on deterministic row; machine-verified deterministic expectations"
)
_DISAGREEMENT_RATIONALE = (
    "dual engineer disagreement on retrieval_relevance; SME adjudication required"
)
_NON_DETERMINISTIC_RATIONALE = "row is not deterministically graded; SME adjudication pending"


class _MergeInputs(NamedTuple):
    """Everything loaded from disk before the merge rules run."""

    metadata_line: str | None
    rows: list[dict[str, Any]]
    by_case_a: dict[str, dict[str, Any]]
    by_case_b: dict[str, dict[str, Any]]
    report: dict[str, Any]


def _utc_now() -> str:
    """Return the current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_gold_rows(path: Path, errors: list[str]) -> tuple[str | None, list[dict[str, Any]]]:
    """Load the gold JSONL: metadata line 1 plus one query object per row."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{path}: unreadable ({exc})")
        return None, []
    text = text.removesuffix("\n")  # drop one trailing newline, keep interior blank lines
    lines = text.split("\n")
    if not lines:
        errors.append(f"{path}: gold set is empty")
        return None, []
    metadata_line = lines[0]
    try:
        metadata = json.loads(metadata_line)
    except json.JSONDecodeError as exc:
        errors.append(f"{path}:1: malformed metadata JSON ({exc.msg})")
        return None, []
    if not isinstance(metadata, dict) or not isinstance(metadata.get("version"), str):
        errors.append(f"{path}:1: metadata must be an object with a version string")
        return None, []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines[1:], start=2):
        if not line.strip():
            errors.append(f"{path}:{line_no}: blank line inside gold set")
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_no}: malformed JSON ({exc.msg})")
            continue
        if not isinstance(parsed, dict) or not isinstance(parsed.get("case_id"), str):
            errors.append(f"{path}:{line_no}: row must be an object with a case_id")
            continue
        if not isinstance(parsed.get("provenance"), dict):
            errors.append(f"{path}:{line_no}: row provenance is missing")
            continue
        rows.append(parsed)
    return metadata_line, rows


def load_iaa_report(path: Path, errors: list[str]) -> dict[str, Any]:
    """Load the IAA gate report emitted by ``compute_iaa.py``."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: unreadable or malformed IAA report ({exc})")
        return {}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("gate"), dict):
        errors.append(f"{path}: not a compute_iaa report (missing gate object)")
        return {}
    return parsed


def _release_adjudication(existing: object) -> dict[str, Any]:
    """Build the release-tier adjudication record, preserving decided_at on re-runs."""
    if (
        isinstance(existing, dict)
        and existing.get("status") == "not_required"
        and existing.get("method") == "independent_sme"
        and existing.get("rationale") == _RELEASE_RATIONALE
    ):
        adjudication = dict(existing)
        adjudication.setdefault("adjudicator", None)
        adjudication.setdefault("decided_at", _utc_now())
        adjudication.setdefault("checklist", {})
        return adjudication
    return {
        "status": "not_required",
        "method": "independent_sme",
        "adjudicator": None,
        "rationale": _RELEASE_RATIONALE,
        "checklist": {},
        "decided_at": _utc_now(),
    }


def _pending_sme_adjudication(existing: object, rationale: str) -> dict[str, Any]:
    """Build the pending-SME adjudication record without ever naming an adjudicator."""
    adjudication = dict(existing) if isinstance(existing, dict) else {}
    adjudication["status"] = "pending_sme"
    adjudication.setdefault("rationale", rationale)
    return adjudication


def _annotation_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Strip the case binding key; attached records must match AnnotationRecord."""
    return {key: value for key, value in record.items() if key != "case_id"}


def merge_row(row: dict[str, Any], record_a: dict[str, Any], record_b: dict[str, Any]) -> list[str]:
    """Fold both annotation records into one gold row; return change descriptions."""
    changes: list[str] = []
    provenance: dict[str, Any] = row["provenance"]
    payload_a = _annotation_payload(record_a)
    payload_b = _annotation_payload(record_b)
    annotator_ids = {payload_a["annotator_id"], payload_b["annotator_id"]}

    existing_annotations = provenance.get("annotations")
    prior_annotations = existing_annotations if isinstance(existing_annotations, list) else []
    retained = [
        record
        for record in prior_annotations
        if isinstance(record, dict) and record.get("annotator_id") not in annotator_ids
    ]
    merged_annotations = [*retained, payload_a, payload_b]
    if merged_annotations != prior_annotations:
        changes.append(f"annotations: {len(prior_annotations)} -> {len(merged_annotations)}")
    provenance["annotations"] = merged_annotations

    if provenance.get("human_review_status") == "pending":
        provenance["human_review_status"] = "engineer_reviewed"
        changes.append("human_review_status: pending -> engineer_reviewed")

    agreed = relevance_of(record_a) == relevance_of(record_b)
    deterministic = row.get("grader") == "deterministic"
    prior_tier = provenance.get("quality_tier")
    existing_adjudication: object = provenance.get("adjudication")
    if agreed and deterministic:
        provenance["quality_tier"] = "release"
        adjudication = _release_adjudication(existing_adjudication)
        if adjudication != existing_adjudication:
            changes.append("adjudication: set to not_required (independent_sme)")
        if prior_tier != "release":
            changes.append(f"quality_tier: {prior_tier} -> release")
    else:
        provenance["quality_tier"] = "candidate"
        rationale = _DISAGREEMENT_RATIONALE if not agreed else _NON_DETERMINISTIC_RATIONALE
        adjudication = _pending_sme_adjudication(existing_adjudication, rationale)
        if adjudication != existing_adjudication:
            changes.append("adjudication.status: -> pending_sme")
        if prior_tier != "candidate":
            changes.append(f"quality_tier: {prior_tier} -> candidate")
    provenance["adjudication"] = adjudication
    return changes


def _run_gold_validation(target: Path, repo_root: Path) -> tuple[bool, str]:
    """Run the exact CI validation module against a candidate gold file."""
    src_path = str(repo_root / "src")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path
    env = {**os.environ, "PYTHONPATH": pythonpath}
    completed = subprocess.run(  # noqa: S603 — fixed module invocation; path comes from CLI args
        [sys.executable, "-m", "app.ai.evaluation.validate", str(target)],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stderr + completed.stdout).strip()
    return completed.returncode == 0, output


def _write_and_validate(payload: str, out_path: Path, repo_root: Path) -> list[str]:
    """Write to a temp file, validate it, and move into place; return error strings.

    The temp file never outlives this call: every exit path either unlinks it
    explicitly or moves it onto ``out_path``.  Any exception raised inside the
    write/validate window (including ``KeyboardInterrupt`` and ``SystemExit``)
    unlinks the temp file before propagating, so a crashed merge can never
    leave ``<gold>.merge-*.tmp`` litter behind.
    """
    handle_fd, temp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=f"{out_path.name}.merge-", suffix=".tmp"
    )
    os.close(handle_fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(payload, encoding="utf-8")
        passed, message = _run_gold_validation(temp_path, repo_root)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        return [f"write failed: {exc}"]
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    if not passed:
        temp_path.unlink(missing_ok=True)
        return [f"merged gold failed validation; rolled back: {message}"]
    # mkstemp creates the file 0o600; normalize to the plain-file default so
    # the replaced gold set keeps world-readable permissions.
    temp_path.chmod(0o644)
    temp_path.replace(out_path)
    return []


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Declare and parse the merge CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path, help="gold JSONL to update")
    parser.add_argument("--a", required=True, type=Path, help="annotator A sidecar JSONL")
    parser.add_argument("--b", required=True, type=Path, help="annotator B sidecar JSONL")
    parser.add_argument("--iaa", required=True, type=Path, help="compute_iaa gate report")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: the --gold path, i.e. in place)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the merged gold set (default: dry run)"
    )
    return parser.parse_args(argv)


def _load_inputs(args: argparse.Namespace, errors: list[str]) -> _MergeInputs:
    """Load gold rows, both sidecars, and the IAA gate report."""
    metadata_line, rows = load_gold_rows(args.gold, errors)
    records_a = load_jsonl_objects(args.a, errors)
    records_b = load_jsonl_objects(args.b, errors)
    by_case_a = validate_sidecar(args.a, records_a, errors)
    by_case_b = validate_sidecar(args.b, records_b, errors)
    report = load_iaa_report(args.iaa, errors)
    return _MergeInputs(metadata_line, rows, by_case_a, by_case_b, report)


def _check_consistency(inputs: _MergeInputs, args: argparse.Namespace, errors: list[str]) -> None:
    """Fail on duplicate/missing/extra cases and on a report older than the sidecars."""
    rows_by_case: dict[str, dict[str, Any]] = {}
    for row in inputs.rows:
        case_id: str = row["case_id"]
        if case_id in rows_by_case:
            errors.append(f"{args.gold}: duplicate case_id {case_id!r}")
        rows_by_case[case_id] = row
    gold_case_ids = set(rows_by_case)
    for name, by_case in (("annotator A", inputs.by_case_a), ("annotator B", inputs.by_case_b)):
        errors.extend(
            f"{name}: no annotation for gold case {missing!r}"
            for missing in sorted(gold_case_ids - set(by_case))
        )
        errors.extend(
            f"{name}: annotation for unknown gold case {extra!r}"
            for extra in sorted(set(by_case) - gold_case_ids)
        )
    if not inputs.report or errors:
        return
    disagreements = sorted(
        case_id
        for case_id in rows_by_case
        if relevance_of(inputs.by_case_a[case_id]) != relevance_of(inputs.by_case_b[case_id])
    )
    reported = inputs.report.get("disagreements")
    reported_ids = sorted(
        entry["case_id"]
        for entry in (reported if isinstance(reported, list) else [])
        if isinstance(entry, dict) and isinstance(entry.get("case_id"), str)
    )
    if reported_ids != disagreements or inputs.report.get("n") != len(inputs.rows):
        errors.append(
            "IAA report is stale: it does not match the sidecars/gold row set "
            "(recompute with scripts/compute_iaa.py)"
        )


def _merge_all(inputs: _MergeInputs) -> tuple[list[tuple[str, list[str]]], int, int]:
    """Apply the merge rules to every gold row; return changes and tier counts."""
    row_changes: list[tuple[str, list[str]]] = []
    release_count = 0
    candidate_count = 0
    for row in inputs.rows:
        case_id = row["case_id"]
        changes = merge_row(row, inputs.by_case_a[case_id], inputs.by_case_b[case_id])
        row_changes.append((case_id, changes))
        if row["provenance"]["quality_tier"] == "release":
            release_count += 1
        else:
            candidate_count += 1
    return row_changes, release_count, candidate_count


def _print_plan(
    inputs: _MergeInputs,
    row_changes: list[tuple[str, list[str]]],
    release_count: int,
    candidate_count: int,
) -> None:
    """Print the per-row change summary."""
    sys.stdout.write(
        f"{len(inputs.rows)} rows; {release_count} release, "
        f"{candidate_count} candidate (pending_sme)\n"
    )
    for case_id, changes in row_changes:
        if changes:
            sys.stdout.write(f"  {case_id}: " + "; ".join(changes) + "\n")


def _serialize(metadata_line: str | None, rows: list[dict[str, Any]]) -> str:
    """Re-emit the gold JSONL: original metadata line plus merged rows in order."""
    header = metadata_line if metadata_line is not None else ""
    return header + "\n" + "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def run(argv: list[str] | None = None) -> int:
    """Run the merge; return the process exit code."""
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    out_path = args.out if args.out is not None else args.gold

    errors: list[str] = []
    inputs = _load_inputs(args, errors)
    _check_consistency(inputs, args, errors)
    if errors:
        for error in errors:
            sys.stderr.write(f"error: {error}\n")
        return _EXIT_STRUCTURAL

    if inputs.report["gate"].get("passed") is not True:
        sys.stderr.write(
            "error: IAA gate has not passed; refusing to merge "
            "(run scripts/compute_iaa.py and clear the gate first)\n"
        )
        return _EXIT_GATE_FAILED

    row_changes, release_count, candidate_count = _merge_all(inputs)
    if not args.apply:
        sys.stdout.write("dry-run: ")
        _print_plan(inputs, row_changes, release_count, candidate_count)
        sys.stdout.write("no files written; re-run with --apply to write\n")
        return _EXIT_OK

    write_errors = _write_and_validate(
        _serialize(inputs.metadata_line, inputs.rows), out_path, repo_root
    )
    if write_errors:
        for error in write_errors:
            sys.stderr.write(f"error: {error}\n")
        return _EXIT_STRUCTURAL
    sys.stdout.write("applied: ")
    _print_plan(inputs, row_changes, release_count, candidate_count)
    sys.stdout.write(f"wrote {out_path}; validation passed\n")
    return _EXIT_OK


def main() -> None:
    """CLI entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
