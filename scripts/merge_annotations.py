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
untouched and the script exits 1.  The validator looks up a manifest named
after the file it loads, so the real gold file's sibling manifest is copied
next to the randomly named temp file before validation (and unlinked with it
on every exit path); a v2 gold set without a sibling manifest is refused
instead of being validated manifest-less.  When ``--out`` names a different
path, the same manifest is kept next to the output as its sibling
``<output_stem>.manifest.json`` so the written set can be loaded and
re-validated; an in-place apply keeps the gold's existing sibling untouched.
Without ``--apply`` it is a dry run that prints what would change.  The
metadata line 1 and row order are preserved.

The IAA report is never trusted as-is: the gate is recomputed from the two
sidecars' labels (Cohen's kappa + raw agreement, implemented inline and
cross-checked against the report's ``gate.thresholds``) and the merge refuses
when the recomputed gate fails, the disagreement set differs, or the reported
kappa no longer matches the sidecars.

Rows an SME has already decided (``adjudication.status == "adjudicated"`` or
``"waived_with_reason"`` — a waiver satisfies the same release gate) are
preserved verbatim — only their annotation records are refreshed — so a
re-run after SME review cannot clobber the decision; pass ``--reopen`` to
downgrade those rows to ``pending_sme`` and let the normal rules rewrite them.

Usage:
    uv run python scripts/merge_annotations.py \
        --gold evals/golden/legal_retrieval_v2.jsonl \
        --a evals/annotations/wave3-annotator-a.jsonl \
        --b evals/annotations/wave3-annotator-b.jsonl \
        --iaa evals/reports/iaa_wave3.json [--apply] [--reopen]

Exit codes: 0 success (or clean dry run), 1 structural error (including a
failed post-merge validation), 2 the IAA gate has not passed (reported or
recomputed).

Dependency-free: standard library only, mirroring ``compute_iaa.py``.  The
sidecar validation helpers are imported from ``compute_iaa`` so both tools
enforce one identical contract; the fallback import keeps the script runnable
when executed directly as a file (``python scripts/merge_annotations.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # noqa: S404 — invokes the project's own validate module only
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Any

try:
    from scripts.compute_iaa import (
        annotator_id_overlap,
        load_jsonl_objects,
        relevance_of,
        validate_sidecar,
    )
except ImportError:  # pragma: no cover - direct script execution puts scripts/ on sys.path
    from compute_iaa import (
        annotator_id_overlap,
        load_jsonl_objects,
        relevance_of,
        validate_sidecar,
    )


_EXIT_OK = 0
_EXIT_STRUCTURAL = 1
_EXIT_GATE_FAILED = 2

_GOLD_SET_V2_VERSION = "legal_retrieval_v2"
_KAPPA_TOLERANCE = 1e-9
_MIN_N_FOR_KAPPA_GATE = 5
_DEFAULT_MIN_KAPPA = 0.70
_DEFAULT_MIN_AGREEMENT = 0.80
_RELEASE_RATIONALE = (
    "dual engineer agreement on deterministic row; machine-verified deterministic expectations"
)
_DISAGREEMENT_RATIONALE = (
    "dual engineer disagreement on retrieval_relevance; SME adjudication required"
)
_NON_DETERMINISTIC_RATIONALE = "row is not deterministically graded; SME adjudication pending"
# Adjudication statuses that mean an SME already made the call: a rerun must
# preserve them verbatim and --reopen resets them to pending_sme.
_HUMAN_DECIDED_STATUSES = frozenset({"adjudicated", "waived_with_reason"})


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


def _percent_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    """Return the raw proportion of identical label pairs (0.0 when empty).

    Mirrors ``compute_iaa.percent_agreement``; the duplication is deliberate so
    the merge cross-checks the report against an independent recomputation
    rather than a shared code path.
    """
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return 0.0
    return sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Return unweighted Cohen's kappa over paired relevance labels.

    Mirrors ``compute_iaa.cohen_kappa`` (stdlib-only): ``p_o`` is the proportion
    of identical label pairs, ``p_e`` the sum over labels of the product of the
    two annotators' marginal proportions, with the same degenerate-case
    contract when the expected agreement reaches 1.0.
    """
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return 0.0
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    expected = 0.0
    for label in (
        "relevant",
        "partially_relevant",
        "irrelevant",
        "not_applicable",
    ):
        expected += (labels_a.count(label) / n) * (labels_b.count(label) / n)
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def recompute_iaa(
    case_ids: list[str],
    by_case_a: dict[str, dict[str, Any]],
    by_case_b: dict[str, dict[str, Any]],
) -> tuple[float, float, list[dict[str, str]]]:
    """Recompute (kappa, raw agreement, disagreement entries) from the sidecars.

    ``case_ids`` must be sorted so the disagreement entries are comparable to
    the report's; the formula mirrors ``compute_iaa.py`` but is implemented
    here so a tampered or stale report cannot vouch for itself.
    """
    labels_a = [relevance_of(by_case_a[case_id]) for case_id in case_ids]
    labels_b = [relevance_of(by_case_b[case_id]) for case_id in case_ids]
    disagreements = [
        {"case_id": case_id, "a": label_a, "b": label_b}
        for case_id, label_a, label_b in zip(case_ids, labels_a, labels_b, strict=True)
        if label_a != label_b
    ]
    return _cohen_kappa(labels_a, labels_b), _percent_agreement(labels_a, labels_b), disagreements


def _recheck_iaa_gate(inputs: _MergeInputs) -> list[str]:
    """Recompute the IAA gate from the sidecars and cross-check the report.

    Refuses (exit 2) when the recomputed gate fails under the stricter of the
    report's ``gate.thresholds`` and the module defaults (both are lower
    bounds, so stricter means higher — a report may not weaken the gate),
    when the recomputed disagreement set differs from the report's, or when
    the recomputed kappa differs from the report's value by more than 1e-9 —
    a stale or tampered report in either case.
    """
    case_ids = sorted({row["case_id"] for row in inputs.rows})
    kappa, raw, disagreements = recompute_iaa(case_ids, inputs.by_case_a, inputs.by_case_b)
    report = inputs.report
    errors: list[str] = []
    reported = report.get("disagreements")
    reported_entries = [
        entry for entry in (reported if isinstance(reported, list) else []) if isinstance(entry, dict)
    ]
    if reported_entries != disagreements:
        errors.append(
            "IAA report is stale: recomputed disagreement set does not match the report "
            "(recompute with scripts/compute_iaa.py)"
        )
    reported_kappa = report.get("kappa")
    if (
        isinstance(reported_kappa, bool)
        or not isinstance(reported_kappa, (int, float))
        or abs(float(reported_kappa) - kappa) > _KAPPA_TOLERANCE
    ):
        errors.append(
            f"IAA report is stale: reported kappa {reported_kappa!r} differs from the "
            f"recomputed kappa {kappa:.6f} (recompute with scripts/compute_iaa.py)"
        )
    thresholds = report["gate"].get("thresholds")
    threshold_map = thresholds if isinstance(thresholds, dict) else {}
    # Both thresholds are lower bounds, so the effective gate is the stricter
    # (higher) of the report's values and the module defaults: a report can
    # tighten the gate but can never weaken it below 0.70/0.80.
    min_kappa = max(float(threshold_map.get("min_kappa", _DEFAULT_MIN_KAPPA)), _DEFAULT_MIN_KAPPA)
    min_agreement = max(
        float(threshold_map.get("min_agreement", _DEFAULT_MIN_AGREEMENT)), _DEFAULT_MIN_AGREEMENT
    )
    n = len(case_ids)
    gate_passed = (
        raw >= 1.0
        if n < _MIN_N_FOR_KAPPA_GATE
        else kappa >= min_kappa and raw >= min_agreement
    )
    if not gate_passed:
        errors.append(
            f"recomputed IAA gate fails: n={n} raw_agreement={raw:.4f} kappa={kappa:.4f} "
            f"against min_kappa={min_kappa} min_agreement={min_agreement}"
        )
    return errors


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
    """Build the pending-SME adjudication record without ever naming an adjudicator.

    A record that is already ``pending_sme`` is preserved as-is (only a
    missing rationale is filled in), which is what makes repeated merges
    byte-identical.  Converting any other existing record back to
    ``pending_sme`` discards its stale decision fields (``method``,
    ``decided_at``, a named ``adjudicator`` and the old ``rationale``) and
    writes a fresh rationale describing the reopen.
    """
    adjudication = dict(existing) if isinstance(existing, dict) else {}
    if adjudication.get("status") == "pending_sme":
        adjudication["status"] = "pending_sme"
        adjudication.setdefault("rationale", rationale)
        return adjudication
    for stale_field in ("method", "decided_at", "adjudicator", "rationale"):
        adjudication.pop(stale_field, None)
    adjudication["status"] = "pending_sme"
    adjudication["rationale"] = f"reopened by merge run {_utc_now()}"
    return adjudication


def _annotation_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Strip the case binding key; attached records must match AnnotationRecord."""
    return {key: value for key, value in record.items() if key != "case_id"}


def merge_row(
    row: dict[str, Any],
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    *,
    reopen: bool = False,
) -> list[str]:
    """Fold both annotation records into one gold row; return change descriptions.

    A row an SME has already decided (``adjudication.status == "adjudicated"``
    or ``"waived_with_reason"``) is frozen: only its annotation records are
    refreshed and its adjudication record, ``human_review_status``, and
    ``quality_tier`` are left verbatim, so re-running a merge after SME review
    cannot clobber the decision.  ``reopen=True`` downgrades such rows to
    ``pending_sme`` first (reporting which status was reopened) and lets the
    normal rules rewrite them.
    """
    changes: list[str] = []
    provenance: dict[str, Any] = row["provenance"]
    annotator_ids = {record_a["annotator_id"], record_b["annotator_id"]}

    existing_annotations = provenance.get("annotations")
    prior_annotations = existing_annotations if isinstance(existing_annotations, list) else []
    retained = [
        record
        for record in prior_annotations
        if isinstance(record, dict) and record.get("annotator_id") not in annotator_ids
    ]
    merged_annotations = [
        *retained,
        _annotation_payload(record_a),
        _annotation_payload(record_b),
    ]
    if merged_annotations != prior_annotations:
        changes.append(f"annotations: {len(prior_annotations)} -> {len(merged_annotations)}")
    provenance["annotations"] = merged_annotations

    existing_adjudication: object = provenance.get("adjudication")
    pre_decided_status: str | None = None
    if isinstance(existing_adjudication, dict) and existing_adjudication.get("status") in (
        _HUMAN_DECIDED_STATUSES
    ):
        pre_decided_status = str(existing_adjudication["status"])
    if pre_decided_status is not None and reopen:
        provenance["human_review_status"] = "pending"
        provenance["adjudication"] = {"status": "pending_sme"}
        changes.append(f"adjudication: reopened ({pre_decided_status} -> pending_sme)")
        pre_decided_status = None
    if pre_decided_status is not None:
        return changes

    if provenance.get("human_review_status") == "pending":
        provenance["human_review_status"] = "engineer_reviewed"
        changes.append("human_review_status: pending -> engineer_reviewed")

    agreed = relevance_of(record_a) == relevance_of(record_b)
    deterministic = row.get("grader") == "deterministic"
    prior_tier = provenance.get("quality_tier")
    existing_adjudication = provenance.get("adjudication")
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


def _manifest_path_for(golden_set_path: Path) -> Path:
    """Return the provenance manifest path paired with a golden-set JSONL.

    Mirrors ``app.ai.evaluation.schema.manifest_path_for`` locally: the merge
    tool stays stdlib-only, and this path convention is exactly what the
    validator's ``load_validated_golden_set`` uses to find the manifest.
    """
    return golden_set_path.with_name(f"{golden_set_path.stem}.manifest.json")


def _temp_file_version(temp_path: Path) -> str | None:
    """Return the golden-set version declared by metadata line 1 of a file."""
    try:
        first_line = temp_path.read_text(encoding="utf-8").split("\n", 1)[0]
        metadata = json.loads(first_line)
    except (OSError, json.JSONDecodeError):
        return None
    return metadata.get("version") if isinstance(metadata, dict) else None


def _write_and_validate(
    payload: str, out_path: Path, repo_root: Path, gold_path: Path
) -> list[str]:
    """Write to a temp file, validate it, and move into place; return error strings.

    The temp file never outlives this call: every exit path either unlinks it
    explicitly or moves it onto ``out_path``.  Any exception raised inside the
    write/validate window (including ``KeyboardInterrupt`` and ``SystemExit``)
    unlinks the temp file before propagating, so a crashed merge can never
    leave ``<gold>.merge-*.tmp`` litter behind.

    Validation consults the REAL manifest: the validator resolves the manifest
    from the stem of the file it loads, so a randomly named temp file would
    otherwise be validated manifest-less — silently skipping the v2
    cross-reference and checksum audit that CI will run on the real file.  The
    gold file's sibling manifest is therefore copied next to the temp file
    before validation, and the temp manifest is unlinked on every exit path
    alongside the temp gold file.  When the output path's stem differs from
    the gold's, the manifest is kept next to the output instead (as
    ``<output_stem>.manifest.json``) so the written set can be loaded again;
    when the output replaces the gold in place, its existing sibling manifest
    stays and is not duplicated.  A v2 gold set without a sibling manifest is
    refused outright instead of being validated manifest-less.
    """
    handle_fd, temp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=f"{out_path.name}.merge-", suffix=".tmp"
    )
    os.close(handle_fd)
    temp_path = Path(temp_name)
    temp_manifest = _manifest_path_for(temp_path)
    real_manifest = _manifest_path_for(gold_path)
    out_manifest = _manifest_path_for(out_path)
    try:
        temp_path.write_text(payload, encoding="utf-8")
        if _temp_file_version(temp_path) == _GOLD_SET_V2_VERSION and not real_manifest.is_file():
            temp_path.unlink(missing_ok=True)
            temp_manifest.unlink(missing_ok=True)
            return [
                f"{gold_path}: v2 gold set ({_GOLD_SET_V2_VERSION}) requires a sibling "
                f"provenance manifest {real_manifest.name}; refusing to merge without it"
            ]
        if real_manifest.is_file():
            shutil.copyfile(real_manifest, temp_manifest)
        passed, message = _run_gold_validation(temp_path, repo_root)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
        return [f"write failed: {exc}"]
    except BaseException:
        temp_path.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
        raise
    if not passed:
        temp_path.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
        return [f"merged gold failed validation; rolled back: {message}"]
    # mkstemp creates the file 0o600; normalize to the plain-file default so
    # the replaced gold set keeps world-readable permissions.
    temp_path.chmod(0o644)
    temp_path.replace(out_path)
    if out_manifest != real_manifest and real_manifest.is_file():
        # A different output path needs its own sibling manifest, or the
        # written set cannot be loaded again; in place, the gold's existing
        # sibling remains and the redundant temp copy is dropped.
        temp_manifest.replace(out_manifest)
    else:
        temp_manifest.unlink(missing_ok=True)
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
        help=(
            "output path (default: the --gold path, i.e. in place); a different path also "
            "receives a copy of the gold's provenance manifest as its sibling "
            "<output_stem>.manifest.json"
        ),
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the merged gold set (default: dry run)"
    )
    parser.add_argument(
        "--reopen",
        action="store_true",
        help=(
            "downgrade rows an SME has already decided (adjudicated or waived_with_reason) back "
            "to pending_sme and let the normal merge rules rewrite them (default: preserve "
            "decided rows verbatim, refreshing only their annotation records)"
        ),
    )
    return parser.parse_args(argv)


def _load_inputs(args: argparse.Namespace, errors: list[str]) -> _MergeInputs:
    """Load gold rows, both sidecars, and the IAA gate report."""
    metadata_line, rows = load_gold_rows(args.gold, errors)
    records_a = load_jsonl_objects(args.a, errors)
    records_b = load_jsonl_objects(args.b, errors)
    by_case_a = validate_sidecar(args.a, records_a, errors)
    by_case_b = validate_sidecar(args.b, records_b, errors)
    overlap = annotator_id_overlap(by_case_a, by_case_b)
    if overlap:
        errors.append(
            "the same annotator_id appears on both sides: "
            f"{', '.join(overlap)}; a paired merge needs two distinct annotators"
        )
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


def _merge_all(
    inputs: _MergeInputs, *, reopen: bool = False
) -> tuple[list[tuple[str, list[str]]], int, int]:
    """Apply the merge rules to every gold row; return changes and tier counts."""
    row_changes: list[tuple[str, list[str]]] = []
    release_count = 0
    candidate_count = 0
    for row in inputs.rows:
        case_id = row["case_id"]
        changes = merge_row(row, inputs.by_case_a[case_id], inputs.by_case_b[case_id], reopen=reopen)
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

    gate_errors = _recheck_iaa_gate(inputs)
    if gate_errors:
        for error in gate_errors:
            sys.stderr.write(f"error: {error}\n")
        return _EXIT_GATE_FAILED

    row_changes, release_count, candidate_count = _merge_all(inputs, reopen=args.reopen)
    if not args.apply:
        sys.stdout.write("dry-run: ")
        _print_plan(inputs, row_changes, release_count, candidate_count)
        sys.stdout.write("no files written; re-run with --apply to write\n")
        return _EXIT_OK

    write_errors = _write_and_validate(
        _serialize(inputs.metadata_line, inputs.rows), out_path, repo_root, args.gold
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
