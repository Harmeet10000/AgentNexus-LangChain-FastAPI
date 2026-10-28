"""Inter-annotator agreement for the wave-3 dual engineer relevance pass.

Reads two blind annotation sidecars (one ``AnnotationRecord``-shaped JSON
object per line, as produced by the two independent engineer annotators),
validates them against the ``evals/annotations/questions.jsonl`` case universe
and the ``retrieval_relevance`` vocabulary, and computes Cohen's kappa (the
standard unweighted observed/expected formula) plus raw percent agreement and
a full confusion matrix over the four relevance labels.

Because annotators are blind to ``case_type``, stratification uses only
annotator-visible signals: the paired confidence stratum (a pair is only as
strong as its least confident annotator) and the coarse agreement pattern
(exact / adjacent / distant on the relevance ordinal scale).

Gate (exit codes: 0 pass, 1 structural error, 2 gate failure): for n >= 5
dual-annotated cases, kappa >= --min-kappa (0.70) AND raw agreement >=
--min-agreement (0.80); for n < 5, 100% agreement is required.

Usage:
    uv run python scripts/compute_iaa.py \
        --a evals/annotations/wave3-annotator-a.jsonl \
        --b evals/annotations/wave3-annotator-b.jsonl \
        [--out evals/reports/iaa_wave3.json]

Dependency-free: standard library only, so it can run in CI before services
exist.  The kappa math is duplicated from
``app.ai.evaluation.metrics.cohen_kappa_unweighted`` (see the test suite for a
cross-check that both implementations agree); the duplication is deliberate so
this guard stays importable without the application package.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

_EXIT_OK = 0
_EXIT_STRUCTURAL = 1
_EXIT_GATE_FAILED = 2

RELEVANCE_LABELS: tuple[str, ...] = (
    "relevant",
    "partially_relevant",
    "irrelevant",
    "not_applicable",
)
_CONFIDENCE_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}
_CONFIDENCE_STRATA: tuple[str, ...] = ("high", "medium", "low")
_ENGINEER_ROLES: frozenset[str] = frozenset({"engineer"})

_PATTERN_EXACT = "exact_agreement"
_PATTERN_ADJACENT = "adjacent_disagreement"
_PATTERN_DISTANT = "distant_disagreement"
_AGREEMENT_PATTERNS: tuple[str, ...] = (_PATTERN_EXACT, _PATTERN_ADJACENT, _PATTERN_DISTANT)

_DEFAULT_QUESTIONS = Path("evals/annotations/questions.jsonl")
_DEFAULT_OUT = Path("evals/reports/iaa_wave3.json")
_MIN_KAPPA_DEFAULT = 0.70
_MIN_AGREEMENT_DEFAULT = 0.80
_MIN_N_FOR_KAPPA_GATE = 5

_GATE_RULE = (
    "n >= 5: kappa >= min_kappa and raw_agreement >= min_agreement; n < 5: raw_agreement == 1.0"
)


def kappa_from_margins(observed_agreement: float, expected_agreement: float) -> float:
    """Return Cohen's kappa from pre-computed observed/expected agreement.

    ``(p_o - p_e) / (1 - p_e)`` with the degenerate-case contract: when the
    expected agreement reaches 1.0 the denominator vanishes, so a totally
    agreeing sample is defined as kappa 1.0 and any non-total agreement on a
    single-category sample is defined as kappa 0.0.
    """
    if expected_agreement >= 1.0:
        return 1.0 if observed_agreement >= 1.0 else 0.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Return unweighted Cohen's kappa over paired relevance labels.

    Uses the exact observed/expected formula: ``p_o`` is the proportion of
    identical label pairs; ``p_e`` is the sum over labels of the product of the
    two annotators' marginal proportions.
    """
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return 0.0
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    expected = 0.0
    for label in RELEVANCE_LABELS:
        expected += (labels_a.count(label) / n) * (labels_b.count(label) / n)
    return kappa_from_margins(observed, expected)


def percent_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    """Return the raw proportion of identical label pairs (0.0 when empty)."""
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return 0.0
    return sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n


def load_jsonl_objects(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    """Parse one JSON object per line; whitespace-only lines are skipped."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{path}: unreadable ({exc})")
        return []
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_no}: malformed JSON ({exc.msg})")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"{path}:{line_no}: expected a JSON object")
            continue
        records.append(parsed)
    return records


def _validate_question_universe(path: Path, errors: list[str]) -> set[str]:
    """Return the case_id universe declared by the blind question file."""
    records = load_jsonl_objects(path, errors)
    universe: set[str] = set()
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{path}: question row without case_id")
            continue
        universe.add(case_id)
    if not universe:
        errors.append(f"{path}: question universe is empty")
    return universe


def validate_sidecar(
    path: Path, records: list[dict[str, Any]], errors: list[str]
) -> dict[str, dict[str, Any]]:
    """Validate one sidecar and return its case_id -> record map.

    Checks: single constant non-empty ``annotator_id``, engineer role, the
    retrieval-relevance vocabulary, a confidence stratum, and unique case ids.
    """
    by_case: dict[str, dict[str, Any]] = {}
    annotator_ids: set[str] = set()
    for line_no, record in enumerate(records, start=1):
        where = f"{path}:{line_no}"
        case_id = record.get("case_id")
        annotator_id = record.get("annotator_id")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{where}: missing case_id")
            continue
        if not isinstance(annotator_id, str) or not annotator_id:
            errors.append(f"{where} ({case_id}): missing annotator_id")
            continue
        annotator_ids.add(annotator_id)
        if record.get("role") not in _ENGINEER_ROLES:
            errors.append(
                f"{where} ({case_id}): role must be 'engineer' for the relevance dimension"
            )
        if record.get("confidence") not in _CONFIDENCE_ORDER:
            errors.append(f"{where} ({case_id}): confidence must be high/medium/low")
        labels = record.get("labels")
        relevance = labels.get("retrieval_relevance") if isinstance(labels, dict) else None
        if relevance not in RELEVANCE_LABELS:
            errors.append(
                f"{where} ({case_id}): labels.retrieval_relevance must be one of "
                f"{', '.join(RELEVANCE_LABELS)}"
            )
        if case_id in by_case:
            errors.append(f"{path}: duplicate case_id {case_id!r}")
        by_case[case_id] = record
    if len(annotator_ids) > 1:
        errors.append(f"{path}: mixed annotator_ids {sorted(annotator_ids)} in one sidecar")
    return by_case


def relevance_of(record: dict[str, Any]) -> str:
    """Return the retrieval_relevance label of one annotation record."""
    labels = record.get("labels")
    value = labels.get("retrieval_relevance") if isinstance(labels, dict) else None
    return value if isinstance(value, str) else ""


def annotator_id_overlap(
    by_case_a: dict[str, dict[str, Any]],
    by_case_b: dict[str, dict[str, Any]],
) -> list[str]:
    """Return sorted annotator_ids present in BOTH sidecars (empty when distinct).

    Shared by ``compute_iaa.py`` and ``merge_annotations.py`` so both tools
    enforce one identical contract: a paired computation over the same
    annotator twice would report perfect agreement and promote rows.
    """
    annotators_a = {record["annotator_id"] for record in by_case_a.values()}
    annotators_b = {record["annotator_id"] for record in by_case_b.values()}
    return sorted(annotators_a & annotators_b)


def _confidence_of(record: dict[str, Any]) -> str:
    value = record.get("confidence")
    return value if isinstance(value, str) and value in _CONFIDENCE_ORDER else "low"


def pair_confidence(record_a: dict[str, Any], record_b: dict[str, Any]) -> str:
    """Return the paired stratum: the least confident of the two annotators."""
    confidence_a = _confidence_of(record_a)
    confidence_b = _confidence_of(record_b)
    if _CONFIDENCE_ORDER[confidence_a] >= _CONFIDENCE_ORDER[confidence_b]:
        return confidence_a
    return confidence_b


def agreement_pattern(label_a: str, label_b: str) -> str:
    """Classify a label pair on the relevance ordinal scale."""
    if label_a == label_b:
        return _PATTERN_EXACT
    distance = abs(RELEVANCE_LABELS.index(label_a) - RELEVANCE_LABELS.index(label_b))
    return _PATTERN_ADJACENT if distance == 1 else _PATTERN_DISTANT


def _stratum_stats(labels_a: list[str], labels_b: list[str]) -> dict[str, Any]:
    """Kappa + raw agreement for one stratum (nulls when the stratum is empty)."""
    if not labels_a:
        return {"n": 0, "raw_agreement": None, "kappa": None}
    return {
        "n": len(labels_a),
        "raw_agreement": percent_agreement(labels_a, labels_b),
        "kappa": cohen_kappa(labels_a, labels_b),
    }


def _confusion_matrix(labels_a: list[str], labels_b: list[str]) -> dict[str, dict[str, int]]:
    """Rows = annotator A, columns = annotator B; every cell is present."""
    matrix = {a: dict.fromkeys(RELEVANCE_LABELS, 0) for a in RELEVANCE_LABELS}
    for a, b in zip(labels_a, labels_b, strict=True):
        matrix[a][b] += 1
    return matrix


def _evaluate_gate(
    n: int, raw_agreement: float, kappa: float, min_kappa: float, min_agreement: float
) -> bool:
    """Apply the exact gate: kappa+agreement thresholds for n >= 5, else 100%."""
    if n < _MIN_N_FOR_KAPPA_GATE:
        # percent_agreement is bounded by 1.0, so >= 1.0 means exactly total.
        return raw_agreement >= 1.0
    return kappa >= min_kappa and raw_agreement >= min_agreement


def _load_inputs(
    args: argparse.Namespace, questions_path: Path, errors: list[str]
) -> tuple[
    set[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Load and validate the question universe plus both sidecars."""
    universe = _validate_question_universe(questions_path, errors)
    records_a = load_jsonl_objects(args.a, errors)
    records_b = load_jsonl_objects(args.b, errors)
    by_case_a = validate_sidecar(args.a, records_a, errors)
    by_case_b = validate_sidecar(args.b, records_b, errors)
    overlap = annotator_id_overlap(by_case_a, by_case_b)
    if overlap:
        errors.append(
            "the same annotator_id appears on both sides: "
            f"{', '.join(overlap)}; a paired computation needs two distinct annotators"
        )
    if universe:
        errors.extend(
            f"question-universe case {missing!r} has no annotation in {args.a}"
            for missing in sorted(universe - set(by_case_a))
        )
        errors.extend(
            f"question-universe case {missing!r} has no annotation in {args.b}"
            for missing in sorted(universe - set(by_case_b))
        )
        errors.extend(
            f"{case_id!r}: case_id not in the question universe"
            for case_id in sorted((set(by_case_a) | set(by_case_b)) - universe)
        )
    return universe, records_a, records_b, by_case_a, by_case_b


def _collect_strata(
    case_ids: list[str],
    by_case_a: dict[str, dict[str, Any]],
    by_case_b: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, list[str]],
    list[dict[str, str]],
]:
    """Bucket label pairs by confidence stratum and agreement pattern."""
    confidence_a: dict[str, list[str]] = {name: [] for name in _CONFIDENCE_STRATA}
    confidence_b: dict[str, list[str]] = {name: [] for name in _CONFIDENCE_STRATA}
    pattern_a: dict[str, list[str]] = {name: [] for name in _AGREEMENT_PATTERNS}
    pattern_b: dict[str, list[str]] = {name: [] for name in _AGREEMENT_PATTERNS}
    disagreements: list[dict[str, str]] = []
    for case_id in case_ids:
        label_a = relevance_of(by_case_a[case_id])
        label_b = relevance_of(by_case_b[case_id])
        stratum = pair_confidence(by_case_a[case_id], by_case_b[case_id])
        confidence_a[stratum].append(label_a)
        confidence_b[stratum].append(label_b)
        pattern = agreement_pattern(label_a, label_b)
        pattern_a[pattern].append(label_a)
        pattern_b[pattern].append(label_b)
        if label_a != label_b:
            disagreements.append({"case_id": case_id, "a": label_a, "b": label_b})
    return confidence_a, confidence_b, pattern_a, pattern_b, disagreements


def _build_report(
    args: argparse.Namespace,
    *,
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    case_ids: list[str],
    labels_a: list[str],
    labels_b: list[str],
    by_case_a: dict[str, dict[str, Any]],
    by_case_b: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the emitted IAA report object."""
    confidence_a, confidence_b, pattern_a, pattern_b, disagreements = _collect_strata(
        case_ids, by_case_a, by_case_b
    )
    n = len(case_ids)
    raw = percent_agreement(labels_a, labels_b)
    kappa = cohen_kappa(labels_a, labels_b)
    return {
        "annotators": [records_a[0]["annotator_id"], records_b[0]["annotator_id"]],
        "n": n,
        "raw_agreement": raw,
        "kappa": kappa,
        "confusion_matrix": _confusion_matrix(labels_a, labels_b),
        "disagreements": disagreements,
        "per_confidence": {
            name: _stratum_stats(confidence_a[name], confidence_b[name])
            for name in _CONFIDENCE_STRATA
        },
        "per_agreement_pattern": {
            name: _stratum_stats(pattern_a[name], pattern_b[name]) for name in _AGREEMENT_PATTERNS
        },
        "gate": {
            "passed": _evaluate_gate(n, raw, kappa, args.min_kappa, args.min_agreement),
            "thresholds": {
                "min_kappa": args.min_kappa,
                "min_agreement": args.min_agreement,
                "rule": _GATE_RULE,
            },
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the question-universe and report paths (repo-root defaults)."""
    repo_root = Path(__file__).resolve().parent.parent
    questions_path = (
        args.questions if args.questions is not None else repo_root / _DEFAULT_QUESTIONS
    )
    out_path = args.out if args.out is not None else repo_root / _DEFAULT_OUT
    return questions_path, out_path


def run(argv: list[str] | None = None) -> int:
    """Run the IAA computation; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path, help="annotator A sidecar JSONL")
    parser.add_argument("--b", required=True, type=Path, help="annotator B sidecar JSONL")
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help=f"blind question universe JSONL (default: {_DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help=f"report destination (default: {_DEFAULT_OUT})"
    )
    parser.add_argument("--min-kappa", type=float, default=_MIN_KAPPA_DEFAULT)
    parser.add_argument("--min-agreement", type=float, default=_MIN_AGREEMENT_DEFAULT)
    args = parser.parse_args(argv)

    questions_path, out_path = _resolve_paths(args)
    errors: list[str] = []
    universe, records_a, records_b, by_case_a, by_case_b = _load_inputs(
        args, questions_path, errors
    )
    if errors:
        for error in errors:
            sys.stderr.write(f"error: {error}\n")
        return _EXIT_STRUCTURAL

    case_ids = sorted(universe)
    labels_a = [relevance_of(by_case_a[case_id]) for case_id in case_ids]
    labels_b = [relevance_of(by_case_b[case_id]) for case_id in case_ids]
    report = _build_report(
        args,
        records_a=records_a,
        records_b=records_b,
        case_ids=case_ids,
        labels_a=labels_a,
        labels_b=labels_b,
        by_case_a=by_case_a,
        by_case_b=by_case_b,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    passed = report["gate"]["passed"]
    sys.stdout.write(
        f"IAA {'PASS' if passed else 'FAIL'}: n={report['n']} "
        f"raw_agreement={report['raw_agreement']:.4f} kappa={report['kappa']:.4f} "
        f"disagreements={len(report['disagreements'])} report={out_path}\n"
    )
    return _EXIT_OK if passed else _EXIT_GATE_FAILED


def main() -> None:
    """CLI entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
