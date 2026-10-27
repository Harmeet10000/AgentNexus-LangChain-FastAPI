"""Unit tests for the wave-3 IAA tooling: compute_iaa.py and merge_annotations.py.

All fixtures are authored locally with the real schema models; the parallel
annotator sidecars are never read.  Offline: the only subprocess is the
project's own ``app.ai.evaluation.validate`` module.
"""

import asyncio
import json
import subprocess  # noqa: S404 — spawns the project's own compute_iaa script only
import sys
from pathlib import Path
from typing import Any

import pytest
import scripts.compute_iaa as iaa
from returns.result import Success
from scripts.compute_iaa import kappa_from_margins as script_kappa_from_margins
from scripts.compute_iaa import run as run_iaa
from scripts.merge_annotations import run as run_merge

from app.ai.evaluation.metrics import cohen_kappa_unweighted, kappa_from_margins
from app.ai.evaluation.schema import (
    AdjudicationRecord,
    AnnotationRecord,
    AnnotatorRoles,
    GoldenItemProvenance,
    GoldenQuery,
    load_validated_golden_set,
)

ANNOTATOR_A = "wave3-annotator-a"
ANNOTATOR_B = "wave3-annotator-b"
RELEASE_RATIONALE = (
    "dual engineer agreement on deterministic row; machine-verified deterministic expectations"
)

# (case, label_a, label_b, grader): n=12, raw=10/12, p_e=43/144, kappa~0.7624
# so the default gate (kappa >= 0.70, agreement >= 0.80) genuinely passes while
# still containing two disagreements, two sme-graded rows and eight releases.
WAVE3_FIXTURE: tuple[tuple[str, str, str, str], ...] = (
    ("direct_answer-unit-001", "relevant", "relevant", "deterministic"),
    ("direct_answer-unit-002", "relevant", "relevant", "deterministic"),
    ("direct_answer-unit-003", "relevant", "relevant", "deterministic"),
    ("direct_answer-unit-004", "relevant", "relevant", "deterministic"),
    ("direct_answer-unit-005", "partially_relevant", "partially_relevant", "sme"),
    ("direct_answer-unit-006", "partially_relevant", "partially_relevant", "sme"),
    ("direct_answer-unit-007", "partially_relevant", "partially_relevant", "deterministic"),
    ("direct_answer-unit-008", "irrelevant", "irrelevant", "deterministic"),
    ("direct_answer-unit-009", "irrelevant", "irrelevant", "deterministic"),
    ("direct_answer-unit-010", "not_applicable", "not_applicable", "deterministic"),
    ("direct_answer-unit-011", "relevant", "partially_relevant", "deterministic"),
    ("direct_answer-unit-012", "partially_relevant", "irrelevant", "deterministic"),
)


def annotation(
    case_id: str,
    relevance: str,
    *,
    annotator: str = ANNOTATOR_A,
    confidence: str = "high",
) -> dict[str, Any]:
    record = AnnotationRecord(
        annotator_id=annotator,
        role="engineer",
        timestamp_utc="2026-10-28T00:00:00Z",
        labels={"retrieval_relevance": relevance},
        confidence=confidence,  # type: ignore[arg-type]
        ambiguity_notes="",
        ocr_quality_impact="none",
        harness_version="wave3-test",
    )
    # sidecar line = the case binding plus the AnnotationRecord payload
    return {"case_id": case_id, **record.model_dump(mode="json")}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_questions(path: Path, case_ids: list[str]) -> None:
    write_jsonl(path, [{"case_id": case_id} for case_id in case_ids])


def gold_row(case_id: str, *, grader: str = "deterministic") -> dict[str, Any]:
    adjudication_status = "pending_sme" if grader != "deterministic" else "not_required"
    query = GoldenQuery(
        query="fixture query",
        expected_chunk_ids=["chunk-1"],
        document_kind="contracts",
        difficulty="easy",
        notes="fixture",
        awaiting_sme_expansion=False,
        case_type="direct_answer",
        retrieval_relevance="relevant",
        claim_entailment="not_applicable",
        case_id=case_id,
        grader=grader,  # type: ignore[arg-type]
        provenance=GoldenItemProvenance(
            source_type="synthetic",
            source_reference="evals/corpus synthetic fixture",
            permission_status="synthetic_owned",
            license_terms_status="not_required",
            is_public=False,
            is_synthetic=True,
            human_review_status="pending",
            annotators=AnnotatorRoles(
                engineer_relevance=ANNOTATOR_A, legal_sme_entailment="pending-sme"
            ),
            annotations=[],
            adjudication=AdjudicationRecord(status=adjudication_status),  # type: ignore[arg-type]
        ),
    )
    return query.model_dump(mode="json")


def write_gold(path: Path, rows: list[dict[str, Any]]) -> str:
    metadata = json.dumps(
        {"version": "legal_retrieval_v2", "provenance_schema_version": "golden_provenance_v1"}
    )
    path.write_text(
        metadata + "\n" + "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return metadata


def sidecars_for(spec: tuple[tuple[str, str, str, str], ...]) -> tuple[list, list]:
    sidecar_a = [annotation(case, label_a) for case, label_a, _, _ in spec]
    sidecar_b = [annotation(case, label_b, annotator=ANNOTATOR_B) for case, _, label_b, _ in spec]
    return sidecar_a, sidecar_b


def compute_iaa_report(
    base: Path,
    spec: tuple[tuple[str, str, str, str], ...],
    *,
    extra_args: list[str] | None = None,
) -> tuple[int, Path]:
    """Run compute_iaa over fixture sidecars under ``base``; return (code, report)."""
    base.mkdir(parents=True, exist_ok=True)
    case_ids = [case for case, _, _, _ in spec]
    sidecar_a, sidecar_b = sidecars_for(spec)
    questions = base / "questions.jsonl"
    path_a = base / "sidecar-a.jsonl"
    path_b = base / "sidecar-b.jsonl"
    report = base / "iaa_wave3.json"
    write_questions(questions, case_ids)
    write_jsonl(path_a, sidecar_a)
    write_jsonl(path_b, sidecar_b)
    code = run_iaa(
        [
            "--a",
            str(path_a),
            "--b",
            str(path_b),
            "--questions",
            str(questions),
            "--out",
            str(report),
            *(extra_args or []),
        ]
    )
    return code, report


def wave3_fixture(tmp_path: Path) -> dict[str, Path]:
    """Full fixture: gold set + sidecars + passing IAA report (thresholds met)."""
    rows = [gold_row(case, grader=grader) for case, _, _, grader in WAVE3_FIXTURE]
    gold = tmp_path / "legal_retrieval_v2.jsonl"
    write_gold(gold, rows)
    _, report = compute_iaa_report(tmp_path, WAVE3_FIXTURE)
    return {"gold": gold, "report": report, "out": tmp_path / "merged.jsonl"}


def merge_argv(fixture: dict[str, Path], tmp_path: Path, *, apply: bool = True) -> list[str]:
    argv = [
        "--gold",
        str(fixture["gold"]),
        "--a",
        str(tmp_path / "sidecar-a.jsonl"),
        "--b",
        str(tmp_path / "sidecar-b.jsonl"),
        "--iaa",
        str(fixture["report"]),
        "--out",
        str(fixture["out"]),
    ]
    if apply:
        argv.append("--apply")
    return argv


def read_gold_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0], [json.loads(line) for line in lines[1:]]


# --- kappa math -----------------------------------------------------------


def test_metrics_kappa_perfect_agreement_is_one() -> None:
    matrix = {"relevant": {"relevant": 3}, "irrelevant": {"irrelevant": 2}}
    assert cohen_kappa_unweighted(matrix) == pytest.approx(1.0)


def test_metrics_kappa_matches_hand_computed_example() -> None:
    # rows=A, cols=B; p_o=0.7, marginals 6/4 vs 5/5, p_e=0.5, kappa=0.4.
    matrix = {
        "relevant": {"relevant": 4, "irrelevant": 2},
        "irrelevant": {"relevant": 1, "irrelevant": 3},
    }
    assert cohen_kappa_unweighted(matrix) == pytest.approx(0.4, abs=0.01)


def test_metrics_kappa_degenerate_all_one_class_is_one() -> None:
    assert cohen_kappa_unweighted({"relevant": {"relevant": 4}}) == pytest.approx(1.0)


def test_metrics_kappa_empty_matrix_is_zero() -> None:
    assert cohen_kappa_unweighted({}) == pytest.approx(0.0)


def test_kappa_guard_when_expected_agreement_is_one() -> None:
    assert kappa_from_margins(1.0, 1.0) == pytest.approx(1.0)
    assert kappa_from_margins(0.8, 1.0) == pytest.approx(0.0)
    assert script_kappa_from_margins(0.8, 1.0) == pytest.approx(0.0)
    assert script_kappa_from_margins(1.0, 1.0) == pytest.approx(1.0)


def test_script_and_metrics_kappa_agree_on_shared_matrices() -> None:
    matrices = [
        {
            "relevant": {"relevant": 4, "irrelevant": 2},
            "irrelevant": {"relevant": 1, "irrelevant": 3},
        },
        {
            "relevant": {"relevant": 3, "partially_relevant": 1},
            "partially_relevant": {"partially_relevant": 2, "irrelevant": 1},
            "irrelevant": {"irrelevant": 2, "not_applicable": 1},
        },
    ]
    for matrix in matrices:
        labels_a = [label for label, row in matrix.items() for _ in range(sum(row.values()))]
        labels_b = [
            label for row in matrix.values() for label, count in row.items() for _ in range(count)
        ]
        assert iaa.cohen_kappa(labels_a, labels_b) == pytest.approx(cohen_kappa_unweighted(matrix))


# --- compute_iaa gate and report ------------------------------------------


def test_iaa_passes_gate_when_thresholds_met(tmp_path: Path) -> None:
    code, report_path = compute_iaa_report(tmp_path, WAVE3_FIXTURE)
    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["gate"]["passed"] is True
    assert report["raw_agreement"] == pytest.approx(10 / 12)
    assert report["kappa"] == pytest.approx(0.7624, abs=0.01)
    assert report["disagreements"] == [
        {"case_id": "direct_answer-unit-011", "a": "relevant", "b": "partially_relevant"},
        {"case_id": "direct_answer-unit-012", "a": "partially_relevant", "b": "irrelevant"},
    ]


def test_iaa_gate_failure_exits_two_on_raw_agreement(tmp_path: Path) -> None:
    all_agree = tuple(
        (case, label_a, label_a, grader) for case, label_a, _, grader in WAVE3_FIXTURE
    )
    code, report_path = compute_iaa_report(
        tmp_path, all_agree, extra_args=["--min-agreement", "1.1"]
    )
    assert code == 2  # kappa 1.0 >= 0.70 but raw 1.0 < 1.1
    assert json.loads(report_path.read_text())["gate"]["passed"] is False


def test_iaa_gate_failure_exits_two_on_kappa(tmp_path: Path) -> None:
    spec = (
        ("direct_answer-unit-001", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-002", "relevant", "irrelevant", "deterministic"),
        ("direct_answer-unit-003", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-004", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-005", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-006", "relevant", "relevant", "deterministic"),
    )
    code, _ = compute_iaa_report(tmp_path, spec)
    assert code == 2  # raw 5/6 >= 0.80 but kappa 0.0 < 0.70


def test_iaa_hand_computed_kappa_lands_in_report(tmp_path: Path) -> None:
    spec = (
        ("direct_answer-unit-001", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-002", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-003", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-004", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-005", "relevant", "irrelevant", "deterministic"),
        ("direct_answer-unit-006", "relevant", "irrelevant", "deterministic"),
        ("direct_answer-unit-007", "irrelevant", "relevant", "deterministic"),
        ("direct_answer-unit-008", "irrelevant", "irrelevant", "deterministic"),
        ("direct_answer-unit-009", "irrelevant", "irrelevant", "deterministic"),
        ("direct_answer-unit-010", "irrelevant", "irrelevant", "deterministic"),
    )
    code, report_path = compute_iaa_report(tmp_path, spec)
    assert code == 2  # raw agreement 0.7 < 0.80
    report = json.loads(report_path.read_text())
    assert report["kappa"] == pytest.approx(0.4, abs=0.01)
    assert cohen_kappa_unweighted(report["confusion_matrix"]) == pytest.approx(report["kappa"])


def test_iaa_small_sample_requires_total_agreement(tmp_path: Path) -> None:
    agreeing = (
        ("direct_answer-unit-001", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-002", "partially_relevant", "partially_relevant", "deterministic"),
        ("direct_answer-unit-003", "irrelevant", "irrelevant", "deterministic"),
        ("direct_answer-unit-004", "not_applicable", "not_applicable", "deterministic"),
    )
    code, _ = compute_iaa_report(tmp_path / "agree", agreeing)
    assert code == 0  # n < 5 with 100% agreement passes
    disputed = (
        ("direct_answer-unit-001", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-002", "partially_relevant", "partially_relevant", "deterministic"),
        ("direct_answer-unit-003", "irrelevant", "irrelevant", "deterministic"),
        ("direct_answer-unit-004", "relevant", "irrelevant", "deterministic"),
    )
    code, _ = compute_iaa_report(tmp_path / "dispute", disputed)
    assert code == 2  # n < 5 with any disagreement fails even at kappa thresholds


def test_iaa_report_shape_and_strata(tmp_path: Path) -> None:
    spec = (
        *WAVE3_FIXTURE[:3],
        ("direct_answer-unit-004", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-005", "relevant", "partially_relevant", "deterministic"),
    )
    sidecar_a, sidecar_b = sidecars_for(spec)
    sidecar_a[1]["confidence"] = "medium"
    sidecar_a[2]["confidence"] = "low"
    base = tmp_path / "shape"
    base.mkdir()
    write_questions(base / "questions.jsonl", [case for case, _, _, _ in spec])
    write_jsonl(base / "sidecar-a.jsonl", sidecar_a)
    write_jsonl(base / "sidecar-b.jsonl", sidecar_b)
    code = run_iaa(
        [
            "--a",
            str(base / "sidecar-a.jsonl"),
            "--b",
            str(base / "sidecar-b.jsonl"),
            "--questions",
            str(base / "questions.jsonl"),
            "--out",
            str(base / "report.json"),
            "--min-kappa",
            "0.0",
            "--min-agreement",
            "0.6",
        ]
    )
    assert code == 0  # raw 0.8 >= 0.6, kappa 0.0 >= 0.0
    report = json.loads((base / "report.json").read_text())
    assert set(report) >= {
        "annotators",
        "n",
        "raw_agreement",
        "kappa",
        "confusion_matrix",
        "disagreements",
        "per_confidence",
        "per_agreement_pattern",
        "gate",
    }
    assert report["annotators"] == [ANNOTATOR_A, ANNOTATOR_B]
    assert report["n"] == 5
    labels = ("relevant", "partially_relevant", "irrelevant", "not_applicable")
    assert set(report["confusion_matrix"]) == set(labels)
    assert all(set(row) == set(labels) for row in report["confusion_matrix"].values())
    assert report["disagreements"] == [
        {"case_id": "direct_answer-unit-005", "a": "relevant", "b": "partially_relevant"}
    ]
    assert report["per_confidence"]["high"]["n"] == 3
    assert report["per_confidence"]["medium"]["n"] == 1  # min(high, medium)
    assert report["per_confidence"]["low"]["n"] == 1  # least confident annotator wins
    assert report["per_agreement_pattern"]["exact_agreement"]["n"] == 4
    assert report["per_agreement_pattern"]["adjacent_disagreement"]["n"] == 1
    assert report["per_agreement_pattern"]["distant_disagreement"]["n"] == 0
    assert report["gate"]["thresholds"]["min_kappa"] == pytest.approx(0.0)
    assert report["gate"]["thresholds"]["min_agreement"] == pytest.approx(0.6)


def test_iaa_structural_errors_exit_one(tmp_path: Path) -> None:
    base = (
        ("direct_answer-unit-001", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-002", "relevant", "relevant", "deterministic"),
    )
    questions = tmp_path / "questions.jsonl"
    write_questions(questions, [case for case, _, _, _ in base])

    def run(base_dir: Path) -> int:
        return run_iaa(
            [
                "--a",
                str(base_dir / "sidecar-a.jsonl"),
                "--b",
                str(base_dir / "sidecar-b.jsonl"),
                "--questions",
                str(questions),
                "--out",
                str(base_dir / "report.json"),
            ]
        )

    # case universe mismatch: annotator B missed one case
    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    write_jsonl(mismatch / "sidecar-a.jsonl", sidecars_for(base)[0])
    write_jsonl(mismatch / "sidecar-b.jsonl", sidecars_for(base)[1][:1])
    assert run(mismatch) == 1

    # malformed JSON line
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    write_jsonl(malformed / "sidecar-a.jsonl", sidecars_for(base)[0])
    (malformed / "sidecar-b.jsonl").write_text("not-json\n", encoding="utf-8")
    assert run(malformed) == 1

    # out-of-vocabulary label, duplicate case_id, wrong role, mixed annotators
    corruptions = (
        "out-of-vocab",
        lambda records: records[0]["labels"].__setitem__("retrieval_relevance", "mostly_relevant"),
        "duplicate",
        lambda records: records.append(dict(records[0])),
        "role",
        lambda records: records[0].__setitem__("role", "legal_sme"),
        "mixed-annotators",
        lambda records: records[1].__setitem__("annotator_id", "someone-else"),
    )
    for name, corrupt in zip(corruptions[::2], corruptions[1::2], strict=True):
        case_dir = tmp_path / str(name)
        case_dir.mkdir()
        records = sidecars_for(base)[1]
        corrupt(records)
        write_jsonl(case_dir / "sidecar-b.jsonl", records)
        write_jsonl(case_dir / "sidecar-a.jsonl", sidecars_for(base)[0])
        assert run(case_dir) == 1, name


def test_iaa_cli_subprocess_reports_gate_failure(tmp_path: Path) -> None:
    spec = (
        ("direct_answer-unit-001", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-002", "relevant", "irrelevant", "deterministic"),
        ("direct_answer-unit-003", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-004", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-005", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-006", "relevant", "relevant", "deterministic"),
    )
    case_ids = [case for case, _, _, _ in spec]
    sidecar_a, sidecar_b = sidecars_for(spec)
    write_questions(tmp_path / "questions.jsonl", case_ids)
    write_jsonl(tmp_path / "sidecar-a.jsonl", sidecar_a)
    write_jsonl(tmp_path / "sidecar-b.jsonl", sidecar_b)
    script = Path(iaa.__file__).resolve()
    completed = subprocess.run(  # noqa: S603 — fixed project script, fixture paths only
        [
            sys.executable,
            str(script),
            "--a",
            str(tmp_path / "sidecar-a.jsonl"),
            "--b",
            str(tmp_path / "sidecar-b.jsonl"),
            "--questions",
            str(tmp_path / "questions.jsonl"),
            "--out",
            str(tmp_path / "report.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 2
    assert json.loads((tmp_path / "report.json").read_text())["gate"]["passed"] is False


# --- merge_annotations ------------------------------------------------------


def test_merge_applies_release_rule_for_deterministic_agreement(tmp_path: Path) -> None:
    fixture = wave3_fixture(tmp_path)
    assert run_merge(merge_argv(fixture, tmp_path)) == 0
    metadata, rows = read_gold_rows(fixture["out"])
    assert metadata == read_gold_rows(fixture["gold"])[0]
    assert [row["case_id"] for row in rows] == [case for case, _, _, _ in WAVE3_FIXTURE]
    by_case = {row["case_id"]: row for row in rows}

    release = by_case["direct_answer-unit-001"]["provenance"]
    assert release["quality_tier"] == "release"
    assert release["human_review_status"] == "engineer_reviewed"
    assert [record["annotator_id"] for record in release["annotations"]] == [
        ANNOTATOR_A,
        ANNOTATOR_B,
    ]
    # the case binding key is stripped; attached records are pure AnnotationRecords
    assert all("case_id" not in record for record in release["annotations"])
    adjudication = release["adjudication"]
    assert adjudication["status"] == "not_required"
    assert adjudication["method"] == "independent_sme"
    assert adjudication["adjudicator"] is None
    assert adjudication["rationale"] == RELEASE_RATIONALE
    assert adjudication["checklist"] == {}
    assert adjudication["decided_at"]

    sme = by_case["direct_answer-unit-005"]["provenance"]
    assert sme["quality_tier"] == "candidate"
    assert sme["human_review_status"] == "engineer_reviewed"
    assert sme["adjudication"]["status"] == "pending_sme"
    assert sme["adjudication"].get("adjudicator") is None
    assert len(sme["annotations"]) == 2

    disputed = by_case["direct_answer-unit-011"]["provenance"]
    assert disputed["quality_tier"] == "candidate"
    assert disputed["adjudication"]["status"] == "pending_sme"
    assert len(disputed["annotations"]) == 2


def test_merge_row_counts_match_gate_rules(tmp_path: Path) -> None:
    fixture = wave3_fixture(tmp_path)
    run_merge(merge_argv(fixture, tmp_path))
    _, rows = read_gold_rows(fixture["out"])
    release = sum(row["provenance"]["quality_tier"] == "release" for row in rows)
    assert release == 8  # 10 agree, 2 of those are sme-graded => 8 release
    assert all(
        row["provenance"]["adjudication"]["status"]
        == ("not_required" if row["provenance"]["quality_tier"] == "release" else "pending_sme")
        for row in rows
    )
    assert all(row["provenance"]["human_review_status"] == "engineer_reviewed" for row in rows)


def test_merge_dry_run_writes_nothing(tmp_path: Path) -> None:
    fixture = wave3_fixture(tmp_path)
    assert run_merge(merge_argv(fixture, tmp_path, apply=False)) == 0
    assert not fixture["out"].exists()


def test_merge_refuses_gate_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = (
        ("direct_answer-unit-001", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-002", "relevant", "irrelevant", "deterministic"),
        ("direct_answer-unit-003", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-004", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-005", "relevant", "relevant", "deterministic"),
        ("direct_answer-unit-006", "relevant", "relevant", "deterministic"),
    )
    rows = [gold_row(case, grader=grader) for case, _, _, grader in spec]
    gold = tmp_path / "gold.jsonl"
    write_gold(gold, rows)
    _, report = compute_iaa_report(tmp_path, spec)  # failing gate (kappa 0.0)
    fixture = {"gold": gold, "report": report, "out": tmp_path / "merged.jsonl"}
    code = run_merge(merge_argv(fixture, tmp_path))
    assert code == 2
    assert "gate" in capsys.readouterr().err
    assert not fixture["out"].exists()


def test_merge_rejects_stale_report(tmp_path: Path) -> None:
    fixture = wave3_fixture(tmp_path)
    # report computed from a different (all-agree) sidecar set in its own directory,
    # so the merge-input sidecars in tmp_path stay the disagreeing originals
    all_agree = tuple(
        (case, label_a, label_a, grader) for case, label_a, _, grader in WAVE3_FIXTURE
    )
    stale_report = compute_iaa_report(tmp_path / "stale", all_agree)[1]
    argv = merge_argv(fixture, tmp_path)
    argv[argv.index("--iaa") + 1] = str(stale_report)
    code = run_merge(argv)
    assert code == 1
    assert not fixture["out"].exists()


def test_merge_validation_failure_rolls_back(tmp_path: Path) -> None:
    fixture = wave3_fixture(tmp_path)
    original = fixture["gold"].read_bytes()
    sidecar_a, sidecar_b = sidecars_for(WAVE3_FIXTURE)
    sidecar_b[0]["sentiment"] = "unvalidated-extra-field"  # breaks AnnotationRecord extra=forbid
    write_jsonl(tmp_path / "sidecar-a.jsonl", sidecar_a)
    write_jsonl(tmp_path / "sidecar-b.jsonl", sidecar_b)
    code = run_merge(merge_argv(fixture, tmp_path))
    assert code == 1
    assert fixture["gold"].read_bytes() == original  # untouched
    assert list(tmp_path.glob("*.merge-*.tmp")) == []  # temp rolled back


def test_merge_temp_file_removed_on_any_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a KeyboardInterrupt mid-merge must not leave temp-file litter."""
    from scripts import merge_annotations

    fixture = wave3_fixture(tmp_path)
    original = fixture["gold"].read_bytes()

    def interrupting_validation(target: Path, repo_root: Path) -> tuple[bool, str]:
        raise KeyboardInterrupt

    monkeypatch.setattr(merge_annotations, "_run_gold_validation", interrupting_validation)
    with pytest.raises(KeyboardInterrupt):
        run_merge(merge_argv(fixture, tmp_path))
    assert fixture["gold"].read_bytes() == original  # untouched
    assert list(tmp_path.glob("*.merge-*.tmp")) == []  # temp unlinked before the raise


def test_merge_replaced_output_is_world_readable(tmp_path: Path) -> None:
    """The mkstemp 0o600 mode is normalized to 0o644 before the replace."""
    fixture = wave3_fixture(tmp_path)
    assert run_merge(merge_argv(fixture, tmp_path)) == 0
    assert fixture["out"].stat().st_mode & 0o777 == 0o644


def test_merge_is_idempotent(tmp_path: Path) -> None:
    fixture = wave3_fixture(tmp_path)
    argv = merge_argv(fixture, tmp_path)
    assert run_merge(argv) == 0
    first_pass = fixture["out"].read_bytes()
    # second pass runs against the already-merged output
    argv[argv.index("--gold") + 1] = str(fixture["out"])
    assert run_merge(argv) == 0
    assert fixture["out"].read_bytes() == first_pass
    _, rows = read_gold_rows(fixture["out"])
    for row in rows:
        annotations = row["provenance"]["annotations"]
        assert len(annotations) == 2
        assert {record["annotator_id"] for record in annotations} == {ANNOTATOR_A, ANNOTATOR_B}


def test_merge_missing_sidecar_case_exits_one(tmp_path: Path) -> None:
    fixture = wave3_fixture(tmp_path)
    sidecar_b = [
        json.loads(line)
        for line in (tmp_path / "sidecar-b.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    write_jsonl(tmp_path / "sidecar-b.jsonl", sidecar_b[:-1])
    code = run_merge(merge_argv(fixture, tmp_path))
    assert code == 1
    assert not fixture["out"].exists()


def test_merged_output_passes_real_validation(tmp_path: Path) -> None:
    fixture = wave3_fixture(tmp_path)
    assert run_merge(merge_argv(fixture, tmp_path)) == 0
    result = asyncio.run(load_validated_golden_set(fixture["out"]))
    assert isinstance(result, Success)
    queries = result.unwrap().queries
    assert len(queries) == len(WAVE3_FIXTURE)
    release = [
        query
        for query in queries
        if query.provenance is not None and query.provenance.quality_tier == "release"
    ]
    assert len(release) == 8
    assert all(
        query.provenance is not None
        and query.provenance.adjudication is not None
        and query.provenance.adjudication.status == "not_required"
        and query.provenance.adjudication.adjudicator is None
        and len(query.provenance.annotations) == 2
        for query in release
    )
