"""Regression protection for the context-comparison CLI options (wave 3).

Greptile finding: ``main()`` accepted ``--k`` and ``--commit`` but neither
value reached ``run_comparison`` — ``--k 5`` still evaluated and wrote the
default cutoff of ten, and the requested commit was absent from the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ai.evaluation.context_comparison import run_comparison

REPO_ROOT = Path(__file__).resolve().parents[4]
GOLDEN_V1 = REPO_ROOT / "evals/golden/legal_retrieval_v1.jsonl"


def _seed_corpus(tmp_path: Path) -> Path:
    """One recorded source whose pages the offline index can score."""
    source_dir = tmp_path / "SRC-TST"
    source_dir.mkdir(parents=True)
    payload = {
        "pages": [
            {
                "page_no": 1,
                "text": (
                    "The Concessionaire shall provide the Performance Security "
                    "within 180 days.\n\nThe Authority shall appoint the "
                    "Arbitral Tribunal for disputes."
                ),
            }
        ]
    }
    (source_dir / "pages.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _fixture_golden_set(tmp_path: Path) -> Path:
    """A v1 golden set with one row cloned from the committed set."""
    committed = [
        json.loads(line)
        for line in GOLDEN_V1.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata, first_row = committed[0], committed[1]
    row = dict(first_row)
    row["query"] = "termination obligations"
    row["expected_chunk_ids"] = ["SRC-TST:p1:c01"]
    row["expected_document_ids"] = []
    path = tmp_path / "fixture_v1.jsonl"
    path.write_text(
        json.dumps(metadata) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )
    return path


def _fixture_overlay(tmp_path: Path) -> Path:
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps({"SRC-TST:p1:c01": "Performance Security"}), encoding="utf-8")
    return overlay_path


@pytest.mark.asyncio
async def test_run_comparison_records_the_requested_cutoff_and_commit(tmp_path: Path) -> None:
    report = await run_comparison(
        golden_set_path=_fixture_golden_set(tmp_path),
        corpus_root=_seed_corpus(tmp_path),
        overlay_path=_fixture_overlay(tmp_path),
        cutoff=3,
        commit="abc1234",
    )

    assert report is not None
    assert report.cutoff == 3
    assert report.commit == "abc1234"


@pytest.mark.asyncio
async def test_run_comparison_defaults_match_the_reported_schema(tmp_path: Path) -> None:
    report = await run_comparison(
        golden_set_path=_fixture_golden_set(tmp_path),
        corpus_root=_seed_corpus(tmp_path),
        overlay_path=_fixture_overlay(tmp_path),
    )

    assert report is not None
    assert report.cutoff == 10
    assert report.commit is None
