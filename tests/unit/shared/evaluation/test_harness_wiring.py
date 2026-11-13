"""Harness wiring tests (task 8.3).

Two invariants:

1. The harness's scoring path reaches seeded chunk ids: a seeded corpus index
   (and the real recorded corpus) yields non-zero recall through the exact
   retrieve closure the CLI wires, and every golden expected chunk id exists in
   the chunk universe the harness scores against.
2. A wiring failure is reported, never silently scored: an empty corpus (or a
   path where every query retrieves zero chunks) exits with the wiring-failure
   code 3 and writes no report, unless ``--allow-empty-corpus`` was passed
   deliberately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from app.ai.evaluation.deterministic import OfflineCorpusIndex, fuse_branch_ranks
from app.ai.evaluation.harness import _run
from app.ai.evaluation.runner import BranchAwareRetrieval, BranchContribution, run_retrieval_eval
from app.ai.evaluation.schema import GoldenQuery, load_golden_set

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
CORPUS_ROOT = REPO_ROOT / "evals/corpus/artifacts"
GOLDEN_V2 = REPO_ROOT / "evals/golden/legal_retrieval_v2.jsonl"
GOLDEN_V1 = REPO_ROOT / "evals/golden/legal_retrieval_v1.jsonl"


def _namespace(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "golden_set": GOLDEN_V1,
        "corpus": CORPUS_ROOT,
        "output": None,
        "k": 10,
        "commit": "test-commit",
        "branches": "bm25,trigram",
        "weights": None,
        "thresholds": None,
        "allow_empty_corpus": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _seed_corpus(tmp_path: Path, pages: dict[int, list[str]]) -> Path:
    """Write a recorded-corpus fixture whose convention chunks are known."""
    source_dir = tmp_path / "SRC-TST"
    source_dir.mkdir(parents=True)
    payload = {
        "pages": [
            {"page_no": page_no, "text": "\n\n".join(chunks)} for page_no, chunks in pages.items()
        ]
    }
    (source_dir / "pages.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def _retrieve_like_cli(
    index: OfflineCorpusIndex, cutoff: int
) -> Callable[[GoldenQuery], Awaitable[BranchAwareRetrieval]]:
    """The exact retrieve closure harness._run wires (branch ranks + RRF)."""

    async def retrieve(query: GoldenQuery) -> BranchAwareRetrieval:
        branch_ranks = index.branch_ranks(query.query, cutoff, branches=("bm25", "trigram"))
        return BranchAwareRetrieval(
            fused_chunk_ids=fuse_branch_ranks(branch_ranks, cutoff),
            branches=[
                BranchContribution(branch=name, ranked_chunk_ids=ranks)
                for name, ranks in branch_ranks.items()
            ],
        )

    return retrieve


@pytest.mark.asyncio
async def test_harness_reaches_seeded_chunk_ids_in_a_seeded_index(tmp_path: Path) -> None:
    """A seeded corpus drives non-zero recall through the live scoring path."""
    corpus = _seed_corpus(
        tmp_path,
        {
            1: [
                "The Concessionaire shall provide the Performance Security within 180 days.",
                "The Authority shall appoint the Arbitral Tribunal for disputes.",
            ],
            2: ["Unrelated filler text about procurement of stationery supplies."],
        },
    )
    index = OfflineCorpusIndex(corpus)
    seeded = {"SRC-TST:p1:c01", "SRC-TST:p1:c02", "SRC-TST:p2:c01"}
    assert seeded == set(index.chunk_ids)
    queries = [
        GoldenQuery(
            query="Within how many days must the Concessionaire provide the Performance Security?",
            expected_chunk_ids=["SRC-TST:p1:c01"],
            expected_document_ids=["SRC-TST"],
            document_kind="contracts",
            difficulty="easy",
            notes="wiring probe",
            awaiting_sme_expansion=True,
        ),
        GoldenQuery(
            query="Who appoints the Arbitral Tribunal for disputes?",
            expected_chunk_ids=["SRC-TST:p1:c02"],
            expected_document_ids=["SRC-TST"],
            document_kind="contracts",
            difficulty="easy",
            notes="wiring probe",
            awaiting_sme_expansion=True,
        ),
    ]
    evaluation = await run_retrieval_eval(
        queries=queries, retrieve=_retrieve_like_cli(index, cutoff=10), cutoff=10
    )
    assert evaluation.aggregates.recall_at_k == pytest.approx(1.0)
    retrieved = {chunk_id for row in evaluation.rows for chunk_id in row.retrieved_chunk_ids}
    assert {"SRC-TST:p1:c01", "SRC-TST:p1:c02"} <= retrieved


@pytest.mark.asyncio
async def test_every_golden_expected_chunk_id_exists_in_the_scored_universe() -> None:
    """The real corpus index contains every expected chunk id of the v2 set."""
    result = await load_golden_set(GOLDEN_V2)
    golden_set = result.unwrap()
    index = OfflineCorpusIndex(CORPUS_ROOT)
    universe = set(index.chunk_ids)
    missing = [
        (query.case_id, chunk_id)
        for query in golden_set.queries
        for chunk_id in query.expected_chunk_ids
        if chunk_id not in universe
    ]
    assert missing == []


@pytest.mark.asyncio
async def test_empty_corpus_reports_wiring_failure_and_writes_no_report(tmp_path: Path) -> None:
    """An empty chunk universe exits 3 with a wiring-failure message."""
    empty_corpus = tmp_path / "empty-corpus"
    empty_corpus.mkdir()
    output = tmp_path / "should-not-exist.json"
    code = await _run(
        _namespace(corpus=empty_corpus, output=output),
        branch_names=["bm25", "trigram"],
        branch_weights=None,
        thresholds=None,
    )
    assert code == 3
    assert not output.exists()


@pytest.mark.asyncio
async def test_allow_empty_corpus_deliberately_scores_the_empty_universe(tmp_path: Path) -> None:
    """The explicit override writes the all-zero report instead of failing."""
    empty_corpus = tmp_path / "empty-corpus"
    empty_corpus.mkdir()
    output = tmp_path / "deliberate-empty.json"
    code = await _run(
        _namespace(corpus=empty_corpus, output=output, allow_empty_corpus=True),
        branch_names=["bm25", "trigram"],
        branch_weights=None,
        thresholds=None,
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["aggregates"]["recall_at_k"] == pytest.approx(0.0)
