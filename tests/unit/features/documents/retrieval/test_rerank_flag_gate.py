"""The served-path rerank gate: flag-off passes fused order through unchanged.

`get_configured_reranker` returning ``None`` is a normal serving state (the
settings comment cites the recorded measurement that justified default-on).
The node contract under flag-off mirrors the degraded-shape contract: fused
order truncated to the rerank limit, stage unspent, no provider call.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from app.config.settings import get_settings
from app.features.documents.application.retrieval.graph.budgets import STAGE_RERANK
from app.features.documents.application.retrieval.graph.nodes import (
    RERANK_LIMIT,
    make_reranker_node,
)
from app.features.documents.application.retrieval.graph.state import RetrievedChunk, StageUsage


def _chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"SRC-01:p1:c{index}",
        chunk_text=f"clause {index}",
        preamble="",
        clause_type="term",
        parent_doc_id="doc-1",
        chunk_index=index,
        metadata_={},
        custom_metadata={},
        score=10.0 - index,
    )


def _state(chunks: list[RetrievedChunk]) -> dict[str, Any]:
    return {
        "query_plan": SimpleNamespace(rewritten_query="governing law"),
        "retrieved_chunks": chunks,
    }


class _ExplodingReranker:
    """If the flag-off resolution ever reaches a reranker, this test fails."""

    async def rerank(
        self, _query: str, _chunks: list[RetrievedChunk], *, limit: int
    ) -> list[RetrievedChunk]:
        flag_violation = "rerank provider called while the flag is off"
        raise AssertionError(flag_violation)


@pytest.fixture
def _rerank_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_RERANK_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def _rerank_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_RERANK_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("_rerank_off")
async def test_flag_off_passes_fused_order_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The exploding stub sits behind the gate (on get_shared_reranker): under
    # flag-off the accessor must return None before the shared reranker is
    # ever consulted, and the node must pass fused order through.
    monkeypatch.setattr(
        "app.features.documents.application.retrieval.reranking.reranker.get_shared_reranker",
        _ExplodingReranker,
    )
    node = make_reranker_node()
    chunks = [_chunk(i) for i in range(8)]

    result = await node(_state(chunks), runtime=None)

    assert result["reranked_chunks"] == chunks[:RERANK_LIMIT]
    assert result["stage_usage"] == [StageUsage(stage=STAGE_RERANK, calls=0)]


@pytest.mark.usefixtures("_rerank_off")
async def test_flag_off_truncates_to_the_rerank_limit() -> None:
    node = make_reranker_node()
    chunks = [_chunk(i) for i in range(RERANK_LIMIT + 3)]

    result = await node(_state(chunks), runtime=None)

    assert len(result["reranked_chunks"]) == RERANK_LIMIT


@pytest.mark.usefixtures("_rerank_on")
async def test_flag_on_still_reranks(monkeypatch: pytest.MonkeyPatch) -> None:
    node = make_reranker_node()
    chunks = [_chunk(i) for i in range(8)]
    seen: dict[str, Any] = {}

    class _RecordingReranker:
        async def rerank(
            self, query: str, candidates: list[RetrievedChunk], *, limit: int
        ) -> list[RetrievedChunk]:
            seen.update(query=query, count=len(candidates), limit=limit)
            return list(reversed(candidates))[:limit]

    monkeypatch.setattr(
        "app.features.documents.application.retrieval.graph.nodes.get_configured_reranker",
        _RecordingReranker,
    )

    result = await node(_state(chunks), runtime=None)

    assert seen == {"query": "governing law", "count": 8, "limit": RERANK_LIMIT}
    assert [chunk.chunk_id for chunk in result["reranked_chunks"]] == [
        f"SRC-01:p1:c{len(chunks) - 1 - i}" for i in range(RERANK_LIMIT)
    ]
