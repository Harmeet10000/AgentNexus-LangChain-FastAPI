"""The shared `retrieve(...)` service (harden 5.1/5.2/5.6).

Pins the typed-status contract: an outage, a deadline, an exhausted budget,
and an empty corpus are four different answers — never four shapes of the
same empty list — and a Graphiti outage degrades the coverage signal while
canonical chunks stand.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from graphiti_core.errors import GraphitiError
from returns.result import Failure, Success

from app.config.settings import get_settings
from app.features.documents.application.queries import documents as queries_documents
from app.features.documents.application.queries.documents import RankedChunk
from app.features.documents.domain.errors import DocumentDatabaseError
from app.features.documents.retrieval import service as retrieval_service
from app.features.documents.retrieval.service import (
    RetrievalFilters,
    RetrievalRequestContext,
    retrieve,
)

if TYPE_CHECKING:
    from typing import Any

_DOC = "00000000-0000-0000-0000-000000000001"

_ROWS = [
    {
        "chunk_id": "00000000-0000-0000-0000-0000000000aa",
        "document_id": _DOC,
        "title": "contract.pdf",
        "content": "The termination clause of the agreement.",
        "preamble": "",
        "document_version": 1,
        "chunk_index": 0,
        "chunk_kind": "legal_contract",
        "clause_type": "termination",
        "page_no": 4,
        "chunk_metadata": {"pipeline_version": "p1"},
        "document_metadata": {},
        "quality_warnings": [],
        "graphiti_verified": True,
    }
]

_FUSED = [
    RankedChunk(chunk_id="00000000-0000-0000-0000-0000000000aa", score=0.9, rank=1),
]


class _Repo:
    """Three-leg fused-search stub returning one canonical chunk."""

    async def bm25_search(self, **_kwargs: Any) -> Any:
        return Success([{"chunk_id": _FUSED[0].chunk_id, "score": -5.0}])

    async def vector_search(self, **_kwargs: Any) -> Any:
        return Success([])

    async def trigram_search(self, **_kwargs: Any) -> Any:
        return Success([])

    async def fetch_chunks_by_ids(self, chunk_ids: Any, **_kwargs: Any) -> Any:
        return Success(
            {row["chunk_id"]: dict(row) for row in _ROWS if row["chunk_id"] in chunk_ids}
        )


class _OutagedGraphiti:
    async def search(self, **_kwargs: Any) -> list[Any]:
        message = "simulated outage"
        raise GraphitiError(message)


def _context(repo: Any = None, **overrides: Any) -> RetrievalRequestContext:
    defaults: dict[str, Any] = {
        "principal_id": "alice",
        "tenant_id": "tenant-a",
        "allowed_doc_ids": (_DOC,),
        "repo": repo or _Repo(),
        "graphiti": None,
        "reranker": None,
    }
    defaults.update(overrides)
    return RetrievalRequestContext(**defaults)


def _filters(**overrides: Any) -> RetrievalFilters:
    return RetrievalFilters(**overrides)


@pytest.fixture
def _patched_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _embed(*_args: Any, **_kwargs: Any) -> list[float]:
        return [0.0]

    async def _retrieve_fused(**_kwargs: Any) -> Any:
        return Success(
            (
                [
                    RankedChunk(chunk_id=item.chunk_id, score=item.score, rank=item.rank)
                    for item in _FUSED
                ],
                {row["chunk_id"]: dict(row) for row in _ROWS},
            )
        )

    monkeypatch.setattr(retrieval_service, "embed_text", _embed)
    monkeypatch.setattr(queries_documents, "retrieve_fused", _retrieve_fused)


@pytest.fixture
def _patched_embed_only(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _embed(*_args: Any, **_kwargs: Any) -> list[float]:
        return [0.0]

    monkeypatch.setattr(retrieval_service, "embed_text", _embed)


@pytest.fixture
def _fusion_flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_GRAPHITI_FUSION_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_empty_scope_is_unauthorized_not_empty() -> None:
    result = await retrieve(
        _context(repo=_Repo(), allowed_doc_ids=()), "query", _filters(), 5, None
    )
    assert result.status == "unauthorized"
    assert result.retryable is False
    assert result.rows == []


@pytest.mark.usefixtures("_patched_fusion")
async def test_no_evidence_is_full_coverage_valid_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _empty(**_kwargs: Any) -> Any:
        return Success(([], {}))

    monkeypatch.setattr(queries_documents, "retrieve_fused", _empty)
    result = await retrieve(_context(), "query", _filters(), 5, None)
    assert result.status == "no_evidence"
    assert result.retryable is False
    assert result.rows == []
    assert result.coverage.documents_requested == 1


@pytest.mark.usefixtures("_patched_fusion")
async def test_postgres_outage_is_transient_not_no_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RetryableDBError(DocumentDatabaseError):
        retryable = True

    async def _failing(**_kwargs: Any) -> Any:
        message = "connection reset"
        return Failure(_RetryableDBError(message=message))

    monkeypatch.setattr(queries_documents, "retrieve_fused", _failing)
    result = await retrieve(_context(), "query", _filters(), 5, None)
    assert result.status == "transient_failure"
    assert result.retryable is True
    assert result.rows == []


@pytest.mark.usefixtures("_patched_fusion")
async def test_fused_deadline_is_a_typed_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_RETRIEVAL_POSTGRES_TIMEOUT_SECONDS", "0.01")
    get_settings.cache_clear()
    try:

        async def _slow(**_kwargs: Any) -> Any:
            await asyncio.sleep(0.2)
            return Success(([], {}))

        monkeypatch.setattr(queries_documents, "retrieve_fused", _slow)
        result = await retrieve(_context(), "query", _filters(), 5, None)
    finally:
        get_settings.cache_clear()
    assert result.status == "timeout"
    assert result.retryable is True


@pytest.mark.usefixtures("_patched_embed_only", "_fusion_flag_on")
async def test_graphiti_outage_preserves_canonical_chunks() -> None:
    """The legal-path outage contract (harden 5.2 / legal-corpus spec).

    Graphiti is a coverage signal, not legal truth: an outage degrades the
    coverage signal and returns the canonical Postgres chunks — through the
    REAL fused branch registry, not a stubbed fusion.
    """
    result = await retrieve(
        _context(
            graphiti=_OutagedGraphiti(),
            # Version/contract scope so the Graphiti partition fan-out
            # resolves; without it the branch contributes nothing without
            # reading (the no-scope contract) and no outage is exercised.
            document_versions=((_DOC, 1, None),),
            document_contracts=((_DOC, 1, (None,), True),),
        ),
        "query",
        _filters(),
        5,
        None,
    )
    assert result.status == "ok"
    assert len(result.rows) == 1
    assert result.rows[0]["chunk_id"] == _FUSED[0].chunk_id
    assert result.graph_available is False
    assert [warning.code for warning in result.warnings] == ["GRAPHITI_UNAVAILABLE"]
    assert result.coverage.graph_indexed_documents == 1  # from the row flags


@pytest.mark.usefixtures("_patched_fusion")
async def test_budget_exhaustion_is_budget_exhausted_not_no_evidence() -> None:
    from app.features.documents.retrieval.graph.budgets import StageBudget, StageBudgetTracker

    tracker = StageBudgetTracker(
        budgets={"retrieve_fusion": StageBudget(latency_ms=60_000, max_calls=0)},
        alert_threshold=100,
        alert_window_seconds=300,
    )
    result = await retrieve(_context(), "query", _filters(), 5, tracker)
    assert result.status == "budget_exhausted"
    assert result.retryable is False
    assert result.rows == []


@pytest.mark.usefixtures("_patched_fusion")
async def test_out_of_scope_request_documents_are_unauthorized() -> None:
    result = await retrieve(
        _context(),
        "query",
        _filters(document_ids=("00000000-0000-0000-0000-000000000999",)),
        5,
        None,
    )
    assert result.status == "unauthorized"


@pytest.mark.usefixtures("_patched_fusion")
async def test_ok_result_carries_provenance_rows() -> None:
    from app.features.documents.retrieval.row_hydration import pipeline_version

    result = await retrieve(_context(), "query", _filters(), 5, None)
    assert result.status == "ok"
    row = result.rows[0]
    assert row["page_no"] == 4
    assert row["document_version"] == 1
    assert pipeline_version(row) == "p1"
    assert result.rank_by_chunk_id[_FUSED[0].chunk_id] == 1
