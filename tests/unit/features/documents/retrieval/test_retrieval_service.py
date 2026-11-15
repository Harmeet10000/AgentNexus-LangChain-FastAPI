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


_GRAPH_ONLY_CHUNK = "00000000-0000-0000-0000-0000000000bb"


@pytest.fixture
def _patched_fusion_with_graph_only_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fusion returns the canonical chunk plus a graph-only match the SQL legs
    never return (they exclude it on the active filters)."""
    canonical_row = {
        **_ROWS[0],
        "chunk_metadata": {"pipeline_version": "p1", "jurisdiction": "Ontario"},
    }
    graph_row = {
        **_ROWS[0],
        "chunk_id": _GRAPH_ONLY_CHUNK,
        "clause_type": "payment",
        "graphiti_verified": False,
        "chunk_metadata": {"jurisdiction": "Alberta"},
    }

    async def _embed(*_args: Any, **_kwargs: Any) -> list[float]:
        return [0.0]

    async def _retrieve_fused(**_kwargs: Any) -> Any:
        return Success(
            (
                [
                    RankedChunk(chunk_id=_FUSED[0].chunk_id, score=0.9, rank=1),
                    RankedChunk(chunk_id=_GRAPH_ONLY_CHUNK, score=0.8, rank=2),
                ],
                {_FUSED[0].chunk_id: dict(canonical_row), _GRAPH_ONLY_CHUNK: dict(graph_row)},
            )
        )

    monkeypatch.setattr(retrieval_service, "embed_text", _embed)
    monkeypatch.setattr(queries_documents, "retrieve_fused", _retrieve_fused)


@pytest.mark.usefixtures("_patched_fusion_with_graph_only_match")
async def test_graph_only_match_is_held_to_the_active_clause_filter() -> None:
    """The Graphiti branch bypasses the SQL filters, so the fusion's predicate
    re-runs post-fusion: a graph-only match excluded by clause_type earns no
    row — and downstream, no evidence handle."""
    result = await retrieve(_context(), "query", _filters(clause_type="termination"), 5, None)
    assert result.status == "ok"
    assert [row["chunk_id"] for row in result.rows] == [_FUSED[0].chunk_id]


@pytest.mark.usefixtures("_patched_fusion_with_graph_only_match")
async def test_jurisdiction_filter_excludes_the_graph_only_match() -> None:
    result = await retrieve(_context(), "query", _filters(jurisdiction="Ontario"), 5, None)
    assert result.status == "ok"
    assert [row["chunk_id"] for row in result.rows] == [_FUSED[0].chunk_id]


@pytest.mark.usefixtures("_patched_fusion_with_graph_only_match")
async def test_graph_indexed_requirement_excludes_the_unverified_graph_match() -> None:
    result = await retrieve(_context(), "query", _filters(require_graphiti_verified=True), 5, None)
    assert result.status == "ok"
    assert [row["chunk_id"] for row in result.rows] == [_FUSED[0].chunk_id]


@pytest.mark.usefixtures("_patched_fusion_with_graph_only_match")
async def test_every_match_excluded_by_a_filter_is_no_evidence() -> None:
    """A filter that excludes every candidate — graph-only ones included — is
    the valid-negative answer, never a silently narrowed ok result."""
    result = await retrieve(_context(), "query", _filters(clause_type="confidentiality"), 5, None)
    assert result.status == "no_evidence"
    assert result.rows == []


@pytest.mark.usefixtures("_patched_fusion_with_graph_only_match")
async def test_unfiltered_search_keeps_the_graph_only_match() -> None:
    """With no active filter the predicate is a no-op: the graph match stands."""
    result = await retrieve(_context(), "query", _filters(), 5, None)
    assert result.status == "ok"
    assert [row["chunk_id"] for row in result.rows] == [_FUSED[0].chunk_id, _GRAPH_ONLY_CHUNK]


# --- chunk_ids forwarding (harden 5.1 filters) -------------------------------


@pytest.mark.usefixtures("_patched_fusion")
async def test_chunk_ids_filter_is_forwarded_to_the_fused_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`RetrievalFilters.chunk_ids` reaches the SQL legs: the caller's chunk
    allowlist narrows the search instead of being silently dropped."""
    captured: dict[str, Any] = {}

    async def _fused(**kwargs: Any) -> Any:
        captured["filter_params"] = kwargs["filter_params"]
        return Success((list(_FUSED), {row["chunk_id"]: dict(row) for row in _ROWS}))

    monkeypatch.setattr(queries_documents, "retrieve_fused", _fused)
    result = await retrieve(_context(), "query", _filters(chunk_ids=(_FUSED[0].chunk_id,)), 5, None)
    assert result.status == "ok"
    assert captured["filter_params"]["chunk_ids"] == [_FUSED[0].chunk_id]


@pytest.mark.usefixtures("_patched_fusion_with_graph_only_match")
async def test_chunk_allowlist_excludes_matches_outside_it() -> None:
    """A fused match outside the requested chunk allowlist earns no row — the
    same predicate the SQL legs enforce, re-run over the fusion output."""
    result = await retrieve(_context(), "query", _filters(chunk_ids=(_FUSED[0].chunk_id,)), 5, None)
    assert result.status == "ok"
    assert [row["chunk_id"] for row in result.rows] == [_FUSED[0].chunk_id]


# --- Graphiti warning honesty ------------------------------------------------


@pytest.mark.usefixtures("_patched_fusion")
async def test_no_graphiti_warning_when_fusion_is_disabled() -> None:
    """A healthy canonical-only search (fusion flag off, the default) is not
    degraded: the GRAPHITI_UNAVAILABLE warning fires only when a graph read was
    attempted and failed — never on every successful disabled-mode search."""
    result = await retrieve(_context(), "query", _filters(), 5, None)
    assert result.status == "ok"
    assert result.warnings == []
    assert result.graph_available is False


@pytest.mark.usefixtures("_patched_embed_only", "_fusion_flag_on")
async def test_graphiti_warning_survives_when_an_attempted_read_fails() -> None:
    result = await retrieve(
        _context(
            graphiti=_OutagedGraphiti(),
            document_versions=((_DOC, 1, None),),
            document_contracts=((_DOC, 1, (None,), True),),
        ),
        "query",
        _filters(),
        5,
        None,
    )
    assert result.status == "ok"
    assert [warning.code for warning in result.warnings] == ["GRAPHITI_UNAVAILABLE"]


# --- fusion weights ----------------------------------------------------------


@pytest.mark.usefixtures("_patched_embed_only", "_fusion_flag_on")
async def test_graphiti_leg_fuses_with_the_measured_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the graph branch registered, the fused call carries one weight per
    leg: the canonical legs keep their unweighted 1.0 and the graph leg carries
    the measured RRF_WEIGHT_GRAPHITI — never the 1.0 a None-weights fusion
    would hand it (ten times its measured contribution)."""
    from app.features.documents.domain.constants import RRF_WEIGHT_GRAPHITI

    captured: dict[str, Any] = {}

    async def _fused(**kwargs: Any) -> Any:
        captured["weights"] = kwargs["weights"]
        return Success((list(_FUSED), {row["chunk_id"]: dict(row) for row in _ROWS}))

    monkeypatch.setattr(queries_documents, "retrieve_fused", _fused)
    result = await retrieve(_context(graphiti=_OutagedGraphiti()), "query", _filters(), 5, None)
    assert result.status == "ok"
    assert captured["weights"] == [1.0, 1.0, RRF_WEIGHT_GRAPHITI]


@pytest.mark.usefixtures("_patched_fusion")
async def test_canonical_only_fusion_keeps_the_unweighted_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fused(**kwargs: Any) -> Any:
        captured["weights"] = kwargs["weights"]
        return Success((list(_FUSED), {row["chunk_id"]: dict(row) for row in _ROWS}))

    monkeypatch.setattr(queries_documents, "retrieve_fused", _fused)
    result = await retrieve(_context(), "query", _filters(), 5, None)
    assert result.status == "ok"
    assert captured["weights"] is None


# --- rerank over the candidate set (harden 5.1 ordering) ---------------------


_CANDIDATE_IDS = tuple(f"00000000-0000-0000-0000-0000000000{index:02d}" for index in range(1, 7))


class _PromotingReranker:
    """Records its input and promotes the last candidate to the top."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def rerank(self, _query: str, chunks: list[Any], *, limit: int) -> list[Any]:
        self.seen = [chunk.chunk_id for chunk in chunks]
        reordered = list(chunks)
        reordered.insert(0, reordered.pop())
        return reordered[:limit]


@pytest.fixture
def _patched_fusion_with_six_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fused order ranks six candidates; only the last one is relevant."""
    rows = [
        {**_ROWS[0], "chunk_id": chunk_id, "chunk_metadata": {"pipeline_version": "p1"}}
        for chunk_id in _CANDIDATE_IDS
    ]
    fused = [
        RankedChunk(chunk_id=chunk_id, score=1.0 - index * 0.1, rank=index + 1)
        for index, chunk_id in enumerate(_CANDIDATE_IDS)
    ]

    async def _embed(*_args: Any, **_kwargs: Any) -> list[float]:
        return [0.0]

    async def _retrieve_fused(**_kwargs: Any) -> Any:
        return Success((fused, {row["chunk_id"]: dict(row) for row in rows}))

    monkeypatch.setattr(retrieval_service, "embed_text", _embed)
    monkeypatch.setattr(queries_documents, "retrieve_fused", _retrieve_fused)


@pytest.mark.usefixtures("_patched_fusion_with_six_candidates")
async def test_reranker_sees_the_candidate_set_before_k_is_applied() -> None:
    """The reranker ranks the whole filtered candidate set, then k and the
    token budget apply — a better candidate just past the k-cut can be
    promoted (the served graph's reranker node over the same candidates)."""
    reranker = _PromotingReranker()
    result = await retrieve(_context(reranker=reranker), "query", _filters(), 2, None)
    assert reranker.seen == list(_CANDIDATE_IDS)
    assert result.status == "ok"
    assert [row["chunk_id"] for row in result.rows] == [
        _CANDIDATE_IDS[-1],
        _CANDIDATE_IDS[0],
    ]


@pytest.mark.usefixtures("_patched_fusion_with_six_candidates")
async def test_rerank_fallback_keeps_fused_order_truncated_to_k() -> None:
    """Without a working reranker the fused order stands, truncated to k."""
    result = await retrieve(_context(), "query", _filters(), 2, None)
    assert result.status == "ok"
    assert [row["chunk_id"] for row in result.rows] == list(_CANDIDATE_IDS[:2])


# --- the graph leg's own deadline -------------------------------------------


class _HangingGraphiti:
    async def search(self, **_kwargs: Any) -> list[Any]:
        await asyncio.sleep(0.5)
        return []


async def test_graphiti_leg_deadline_is_a_branch_attributed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`retrieve_fused(graphiti_timeout_seconds=...)` bounds the graph leg on
    its own: a hang past the leg deadline is the same typed, branch-attributed
    Failure a transport failure produces — not a fused-search timeout."""
    from app.features.documents.adapters.persistence.postgres.document_repository import (
        build_search_filter_params,
    )
    from app.features.documents.application.queries.documents import retrieve_fused

    fused_result = await retrieve_fused(
        repo=_Repo(),
        user_id="alice",
        tenant_id="tenant-a",
        query_text="query",
        query_embedding=[0.0],
        candidate_limit=10,
        limit=10,
        filter_params=build_search_filter_params(metadata_filter={"document_ids": [_DOC]}),
        graphiti=_HangingGraphiti(),
        document_versions=((_DOC, 1, None),),
        document_contracts=((_DOC, 1, (None,), True),),
        graphiti_timeout_seconds=0.05,
    )
    assert isinstance(fused_result, Failure)
    error = fused_result.failure()
    assert error.details.get("branch") == "graphiti"
    assert "deadline" in error.message


@pytest.mark.usefixtures("_patched_embed_only", "_fusion_flag_on")
async def test_slow_graphiti_misses_its_own_deadline_and_canonical_chunks_stand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legal-path outage contract, for the slow case: a Graphiti leg that
    misses its own deadline degrades to the canonical-only fusion — healthy
    canonical results are returned, never a fused-search timeout."""
    monkeypatch.setenv("DOCUMENTS_RETRIEVAL_POSTGRES_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("DOCUMENTS_RETRIEVAL_GRAPHITI_TIMEOUT_SECONDS", "0.05")
    get_settings.cache_clear()
    try:
        result = await retrieve(
            _context(
                graphiti=_HangingGraphiti(),
                document_versions=((_DOC, 1, None),),
                document_contracts=((_DOC, 1, (None,), True),),
            ),
            "query",
            _filters(),
            5,
            None,
        )
    finally:
        get_settings.cache_clear()
    assert result.status == "ok"
    assert [row["chunk_id"] for row in result.rows] == [_FUSED[0].chunk_id]
    assert result.graph_available is False
    assert [warning.code for warning in result.warnings] == ["GRAPHITI_UNAVAILABLE"]
