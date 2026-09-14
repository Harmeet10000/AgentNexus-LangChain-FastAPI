"""Source identifier: narrow before retrieval, widen on retry, terminate (tasks 3.1–3.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from returns.result import Success

if TYPE_CHECKING:
    from typing import Any

from app.shared.langgraph_layer.retrieval_kb.graph import build_retrieval_graph
from app.shared.langgraph_layer.retrieval_kb.nodes import (
    FALLBACK_ANSWER,
    make_hybrid_retrieval_node,
    make_source_identifier_node,
)
from app.shared.langgraph_layer.retrieval_kb.state import QueryPlan


def _plan(**overrides: Any) -> QueryPlan:
    values: dict[str, Any] = {"rewritten_query": "termination rights India"}
    values.update(overrides)
    return QueryPlan(**values)


class _CriteriaLlm:
    """Stub language model returning fixed source criteria."""

    def __init__(self, criteria: dict[str, Any]) -> None:
        self._criteria = criteria

    async def ainvoke(self, messages: Any) -> dict[str, Any]:
        return dict(self._criteria)


class _ResolvingRepo:
    """Stub repository resolving one allowlist and recording filter params."""

    def __init__(self, allowlist: list[str]) -> None:
        self._allowlist = allowlist
        self.seen_filter_params: list[dict[str, Any]] = []

    async def find_document_ids_by_metadata(self, **kwargs: Any) -> Any:
        return Success(list(self._allowlist))

    async def bm25_search(self, **kwargs: Any) -> Any:
        self.seen_filter_params.append(dict(kwargs.get("filter_params", {})))
        return Success([{"chunk_id": "c-1", "score": -5.0}])

    async def vector_search(self, **kwargs: Any) -> Any:
        return Success([{"chunk_id": "c-1", "score": 0.9}])

    async def trigram_search(self, **kwargs: Any) -> Any:
        return Success([{"chunk_id": "c-1", "score": 0.7}])

    async def fetch_chunks_by_ids(self, chunk_ids: Any) -> Any:
        return Success({
            cid: {
                "chunk_id": cid,
                "document_id": "doc-1",
                "title": "t",
                "content": "content",
                "preamble": "",
                "search_text": "content",
                "chunk_index": 0,
                "chunk_kind": "contracts",
                "clause_type": None,
                "chunk_metadata": {},
                "quality_warnings": [],
                "graphiti_verified": False,
            }
            for cid in chunk_ids
        })


def _state(plan: QueryPlan, iteration: int = 0) -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "query": "termination rights India",
        "messages": [],
        "doc_ids_filter": [],
        "query_plan": plan,
        "iteration_count": iteration,
    }


async def test_allowlist_populates_the_plan() -> None:
    node = make_source_identifier_node(
        _CriteriaLlm({"jurisdictions": ["India"], "document_kinds": [], "matters": []}),
        cast("Any", _ResolvingRepo(["doc-1", "doc-2"])),
    )

    result = await node(_state(_plan()))

    assert result["query_plan"].allowlist == ["doc-1", "doc-2"]


async def test_allowlist_reaches_branch_filter_params(monkeypatch: Any) -> None:
    async def _fake_embed(*args: Any, **kwargs: Any) -> list[float]:
        return [0.0, 0.1, 0.2]

    monkeypatch.setattr(
        "app.shared.langgraph_layer.retrieval_kb.nodes.embed_text", _fake_embed
    )
    repo = _ResolvingRepo(["doc-1", "doc-2"])
    identifier = make_source_identifier_node(
        _CriteriaLlm({"jurisdictions": ["India"], "document_kinds": [], "matters": []}),
        cast("Any", repo),
    )
    identified = await identifier(_state(_plan()))

    hybrid = make_hybrid_retrieval_node(cast("Any", repo), redis=None)
    await hybrid({**_state(_plan()), "query_plan": identified["query_plan"]})

    assert repo.seen_filter_params
    # The allowlist constrains the search itself; it is not applied to results
    # after they return.
    assert repo.seen_filter_params[0]["document_ids"] == ["doc-1", "doc-2"]


async def test_retry_widens_the_allowlist() -> None:
    node = make_source_identifier_node(
        _CriteriaLlm({"jurisdictions": ["India"], "document_kinds": [], "matters": []}),
        cast("Any", _ResolvingRepo(["doc-1"])),
    )

    result = await node(_state(_plan(allowlist=["doc-1"]), iteration=1))

    assert result["query_plan"].allowlist == []


async def test_retry_without_narrowing_is_a_noop() -> None:
    node = make_source_identifier_node(
        _CriteriaLlm({"jurisdictions": [], "document_kinds": [], "matters": []}),
        cast("Any", _ResolvingRepo([])),
    )

    result = await node(_state(_plan(), iteration=1))

    assert result["query_plan"].allowlist == []


class _SchemaLlm:
    """Stub model dispatching canned structured values per requested schema."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def with_structured_output(self, schema: Any) -> Any:
        value = self._values[schema.__name__]

        class _Bound:
            async def ainvoke(self, messages: Any) -> Any:
                return value

        return _Bound()


class _FusedReranker:
    async def rerank(self, query: str, chunks: Any, *, limit: int = 5) -> Any:
        return list(chunks)[:limit]


def _loop_graph(repo: _LoopRepo) -> Any:
    llm = _SchemaLlm(
        {
            "QueryPlan": {"rewritten_query": "termination rights India", "route": "hybrid_postgres"},
            "SourceCriteria": {"jurisdictions": ["India"]},
            "ContextGrade": {
                "sufficient": False,
                "missing_aspects": ["termination clause"],
                "rewrite_suggestion": "termination rights India agreements",
            },
            "GeneratedAnswer": {"answer": "x", "citations": [], "confidence": "uncertain"},
        }
    )
    return build_retrieval_graph(
        llm=llm, repo=repo, redis=None, graphiti=None, reranker=_FusedReranker()
    )


class _LoopRepo(_ResolvingRepo):
    def __init__(self) -> None:
        super().__init__(["doc-1"])
        self.retrieval_filters: list[list[str]] = []

    async def bm25_search(self, **kwargs: Any) -> Any:
        self.retrieval_filters.append(list(kwargs.get("filter_params", {}).get("document_ids", [])))
        return await super().bm25_search(**kwargs)


async def test_insufficient_context_widens_then_terminates(monkeypatch: Any) -> None:
    async def _fake_embed(*args: Any, **kwargs: Any) -> list[float]:
        return [0.0, 0.1, 0.2]

    monkeypatch.setattr(
        "app.shared.langgraph_layer.retrieval_kb.nodes.embed_text", _fake_embed
    )
    repo = _LoopRepo()
    graph = _loop_graph(repo)

    final = await graph.ainvoke(
        {
            "user_id": "user-1",
            "query": "termination rights India",
            "messages": [],
            "doc_ids_filter": [],
            "iteration_count": 0,
        }
    )

    # The first retrieval runs narrowed; the retry runs widened (empty).
    assert repo.retrieval_filters[0] == ["doc-1"]
    assert repo.retrieval_filters[-1] == []
    # Widening does not extend the cap: the loop stops at two iterations with
    # the grounded fallback.
    assert final["iteration_count"] == 2
    assert final["generated_answer"].answer == FALLBACK_ANSWER
