"""Nodes for clauses-backed legal retrieval graph."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.shared.langchain_layer.models import serialize_to_toon
from app.shared.langgraph_layer.kb_retry import retry_immediate
from app.utils import logger

from .reranker import CrossEncoderReranker
from .state import ContextGrade, GeneratedAnswer, QueryPlan, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from redis.asyncio import Redis

    from app.features.search.repository import SearchRepository

    from .state import RetrievalState

EmbeddingFunction = Any

QUERY_ANALYZER_PROMPT = """
Analyze the legal retrieval query. Rewrite coreferences using the conversation,
decompose multi-part questions, choose route hybrid_postgres, graph_neo4j, or both,
and choose vector_weight/keyword_weight. Exact clause-reference queries should
favor BM25; conceptual obligation/risk questions should favor vector search.
Return only QueryPlan.
"""

CONTEXT_GRADER_PROMPT = """
Grade whether the retrieved chunks are sufficient to answer the query without
hallucination. If not sufficient, provide missing_aspects and a concise
rewrite_suggestion. Return only ContextGrade.
"""

GENERATOR_PROMPT = """
Answer using only retrieved chunks. Every factual claim must cite exact chunk_id
and clause_type in the citations list. If support is weak, set confidence to
uncertain. Return only GeneratedAnswer.
"""

FALLBACK_ANSWER = (
    "I do not have enough grounded contract context to answer this reliably. "
    "Please narrow the question or ingest the relevant document sections."
)


def make_query_analyzer_node(
    query_llm: Any,
    redis: Redis | None,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    async def query_analyzer_node(state: RetrievalState) -> dict[str, object]:
        query = state["query"]
        plan_input = serialize_to_toon(
            {
                "query": query,
                "messages": [str(message.content) for message in state.get("messages", [])[-8:]],
                "doc_ids_filter": state.get("doc_ids_filter", []),
                "iteration_count": state.get("iteration_count", 0),
                "rewrite_suggestion": getattr(
                    state.get("context_grade"), "rewrite_suggestion", None
                ),
            }
        )
        messages = [SystemMessage(content=QUERY_ANALYZER_PROMPT), HumanMessage(content=plan_input)]
        try:
            raw_plan = await retry_immediate(
                lambda: query_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_query_analyzer",
            )
            plan = _normalize_plan(QueryPlan.model_validate(raw_plan))
        except Exception as exc:  # noqa: BLE001 - default plan keeps retrieval available.
            logger.bind(error=str(exc)).warning("query_analyzer_failed_using_default")
            plan = QueryPlan(rewritten_query=query, sub_queries=[query])

        cache_key = _answer_cache_key(plan.rewritten_query, state.get("doc_ids_filter", []))
        if redis is not None:
            cached = await redis.get(cache_key)
            if cached:
                raw = cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)
                return {
                    "query_plan": plan,
                    "cache_hit": True,
                    "cached_answer": GeneratedAnswer.model_validate_json(raw),
                }

        return {"query_plan": plan, "cache_hit": False}

    return query_analyzer_node


def make_graph_retrieval_node(
    graphiti: Any,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    async def graph_retrieval_node(state: RetrievalState) -> dict[str, object]:
        if graphiti is None:
            return {"graph_chunk_ids": []}
        plan = state["query_plan"]
        try:
            raw_results = await retry_immediate(
                lambda: graphiti.search(
                    query=plan.rewritten_query,
                    group_ids=[state["user_id"], *state.get("doc_ids_filter", [])],
                    num_results=20,
                ),
                label="graphiti_retrieval_search",
            )
        except Exception as exc:  # noqa: BLE001 - graph route is optional enrichment.
            logger.bind(error=str(exc)).warning("graph_retrieval_failed")
            return {"graph_chunk_ids": []}

        chunk_ids: list[str] = []
        for result in raw_results or []:
            source_description = getattr(result, "source_description", "") or ""
            content = getattr(result, "content", "") or getattr(result, "episode_body", "") or ""
            chunk_ids.extend(_extract_postgres_chunk_ids(source_description))
            chunk_ids.extend(_extract_postgres_chunk_ids(content))
        return {"graph_chunk_ids": sorted(set(chunk_ids))}

    return graph_retrieval_node


def make_hybrid_retrieval_node(
    repo: SearchRepository,
    embedding_fn: EmbeddingFunction,
    redis: Redis | None,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    async def hybrid_retrieval_node(state: RetrievalState) -> dict[str, object]:
        plan = state["query_plan"]
        embedding = await _cached_embedding(redis, embedding_fn, plan.rewritten_query)
        chunk_ids = state.get("graph_chunk_ids") or None
        rows = await retry_immediate(
            lambda: repo.legal_rrf_search(
                query_text=plan.rewritten_query,
                query_embedding=embedding,
                limit=20,
                vector_weight=plan.vector_weight,
                keyword_weight=plan.keyword_weight,
                jurisdiction=plan.jurisdiction,
                contract_type=plan.contract_type,
                chunk_ids=chunk_ids,
                bm25_threshold=plan.bm25_threshold,
                exact_phrase=plan.exact_phrase,
            ),
            label="postgres_legal_rrf_search",
        )
        return {"retrieved_chunks": [_row_to_chunk(row) for row in rows]}

    return hybrid_retrieval_node


def make_reranker_node(
    reranker: CrossEncoderReranker | None = None,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    resolved = reranker or CrossEncoderReranker()

    async def reranker_node(state: RetrievalState) -> dict[str, object]:
        plan = state["query_plan"]
        chunks = state.get("retrieved_chunks", [])[:20]
        reranked = await resolved.rerank(plan.rewritten_query, chunks, limit=5)
        return {"reranked_chunks": reranked}

    return reranker_node


def make_context_grader_node(
    grader_llm: Any,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    async def context_grader_node(state: RetrievalState) -> dict[str, object]:
        plan = state["query_plan"]
        chunks = state.get("reranked_chunks", [])
        payload = serialize_to_toon(
            {
                "query": plan.rewritten_query,
                "chunks": [chunk.model_dump() for chunk in chunks],
            }
        )
        messages = [SystemMessage(content=CONTEXT_GRADER_PROMPT), HumanMessage(content=payload)]
        if not chunks:
            grade = ContextGrade(
                sufficient=False,
                missing_aspects=["No retrieved chunks"],
                rewrite_suggestion=plan.rewritten_query,
            )
            return {"context_grade": grade, "iteration_count": state.get("iteration_count", 0) + 1}
        try:
            raw_grade = await retry_immediate(
                lambda: grader_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_context_grader",
            )
            grade = ContextGrade.model_validate(raw_grade)
        except Exception as exc:  # noqa: BLE001 - fall back to chunk-presence heuristic.
            logger.bind(error=str(exc)).warning("context_grader_failed_using_chunk_presence")
            grade = ContextGrade(sufficient=bool(chunks), missing_aspects=[])
        return {"context_grade": grade, "iteration_count": state.get("iteration_count", 0) + 1}

    return context_grader_node


def make_generator_node(
    generator_llm: Any,
    redis: Redis | None,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    async def generator_node(state: RetrievalState) -> dict[str, object]:
        if state.get("cache_hit") and state.get("cached_answer"):
            return {
                "generated_answer": state["cached_answer"],
                "messages": [AIMessage(content=state["cached_answer"].answer)],
            }

        grade = state.get("context_grade")
        if grade is not None and not grade.sufficient and state.get("iteration_count", 0) >= 2:
            answer = GeneratedAnswer(answer=FALLBACK_ANSWER, citations=[], confidence="uncertain")
            return {"generated_answer": answer, "messages": [AIMessage(content=answer.answer)]}

        plan = state["query_plan"]
        chunks = state.get("reranked_chunks", [])
        payload = serialize_to_toon(
            {
                "query": plan.rewritten_query,
                "chunks": [chunk.model_dump() for chunk in chunks],
            }
        )
        messages = [SystemMessage(content=GENERATOR_PROMPT), HumanMessage(content=payload)]
        try:
            raw_answer = await retry_immediate(
                lambda: generator_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_grounded_generator",
            )
            answer = GeneratedAnswer.model_validate(raw_answer)
        except Exception as exc:  # noqa: BLE001 - generator failure must return hard fallback.
            logger.bind(error=str(exc)).warning("generator_failed_using_fallback")
            answer = GeneratedAnswer(answer=FALLBACK_ANSWER, citations=[], confidence="uncertain")

        if answer.confidence == "uncertain" and FALLBACK_ANSWER not in answer.answer:
            answer = answer.model_copy(update={"answer": f"{answer.answer}\n\n{FALLBACK_ANSWER}"})

        if redis is not None:
            cache_key = _answer_cache_key(plan.rewritten_query, state.get("doc_ids_filter", []))
            ttl = 60 * 60 * 24 if state.get("doc_ids_filter") else 60 * 60
            await redis.setex(cache_key, ttl, answer.model_dump_json())

        return {"generated_answer": answer, "messages": [AIMessage(content=answer.answer)]}

    return generator_node


def should_run_graph(state: RetrievalState) -> str:
    if state.get("cache_hit"):
        return "generate"
    route = state["query_plan"].route
    return "graph" if route in {"graph_neo4j", "both"} else "hybrid"


def should_retry_or_generate(state: RetrievalState) -> str:
    grade = state.get("context_grade")
    if grade is not None and not grade.sufficient and state.get("iteration_count", 0) < 2:
        return "query_analyzer"
    return "generate"


def _normalize_plan(plan: QueryPlan) -> QueryPlan:
    vector_weight = plan.vector_weight
    keyword_weight = plan.keyword_weight
    total = vector_weight + keyword_weight
    if total <= 0:
        vector_weight = 0.4
        keyword_weight = 0.6
    else:
        vector_weight = vector_weight / total
        keyword_weight = keyword_weight / total
    if plan.query_type == "exact":
        vector_weight = 0.25
        keyword_weight = 0.75
    if plan.query_type == "conceptual":
        vector_weight = 0.65
        keyword_weight = 0.35
    return plan.model_copy(
        update={"vector_weight": vector_weight, "keyword_weight": keyword_weight}
    )


async def _cached_embedding(
    redis: Redis | None,
    embedding_fn: EmbeddingFunction,
    text_to_embed: str,
) -> list[float]:
    key = "kb:embedding:" + hashlib.sha256(text_to_embed.encode("utf-8")).hexdigest()
    if redis is not None:
        cached = await redis.get(key)
        if cached:
            raw = cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)
            return cast("list[float]", json.loads(raw))
    embedding = await retry_immediate(
        lambda: _call_embedding_fn(embedding_fn, text_to_embed),
        label="gemini_query_embedding",
    )
    embedding = _normalize_embedding(embedding)
    if redis is not None:
        await redis.setex(key, 60 * 60 * 24, json.dumps(embedding))
    return embedding


async def _call_embedding_fn(embedding_fn: EmbeddingFunction, text_to_embed: str) -> list[float]:
    if hasattr(embedding_fn, "aembed_query"):
        return cast("list[float]", await embedding_fn.aembed_query(text_to_embed))
    if hasattr(embedding_fn, "ainvoke"):
        return cast("list[float]", await embedding_fn.ainvoke(text_to_embed))
    result = embedding_fn(text_to_embed)
    if hasattr(result, "__await__"):
        return cast("list[float]", await result)
    return cast("list[float]", result)


def _normalize_embedding(embedding: list[float]) -> list[float]:
    if len(embedding) == 768:
        return embedding
    if len(embedding) > 768:
        return embedding[:768]
    return [*embedding, *([0.0] * (768 - len(embedding)))]


def _row_to_chunk(row: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(row["chunk_id"]),
        chunk_text=str(row["chunk_text"]),
        preamble=str(row["preamble"] or ""),
        clause_type=str(row["clause_type"]),
        parent_doc_id=str(row["parent_doc_id"]),
        metadata_=dict(row["metadata_"] or {}),
        custom_metadata=dict(row["custom_metadata"] or {}),
        score=float(row["rrf_score"]),
    )


def _answer_cache_key(rewritten_query: str, doc_ids_filter: list[str]) -> str:
    raw = json.dumps(
        {"query": " ".join(rewritten_query.lower().split()), "doc_ids": sorted(doc_ids_filter)},
        sort_keys=True,
    )
    return "kb:answer:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_postgres_chunk_ids(value: str) -> list[str]:
    return re.findall(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        value,
    )
