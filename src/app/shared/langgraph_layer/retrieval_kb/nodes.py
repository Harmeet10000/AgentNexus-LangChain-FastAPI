"""Nodes for the unified-chunk-backed legal retrieval graph."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, cast

from graphiti_core.errors import GraphitiError
from langchain_core.exceptions import LangChainException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from returns.result import Failure

from app.shared.langchain_layer import render_prompt_sections, serialize_to_toon
from app.shared.langchain_layer.embeddings import EmbeddingTaskType, embed_text
from app.shared.langgraph_layer.kb_retry import TransientExternalError, retry_immediate
from app.shared.result import log_expected_failure
from app.utils import InfrastructureException, logger

from .reranker import get_shared_reranker
from .state import ContextGrade, GeneratedAnswer, QueryPlan, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from redis.asyncio import Redis

    from app.features.documents.repository import DocumentRepository

    from .reranker import CrossEncoderReranker
    from .state import RetrievalState

_QUERY_ANALYZER_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a legal retrieval query planning engine."),
    (
        "OBJECTIVE",
        "Analyze the legal retrieval query and produce a QueryPlan that maximizes grounded retrieval.",
    ),
    (
        "CONTEXT POLICY",
        "Use recent conversation messages only to resolve references and clarify retrieval intent.",
    ),
    (
        "EXECUTION POLICY",
        "Rewrite coreferences using the conversation, decompose multi-part questions, choose route "
        "hybrid_postgres, graph_neo4j, or both, and choose vector_weight and keyword_weight. "
        "Exact clause-reference queries should favor BM25; conceptual obligation or risk questions should favor vector search.",
    ),
    ("CONSTRAINTS", "Return only QueryPlan."),
)

_CONTEXT_GRADER_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a retrieval sufficiency grader."),
    (
        "OBJECTIVE",
        "Determine whether the retrieved chunks are sufficient to answer the query without hallucination.",
    ),
    (
        "EXECUTION POLICY",
        "If the evidence is insufficient, identify missing aspects and provide a concise rewrite suggestion.",
    ),
    ("CONSTRAINTS", "Return only ContextGrade."),
)

_GENERATOR_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a grounded legal answer generator."),
    (
        "OBJECTIVE",
        "Answer the user's question using only the retrieved chunks.",
    ),
    (
        "CONTEXT POLICY",
        "Treat retrieved chunks as the only admissible basis for factual claims.",
    ),
    (
        "CONSTRAINTS",
        "Every factual claim must cite exact chunk_id and clause_type in the citations list. "
        "Return only GeneratedAnswer.",
    ),
    (
        "UNCERTAINTY POLICY",
        "If support is weak, set confidence to uncertain and avoid stronger claims than the evidence supports.",
    ),
)

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
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=_QUERY_ANALYZER_SYSTEM_PROMPT),
            HumanMessage(content=plan_input),
        ]
        try:
            raw_plan = await retry_immediate(
                lambda: query_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_query_analyzer",
            )
            plan = _normalize_plan(QueryPlan.model_validate(raw_plan))
        # Both routes into this branch (the C6 contract): a deterministic
        # framework failure arrives unwrapped; a transient one arrives retried
        # and, once the budget is spent, as the boundary's transient type with
        # the original reachable through its cause. Catching only the first
        # left this degradation branch dead in production.
        except (LangChainException, TransientExternalError) as exc:
            exc.add_note(f"query={query[:80]}, operation=query_analyzer")
            logger.bind(error=str(exc)).warning("query_analyzer_failed_using_default")
            plan = QueryPlan(rewritten_query=query, sub_queries=[query])

        cache_key = _answer_cache_key(plan.rewritten_query, state.get("doc_ids_filter", []))
        if redis is not None:
            cached = await redis.get(cache_key)
            if cached:
                raw = str(cached)
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
        # Same C6 contract as above: a retry-exhausted transient failure
        # arrives as the boundary's type, not as `GraphitiError`.
        except (GraphitiError, TransientExternalError) as exc:
            exc.add_note(f"query={plan.rewritten_query[:80]}, operation=graph_retrieval")
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
    repo: DocumentRepository,
    redis: Redis | None,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    async def hybrid_retrieval_node(state: RetrievalState) -> dict[str, object]:
        plan = state["query_plan"]
        # `QUERY`, not `DOCUMENT`. This is the side of the asymmetry that was never declared:
        # the prior helper passed no task type at all, so a query vector was drawn from the
        # document projection and compared against stored vectors drawn from the same one.
        # Mutually consistent, and both wrong — which is why nothing ever errored.
        embedding = await retry_immediate(
            lambda: embed_text(
                plan.rewritten_query,
                task_type=EmbeddingTaskType.QUERY,
                redis=redis,
            ),
            label="gemini_query_embedding",
        )
        chunk_ids = state.get("graph_chunk_ids") or None
        # `user_id` is not a translation of an old argument — the reader this replaces had no
        # tenant predicate at all, so every fused search read across all owners and was held
        # back only by the caller never passing another user's chunk ids. The unified query
        # scopes on the parent document's owner, which is why the state field is required here
        # rather than optional.
        rows_result = await retry_immediate(
            lambda: repo.legal_rrf_search(
                user_id=state["user_id"],
                query_text=plan.rewritten_query,
                query_embedding=embedding,
                limit=20,
                vector_weight=plan.vector_weight,
                keyword_weight=plan.keyword_weight,
                jurisdiction=plan.jurisdiction,
                contract_type=plan.contract_type,
                # The one filter the old reader accepted from the request and then dropped on
                # the floor: `doc_ids_filter` reached the analyzer prompt, the Graphiti group
                # ids and the answer cache key, but never the SQL.
                document_ids=state.get("doc_ids_filter") or None,
                chunk_ids=chunk_ids,
                # Explicit, not defaulted: `QueryPlan` forbids extra fields, so the graph has
                # nowhere to carry either of these. Passing them by name records that the
                # omission is a property of the plan object, not an oversight here.
                clause_type=None,
                require_graphiti_verified=False,
                bm25_threshold=plan.bm25_threshold,
                exact_phrase=plan.exact_phrase,
            ),
            label="postgres_legal_rrf_search",
        )
        if isinstance(rows_result, Failure):
            error = rows_result.failure()
            log_expected_failure(error, operation="postgres_legal_rrf_search")
            raise InfrastructureException(
                detail=error.message,
                error_code=error.code,
                retryable=error.retryable,
                data=error.details,
            )
        rows = rows_result.unwrap()
        return {"retrieved_chunks": [_row_to_chunk(row) for row in rows]}

    return hybrid_retrieval_node


def make_reranker_node(
    reranker: CrossEncoderReranker | None = None,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    resolved: CrossEncoderReranker = reranker or get_shared_reranker()

    async def reranker_node(state: RetrievalState) -> dict[str, object]:
        plan: QueryPlan = state["query_plan"]
        chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])[:20]
        reranked: list[RetrievedChunk] = await resolved.rerank(
            plan.rewritten_query, chunks, limit=5
        )
        return {"reranked_chunks": reranked}

    return reranker_node


def make_context_grader_node(
    grader_llm: Any,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    async def context_grader_node(state: RetrievalState) -> dict[str, object]:
        plan: QueryPlan = state["query_plan"]
        chunks: list[RetrievedChunk] = state.get("reranked_chunks", [])
        payload: str = serialize_to_toon(
            {
                "query": plan.rewritten_query,
                "chunks": [chunk.model_dump() for chunk in chunks],
            }
        )
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=_CONTEXT_GRADER_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
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
            grade: ContextGrade = ContextGrade.model_validate(raw_grade)
        except Exception as exc:  # noqa: BLE001 — fall back to chunk-presence heuristic
            exc.add_note("operation=context_grader")
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

        grade: ContextGrade | None = state.get("context_grade")
        if grade is not None and not grade.sufficient and state.get("iteration_count", 0) >= 2:
            answer = GeneratedAnswer(answer=FALLBACK_ANSWER, citations=[], confidence="uncertain")
            return {"generated_answer": answer, "messages": [AIMessage(content=answer.answer)]}

        plan: QueryPlan = state["query_plan"]
        chunks: list[RetrievedChunk] = state.get("reranked_chunks", [])
        payload = serialize_to_toon(
            {
                "query": plan.rewritten_query,
                "chunks": [chunk.model_dump() for chunk in chunks],
            }
        )
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
        try:
            raw_answer = await retry_immediate(
                lambda: generator_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_grounded_generator",
            )
            answer: GeneratedAnswer = GeneratedAnswer.model_validate(raw_answer)
        except Exception as exc:  # noqa: BLE001 — generator failure must return hard fallback
            exc.add_note("operation=generator")
            logger.bind(error=str(exc)).warning("generator_failed_using_fallback")
            answer = GeneratedAnswer(answer=FALLBACK_ANSWER, citations=[], confidence="uncertain")

        if answer.confidence == "uncertain" and FALLBACK_ANSWER not in answer.answer:
            answer: GeneratedAnswer = answer.model_copy(
                update={"answer": f"{answer.answer}\n\n{FALLBACK_ANSWER}"}
            )

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
        vector_weight /= total
        keyword_weight /= total
    if plan.query_type == "exact":
        vector_weight = 0.25
        keyword_weight = 0.75
    if plan.query_type == "conceptual":
        vector_weight = 0.65
        keyword_weight = 0.35
    return plan.model_copy(
        update={"vector_weight": vector_weight, "keyword_weight": keyword_weight}
    )


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
