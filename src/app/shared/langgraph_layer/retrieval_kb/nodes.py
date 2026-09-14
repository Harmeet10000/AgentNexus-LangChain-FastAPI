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

from .reranker import get_configured_reranker
from .state import (  # noqa: TC001 — RetrievalState resolves at runtime for branch schemas
    ContextGrade,
    GeneratedAnswer,
    QueryPlan,
    RetrievalState,
    RetrievedChunk,
    SourceCriteria,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from redis.asyncio import Redis

    from app.features.documents.repository import DocumentRepository
    from app.shared.rag.token_counter import CountTokens

    from .reranker import Reranker

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
            logger.bind(query=query[:80], operation="query_analyzer", error=str(exc)).warning(
                "query_analyzer_failed_using_default"
            )
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


_SOURCE_IDENTIFIER_SYSTEM_PROMPT = render_prompt_sections(
    ("IDENTITY", "You are a corpus-narrowing source identifier."),
    (
        "OBJECTIVE",
        "Extract the cheap-metadata narrowing the query names — jurisdiction, document kind, "
        "matter — so retrieval searches a smaller corpus. Name only what the query states; "
        "when it names nothing, return empty lists.",
    ),
    (
        "FEW-SHOT EXAMPLES",
        "Query: 'What termination rights survive under our India services agreements?' "
        "-> {jurisdictions: ['India'], document_kinds: ['contracts'], matters: []}. "
        "Query: 'What does section 73 provide about compensation for breach?' "
        "-> {jurisdictions: [], document_kinds: ['statutes'], matters: []}. "
        "Query: 'Summarise the key risks in this filing.' "
        "-> {jurisdictions: [], document_kinds: [], matters: []}.",
    ),
    ("CONSTRAINTS", "Return only SourceCriteria."),
)

#: Upper bound on a resolved allowlist. More identifiers than this narrow
#: nothing and only bloat the filter parameters, so an over-wide resolution is
#: treated as "cannot narrow" and retrieval proceeds unconstrained.
SOURCE_ALLOWLIST_MAX_IDS = 100


def _intersect_doc_ids(*, request_ids: list[str], allowlist: list[str]) -> list[str]:
    """Combine the request document filter with the identifier allowlist."""
    if not allowlist:
        return request_ids
    if not request_ids:
        return allowlist
    narrowed = [doc_id for doc_id in allowlist if doc_id in request_ids]
    return narrowed or request_ids


def make_source_identifier_node(
    identifier_llm: Any,
    repo: DocumentRepository,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    """Narrow the corpus to a document allowlist before retrieval runs.

    On a retry iteration the previous narrowing is dropped instead: a wrong
    allowlist that excluded the answer is unrecoverable any other way, and the
    iteration cap (not the allowlist) is what bounds the loop.
    """

    async def source_identifier_node(state: RetrievalState) -> dict[str, object]:
        plan: QueryPlan = state["query_plan"]
        if state.get("iteration_count", 0) > 0:
            if not plan.allowlist:
                return {"query_plan": plan}
            logger.bind(operation="source_identifier", iteration=state.get("iteration_count")).info(
                "source_allowlist_widened_on_retry"
            )
            return {"query_plan": plan.model_copy(update={"allowlist": []})}

        query = plan.rewritten_query
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=_SOURCE_IDENTIFIER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        try:
            raw_criteria = await retry_immediate(
                lambda: identifier_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_source_identifier",
            )
            criteria = SourceCriteria.model_validate(raw_criteria)
        except Exception as exc:  # noqa: BLE001 — cannot narrow means unconstrained, not failed
            exc.add_note(f"query={query[:80]}, operation=source_identifier")
            logger.bind(query=query[:80], operation="source_identifier", error=str(exc)).warning(
                "source_identifier_failed_proceeding_unconstrained"
            )
            return {"query_plan": plan}

        if not (criteria.jurisdictions or criteria.document_kinds or criteria.matters):
            return {"query_plan": plan}
        resolved = await repo.find_document_ids_by_metadata(
            user_id=state["user_id"],
            jurisdictions=criteria.jurisdictions,
            document_kinds=criteria.document_kinds,
            matters=criteria.matters,
            limit=SOURCE_ALLOWLIST_MAX_IDS + 1,
        )
        if isinstance(resolved, Failure):
            error = resolved.failure()
            log_expected_failure(error, operation="source_allowlist_resolve")
            return {"query_plan": plan}
        allowlist = resolved.unwrap()
        if len(allowlist) > SOURCE_ALLOWLIST_MAX_IDS:
            return {"query_plan": plan}
        return {"query_plan": plan.model_copy(update={"allowlist": allowlist})}

    return source_identifier_node


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
            logger.bind(
                query=plan.rewritten_query[:80], operation="graph_retrieval", error=str(exc)
            ).warning("graph_retrieval_failed")
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
        # Local imports (noqa: PLC0415): `documents.service` imports this package at
        # module load, so top-level imports would close a cycle. The fused
        # retrieval helper lives with the other retrieval-region policy in the
        # service module, and this node reaches the same shared path as `ask`.
        from app.features.documents.constants import (  # noqa: PLC0415
            HYBRID_CANDIDATE_LIMIT,
            RRF_WEIGHT_TRIGRAM,
        )
        from app.features.documents.repository import (  # noqa: PLC0415
            build_search_filter_params,
        )
        from app.features.documents.service import (  # noqa: PLC0415
            lookup_to_rows,
            retrieve_fused,
        )

        # `user_id` is not a translation of an old argument — the reader this replaces had no
        # tenant predicate at all, so every fused search read across all owners and was held
        # back only by the caller never passing another user's chunk ids. The unified query
        # scopes on the parent document's owner, which is why the state field is required here
        # rather than optional.
        filter_params = build_search_filter_params(
            metadata_filter={
                # The request filter and the identifier allowlist both constrain:
                # intersect when both name documents, otherwise take whichever
                # is present. A conflicting inference loses to the explicit
                # request filter rather than silently emptying retrieval.
                "document_ids": _intersect_doc_ids(
                    request_ids=list(state.get("doc_ids_filter") or []),
                    allowlist=list(plan.allowlist or []),
                ),
                "chunk_ids": list(chunk_ids or []),
                "jurisdiction": plan.jurisdiction,
                "contract_type": plan.contract_type,
            }
        )
        rows_result = await retry_immediate(
            lambda: retrieve_fused(
                repo=repo,
                user_id=state["user_id"],
                query_text=plan.rewritten_query,
                query_embedding=embedding,
                candidate_limit=HYBRID_CANDIDATE_LIMIT,
                limit=20,
                filter_params=filter_params,
                weights=[plan.keyword_weight, plan.vector_weight, RRF_WEIGHT_TRIGRAM],
                bm25_threshold=plan.bm25_threshold,
                exact_phrase=plan.exact_phrase,
            ),
            label="postgres_fused_search",
        )
        if isinstance(rows_result, Failure):
            error = rows_result.failure()
            log_expected_failure(error, operation="postgres_fused_search")
            raise InfrastructureException(
                detail=error.message,
                error_code=error.code,
                retryable=error.retryable,
                data=error.details,
            )
        fused, lookup = rows_result.unwrap()
        rows = lookup_to_rows(fused, lookup)
        return {"retrieved_chunks": [_row_to_chunk(row) for row in rows]}

    return hybrid_retrieval_node


def make_reranker_node(
    reranker: Reranker | None = None,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    resolved: Reranker = reranker or get_configured_reranker()

    async def reranker_node(state: RetrievalState) -> dict[str, object]:
        plan: QueryPlan = state["query_plan"]
        chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])[:20]
        reranked: list[RetrievedChunk] = await resolved.rerank(
            plan.rewritten_query, chunks, limit=5
        )
        return {"reranked_chunks": reranked}

    return reranker_node


def make_post_process_node(
    *,
    max_tokens: int,
    count_tokens: CountTokens,
) -> Callable[[RetrievalState], Awaitable[dict[str, object]]]:
    """Deduplicate reranked chunks and assemble document-ordered context.

    A chunk returned by two branches reaches generation once, and chunks of
    one document read in ascending in-document order rather than relevance
    order. Assembly reuses `assemble_rag_context` — no second implementation.
    Both the reordered chunk list (what the grader and generator read) and the
    assembled sections are returned so the two stay consistent.
    """

    async def post_process_node(state: RetrievalState) -> dict[str, object]:
        # Local imports (noqa: PLC0415): `documents` modules import this
        # package at module load, so top-level imports would close a cycle.
        from app.features.documents.fusion import RankedChunk  # noqa: PLC0415
        from app.features.documents.rag import (  # noqa: PLC0415
            SearchChunkRecord,
            assemble_rag_context,
        )

        chunks: list[RetrievedChunk] = state.get("reranked_chunks", [])
        unique: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            unique.setdefault(chunk.chunk_id, chunk)
        grouped: dict[str, list[RetrievedChunk]] = {}
        for chunk in unique.values():
            grouped.setdefault(chunk.parent_doc_id, []).append(chunk)
        ordered: list[RetrievedChunk] = [
            chunk
            for document_chunks in grouped.values()
            for chunk in sorted(document_chunks, key=lambda item: item.chunk_index)
        ]
        ranked = [
            RankedChunk(chunk_id=chunk.chunk_id, score=chunk.score, rank=index)
            for index, chunk in enumerate(ordered, start=1)
        ]
        lookup = {
            chunk.chunk_id: SearchChunkRecord(
                document_id=chunk.parent_doc_id,
                title="",
                content=chunk.chunk_text,
                chunk_index=chunk.chunk_index,
                chunk_metadata=dict(chunk.metadata_),
            )
            for chunk in ordered
        }
        sections = assemble_rag_context(
            ranked, lookup, max_tokens=max_tokens, count_tokens=count_tokens
        )
        return {"reranked_chunks": ordered, "assembled_context": sections}

    return post_process_node


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
            logger.bind(operation="context_grader", error=str(exc)).warning(
                "context_grader_failed_using_chunk_presence"
            )
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
            logger.bind(operation="generator", error=str(exc)).warning(
                "generator_failed_using_fallback"
            )
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
    chunk_index = row.get("chunk_index")
    return RetrievedChunk(
        chunk_id=str(row["chunk_id"]),
        chunk_text=str(row["chunk_text"]),
        preamble=str(row["preamble"] or ""),
        clause_type=str(row["clause_type"]),
        parent_doc_id=str(row["parent_doc_id"]),
        chunk_index=int(chunk_index) if chunk_index is not None else 0,
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
