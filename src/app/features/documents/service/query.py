"""Unified document feature services."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, NamedTuple, cast

from langchain_core.messages import HumanMessage, SystemMessage
from returns.result import Failure, Success

from app.ai.embeddings import EmbeddingTaskType, embed_text
from app.ai.graph.retry import retry_immediate
from app.config import get_settings
from app.features.documents.constants import (
    DEFAULT_SEARCH_CACHE_TTL_SECONDS,
    HYBRID_CANDIDATE_LIMIT,
    RRF_K,
    RRF_WEIGHT_TRIGRAM,
)
from app.features.documents.dto import (
    DocumentSearchResultItem,
    LegalCitationResponse,
    QualityWarningDTO,
    RagContextSectionResponse,
    UnifiedAskResponse,
    UnifiedRagResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from app.features.documents.rag import SearchChunkRecord, assemble_rag_context
from app.features.documents.repository import (
    DocumentRepository,
    build_search_filter_params,
)
from app.features.documents.retrieval_graph import (
    ContextGrade,
    GeneratedAnswer,
    QueryPlan,
    RetrievedChunk,
    _extract_postgres_chunk_ids,
    build_retrieval_graph,
    get_shared_reranker,
)
from app.observability.logging import trace_layer
from app.retrieval.fusion import RankedChunk, RankedResultRow, reciprocal_rank_fusion
from app.shared.langchain_layer import serialize_to_toon
from app.shared.rag.structural_navigator import navigate_tree
from app.shared.rag.token_counter import count_tokens as default_count_tokens
from app.shared.result import log_expected_failure
from app.utils import to_sorted_key_bytes

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import Any

    from graphiti_core.graphiti import Graphiti
    from langchain_core.language_models import BaseChatModel
    from redis.asyncio import Redis
    from ty_extensions import Unknown

    from app.config.settings import Settings
    from app.features.documents import dto as documents_dto
    from app.features.documents.rag import ContextSection

    from app.features.documents.classification import QualityWarning
    from app.features.documents.exceptions import DocumentResult
    from app.features.documents.legal_metadata import (
        LegalMetadataExtraction,
    )
_GENERATOR_SYSTEM_PROMPT = (
    "You are a grounded answer generator. Use only the provided chunks. "
    "Every factual claim must cite exact chunk_id and clause_type in the citations list. "
    "Return only GeneratedAnswer."
)
_CONTEXT_GRADER_SYSTEM_PROMPT = "You are a retrieval sufficiency grader. Return only ContextGrade."
_QUERY_ANALYZER_SYSTEM_PROMPT = "You are a legal retrieval query planner. Return only QueryPlan."
_FALLBACK_ANSWER = (
    "I do not have enough grounded document context to answer this reliably. "
    "Please narrow the question or ingest the relevant document sections."
)
_KNOWLEDGE_EXTRACTION_PROMPT = (
    "Extract grounded legal clauses and their attributes. Use extraction classes such as "
    "indemnity, termination, payment, confidentiality, governing_law, arbitration, "
    "limitation_of_liability, ip_ownership, or obliges. Preserve source character offsets."
)


# Retrieval branch registry. Each branch is a (name, run) record: `name` is the
# attribution label a failure carries, and `run` builds that branch's coroutine
# from one shared input. The pairing between names and `asyncio.gather`'s
# positional results used to live in the `_SEARCH_BRANCHES` tuple alone; a
# reorder there silently misattributed every failure, so the name now travels
# with the callable and the zip in `_run_branches` pairs records with
# results. Adding branch #4 is a registration below — the gather/zip/fusion
# does not change. Per-leg fusion weights travel on `retrieve_fused`'s `weights`
# argument (one weight per leg, defaulting to unweighted); the query plan's
# vector and keyword weights map onto the first two legs at the call sites.


class _BranchInput(NamedTuple):
    """Shared input every fused-retrieval branch runs from."""

    user_id: str
    query_text: str
    query_embedding: list[float]
    candidate_limit: int
    filter_params: dict[str, Any]
    bm25_threshold: float | None = None
    exact_phrase: str | None = None


class _LegalPreparation(NamedTuple):
    metadata: LegalMetadataExtraction | None
    warnings: list[QualityWarning]


async def _run_bm25_branch(
    repo: DocumentRepository, args: _BranchInput
) -> DocumentResult[list[dict[str, Any]]]:
    """Keyword branch coroutine."""
    return await repo.bm25_search(
        user_id=args.user_id,
        query=args.query_text,
        candidate_limit=args.candidate_limit,
        filter_params=args.filter_params,
        bm25_threshold=args.bm25_threshold,
        exact_phrase=args.exact_phrase,
    )


async def _run_vector_branch(
    repo: DocumentRepository, args: _BranchInput
) -> DocumentResult[list[dict[str, Any]]]:
    """Dense-vector branch coroutine."""
    return await repo.vector_search(
        user_id=args.user_id,
        embedding=args.query_embedding,
        candidate_limit=args.candidate_limit,
        filter_params=args.filter_params,
    )


async def _run_trigram_branch(
    repo: DocumentRepository, args: _BranchInput
) -> DocumentResult[list[dict[str, Any]]]:
    """Fuzzy-trigram branch coroutine."""
    return await repo.trigram_search(
        user_id=args.user_id,
        query=args.query_text,
        candidate_limit=args.candidate_limit,
        filter_params=args.filter_params,
    )


class RetrievalBranchPolicy(NamedTuple):
    """One fused-retrieval branch: attribution name plus how to run it."""

    name: str
    run: Callable[
        [DocumentRepository, _BranchInput], Awaitable[DocumentResult[list[dict[str, Any]]]]
    ]


RETRIEVAL_BRANCHES: tuple[RetrievalBranchPolicy, ...] = (
    RetrievalBranchPolicy(name="bm25", run=_run_bm25_branch),
    RetrievalBranchPolicy(name="vector", run=_run_vector_branch),
    RetrievalBranchPolicy(name="trigram", run=_run_trigram_branch),
)

# Derived, not authored: the i-th label names the i-th gathered result, and the
# pairing tests pin this derivation rather than a hand-kept tuple.
_SEARCH_BRANCHES = tuple(branch.name for branch in RETRIEVAL_BRANCHES)


async def _run_branches(
    *, repo: DocumentRepository, branch_input: _BranchInput
) -> DocumentResult[list[list[RankedResultRow]]]:
    """Run every retrieval branch and return ranked row sets, or fail with attribution.

    A branch that *raised* keeps its own error kind via `model_copy` — re-wrapping
    would flatten the taxonomy and turn a 422 from one branch into a 503 for the
    whole request — with only the branch name added.
    """
    results = await asyncio.gather(
        *(branch.run(repo, branch_input) for branch in RETRIEVAL_BRANCHES)
    )
    row_sets: list[list[RankedResultRow]] = []
    for branch, branch_result in zip(RETRIEVAL_BRANCHES, results, strict=True):
        if isinstance(branch_result, Failure):
            error = branch_result.failure()
            return Failure(
                error.model_copy(
                    update={
                        "message": f"{branch.name} retrieval branch failed: {error.message}",
                        "details": {**(error.details or {}), "branch": branch.name},
                    }
                )
            )
        # `else`, not `elif isinstance(..., Success)`. The old form had no final branch, so a
        # value that was neither would have been dropped from `row_sets` entirely — shrinking
        # the fusion input with no log line and no failure.
        row_sets.append(_to_ranked_rows(branch_result.unwrap()))
    return Success(row_sets)


async def retrieve_fused(
    *,
    repo: DocumentRepository,
    user_id: str,
    query_text: str,
    query_embedding: list[float],
    candidate_limit: int,
    limit: int,
    filter_params: dict[str, Any],
    weights: Sequence[float] | None = None,
    bm25_threshold: float | None = None,
    exact_phrase: str | None = None,
) -> DocumentResult[tuple[list[RankedChunk], dict[str, dict[str, Any]]]]:
    """Run the three branch legs over base chunks and fuse them in Python.

    The single fused path shared by the search endpoint, `ask`, and the
    retrieval-graph hybrid node: every caller issues the same branch SQL with the
    same inputs and fuses with the same weights, so one query yields identical
    chunk-id order from every door. A branch that *raised* keeps its attribution
    (branch name in details); a branch that legitimately matches nothing
    contributes an empty rank list and fusion proceeds.
    """
    branch_input = _BranchInput(
        user_id=user_id,
        query_text=query_text,
        query_embedding=query_embedding,
        candidate_limit=candidate_limit,
        filter_params=filter_params,
        bm25_threshold=bm25_threshold,
        exact_phrase=exact_phrase,
    )
    results = await _run_branches(repo=repo, branch_input=branch_input)
    if isinstance(results, Failure):
        return Failure(results.failure())
    row_sets = results.unwrap()
    # Phrase matches may rank below many vector/trigram-only candidates. Preserve
    # every candidate already bounded by the branch limits until the phrase
    # constraint has been applied; a second pre-filter cutoff can lose valid hits.
    fusion_limit = sum(len(row_set) for row_set in row_sets) if exact_phrase else limit
    fused = reciprocal_rank_fusion(*row_sets, k=RRF_K, limit=fusion_limit, weights=weights)
    if not fused:
        return Success(([], {}))
    chunk_ids = [item.chunk_id for item in fused]
    chunk_lookup_result = (
        await repo.fetch_chunks_by_ids(chunk_ids, exact_phrase=exact_phrase)
        if exact_phrase
        else await repo.fetch_chunks_by_ids(chunk_ids)
    )
    if isinstance(chunk_lookup_result, Failure):
        return Failure(chunk_lookup_result.failure())
    lookup: dict[str, dict[str, Any]] = chunk_lookup_result.unwrap()
    if exact_phrase:
        # Hydration applies the same escaped PostgreSQL ILIKE predicate used by
        # the BM25 leg, so vector/trigram candidates cannot diverge on Unicode
        # or wildcard semantics.
        fused = [item for item in fused if item.chunk_id in lookup][:limit]
        lookup = {item.chunk_id: lookup[item.chunk_id] for item in fused if item.chunk_id in lookup}
    return Success((fused, lookup))


def lookup_to_rows(
    fused: Sequence[RankedChunk],
    lookup: dict[str, dict[str, Any]],
) -> list[dict[str, object]]:
    """Adapt fused ranks plus the chunk lookup to `_row_to_chunk` row shape."""
    rows: list[dict[str, object]] = []
    for item in fused:
        record = lookup.get(item.chunk_id)
        if record is None:
            continue
        rows.append(
            {
                "chunk_id": item.chunk_id,
                "chunk_text": record.get("content"),
                "preamble": record.get("preamble") or "",
                "clause_type": record.get("clause_type"),
                "parent_doc_id": record.get("document_id"),
                "metadata_": record.get("chunk_metadata") or {},
                "custom_metadata": {},
                "quality_warnings": record.get("quality_warnings", []),
                "graphiti_verified": bool(record.get("graphiti_verified", False)),
                "rrf_score": item.score,
                "title": record.get("title"),
                "chunk_index": record.get("chunk_index"),
                "chunk_kind": record.get("chunk_kind"),
            }
        )
    return rows


class DocumentQueryService:
    """Unified retrieval and grounded QA service."""

    def __init__(
        self,
        repo: DocumentRepository,
        llm_factory: Callable[[], BaseChatModel],
        redis: Redis | None,
        graphiti: Graphiti | None,
    ):
        self.repo: DocumentRepository = repo
        # The model client is constructed on first use, not at dependency
        # resolution: building it eagerly lets an environment failure (missing
        # provider package, bad key) answer an unauthenticated request with a
        # 500 that masks the 401 auth already earned.
        self._llm: BaseChatModel | None = None
        self._llm_factory: Callable[[], BaseChatModel] = llm_factory
        self.redis: Redis | None = redis
        self.graphiti: Graphiti | None = graphiti

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = self._llm_factory()
        return self._llm

    @trace_layer("service")
    async def navigate_structure(
        self,
        *,
        user_id: str,
        query: str,
        document_ids: list[str] | None = None,
        limit: int = 5,
    ) -> DocumentResult[list[tuple[str, ...]]]:
        """Navigate stored document trees through the repository boundary."""
        trees_result = await self.repo.fetch_structural_trees(
            user_id=user_id,
            document_ids=document_ids or [],
        )
        if isinstance(trees_result, Failure):
            return Failure(trees_result.failure())
        paths: list[tuple[str, ...]] = []
        for row in trees_result.unwrap():
            tree = row.get("structural_tree")
            if not isinstance(tree, dict):
                continue
            document_id = str(row.get("document_id") or "")
            paths.extend((document_id, *path) for path in navigate_tree(tree, query, limit=limit))
            if len(paths) >= limit:
                break
        return Success(paths[:limit])

    @trace_layer("service")
    async def search(
        self,
        *,
        user_id: str,
        payload: UnifiedSearchRequest,
        weights: Sequence[float] | None = None,
    ) -> DocumentResult[UnifiedSearchResponse]:
        cache_key = _build_cache_key("documents:search", payload, user_id=user_id)
        lock_key = f"{cache_key}:lock"
        lock_acquired = False
        if not payload.bypass_cache and self.redis is not None:
            cached = await self.redis.get(cache_key)
            if cached is not None:
                response: UnifiedSearchResponse = UnifiedSearchResponse.model_validate_json(cached)
                return Success(response.model_copy(update={"cache_hit": True}))

            # ponytail: setnx lock prevents concurrent duplicate compute for same query
            lock_acquired = await self.redis.setnx(lock_key, "1")
            if not lock_acquired:
                for _ in range(30):
                    await asyncio.sleep(0.05)
                    cached = await self.redis.get(cache_key)
                    if cached is not None:
                        return Success(
                            UnifiedSearchResponse.model_validate_json(cached).model_copy(
                                update={"cache_hit": True}
                            )
                        )
            else:
                await self.redis.expire(lock_key, 15)

        query_embedding = await embed_text(
            payload.query,
            task_type=EmbeddingTaskType.QUERY,
            redis=None if payload.bypass_cache else self.redis,
        )
        filter_params = build_search_filter_params(
            metadata_filter=payload.metadata_filter.model_dump()
        )
        fused_result = await retrieve_fused(
            repo=self.repo,
            user_id=user_id,
            query_text=payload.query,
            query_embedding=query_embedding,
            candidate_limit=payload.candidate_limit,
            limit=payload.limit,
            filter_params=filter_params,
            weights=weights,
        )
        if isinstance(fused_result, Failure):
            error = fused_result.failure()
            log_expected_failure(error, operation="hybrid_search")
            # The setnx lock above is not released on this path. It carries a 15s expiry for
            # exactly this reason, and every other raise in this method already relied on it.
            return Failure(error)
        fused_results, chunk_lookup = fused_result.unwrap()
        items: list[DocumentSearchResultItem] = _build_search_items(
            fused_results=fused_results, chunk_lookup=chunk_lookup
        )
        response = UnifiedSearchResponse(items=items, cache_hit=False)
        if not payload.bypass_cache and self.redis is not None:
            await self.redis.setex(
                name=cache_key,
                time=DEFAULT_SEARCH_CACHE_TTL_SECONDS,
                value=response.model_dump_json(),
            )
            if lock_acquired:
                await self.redis.delete(lock_key)
        return Success(response)

    async def _fuse_search_branches(
        self,
        *,
        user_id: str,
        payload: UnifiedSearchRequest,
        query_embedding: list[float],
        filter_params: dict[str, Any],
        weights: Sequence[float] | None = None,
    ) -> DocumentResult[list[RankedChunk]]:
        """Run the three retrieval modes and fuse them, or fail naming the branch that broke.

        The distinction this method exists to draw: **an empty result from a healthy branch is
        not a failure.** A keyword branch that legitimately matches nothing contributes an empty
        rank list and the fusion proceeds over two modes; a keyword branch that *raised* used to
        contribute an identical empty rank list, so a partially-broken index answered `200` with
        results silently fused from fewer modes than the caller asked for. The two cases were
        indistinguishable in the response, which is what made the old degrade path a correctness
        problem rather than a resilience feature.

        Returned as a `Result` rather than raised because the branch identity is the payload: the
        caller is the ownership boundary, and a test can assert the branch name without having to
        catch an exception and re-parse its message.
        """
        branch_input = _BranchInput(
            user_id=user_id,
            query_text=payload.query,
            query_embedding=query_embedding,
            candidate_limit=payload.candidate_limit,
            filter_params=filter_params,
        )
        row_sets_result = await _run_branches(repo=self.repo, branch_input=branch_input)
        if isinstance(row_sets_result, Failure):
            return Failure(row_sets_result.failure())
        return Success(
            reciprocal_rank_fusion(
                *row_sets_result.unwrap(),
                k=RRF_K,
                limit=payload.limit,
                weights=weights,
            )
        )

    @trace_layer("service")
    async def rag(
        self, *, user_id: str, payload: documents_dto.UnifiedRagRequest
    ) -> DocumentResult[UnifiedRagResponse]:
        search_result = await self.search(
            user_id=user_id, payload=UnifiedSearchRequest.model_validate(payload.model_dump())
        )
        if isinstance(search_result, Failure):
            return Failure(search_result.failure())
        response = search_result.unwrap()
        chunk_lookup: dict[str, SearchChunkRecord] = {
            item.chunk_id: SearchChunkRecord(
                document_id=item.document_id,
                title=item.title,
                content=item.content,
                chunk_index=item.chunk_index,
                chunk_metadata=item.chunk_metadata,
            )
            for item in response.items
        }
        ranked_chunks: list[RankedChunk] = [
            RankedChunk(chunk_id=item.chunk_id, score=item.score, rank=item.rank)
            for item in response.items
        ]
        context_sections: list[ContextSection] = assemble_rag_context(
            ranked_chunks,
            chunk_lookup,
            max_tokens=payload.max_tokens,
            count_tokens=default_count_tokens,
        )
        return Success(
            UnifiedRagResponse(
                items=response.items,
                context=[
                    RagContextSectionResponse(
                        document_id=section.document_id,
                        title=section.title,
                        content=section.content,
                        chunk_indices=section.chunk_indices,
                        chunk_metadata=section.chunk_metadata,
                    )
                    for section in context_sections
                ],
                cache_hit=response.cache_hit,
            )
        )

    @trace_layer("service")
    async def ask_via_retrieval_graph(
        self, *, user_id: str, payload: documents_dto.UnifiedAskRequest
    ) -> DocumentResult[UnifiedAskResponse]:
        """Answer through the compiled retrieval graph rather than the inline loop in `ask`.

        **Deliberately not exposed by any router**, and that is the whole point of it existing.
        `build_retrieval_graph` is change 1's foundation — the compiled plan/retrieve/grade/generate
        machine in `features/documents/retrieval_graph/` — and its only caller in the tree was
        `features/search/service.py:ask_legal`, which this change deletes. Deleting the last caller
        would leave the graph builder, its nodes and the retarget at step 5 with nothing reaching
        them: type-checked, unit-tested, and unreferenced from the application. So the caller moves
        here rather than disappearing.

        It is honest to say what this costs. `ask` above re-implements the same node sequence inline
        as module-level helpers (`_build_query_plan`, `_grade_context`, `_generate_answer`) and is
        the path the mounted router actually serves, so the two are duplicate expressions of one
        behaviour. That duplication is recorded as debt in this change's notes rather than resolved
        here: collapsing them is a behavioural change to a live endpoint, and this step is a
        deletion. The graph is the better-factored of the two and is the intended survivor.
        """
        graph = build_retrieval_graph(
            llm=self.llm,
            repo=self.repo,
            redis=None if payload.bypass_cache else self.redis,
            graphiti=self.graphiti,
        )
        result = await graph.ainvoke(
            {
                "user_id": user_id,
                "query": payload.query,
                "doc_ids_filter": payload.doc_ids_filter,
                "messages": [],
                "iteration_count": 0,
            }
        )
        answer = GeneratedAnswer.model_validate(result["generated_answer"])
        return Success(
            UnifiedAskResponse(
                answer=answer.answer,
                citations=[
                    LegalCitationResponse(
                        chunk_id=citation.chunk_id,
                        clause_type=citation.clause_type,
                        claim=citation.claim,
                    )
                    for citation in answer.citations
                ],
                confidence=answer.confidence,
                cache_hit=bool(result.get("cache_hit")),
            )
        )

    @trace_layer("service")
    async def ask(  # noqa: PLR0914
        self,
        *,
        user_id: str,
        payload: documents_dto.UnifiedAskRequest,
        require_graphiti_verified: bool,
    ) -> DocumentResult[UnifiedAskResponse]:
        answer_cache_key = _build_answer_cache_key(
            user_id=user_id,
            query=payload.query,
            doc_ids_filter=payload.doc_ids_filter,
            jurisdiction=payload.jurisdiction,
            contract_type=payload.contract_type,
            clause_type=payload.clause_type,
            require_graphiti_verified=require_graphiti_verified,
        )
        if not payload.bypass_cache and self.redis is not None:
            cached = await self.redis.get(answer_cache_key)
            if cached is not None:
                response = UnifiedAskResponse.model_validate_json(cached)
                return Success(response.model_copy(update={"cache_hit": True}))

        settings: Settings = get_settings()
        _ = settings
        llm = self.llm
        query_llm = llm.with_structured_output(QueryPlan)
        grader_llm = llm.with_structured_output(ContextGrade)
        generator_llm = llm.with_structured_output(GeneratedAnswer)
        response_warnings: list[QualityWarningDTO] = []
        answer = GeneratedAnswer(answer=_FALLBACK_ANSWER, citations=[], confidence="uncertain")
        grade = ContextGrade(sufficient=False, missing_aspects=["No retrieved chunks"])

        for iteration in range(2):
            rewritten_query = (
                payload.query if iteration == 0 else (grade.rewrite_suggestion or payload.query)
            )
            plan = await _build_query_plan(
                query_llm=query_llm,
                query=rewritten_query,
                doc_ids_filter=payload.doc_ids_filter,
                jurisdiction=payload.jurisdiction,
                contract_type=payload.contract_type,
                clause_type=payload.clause_type,
            )
            graph_chunk_ids = await _graphiti_filter_chunk_ids(
                graphiti=self.graphiti,
                user_id=user_id,
                query=plan.rewritten_query,
                doc_ids_filter=payload.doc_ids_filter,
            )
            embedding = await retry_immediate(
                # `query=` binds the loop variable at definition rather than at call. It is awaited
                # inside the same iteration so late binding would not bite today, but this loop
                # retries on a grade and the binding is what keeps that true.
                lambda query=plan.rewritten_query: embed_text(
                    query,
                    task_type=EmbeddingTaskType.QUERY,
                    redis=None if payload.bypass_cache else self.redis,
                ),
                label="documents_query_embedding",
            )
            filter_params = build_search_filter_params(
                metadata_filter={
                    "document_ids": payload.doc_ids_filter,
                    "chunk_ids": graph_chunk_ids or [],
                    "jurisdiction": payload.jurisdiction or plan.jurisdiction,
                    "contract_type": payload.contract_type or plan.contract_type,
                    "clause_type": payload.clause_type,
                    "require_graphiti_verified": require_graphiti_verified,
                }
            )
            fused_result = await retrieve_fused(
                repo=self.repo,
                user_id=user_id,
                query_text=plan.rewritten_query,
                query_embedding=embedding,
                candidate_limit=HYBRID_CANDIDATE_LIMIT,
                limit=20,
                filter_params=filter_params,
                weights=[plan.keyword_weight, plan.vector_weight, RRF_WEIGHT_TRIGRAM],
                bm25_threshold=plan.bm25_threshold,
                exact_phrase=plan.exact_phrase,
            )
            if isinstance(fused_result, Failure):
                return Failure(fused_result.failure())
            fused, lookup = fused_result.unwrap()
            rows = lookup_to_rows(fused, lookup)
            retrieved_chunks = [_row_to_chunk(row) for row in rows]
            reranked = await get_shared_reranker().rerank(
                plan.rewritten_query, retrieved_chunks, limit=5
            )
            grade = await _grade_context(
                grader_llm=grader_llm, query=plan.rewritten_query, reranked=reranked
            )
            response_warnings = _merge_warning_lists([_warnings_from_rows(rows)])
            if grade.sufficient or iteration == 1:
                if grade.sufficient:
                    answer = await _generate_answer(
                        generator_llm=generator_llm,
                        query=plan.rewritten_query,
                        reranked=reranked,
                    )
                break

        response = UnifiedAskResponse(
            answer=answer.answer,
            citations=[
                LegalCitationResponse(
                    chunk_id=citation.chunk_id,
                    clause_type=citation.clause_type,
                    claim=citation.claim,
                )
                for citation in answer.citations
            ],
            confidence=answer.confidence,
            warnings=response_warnings,
            cache_hit=False,
        )
        if not payload.bypass_cache and self.redis is not None:
            await self.redis.setex(
                answer_cache_key,
                DEFAULT_SEARCH_CACHE_TTL_SECONDS,
                response.model_dump_json(),
            )
        return Success(response)


async def _build_query_plan(
    *,
    query_llm: Any,
    query: str,
    doc_ids_filter: list[str],
    jurisdiction: str | None,
    contract_type: str | None,
    clause_type: str | None,
) -> QueryPlan:
    payload = serialize_to_toon(
        {
            "query": query,
            "doc_ids_filter": doc_ids_filter,
            "jurisdiction": jurisdiction,
            "contract_type": contract_type,
            "clause_type": clause_type,
        }
    )
    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=_QUERY_ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=payload),
    ]
    try:
        raw: Unknown = await retry_immediate(
            lambda: query_llm.ainvoke(messages),
            label="documents_query_analyzer",
        )
        plan = QueryPlan.model_validate(raw)
    except (ValueError, TypeError):
        plan = QueryPlan(
            rewritten_query=query,
            sub_queries=[query],
            jurisdiction=jurisdiction,
            contract_type=contract_type,
        )
    total = plan.vector_weight + plan.keyword_weight
    if total > 0:
        plan = plan.model_copy(
            update={
                "vector_weight": plan.vector_weight / total,
                "keyword_weight": plan.keyword_weight / total,
            }
        )
    return plan


async def _graphiti_filter_chunk_ids(
    *,
    graphiti: Graphiti | None,
    user_id: str,
    query: str,
    doc_ids_filter: list[str],
) -> list[str]:
    if graphiti is None:
        return []
    try:
        raw_results = await retry_immediate(
            lambda: graphiti.search(
                query=query, group_ids=[user_id, *doc_ids_filter], num_results=20
            ),
            label="documents_graphiti_filter",
        )
    except (ValueError, TypeError):
        return []
    chunk_ids: list[str] = []
    for result in raw_results or []:
        source_description = getattr(result, "source_description", "") or ""
        content = getattr(result, "content", "") or getattr(result, "episode_body", "") or ""
        chunk_ids.extend(_extract_postgres_chunk_ids(source_description))
        chunk_ids.extend(_extract_postgres_chunk_ids(content))
    return sorted(set(chunk_ids))


async def _grade_context(
    *,
    grader_llm: Any,
    query: str,
    reranked: Sequence[RetrievedChunk],
) -> ContextGrade:
    if not reranked:
        return ContextGrade(
            sufficient=False, missing_aspects=["No retrieved chunks"], rewrite_suggestion=query
        )
    payload = serialize_to_toon(
        {"query": query, "chunks": [chunk.model_dump() for chunk in reranked]}
    )
    messages = [SystemMessage(content=_CONTEXT_GRADER_SYSTEM_PROMPT), HumanMessage(content=payload)]
    try:
        raw = await retry_immediate(
            lambda: grader_llm.ainvoke(messages), label="documents_context_grader"
        )
        return ContextGrade.model_validate(raw)
    except (ValueError, TypeError):
        return ContextGrade(sufficient=True, missing_aspects=[])


async def _generate_answer(
    *,
    generator_llm: Any,
    query: str,
    reranked: Sequence[RetrievedChunk],
) -> GeneratedAnswer:
    payload = serialize_to_toon(
        {"query": query, "chunks": [chunk.model_dump() for chunk in reranked]}
    )
    messages = [SystemMessage(content=_GENERATOR_SYSTEM_PROMPT), HumanMessage(content=payload)]
    try:
        raw = await retry_immediate(
            lambda: generator_llm.ainvoke(messages), label="documents_answer_generator"
        )
        answer = GeneratedAnswer.model_validate(raw)
    except (ValueError, TypeError):
        answer = GeneratedAnswer(answer=_FALLBACK_ANSWER, citations=[], confidence="uncertain")
    if answer.confidence == "uncertain" and _FALLBACK_ANSWER not in answer.answer:
        return answer.model_copy(update={"answer": f"{answer.answer}\n\n{_FALLBACK_ANSWER}"})
    return answer


def _to_ranked_rows(rows: list[dict[str, object]]) -> list[RankedResultRow]:
    return [
        RankedResultRow(
            chunk_id=str(row["chunk_id"]),
            score=float(cast("float | int | str", row["score"])),
            rank=index,
        )
        for index, row in enumerate(rows, start=1)
    ]


def _build_search_items(
    *,
    fused_results: Sequence[RankedChunk],
    chunk_lookup: dict[str, dict[str, object]],
) -> list[DocumentSearchResultItem]:
    items: list[DocumentSearchResultItem] = []
    for ranked_chunk in fused_results:
        row = chunk_lookup.get(ranked_chunk.chunk_id)
        if row is None:
            continue
        items.append(
            DocumentSearchResultItem(
                chunk_id=ranked_chunk.chunk_id,
                document_id=str(row["document_id"]),
                title=str(row["title"]),
                content=str(row["content"]),
                chunk_index=int(cast("int | str", row["chunk_index"])),
                chunk_kind=str(row["chunk_kind"]),
                clause_type=str(row["clause_type"]) if row["clause_type"] is not None else None,
                chunk_metadata=cast("dict[str, object]", row["chunk_metadata"] or {}),
                quality_warnings=_flatten_warnings(row.get("quality_warnings", [])),  # ty: ignore[invalid-argument-type]
                graphiti_verified=bool(row.get("graphiti_verified", False)),
                score=ranked_chunk.score,
                rank=ranked_chunk.rank,
            )
        )
    return items


def _build_cache_key(
    kind: str,
    payload: UnifiedSearchRequest,
    *,
    user_id: str,
) -> str:
    normalized_query = " ".join(payload.query.lower().split())
    filter_json = to_sorted_key_bytes(payload.metadata_filter.model_dump())
    raw = b"|".join(
        [
            kind.encode("utf-8"),
            user_id.encode("utf-8"),
            normalized_query.encode("utf-8"),
            filter_json,
            str(payload.limit).encode("utf-8"),
            str(payload.candidate_limit).encode("utf-8"),
        ]
    )
    return "documents:" + hashlib.sha256(raw).hexdigest()


def _build_answer_cache_key(
    *,
    user_id: str,
    query: str,
    doc_ids_filter: list[str],
    jurisdiction: str | None,
    contract_type: str | None,
    clause_type: str | None,
    require_graphiti_verified: bool,
) -> str:
    raw = to_sorted_key_bytes(
        {
            "user_id": user_id,
            "query": " ".join(query.lower().split()),
            "doc_ids_filter": sorted(doc_ids_filter),
            "jurisdiction": jurisdiction,
            "contract_type": contract_type,
            "clause_type": clause_type,
            "require_graphiti_verified": require_graphiti_verified,
        },
    )
    return "documents:answer:" + hashlib.sha256(raw).hexdigest()


def _batched[T](values: Sequence[T], batch_size: int) -> list[Sequence[T]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _flatten_warnings(
    raw_groups: list[Any],
) -> list[QualityWarningDTO]:  # ponytail: SQL results are untyped dicts
    warnings: list[QualityWarningDTO] = []
    for group in raw_groups:
        if isinstance(group, list):
            warnings.extend(
                QualityWarningDTO.model_validate(warning)
                for warning in group
                if isinstance(warning, dict)
            )
        elif isinstance(group, dict):
            warnings.append(QualityWarningDTO.model_validate(group))
    return warnings


def _merge_warning_lists(groups: list[list[QualityWarningDTO]]) -> list[QualityWarningDTO]:
    merged: dict[tuple[str, str, str, str], QualityWarningDTO] = {}
    for group in groups:
        for warning in group:
            merged[warning.stage, warning.code, warning.message, warning.severity] = warning
    return list(merged.values())


def _warnings_from_rows(rows: list[dict[str, object]]) -> list[QualityWarningDTO]:
    return _flatten_warnings([row.get("quality_warnings", []) for row in rows])


def _row_to_chunk(row: dict[str, object]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(row["chunk_id"]),
        chunk_text=str(row["chunk_text"]),
        preamble=str(row.get("preamble") or ""),
        clause_type=str(row.get("clause_type") or "other"),
        parent_doc_id=str(row["parent_doc_id"]),
        metadata_={
            **cast("dict[str, object]", row.get("metadata_") or {}),
            "quality_warnings": row.get("quality_warnings", []),
            "graphiti_verified": bool(row.get("graphiti_verified")),
        },
        custom_metadata=cast("dict[str, object]", row.get("custom_metadata") or {}),
        score=float(cast("float | int | str", row["rrf_score"])),
    )
