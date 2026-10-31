"""Unified document feature services."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from operator import itemgetter
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast, runtime_checkable

from graphiti_core.errors import GraphitiError
from langchain_core.messages import HumanMessage, SystemMessage
from returns.result import Failure, Success

from app.ai.embeddings import EmbeddingTaskType, embed_text
from app.ai.generation import TextGenerationRequest
from app.ai.graph.retry import TransientExternalError, retry_immediate
from app.config import get_settings
from app.features.auth.domain.tenancy import personal_tenant_id
from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
    build_search_filter_params,
)
from app.features.documents.api.http.dto import (
    DocumentSearchResultItem,
    EvidenceCoverage,
    EvidencePolicy,
    LegalCitationResponse,
    QualityWarningDTO,
    RagContextSectionResponse,
    UnifiedAskResponse,
    UnifiedRagResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from app.features.documents.application.access_audit import (
    DocumentAccessAuditEvent,
    DocumentAccessAuditRecorder,
    InMemoryDocumentAccessAuditRecorder,
)
from app.features.documents.domain.constants import (
    CANONICAL_PIPELINE_VERSION,
    DEFAULT_SEARCH_CACHE_TTL_SECONDS,
    HYBRID_CANDIDATE_LIMIT,
    RRF_K,
    RRF_WEIGHT_TRIGRAM,
)
from app.features.documents.retrieval.context_assembly import (
    SearchChunkRecord,
    assemble_rag_context,
)
from app.features.documents.retrieval.graph import (
    Citation,
    ContextGrade,
    GeneratedAnswer,
    QueryPlan,
    RetrievedChunk,
    _extract_postgres_chunk_ids,
    build_retrieval_graph,
    get_shared_reranker,
)
from app.kernel.result import log_expected_failure
from app.observability.logging import logger, trace_layer
from app.retrieval.edge_expansion import (
    EdgeContextBlock,
    EdgeExpansion,
    EdgeSeedChunk,
    expand_along_edges,
)
from app.retrieval.fusion import RankedChunk, RankedResultRow, reciprocal_rank_fusion
from app.retrieval.graphiti.client import fetch_relationship_edges
from app.retrieval.graphiti.grouping import build_graphiti_group_id
from app.retrieval.structural_navigator import navigate_tree
from app.retrieval.token_counter import count_tokens as default_count_tokens
from app.shared.langchain_layer import serialize_to_toon
from app.utils import to_sorted_key_bytes

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from typing import Any

    from graphiti_core.graphiti import Graphiti
    from langchain_core.language_models import BaseChatModel
    from redis.asyncio import Redis
    from ty_extensions import Unknown

    from app.ai.generation import TextGenerationStructuredResult
    from app.config.settings import Settings
    from app.features.documents.api.http import dto as documents_dto
    from app.features.documents.domain.errors import DocumentResult
    from app.features.documents.ingestion.chunking.classification import QualityWarning
    from app.features.documents.ingestion.enrichment.legal_metadata import (
        LegalMetadataExtraction,
    )
    from app.features.documents.retrieval.context_assembly import ContextSection
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
# Edge-aware context expansion bounds. Additive and small by construction: at
# most a few server-authored grounding lines per reranked chunk, never a
# second retrieval pass, and only for queries that actually touch edges.
_EDGE_EXPANSION_MAX_TOKENS: int = 256
_EDGE_EXPANSION_MAX_BLOCKS_PER_SEED: int = 4
_EDGE_FETCH_LIMIT: int = 32
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
    tenant_id: str
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
        tenant_id=args.tenant_id,
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
        tenant_id=args.tenant_id,
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
        tenant_id=args.tenant_id,
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
    tenant_id: str | None = None,
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
        tenant_id=_resolve_tenant_id(tenant_id, user_id),
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
    fetch_kwargs: dict[str, Any] = {}
    # Keep narrow repository test doubles and downstream adapters source
    # compatible while using the tenant-aware repository in production.
    fetch_parameters = inspect.signature(repo.fetch_chunks_by_ids).parameters
    if "exact_phrase" in fetch_parameters:
        fetch_kwargs["exact_phrase"] = exact_phrase
    if "tenant_id" in fetch_parameters:
        fetch_kwargs["tenant_id"] = _resolve_tenant_id(tenant_id, user_id)
    if "allowed_doc_ids" in fetch_parameters:
        fetch_kwargs["allowed_doc_ids"] = filter_params.get("document_ids") or None
    chunk_lookup_result = await repo.fetch_chunks_by_ids(chunk_ids, **fetch_kwargs)
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
                # `document_id` backs citation provenance and coverage counts;
                # `parent_doc_id` is the historical `_row_to_chunk` key. Both
                # name the same hydrated identity.
                "document_id": record.get("document_id"),
                "parent_doc_id": record.get("document_id"),
                "metadata_": record.get("chunk_metadata") or {},
                "custom_metadata": {},
                "quality_warnings": record.get("quality_warnings", []),
                "graphiti_verified": bool(record.get("graphiti_verified", False)),
                "graph_indexed": bool(record.get("graphiti_verified", False)),
                "graph_indexed_at": record.get("graph_indexed_at"),
                "document_version": record.get("document_version", 1),
                "contract_index": record.get("contract_index"),
                "pipeline_version": record.get("pipeline_version"),
                "rrf_score": item.score,
                "title": record.get("title"),
                "chunk_index": record.get("chunk_index"),
                "chunk_kind": record.get("chunk_kind"),
            }
        )
    return rows


@runtime_checkable
class ContextGrader(Protocol):
    """The narrow slice of the text-generation seam the context grader consumes.

    Structural, so the lifespan-owned ``LangChainTextGenerator`` and any test
    double with the same single method both satisfy it; documents never import
    the concrete adapter.
    """

    def generate_structured(
        self, request: TextGenerationRequest
    ) -> Awaitable[TextGenerationStructuredResult]: ...


class DocumentQueryService:
    """Unified retrieval and grounded QA service."""

    def __init__(
        self,
        repo: DocumentRepository,
        llm_factory: Callable[[], BaseChatModel],
        redis: Redis | None,
        graphiti: Graphiti | None,
        *,
        grader: ContextGrader | None = None,
        audit_recorder: DocumentAccessAuditRecorder | None = None,
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
        # Optional structured-grading seam (item 274 reopen pilot): the lifespan
        # owns `app.state.text_generator`, which text generation disables when
        # Redis or the provider key is missing. A disabled seam must not take
        # `ask` down, so ``None`` keeps grading on the legacy inline path below;
        # every other consumer of `self.llm` is unchanged either way.
        self.grader: ContextGrader | None = grader
        self.audit_recorder = audit_recorder or InMemoryDocumentAccessAuditRecorder()

    async def _record_access(
        self,
        result: DocumentResult[Any],
        *,
        operation: str,
        user_id: str,
        tenant_id: str,
        requested_document_ids: Sequence[str],
        document_ids: Sequence[str] = (),
        reason: str | None = None,
        outcome: str | None = None,
    ) -> DocumentResult[Any]:
        outcome = outcome or ("ok" if isinstance(result, Success) else "denied")
        if reason is None and isinstance(result, Failure):
            reason = result.failure().message
        try:
            await self.audit_recorder.append(
                DocumentAccessAuditEvent(
                    tenant_id=tenant_id,
                    principal_id=user_id,
                    operation=operation,
                    requested_document_ids=tuple(requested_document_ids),
                    document_ids=tuple(document_ids),
                    policy="mechanically_verified",
                    outcome=outcome,
                    reason=reason,
                )
            )
        except Exception as exc:  # noqa: BLE001 — audit is a fail-closed boundary
            exc.add_note(f"operation={operation}, tenant_id={tenant_id}")
            logger.bind(operation=operation, tenant_id=tenant_id).exception(
                "document_access_audit_failed"
            )
        return result

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
        tenant_id: str | None = None,
        query: str,
        document_ids: list[str] | None = None,
        limit: int = 5,
    ) -> DocumentResult[list[tuple[str, ...]]]:
        """Navigate stored document trees through the repository boundary."""
        trees_result = await self.repo.fetch_structural_trees(
            user_id=user_id,
            tenant_id=tenant_id,
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
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: UnifiedSearchRequest,
        weights: Sequence[float] | None = None,
    ) -> DocumentResult[UnifiedSearchResponse]:
        resolved_tenant = _resolve_tenant_id(tenant_id, user_id)
        result = await self._search(
            user_id=user_id,
            tenant_id=resolved_tenant,
            allowed_doc_ids=allowed_doc_ids,
            payload=payload,
            weights=weights,
        )
        scope = tuple(allowed_doc_ids or ())
        document_ids = (
            tuple(item.document_id for item in result.unwrap().items)
            if isinstance(result, Success)
            else ()
        )
        return cast(
            "DocumentResult[UnifiedSearchResponse]",
            await self._record_access(
                result,
                operation="document.search",
                user_id=user_id,
                tenant_id=resolved_tenant,
                requested_document_ids=scope,
                document_ids=document_ids,
                reason=("requested document scope is empty" if allowed_doc_ids == () else None),
                outcome="denied" if allowed_doc_ids == () else None,
            ),
        )

    async def _search(  # noqa: PLR0914
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: UnifiedSearchRequest,
        weights: Sequence[float] | None = None,
    ) -> DocumentResult[UnifiedSearchResponse]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        explicit_scope = allowed_doc_ids is not None
        scope_ids = tuple(dict.fromkeys(allowed_doc_ids or ()))
        if explicit_scope and not scope_ids:
            return Success(UnifiedSearchResponse(items=[], cache_hit=False))
        scope_versions = await _resolve_scope_versions(
            self.repo, tenant_id=tenant_id, allowed_doc_ids=scope_ids
        )
        cache_key = _build_cache_key(
            "documents:search",
            payload,
            user_id=user_id,
            tenant_id=tenant_id,
            allowed_doc_ids=scope_ids,
            document_versions=scope_versions,
            evidence_policy="mechanically_verified",
        )
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
            metadata_filter={
                **payload.metadata_filter.model_dump(),
                "document_ids": list(scope_ids)
                if explicit_scope
                else payload.metadata_filter.document_ids,
                "scope_bound": explicit_scope,
            }
        )
        fused_result = await retrieve_fused(
            repo=self.repo,
            user_id=user_id,
            tenant_id=tenant_id,
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
        tenant_id: str | None = None,
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
            tenant_id=_resolve_tenant_id(tenant_id, user_id),
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
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: documents_dto.UnifiedRagRequest,
    ) -> DocumentResult[UnifiedRagResponse]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        search_result = await self.search(
            user_id=user_id,
            tenant_id=tenant_id,
            allowed_doc_ids=allowed_doc_ids,
            payload=UnifiedSearchRequest.model_validate(payload.model_dump()),
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
                        contract_index=_context_contract_index(section.chunk_metadata),
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
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: documents_dto.UnifiedAskRequest,
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
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        if allowed_doc_ids is not None and not allowed_doc_ids:
            return Success(
                UnifiedAskResponse(
                    answer=_FALLBACK_ANSWER,
                    citations=[],
                    confidence="uncertain",
                    evidence_policy="mechanically_verified",
                )
            )
        scope_ids = tuple(dict.fromkeys(allowed_doc_ids or ()))
        scope_versions = await _resolve_scope_versions(
            self.repo, tenant_id=tenant_id, allowed_doc_ids=scope_ids
        )
        scope_contracts = await _resolve_scope_contracts(
            self.repo, tenant_id=tenant_id, allowed_doc_ids=scope_ids
        )
        version_by_document = {document_id: version for document_id, version, _ in scope_versions}
        contracts_by_document = {
            document_id: (version, indexes, legacy_null)
            for document_id, version, indexes, legacy_null in scope_contracts
        }
        unavailable_scope_documents: list[str] = []
        for document_id in scope_ids:
            version = version_by_document.get(document_id)
            contract_scope = contracts_by_document.get(document_id)
            if version is None or contract_scope is None or contract_scope[0] != version:
                unavailable_scope_documents.append(document_id)
                continue
            _scope_version, indexes, legacy_null = contract_scope
            if payload.contract_index is not None:
                if payload.contract_index not in indexes:
                    unavailable_scope_documents.append(document_id)
            elif not indexes and not legacy_null:
                unavailable_scope_documents.append(document_id)
        graph = build_retrieval_graph(
            llm=self.llm,
            repo=self.repo,
            redis=None if payload.bypass_cache else self.redis,
            graphiti=self.graphiti,
        )
        result = await graph.ainvoke(
            {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "allowed_doc_ids": list(scope_ids),
                "document_versions": list(scope_versions),
                "document_contracts": list(scope_contracts),
                "scope_unavailable_documents": unavailable_scope_documents,
                "scope_bound": allowed_doc_ids is not None,
                "query": payload.query,
                "doc_ids_filter": payload.doc_ids_filter,
                "contract_index": payload.contract_index,
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
                        contract_index=(
                            citation.contract_index
                            if citation.contract_index is not None
                            else _retrieved_contract_index(
                                citation.chunk_id, result.get("retrieved_chunks", [])
                            )
                        ),
                    )
                    for citation in answer.citations
                ],
                confidence=answer.confidence,
                evidence_policy="mechanically_verified",
                coverage=EvidenceCoverage(
                    documents_requested=len(scope_ids),
                    documents_ready=len(scope_ids) - len(unavailable_scope_documents),
                    graph_unavailable_documents=len(unavailable_scope_documents),
                ),
                warnings=(
                    [
                        QualityWarningDTO(
                            stage="graph_scope",
                            code="GRAPH_SCOPE_UNAVAILABLE",
                            message=(
                                "Authoritative document version/contract scope was unavailable "
                                f"for {len(unavailable_scope_documents)} authorized document(s)."
                            ),
                            severity="warning",
                        )
                    ]
                    if unavailable_scope_documents
                    else []
                ),
                cache_hit=bool(result.get("cache_hit")),
            )
        )

    @trace_layer("service")
    async def ask(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: documents_dto.UnifiedAskRequest,
        require_graphiti_verified: bool,
    ) -> DocumentResult[UnifiedAskResponse]:
        resolved_tenant = _resolve_tenant_id(tenant_id, user_id)
        result = await self._ask(
            user_id=user_id,
            tenant_id=resolved_tenant,
            allowed_doc_ids=allowed_doc_ids,
            payload=payload,
            require_graphiti_verified=require_graphiti_verified,
        )
        scope = tuple(allowed_doc_ids or ())
        document_ids = scope if isinstance(result, Success) else ()
        return cast(
            "DocumentResult[UnifiedAskResponse]",
            await self._record_access(
                result,
                operation="document.read",
                user_id=user_id,
                tenant_id=resolved_tenant,
                requested_document_ids=scope,
                document_ids=document_ids,
                reason=("requested document scope is empty" if allowed_doc_ids == () else None),
                outcome="denied" if allowed_doc_ids == () else None,
            ),
        )

    async def _ask(  # noqa: PLR0914
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        allowed_doc_ids: tuple[str, ...] | None = None,
        payload: documents_dto.UnifiedAskRequest,
        require_graphiti_verified: bool,
    ) -> DocumentResult[UnifiedAskResponse]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        explicit_scope = allowed_doc_ids is not None
        scope_ids = tuple(dict.fromkeys(allowed_doc_ids or ()))
        if explicit_scope and not scope_ids:
            return Success(
                UnifiedAskResponse(
                    answer=_FALLBACK_ANSWER,
                    citations=[],
                    confidence="uncertain",
                    evidence_policy=(
                        "graph_indexed_only"
                        if require_graphiti_verified
                        else "mechanically_verified"
                    ),
                )
            )
        scope_versions = await _resolve_scope_versions(
            self.repo, tenant_id=tenant_id, allowed_doc_ids=scope_ids
        )
        scope_contracts = await _resolve_scope_contracts(
            self.repo, tenant_id=tenant_id, allowed_doc_ids=scope_ids
        )
        answer_cache_key = _build_answer_cache_key(
            user_id=user_id,
            tenant_id=tenant_id,
            query=payload.query,
            doc_ids_filter=payload.doc_ids_filter,
            allowed_doc_ids=scope_ids,
            document_versions=scope_versions,
            jurisdiction=payload.jurisdiction,
            contract_type=payload.contract_type,
            contract_index=payload.contract_index,
            clause_type=payload.clause_type,
            require_graphiti_verified=require_graphiti_verified,
            evidence_policy=(
                "graph_indexed_only" if require_graphiti_verified else "mechanically_verified"
            ),
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
        # Edge-grounded expansion is additive: it stays empty unless an
        # authorized graph scope, a healthy graphiti, and at least one typed
        # edge touching the reranked chunks all exist.
        edge_expansions: list[EdgeExpansion] = []
        resolved_edges: list[tuple[EdgeContextBlock, dict[str, object]]] = []

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
                tenant_id=tenant_id,
                query=plan.rewritten_query,
                doc_ids_filter=list(scope_ids) if explicit_scope else payload.doc_ids_filter,
                contract_index=payload.contract_index,
                document_versions=scope_versions,
                document_contracts=scope_contracts,
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
                    "document_ids": list(scope_ids) if explicit_scope else payload.doc_ids_filter,
                    "scope_bound": explicit_scope,
                    "chunk_ids": graph_chunk_ids or [],
                    "jurisdiction": payload.jurisdiction or plan.jurisdiction,
                    "contract_type": payload.contract_type or plan.contract_type,
                    "contract_index": payload.contract_index,
                    "clause_type": payload.clause_type,
                    "require_graphiti_verified": require_graphiti_verified,
                }
            )
            fused_result = await retrieve_fused(
                repo=self.repo,
                user_id=user_id,
                tenant_id=tenant_id,
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
            rows_by_chunk_id = {str(row["chunk_id"]): row for row in rows}
            retrieved_chunks = [_row_to_chunk(row) for row in rows]
            reranked = await get_shared_reranker().rerank(
                plan.rewritten_query, retrieved_chunks, limit=5
            )
            grade = await _grade_context(
                grader_llm=grader_llm,
                grader=self.grader,
                query=plan.rewritten_query,
                reranked=reranked,
            )
            response_warnings = _merge_warning_lists([_warnings_from_rows(rows)])
            if grade.sufficient or iteration == 1:
                if grade.sufficient:
                    edge_expansions = await _edge_grounding_for_reranked(
                        graphiti=self.graphiti,
                        query=plan.rewritten_query,
                        tenant_id=tenant_id,
                        scope_ids=scope_ids if explicit_scope else (),
                        scope_versions=scope_versions,
                        scope_contracts=scope_contracts,
                        contract_index=payload.contract_index,
                        reranked=reranked,
                    )
                    resolved_edges = _resolved_edge_blocks(edge_expansions, rows_by_chunk_id)
                    answer = await _generate_answer(
                        generator_llm=generator_llm,
                        query=plan.rewritten_query,
                        reranked=reranked,
                        grounding_edges=[block.text for block, _ in resolved_edges],
                    )
                break

        evidence_policy: EvidencePolicy = (
            "graph_indexed_only" if require_graphiti_verified else "mechanically_verified"
        )
        result_document_ids = {str(row["document_id"]) for row in rows}
        graph_document_ids = {
            str(row["document_id"]) for row in rows if bool(row.get("graphiti_verified", False))
        }
        citations = [
            _citation_response(citation, rows_by_chunk_id.get(citation.chunk_id))
            for citation in answer.citations
        ]
        citations.extend(_edge_citations(resolved_edges))
        response = UnifiedAskResponse(
            answer=answer.answer,
            citations=citations,
            confidence=answer.confidence,
            evidence_policy=evidence_policy,
            coverage=EvidenceCoverage(
                documents_requested=len(scope_ids)
                if explicit_scope
                else len(payload.doc_ids_filter),
                documents_ready=len(result_document_ids),
                graph_indexed_documents=len(graph_document_ids),
                graph_unavailable_documents=len(result_document_ids - graph_document_ids),
            ),
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
    contract_index: int | None = None,
) -> QueryPlan:
    payload = serialize_to_toon(
        {
            "query": query,
            "doc_ids_filter": doc_ids_filter,
            "jurisdiction": jurisdiction,
            "contract_type": contract_type,
            "contract_index": contract_index,
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
    tenant_id: str = "",
    user_id: str | None = None,
    query: str,
    doc_ids_filter: list[str],
    contract_index: int | None = None,
    document_versions: Sequence[tuple[str, int, str | None]] = (),
    document_contracts: Sequence[tuple[str, int, tuple[int | None, ...], bool]] = (),
) -> list[str]:
    """Narrow candidates using the server-selected tenant Graphiti scope.

    ``user_id`` remains accepted for callers that used the pre-tenant helper
    signature, but it is intentionally ignored.  A user identity is an audit
    field, not a Graphiti authority boundary; when the explicit tenant scope
    is absent this helper fails closed instead of deriving one from the user.
    """
    _ = user_id
    if graphiti is None or not tenant_id or not doc_ids_filter:
        return []
    versions = {document_id: version for document_id, version, _ in document_versions}
    contract_scopes = {
        (document_id, version): (indexes, legacy_null)
        for document_id, version, indexes, legacy_null in document_contracts
    }
    group_ids: list[str] = []
    for document_id in doc_ids_filter:
        version = versions.get(document_id)
        if version is None:
            # No authoritative version means no graph read.  In particular,
            # never fall back to the historical bare/v1 document group.
            continue
        scope = contract_scopes.get((document_id, version))
        if scope is None:
            continue
        indexes, legacy_null = scope
        if contract_index is not None:
            if contract_index not in indexes:
                continue
            group_ids.append(
                build_graphiti_group_id(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    document_version=version,
                    contract_index=contract_index,
                )
            )
            continue
        if not indexes and not legacy_null:
            continue
        group_ids.extend(
            build_graphiti_group_id(
                tenant_id=tenant_id,
                document_id=document_id,
                document_version=version,
                contract_index=index,
            )
            for index in indexes
        )
        if legacy_null:
            group_ids.append(
                build_graphiti_group_id(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    document_version=version,
                    contract_index=None,
                )
            )
    if not group_ids:
        return []
    try:
        raw_results = await retry_immediate(
            lambda: graphiti.search(query=query, group_ids=group_ids, num_results=20),
            label="documents_graphiti_filter",
        )
    except (GraphitiError, TransientExternalError, ValueError, TypeError):
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
    grader: ContextGrader | None,
    query: str,
    reranked: Sequence[RetrievedChunk],
) -> ContextGrade:
    """Grade retrieval sufficiency through the structured seam, or its legacy twin.

    Dual paths, deliberate: the injected grader rides the lifespan-owned
    text-generator seam (retries, breaker, schema-validated output), while
    ``grader=None`` — the seam disabled at lifespan — replays the prior inline
    path verbatim so a disabled optional capability cannot change ask() results.
    """
    if not reranked:
        return ContextGrade(
            sufficient=False, missing_aspects=["No retrieved chunks"], rewrite_suggestion=query
        )
    payload = serialize_to_toon(
        {"query": query, "chunks": [chunk.model_dump() for chunk in reranked]}
    )
    if grader is None:
        return await _grade_context_via_llm(grader_llm=grader_llm, payload=payload)

    request = TextGenerationRequest(
        prompt=payload,
        system=_CONTEXT_GRADER_SYSTEM_PROMPT,
        response_schema=ContextGrade,
        # Grading is the same near-deterministic call the legacy inline path
        # made (dependencies.py builds the document LLM at temperature=0.1);
        # without this override the seam would inherit GEMINI_TEMPERATURE (0.7).
        temperature=0.1,
    )
    result = await grader.generate_structured(request)
    if isinstance(result, Failure):
        error = result.failure()
        log_expected_failure(error, operation="documents_context_grader")
        # Same degrade the legacy parse-failure path chose: an unavailable grade
        # proceeds to generation rather than failing the whole ask() loop.
        return ContextGrade(sufficient=True, missing_aspects=[])
    return ContextGrade.model_validate(result.unwrap().output)


async def _grade_context_via_llm(*, grader_llm: Any, payload: str) -> ContextGrade:
    """Legacy inline grading path, kept verbatim for the seam-absent degrade."""
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
    grounding_edges: Sequence[str] | None = None,
) -> GeneratedAnswer:
    payload_data: dict[str, Any] = {
        "query": query,
        "chunks": [chunk.model_dump() for chunk in reranked],
    }
    # The key exists only when edge expansion actually produced blocks, so a
    # query that touches no edges renders the exact pre-edge payload.
    if grounding_edges:
        payload_data["grounding_edges"] = list(grounding_edges)
    payload = serialize_to_toon(payload_data)
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


async def _edge_grounding_for_reranked(
    *,
    graphiti: Graphiti | None,
    query: str,
    tenant_id: str,
    scope_ids: Sequence[str],
    scope_versions: Sequence[tuple[str, int, str | None]],
    scope_contracts: Sequence[tuple[str, int, tuple[int | None, ...], bool]],
    contract_index: int | None,
    reranked: Sequence[RetrievedChunk],
) -> list[EdgeExpansion]:
    """Expand reranked chunk context along persisted typed relationship edges.

    Fails closed on every axis: without a live graphiti, a resolved tenant, or
    an explicit non-empty authorized document scope there is no edge read at
    all — the same boundary ``_graphiti_filter_chunk_ids`` enforces.  A
    transport failure inside the reader degrades to no expansion, so an
    edge-free query keeps the exact pre-edge ask contract.
    """
    if graphiti is None or not tenant_id or not scope_ids:
        return []
    edges = await fetch_relationship_edges(
        graphiti,
        query,
        allowed_doc_ids=tuple(dict.fromkeys(scope_ids)),
        tenant_id=tenant_id,
        contract_index=contract_index,
        document_versions=scope_versions,
        document_contracts=scope_contracts,
        num_results=_EDGE_FETCH_LIMIT,
    )
    if not edges or not reranked:
        return []
    seeds = [
        EdgeSeedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.parent_doc_id,
            content=chunk.chunk_text,
            clause_ids=_chunk_clause_ids(chunk),
            contract_index=_chunk_contract_index(chunk),
            document_version=_chunk_document_version(chunk),
        )
        for chunk in reranked
    ]
    return expand_along_edges(
        seeds=seeds,
        edges=edges,
        max_expansion_tokens=_EDGE_EXPANSION_MAX_TOKENS,
        count_tokens=default_count_tokens,
        max_blocks_per_seed=_EDGE_EXPANSION_MAX_BLOCKS_PER_SEED,
    )


def _chunk_document_version(chunk: RetrievedChunk) -> int | None:
    """Read the chunk's document version from server metadata, if persisted."""
    raw = (chunk.metadata_ or {}).get("document_version")
    return int(raw) if isinstance(raw, int) and raw >= 1 else None


def _chunk_contract_index(chunk: RetrievedChunk) -> int | None:
    """Read the chunk's contract index from server metadata, if persisted.

    ``RetrievedChunk.contract_index`` is never populated by ``_row_to_chunk``
    (the value rides in metadata), so reading the field would leave every
    seed "unknown" and disable the cross-contract matching guard.
    """
    return _context_contract_index(chunk.metadata_ or {})


def _chunk_clause_ids(chunk: RetrievedChunk) -> tuple[str, ...]:
    """Read server-persisted clause identities from chunk metadata, if any."""
    metadata = chunk.metadata_ or {}
    raw = metadata.get("clause_ids", metadata.get("clause_id"))
    if isinstance(raw, str) and raw:
        return (raw,)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in raw if item)
    return ()


def _resolved_edge_blocks(
    expansions: Sequence[EdgeExpansion],
    rows_by_chunk_id: Mapping[str, dict[str, object]],
) -> list[tuple[EdgeContextBlock, dict[str, object]]]:
    """Keep only expansion blocks whose grounded span resolves to a hydrated
    passage — the same resolution the citations use, so the generator never
    sees a relationship it could not cite against a retrieved chunk."""
    rows_by_document: dict[str, list[dict[str, object]]] = {}
    for row in rows_by_chunk_id.values():
        rows_by_document.setdefault(str(row.get("document_id")), []).append(row)
    resolved: list[tuple[EdgeContextBlock, dict[str, object]]] = []
    for expansion in expansions:
        for block in expansion.blocks:
            if block.char_start is None or block.char_end is None:
                continue
            supporting = _row_containing_span(
                rows_by_document.get(expansion.document_id, ()),
                char_start=block.char_start,
                char_end=block.char_end,
                document_version=block.document_version,
                contract_index=block.contract_index,
            )
            if supporting is not None:
                resolved.append((block, supporting))
    return resolved


def _edge_citations(
    resolved: Sequence[tuple[EdgeContextBlock, dict[str, object]]],
) -> list[LegalCitationResponse]:
    """Server-derived citations for edge-grounded expansion blocks.

    Provenance discipline matches ``_citation_response``: document identity,
    version, and contract come from the hydrated chunk row — never from the
    edge text or any model output — and the claim is authored here,
    deterministically, from the edge's grounded span.

    The citation points at the passage the edge's grounded span actually
    falls in, resolved among the hydrated rows — never at the seed chunk
    that merely triggered the expansion (two passages can share an entity
    name, and a relationship from one must not be cited under the other's
    chunk and page).  An edge whose span resolves to no hydrated passage —
    or that carries no span at all — earns no citation: the expansion text
    still reached the generator, but there is no verified passage to point
    at.
    """
    citations: list[LegalCitationResponse] = []
    seen: set[tuple[str, str]] = set()
    for block, supporting in resolved:
        chunk_id = str(supporting.get("chunk_id"))
        span = f"chars {block.char_start}-{block.char_end}"
        claim = (
            f"Grounded by {block.relationship.upper()} edge "
            f"'{block.from_entity}' -> '{block.to_entity}' at {span} "
            f"(clause {block.clause_id}, edge {block.edge_id})"
        )
        key = (chunk_id, claim)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            _citation_response(
                Citation(
                    chunk_id=chunk_id,
                    clause_type=str(supporting.get("clause_type") or "other"),
                    claim=claim,
                ),
                supporting,
            )
        )
    return citations


def _row_containing_span(
    rows: Sequence[dict[str, object]],
    *,
    char_start: int,
    char_end: int,
    document_version: int | None = None,
    contract_index: int | None = None,
) -> dict[str, object] | None:
    """Find the hydrated row whose persisted source span covers the edge span.

    When the edge carries version/contract provenance, the supporting passage
    must match it: two stored versions of one document can both cover a char
    range, and citing the wrong one dresses an unsupported answer as grounded.
    """
    for row in rows:
        metadata = row.get("metadata_") or {}
        if not isinstance(metadata, dict):
            continue
        start = metadata.get("source_start_char")
        end = metadata.get("source_end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not (start <= char_start and char_end <= end):
            continue
        if document_version is not None:
            row_version = metadata.get("document_version")
            if isinstance(row_version, int) and row_version != document_version:
                continue
        if contract_index is not None:
            row_contract = metadata.get("contract_index")
            if isinstance(row_contract, int) and row_contract != contract_index:
                continue
        return row
    return None


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
                document_version=int(cast("int | str", row.get("document_version", 1))),
                contract_index=(
                    int(cast("int | str", row["contract_index"]))
                    if isinstance(row.get("contract_index"), (int, str))
                    and not isinstance(row.get("contract_index"), bool)
                    else None
                ),
                pipeline_version=(
                    str(row["pipeline_version"])
                    if row.get("pipeline_version") is not None
                    else None
                ),
                title=str(row["title"]),
                content=str(row["content"]),
                chunk_index=int(cast("int | str", row["chunk_index"])),
                chunk_kind=str(row["chunk_kind"]),
                clause_type=str(row["clause_type"]) if row["clause_type"] is not None else None,
                chunk_metadata=cast("dict[str, object]", row["chunk_metadata"] or {}),
                quality_warnings=_flatten_warnings(row.get("quality_warnings", [])),  # ty: ignore[invalid-argument-type]
                graphiti_verified=bool(row.get("graphiti_verified", False)),
                graph_indexed=bool(row.get("graphiti_verified", False)),
                graph_indexed_at=(
                    str(row["graph_indexed_at"])
                    if row.get("graph_indexed_at") is not None
                    else None
                ),
                score=ranked_chunk.score,
                rank=ranked_chunk.rank,
            )
        )
    return items


async def _resolve_scope_versions(
    repo: DocumentRepository,
    *,
    tenant_id: str,
    allowed_doc_ids: Sequence[str],
) -> tuple[tuple[str, int, str | None], ...]:
    """Read current document identities before consulting a response cache.

    Narrow repository doubles used by offline evaluation predate this method;
    in that case the empty identity is safe (it disables version invalidation
    without weakening the tenant/allowlist fence).  A repository failure also
    returns an empty identity and retrieval remains the source of truth.
    """
    resolver = getattr(repo, "resolve_scope_versions", None)
    if resolver is None or not allowed_doc_ids:
        return ()
    try:
        result = await resolver(tenant_id=tenant_id, document_ids=tuple(allowed_doc_ids))
    except (AttributeError, TypeError):
        return ()
    if isinstance(result, Failure):
        return ()
    try:
        raw = result.unwrap() if isinstance(result, Success) else result
        return tuple(
            (str(doc_id), int(version), str(pipeline) if pipeline is not None else None)
            for doc_id, version, pipeline in raw
        )
    except (AttributeError, TypeError, ValueError):
        return ()


async def _resolve_scope_contracts(
    repo: DocumentRepository,
    *,
    tenant_id: str,
    allowed_doc_ids: Sequence[str],
) -> tuple[tuple[str, int, tuple[int | None, ...], bool], ...]:
    """Resolve exact contract partitions; an unavailable scope fails closed."""
    resolver = getattr(repo, "resolve_scope_contracts", None)
    if resolver is None or not allowed_doc_ids:
        return ()
    try:
        result = await resolver(tenant_id=tenant_id, document_ids=tuple(allowed_doc_ids))
    except (AttributeError, TypeError):
        return ()
    if isinstance(result, Failure):
        return ()
    try:
        raw = result.unwrap() if isinstance(result, Success) else result
        return tuple(
            (
                str(doc_id),
                int(version),
                tuple(index for index in indexes if index is None or int(index) >= 0),
                bool(legacy_null),
            )
            for doc_id, version, indexes, legacy_null in raw
        )
    except (AttributeError, TypeError, ValueError):
        return ()


def _resolve_tenant_id(tenant_id: str | None, user_id: str) -> str:
    """Migration compatibility boundary for pre-membership callers."""
    return tenant_id if tenant_id is not None else personal_tenant_id(user_id)


def _build_cache_key(
    kind: str,
    payload: UnifiedSearchRequest,
    *,
    user_id: str,
    tenant_id: str | None = None,
    allowed_doc_ids: Sequence[str] = (),
    document_versions: Sequence[tuple[str, int, str | None]] = (),
    evidence_policy: EvidencePolicy = "mechanically_verified",
) -> str:
    tenant_id = tenant_id or user_id
    normalized_query = " ".join(payload.query.lower().split())
    filter_json = to_sorted_key_bytes(payload.metadata_filter.model_dump())
    raw = b"|".join(
        [
            kind.encode("utf-8"),
            user_id.encode("utf-8"),
            tenant_id.encode("utf-8"),
            normalized_query.encode("utf-8"),
            filter_json,
            to_sorted_key_bytes({"allowed_doc_ids": sorted(allowed_doc_ids)}),
            to_sorted_key_bytes(
                {
                    "document_versions": sorted(
                        (
                            (str(doc_id), int(version), pipeline)
                            for doc_id, version, pipeline in document_versions
                        ),
                        key=itemgetter(0),
                    )
                }
            ),
            evidence_policy.encode("utf-8"),
            str(payload.limit).encode("utf-8"),
            str(payload.candidate_limit).encode("utf-8"),
            CANONICAL_PIPELINE_VERSION.encode("utf-8"),
        ]
    )
    return "documents:" + hashlib.sha256(raw).hexdigest()


def _build_answer_cache_key(
    *,
    user_id: str,
    tenant_id: str | None = None,
    query: str,
    doc_ids_filter: list[str],
    jurisdiction: str | None,
    contract_type: str | None,
    clause_type: str | None,
    require_graphiti_verified: bool,
    contract_index: int | None = None,
    allowed_doc_ids: Sequence[str] = (),
    evidence_policy: EvidencePolicy | None = None,
    document_versions: Sequence[tuple[str, int, str | None]] = (),
) -> str:
    tenant_id = tenant_id or user_id
    raw = to_sorted_key_bytes(
        {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "query": " ".join(query.lower().split()),
            "doc_ids_filter": sorted(doc_ids_filter),
            "allowed_doc_ids": sorted(allowed_doc_ids),
            "document_versions": sorted(
                (
                    (str(doc_id), int(version), pipeline)
                    for doc_id, version, pipeline in document_versions
                ),
                key=itemgetter(0),
            ),
            "jurisdiction": jurisdiction,
            "contract_type": contract_type,
            "contract_index": contract_index,
            "clause_type": clause_type,
            "require_graphiti_verified": require_graphiti_verified,
            "evidence_policy": evidence_policy
            or ("graph_indexed_only" if require_graphiti_verified else "mechanically_verified"),
            "pipeline_version": CANONICAL_PIPELINE_VERSION,
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


def _context_contract_index(metadata: dict[str, object]) -> int | None:
    """Return a contract index only when a context section is contract-local."""
    value = metadata.get("contract_index")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


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
            "graph_indexed": bool(row.get("graphiti_verified")),
            "document_version": row.get("document_version", 1),
            "pipeline_version": row.get("pipeline_version"),
            "contract_index": row.get("contract_index"),
            "graph_indexed_at": row.get("graph_indexed_at"),
        },
        custom_metadata=cast("dict[str, object]", row.get("custom_metadata") or {}),
        score=float(cast("float | int | str", row["rrf_score"])),
    )


def _citation_response(citation: Citation, row: dict[str, object] | None) -> LegalCitationResponse:
    """Attach immutable source provenance to a generated citation when available."""
    chunk_id = citation.chunk_id
    response = LegalCitationResponse(
        chunk_id=chunk_id,
        clause_type=citation.clause_type,
        claim=citation.claim,
    )
    if row is None:
        return response
    return response.model_copy(
        update={
            "document_id": str(row["document_id"]) if row.get("document_id") else None,
            "document_version": int(cast("int | str", row.get("document_version", 1))),
            "contract_index": (
                int(cast("int | str", row["contract_index"]))
                if isinstance(row.get("contract_index"), (int, str))
                and not isinstance(row.get("contract_index"), bool)
                else None
            ),
            "pipeline_version": (
                str(row["pipeline_version"]) if row.get("pipeline_version") is not None else None
            ),
            "page_no": int(cast("int | str", row.get("page_no", 0))),
        }
    )


def _retrieved_contract_index(chunk_id: str, chunks: object) -> int | None:
    """Recover server provenance for graph citations without trusting the LLM."""
    if not isinstance(chunks, list):
        return None
    for chunk in chunks:
        if isinstance(chunk, dict):
            chunk_key = chunk.get("chunk_id")
            value = chunk.get("contract_index")
            metadata = chunk.get("metadata_")
        else:
            chunk_key = getattr(chunk, "chunk_id", None)
            value = getattr(chunk, "contract_index", None)
            metadata = getattr(chunk, "metadata_", None)
        if chunk_key != chunk_id:
            continue
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        if isinstance(metadata, dict):
            value = metadata.get("contract_index")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None
