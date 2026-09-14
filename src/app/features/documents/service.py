"""Unified document feature services."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, NamedTuple, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from returns.result import Failure, Success

from app.config import get_settings
from app.connections.celery_task_names import DOCUMENTS_INGEST
from app.shared.langchain_layer import serialize_to_toon
from app.shared.langchain_layer.embeddings import EmbeddingTaskType, embed_text, embed_texts
from app.shared.langgraph_layer.kb_retry import retry_immediate
from app.shared.langgraph_layer.retrieval_kb import (
    ContextGrade,
    GeneratedAnswer,
    QueryPlan,
    RetrievedChunk,
    _extract_postgres_chunk_ids,
    build_retrieval_graph,
    get_shared_reranker,
)
from app.shared.rag.langextract.service import (
    ExtractionFailed,
    ExtractionFailureCode,
    ExtractionRequest,
    ExtractionSucceeded,
)
from app.shared.rag.structural_navigator import navigate_tree
from app.shared.rag.token_counter import count_tokens as default_count_tokens
from app.shared.result import log_expected_failure
from app.shared.services.storage import build_s3_key, key_from_s3_uri
from app.utils import logger, to_sorted_key_bytes, trace_layer

from .classification import classify_document, segment_chunks
from .constants import (
    ANALYZE_THRESHOLD_CHUNKS,
    DEFAULT_SEARCH_CACHE_TTL_SECONDS,
    HYBRID_CANDIDATE_LIMIT,
    INGEST_EMBEDDING_BATCH_SIZE,
    RRF_K,
    RRF_WEIGHT_TRIGRAM,
)
from .dto import (
    DocumentSearchResultItem,
    DocumentStatusResponse,
    DocumentUploadResponse,
    LegalCitationResponse,
    QualityWarningDTO,
    RagContextSectionResponse,
    UnifiedAskResponse,
    UnifiedRagResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from .errors import (
    DocumentNotFoundError,
    DocumentStorageError,
    DocumentValidationError,
)
from .fusion import RankedChunk, RankedResultRow, reciprocal_rank_fusion
from .graphiti_verifier import write_and_verify_chunk
from .legal_metadata import (
    contract_event_dates,
    enrich_legal_chunks,
    extract_legal_metadata,
)
from .parser import parse_document
from .rag import SearchChunkRecord, assemble_rag_context
from .repository import DocumentRepository, build_chunk_rows, build_search_filter_params

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from typing import Any, Literal

    from graphiti_core.graphiti import Graphiti
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from ty_extensions import Unknown

    from app.config.settings import Settings
    from app.features.documents.rag import ContextSection
    from app.shared.langchain_layer.agents.tools.idempotency import IdempotencyGuard
    from app.shared.rag.graphiti.client import GraphitiService
    from app.shared.services.storage import StorageService

    from . import dto as documents_dto
    from .classification import ClassifiedDocument, ParsedDocument, PreparedChunk, QualityWarning
    from .dto import IngestionJob, IngestionRuntime
    from .errors import DocumentResult
    from .legal_metadata import (
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
    fused = reciprocal_rank_fusion(*results.unwrap(), k=RRF_K, limit=limit, weights=weights)
    if not fused:
        return Success(([], {}))
    chunk_lookup_result = await repo.fetch_chunks_by_ids([item.chunk_id for item in fused])
    if isinstance(chunk_lookup_result, Failure):
        return Failure(chunk_lookup_result.failure())
    lookup: dict[str, dict[str, Any]] = chunk_lookup_result.unwrap()
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


class DocumentCommandService:
    """Create, queue, and inspect document ingestion jobs."""

    def __init__(
        self,
        repo: DocumentRepository,
        object_store: StorageService | None,
    ):
        self.repo: DocumentRepository = repo
        self.object_store: StorageService | None = object_store

    @trace_layer("service")
    async def upload_document(
        self,
        *,
        user_id: str,
        filename: str,
        content_type: str,
        raw_bytes: bytes,
    ) -> DocumentResult[DocumentUploadResponse]:
        if not raw_bytes:
            return Failure(
                DocumentValidationError(
                    message="Uploaded document is empty", source="document_service"
                )
            )
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        existing_result = await self.repo.get_document_by_user_hash(
            user_id=user_id, content_hash=content_hash
        )
        if isinstance(existing_result, Success):
            existing = existing_result.unwrap()
            if existing is not None:
                return Success(
                    DocumentUploadResponse(
                        doc_id=str(existing.id),
                        status=existing.status,
                        duplicate=True,
                    )
                )
        elif isinstance(existing_result, Failure):
            failure = existing_result.failure()
            # A miss on the duplicate check is the NORMAL first-upload path,
            # not an error — only real infrastructure failures abort here.
            if not isinstance(failure, DocumentNotFoundError):
                log_expected_failure(error=failure, operation="document_upload")
                return Failure(failure)

        document_id = str(object=uuid4())
        object_key = build_s3_key(
            prefix="documents",
            user_id=user_id,
            document_id=document_id,
            content_hash=content_hash,
            filename=filename,
        )
        if self.object_store is None:
            return Failure(
                DocumentStorageError(
                    message="Object storage is not configured", source="document_service"
                )
            )
        storage_result = await self.object_store.put_object(
            key=object_key,
            data=raw_bytes,
            content_type=content_type,
            metadata={"user_id": user_id, "document_id": document_id, "content_hash": content_hash},
        )
        if isinstance(storage_result, Failure):
            error = storage_result.failure()
            return Failure(
                DocumentStorageError(
                    message=error.message,
                    details=error.details,
                    source="object_storage",
                )
            )
        object_uri = storage_result.unwrap()
        create_result = await self.repo.create_document(
            user_id=user_id,
            title=filename,
            source_uri=filename,
            object_uri=object_uri,
            content_hash=content_hash,
            document_kind="generic",
            status="received",
            jurisdiction=None,
            contract_type=None,
            parties=[],
            metadata_={"content_type": content_type, "filename": filename},
        )
        if isinstance(create_result, Failure):
            log_expected_failure(error=create_result.failure(), operation="document_upload")
            return Failure(create_result.failure())
        document = create_result.unwrap()

        from app.shared.outbox import (
            with_outbox,
        )

        await with_outbox(
            session=self.repo.session,
            aggregate_type="user_document",
            aggregate_id=str(object=document.id),
            event_type=DOCUMENTS_INGEST,
            payload={
                "document_id": str(object=document.id),
                "user_id": user_id,
                "filename": filename,
                "content_type": content_type,
                "object_uri": object_uri,
            },
        )

        logger.bind(document_id=str(object=document.id)).info("documents_ingest_queued")
        return Success(
            DocumentUploadResponse(
                doc_id=str(object=document.id),
                status="queued",
                task_id=None,
                object_uri=object_uri,
                document_kind=document.document_kind,
                warning_count=0,
            )
        )

    @trace_layer("service")
    async def get_status(
        self, *, user_id: str, document_id: str
    ) -> DocumentResult[DocumentStatusResponse]:
        status_result = await self.repo.fetch_status(user_id=user_id, document_id=document_id)
        if isinstance(status_result, Success):
            record: dict[str, Any] | None = status_result.unwrap()
            if record is not None:
                warnings: list[QualityWarningDTO] = _flatten_warnings(record.get("warnings", []))
                return Success(
                    DocumentStatusResponse(
                        doc_id=str(object=record["document_id"]),
                        status=str(object=record["status"]),
                        object_uri=str(object=record["object_uri"]),
                        title=str(object=record["title"]),
                        document_kind=str(object=record["document_kind"]),
                        chunk_count=int(record["chunk_count"]),
                        verified_chunk_count=int(record["verified_chunk_count"]),
                        warning_count=len(warnings),
                        warnings=warnings,
                    )
                )
        if isinstance(status_result, Failure):
            return Failure(status_result.failure())
        return Failure(
            DocumentNotFoundError(
                message="Document not found",
                details={"document_id": document_id},
                source="document_service",
            )
        )


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
            paths.extend(navigate_tree(tree, query, limit=limit))
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
        machine in `shared/langgraph_layer/retrieval_kb/` — and its only caller in the tree was
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


async def _load_document_bytes(
    object_store: StorageService, object_uri: str
) -> DocumentResult[bytes]:
    key_result = key_from_s3_uri(object_uri)
    if isinstance(key_result, Failure):
        error = key_result.failure()
        return Failure(
            DocumentValidationError(
                message=error.message,
                details=error.details,
                source="object_storage",
            )
        )
    object_result = await object_store.get_object(key=key_result.unwrap())
    if isinstance(object_result, Failure):
        return Failure(
            DocumentStorageError(
                message=object_result.failure().message,
                details=object_result.failure().details,
                source="object_storage",
            )
        )
    return Success(object_result.unwrap())


@trace_layer("service")
async def process_document_ingestion(
    *,
    job: IngestionJob,
    runtime: IngestionRuntime,
) -> DocumentResult[dict[str, object]]:
    raw_result = await _load_document_bytes(runtime.object_store, job.object_uri)
    if isinstance(raw_result, Failure):
        return raw_result
    raw_bytes = raw_result.unwrap()
    parsed: ParsedDocument = await parse_document(
        raw_bytes=raw_bytes, filename=job.filename, content_type=job.content_type
    )
    classified: ClassifiedDocument = classify_document(
        markdown=parsed.markdown, filename=job.filename
    )
    legal = await _prepare_legal_metadata(
        runtime=runtime,
        markdown=parsed.markdown,
        classified=classified,
    )
    extraction_outcome = await _extract_document_knowledge(
        runtime=runtime,
        job=job,
        markdown=parsed.markdown,
        jurisdiction=(legal.metadata.jurisdiction if legal.metadata else classified.jurisdiction),
        document_type=classified.document_kind,
    )
    status_result = await runtime.repo.update_document_status(
        document_id=job.document_id,
        status="parsed",
        title=parsed.title,
        document_kind=classified.document_kind,
        jurisdiction=(legal.metadata.jurisdiction if legal.metadata else classified.jurisdiction),
        contract_type=(
            legal.metadata.contract_type if legal.metadata else classified.contract_type
        ),
        parties=[*(legal.metadata.parties if legal.metadata else classified.parties)],
        metadata_={
            "content_type": job.content_type,
            "filename": job.filename,
            **classified.metadata_,
            **(legal.metadata.model_dump(exclude_none=True) if legal.metadata else {}),
        },
        structural_tree=parsed.structural_tree,
        extraction_incomplete=isinstance(extraction_outcome, ExtractionFailed),
    )
    if isinstance(status_result, Failure):
        return Failure(status_result.failure())
    chunks, segmentation_warnings = await segment_chunks(parsed=parsed, classified=classified)
    if legal.metadata is not None:
        chunks = enrich_legal_chunks(
            chunks=chunks,
            classified=classified,
            metadata=legal.metadata,
        )
    chunk_rows = await _embed_chunks(
        user_id=job.user_id,
        document_id=job.document_id,
        chunks=chunks,
        extra_warnings=segmentation_warnings + classified.warnings + legal.warnings,
        extraction_outcome=extraction_outcome,
    )
    upsert_result = await runtime.repo.upsert_chunks(
        build_chunk_rows(document_id=job.document_id, user_id=job.user_id, chunks=chunk_rows)
    )
    if isinstance(upsert_result, Failure):
        log_expected_failure(upsert_result.failure(), operation="document_ingestion")
        return Failure(upsert_result.failure())
    if len(chunk_rows) > ANALYZE_THRESHOLD_CHUNKS:
        analyze_result = await runtime.repo.analyze_chunks()
        if isinstance(analyze_result, Failure):
            return Failure(analyze_result.failure())
    status_result = await runtime.repo.update_document_status(
        document_id=job.document_id, status="stored_postgres"
    )
    if isinstance(status_result, Failure):
        return Failure(status_result.failure())
    if classified.graphiti_required:
        await _write_contract_events(runtime.graphiti, legal.metadata, job.document_id)
        verify_result = await _verify_legal_chunks(
            repo=runtime.repo,
            graphiti=runtime.graphiti,
            user_id=job.user_id,
            document_id=job.document_id,
            chunk_rows=chunk_rows,
        )
        if isinstance(verify_result, Failure):
            return Failure(verify_result.failure())
        verified_count = verify_result.unwrap()
        final_status: Literal["completed", "completed_with_warnings"] = (
            "completed" if verified_count == len(chunk_rows) else "completed_with_warnings"
        )
    else:
        final_status = (
            "completed"
            if not segmentation_warnings and not classified.warnings
            else "completed_with_warnings"
        )
        verified_count = 0
    status_result = await runtime.repo.update_document_status(
        document_id=job.document_id, status=final_status
    )
    if isinstance(status_result, Failure):
        return Failure(status_result.failure())
    return Success(
        {
            "status": final_status,
            "document_id": job.document_id,
            "chunk_count": len(chunk_rows),
            "verified_chunk_count": verified_count,
            "document_kind": classified.document_kind,
        }
    )


async def _prepare_legal_metadata(
    *,
    runtime: IngestionRuntime,
    markdown: str,
    classified: ClassifiedDocument,
) -> _LegalPreparation:
    if not classified.graphiti_required:
        return _LegalPreparation(metadata=None, warnings=[])
    metadata, warnings = await extract_legal_metadata(
        llm=runtime.llm,
        markdown=markdown,
        classified=classified,
    )
    return _LegalPreparation(metadata=metadata, warnings=warnings)


async def _write_contract_events(
    graphiti: Graphiti | None,
    legal_metadata: LegalMetadataExtraction | None,
    document_id: str,
) -> None:
    if graphiti is None or legal_metadata is None:
        return
    for event_name, event_date in contract_event_dates(legal_metadata):
        try:
            await graphiti.add_episode(
                name=f"{event_name}:{document_id}:{event_date}",
                episode_body=f"{event_name} for {document_id} occurs on {event_date}.",
                source_description=(
                    "{"
                    f'"doc_id":"{document_id}",'
                    f'"event_type":"{event_name}",'
                    f'"event_date":"{event_date}"'
                    "}"
                ),
                reference_time=None,  # ty: ignore[invalid-argument-type]
                group_id=document_id,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.bind(
                document_id=document_id,
                event_name=event_name,
                event_date=event_date,
            ).warning("graphiti_event_episode_failed", error=str(exc))


@trace_layer("service")
async def run_document_ingestion_task(
    *,
    document_id: str,
    user_id: str,
    filename: str,
    content_type: str,
    object_uri: str,
    graph: CompiledStateGraph[Any],
    session_local: async_sessionmaker[Any],
) -> dict[str, object]:
    async with session_local() as session, session.begin():
        repo = DocumentRepository(session)
        return await graph.ainvoke(
            {
                "document_id": document_id,
                "user_id": user_id,
                "filename": filename,
                "content_type": content_type,
                "object_uri": object_uri,
            },
            {"configurable": {"document_repository": repo}},
        )


async def _embed_chunks(
    *,
    user_id: str,
    document_id: str,
    chunks: list[PreparedChunk],
    extra_warnings: list[QualityWarning],
    extraction_outcome: ExtractionSucceeded | ExtractionFailed,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch in _batched(chunks, INGEST_EMBEDDING_BATCH_SIZE):
        embeddings = await embed_texts(
            [f"{chunk.preamble}\n\n{chunk.content}".strip() for chunk in batch],
            task_type=EmbeddingTaskType.DOCUMENT,
        )
        for chunk, embedding in zip(batch, embeddings, strict=True):
            extraction_metadata = _extraction_metadata(extraction_outcome)
            rows.append(
                {
                    "id": str(uuid4()),
                    "chunk_index": chunk.chunk_index,
                    "document_version": chunk.document_version,
                    "chunk_kind": chunk.chunk_kind,
                    "content": chunk.content,
                    "preamble": chunk.preamble,
                    "locus": chunk.locus,
                    "clause_type": chunk.clause_type,
                    "page_no": chunk.page_no,
                    "embedding": embedding,
                    "metadata_": chunk.metadata_,
                    "custom_metadata": {**chunk.custom_metadata, **extraction_metadata},
                    "quality_warnings": [
                        warning.model_dump()
                        for warning in [*chunk.quality_warnings, *extra_warnings]
                    ],
                }
            )
    _ = (user_id, document_id)
    return rows


def _extraction_metadata(
    outcome: ExtractionSucceeded | ExtractionFailed,
) -> dict[str, object]:
    if isinstance(outcome, ExtractionFailed):
        return {
            "knowledge_extraction": "incomplete",
            "knowledge_extraction_failure": str(outcome.code),
        }
    classes = sorted(
        {
            extraction.extraction_class
            for document in outcome.documents
            for extraction in document.extractions
        }
    )
    return {
        "knowledge_extraction": "complete",
        "knowledge_extraction_classes": classes,
    }


async def _extract_document_knowledge(
    *,
    runtime: IngestionRuntime,
    job: IngestionJob,
    markdown: str,
    jurisdiction: str | None,
    document_type: str,
) -> ExtractionSucceeded | ExtractionFailed:
    if runtime.extraction is None:
        return ExtractionFailed(
            code=ExtractionFailureCode.UNCONFIGURED,
            message="LangExtract provider is not configured",
        )
    outcome = await runtime.extraction.extract(
        ExtractionRequest(text=markdown, prompt_description=_KNOWLEDGE_EXTRACTION_PROMPT)
    )
    if isinstance(outcome, ExtractionSucceeded):
        await _write_extracted_clause_episodes(
            outcome=outcome,
            graph_writer=runtime.graph_writer,
            idempotency=runtime.idempotency,
            document_id=job.document_id,
            user_id=job.user_id,
            jurisdiction=jurisdiction,
            document_type=document_type,
        )
    return outcome


async def _write_extracted_clause_episodes(
    *,
    outcome: ExtractionSucceeded,
    graph_writer: object | None,
    idempotency: object | None,
    document_id: str,
    user_id: str,
    jurisdiction: str | None,
    document_type: str,
) -> None:
    """Map grounded clause extractions into the existing canonical writer."""
    if graph_writer is None or idempotency is None:
        return
    from app.shared.langgraph_layer.agent_saul.state import ClauseSegment, ClauseType
    from app.shared.rag.graphiti.write_clause_episodes import write_clause_episodes_to_graphiti
    from app.shared.rag.langextract.langextract_to_graph import GraphIngestionContext

    segments: list[ClauseSegment] = []
    valid_types = {member.value: member for member in ClauseType}
    for document in outcome.documents:
        for extraction in document.extractions:
            interval = extraction.char_interval
            if interval is None or interval.start_pos is None or interval.end_pos is None:
                continue
            attributes = extraction.attributes or {}
            raw_type = str(attributes.get("clause_type") or extraction.extraction_class).casefold()
            clause_type = valid_types.get(raw_type, ClauseType.OTHER)
            raw_id = attributes.get("clause_id") or attributes.get("clause_number")
            clause_id = (
                str(raw_id)
                if raw_id
                else hashlib.sha256(
                    f"{document_id}:{interval.start_pos}:{interval.end_pos}:{extraction.extraction_text}".encode()
                ).hexdigest()[:20]
            )
            segments.append(
                ClauseSegment(
                    clause_id=clause_id,
                    clause_type=clause_type,
                    text=extraction.extraction_text,
                    section_ref=str(attributes.get("section_ref") or clause_id),
                    start_char=interval.start_pos,
                    end_char=interval.end_pos,
                )
            )
    if not segments:
        return
    await write_clause_episodes_to_graphiti(
        segments,
        [],
        GraphIngestionContext(
            document_id=document_id,
            user_id=user_id,
            thread_id=f"ingestion:{document_id}",
            jurisdiction=jurisdiction or "unspecified",
            document_type=document_type,
        ),
        graphiti_service=cast("GraphitiService", graph_writer),
        idempotency=cast("IdempotencyGuard", idempotency),
    )


async def _verify_legal_chunks(
    *,
    repo: DocumentRepository,
    graphiti: Graphiti | None,
    user_id: str,
    document_id: str,
    chunk_rows: list[dict[str, object]],
) -> DocumentResult[int]:
    verified_count = 0
    for chunk in chunk_rows:
        result = await write_and_verify_chunk(
            graphiti=graphiti,
            user_id=user_id,
            document_id=document_id,
            chunk_id=str(chunk["id"]),
            clause_type=str(chunk["clause_type"]) if chunk.get("clause_type") is not None else None,
            preamble=str(chunk.get("preamble", "")),
            content=str(chunk["content"]),
        )
        chunk["graphiti_episode_id"] = result.episode_id
        chunk["graphiti_verified"] = result.verified
        if result.verified:
            verified_count += 1
    upsert_result = await repo.upsert_chunks(
        build_chunk_rows(document_id=document_id, user_id=user_id, chunks=chunk_rows)
    )
    if isinstance(upsert_result, Failure):
        log_expected_failure(upsert_result.failure(), operation="verify_legal_chunks")
        return Failure(upsert_result.failure())
    return Success(verified_count)


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
