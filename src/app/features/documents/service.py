"""Unified document feature services."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from returns.result import Failure, Success

from app.config import get_settings
from app.connections import init_db
from app.shared.langchain_layer import serialize_to_toon
from app.shared.langchain_layer.embeddings import EmbeddingTaskType, embed_text, embed_texts
from app.shared.langchain_layer.models import _build_chat_model
from app.shared.langgraph_layer.kb_retry import retry_immediate
from app.shared.langgraph_layer.retrieval_kb import (
    ContextGrade,
    CrossEncoderReranker,
    GeneratedAnswer,
    QueryPlan,
    RetrievedChunk,
    _extract_postgres_chunk_ids,
    build_retrieval_graph,
)
from app.shared.rag.graphiti import close_graphiti, setup_graphiti, setup_graphiti_indices
from app.shared.result import app_error_to_exception, log_expected_failure
from app.shared.services.storage import StorageService, build_s3_key, key_from_s3_uri
from app.utils import (
    NotFoundException,
    ServiceUnavailableException,
    ValidationException,
    logger,
    to_sorted_key_bytes,
)

from .classification import classify_document, segment_chunks
from .constants import (
    ANALYZE_THRESHOLD_CHUNKS,
    DEFAULT_SEARCH_CACHE_TTL_SECONDS,
    INGEST_EMBEDDING_BATCH_SIZE,
    RRF_K,
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
from .fusion import RankedChunk, RankedResultRow, reciprocal_rank_fusion
from .graphiti_verifier import write_and_verify_chunk
from .ingestion_graph import build_document_ingestion_graph
from .legal_metadata import (
    contract_event_dates,
    enrich_legal_chunks,
    extract_legal_metadata,
)
from .parser import parse_document
from .rag import SearchChunkRecord, assemble_rag_context
from .repository import DocumentRepository, build_chunk_rows, build_search_filter_params

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    from graphiti_core.graphiti import Graphiti
    from langchain_core.language_models import BaseChatModel
    from redis.asyncio import Redis
    from ty_extensions import Unknown

    from app.config.settings import Settings
    from app.features.documents.rag import ContextSection
    from app.shared.result import AppResult

    from . import dto as documents_dto
    from .classification import ClassifiedDocument, ParsedDocument, PreparedChunk, QualityWarning
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
# Positional, and `zip(..., strict=True)` below is what keeps it honest: these names label the
# three coroutines handed to `asyncio.gather` in order, and `gather` preserves argument order
# regardless of completion order. Reorder the gather without reordering this and the strict zip
# still passes while every failure is attributed to the wrong branch — so the pairing is asserted
# by a unit test, not by this comment.
_SEARCH_BRANCHES = ("bm25", "vector", "trigram")


class DocumentCommandService:
    """Create, queue, and inspect document ingestion jobs."""

    def __init__(
        self,
        repo: DocumentRepository,
        object_store: StorageService | None,
    ):
        self.repo: DocumentRepository = repo
        self.object_store: StorageService | None = object_store

    async def upload_document(
        self,
        *,
        user_id: str,
        filename: str,
        content_type: str,
        raw_bytes: bytes,
    ) -> DocumentUploadResponse:
        if not raw_bytes:
            message = "Uploaded document is empty"
            raise ValidationException(message)
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        existing_result = await self.repo.get_document_by_user_hash(
            user_id=user_id, content_hash=content_hash
        )
        if isinstance(existing_result, Success):
            existing = existing_result.unwrap()
            if existing is not None:
                return DocumentUploadResponse(
                    doc_id=str(existing.id),
                    status=existing.status,
                    duplicate=True,
                )
        elif isinstance(existing_result, Failure):
            log_expected_failure(error=existing_result.failure(), operation="document_upload")
            raise app_error_to_exception(existing_result.failure())

        document_id = str(object=uuid4())
        object_key = build_s3_key(
            prefix="documents",
            user_id=user_id,
            document_id=document_id,
            content_hash=content_hash,
            filename=filename,
        )
        if self.object_store is None:
            raise ServiceUnavailableException(detail="Object storage is not configured")
        object_uri: str = await self.object_store.put_object(
            key=object_key,
            data=raw_bytes,
            content_type=content_type,
            metadata={"user_id": user_id, "document_id": document_id, "content_hash": content_hash},
        )
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
        if isinstance(create_result, Success):
            document = create_result.unwrap()
        elif isinstance(create_result, Failure):
            log_expected_failure(error=create_result.failure(), operation="document_upload")
            raise app_error_to_exception(error=create_result.failure())

        from app.shared.outbox import (
            with_outbox,
        )

        await with_outbox(
            session=self.repo.session,
            aggregate_type="user_document",
            aggregate_id=str(object=document.id),
            event_type="tasks.documents_ingest",
            payload={
                "document_id": str(object=document.id),
                "user_id": user_id,
                "filename": filename,
                "content_type": content_type,
                "object_uri": object_uri,
            },
        )

        logger.bind(document_id=str(object=document.id)).info("documents_ingest_queued")
        return DocumentUploadResponse(
            doc_id=str(object=document.id),
            status="queued",
            task_id=None,
            object_uri=object_uri,
            document_kind=document.document_kind,
            warning_count=0,
        )

    async def get_status(self, *, user_id: str, document_id: str) -> DocumentStatusResponse:
        status_result: AppResult[dict[str, Any] | None] = await self.repo.fetch_status(
            user_id=user_id, document_id=document_id
        )
        if isinstance(status_result, Success):
            record: dict[str, Any] | None = status_result.unwrap()
            if record is not None:
                warnings: list[QualityWarningDTO] = _flatten_warnings(record.get("warnings", []))
                return DocumentStatusResponse(
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
        resource = "Document"
        raise NotFoundException(resource, document_id)


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

    async def search(self, *, user_id: str, payload: UnifiedSearchRequest) -> UnifiedSearchResponse:
        cache_key = _build_cache_key("documents:search", payload)
        lock_key = f"{cache_key}:lock"
        lock_acquired = False
        if not payload.bypass_cache and self.redis is not None:
            cached = await self.redis.get(cache_key)
            if cached is not None:
                response: UnifiedSearchResponse = UnifiedSearchResponse.model_validate_json(cached)
                return response.model_copy(update={"cache_hit": True})

            # ponytail: setnx lock prevents concurrent duplicate compute for same query
            lock_acquired = await self.redis.setnx(lock_key, "1")
            if not lock_acquired:
                for _ in range(30):
                    await asyncio.sleep(0.05)
                    cached = await self.redis.get(cache_key)
                    if cached is not None:
                        return UnifiedSearchResponse.model_validate_json(cached).model_copy(
                            update={"cache_hit": True}
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
        fused_result: AppResult[list[RankedChunk]] = await self._fuse_search_branches(
            user_id=user_id,
            payload=payload,
            query_embedding=query_embedding,
            filter_params=filter_params,
        )
        if isinstance(fused_result, Failure):
            error = fused_result.failure()
            log_expected_failure(error, operation="hybrid_search")
            # The setnx lock above is not released on this path. It carries a 15s expiry for
            # exactly this reason, and every other raise in this method already relied on it.
            raise app_error_to_exception(error)
        fused_results: list[RankedChunk] = fused_result.unwrap()
        chunk_lookup = await self.repo.fetch_chunks_by_ids(
            [item.chunk_id for item in fused_results]
        )
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
        return response

    async def _fuse_search_branches(
        self,
        *,
        user_id: str,
        payload: UnifiedSearchRequest,
        query_embedding: list[float],
        filter_params: dict[str, Any],
    ) -> AppResult[list[RankedChunk]]:
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
        results = await asyncio.gather(
            self.repo.bm25_search(
                user_id=user_id,
                query=payload.query,
                candidate_limit=payload.candidate_limit,
                filter_params=filter_params,
            ),
            self.repo.vector_search(
                user_id=user_id,
                embedding=query_embedding,
                candidate_limit=payload.candidate_limit,
                filter_params=filter_params,
            ),
            self.repo.trigram_search(
                user_id=user_id,
                query=payload.query,
                candidate_limit=payload.candidate_limit,
                filter_params=filter_params,
            ),
        )
        row_sets: list[list[RankedResultRow]] = []
        for branch, branch_result in zip(_SEARCH_BRANCHES, results, strict=True):
            if isinstance(branch_result, Failure):
                error = branch_result.failure()
                # `model_copy` rather than a fresh `InfrastructureAppError`: re-wrapping would
                # flatten the taxonomy and turn a 422 from one branch into a 503 for the whole
                # request. This keeps the branch's own error kind, so `app_error_to_exception`
                # still maps it to the status the branch earned, and only adds the attribution.
                return Failure(
                    error.model_copy(
                        update={
                            "message": f"{branch} retrieval branch failed: {error.message}",
                            "details": {**(error.details or {}), "branch": branch},
                        }
                    )
                )
            # `else`, not `elif isinstance(..., Success)`. The old form had no final branch, so a
            # value that was neither would have been dropped from `row_sets` entirely — shrinking
            # the fusion input with no log line and no failure.
            row_sets.append(_to_ranked_rows(branch_result.unwrap()))
        return Success(
            reciprocal_rank_fusion(
                *row_sets,
                k=RRF_K,
                limit=payload.limit,
            )
        )

    async def rag(
        self, *, user_id: str, payload: documents_dto.UnifiedRagRequest
    ) -> UnifiedRagResponse:
        response = await self.search(
            user_id=user_id, payload=UnifiedSearchRequest.model_validate(payload.model_dump())
        )
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
            ranked_chunks, chunk_lookup, max_tokens=payload.max_tokens
        )
        return UnifiedRagResponse(
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

    async def ask_via_retrieval_graph(
        self, *, user_id: str, payload: documents_dto.UnifiedAskRequest
    ) -> UnifiedAskResponse:
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
        return UnifiedAskResponse(
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

    async def ask(  # noqa: PLR0914
        self,
        *,
        user_id: str,
        payload: documents_dto.UnifiedAskRequest,
        require_graphiti_verified: bool,
    ) -> UnifiedAskResponse:
        answer_cache_key = _build_answer_cache_key(
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
                return response.model_copy(update={"cache_hit": True})

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
            rows = await self.repo.legal_rrf_search(
                user_id=user_id,
                query_text=plan.rewritten_query,
                query_embedding=embedding,
                limit=20,
                vector_weight=plan.vector_weight,
                keyword_weight=plan.keyword_weight,
                jurisdiction=payload.jurisdiction or plan.jurisdiction,
                contract_type=payload.contract_type or plan.contract_type,
                document_ids=payload.doc_ids_filter,
                chunk_ids=graph_chunk_ids or None,
                clause_type=payload.clause_type,
                require_graphiti_verified=require_graphiti_verified,
                bm25_threshold=plan.bm25_threshold,
                exact_phrase=plan.exact_phrase,
            )
            retrieved_chunks = [_row_to_chunk(row) for row in rows]
            reranked = await CrossEncoderReranker().rerank(
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
        return response


async def process_document_ingestion(
    *,
    document_id: str,
    user_id: str,
    filename: str,
    content_type: str,
    object_uri: str,
    object_store: StorageService,
    repo: DocumentRepository,
    graphiti: Graphiti | None,
    llm: BaseChatModel,
) -> dict[str, object]:
    raw_bytes = await object_store.get_object(key=key_from_s3_uri(object_uri))
    parsed: ParsedDocument = await parse_document(
        raw_bytes=raw_bytes, filename=filename, content_type=content_type
    )
    classified: ClassifiedDocument = classify_document(markdown=parsed.markdown, filename=filename)
    legal_metadata: LegalMetadataExtraction | None = None
    metadata_warnings: list[QualityWarning] = []
    if classified.graphiti_required:
        legal_metadata, metadata_warnings = await extract_legal_metadata(
            llm=llm,
            markdown=parsed.markdown,
            classified=classified,
        )
    await repo.update_document_status(
        document_id=document_id,
        status="parsed",
        title=parsed.title,
        document_kind=classified.document_kind,
        jurisdiction=(legal_metadata.jurisdiction if legal_metadata else classified.jurisdiction),
        contract_type=(
            legal_metadata.contract_type if legal_metadata else classified.contract_type
        ),
        parties=[*(legal_metadata.parties if legal_metadata else classified.parties)],
        metadata_={
            "content_type": content_type,
            "filename": filename,
            **classified.metadata_,
            **(legal_metadata.model_dump(exclude_none=True) if legal_metadata else {}),
        },
    )
    chunks, segmentation_warnings = await segment_chunks(parsed=parsed, classified=classified)
    if legal_metadata is not None:
        chunks = enrich_legal_chunks(
            chunks=chunks,
            classified=classified,
            metadata=legal_metadata,
        )
    chunk_rows = await _embed_chunks(
        user_id=user_id,
        document_id=document_id,
        chunks=chunks,
        extra_warnings=segmentation_warnings + classified.warnings + metadata_warnings,
    )
    upsert_result = await repo.upsert_chunks(
        build_chunk_rows(document_id=document_id, user_id=user_id, chunks=chunk_rows)
    )
    if isinstance(upsert_result, Failure):
        log_expected_failure(upsert_result.failure(), operation="document_ingestion")
        raise app_error_to_exception(upsert_result.failure())
    if len(chunk_rows) > ANALYZE_THRESHOLD_CHUNKS:
        await repo.analyze_chunks()
    await repo.update_document_status(document_id=document_id, status="stored_postgres")
    if classified.graphiti_required:
        if graphiti is not None and legal_metadata is not None:
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
                    continue
        verified_count: int = await _verify_legal_chunks(
            repo=repo,
            graphiti=graphiti,
            user_id=user_id,
            document_id=document_id,
            chunk_rows=chunk_rows,
        )
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
    await repo.update_document_status(document_id=document_id, status=final_status)
    return {
        "status": final_status,
        "document_id": document_id,
        "chunk_count": len(chunk_rows),
        "verified_chunk_count": verified_count,
        "document_kind": classified.document_kind,
    }


async def run_document_ingestion_task(
    *,
    document_id: str,
    user_id: str,
    filename: str,
    content_type: str,
    object_uri: str,
) -> dict[str, object]:
    engine, session_local = await init_db()
    settings: Settings = get_settings()
    object_store: StorageService = StorageService.from_settings(settings=settings)
    llm = _build_chat_model(
        model_name=settings.GEMINI_FLASH_MODEL,
        temperature=0.1,
        implementation="generic",
    )
    graphiti: Graphiti = await setup_graphiti(
        neo4j_uri=settings.NEO4J_URI,
        neo4j_user=settings.NEO4J_USERNAME,
        neo4j_password=settings.NEO4J_PASSWORD.get_secret_value(),
    )
    await setup_graphiti_indices(graphiti)
    try:
        async with session_local() as session, session.begin():
            repo = DocumentRepository(session)
            graph = build_document_ingestion_graph(
                object_store=object_store,
                repo=repo,
                graphiti=graphiti,
                ingest_document_fn=process_document_ingestion,
                llm=llm,
            )
            return await graph.ainvoke(
                {
                    "document_id": document_id,
                    "user_id": user_id,
                    "filename": filename,
                    "content_type": content_type,
                    "object_uri": object_uri,
                }
            )
    finally:
        await close_graphiti(graphiti)
        await engine.dispose()


async def _embed_chunks(
    *,
    user_id: str,
    document_id: str,
    chunks: list[PreparedChunk],
    extra_warnings: list[QualityWarning],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch in _batched(chunks, INGEST_EMBEDDING_BATCH_SIZE):
        embeddings = await embed_texts(
            [f"{chunk.preamble}\n\n{chunk.content}".strip() for chunk in batch],
            task_type=EmbeddingTaskType.DOCUMENT,
        )
        for chunk, embedding in zip(batch, embeddings, strict=True):
            rows.append(
                {
                    "id": str(uuid4()),
                    "chunk_index": chunk.chunk_index,
                    "chunk_kind": chunk.chunk_kind,
                    "content": chunk.content,
                    "preamble": chunk.preamble,
                    "clause_type": chunk.clause_type,
                    "page_no": chunk.page_no,
                    "embedding": embedding,
                    "metadata_": chunk.metadata_,
                    "custom_metadata": chunk.custom_metadata,
                    "quality_warnings": [
                        warning.model_dump()
                        for warning in [*chunk.quality_warnings, *extra_warnings]
                    ],
                }
            )
    _ = (user_id, document_id)
    return rows


async def _verify_legal_chunks(
    *,
    repo: DocumentRepository,
    graphiti: Graphiti | None,
    user_id: str,
    document_id: str,
    chunk_rows: list[dict[str, object]],
) -> int:
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
        raise app_error_to_exception(upsert_result.failure())
    return verified_count


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


def _build_cache_key(kind: str, payload: UnifiedSearchRequest) -> str:
    normalized_query = " ".join(payload.query.lower().split())
    filter_json = to_sorted_key_bytes(payload.metadata_filter.model_dump())
    raw = b"|".join(
        [
            kind.encode("utf-8"),
            normalized_query.encode("utf-8"),
            filter_json,
            str(payload.limit).encode("utf-8"),
            str(payload.candidate_limit).encode("utf-8"),
        ]
    )
    return "documents:" + hashlib.sha256(raw).hexdigest()


def _build_answer_cache_key(
    *,
    query: str,
    doc_ids_filter: list[str],
    jurisdiction: str | None,
    contract_type: str | None,
    clause_type: str | None,
    require_graphiti_verified: bool,
) -> str:
    raw = to_sorted_key_bytes(
        {
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
