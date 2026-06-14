"""Unified document feature services."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.connections import celery_app, init_db
from app.features.search import (
    ANALYZE_THRESHOLD_CHUNKS,
    DEFAULT_SEARCH_CACHE_TTL_SECONDS,
    INGEST_EMBEDDING_BATCH_SIZE,
    RRF_K,
    RankedChunk,
    RankedResultRow,
    SearchChunkRecord,
    assemble_rag_context,
    build_embedding_client,
    reciprocal_rank_fusion,
)
from app.shared.langchain_layer import serialize_to_toon
from app.shared.langgraph_layer.kb_retry import retry_immediate
from app.shared.langgraph_layer.retrieval_kb import (
    ContextGrade,
    CrossEncoderReranker,
    GeneratedAnswer,
    QueryPlan,
    RetrievedChunk,
    _extract_postgres_chunk_ids,
)
from app.shared.rag.graphiti import close_graphiti, setup_graphiti, setup_graphiti_indices
from app.shared.services.storage import StorageService, build_s3_key, key_from_s3_uri
from app.utils import NotFoundException, ServiceUnavailableException, ValidationException, logger
from app.utils.json_serializer import from_json_float_list, to_float_list_str, to_sorted_key_bytes

from .classification import classify_document, segment_chunks
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
from .graphiti_verifier import write_and_verify_chunk
from .ingestion_graph import build_document_ingestion_graph
from .legal_metadata import (
    contract_event_dates,
    enrich_legal_chunks,
    extract_legal_metadata,
)
from .parser import parse_document
from .repository import DocumentRepository, build_chunk_rows, build_search_filter_params

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from graphiti_core.graphiti import Graphiti
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ty_extensions import Unknown

    from app.config.settings import Settings

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


class DocumentCommandService:
    """Create, queue, and inspect document ingestion jobs."""

    def __init__(
        self,
        repo: DocumentRepository,
        object_store: StorageService,
    ):
        self.repo = repo
        self.object_store = object_store

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
        existing = await self.repo.get_document_by_user_hash(
            user_id=user_id, content_hash=content_hash
        )
        if existing is not None:
            return DocumentUploadResponse(
                doc_id=str(existing.id),
                status=existing.status,
                duplicate=True,
            )

        document_id = str(uuid4())
        object_key = build_s3_key(
            prefix="documents",
            user_id=user_id,
            document_id=document_id,
            content_hash=content_hash,
            filename=filename,
        )
        object_uri = await self.object_store.put_object(
            key=object_key,
            data=raw_bytes,
            content_type=content_type,
            metadata={"user_id": user_id, "document_id": document_id, "content_hash": content_hash},
        )
        document = await self.repo.create_document(
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

        try:
            task = celery_app.send_task(
                "tasks.documents_ingest",
                kwargs={
                    "document_id": str(document.id),
                    "user_id": user_id,
                    "filename": filename,
                    "content_type": content_type,
                    "object_uri": object_uri,
                },
            )
        except Exception as exc:
            raise ServiceUnavailableException(
                detail="Task queue unavailable",
                data={"document_id": str(document.id)},
            ) from exc

        logger.bind(document_id=str(document.id), task_id=task.id).info("documents_ingest_queued")
        return DocumentUploadResponse(
            doc_id=str(document.id),
            status="queued",
            task_id=task.id,
            object_uri=object_uri,
            document_kind=document.document_kind,
            warning_count=0,
        )

    async def get_status(self, *, user_id: str, document_id: str) -> DocumentStatusResponse:
        record = await self.repo.fetch_status(user_id=user_id, document_id=document_id)
        if record is None:
            resource = "Document"
            raise NotFoundException(resource, document_id)
        warnings = _flatten_warnings(record.get("warnings", []))
        return DocumentStatusResponse(
            doc_id=str(record["document_id"]),
            status=str(record["status"]),
            object_uri=str(record["object_uri"]),
            title=str(record["title"]),
            document_kind=str(record["document_kind"]),
            chunk_count=int(record["chunk_count"]),
            verified_chunk_count=int(record["verified_chunk_count"]),
            warning_count=len(warnings),
            warnings=warnings,
        )


class DocumentQueryService:
    """Unified retrieval and grounded QA service."""

    def __init__(
        self,
        repo: DocumentRepository,
        redis: object | None,
        graphiti: object | None,
    ):
        self.repo = repo
        self.redis = redis
        self.graphiti = graphiti

    async def search(self, *, user_id: str, payload: UnifiedSearchRequest) -> UnifiedSearchResponse:
        cache_key = _build_cache_key("documents:search", payload)
        if not payload.bypass_cache and self.redis is not None:
            cached = await self.redis.get(cache_key)
            if cached is not None:
                response = UnifiedSearchResponse.model_validate_json(cached)
                return response.model_copy(update={"cache_hit": True})

        embedding_client: GoogleGenerativeAIEmbeddings = build_embedding_client()
        query_embedding = await embedding_client.aembed_query(
            payload.query, task_type="RETRIEVAL_QUERY"
        )
        filter_params = build_search_filter_params(
            metadata_filter=payload.metadata_filter.model_dump()
        )
        bm25_results, vector_results, trigram_results = await asyncio.gather(
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
        fused_results = reciprocal_rank_fusion(
            _to_ranked_rows(bm25_results),
            _to_ranked_rows(vector_results),
            _to_ranked_rows(trigram_results),
            k=RRF_K,
            limit=payload.limit,
        )
        chunk_lookup = await self.repo.fetch_chunks_by_ids(
            [item.chunk_id for item in fused_results]
        )
        items = _build_search_items(fused_results=fused_results, chunk_lookup=chunk_lookup)
        response = UnifiedSearchResponse(items=items, cache_hit=False)
        if not payload.bypass_cache and self.redis is not None:
            await self.redis.setex(
                cache_key, DEFAULT_SEARCH_CACHE_TTL_SECONDS, response.model_dump_json()
            )
        return response

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
        ranked_chunks = [
            RankedChunk(chunk_id=item.chunk_id, score=item.score, rank=item.rank)
            for item in response.items
        ]
        context_sections = assemble_rag_context(
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

    async def ask(
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
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_FLASH_MODEL,
            api_key=settings.GEMINI_API_KEY.get_secret_value()
            if settings.GEMINI_API_KEY.get_secret_value()
            else None,
            temperature=0.1,
            retries=0,
        )
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
            embedding = await _cached_embedding(
                self.redis, build_embedding_client(), plan.rewritten_query
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
    graphiti: object | None,
) -> dict[str, object]:
    raw_bytes = await object_store.get_object(key=key_from_s3_uri(object_uri))
    parsed: ParsedDocument = await parse_document(
        raw_bytes=raw_bytes, filename=filename, content_type=content_type
    )
    classified: ClassifiedDocument = classify_document(markdown=parsed.markdown, filename=filename)
    legal_metadata: LegalMetadataExtraction | None = None
    metadata_warnings: list[QualityWarning] = []
    if classified.graphiti_required:
        llm = ChatGoogleGenerativeAI(
            model=get_settings().GEMINI_FLASH_MODEL,
            api_key=get_settings().GEMINI_API_KEY.get_secret_value()
            if get_settings().GEMINI_API_KEY.get_secret_value()
            else None,
            temperature=0.1,
            retries=0,
        )
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
    chunks, segmentation_warnings = segment_chunks(parsed=parsed, classified=classified)
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
    await repo.upsert_chunks(
        build_chunk_rows(document_id=document_id, user_id=user_id, chunks=chunk_rows)
    )
    if len(chunk_rows) > ANALYZE_THRESHOLD_CHUNKS:
        await repo.analyze_chunks()
    await repo.update_document_status(document_id=document_id, status="stored_postgres")
    if classified.graphiti_required:
        if graphiti is not None and legal_metadata is not None:
            for event_name, event_date in contract_event_dates(legal_metadata):
                try:
                    await graphiti.add_episode(  # type: ignore[attr-defined]
                        name=f"{event_name}:{document_id}:{event_date}",
                        episode_body=f"{event_name} for {document_id} occurs on {event_date}.",
                        source_description=(
                            "{"
                            f'"doc_id":"{document_id}",'
                            f'"event_type":"{event_name}",'
                            f'"event_date":"{event_date}"'
                            "}"
                        ),
                        reference_time=None,
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
    embedding_client: GoogleGenerativeAIEmbeddings = build_embedding_client()
    rows: list[dict[str, object]] = []
    for batch in _batched(chunks, INGEST_EMBEDDING_BATCH_SIZE):
        embeddings = await embedding_client.aembed_documents(
            [f"{chunk.preamble}\n\n{chunk.content}".strip() for chunk in batch],
            task_type="RETRIEVAL_DOCUMENT",
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
    graphiti: object | None,
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
    await repo.upsert_chunks(
        build_chunk_rows(document_id=document_id, user_id=user_id, chunks=chunk_rows)
    )
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
    graphiti: Any | None,
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
            ),  # type: ignore[attr-defined]
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


async def _cached_embedding(
    redis: object | None,
    embedding_fn: object,
    text_to_embed: str,
) -> list[float]:
    cache_key = "documents:embedding:" + hashlib.sha256(text_to_embed.encode("utf-8")).hexdigest()
    if redis is not None:
        cached = await redis.get(cache_key)
        if cached:
            raw = cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)
            return from_json_float_list(raw)
    embedding = await retry_immediate(
        lambda: embedding_fn.aembed_query(text_to_embed, task_type="RETRIEVAL_QUERY"),
        label="documents_query_embedding",
    )
    if redis is not None:
        await redis.setex(cache_key, 60 * 60 * 24, to_float_list_str(embedding))
    return _normalize_embedding(list(embedding))


def _normalize_embedding(embedding: list[float]) -> list[float]:
    if len(embedding) == 768:
        return embedding
    if len(embedding) > 768:
        return embedding[:768]
    return [*embedding, *([0.0] * (768 - len(embedding)))]


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
                quality_warnings=_flatten_warnings(row.get("quality_warnings", [])),
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


def _flatten_warnings(raw_groups: object) -> list[QualityWarningDTO]:
    if not isinstance(raw_groups, list):
        return []
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
            merged[(warning.stage, warning.code, warning.message, warning.severity)] = warning
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
            "graphiti_verified": bool(row.get("graphiti_verified", False)),
        },
        custom_metadata=cast("dict[str, object]", row.get("custom_metadata") or {}),
        score=float(cast("float | int | str", row["rrf_score"])),
    )
