from __future__ import annotations

import hashlib
import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import asyncer
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from graphiti_core.nodes import EpisodeType
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.langchain_layer.models import serialize_to_toon
from app.shared.langgraph_layer.kb_retry import retry_immediate
from app.shared.rag.graphiti.schemas import (
    GRAPHITI_EDGE_TYPE_MAP,
    GRAPHITI_EDGE_TYPES,
    GRAPHITI_ENTITY_TYPES,
)
from app.utils import logger

from .prompt import (
    CLASSIFY_EXTRACT_SYSTEM_PROMPT,
    CONTEXTUALIZE_CHUNK_SYSTEM_PROMPT,
    EXTRACT_SCHEMA_SYSTEM_PROMPT,
    SEGMENT_DOCUMENT_SYSTEM_PROMPT,
)
from .state import (
    ClauseSegment,
    ClauseSegmentationResult,
    ContextualizedChunk,
    ContractMetadata,
    EntityExtractionResult,
    ParsedDocument,
    StoredChunk,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from docling_core.types.doc.document import DoclingDocument
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy.sql.elements import TextClause
    from ty_extensions import Unknown

    from .state import (
        EmbeddingFunction,
        IngestionState,
        StructuredRunnable,
    )

_DEFAULT_LIMIT = 20
# refactor document parser, grapgiti then use the updated code here in ingestion pipeline node, then modify the model.py, tool definations, retry logic, p
# think about the retrival_kb afterwards

def make_parse_document_node() -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def parse_document_node(state: IngestionState) -> dict[str, object]:
        if not state.raw_bytes:
            return {"error": "empty_document", "ingestion_complete": False}

        parsed: ParsedDocument = await retry_immediate(
            lambda: _parse_document_with_docling(state.raw_bytes, state.filename, state.source),
            label="docling_parse_document",
        )
        return {"parsed_document": parsed}

    return parse_document_node


def make_extract_schema_node(
    schema_llm: StructuredRunnable,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def extract_schema_node(state: IngestionState) -> dict[str, object]:
        parsed = state.parsed_document
        if parsed is None:
            return {"error": "missing_parsed_document", "ingestion_complete": False}

        payload = serialize_to_toon(
            {
                "document_type": state.document_type,
                "jurisdiction_hint": state.jurisdiction,
                "source": parsed.source,
                "markdown": parsed.markdown[:40_000],
            }
        )
        messages = [
            SystemMessage(content=EXTRACT_SCHEMA_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
        metadata = await retry_immediate(
            lambda: schema_llm.ainvoke(cast("list[Any]", messages)),
            label="gemini_extract_schema",
        )
        metadata = ContractMetadata.model_validate(metadata)
        if metadata.jurisdiction is None:
            metadata = metadata.model_copy(update={"jurisdiction": state.jurisdiction})
        return {"contract_metadata": metadata}

    return extract_schema_node


def make_segment_document_node(
    segmentation_llm: StructuredRunnable,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def segment_document_node(state: IngestionState) -> dict[str, object]:
        parsed = state.parsed_document
        metadata = state.contract_metadata
        if parsed is None or metadata is None:
            return {"error": "missing_document_or_metadata", "ingestion_complete": False}

        payload = serialize_to_toon(
            {
                "contract_metadata": metadata.model_dump(),
                "source": parsed.source,
                "markdown": parsed.markdown[:50_000],
            }
        )
        messages = [
            SystemMessage(content=SEGMENT_DOCUMENT_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
        try:
            result = await retry_immediate(
                lambda: segmentation_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_segment_document",
            )
            segments = ClauseSegmentationResult.model_validate(result).segments
        except Exception as exc:  # noqa: BLE001 - fallback segmentation keeps ingestion usable.
            logger.bind(doc_id=state.doc_id, error=str(exc)).warning(
                "structured_segmentation_failed_using_fallback"
            )
            segments = _fallback_segments(parsed.markdown)

        return {"segments": _ensure_chunk_enrichment(segments)}

    return segment_document_node


def dispatch_contextualize_chunks(state: IngestionState) -> list[Send]:
    metadata = state.contract_metadata or ContractMetadata()
    parsed = state.parsed_document or ParsedDocument(markdown="", title="", source=state.source)
    return [
        Send(
            "contextualize_chunks",
            {
                "segment": segment.model_dump(),
                "contract_metadata": metadata.model_dump(),
                "source": parsed.source,
            },
        )
        for segment in state.segments
    ]


def make_contextualize_chunk_node(
    contextualize_llm: StructuredRunnable,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, object]]]:
    async def contextualize_chunk_node(state: dict[str, Any]) -> dict[str, object]:
        segment = ClauseSegment.model_validate(state["segment"])
        metadata: ContractMetadata = ContractMetadata.model_validate(state["contract_metadata"])
        preamble = _build_preamble(segment, metadata)
        payload = serialize_to_toon(
            {
                "required_preamble": preamble,
                "segment": segment.model_dump(),
                "contract_metadata": metadata.model_dump(),
            }
        )
        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=CONTEXTUALIZE_CHUNK_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
        try:
            result = await retry_immediate(
                lambda: contextualize_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_contextualize_chunk",
            )
            chunk = ContextualizedChunk.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - deterministic preamble is a safe fallback.
            logger.bind(clause_id=segment.clause_id, error=str(exc)).warning(
                "contextualize_failed_using_deterministic_preamble"
            )
            chunk = ContextualizedChunk(
                clause_id=segment.clause_id,
                chunk_index=segment.chunk_index,
                clause_type=segment.clause_type,
                preamble=preamble,
                text=segment.text,
                tokens=max(1, len(f"{preamble} {segment.text}".split())),
                page_no=segment.page_no,
                chunk_faqs=segment.chunk_faqs,
                chunk_keywords=segment.chunk_keywords,
            )
        return {"contextualized_chunks": [chunk]}

    return contextualize_chunk_node


def make_classify_extract_node(
    extraction_llm: StructuredRunnable,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def classify_extract_node(state: IngestionState) -> dict[str, object]:
        metadata: ContractMetadata | None = state.contract_metadata
        if metadata is None:
            return {"error": "missing_contract_metadata", "ingestion_complete": False}

        payload = serialize_to_toon(
            {
                "contract_metadata": metadata.model_dump(),
                "chunks": [chunk.model_dump() for chunk in state.contextualized_chunks],
            }
        )
        messages = [
            SystemMessage(content=CLASSIFY_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=payload[:80_000]),
        ]
        try:
            result = await retry_immediate(
                lambda: extraction_llm.ainvoke(cast("list[Any]", messages)),
                label="gemini_entity_extraction",
            )
            extraction: EntityExtractionResult = EntityExtractionResult.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - entity extraction is non-critical enrichment.
            logger.bind(doc_id=state.doc_id, error=str(exc)).warning(
                "entity_extraction_failed_continuing_without_entities"
            )
            extraction = EntityExtractionResult()
        return {
            "extracted_entities": extraction.entities,
            "extracted_relationships": extraction.relationships,
        }

    return classify_extract_node


def make_embed_store_node(
    db_engine: AsyncEngine,
    embedding_fn: EmbeddingFunction,
    redis: Redis | None = None,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def embed_store_node(state: IngestionState) -> dict[str, object]:
        parsed: ParsedDocument | None = state.parsed_document
        metadata: ContractMetadata | None = state.contract_metadata
        if parsed is None or metadata is None:
            return {"error": "missing_parsed_document_or_metadata", "ingestion_complete": False}

        async with AsyncSession(db_engine) as session, session.begin():
            parent_doc_id = await retry_immediate(
                lambda: _upsert_parent_document(session, state, parsed, metadata),
                label="postgres_upsert_parent_document",
            )
            stored_entities = await _store_entities(session, state)
            stored_relationships = await _store_relationships(session, state, stored_entities)
            stored_chunks = await _store_chunks(
                session=session,
                state=state,
                parsed=parsed,
                metadata=metadata,
                parent_doc_id=parent_doc_id,
                embedding_fn=embedding_fn,
                redis=redis,
            )
            await retry_immediate(
                lambda: _force_merge_bm25(session),
                label="postgres_bm25_force_merge",
            )

        return {
            "parent_doc_id": parent_doc_id,
            "stored_clause_ids": [chunk.chunk_id for chunk in stored_chunks],
            "stored_chunks": stored_chunks,
            "stored_entity_ids": list(stored_entities.values()),
            "stored_relationship_ids": stored_relationships,
        }

    return embed_store_node


def make_graphiti_upsert_node(
    graphiti: Any,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def graphiti_upsert_node(state: IngestionState) -> dict[str, object]:
        if graphiti is None:
            return {"graphiti_episode_ids": [], "ingestion_complete": True}

        episode_ids: list[str] = []
        for chunk in state.contextualized_chunks:
            postgres_chunk_id = _stored_chunk_id(state, chunk.clause_id)
            if postgres_chunk_id is None:
                continue
            body = (
                f"{chunk.preamble}\n\n{chunk.text}\n\n"
                f"REFERENCES_CLAUSE postgres_chunk_id={postgres_chunk_id}"
            )
            source_description = json.dumps(
                {
                    "doc_id": state.doc_id,
                    "clause_id": chunk.clause_id,
                    "postgres_chunk_id": postgres_chunk_id,
                    "clause_type": chunk.clause_type.value,
                    "edge_type": "REFERENCES_CLAUSE",
                }
            )
            episode_id = await _graphiti_add_episode(
                graphiti=graphiti,
                name=f"clause:{state.doc_id}:{chunk.clause_id}",
                body=body,
                source_description=source_description,
                group_id=state.doc_id,
            )
            if episode_id:
                episode_ids.append(episode_id)

        for event_name, event_date in _contract_events(state.contract_metadata):
            episode_id: str | None = await _graphiti_add_episode(
                graphiti=graphiti,
                name=f"{event_name}:{state.doc_id}:{event_date}",
                body=f"{event_name} for {state.doc_id} occurs on {event_date}.",
                source_description=json.dumps(
                    {"doc_id": state.doc_id, "event_type": event_name, "event_date": event_date}
                ),
                group_id=state.doc_id,
            )
            if episode_id:
                episode_ids.append(episode_id)

        return {"graphiti_episode_ids": episode_ids, "ingestion_complete": True}

    return graphiti_upsert_node


async def _parse_document_with_docling(
    raw_bytes: bytes,
    filename: str,
    source: str,
) -> ParsedDocument:
    def _sync_parse() -> ParsedDocument:
        suffix = Path(filename or "upload.pdf").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            result: ConversionResult = DocumentConverter().convert(tmp.name)
            document: DoclingDocument = result.document
            markdown = document.export_to_markdown()
            tables: list[Any] = [
                table.to_markdown()
                for table in getattr(document, "tables", [])
                if hasattr(table, "to_markdown")
            ]
            elements: list[Unknown] = [
                item.to_dict()
                for item, _level in document.iterate_items()
                if hasattr(item, "to_dict")
            ]
            return ParsedDocument(
                markdown=markdown,
                title=_extract_title(markdown, filename),
                source=source or filename or tmp.name,
                page_count=len(getattr(document, "pages", []) or []),
                tables=tables,
                elements=elements[:500],
            )

    return await asyncer.asyncify(_sync_parse)()


def _fallback_segments(markdown: str) -> list[ClauseSegment]:
    blocks: list[str | Any] = [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]
    if not blocks:
        blocks: list[str | Any] = [markdown.strip()] if markdown.strip() else []
    return [
        ClauseSegment(
            clause_id=f"clause-{index + 1}",
            text=block,
            chunk_index=index,
            chunk_faqs=[f"What does clause {index + 1} cover?"],
            chunk_keywords=_keywords(block),
        )
        for index, block in enumerate(blocks[:200])
    ]


def _ensure_chunk_enrichment(segments: list[ClauseSegment]) -> list[ClauseSegment]:
    enriched: list[ClauseSegment] = []
    for index, segment in enumerate(segments):
        enriched.append(
            segment.model_copy(
                update={
                    "chunk_index": index,
                    "page_no": segment.page_no or 0,
                    "chunk_faqs": segment.chunk_faqs or [f"What does {segment.clause_id} say?"],
                    "chunk_keywords": segment.chunk_keywords or _keywords(segment.text),
                }
            )
        )
    return enriched


def _build_preamble(segment: ClauseSegment, metadata: ContractMetadata) -> str:
    party_a = metadata.party_a or (metadata.parties[0] if metadata.parties else "unknown party")
    party_b = metadata.party_b or (
        metadata.parties[1] if len(metadata.parties) > 1 else "unknown party"
    )
    effective = metadata.effective_date or "unknown date"
    return (
        f"This is {segment.clause_type.value} from {metadata.contract_name} "
        f"between {party_a} and {party_b}, effective {effective}."
    )


async def _upsert_parent_document(
    session: AsyncSession,
    state: IngestionState,
    parsed: ParsedDocument,
    metadata: ContractMetadata,
) -> str:
    content_hash: str = hashlib.sha256(parsed.markdown.encode("utf-8")).hexdigest()
    query: TextClause = text(
        """
        INSERT INTO parent_documents
            (doc_id, user_id, thread_id, source, title, document_type, jurisdiction,
             content_hash, markdown, summary, metadata)
        VALUES
            (:doc_id, :user_id, :thread_id, :source, :title, :document_type,
             :jurisdiction, :content_hash, :markdown, :summary, CAST(:metadata AS JSONB))
        ON CONFLICT (doc_id)
        DO UPDATE SET
            source = EXCLUDED.source,
            title = EXCLUDED.title,
            document_type = EXCLUDED.document_type,
            jurisdiction = EXCLUDED.jurisdiction,
            content_hash = EXCLUDED.content_hash,
            markdown = EXCLUDED.markdown,
            summary = EXCLUDED.summary,
            metadata = EXCLUDED.metadata
        RETURNING id::text
        """
    )
    row = (
        await session.execute(
            query,
            {
                "doc_id": state.doc_id,
                "user_id": state.user_id,
                "thread_id": state.thread_id,
                "source": parsed.source,
                "title": parsed.title,
                "document_type": state.document_type,
                "jurisdiction": metadata.jurisdiction or state.jurisdiction,
                "content_hash": content_hash,
                "markdown": parsed.markdown,
                "summary": metadata.document_summary,
                "metadata": json.dumps(_contract_metadata_json(metadata, parsed.source)),
            },
        )
    ).fetchone()
    return str(row[0])


async def _store_entities(
    session: AsyncSession,
    state: IngestionState,
) -> dict[str, str]:
    entity_id_map: dict[str, str] = {}
    for entity in state.extracted_entities:
        row_id = str(uuid4())
        row = (
            await session.execute(
                text(
                    """
                    INSERT INTO entities
                        (id, entity_type, name, normalized_name, doc_id, user_id,
                         thread_id, metadata, confidence, decay_score)
                    VALUES
                        (:id, :entity_type, :name, :normalized_name, :doc_id, :user_id,
                         :thread_id, CAST(:metadata AS JSONB), :confidence, 1.0)
                    ON CONFLICT (normalized_name, entity_type)
                    DO UPDATE SET
                        confidence = GREATEST(entities.confidence, EXCLUDED.confidence),
                        last_accessed_at = NOW()
                    RETURNING id::text
                    """
                ),
                {
                    "id": row_id,
                    "entity_type": entity.type.value,
                    "name": entity.name,
                    "normalized_name": entity.normalized_name.lower().strip(),
                    "doc_id": state.doc_id,
                    "user_id": state.user_id,
                    "thread_id": state.thread_id,
                    "metadata": json.dumps({"source": state.source}),
                    "confidence": entity.confidence,
                },
            )
        ).fetchone()
        entity_id_map[entity.id] = str(row[0]) if row else row_id
    return entity_id_map


async def _store_relationships(
    session: AsyncSession,
    state: IngestionState,
    entity_id_map: dict[str, str],
) -> list[str]:
    stored: list[str] = []
    for relationship in state.extracted_relationships:
        from_id = entity_id_map.get(relationship.from_entity)
        to_id = entity_id_map.get(relationship.to_entity)
        if from_id is None or to_id is None:
            continue
        relationship_id = str(uuid4())
        row = (
            await session.execute(
                text(
                    """
                    INSERT INTO relationships
                        (id, from_entity_id, to_entity_id, relation_type, doc_id,
                         user_id, clause_id, metadata, valid_from, valid_to,
                         confidence, source)
                    VALUES
                        (:id, :from_id, :to_id, :relation_type, :doc_id, :user_id,
                         :clause_id, CAST(:metadata AS JSONB), :valid_from, :valid_to,
                         :confidence, 'graphiti_extraction')
                    ON CONFLICT DO NOTHING
                    RETURNING id::text
                    """
                ),
                {
                    "id": relationship_id,
                    "from_id": from_id,
                    "to_id": to_id,
                    "relation_type": relationship.type.value,
                    "doc_id": state.doc_id,
                    "user_id": state.user_id,
                    "clause_id": relationship.clause_id,
                    "metadata": json.dumps({"source": state.source}),
                    "valid_from": relationship.valid_from,
                    "valid_to": relationship.valid_to,
                    "confidence": relationship.confidence,
                },
            )
        ).fetchone()
        if row:
            stored.append(str(row[0]))
    return stored


async def _store_chunks(
    *,
    session: AsyncSession,
    state: IngestionState,
    parsed: ParsedDocument,
    metadata: ContractMetadata,
    parent_doc_id: str,
    embedding_fn: EmbeddingFunction,
    redis: Redis | None,
) -> list[StoredChunk]:
    stored: list[StoredChunk] = []
    for chunk in sorted(state.contextualized_chunks, key=lambda item: item.chunk_index):
        row_id = str(uuid4())
        chunk_id = row_id
        text_to_embed = f"{chunk.preamble}\n\n{chunk.text}"
        embedding = await _cached_embedding(redis, embedding_fn, text_to_embed)
        metadata_json = _chunk_metadata_json(
            metadata=metadata,
            source=parsed.source,
            page_no=chunk.page_no,
        )
        custom_metadata = {
            "source": parsed.source,
            "page_no": chunk.page_no,
            "document_summary": metadata.document_summary,
            "chunk_id": chunk_id,
            "chunk_faqs": chunk.chunk_faqs,
            "chunk_keywords": chunk.chunk_keywords,
        }
        query = text(
            """
                    INSERT INTO clauses
                        (id, chunk_id, parent_doc_id, contract_id, doc_id, user_id,
                         clause_id, chunk_index, text, chunk_text, preamble, embedding,
                         clause_type, metadata_, custom_metadata, decay_score)
                    VALUES
                        (:id, :chunk_id, :parent_doc_id, :contract_id, :doc_id, :user_id,
                         :clause_id, :chunk_index, :text, :chunk_text, :preamble,
                         CAST(:embedding AS vector), :clause_type, CAST(:metadata AS JSONB),
                         CAST(:custom_metadata AS JSONB), 1.0)
                    ON CONFLICT (parent_doc_id, chunk_index)
                    DO UPDATE SET
                        chunk_id = EXCLUDED.chunk_id,
                        text = EXCLUDED.text,
                        chunk_text = EXCLUDED.chunk_text,
                        preamble = EXCLUDED.preamble,
                        embedding = EXCLUDED.embedding,
                        clause_type = EXCLUDED.clause_type,
                        metadata_ = EXCLUDED.metadata_,
                        custom_metadata = EXCLUDED.custom_metadata
                    RETURNING chunk_id::text
                    """
        )
        params = {
            "id": row_id,
            "chunk_id": chunk_id,
            "parent_doc_id": parent_doc_id,
            "contract_id": metadata.contract_name,
            "doc_id": state.doc_id,
            "user_id": state.user_id,
            "clause_id": chunk.clause_id,
            "chunk_index": chunk.chunk_index,
            "text": text_to_embed,
            "chunk_text": _naturalize_tables(chunk.text),
            "preamble": chunk.preamble,
            "embedding": _vector_literal(embedding),
            "clause_type": chunk.clause_type.value,
            "metadata": json.dumps(metadata_json),
            "custom_metadata": json.dumps(custom_metadata),
        }
        result = await retry_immediate(
            lambda query=query, params=params: session.execute(query, params),
            label="postgres_store_clause",
        )
        row = result.fetchone()
        stored_chunk_id = str(row[0]) if row else chunk_id
        stored.append(
            StoredChunk(
                chunk_id=stored_chunk_id,
                clause_id=chunk.clause_id,
                chunk_index=chunk.chunk_index,
                clause_type=chunk.clause_type.value,
            )
        )
    return stored


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
        label="gemini_embedding",
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


async def _force_merge_bm25(session: AsyncSession) -> None:
    try:
        await session.execute(text("SELECT bm25_force_merge('clauses_bm25_idx')"))
    except Exception as exc:  # noqa: BLE001 - extension/index may be absent in local/dev DBs.
        logger.bind(error=str(exc)).warning("bm25_force_merge_skipped")


async def _graphiti_add_episode(
    *,
    graphiti: Any,
    name: str,
    body: str,
    source_description: str,
    group_id: str,
) -> str | None:
    try:
        result = await retry_immediate(
            lambda: graphiti.add_episode(
                name=name,
                episode_body=body,
                source=EpisodeType.text,
                source_description=source_description,
                reference_time=datetime.now(tz=UTC),
                group_id=group_id,
                entity_types=GRAPHITI_ENTITY_TYPES,
                edge_types=GRAPHITI_EDGE_TYPES,
                edge_type_map=GRAPHITI_EDGE_TYPE_MAP,
            ),
            label="graphiti_add_episode",
        )
        return str(getattr(result, "uuid", name))
    except Exception as exc:  # noqa: BLE001 - graph write failures should not roll back Postgres ingestion.
        logger.bind(name=name, error=str(exc)).warning("graphiti_episode_upsert_failed")
        return None


def _contract_events(metadata: ContractMetadata | None) -> list[tuple[str, str]]:
    if metadata is None:
        return []
    events = [
        ("contract_signed", metadata.contract_signed),
        ("amendment_effective", metadata.amendment_effective),
        ("expiry_date", metadata.expiry_date),
    ]
    return [(name, date) for name, date in events if date]


def _stored_chunk_id(state: IngestionState, clause_id: str) -> str | None:
    for chunk in state.stored_chunks:
        if chunk.clause_id == clause_id:
            return chunk.chunk_id
    return None


def _contract_metadata_json(metadata: ContractMetadata, source: str) -> dict[str, object]:
    payload = metadata.model_dump()
    payload["source"] = source
    payload["page_no"] = 0
    return payload


def _chunk_metadata_json(
    *,
    metadata: ContractMetadata,
    source: str,
    page_no: int,
) -> dict[str, object]:
    payload = _contract_metadata_json(metadata, source)
    payload["page_no"] = page_no
    payload["jurisdiction"] = metadata.jurisdiction
    payload["contract_type"] = metadata.contract_type
    payload["party_names"] = metadata.parties
    return payload


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def _extract_title(markdown: str, filename: str) -> str:
    for line in markdown.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:500]
    return Path(filename or "uploaded-document").stem[:500]


def _keywords(text_value: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text_value.lower())
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        result.append(word)
        if len(result) >= 12:
            break
    return result


def _naturalize_tables(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_table_line(line):
            table: list[str] = []
            while index < len(lines) and _is_table_line(lines[index]):
                table.append(lines[index])
                index += 1
            output.extend(_table_to_sentences(table))
            continue
        output.append(line)
        index += 1
    return "\n".join(output)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _table_to_sentences(lines: list[str]) -> list[str]:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
        if "---" not in line
    ]
    if not rows:
        return []
    headers = rows[0]
    sentences: list[str] = []
    for row_index, row in enumerate(rows[1:], start=1):
        pairs = [
            f"{headers[index]}: {value}"
            for index, value in enumerate(row)
            if index < len(headers) and value
        ]
        if pairs:
            sentences.append(f"Table row {row_index}: {'; '.join(pairs)}.")
    return sentences
