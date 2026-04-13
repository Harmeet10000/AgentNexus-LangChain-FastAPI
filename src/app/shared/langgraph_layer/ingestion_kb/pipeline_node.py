"""
Ingestion graph for uploaded legal documents.

Pipeline:
  1. Extract entities and relationships from raw text.
  2. Validate extraction confidence and endpoint integrity.
  3. Persist entities, clause embeddings, and relationships.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils import logger

from .prompt import extraction_prompt
from .state import (
    ExtractedEntityType,
    ExtractionPayload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import BaseMessage
    from sqlalchemy.ext.asyncio import AsyncEngine

    from .state import (
        EmbeddingFunction,
        ExtractedEntity,
        ExtractedRelationship,
        ExtractionRunnable,
        IngestionState,
    )


_CONFIDENCE_THRESHOLD = 0.7
_MAX_EXTRACTION_CHARS = 12_000


def make_extract_node(
    extraction_llm: ExtractionRunnable,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def extract_node(state: IngestionState) -> dict[str, object]:
        log = logger.bind(node="ingestion_extract", doc_id=state.doc_id)

        if not state.raw_text:
            log.error("extract_no_text")
            return {
                "error": "empty_document",
                "extracted_entities": [],
                "extracted_relationships": [],
                "extraction_error": "raw_text is empty",
            }

        messages = [
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=state.raw_text[:_MAX_EXTRACTION_CHARS]),
        ]

        try:
            result = await extraction_llm.ainvoke(cast("list[BaseMessage]", messages))
            payload = _parse_extraction_payload(result)
        except (ValidationError, json.JSONDecodeError) as exc:
            log.bind(error=str(exc)).warning("extract_invalid_payload")
            return {
                "extracted_entities": [],
                "extracted_relationships": [],
                "extraction_error": str(exc),
            }
        except Exception as exc:
            log.bind(error=str(exc)).exception("extract_failed")
            return {
                "extracted_entities": [],
                "extracted_relationships": [],
                "extraction_error": str(exc),
            }

        log.info(
            "extract_done",
            entities=len(payload.entities),
            relationships=len(payload.relationships),
        )
        return {
            "extracted_entities": payload.entities,
            "extracted_relationships": payload.relationships,
            "extraction_error": None,
        }

    return extract_node


def make_validate_node() -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def validate_node(state: IngestionState) -> dict[str, object]:
        log = logger.bind(node="ingestion_validate", doc_id=state.doc_id)

        valid_entities = [
            entity
            for entity in state.extracted_entities
            if entity.confidence > _CONFIDENCE_THRESHOLD and entity.name
        ]
        valid_entity_ids = {entity.id for entity in valid_entities}

        valid_relationships = [
            relationship
            for relationship in state.extracted_relationships
            if relationship.confidence > _CONFIDENCE_THRESHOLD
            and relationship.from_entity in valid_entity_ids
            and relationship.to_entity in valid_entity_ids
            and relationship.type
        ]

        dropped_entities = len(state.extracted_entities) - len(valid_entities)
        dropped_relationships = len(state.extracted_relationships) - len(valid_relationships)

        log.info(
            "validate_done",
            valid_entities=len(valid_entities),
            valid_relationships=len(valid_relationships),
            dropped_entities=dropped_entities,
            dropped_relationships=dropped_relationships,
        )
        return {
            "validated_entities": valid_entities,
            "validated_relationships": valid_relationships,
            "dropped_entity_count": dropped_entities,
            "dropped_relationship_count": dropped_relationships,
        }

    return validate_node


def make_embed_store_node(
    db_engine: AsyncEngine,
    embedding_fn: EmbeddingFunction,
) -> Callable[[IngestionState], Awaitable[dict[str, object]]]:
    async def embed_store_node(state: IngestionState) -> dict[str, object]:
        log = logger.bind(node="ingestion_embed_store", doc_id=state.doc_id)

        entity_id_map: dict[str, str] = {}
        stored_entity_ids: list[str] = []
        stored_clause_ids: list[str] = []
        stored_relationship_ids: list[str] = []

        try:
            async with AsyncSession(db_engine) as session, session.begin():
                for entity in state.validated_entities:
                    entity_record_id = await _upsert_entity(
                        session=session,
                        entity=entity,
                        state=state,
                    )
                    entity_id_map[entity.id] = entity_record_id
                    stored_entity_ids.append(entity_record_id)

                    if entity.type is ExtractedEntityType.CLAUSE:
                        clause_row_id = await _store_clause(
                            session=session,
                            entity=entity,
                            state=state,
                            embedding_fn=embedding_fn,
                        )
                        stored_clause_ids.append(clause_row_id)

                for relationship in state.validated_relationships:
                    relationship_id = await _store_relationship(
                        session=session,
                        relationship=relationship,
                        entity_id_map=entity_id_map,
                        state=state,
                    )
                    if relationship_id is not None:
                        stored_relationship_ids.append(relationship_id)
        except Exception as exc:
            log.bind(error=str(exc)).exception("embed_store_failed")
            return {
                "error": str(exc),
                "ingestion_complete": False,
            }

        log.info(
            "embed_store_done",
            entities=len(stored_entity_ids),
            clauses=len(stored_clause_ids),
            relationships=len(stored_relationship_ids),
        )
        return {
            "stored_entity_ids": stored_entity_ids,
            "stored_clause_ids": stored_clause_ids,
            "stored_relationship_ids": stored_relationship_ids,
            "ingestion_complete": True,
        }

    return embed_store_node


async def _upsert_entity(
    *,
    session: AsyncSession,
    entity: ExtractedEntity,
    state: IngestionState,
) -> str:
    metadata = json.dumps(
        {
            "doc_type": state.document_type,
            "jurisdiction": state.jurisdiction,
        }
    )
    query = text(
        """
        INSERT INTO entities
            (id, entity_type, name, normalized_name, doc_id, user_id, thread_id,
             metadata, confidence, decay_score)
        VALUES
            (:id, :entity_type, :name, :normalized_name, :doc_id, :user_id,
             :thread_id, CAST(:metadata AS JSONB), :confidence, 1.0)
        ON CONFLICT (normalized_name, entity_type)
        DO UPDATE SET
            confidence = GREATEST(entities.confidence, EXCLUDED.confidence),
            last_accessed_at = NOW()
        RETURNING id
        """
    )
    generated_id = str(uuid4())
    row = (
        await session.execute(
            query,
            {
                "id": generated_id,
                "entity_type": entity.type.value,
                "name": entity.name,
                "normalized_name": entity.normalized_name.lower().strip(),
                "doc_id": state.doc_id,
                "user_id": state.user_id,
                "thread_id": state.thread_id,
                "metadata": metadata,
                "confidence": entity.confidence,
            },
        )
    ).fetchone()
    return str(row[0]) if row is not None else generated_id


async def _store_clause(
    *,
    session: AsyncSession,
    entity: ExtractedEntity,
    state: IngestionState,
    embedding_fn: EmbeddingFunction,
) -> str:
    embedding: list[float] | None = None
    try:
        embedding = await embedding_fn(entity.name)
    except Exception as exc:
        logger.bind(
            clause_id=entity.id,
            doc_id=state.doc_id,
            error=str(exc),
        ).warning("clause_embedding_failed")

    clause_row_id = str(uuid4())
    query = text(
        """
        INSERT INTO clauses
            (id, doc_id, user_id, clause_id, text, embedding, clause_type, decay_score)
        VALUES
            (:id, :doc_id, :user_id, :clause_id, :text, :embedding, :clause_type, 1.0)
        ON CONFLICT DO NOTHING
        """
    )
    await session.execute(
        query,
        {
            "id": clause_row_id,
            "doc_id": state.doc_id,
            "user_id": state.user_id,
            "clause_id": entity.id,
            "text": entity.name,
            "embedding": embedding,
            "clause_type": "other",
        },
    )
    return clause_row_id


async def _store_relationship(
    *,
    session: AsyncSession,
    relationship: ExtractedRelationship,
    entity_id_map: dict[str, str],
    state: IngestionState,
) -> str | None:
    from_entity_id = entity_id_map.get(relationship.from_entity)
    to_entity_id = entity_id_map.get(relationship.to_entity)
    if from_entity_id is None or to_entity_id is None:
        return None

    query = text(
        """
        INSERT INTO relationships
            (id, from_entity_id, to_entity_id, relation_type,
             doc_id, user_id, confidence, valid_from, valid_to, source)
        VALUES
            (:id, :from_id, :to_id, :rel_type,
             :doc_id, :user_id, :confidence,
             :valid_from, :valid_to, 'graphiti_extraction')
        ON CONFLICT DO NOTHING
        RETURNING id
        """
    )
    row = (
        await session.execute(
            query,
            {
                "id": str(uuid4()),
                "from_id": from_entity_id,
                "to_id": to_entity_id,
                "rel_type": relationship.type,
                "doc_id": state.doc_id,
                "user_id": state.user_id,
                "confidence": relationship.confidence,
                "valid_from": relationship.valid_from,
                "valid_to": relationship.valid_to,
            },
        )
    ).fetchone()
    return str(row[0]) if row is not None else None


def _parse_extraction_payload(result: object) -> ExtractionPayload:
    content = _read_llm_content(result)
    cleaned_content = _strip_markdown_fences(content)
    return ExtractionPayload.model_validate_json(cleaned_content)


def _read_llm_content(result: object) -> str:
    content = getattr(result, "content", result)
    return content if isinstance(content, str) else str(content)


def _strip_markdown_fences(content: str) -> str:
    cleaned = content.strip()
    if not cleaned.startswith("```"):
        return cleaned

    segments = cleaned.split("```")
    if len(segments) < 2:
        return cleaned

    fenced_content = segments[1].strip()
    if fenced_content.startswith("json"):
        return fenced_content[4:].strip()
    return fenced_content
