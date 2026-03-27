"""
IngestionGraph: extract → validate → embed_store

Runs at HTTP upload time (before the WebSocket session starts).
agent_saul's ingestion_node becomes a simple doc_text lookup — it no longer
processes documents.

Pipeline:
  extract_node:     Graphiti EXTRACTION_PROMPT → raw entities + relationships
  validate_node:    confidence > 0.7 filter + dangling reference check
  embed_store_node: embed clause text (pgvector) + INSERT into 5 tables

State uses total=False so nodes can return partial updates cleanly.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict
from uuid import uuid4

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src.app.shared.rag.graphiti.client import GraphitiService

logger = structlog.get_logger(__name__)

_CONFIDENCE_THRESHOLD: float = 0.7
_EMBEDDING_DIM: int = 1536

# ---------------------------------------------------------------------------
# EXTRACTION_PROMPT (from the plan — enforces structured output)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """
You are a legal knowledge extraction system.

Extract from the document:

1. ENTITIES:
   - Parties (PERSON, ORG)
   - Contracts (CONTRACT)
   - Clauses (CLAUSE)
   - Obligations (OBLIGATION)

2. RELATIONSHIPS:
   - SIGNED_BY, OWES, GOVERNED_BY, TERMINATES_ON, LIABLE_FOR,
     INDEMNIFIES, TRIGGERED_BY, OVERRIDDEN_BY, RESTRICTS

Rules:
   - Normalize entity names: lowercase, strip whitespace, collapse aliases
     (e.g. "Acme Corp", "Acme Corporation" → normalized_name: "acme corp")
   - Include confidence (0.0–1.0) for every entity and relationship
   - DO NOT hallucinate parties, obligations, or clause references
   - valid_from / valid_to: ISO8601 strings if temporally bounded, else null

Output ONLY this JSON structure, no prose:
{
  "entities": [
    {
      "id": "uuid-string",
      "type": "PERSON|ORG|CLAUSE|CONTRACT|OBLIGATION",
      "name": "...",
      "normalized_name": "...",
      "confidence": 0.0
    }
  ],
  "relationships": [
    {
      "from": "entity-id",
      "to": "entity-id",
      "type": "SIGNED_BY|OWES|...",
      "confidence": 0.0,
      "valid_from": null,
      "valid_to": null
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# IngestionState
# ---------------------------------------------------------------------------


class IngestionState(TypedDict, total=False):
    doc_id: str
    user_id: str
    thread_id: str
    raw_text: str
    document_type: str
    jurisdiction: str

    # Extraction output
    extracted_entities: list[dict[str, Any]]
    extracted_relationships: list[dict[str, Any]]
    extraction_error: str | None

    # Validation output
    validated_entities: list[dict[str, Any]]
    validated_relationships: list[dict[str, Any]]
    dropped_entity_count: int
    dropped_relationship_count: int

    # Storage output
    stored_entity_ids: list[str]
    stored_clause_ids: list[str]
    stored_relationship_ids: list[str]
    ingestion_complete: bool
    error: str | None


# ---------------------------------------------------------------------------
# Node: extract
# ---------------------------------------------------------------------------


def make_extract_node(
    extraction_llm: Runnable[list[Any], Any],
    graphiti_service: GraphitiService,
) -> Any:
    async def extract_node(state: IngestionState) -> IngestionState:
        log = logger.bind(node="ingestion_extract", doc_id=state.get("doc_id"))
        raw_text = state.get("raw_text", "")

        if not raw_text:
            log.error("extract_no_text")
            return {"extraction_error": "raw_text is empty", "error": "empty_document"}

        import json as _json

        messages = [
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(content=raw_text[:12_000]),  # cap to avoid token explosion
        ]

        try:
            # Use structured LLM call for extraction
            result = await extraction_llm.ainvoke(messages)
            content: str = result.content if hasattr(result, "content") else str(result)

            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = _json.loads(content)
            entities = parsed.get("entities", [])
            relationships = parsed.get("relationships", [])

            log.info("extract_done", entities=len(entities), relationships=len(relationships))
            return {
                "extracted_entities": entities,
                "extracted_relationships": relationships,
                "extraction_error": None,
            }
        except Exception as exc:  # noqa: BLE001
            log.error("extract_failed", error=str(exc))
            return {"extraction_error": str(exc), "extracted_entities": [], "extracted_relationships": []}

    return extract_node


# ---------------------------------------------------------------------------
# Node: validate
# ---------------------------------------------------------------------------


def make_validate_node() -> Any:
    async def validate_node(state: IngestionState) -> IngestionState:
        log = logger.bind(node="ingestion_validate", doc_id=state.get("doc_id"))

        entities = state.get("extracted_entities", [])
        relationships = state.get("extracted_relationships", [])

        # Filter entities: confidence > threshold AND name present
        valid_entities = [
            e for e in entities
            if float(e.get("confidence", 0.0)) > _CONFIDENCE_THRESHOLD
            and e.get("name")
            and e.get("type") in {"PERSON", "ORG", "CLAUSE", "CONTRACT", "OBLIGATION"}
        ]

        # Index valid IDs for relationship validation
        valid_entity_ids: set[str] = {e["id"] for e in valid_entities}

        # Filter relationships: confidence > threshold AND both endpoints valid
        valid_relationships = [
            r for r in relationships
            if float(r.get("confidence", 0.0)) > _CONFIDENCE_THRESHOLD
            and r.get("from") in valid_entity_ids
            and r.get("to") in valid_entity_ids
            and r.get("type")
        ]

        dropped_e = len(entities) - len(valid_entities)
        dropped_r = len(relationships) - len(valid_relationships)

        log.info(
            "validate_done",
            valid_entities=len(valid_entities),
            valid_relationships=len(valid_relationships),
            dropped_entities=dropped_e,
            dropped_relationships=dropped_r,
        )

        return {
            "validated_entities": valid_entities,
            "validated_relationships": valid_relationships,
            "dropped_entity_count": dropped_e,
            "dropped_relationship_count": dropped_r,
        }

    return validate_node


# ---------------------------------------------------------------------------
# Node: embed_store
# ---------------------------------------------------------------------------


def make_embed_store_node(
    db_engine: AsyncEngine,
    embedding_fn: Any,  # async callable: (text: str) -> list[float]
) -> Any:
    async def embed_store_node(state: IngestionState) -> IngestionState:
        log = logger.bind(node="ingestion_embed_store", doc_id=state.get("doc_id"))

        doc_id = state.get("doc_id", "")
        user_id = state.get("user_id", "")
        thread_id = state.get("thread_id", "")
        entities = state.get("validated_entities", [])
        relationships = state.get("validated_relationships", [])
        raw_text = state.get("raw_text", "")
        document_type = state.get("document_type", "unknown")
        jurisdiction = state.get("jurisdiction", "India")

        entity_id_map: dict[str, str] = {}  # extraction_id → postgres_uuid
        stored_entity_ids: list[str] = []
        stored_clause_ids: list[str] = []
        stored_relationship_ids: list[str] = []

        try:
            async with AsyncSession(db_engine) as session:
                async with session.begin():
                    # --- Insert entities (upsert on normalized_name + type) ---
                    for ent in entities:
                        pg_id = str(uuid4())
                        normalized = (ent.get("normalized_name") or ent.get("name", "")).lower().strip()

                        upsert_entity = text("""
                            INSERT INTO entities
                                (id, entity_type, name, normalized_name, doc_id, user_id, thread_id,
                                 metadata, confidence, decay_score)
                            VALUES
                                (:id, :entity_type, :name, :normalized_name, :doc_id, :user_id,
                                 :thread_id, :metadata::jsonb, :confidence, 1.0)
                            ON CONFLICT (normalized_name, entity_type)
                            DO UPDATE SET
                                confidence = GREATEST(entities.confidence, EXCLUDED.confidence),
                                last_accessed_at = NOW()
                            RETURNING id
                        """)
                        import json as _json
                        row = (await session.execute(upsert_entity, {
                            "id": pg_id,
                            "entity_type": ent.get("type", "ORG"),
                            "name": ent.get("name", ""),
                            "normalized_name": normalized,
                            "doc_id": doc_id,
                            "user_id": user_id,
                            "thread_id": thread_id,
                            "metadata": _json.dumps({"doc_type": document_type, "jurisdiction": jurisdiction}),
                            "confidence": float(ent.get("confidence", 0.0)),
                        })).fetchone()
                        real_id = str(row[0]) if row else pg_id
                        entity_id_map[ent["id"]] = real_id
                        stored_entity_ids.append(real_id)

                        # --- Store CLAUSE entities in clauses table + embed ---
                        if ent.get("type") == "CLAUSE":
                            clause_text = ent.get("name", "")
                            embedding: list[float] | None = None
                            try:
                                embedding = await embedding_fn(clause_text)
                            except Exception:  # noqa: BLE001
                                pass

                            insert_clause = text("""
                                INSERT INTO clauses
                                    (id, doc_id, user_id, clause_id, text, embedding, clause_type, decay_score)
                                VALUES
                                    (:id, :doc_id, :user_id, :clause_id, :text,
                                     :embedding, :clause_type, 1.0)
                                ON CONFLICT DO NOTHING
                            """)
                            await session.execute(insert_clause, {
                                "id": str(uuid4()),
                                "doc_id": doc_id,
                                "user_id": user_id,
                                "clause_id": ent.get("id", real_id),
                                "text": clause_text,
                                "embedding": embedding,
                                "clause_type": "other",
                            })
                            stored_clause_ids.append(real_id)

                    # --- Insert relationships ---
                    for rel in relationships:
                        from_pg_id = entity_id_map.get(rel.get("from", ""))
                        to_pg_id = entity_id_map.get(rel.get("to", ""))
                        if not from_pg_id or not to_pg_id:
                            continue

                        insert_rel = text("""
                            INSERT INTO relationships
                                (id, from_entity_id, to_entity_id, relation_type,
                                 doc_id, user_id, confidence, valid_from, valid_to, source)
                            VALUES
                                (:id, :from_id, :to_id, :rel_type,
                                 :doc_id, :user_id, :confidence,
                                 :valid_from, :valid_to, 'graphiti_extraction')
                            ON CONFLICT DO NOTHING
                            RETURNING id
                        """)
                        rel_row = (await session.execute(insert_rel, {
                            "id": str(uuid4()),
                            "from_id": from_pg_id,
                            "to_id": to_pg_id,
                            "rel_type": rel.get("type", ""),
                            "doc_id": doc_id,
                            "user_id": user_id,
                            "confidence": float(rel.get("confidence", 0.0)),
                            "valid_from": rel.get("valid_from"),
                            "valid_to": rel.get("valid_to"),
                        })).fetchone()
                        if rel_row:
                            stored_relationship_ids.append(str(rel_row[0]))

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

        except Exception as exc:  # noqa: BLE001
            log.error("embed_store_failed", error=str(exc))
            return {"error": str(exc), "ingestion_complete": False}

    return embed_store_node


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_ingestion_graph(
    extraction_llm: Any,
    db_engine: AsyncEngine,
    embedding_fn: Any,
    graphiti_service: GraphitiService,
) -> CompiledStateGraph:
    """Build the IngestionGraph. Compiled once at lifespan, stored in app.state.

    No checkpointer — ingestion is fire-and-forget, not resumable.
    If it fails the user re-uploads.
    """
    extract_node = make_extract_node(extraction_llm, graphiti_service)
    validate_node = make_validate_node()
    embed_store_node = make_embed_store_node(db_engine, embedding_fn)

    graph = StateGraph(IngestionState)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("embed_store", embed_store_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "embed_store")
    graph.add_edge("embed_store", END)

    return graph.compile()
