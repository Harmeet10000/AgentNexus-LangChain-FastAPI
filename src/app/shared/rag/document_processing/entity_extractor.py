"""
Entity extraction using Graphiti for knowledge graph construction.

Integrates with existing graphiti-core dependency for entity/relationship
extraction from documents.
"""

import asyncio
import os
import re
import time
from datetime import UTC, datetime
from typing import Any

from app.utils.logger import logger as loguru_logger

from .models import Entity, ExtractionResult, Relationship


async def extract_with_graphiti(
    text: str,
    document_id: str,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    config: dict[str, Any] | None = None,
) -> ExtractionResult:
    """
    Extract entities and relationships using Graphiti.

    Args:
        text: Input text to extract from
        document_id: Source document identifier
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        config: Optional extraction configuration

    Returns:
        ExtractionResult with entities and relationships
    """
    start_time = time.time()

    neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

    try:
        from graphiti_graph import Graphiti

        client = Graphiti(
            neo4j_uri=neo4j_uri,
            neo4j_auth=(neo4j_user, neo4j_password),
        )
        loguru_logger.info("Graphiti client initialized")

        episode = await client.add_episode(
            name=f"document_{document_id}",
            text=text,
        )

        entities = []
        relationships = []

        if hasattr(episode, "nodes"):
            for node in episode.nodes:
                entity = Entity(
                    id=getattr(node, "id", None),
                    name=getattr(node, "name", ""),
                    entity_type=getattr(node, "type", "UNKNOWN"),
                    description=getattr(node, "description", None),
                    properties=getattr(node, "properties", {}),
                    source_document_id=document_id,
                )
                entities.append(entity)

        if hasattr(episode, "edges"):
            for edge in episode.edges:
                relationship = Relationship(
                    id=getattr(edge, "id", None),
                    source_entity_id=str(getattr(edge, "source", "")),
                    target_entity_id=str(getattr(edge, "target", "")),
                    relationship_type=getattr(edge, "name", "RELATED_TO"),
                    properties=getattr(edge, "properties", {}),
                    source_document_id=document_id,
                )
                relationships.append(relationship)

        loguru_logger.info(
            f"Extracted {len(entities)} entities and {len(relationships)} relationships"
        )

        processing_time = (time.time() - start_time) * 1000
        return ExtractionResult(
            document_id=document_id,
            entities=entities,
            relationships=relationships,
            processing_time_ms=processing_time,
        )

    except ImportError:
        loguru_logger.warning("graphiti_graph not available, using fallback extraction")
        return await extract_with_fallback(text, document_id, start_time)
    except Exception as e:
        loguru_logger.error(f"Graphiti extraction failed: {e}")
        return await extract_with_fallback(text, document_id, start_time)


async def extract_with_fallback(
    text: str, document_id: str, start_time: float | None = None
) -> ExtractionResult:
    """Fallback extraction using simple NLP patterns."""
    if start_time is None:
        start_time = time.time()

    entities = _extract_entities_simple(text, document_id)
    relationships = _extract_relationships_simple(text, entities)

    processing_time = (time.time() - start_time) * 1000

    return ExtractionResult(
        document_id=document_id,
        entities=entities,
        relationships=relationships,
        processing_time_ms=processing_time,
        metadata={"extraction_method": "fallback"},
    )


def _extract_entities_simple(text: str, document_id: str) -> list[Entity]:
    """Simple pattern-based entity extraction."""

    entities = []
    entity_id = 0

    person_pattern = r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b"
    for match in re.finditer(person_pattern, text):
        entities.append(
            Entity(
                id=f"entity_{entity_id}",
                name=match.group(1),
                entity_type="PERSON",
                source_document_id=document_id,
                created_at=datetime.now(UTC),
            )
        )
        entity_id += 1

    org_pattern = (
        r"\b([A-Z][a-zA-Z]+ (Inc|Corp|LLC|Ltd|Company|Group|Foundation))\b"
    )
    for match in re.finditer(org_pattern, text):
        entities.append(
            Entity(
                id=f"entity_{entity_id}",
                name=match.group(1),
                entity_type="ORGANIZATION",
                source_document_id=document_id,
                created_at=datetime.now(UTC),
            )
        )
        entity_id += 1

    tech_pattern = r"\b(Python|JavaScript|TypeScript|React|Angular|Vue|Node\.js|FastAPI|PyTorch|TensorFlow|Gemini|GPT|LLM|AI|ML)\b"
    for match in re.finditer(tech_pattern, text):
        entities.append(
            Entity(
                id=f"entity_{entity_id}",
                name=match.group(1),
                entity_type="TECHNOLOGY",
                source_document_id=document_id,
                created_at=datetime.now(UTC),
            )
        )
        entity_id += 1

    loguru_logger.info(f"Fallback extraction found {len(entities)} entities")
    return entities


def _extract_relationships_simple(
    text: str, entities: list[Entity]
) -> list[Relationship]:
    """Simple pattern-based relationship extraction."""
    relationships = []
    rel_id = 0

    for entity in entities:
        if entity.entity_type == "PERSON":
            if " works at " in text or " employed by " in text:
                for org in entities:
                    if org.entity_type == "ORGANIZATION":
                        relationships.append(
                            Relationship(
                                id=f"rel_{rel_id}",
                                source_entity_id=entity.id,
                                target_entity_id=org.id,
                                relationship_type="WORKS_AT",
                                source_document_id=entity.source_document_id,
                                created_at=datetime.now(UTC),
                            )
                        )
                        rel_id += 1

        if entity.entity_type == "TECHNOLOGY":
            if " uses " in text or " built with " in text:
                for org in entities:
                    if org.entity_type == "ORGANIZATION":
                        relationships.append(
                            Relationship(
                                id=f"rel_{rel_id}",
                                source_entity_id=org.id,
                                target_entity_id=entity.id,
                                relationship_type="USES",
                                source_document_id=entity.source_document_id,
                                created_at=datetime.now(UTC),
                            )
                        )
                        rel_id += 1

    return relationships


async def extract_entities_batch(
    documents: list[tuple[str, str]],
    use_graphiti: bool = True,
    progress_callback: None | callable = None,
    neo4j_config: dict[str, Any] | None = None,
) -> list[ExtractionResult]:
    """
    Extract from multiple documents.

    Args:
        documents: List of (text, document_id) tuples
        use_graphiti: Whether to use Graphiti (requires Neo4j)
        progress_callback: Optional progress callback
        neo4j_config: Neo4j configuration dict

    Returns:
        List of ExtractionResults
    """
    neo4j_config = neo4j_config or {}
    results = []

    for idx, (text, doc_id) in enumerate(documents):
        if progress_callback:
            progress_callback(idx + 1, len(documents))

        if use_graphiti:
            result = await extract_with_graphiti(
                text,
                doc_id,
                neo4j_uri=neo4j_config.get("uri"),
                neo4j_user=neo4j_config.get("user"),
                neo4j_password=neo4j_config.get("password"),
            )
        else:
            result = await extract_with_fallback(text, doc_id)

        results.append(result)

        await asyncio.sleep(0.1)

    return results


def extract_entities_simple(text: str, document_id: str) -> ExtractionResult:
    """
    Extract entities using lightweight pattern matching.

    Args:
        text: Text to extract from
        document_id: Source document identifier

    Returns:
        ExtractionResult with entities and inferred relationships
    """
    start_time = time.time()

    entities = _extract_entities_simple(text, document_id)
    relationships = _infer_relationships(entities)

    processing_time = (time.time() - start_time) * 1000

    return ExtractionResult(
        document_id=document_id,
        entities=entities,
        relationships=relationships,
        processing_time_ms=processing_time,
        metadata={"extraction_method": "simple_patterns"},
    )


def _infer_relationships(entities: list[Entity]) -> list[Relationship]:
    """Infer relationships between extracted entities."""
    relationships = []
    rel_id = 0

    persons = [e for e in entities if e.entity_type == "PERSON"]
    orgs = [e for e in entities if e.entity_type == "ORGANIZATION"]

    for person in persons:
        for org in orgs:
            relationships.append(
                Relationship(
                    id=f"rel_{rel_id}",
                    source_entity_id=person.id,
                    target_entity_id=org.id,
                    relationship_type="RELATED_TO",
                    created_at=datetime.now(UTC),
                )
            )
            rel_id += 1

    return relationships


async def extract(
    text: str,
    document_id: str,
    use_graphiti: bool = False,
    neo4j_config: dict[str, Any] | None = None,
) -> ExtractionResult:
    """
    Extract entities and relationships from text.

    Args:
        text: Input text to extract from
        document_id: Source document identifier
        use_graphiti: Whether to use Graphiti
        neo4j_config: Neo4j configuration

    Returns:
        ExtractionResult
    """
    if use_graphiti:
        neo4j_config = neo4j_config or {}
        return await extract_with_graphiti(
            text,
            document_id,
            neo4j_uri=neo4j_config.get("uri"),
            neo4j_user=neo4j_config.get("user"),
            neo4j_password=neo4j_config.get("password"),
        )
    else:
        return extract_entities_simple(text, document_id)


def create_extractor(
    use_graphiti: bool = True,
    neo4j_config: dict[str, Any] | None = None,
) -> callable:
    """
    Factory function to create entity extractor.

    Args:
        use_graphiti: Whether to use Graphiti (requires Neo4j)
        neo4j_config: Neo4j configuration dict

    Returns:
        Entity extractor callable
    """
    neo4j_config = neo4j_config or {}

    async def extractor(text: str, document_id: str) -> ExtractionResult:
        return await extract(text, document_id, use_graphiti, neo4j_config)

    return extractor
