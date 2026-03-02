"""
Entity extraction using Graphiti for knowledge graph construction.

Integrates with existing graphiti-core dependency for entity/relationship
extraction from documents.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime

from app.utils.logger import logger


@dataclass
class Entity:
    """Represents an extracted entity."""

    id: str | None = None
    name: str
    entity_type: str
    description: str | None = None
    properties: dict = field(default_factory=dict)
    source_document_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    id: str | None = None
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: dict = field(default_factory=dict)
    source_document_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass
class ExtractionResult:
    """Result of entity extraction."""

    document_id: str
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
    error: str | None = None


class GraphitiExtractor:
    """
    Entity and relationship extractor using Graphiti.

    Graphiti is a knowledge graph builder for RAG applications that
    extracts entities and relationships from text documents.
    """

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        embedding_model: str = "gemini-embedding-001",
    ):
        """
        Initialize Graphiti extractor.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            embedding_model: Embedding model for entity encoding
        """
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.embedding_model = embedding_model
        self._client = None
        self._graph = None

    def _initialize_client(self):
        """Initialize Graphiti client."""
        try:
            from graphiti_graph import Graphiti

            if self._client is None:
                self._client = Graphiti(
                    neo4j_uri=self.neo4j_uri,
                    neo4j_auth=(self.neo4j_user, self.neo4j_password),
                )
                logger.info("Graphiti client initialized")
        except ImportError:
            logger.warning("graphiti_graph not available, using fallback extraction")
            self._client = None

    async def extract(
        self,
        text: str,
        document_id: str,
        config: dict | None = None,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from text.

        Args:
            text: Input text to extract from
            document_id: Source document identifier
            config: Optional extraction configuration

        Returns:
            ExtractionResult with entities and relationships
        """
        import time

        start_time = time.time()

        self._initialize_client()

        if self._client is None:
            return await self._extract_fallback(text, document_id, start_time)

        try:
            result = await self._extract_with_graphiti(text, document_id, config or {})
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            logger.error(f"Graphiti extraction failed: {e}")
            return await self._extract_fallback(text, document_id, start_time)

    async def _extract_with_graphiti(
        self,
        text: str,
        document_id: str,
        config: dict,
    ) -> ExtractionResult:
        """Extract using Graphiti."""

        try:
            episode = await self._client.add_episode(
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

            logger.info(
                f"Extracted {len(entities)} entities and "
                f"{len(relationships)} relationships"
            )

            return ExtractionResult(
                document_id=document_id,
                entities=entities,
                relationships=relationships,
            )

        except Exception as e:
            logger.error(f"Graphiti extraction error: {e}")
            raise

    async def _extract_fallback(
        self,
        text: str,
        document_id: str,
        start_time: float,
    ) -> ExtractionResult:
        """Fallback extraction using simple NLP patterns."""
        entities = self._extract_entities_simple(text, document_id)
        relationships = self._extract_relationships_simple(text, entities)

        import time

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            document_id=document_id,
            entities=entities,
            relationships=relationships,
            processing_time_ms=processing_time,
            metadata={"extraction_method": "fallback"},
        )

    def _extract_entities_simple(self, text: str, document_id: str) -> list[Entity]:
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
                )
            )
            entity_id += 1

        logger.info(f"Fallback extraction found {len(entities)} entities")
        return entities

    def _extract_relationships_simple(
        self, text: str, entities: list[Entity]
    ) -> list[Relationship]:
        """Simple pattern-based relationship extraction."""
        relationships = []
        rel_id = 0

        entity_names = {e.name for e in entities}

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
                                )
                            )
                            rel_id += 1

        return relationships

    async def batch_extract(
        self,
        documents: list[tuple[str, str]],
        progress_callback: callable | None = None,
    ) -> list[ExtractionResult]:
        """
        Extract from multiple documents.

        Args:
            documents: List of (text, document_id) tuples
            progress_callback: Optional progress callback

        Returns:
            List of ExtractionResults
        """
        import asyncio

        results = []

        for idx, (text, doc_id) in enumerate(documents):
            if progress_callback:
                progress_callback(idx + 1, len(documents))

            result = await self.extract(text, doc_id)
            results.append(result)

            await asyncio.sleep(0.1)

        return results


class SimpleEntityExtractor:
    """Lightweight entity extractor without Graphiti dependency."""

    def __init__(self):
        """Initialize simple extractor."""
        self.entity_patterns = {
            "PERSON": r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",
            "ORGANIZATION": r"\b([A-Z][a-zA-Z]+ (Inc|Corp|LLC|Ltd|Company|Group))\b",
            "TECHNOLOGY": r"\b(Python|JavaScript|React|FastAPI|Gemini|AI|ML|LLM)\b",
            "LOCATION": r"\b([A-Z][a-z]+ (City|Country|State|County))\b",
        }

    async def extract(
        self,
        text: str,
        document_id: str,
    ) -> ExtractionResult:
        """Extract entities using pattern matching."""
        import time

        start_time = time.time()

        entities = []
        entity_id = 0

        for entity_type, pattern in self.entity_patterns.items():
            for match in re.finditer(pattern, text):
                entities.append(
                    Entity(
                        id=f"entity_{entity_id}",
                        name=match.group(1),
                        entity_type=entity_type,
                        source_document_id=document_id,
                    )
                )
                entity_id += 1

        relationships = self._infer_relationships(entities)

        return ExtractionResult(
            document_id=document_id,
            entities=entities,
            relationships=relationships,
            processing_time_ms=(time.time() - start_time) * 1000,
            metadata={"extraction_method": "simple_patterns"},
        )

    def _infer_relationships(self, entities: list[Entity]) -> list[Relationship]:
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
                    )
                )
                rel_id += 1

        return relationships


def create_extractor(
    use_graphiti: bool = True,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> GraphitiExtractor | SimpleEntityExtractor:
    """
    Factory function to create entity extractor.

    Args:
        use_graphiti: Whether to use Graphiti (requires Neo4j)
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        Entity extractor instance
    """
    if use_graphiti:
        return GraphitiExtractor(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
    return SimpleEntityExtractor()
