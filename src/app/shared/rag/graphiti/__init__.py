"""Graphiti RAG integration for legal domain knowledge graphs.

This module provides async-first integration with Graphiti for building and
querying legal knowledge graphs stored in Neo4j. It handles:

- Clause episode creation and persistence
- Relationship edge mapping between legal concepts
- Final report episode storage
- Multi-objective memory retrieval with semantic, recency, trust, and task relevance scoring

Key components:
- GraphitiService: Main async wrapper around graphiti-core
- Memory pipeline: Context building for agent reasoning
- Tool registry: Compliance and risk assessment tools
- Write operations: Clause episodes, relationships, final reports
- Schemas: Domain types for legal episodes and search results

Usage:
    from src.app.shared.rag.graphiti.client import GraphitiService

    # In lifespan.py
    graphiti_service = await GraphitiService.create(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password.get_secret_value(),
    )
    await graphiti_service.setup()
    app.state.graphiti = graphiti_service

    # On shutdown
    await app.state.graphiti.close()
"""

from src.app.shared.rag.graphiti.client import GraphitiService
from src.app.shared.rag.graphiti.schemas import (
    ClauseEpisodeMetadata,
    FinalReportEpisodeMetadata,
    GraphitiSearchResult,
    LegalEdgeInput,
    LegalEpisodeType,
)

__all__ = [
    "GraphitiService",
    "ClauseEpisodeMetadata",
    "FinalReportEpisodeMetadata",
    "GraphitiSearchResult",
    "LegalEdgeInput",
    "LegalEpisodeType",
]
