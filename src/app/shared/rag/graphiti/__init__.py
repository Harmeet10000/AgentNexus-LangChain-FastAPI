"""Graphiti RAG integration for legal domain knowledge graphs.

This module provides async-first integration with Graphiti for building and
querying legal knowledge graphs stored in Neo4j. It handles:

- Clause episode creation and persistence
- Relationship edge mapping between legal concepts
- Final report episode storage
- Multi-objective memory retrieval with semantic, recency, trust, and task relevance scoring

Key components:
- Pure async functions for setup, write, read operations
- Memory pipeline: Context building for agent reasoning
- Tool registry: Compliance and risk assessment tools
- Schemas: Domain types for legal episodes and search results

Usage:
    from src.app.shared.rag.graphiti.client import (
        setup_graphiti,
        setup_graphiti_indices,
        close_graphiti,
        write_clause_episode,
        search_for_risk_context,
    )

    # In lifespan.py
    graphiti = await setup_graphiti(
        neo4j_uri=settings.NEO4J_URI,
        neo4j_user=settings.NEO4J_USERNAME,
        neo4j_password=settings.NEO4J_PASSWORD,
    )
    await setup_graphiti_indices(graphiti)
    app.state.graphiti = graphiti

    # On shutdown
    await close_graphiti(graphiti)
"""

from .client import (
    close_graphiti,
    get_obligation_chain,
    search_for_precedent_chains,
    search_for_risk_context,
    setup_graphiti,
    setup_graphiti_indices,
    write_clause_episode,
    write_final_report_episode,
    write_relationship_edge,
)

__all__ = [
    "close_graphiti",
    "get_obligation_chain",
    "search_for_precedent_chains",
    "search_for_risk_context",
    "setup_graphiti",
    "setup_graphiti_indices",
    "write_clause_episode",
    "write_final_report_episode",
    "write_relationship_edge",
]
