"""Graphiti RAG integration for the legal domain knowledge graph.

Connection LIFECYCLE is not here: ``setup_graphiti`` /
``setup_graphiti_indices`` / ``close_graphiti`` are real implementations in
``app.connections.graphiti`` (the lifespan-facing adapters package). This
package owns the domain behavior bound to a live Graphiti instance:

- Bound service: every Graphiti operation below one wrapper
  (``BoundGraphitiService``)
- Clause episode creation and persistence, relationship edges between legal
  concepts
- Multi-objective retrieval with semantic, recency, trust, and task-relevance
  scoring
- Domain types for episodes and search results (``schemas.py``)

Usage:
    from app.retrieval.graphiti.client import (
        BoundGraphitiService,
        write_clause_episode,
        search_for_risk_context,
    )

    # in lifespan wiring
    graphiti = await setup_graphiti(...)  # app.connections.graphiti
    await setup_graphiti_indices(graphiti)
    bound = BoundGraphitiService(graphiti)
"""

from .client import (
    BoundGraphitiService,
    get_obligation_chain,
    search_for_precedent_chains,
    search_for_risk_context,
    write_clause_episode,
    write_relationship_edge,
)
from .write_clause_episodes import write_clause_episodes_to_graphiti

__all__ = [
    "BoundGraphitiService",
    "get_obligation_chain",
    "search_for_precedent_chains",
    "search_for_risk_context",
    "write_clause_episode",
    "write_clause_episodes_to_graphiti",
    "write_relationship_edge",
]
