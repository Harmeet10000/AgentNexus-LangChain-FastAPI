"""
Precedent agent tools: hybrid retrieval combining pgvector + Neo4j subgraph expansion.

This implements the "Subgraph Retrieval > Vector Retrieval" pattern:
  seed_nodes = vector.search(query)       ← pgvector cosine similarity on clauses
  subgraph   = expand_neighbors(seeds)    ← Neo4j Cypher depth-N traversal

The combination gives REASONING, not just recall:
  Vector finds semantically similar clauses.
  Subgraph follows their obligation chains to uncover connected risks.

scope=PRECEDENT_SCOPE enforced: all entity types, all sources, depth=3, all time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

from app.shared.agents.memory.memory_scope import PRECEDENT_SCOPE
from app.shared.agents.tools.idempotency import IdempotencyGuard, ToolResult
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.tools.base import BaseTool
    from sqlalchemy.ext.asyncio import AsyncEngine

    from app.shared.rag.graphiti.client import GraphitiService
    from app.shared.rag.graphiti.subgraph import Neo4jSubgraphExpander

def make_hybrid_retrieve_precedents_tool(
    graphiti_service: GraphitiService,
    subgraph_expander: Neo4jSubgraphExpander,
    db_engine: AsyncEngine,
    idempotency: IdempotencyGuard,
) -> BaseTool:
    """Factory: hybrid precedent retrieval tool.

    Step 1: pgvector — find N most similar clauses from user's history
    Step 2: Graphiti — semantic entity search scoped to user
    Step 3: Neo4j subgraph — expand from seed entities (depth=PRECEDENT_SCOPE.graph_depth=3)
    Step 4: Merge + deduplicate results
    """

    @tool
    async def hybrid_retrieve_precedents(
        query: str,
        user_id: str,
        doc_id: str,
        thread_id: str,
        step_id: str,
        num_results: int = 5,
    ) -> dict[str, Any]:
        """Retrieve legal precedents using hybrid vector + knowledge graph search.

        Combines:
          1. pgvector semantic clause search (user's historical clauses)
          2. Graphiti knowledge graph search (cross-document entity patterns)
          3. Neo4j subgraph expansion from seed entities (depth=3 hop traversal)

        Returns structured precedents with similarity scores and obligation chains.
        Scope: PRECEDENT_SCOPE — all entity types, all sources, all time.

        Args:
            query: Precedent search query (e.g., 'unlimited liability NDA India')
            user_id: For memory namespace scoping
            doc_id: Current document (included in scope for context)
            thread_id: For idempotency audit
            step_id: Plan step ID
            num_results: Max precedents to return
        """
        log = logger.bind(tool="hybrid_retrieve_precedents", user_id=user_id)
        scope = PRECEDENT_SCOPE

        idem_key = IdempotencyGuard.make_key(
            step_id=step_id,
            input_data={"query": query, "user_id": user_id, "num_results": num_results},
            user_id=user_id,
        )
        cached = await idempotency.get(idem_key)
        if cached is not None:
            log.debug("precedent_hybrid_cache_hit")
            return cached.model_dump()

        # Layer 1: pgvector clause similarity
        vector_results = await _vector_search_clauses(
            _db_engine=db_engine,
            _user_id=user_id,
            _query=query,
            _num_results=num_results,
            _time_filter=time_filter,
        )

        # Layer 2: Graphiti semantic entity search
        graphiti_results = await graphiti_service.search_for_precedent_chains(
            query=query,
            user_id=user_id,
            num_results=scope.top_k,
        )

        # Layer 3: Subgraph expansion from Graphiti seed entities
        seed_uuids = [r.uuid for r in graphiti_results if r.uuid]
        group_ids = [user_id, doc_id]
        subgraph = await subgraph_expander.expand_from_seeds(
            seed_uuids=seed_uuids,
            scope=scope,
            group_ids=group_ids,
        )

        result = ToolResult.ok(
            data={
                "vector_clauses": vector_results[:num_results],
                "graphiti_precedents": [
                    {
                        "name": r.name,
                        "content": r.content,
                        "relevance_score": r.relevance_score,
                        "source": "graphiti",
                    }
                    for r in graphiti_results
                ],
                "subgraph_context": subgraph.to_context_text(),
                "subgraph_node_count": len(subgraph.nodes),
                "subgraph_edge_count": len(subgraph.edges),
                "total_sources": len(vector_results) + len(graphiti_results),
            },
            tool="hybrid_retrieve_precedents",
        )

        await idempotency.set(
            key=idem_key,
            result=result,
            tool_name="hybrid_retrieve_precedents",
            user_id=user_id,
            thread_id=thread_id,
            step_id=step_id,
        )
        log.info(
            "hybrid_precedent_done",
            vector=len(vector_results),
            graphiti=len(graphiti_results),
            subgraph_nodes=len(subgraph.nodes),
        )
        return result.model_dump()

    return hybrid_retrieve_precedents


def make_detect_graph_conflicts_tool(
    subgraph_expander: Neo4jSubgraphExpander,
    idempotency: IdempotencyGuard,
) -> BaseTool:
    """Tool: detect contradicting obligations in the knowledge graph.

    Finds circular obligations (A owes B who owes A) and override chains
    (clause overrides clause that overrides original clause).
    Runs direct Neo4j Cypher — Graphiti's API doesn't expose this.
    """

    @tool
    async def detect_graph_conflicts(
        user_id: str,
        doc_id: str,
        thread_id: str,
        step_id: str,
    ) -> dict[str, Any]:
        """Detect contradicting obligation patterns in the legal knowledge graph.

        Finds:
          - Circular obligations: Party A owes Party B who owes Party A
          - Override chains: Clause C1 overrides C2 which overrides C1

        These are structural contradictions that pure text analysis misses.
        Use this during risk analysis when relationships are complex.

        Args:
            user_id: For graph namespace scoping
            doc_id: Current document
            thread_id: For idempotency audit
            step_id: Plan step ID
        """
        log = logger.bind(tool="detect_graph_conflicts", doc_id=doc_id)
        idem_key = IdempotencyGuard.make_key(
            step_id=step_id,
            input_data={"doc_id": doc_id},
            user_id=user_id,
        )
        cached = await idempotency.get(idem_key)
        if cached is not None:
            return cached.model_dump()

        conflicts = await subgraph_expander.detect_conflicts(
            group_ids=[user_id, doc_id]
        )

        result = ToolResult.ok(
            data={
                "conflicts": conflicts,
                "conflict_count": len(conflicts),
                "has_conflicts": len(conflicts) > 0,
            },
            tool="detect_graph_conflicts",
        )
        await idempotency.set(
            key=idem_key,
            result=result,
            tool_name="detect_graph_conflicts",
            user_id=user_id,
            thread_id=thread_id,
            step_id=step_id,
        )
        log.info("conflict_detection_done", conflicts=len(conflicts))
        return result.model_dump()

    return detect_graph_conflicts


async def _vector_search_clauses(
    _db_engine: AsyncEngine,
    _user_id: str,
    _query: str,
    _num_results: int,
    _time_filter: str,
) -> list[dict[str, Any]]:
    """pgvector cosine similarity search on clauses table.

    Time filter: "recent" = last 90 days; "all" = no constraint.
    TODO: embed query using same model as ingestion and pass as vector.
    Until then returns empty — prevents hallucinated precedents.
    """
    # TODO: embed query and use <=> cosine operator
    # query_vector = await embedding_fn(query)
    # Currently returns empty — safe default
    return []
