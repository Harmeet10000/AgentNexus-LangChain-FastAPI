"""
Tool: query_knowledge_graph

Risk analysis agent tool.  Multi-hop semantic search over Graphiti.

This is the primary tool for risk_analysis_node.  It answers questions like:
  "What obligations does Party A have in this document?"
  "Are there any unlimited liability clauses across my prior contracts?"
  "What clauses conflict with each other?"

Graphiti's search traverses the knowledge graph — it doesn't just match
embeddings.  It follows entity links.  That's why risk analysis uses this
instead of a plain vector search: context + connections, not just similarity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

from app.utils import logger

from .idempotency import IdempotencyGuard, ToolResult

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.tools.base import BaseTool

    from app.shared.rag.graphiti.client import GraphitiService

def make_query_knowledge_graph_tool(
    graphiti_service: GraphitiService,
    idempotency: IdempotencyGuard,
) -> BaseTool:

    @tool
    async def query_knowledge_graph(
        query: str,
        user_id: str,
        doc_id: str,
        thread_id: str,
        step_id: str,
        num_results: int = 8,
    ) -> dict[str, Any]:
        """Query the legal knowledge graph for risk-relevant context.

        Searches across:
          - Current document's clause episodes (doc-scoped)
          - User's historical approved documents (user-scoped)

        Use this for multi-hop reasoning:
          "What are all the obligations connected to the termination clause?"
          "How does the indemnity clause interact with the liability cap?"

        Args:
            query: Risk-focused search query
            user_id: For Graphiti namespace scoping
            doc_id: Current document ID (scopes to this doc + user history)
            thread_id: For idempotency audit
            step_id: Plan step ID for idempotency key
            num_results: Max results to return (default: 8)
        """
        log = logger.bind(tool="query_knowledge_graph", doc_id=doc_id)

        idem_key = IdempotencyGuard.make_key(
            step_id=step_id,
            input_data={"query": query, "doc_id": doc_id, "num_results": num_results},
            user_id=user_id,
        )
        cached = await idempotency.get(idem_key)
        if cached is not None:
            log.debug("knowledge_graph_cache_hit")
            return cached.model_dump()

        results = await graphiti_service.search_for_risk_context(
            query=query,
            user_id=user_id,
            doc_id=doc_id,
            num_results=num_results,
        )

        # Sort by multi-objective score descending
        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)

        result = ToolResult.ok(
            data={
                "results": [
                    {
                        "uuid": r.uuid,
                        "name": r.name,
                        "content": r.content,
                        "relevance_score": r.relevance_score,
                        "group_id": r.group_id,
                        "metadata": r.metadata_raw,
                    }
                    for r in sorted_results
                ],
                "total_found": len(sorted_results),
                "query": query,
            },
            tool="query_knowledge_graph",
        )

        await idempotency.set(
            key=idem_key,
            result=result,
            tool_name="query_knowledge_graph",
            user_id=user_id,
            thread_id=thread_id,
            step_id=step_id,
        )
        log.info("knowledge_graph_queried", results=len(sorted_results))
        return result.model_dump()

    return query_knowledge_graph
