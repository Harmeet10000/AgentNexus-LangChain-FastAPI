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

from typing import TYPE_CHECKING, Annotated, Any

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.tools.base import BaseTool

from app.observability.logging import logger
from app.resilience.idempotency import IdempotencyGuard, ToolResult

from .authorization import trusted_tool_context

if TYPE_CHECKING:
    from app.retrieval.graphiti.client import BoundGraphitiService as GraphitiService


def make_query_knowledge_graph_tool(
    graphiti_service: GraphitiService,
    idempotency: IdempotencyGuard,
) -> BaseTool:

    @tool
    async def query_knowledge_graph(
        query: str,
        *,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
        user_id: Annotated[str | None, InjectedToolArg] = None,
        doc_id: Annotated[str | None, InjectedToolArg] = None,
        thread_id: Annotated[str | None, InjectedToolArg] = None,
        step_id: Annotated[str | None, InjectedToolArg] = None,
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
            Identity and document scope are injected from trusted graph runtime.
            num_results: Max results to return (default: 8)
        """
        authz = trusted_tool_context(
            runtime,
            legacy_user_id=user_id,
            legacy_doc_id=doc_id,
            legacy_thread_id=thread_id,
            legacy_step_id=step_id,
        )
        authz.require_operation("document.read")
        if not authz.allowed_doc_ids:
            return ToolResult.fail(
                "No authorized documents were supplied for this analysis.",
                tool="query_knowledge_graph",
            ).model_dump()
        doc_id = authz.allowed_doc_ids[0] if authz.allowed_doc_ids else ""
        authz.require_document(doc_id)
        log = logger.bind(tool="query_knowledge_graph", doc_id=doc_id)

        idem_key = IdempotencyGuard.make_key(
            step_id=authz.step_id,
            structural={"doc_id": doc_id, "num_results": num_results},
            content={"query": query},
            user_id=authz.user_id,
            tool_name="query_knowledge_graph",
            tenant_id=authz.tenant_id,
            allowed_doc_ids=authz.allowed_doc_ids,
            evidence_policy="mechanically_verified",
            document_versions=authz.document_versions,
            thread_id=authz.thread_id,
            run_id=authz.run_id,
            checkpoint_id=authz.checkpoint_id,
        )
        cached = await idempotency.get(idem_key)
        if cached is not None:
            log.debug("knowledge_graph_cache_hit")
            return cached.model_dump()

        results = await graphiti_service.search_for_risk_context(
            query=query,
            user_id=authz.tenant_id,
            doc_id=doc_id,
            num_results=num_results,
            allowed_doc_ids=authz.allowed_doc_ids,
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
            user_id=authz.tenant_id,
            thread_id=authz.thread_id,
            step_id=authz.step_id,
        )
        log.info("knowledge_graph_queried", results=len(sorted_results))
        return result.model_dump()

    return query_knowledge_graph
