"""
Tool: get_obligation_chain

Risk analysis agent tool.  Forward-chains obligations from a named entity.

This is the constraint propagation tool from the "Chosen Ones" insight:
  Party A → INDEMNIFIES → Party B
  Payment → TRIGGERS → Penalty
  Breach → TRIGGERS → Termination right

Use this when the risk agent needs to trace what consequences flow from
a given party's obligations — the full causal chain, not just direct edges.

Combined with query_knowledge_graph, the risk agent can reason:
  1. query_knowledge_graph: "What are the indemnity clauses?"
  2. get_obligation_chain: "What obligations does Party A have from clause C-007?"
  3. Combine: "Party A has unlimited liability under clause C-007, which triggers
     a personal guarantee obligation under clause C-012."

That's multi-hop legal reasoning.  Most RAG systems can't do step 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

from app.shared.agents.tools.idempotency import IdempotencyGuard, ToolResult
from app.utils import logger

if TYPE_CHECKING:
    from langchain_core.tools.base import BaseTool

    from app.shared.rag.graphiti.client import GraphitiService
    from app.shared.rag.graphiti.schemas import GraphitiSearchResult


def make_get_obligation_chain_tool(
    graphiti_service: GraphitiService,
    idempotency: IdempotencyGuard,
) -> BaseTool:
    """Create the obligation-chain LangChain tool."""

    @tool
    async def get_obligation_chain(
        entity_name: str,
        user_id: str,
        doc_id: str,
        thread_id: str,
        step_id: str,
        depth: int = 3,
    ) -> dict[str, object]:
        """Trace obligations and consequences connected to a named entity.

        Use this when the agent needs forward-chained legal consequences such
        as liability propagation, triggered penalties, or downstream duties.
        """
        log = logger.bind(
            tool="get_obligation_chain",
            entity_name=entity_name,
            doc_id=doc_id,
            depth=depth,
        )

        idempotency_key = IdempotencyGuard.make_key(
            step_id=step_id,
            input_data={
                "doc_id": doc_id,
                "entity_name": entity_name,
                "depth": depth,
            },
            user_id=user_id,
        )

        cached_result = await idempotency.get(idempotency_key)
        if cached_result is not None:
            log.debug("obligation_chain_cache_hit")
            return cached_result.model_dump()

        graph_results = await graphiti_service.get_obligation_chain(
            entity_name=entity_name,
            user_id=user_id,
            doc_id=doc_id,
            depth=depth,
        )
        sorted_results = sorted(
            graph_results,
            key=lambda result: result.relevance_score,
            reverse=True,
        )

        tool_result = ToolResult.ok(
            data={
                "entity": entity_name,
                "chain": [_serialize_result(result) for result in sorted_results],
                "chain_summary": _format_obligation_chain(entity_name, sorted_results),
                "hop_count": len(sorted_results),
            },
            tool="get_obligation_chain",
        )

        await idempotency.set(
            key=idempotency_key,
            result=tool_result,
            tool_name="get_obligation_chain",
            user_id=user_id,
            thread_id=thread_id,
            step_id=step_id,
        )

        log.info("obligation_chain_traced", hop_count=len(sorted_results))
        return tool_result.model_dump()

    return get_obligation_chain


def _serialize_result(result: GraphitiSearchResult) -> dict[str, object]:
    return {
        "uuid": result.uuid,
        "name": result.name,
        "content": result.content,
        "relevance_score": result.relevance_score,
        "group_id": result.group_id,
        "metadata": result.metadata_raw,
    }


def _format_obligation_chain(
    entity_name: str,
    results: list[GraphitiSearchResult],
) -> str:
    """Format graph results into a compact chain summary for the LLM."""
    if not results:
        return f"No obligations found for {entity_name} in the knowledge graph."

    lines = [f"Obligation chain starting from '{entity_name}':"]
    for index, result in enumerate(results, start=1):
        lines.append(f"  {index}. {result.name}: {result.content[:200]}")
    return "\n".join(lines)
