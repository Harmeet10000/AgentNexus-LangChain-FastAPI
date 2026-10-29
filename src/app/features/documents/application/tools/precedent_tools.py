"""Optional graph-structure tools for legal analysis.

The former ``hybrid_retrieve_precedents`` tool lived here, but its vector leg
was a permanent stub and it was not bound to any live agent.  Precedent and
statute discovery now belongs to the scoped evidence service; this module only
retains the distinct graph-conflict operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.tools.base import BaseTool

from app.observability.logging import logger
from app.resilience.idempotency import IdempotencyGuard, ToolResult
from app.retrieval.graphiti.grouping import (
    GraphitiDocumentScope,
    fanout_graphiti_group_ids,
)

from .authorization import trusted_tool_context

if TYPE_CHECKING:
    from app.retrieval.graphiti.subgraph import Neo4jSubgraphConfig


def make_detect_graph_conflicts_tool(
    subgraph_expander: Neo4jSubgraphConfig,
    idempotency: IdempotencyGuard,
) -> BaseTool:
    """Tool: detect contradicting obligations in the knowledge graph.

    Finds circular obligations (A owes B who owes A) and override chains
    (clause overrides clause that overrides original clause).
    Runs direct Neo4j Cypher — Graphiti's API doesn't expose this.
    """

    @tool
    async def detect_graph_conflicts(
        *,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> dict[str, Any]:
        """Detect contradicting obligation patterns in the legal knowledge graph.

        Finds:
          - Circular obligations: Party A owes Party B who owes Party A
          - Override chains: Clause C1 overrides C2 which overrides C1

        These are structural contradictions that pure text analysis misses.
        Use this during risk analysis when relationships are complex.

        Args:
            Runtime identity and document scope are injected by LangGraph.
        """
        authz = trusted_tool_context(runtime)
        if not authz.allowed_doc_ids:
            return ToolResult.fail(
                "No authorized documents were supplied for this analysis.",
                tool="detect_graph_conflicts",
            ).model_dump()
        doc_id = authz.allowed_doc_ids[0]
        log = logger.bind(tool="detect_graph_conflicts", doc_id=doc_id)
        idem_key = IdempotencyGuard.make_key(
            step_id=authz.step_id,
            structural={"doc_id": doc_id},
            content=None,
            user_id=authz.user_id,
            tool_name="detect_graph_conflicts",
            tenant_id=authz.tenant_id,
            allowed_doc_ids=authz.allowed_doc_ids,
            evidence_policy="graph_indexed_only",
            document_versions=authz.document_versions,
            thread_id=authz.thread_id,
            run_id=authz.run_id,
            checkpoint_id=authz.checkpoint_id,
        )
        cached = await idempotency.get(idem_key)
        if cached is not None:
            return cached.model_dump()

        # Neo4j receives the same canonical partition IDs as Graphiti writes;
        # use the shared scope fan-out so this tool cannot drift from ordinary
        # retrieval on legacy-null contracts or version-only grants.
        versions = {doc_id: version for doc_id, version, _ in authz.document_versions}
        contracts = {
            (doc_id, version): (indexes, legacy_null)
            for doc_id, version, indexes, legacy_null in authz.document_contracts
        }
        scopes: list[GraphitiDocumentScope] = []
        for allowed_doc_id in authz.allowed_doc_ids:
            version = versions.get(allowed_doc_id)
            if version is None:
                continue
            indexes, legacy_null = contracts.get((allowed_doc_id, version), ((), False))
            scopes.append(
                GraphitiDocumentScope(
                    tenant_id=authz.tenant_id,
                    document_id=allowed_doc_id,
                    document_version=version,
                    contract_indexes=indexes,
                    legacy_null_contract=legacy_null,
                )
            )
        group_ids = fanout_graphiti_group_ids(scopes)
        if not group_ids:
            result = ToolResult.fail(
                "No authorized graph partitions were available for this analysis.",
                tool="detect_graph_conflicts",
            )
            await idempotency.set(
                key=idem_key,
                result=result,
                tool_name="detect_graph_conflicts",
                user_id=authz.user_id,
                thread_id=authz.thread_id,
                step_id=authz.step_id,
            )
            return result.model_dump()
        conflicts = await subgraph_expander.detect_conflicts(group_ids=group_ids)  # ty: ignore[unresolved-attribute]

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
            user_id=authz.user_id,
            thread_id=authz.thread_id,
            step_id=authz.step_id,
        )
        log.info("conflict_detection_done", conflicts=len(conflicts))
        return result.model_dump()

    return detect_graph_conflicts
