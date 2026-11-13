"""
AgentToolBundle: all LangChain tools assembled once at graph-build time.

build_tool_bundle() is called by the Agent Saul provider registered in the
application lifespan (see `app.lifecycle.graphs`). The bundle is passed into
build_agent_registry() in factory.py so agents get their tools at compile
time — never at node execution time. (An earlier class of the same name
collided with an unrelated, since-deleted tool registry and was retired.)

Lifespan wiring (in src/app/lifecycle/graphs.py):
    from app.features.documents.application.tools.bundle import AgentToolBundle, build_tool_bundle
    from app.resilience.idempotency import IdempotencyGuard

    idempotency_guard = IdempotencyGuard(
        redis=app.state.redis,
        db_engine=app.state.db_engine,
    )
    tool_registry = build_tool_bundle(
        graphiti_service=app.state.graphiti,
        db_engine=app.state.db_engine,
        idempotency=idempotency_guard,
    )
    app.state.tool_registry = tool_registry
    app.state.idempotency_guard = idempotency_guard

    # Pass to graph factory:
    app.state.saul_graph = build_saul_graph(
        checkpointer=app.state.langgraph_checkpointer,
        pro_llm=pro_llm,
        flash_llm=flash_llm,
        memory_service=app.state.agent_memory_service,
        tool_registry=tool_registry,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools.base import BaseTool
from pydantic import BaseModel, ConfigDict

from app.features.documents.application.evidence.service import EvidenceService
from app.features.documents.application.tools.evidence import (
    make_get_ingestion_state_tool,
    make_read_evidence_tool,
    make_search_evidence_tool,
)
from app.features.documents.application.tools.get_obligation_chain import (
    make_get_obligation_chain_tool,
)
from app.features.documents.application.tools.query_knowledge_graph import (
    make_query_knowledge_graph_tool,
)
from app.features.documents.application.tools.retrieve_statute_section import (
    make_retrieve_statute_section_tool,
)
from app.features.documents.application.tools.search_legal_precedents import (
    make_search_legal_precedents_tool,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from app.resilience.idempotency import IdempotencyGuard
    from app.retrieval.graphiti.client import BoundGraphitiService as GraphitiService


class AgentToolBundle(BaseModel):
    """
    Immutable collection of all pre-built LangChain tools.

    Tool assignment to agents is bounded by the authoritative registry
    (``application/tools/registry.py``, harden 5.8) and asserted at
    ``agent_saul.graph.factory`` build time:
      compliance_agent  → scoped evidence search/read/readiness, statute
      retrieval, and handoff
      risk_agent        → scoped evidence search/read/readiness, the knowledge
      graph and obligation chain tools, and handoff
      deep_research     → the scoped EvidenceService adapter (research must not
      bypass the handle-based evidence seam).

    ``search_legal_precedents`` is a TRANSITIONAL seat: it binds to the
    compliance specialist only when no EvidenceService is provisioned, and the
    scoped evidence seam replaces it once that is provisioned.

    Graphiti write tools (write_clause_episodes_to_graphiti /
    write_final_report_to_memory) are implemented but deliberately unwired —
    no node's toolset may carry them (the registry rejects any binding).
    Persistence runs only through the approval-gated ``persist_memory`` node
    outside any tool surface.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    # Compliance agent tools
    search_legal_precedents: BaseTool
    retrieve_statute_section: BaseTool

    # Risk agent tools
    query_knowledge_graph: BaseTool
    get_obligation_chain: BaseTool

    # Scoped evidence contract. Optional keeps worker/test startup compatible
    # while the lifecycle provider migrates to EvidenceService wiring.
    search_evidence: BaseTool | None = None
    read_evidence_by_id: BaseTool | None = None
    get_ingestion_state: BaseTool | None = None
    evidence_service: EvidenceService | None = None


def build_tool_bundle(
    graphiti_service: GraphitiService,
    db_engine: AsyncEngine,
    idempotency: IdempotencyGuard,
    evidence_service: EvidenceService | None = None,
) -> AgentToolBundle:
    """Build all tools once.  Call at lifespan startup only."""
    evidence_tools = (
        {
            "search_evidence": make_search_evidence_tool(evidence_service),
            "read_evidence_by_id": make_read_evidence_tool(evidence_service),
            "get_ingestion_state": make_get_ingestion_state_tool(evidence_service),
        }
        if evidence_service is not None
        else {}
    )
    return AgentToolBundle(
        search_legal_precedents=make_search_legal_precedents_tool(
            graphiti_service=graphiti_service,
            db_engine=db_engine,
            idempotency=idempotency,
        ),
        retrieve_statute_section=make_retrieve_statute_section_tool(
            db_engine=db_engine,
            idempotency=idempotency,
        ),
        query_knowledge_graph=make_query_knowledge_graph_tool(
            graphiti_service=graphiti_service,
            idempotency=idempotency,
        ),
        get_obligation_chain=make_get_obligation_chain_tool(
            graphiti_service=graphiti_service,
            idempotency=idempotency,
        ),
        evidence_service=evidence_service,
        **evidence_tools,
    )
