"""
AgentToolBundle: all LangChain tools assembled once at graph-build time.

build_tool_bundle() is called wherever the Saul graph is wired (see
lifespan.py — currently commented out pending the graph wiring; task 11.4
retargets it when group 11 lands). The bundle is passed into
build_agent_registry() in factory.py so agents get their tools at compile
time — never at node execution time. The old class name collided with the
unrelated tool-registry in agents/tools/base.py and is retired.

Lifespan wiring (in src/app/lifecycle/lifespan.py):
    from app.shared.rag.graphiti.registry import AgentToolBundle, build_tool_bundle
    from app.shared.langchain_layer.agents.tools.idempotency import IdempotencyGuard

    idempotency_guard = IdempotencyGuard(
        redis=app.state.redis,
        db_engine=app.state.db_engine,
    )
    tool_registry = build_tool_registry(
        graphiti_service=app.state.graphiti,
        db_engine=app.state.db_engine,
        idempotency=idempotency_guard,
    )
    app.state.tool_registry = tool_registry
    app.state.idempotency_guard = idempotency_guard

    # Pass to graph factory:
    app.state.saul_graph = build_saul_graph(
        checkpointer=app.state.saul_checkpointer,
        pro_llm=pro_llm,
        flash_llm=flash_llm,
        memory_service=app.state.agent_memory_service,
        tool_registry=tool_registry,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.shared.langchain_layer.agents.tools import (
    make_get_obligation_chain_tool,
    make_query_knowledge_graph_tool,
    make_retrieve_statute_section_tool,
    make_search_legal_precedents_tool,
)

if TYPE_CHECKING:
    from langchain_core.tools.base import BaseTool
    from sqlalchemy.ext.asyncio import AsyncEngine

    from app.shared.langchain_layer.agents.tools.idempotency import IdempotencyGuard

    from .client import GraphitiService


class AgentToolBundle(BaseModel):
    """
    Immutable collection of all pre-built LangChain tools.

    Tool assignment to agents (pending — agents are currently built with
    empty tool lists in factory.py):
      compliance_agent  → [search_legal_precedents, retrieve_statute_section]
      risk_agent        → [query_knowledge_graph, get_obligation_chain]
      deep_research     → deep_research_tool (delegates to search_legal_precedents)

    Memory writers (write_clause_episodes_to_graphiti / write_final_report_to_memory)
    are implemented but not yet wired into any node.
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

    @property
    def deep_research_tool(self) -> BaseTool:
        return self.search_legal_precedents


def build_tool_bundle(
    graphiti_service: GraphitiService,
    db_engine: AsyncEngine,
    idempotency: IdempotencyGuard,
) -> AgentToolBundle:
    """Build all tools once.  Call at lifespan startup only."""
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
    )
