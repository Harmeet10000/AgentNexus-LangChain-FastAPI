"""
ToolRegistry: all LangChain tools assembled once at lifespan startup.

build_tool_registry() is called from lifespan.py alongside build_saul_graph().
The ToolRegistry is passed into build_agent_registry() in factory.py so
agents get their tools at graph compile time — never at node execution time.

Lifespan wiring (in src/app/lifecycle/lifespan.py):
    from app.shared.agents.tools.registry import ToolRegistry, build_tool_registry
    from app.shared.agents.tools.idempotency import IdempotencyGuard

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
        cognee_client=app.state.cognee,
        tool_registry=tool_registry,
        graphiti_service=app.state.graphiti,
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

    from app.shared.langchain_layer.agents import (
        IdempotencyGuard,
    )

    from .client import GraphitiService


class ToolRegistry(BaseModel):
    """
    Immutable collection of all pre-built LangChain tools.

    Tool assignment to agents:
      compliance_agent  → [search_legal_precedents, retrieve_statute_section]
      risk_agent        → [query_knowledge_graph, get_obligation_chain]
      orchestrator      → [] (uses structured output, no tools needed)

    Non-@tool functions (called directly from nodes, not via agent):
      write_clause_episodes_to_graphiti  → relationship_mapping node
      write_final_report_to_memory       → persist_memory node
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
    def compliance_tools(self) -> list[BaseTool]:
        return [self.search_legal_precedents, self.retrieve_statute_section]

    @property
    def risk_tools(self) -> list[BaseTool]:
        return [self.query_knowledge_graph, self.get_obligation_chain]


def build_tool_registry(
    graphiti_service: GraphitiService,
    db_engine: AsyncEngine,
    idempotency: IdempotencyGuard,
) -> ToolRegistry:
    """Build all tools once.  Call at lifespan startup only."""
    return ToolRegistry(
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
