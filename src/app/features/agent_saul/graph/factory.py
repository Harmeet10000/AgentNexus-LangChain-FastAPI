"""Startup-time Agent Saul graph composition helpers."""

from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict

from app.ai.tools.registry import (
    get_tool_registry,
    register_deep_research_tool,
    register_default_tools,
    register_portfolio_traversal_tools,
)
from app.features.agent_saul.domain.state import (
    ComplianceOutput,
    FinalReport,
    GroundingVerificationOutput,
    NormalizedDocument,
    OrchestratorAction,
    RiskAnalysisOutput,
)
from app.features.documents.application.tools.authorization import ToolRuntimeContext
from app.features.documents.application.tools.bundle import AgentToolBundle
from app.features.documents.application.tools.portfolio import (
    RelationshipEdgeProbe,
    RelationshipEdgeVerdict,
    make_portfolio_traversal_tools,
    require_traversal_gate,
)
from app.retrieval.graphiti.client import BoundGraphitiService

from .deep_research import make_deep_research_tool
from .nodes import (
    ClauseSegmentationOutput,
    EntityExtractionOutput,
    PlannerOutput,
    QnAOutput,
    RelationshipMappingOutput,
    make_compliance_node,
    make_deep_research_node,
    make_entity_extraction_node,
    make_finalization_node,
    make_gateway_node,
    make_grounding_verification_node,
    make_human_review_entry,
    make_ingestion_node,
    make_normalization_node,
    make_orchestrator_node,
    make_persist_memory_node,
    make_plan_approval_node,
    make_plan_generation_node,
    make_qna_node,
    make_relationship_mapping_node,
    make_risk_analysis_node,
    make_segmentation_node,
)
from .prompts import (
    _COMPLIANCE_SYSTEM_PROMPT,
    _RISK_ANALYSIS_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------


class AgentRegistry(BaseModel):
    """
    Holds all pre-built agents and structured-output LLM chains.
    Created once at lifespan startup; referenced by closure in every node.

    Separation:
      create_react_agent → nodes that need tool-calling (orchestrator, risk, compliance)
      llm.with_structured_output → schema-locked nodes (qna, planner, pipeline nodes)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # create_react_agent compiled sub-graphs
    risk_agent: Any  # CompiledStateGraph
    compliance_agent: Any  # CompiledStateGraph

    # Structured output LLM chains
    qna_llm: Runnable[list[Any], QnAOutput]
    planner_llm: Runnable[list[Any], PlannerOutput]
    normalization_llm: Runnable[list[Any], NormalizedDocument]
    segmentation_llm: Runnable[list[Any], ClauseSegmentationOutput]
    entity_llm: Runnable[list[Any], EntityExtractionOutput]
    relationship_llm: Runnable[list[Any], RelationshipMappingOutput]
    grounding_llm: Runnable[list[Any], GroundingVerificationOutput]
    finalization_llm: Runnable[list[Any], Any]  # FinalReport

    # The compiled deep-research subgraph behind its single tool wrapper
    # (D5). ``None`` only when no EvidenceService is provisioned; the node
    # then refuses with a typed service-unavailable failure.
    deep_research_tool: Any | None = None


class SaulGraphNodes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gateway: Any
    qna: Any
    orchestrator: Any
    planner_generate: Any
    planner: Any
    ingestion: Any
    normalization: Any
    segmentation: Any
    entity_extraction: Any
    relationship_mapping: Any
    risk_analysis: Any
    compliance: Any
    grounding_verification: Any
    human_review: Any
    # The validation layer's HumanReviewSink wired to the human_review node
    # (D7 / ADR-2): every Saul-side ESCALATE routes into the HITL review
    # through this sink, never around it. ``None`` only for hand-built node
    # sets that predate the sink seam.
    human_review_sink: Any | None = None
    finalization: Any
    persist_memory: Any
    deep_research: Any


def build_agent_registry(  # noqa: PLR0914 — one readable composition of agents, tools, and chains
    pro_llm: BaseChatModel,
    flash_llm: BaseChatModel,
    tools: AgentToolBundle | None = None,
    *,
    traversal_graphiti_service: BoundGraphitiService | None = None,
    traversal_gate_probe: RelationshipEdgeProbe | None = None,
) -> AgentRegistry:
    """
    Instantiate all agents + LLM chains once.
    Called from build_saul_graph — never call this inside a node function.

    Model assignment:
      Pro  (thinking_level=high)  → deep reasoning: orchestrator, risk, compliance
      Flash (thinking_level=none) → fast structural: qna, planner, pipeline nodes

    Portfolio traversal binding (openspec ``agent-portfolio-traversal-tools``,
    design D2): the binding-time gate. Traversal tools are bound to the
    specialist projections only when ``traversal_graphiti_service`` is
    provided AND ``traversal_gate_probe`` — obtained by the async caller via
    ``relationship_edges_available(...)`` before this sync builder — reports
    ``edges_present``. A missing probe is treated as a scopeless corpus and a
    graphless (edge-less) corpus fails the build loudly with a
    ``TraversalGateError`` naming the gate; no role's toolset ever contains a
    traversal tool otherwise. The catalog itself always carries the three
    tools (registered backend-less by ``register_default_tools``), so the
    policy's ``portfolio.traversal`` grants always name a declared capability.
    """
    # --- create_agent nodes (Pro LLM + bound tools) ------------------------
    # Each role binds the tools its spec assigns (agent-tool-registry):
    #   risk       -> knowledge-graph query + obligation chain (+ handoff)
    #   compliance -> precedent search + statute retrieval (+ handoff)
    # Evidence tools come from the AgentToolBundle wired at graph-build time;
    # handoff tools resolve through the explicit global registry. Errors
    # surface through the standard retry middleware — the only mechanism
    # create_agent runs.
    register_default_tools()
    handoff = get_tool_registry()

    # Portfolio traversal catalog seats (task 3.1): backend-less instances so
    # the policy's portfolio.traversal grants name declared capabilities from
    # the first validation. The features-side import is legal here; the
    # registry itself never imports app.features (shared-runtime contract).
    register_portfolio_traversal_tools(make_portfolio_traversal_tools(None), validate=False)

    # D5: the deep-research subgraph wrapper registers once in the single
    # registry, compiled here at graph-build time, with read-only authority
    # tags. The deep_research role's projection stays at exactly its transfer
    # tool plus this one research tool.
    deep_research_tool: Any | None = None
    if tools is not None and tools.evidence_service is not None:
        deep_research_tool = make_deep_research_tool(tools.evidence_service)
        register_deep_research_tool(deep_research_tool)

    # Gated portfolio traversal (task 3.3): backend-bound instances replace
    # the catalog's backend-less registrations only behind the binding-time
    # gate; a scopeless probe (no probe supplied) fails closed here too.
    traversal_tools: tuple[Any, ...] = ()
    if traversal_graphiti_service is not None:
        require_traversal_gate(
            traversal_gate_probe
            if traversal_gate_probe is not None
            else RelationshipEdgeProbe(verdict=RelationshipEdgeVerdict.NO_SCOPE)
        )
        traversal_tools = make_portfolio_traversal_tools(traversal_graphiti_service)
        register_portfolio_traversal_tools(traversal_tools)

    def _transfer(name: str) -> Any:
        return handoff.get(name)

    risk_tools: list[Any] = [_transfer("transfer_to_orchestrator")]
    compliance_tools: list[Any] = [
        _transfer("transfer_to_orchestrator"),
        _transfer("transfer_to_risk"),
    ]
    if tools is not None:
        # Keep each specialist's active toolset bounded.  The evidence tools
        # are the authoritative search/read seam; legacy specialist tools stay
        # available only where they add a distinct graph capability.
        risk_tools += [tools.query_knowledge_graph, tools.get_obligation_chain]
        compliance_tools += [tools.search_legal_precedents, tools.retrieve_statute_section]
        if tools.search_evidence is not None and tools.read_evidence_by_id is not None:
            # Both specialists use the same scoped evidence seam; both receive
            # readiness (D3 parity) so missing or incomplete uploaded
            # authorities cannot be interpreted as a complete analysis.
            compliance_tools = [
                _transfer("transfer_to_orchestrator"),
                _transfer("transfer_to_risk"),
                tools.search_evidence,
                tools.read_evidence_by_id,
            ]
            risk_tools = [
                _transfer("transfer_to_orchestrator"),
                tools.query_knowledge_graph,
                tools.get_obligation_chain,
                tools.search_evidence,
                tools.read_evidence_by_id,
            ]
            if tools.get_ingestion_state is not None:
                compliance_tools.append(tools.get_ingestion_state)
                risk_tools.append(tools.get_ingestion_state)

    # Portfolio traversal bindings (task 3.3): reached only when the
    # binding-time gate passed above. risk was already at 6 tools with
    # readiness parity, so it gains the cross-document obligations sweep;
    # compliance gains the conflict finder plus the party web. Both
    # projections land at exactly the ≤7 cap without a same-role removal —
    # the no-swap decision recorded in the change's design.md addendum.
    if traversal_tools:
        cross_document_obligations, find_conflicting_clauses, party_obligation_map = traversal_tools
        risk_tools.append(cross_document_obligations)
        compliance_tools.extend([find_conflicting_clauses, party_obligation_map])

    risk_agent = create_agent(
        model=pro_llm,
        tools=risk_tools,
        system_prompt=_RISK_ANALYSIS_SYSTEM_PROMPT,
        # LangChain's ToolRetryMiddleware is intentionally absent here: this graph
        # combines reads, handoffs, and potentially side-effecting tools. A retry
        # allowlist must be supplied at the application middleware configuration seam.
        middleware=[],
        response_format=RiskAnalysisOutput,
        # ``TypedDict`` is valid at runtime for LangChain's context schema, but
        # ty cannot express it as ``type[ContextT]`` in this overloaded call.
        context_schema=cast("type[Any]", ToolRuntimeContext),
    )

    compliance_agent = create_agent(
        model=pro_llm,
        tools=compliance_tools,
        system_prompt=_COMPLIANCE_SYSTEM_PROMPT,
        middleware=[],
        response_format=ComplianceOutput,
        context_schema=cast("type[Any]", ToolRuntimeContext),
    )

    # --- with_structured_output chains (Flash LLM) -------------------------
    qna_llm = cast(
        "Runnable[list[Any], QnAOutput]",
        flash_llm.with_structured_output(QnAOutput),
    )
    planner_llm = cast(
        "Runnable[list[Any], PlannerOutput]",
        flash_llm.with_structured_output(PlannerOutput),
    )
    normalization_llm = cast(
        "Runnable[list[Any], NormalizedDocument]",
        flash_llm.with_structured_output(NormalizedDocument),
    )
    segmentation_llm = cast(
        "Runnable[list[Any], ClauseSegmentationOutput]",
        flash_llm.with_structured_output(ClauseSegmentationOutput),
    )
    entity_llm = cast(
        "Runnable[list[Any], EntityExtractionOutput]",
        flash_llm.with_structured_output(EntityExtractionOutput),
    )
    relationship_llm = cast(
        "Runnable[list[Any], RelationshipMappingOutput]",
        flash_llm.with_structured_output(RelationshipMappingOutput),
    )
    grounding_llm = cast(
        "Runnable[list[Any], GroundingVerificationOutput]",
        flash_llm.with_structured_output(GroundingVerificationOutput),
    )
    finalization_llm = cast("Runnable[list[Any], Any]", pro_llm.with_structured_output(FinalReport))

    return AgentRegistry(
        risk_agent=risk_agent,
        compliance_agent=compliance_agent,
        qna_llm=qna_llm,
        planner_llm=planner_llm,
        normalization_llm=normalization_llm,
        segmentation_llm=segmentation_llm,
        entity_llm=entity_llm,
        relationship_llm=relationship_llm,
        grounding_llm=grounding_llm,
        finalization_llm=finalization_llm,
        deep_research_tool=deep_research_tool,
    )


def _build_graph_nodes(
    registry: AgentRegistry,
    pro_llm: BaseChatModel,
    memory_service: Any,
    tool_registry: AgentToolBundle,
) -> SaulGraphNodes:
    # D7 / ADR-2: the review node is built only through its sink entry, so a
    # Saul-side ESCALATE reaches HITL through the HumanReviewSink contract.
    review_sink, review_node = make_human_review_entry()
    return SaulGraphNodes(
        gateway=make_gateway_node(),
        qna=make_qna_node(registry.qna_llm),
        orchestrator=make_orchestrator_node(
            cast(
                "Runnable[list[Any], OrchestratorAction]",
                pro_llm.with_structured_output(OrchestratorAction),
            )
        ),
        planner_generate=make_plan_generation_node(registry.planner_llm),
        planner=make_plan_approval_node(),
        ingestion=make_ingestion_node(),
        normalization=make_normalization_node(registry.normalization_llm),
        segmentation=make_segmentation_node(registry.segmentation_llm),
        entity_extraction=make_entity_extraction_node(registry.entity_llm),
        relationship_mapping=make_relationship_mapping_node(registry.relationship_llm),
        risk_analysis=make_risk_analysis_node(registry.risk_agent),
        compliance=make_compliance_node(registry.compliance_agent),
        # The verdict path escalates through the same sink the review node
        # drains: a real grounding failure reaches HITL via the sink's
        # thread-buffered enqueue, never around the seam.
        grounding_verification=make_grounding_verification_node(
            registry.grounding_llm, human_review_sink=review_sink
        ),
        human_review=review_node,
        human_review_sink=review_sink,
        finalization=make_finalization_node(registry.finalization_llm),
        persist_memory=make_persist_memory_node(memory_service),
        deep_research=make_deep_research_node(
            deep_research_tool=registry.deep_research_tool,
            evidence_service=tool_registry.evidence_service,
        ),
    )
