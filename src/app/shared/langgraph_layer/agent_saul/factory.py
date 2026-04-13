"""
Graph factory for Agent Saul.

Entry point: build_saul_graph(checkpointer, pro_llm, flash_llm, cognee_client)
Returns: CompiledStateGraph — store this in app.state.saul_graph during lifespan.

Lifespan wiring (in src/app/lifecycle/lifespan.py):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from app.shared.langchain_layer.models import build_chat_model
    from app.shared.langgraph_layer.agent_saul.graph.nodes.factory import build_saul_graph

    pro_llm = build_chat_model(
        model_name=settings.model.gemini_pro_model,
        streaming=True,
    )
    flash_llm = build_chat_model(
        model_name=settings.model.gemini_flash_model,
        streaming=True,
    )

    saul_checkpointer = AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL_ASYNC)
    await saul_checkpointer.setup()
    app.state.saul_checkpointer = saul_checkpointer

    app.state.saul_graph = build_saul_graph(
        checkpointer=saul_checkpointer,
        pro_llm=pro_llm,
        flash_llm=flash_llm,
        cognee_client=app.state.cognee,  # your injected Cognee client
    )

Performance guarantee:
    All agents and structured-output LLM chains are created ONCE here,
    at lifespan startup.  Node functions are closures that reference
    pre-built agents — never re-initialise inside a node call.
    500ms+ latency → ~30ms per node call.
"""
from dataclasses import dataclass
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .analysis_nodes import (
    make_compliance_node,
    make_finalization_node,
    make_grounding_verification_node,
    make_human_review_node,
    make_persist_memory_node,
    make_risk_analysis_node,
)
from .gateway import make_gateway_node
from .ingestion import make_ingestion_node
from .orchestrator import (
    make_orchestrator_node,
    route_from_orchestrator,
)
from .pipeline_nodes import (
    ClauseSegmentationOutput,
    EntityExtractionOutput,
    NormalizedDocument,
    RelationshipMappingOutput,
    dispatch_entity_extraction,
    make_entity_extraction_node,
    make_normalization_node,
    make_relationship_mapping_node,
    make_segmentation_node,
)
from .planner import PlannerOutput, make_planner_node
from .prompt import (
    _COMPLIANCE_SYSTEM_PROMPT,
    _ORCHESTRATOR_SYSTEM_PROMPT,
    _RISK_ANALYSIS_SYSTEM_PROMPT,
)
from .qna import QnAOutput, make_qna_node, route_after_qna
from .state import (
    GroundingVerificationOutput,
    LegalAgentState,
    OrchestratorAction,
)

# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------


@dataclass
class AgentRegistry:
    """
    Holds all pre-built agents and structured-output LLM chains.
    Created once at lifespan startup; referenced by closure in every node.

    Separation:
      create_react_agent → nodes that need tool-calling (orchestrator, risk, compliance)
      llm.with_structured_output → schema-locked nodes (qna, planner, pipeline nodes)
    """

    # create_react_agent compiled sub-graphs
    orchestrator_agent: Any  # CompiledStateGraph
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


@dataclass
class SaulGraphNodes:
    gateway: Any
    qna: Any
    orchestrator: Any
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
    finalization: Any
    persist_memory: Any


def build_agent_registry(
    pro_llm: BaseChatModel,
    flash_llm: BaseChatModel,
) -> AgentRegistry:
    """
    Instantiate all agents + LLM chains once.
    Called from build_saul_graph — never call this inside a node function.

    Model assignment:
      Pro  (thinking_level=high)  → deep reasoning: orchestrator, risk, compliance
      Flash (thinking_level=none) → fast structural: qna, planner, pipeline nodes
    """
    # --- create_agent nodes (Pro LLM + tool stubs) -------------------------
    orchestrator_agent = create_agent(
        model=pro_llm,
        tools=[],  # TODO: add delegation tools when available
        system_prompt=_ORCHESTRATOR_SYSTEM_PROMPT,
    )

    risk_agent = create_agent(
        model=pro_llm,
        tools=[],  # TODO: add search_caselaw, retrieve_statute tools
        system_prompt=_RISK_ANALYSIS_SYSTEM_PROMPT,
    )

    compliance_agent = create_agent(
        model=pro_llm,
        tools=[],  # TODO: add retrieve_precedent, check_statute tools
        system_prompt=_COMPLIANCE_SYSTEM_PROMPT,
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
    finalization_llm = cast(
        "Runnable[list[Any], Any]",
        pro_llm.with_structured_output(
        # FinalReport — imported inline to avoid circular at module level
            __import__(
                "app.shared.langgraph_layer.agent_saul.graph.state",
                fromlist=["FinalReport"],
            ).FinalReport
        ),
    )

    return AgentRegistry(
        orchestrator_agent=orchestrator_agent,
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
    )


def _build_graph_nodes(
    registry: AgentRegistry,
    pro_llm: BaseChatModel,
    cognee_client: Any,
) -> SaulGraphNodes:
    return SaulGraphNodes(
        gateway=make_gateway_node(),
        qna=make_qna_node(registry.qna_llm),
        orchestrator=make_orchestrator_node(
            cast(
                "Runnable[list[Any], OrchestratorAction]",
                pro_llm.with_structured_output(OrchestratorAction),
            )
        ),
        planner=make_planner_node(registry.planner_llm),
        ingestion=make_ingestion_node(),
        normalization=make_normalization_node(registry.normalization_llm),
        segmentation=make_segmentation_node(registry.segmentation_llm),
        entity_extraction=make_entity_extraction_node(registry.entity_llm),
        relationship_mapping=make_relationship_mapping_node(registry.relationship_llm),
        risk_analysis=make_risk_analysis_node(registry.risk_agent),
        compliance=make_compliance_node(registry.compliance_agent),
        grounding_verification=make_grounding_verification_node(registry.grounding_llm),
        human_review=make_human_review_node(),
        finalization=make_finalization_node(registry.finalization_llm),
        persist_memory=make_persist_memory_node(cognee_client),
    )


def _wire_graph(graph: Any, nodes: SaulGraphNodes) -> None:
    graph.add_node("gateway", nodes.gateway)
    graph.add_node("qna", nodes.qna)
    graph.add_node("orchestrator", nodes.orchestrator)
    graph.add_node("planner", nodes.planner)
    graph.add_node("ingestion", nodes.ingestion)
    graph.add_node("normalization", nodes.normalization)
    graph.add_node("segmentation", nodes.segmentation)
    graph.add_node("entity_extraction", nodes.entity_extraction)
    graph.add_node("relationship_mapping", nodes.relationship_mapping)
    graph.add_node("risk_analysis", nodes.risk_analysis)
    graph.add_node("compliance", nodes.compliance)
    graph.add_node("grounding_verification", nodes.grounding_verification)
    graph.add_node("human_review", nodes.human_review)
    graph.add_node("finalization", nodes.finalization)
    graph.add_node("persist_memory", nodes.persist_memory)

    # Entry
    graph.set_entry_point("gateway")
    graph.add_edge("gateway", "qna")

    # QnA: self-loop on low confidence, proceed on threshold met
    graph.add_conditional_edges("qna", route_after_qna, {"qna": "qna", "orchestrator": "orchestrator"})

    # Orchestrator: dynamic routing via action schema
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "planner": "planner",
            "ingestion": "ingestion",
            "finalization": "finalization",
            END: END,
        },
    )

    # Planner → back to orchestrator (after HITL approval)
    graph.add_edge("planner", "orchestrator")

    # Linear pipeline
    graph.add_edge("ingestion", "normalization")
    graph.add_edge("normalization", "segmentation")

    # Segmentation → Send fan-out → parallel entity_extraction
    graph.add_conditional_edges(
        "segmentation",
        dispatch_entity_extraction,
        # No path_map needed: Send targets are resolved dynamically
    )

    # entity_extraction → relationship_mapping (join: all parallel executions must complete)
    graph.add_edge("entity_extraction", "relationship_mapping")

    # Parallel branches: relationship_mapping → risk_analysis AND compliance
    graph.add_edge("relationship_mapping", "risk_analysis")
    graph.add_edge("relationship_mapping", "compliance")

    # Join at grounding_verification (waits for both parallel branches)
    graph.add_edge("risk_analysis", "grounding_verification")
    graph.add_edge("compliance", "grounding_verification")

    # Mandatory human review → back to orchestrator for reflection
    graph.add_edge("grounding_verification", "human_review")
    graph.add_edge("human_review", "orchestrator")

    # Terminal
    graph.add_edge("finalization", "persist_memory")
    graph.add_edge("persist_memory", END)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_saul_graph(
    checkpointer: AsyncPostgresSaver,
    pro_llm: BaseChatModel,
    flash_llm: BaseChatModel,
    cognee_client: Any,
) -> CompiledStateGraph:
    """
    Build and compile the Agent Saul LangGraph.

    Graph topology:
      START → gateway → qna → [conditional: loop or orchestrator]
      orchestrator → [conditional: planner | ingestion | finalization | END]
      planner → [HITL] → orchestrator
      ingestion → normalization → segmentation
      segmentation → [Send fan-out] → entity_extraction (parallel N)
      entity_extraction → relationship_mapping  (join via operator.add reducer)
      relationship_mapping → risk_analysis       (parallel branch 1)
      relationship_mapping → compliance          (parallel branch 2)
      risk_analysis  → grounding_verification   (join)
      compliance     → grounding_verification   (join)
      grounding_verification → human_review → [HITL]
      human_review → orchestrator               (reflect loop)
      finalization → persist_memory → END
    """
    registry = build_agent_registry(pro_llm, flash_llm)
    nodes = _build_graph_nodes(registry=registry, pro_llm=pro_llm, cognee_client=cognee_client)

    # --- Wire the graph ----------------------------------------------------
    state_graph_factory = cast("Any", StateGraph)
    graph: Any = state_graph_factory(LegalAgentState)
    _wire_graph(graph=graph, nodes=nodes)

    # --- Compile with async Postgres checkpointer --------------------------
    return cast("CompiledStateGraph", graph.compile(checkpointer=checkpointer))
