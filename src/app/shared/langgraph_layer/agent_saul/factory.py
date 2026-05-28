"""Startup-time Agent Saul graph composition helpers."""
from dataclasses import dataclass
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from .nodes import (
    ClauseSegmentationOutput,
    EntityExtractionOutput,
    PlannerOutput,
    QnAOutput,
    RelationshipMappingOutput,
    make_compliance_node,
    make_entity_extraction_node,
    make_finalization_node,
    make_gateway_node,
    make_grounding_verification_node,
    make_human_review_node,
    make_ingestion_node,
    make_normalization_node,
    make_orchestrator_node,
    make_persist_memory_node,
    make_planner_node,
    make_qna_node,
    make_relationship_mapping_node,
    make_risk_analysis_node,
    make_segmentation_node,
)
from .prompts import (
    _COMPLIANCE_SYSTEM_PROMPT,
    _ORCHESTRATOR_SYSTEM_PROMPT,
    _RISK_ANALYSIS_SYSTEM_PROMPT,
)
from .state import (
    FinalReport,
    GroundingVerificationOutput,
    NormalizedDocument,
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
    finalization_llm = cast("Runnable[list[Any], Any]", pro_llm.with_structured_output(FinalReport))

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
