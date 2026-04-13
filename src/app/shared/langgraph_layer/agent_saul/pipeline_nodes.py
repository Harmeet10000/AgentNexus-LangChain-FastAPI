"""
Pipeline nodes: normalization, segmentation, entity_extraction, relationship_mapping.

All use llm.with_structured_output() — no create_react_agent.
These are schema-locked nodes: inputs and outputs are fully typed.

entity_extraction is the Send fan-out target.  It receives ClauseExtractionInput
(not the full LegalAgentState) and returns {"extracted_entities": [...]}.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.types import Send
from pydantic import BaseModel, Field

from app.utils import logger

from .prompt import (
    _ENTITY_EXTRACTION_SYSTEM_PROMPT,
    _NORMALIZATION_SYSTEM_PROMPT,
    _RELATIONSHIP_MAPPING_SYSTEM_PROMPT,
    _SEGMENTATION_SYSTEM_PROMPT,
)
from .state import (
    AgentError,
    CitedEntity,
    ClauseExtractionInput,
    ClauseSegment,
    LegalAgentState,
    LegalRelationship,
    NormalizedDocument,
    WorkflowStatus,
)

# ===========================================================================
# Normalization Node
# ===========================================================================




class NormalizationInput(BaseModel):
    document_text: str


def make_normalization_node(
    normalization_llm: Runnable[list[Any], NormalizedDocument],
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def normalization_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="normalization", doc_id=state["doc_id"])

        if not state.get("document_text"):
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="normalization",
                        code="MISSING_DOCUMENT_TEXT",
                        message="document_text not populated by ingestion",
                        retryable=False,
                    )
                ],
            }

        messages = [
            SystemMessage(content=_NORMALIZATION_SYSTEM_PROMPT),
            HumanMessage(content=state["document_text"]),
        ]
        result: NormalizedDocument = await normalization_llm.ainvoke(messages)
        log.info("normalization_completed", section_count=len(result.sections))

        return {
            "normalized_document": result,
            "status": WorkflowStatus.SEGMENTING,
        }

    return normalization_node


# ===========================================================================
# Segmentation Node
# ===========================================================================




class ClauseSegmentationOutput(BaseModel):
    segments: list[ClauseSegment] = Field(min_length=1)


def make_segmentation_node(
    segmentation_llm: Runnable[list[Any], ClauseSegmentationOutput],
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def segmentation_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="segmentation", doc_id=state["doc_id"])

        if not state.get("normalized_document"):
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="segmentation",
                        code="MISSING_NORMALIZED_DOCUMENT",
                        message="normalized_document not populated by normalization node",
                        retryable=False,
                    )
                ],
            }

        doc_text = "\n".join(
            s.content for s in state["normalized_document"].sections  # type: ignore[union-attr]
        )
        messages = [
            SystemMessage(content=_SEGMENTATION_SYSTEM_PROMPT),
            HumanMessage(content=doc_text),
        ]
        result: ClauseSegmentationOutput = await segmentation_llm.ainvoke(messages)
        log.info("segmentation_completed", segment_count=len(result.segments))

        # Return segments — operator.add reducer appends to state["segments"]
        return {
            "segments": list(result.segments),
            "status": WorkflowStatus.EXTRACTING_ENTITIES,
        }

    return segmentation_node


# ---------------------------------------------------------------------------
# Send dispatcher: fan-out segmentation → parallel entity_extraction
# ---------------------------------------------------------------------------


def dispatch_entity_extraction(state: LegalAgentState) -> list[Send]:
    """Conditional edge function.

    Returns one Send per clause segment.  LangGraph runs all entity_extraction
    nodes in parallel.  Results accumulate via operator.add on extracted_entities.
    """
    document_context: dict[str, Any] = {
        "jurisdiction": state["working_memory"].get("jurisdiction", "India"),
        "document_type": state["working_memory"].get("document_type", "unknown"),
        "doc_id": state["doc_id"],
    }
    return [
        Send(
            "entity_extraction",
            ClauseExtractionInput(
                clause=segment,
                document_context=document_context,
            ).model_dump(),
        )
        for segment in state["segments"]
    ]


# ===========================================================================
# Entity Extraction Node (Send fan-out target)
# ===========================================================================




class EntityExtractionOutput(BaseModel):
    entities: list[CitedEntity]
    overall_confidence: float = Field(ge=0.0, le=1.0)


def make_entity_extraction_node(
    entity_llm: Runnable[list[Any], EntityExtractionOutput],
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """
    NOTE: This node receives ClauseExtractionInput (via Send), NOT LegalAgentState.
    It returns a partial state dict that is merged into the main graph state
    via the operator.add reducer on extracted_entities.
    """

    async def entity_extraction_node(state: dict[str, Any]) -> dict[str, Any]:
        # Deserialise from Send payload
        clause = ClauseSegment.model_validate(state["clause"])
        doc_context: dict[str, Any] = state.get("document_context", {})

        log = logger.bind(node="entity_extraction", clause_id=clause.clause_id)

        prompt = (
            f"Document type: {doc_context.get('document_type', 'unknown')}\n"
            f"Jurisdiction: {doc_context.get('jurisdiction', 'India')}\n"
            f"Clause ID: {clause.clause_id}\n"
            f"Clause type: {clause.clause_type}\n\n"
            f"Clause text:\n{clause.text}"
        )
        messages = [
            SystemMessage(content=_ENTITY_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        result: EntityExtractionOutput = await entity_llm.ainvoke(messages)
        log.info("entity_extraction_done", entity_count=len(result.entities))

        # operator.add on extracted_entities appends these to the main state list
        return {"extracted_entities": list(result.entities)}

    return entity_extraction_node


# ===========================================================================
# Relationship Mapping Node
# ===========================================================================




class RelationshipMappingOutput(BaseModel):
    relationships: list[LegalRelationship]


def make_relationship_mapping_node(
    relationship_llm: Runnable[list[Any], RelationshipMappingOutput],
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def relationship_mapping_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="relationship_mapping",
            entity_count=len(state["extracted_entities"]),
        )

        entity_summary = "\n".join(
            f"[{e.entity_type}] {e.value} (clause: {e.clause_id}, party: {e.party or 'N/A'})"
            for e in state["extracted_entities"]
        )
        messages = [
            SystemMessage(content=_RELATIONSHIP_MAPPING_SYSTEM_PROMPT),
            HumanMessage(content=entity_summary),
        ]
        result: RelationshipMappingOutput = await relationship_llm.ainvoke(messages)
        log.info("relationship_mapping_done", relationship_count=len(result.relationships))

        return {
            "relationships": list(result.relationships),
            "status": WorkflowStatus.ANALYZING_RISKS,
        }

    return relationship_mapping_node
