"""Plan-conditional entity extraction and relationship mapping (task 6.3).

Extraction and mapping only run for plans whose steps consume them; the
persisted LangExtract metadata recorded at ingestion is the preferred scope
authority over model-invented entity classes.
"""

from typing import Any

import pytest

from app.features.agent_saul.domain.state import (
    ClauseSegment,
    ClauseType,
    PlanActionType,
    PlanStep,
    WorkflowStatus,
)
from app.features.agent_saul.graph.nodes import (
    dispatch_entity_extraction,
    make_entity_extraction_node,
    make_relationship_mapping_node,
)

pytestmark = pytest.mark.unit


def _step(action: PlanActionType, step_id: str = "S-1") -> PlanStep:
    return PlanStep(step_id=step_id, action=action, description=str(action.value))


def _segment(*, index: int = 0, metadata: dict[str, Any] | None = None) -> ClauseSegment:
    return ClauseSegment(
        clause_id=f"canonical:doc-a:v1:chunk:{index}",
        clause_type=ClauseType.INDEMNITY,
        section_ref=f"7.{index}",
        start_char=0,
        end_char=40,
        text="The Supplier shall indemnify the Customer against losses.",
        document_id="doc-a",
        document_version=1,
        chunk_id=f"chunk-{index}",
        chunk_index=index,
        contract_index=0,
        metadata=metadata or {},
    )


def _entity() -> Any:
    from app.features.agent_saul.domain.state import Citation, CitedEntity, EntityType

    return CitedEntity(
        entity_id=f"E-{_entity.counter}",
        entity_type=EntityType.PARTY,
        value="Supplier",
        clause_id="canonical:doc-a:v1:chunk:0",
        citation=Citation(claim="party named", source="chunk-0", confidence=0.9),
    )


_entity.counter = 0


def _entities(count: int) -> list[Any]:
    produced = [_entity() for _ in range(count)]
    _entity.counter += count
    return produced


def _extract_state(*plan: PlanActionType, entities: int = 0) -> dict[str, Any]:
    return {
        "doc_id": "doc-a",
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "allowed_doc_ids": ["doc-a"],
        "working_memory": {},
        "status": WorkflowStatus.PLAN_APPROVED,
        "plan": [_step(action, f"S-{number}") for number, action in enumerate(plan, start=1)],
        "segments": [_segment(index=index) for index in range(2)],
        "extracted_entities": _entities(entities),
        "relationships": [],
    }


class _RecordingLLM:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        return self.result


# --- dispatch gate ----------------------------------------------------------


def test_research_and_summary_plan_routes_past_the_extraction_fanout() -> None:
    state = _extract_state(PlanActionType.SEARCH_PRECEDENTS, PlanActionType.SUMMARIZE)
    assert dispatch_entity_extraction(state) == "relationship_mapping"  # type: ignore[arg-type]


def test_risk_plan_fans_out_one_send_per_segment() -> None:
    state = _extract_state(PlanActionType.RISK_ANALYSIS)
    sends = dispatch_entity_extraction(state)  # type: ignore[arg-type]
    assert isinstance(sends, list)
    assert len(sends) == 2
    assert all(send.node == "entity_extraction" for send in sends)


def test_compliance_plan_also_fans_out() -> None:
    state = _extract_state(PlanActionType.COMPLIANCE_CHECK)
    sends = dispatch_entity_extraction(state)  # type: ignore[arg-type]
    assert isinstance(sends, list)
    assert len(sends) == 2


# --- entity extraction node -------------------------------------------------


def _send_payload(segment: ClauseSegment, *, entities_required: bool) -> dict[str, Any]:
    return {
        "clause": segment,
        "document_context": {
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "allowed_doc_ids": ("doc-a",),
            "entities_required": entities_required,
        },
    }


def _entity_llm() -> Any:
    from app.features.agent_saul.graph.nodes import EntityExtractionOutput

    return _RecordingLLM(EntityExtractionOutput(entities=[], overall_confidence=1.0))


async def test_entity_node_performs_no_model_call_for_uninterested_plan() -> None:
    llm = _entity_llm()
    node = make_entity_extraction_node(llm)  # type: ignore[arg-type]

    result = await node(_send_payload(_segment(), entities_required=False))  # type: ignore[arg-type]

    assert result == {}
    assert llm.calls == []


async def test_entity_node_prefers_persisted_langextract_class_scope() -> None:
    llm = _entity_llm()
    node = make_entity_extraction_node(llm)  # type: ignore[arg-type]
    segment = _segment(
        metadata={"knowledge_extraction": "complete", "knowledge_extraction_classes": ["PARTY"]}
    )

    await node(_send_payload(segment, entities_required=True))  # type: ignore[arg-type]

    assert llm.calls, "an interested plan must run extraction"
    prompt = str(llm.calls[0][1].content)
    assert "authoritative" in prompt
    assert "PARTY" in prompt


async def test_entity_node_discloses_incomplete_persisted_extraction() -> None:
    llm = _entity_llm()
    node = make_entity_extraction_node(llm)  # type: ignore[arg-type]
    segment = _segment(metadata={"knowledge_extraction": "incomplete"})

    await node(_send_payload(segment, entities_required=True))  # type: ignore[arg-type]

    prompt = str(llm.calls[0][1].content)
    assert "did not complete" in prompt


# --- relationship mapping node ----------------------------------------------


def _relationship_result() -> Any:
    from app.features.agent_saul.domain.state import Citation, LegalRelationship
    from app.retrieval.graphiti.schemas import RelationshipType

    relationship = LegalRelationship(
        edge_id="E-1",
        from_node="PARTY:Supplier",
        relationship=RelationshipType.OBLIGES,
        to_node="OBLIGATION:indemnify",
        clause_id="canonical:doc-a:v1:chunk:0",
        citation=Citation(claim="supplier must indemnify", source="chunk-0", confidence=0.9),
    )
    return type("R", (), {"relationships": [relationship]})


async def test_relationship_mapping_skips_model_for_uninterested_plan() -> None:
    llm = _RecordingLLM(_relationship_result())
    node = make_relationship_mapping_node(llm)  # type: ignore[arg-type]
    state = _extract_state(PlanActionType.SUMMARIZE, entities=2)

    result = await node(state)  # type: ignore[arg-type]

    assert llm.calls == []
    assert result["status"] is WorkflowStatus.ANALYZING_RISKS


async def test_relationship_mapping_skips_model_without_entities() -> None:
    llm = _RecordingLLM(_relationship_result())
    node = make_relationship_mapping_node(llm)  # type: ignore[arg-type]
    state = _extract_state(PlanActionType.RISK_ANALYSIS, entities=0)

    result = await node(state)  # type: ignore[arg-type]

    assert llm.calls == []
    assert result["status"] is WorkflowStatus.ANALYZING_RISKS


async def test_relationship_mapping_runs_for_risk_plan_with_entities() -> None:
    llm = _RecordingLLM(_relationship_result())
    node = make_relationship_mapping_node(llm)  # type: ignore[arg-type]
    state = _extract_state(PlanActionType.RISK_ANALYSIS, entities=2)

    result = await node(state)  # type: ignore[arg-type]

    assert len(llm.calls) == 1
    assert result["status"] is WorkflowStatus.ANALYZING_RISKS
