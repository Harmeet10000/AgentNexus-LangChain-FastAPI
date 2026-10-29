"""Node-boundary delegation pairs (agent-message-standard 2.1)."""

import json
from typing import Any

import pytest
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.features.agent_saul.domain.state import (
    IngestionDocumentState,
    IngestionDocumentStatus,
    OrchestratorAction,
    OrchestratorActionType,
    WorkflowStatus,
)
from app.features.agent_saul.graph.nodes import (
    delegation_pairs_from,
    make_compliance_node,
    make_orchestrator_node,
    make_risk_analysis_node,
)


def _transfer_messages() -> list[Any]:
    return [
        HumanMessage(content="review this clause"),
        AIMessage(
            content="handing off",
            tool_calls=[
                {
                    "name": "transfer_to_orchestrator",
                    "args": {"reason": "needs replan"},
                    "id": "call_internal_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="{'transfer_to': 'orchestrator'}", tool_call_id="call_internal_1"),
    ]


class _FakeAgent:
    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages

    async def ainvoke(self, _input: dict[str, Any]) -> dict[str, Any]:
        return {"messages": self._messages}


def _state() -> dict[str, Any]:
    return {
        "user_id": "u",
        "thread_id": "t",
        "doc_id": "d",
        "segments": [],
        "extracted_entities": [],
        "relationships": [],
        "messages": [],
    }


def test_pairs_extracted_from_sub_agent_messages() -> None:
    pairs = delegation_pairs_from(_transfer_messages())

    assert len(pairs) == 1
    request, answer = pairs[0]
    assert isinstance(request, AIMessage)
    assert isinstance(answer, ToolMessage)
    assert request.tool_calls[0]["name"] == "transfer_to_orchestrator"
    assert answer.tool_call_id == request.tool_calls[0]["id"]


def test_no_transfer_means_no_pairs() -> None:
    assert delegation_pairs_from([HumanMessage(content="plain")]) == []


async def test_risk_node_appends_linked_pair() -> None:
    node = make_risk_analysis_node(_FakeAgent(_transfer_messages()))

    result = await node(_state())  # type: ignore[arg-type]

    names = [type(m).__name__ for m in result["messages"]]
    assert names == ["AIMessage", "ToolMessage"]
    assert result["messages"][1].tool_call_id == result["messages"][0].tool_calls[0]["id"]
    assert result["risk_analysis"] is not None


async def test_compliance_node_appends_linked_pair() -> None:
    node = make_compliance_node(_FakeAgent(_transfer_messages()))

    result = await node(_state())  # type: ignore[arg-type]

    assert len(result["messages"]) == 2
    assert result["compliance_result"] is not None


async def test_compliance_node_maps_invalid_structured_output_to_graph_failure() -> None:
    class InvalidAgent:
        async def ainvoke(self, _input: dict[str, Any]) -> dict[str, Any]:
            message = "missing required citation"
            tool_name = "ComplianceOutput"
            raise StructuredOutputValidationError(
                tool_name,
                message,
                AIMessage(content="invalid structured output"),
            )

    result = await make_compliance_node(InvalidAgent())(_state())  # type: ignore[arg-type]

    assert result["status"].value == "failed"
    assert result["errors"][0].code == "INVALID_COMPLIANCE_OUTPUT"
    assert result["errors"][0].retryable is True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (WorkflowStatus.FINALIZING, OrchestratorActionType.SYNTHESIZE),
        (WorkflowStatus.FAILED, OrchestratorActionType.DONE),
        (WorkflowStatus.COMPLETED, OrchestratorActionType.DONE),
    ],
)
async def test_orchestrator_routes_terminal_workflow_states_without_model_call(
    status: WorkflowStatus,
    expected: OrchestratorActionType,
) -> None:
    class NeverCalled:
        async def ainvoke(self, _input: list[Any]) -> Any:
            message = "terminal orchestrator must not invoke the model"
            raise AssertionError(message)

    state = {
        **_state(),
        "status": status,
        "current_step": 3,
    }
    result = await make_orchestrator_node(NeverCalled())(state)  # type: ignore[arg-type]

    assert result["orchestrator_action"].action_type == expected
    assert result["current_step"] == 4


async def test_orchestrator_receives_bounded_ingestion_and_delegation_context() -> None:
    class Capture:
        def __init__(self) -> None:
            self.inputs: list[Any] = []

        async def ainvoke(self, messages: list[Any]) -> OrchestratorAction:
            self.inputs.append(messages)
            return OrchestratorAction(
                action_type=OrchestratorActionType.CONTINUE,
                target_node="risk_analysis",
                reflection="continue with scoped analysis",
            )

    capture = Capture()
    state = {
        **_state(),
        "allowed_doc_ids": ["doc-a", "doc-b"],
        "status": WorkflowStatus.PLAN_APPROVED,
        "current_step": 2,
        "plan": [],
        "plan_approved": True,
        "ingestion_state": [
            IngestionDocumentState(
                document_id="doc-a",
                status=IngestionDocumentStatus.READY,
            ),
            IngestionDocumentState(
                document_id="doc-b",
                status=IngestionDocumentStatus.WARNINGS,
                warning_count=1,
            ),
        ],
        "risk_analysis": None,
        "compliance_result": None,
        "grounding": None,
        "human_review": None,
        "errors": [],
    }

    result = await make_orchestrator_node(capture)(state)  # type: ignore[arg-type]
    context = json.loads(capture.inputs[0][1].content)
    documents = context["capabilities"]["documents"]

    assert [item["document_id"] for item in documents] == ["doc-a", "doc-b"]
    assert [item["status"] for item in documents] == ["ready", "warnings"]
    assert context["capabilities"]["delegation"]["allowed_targets"]
    assert context["capabilities"]["delegation"]["remaining_model_calls"] >= 0
    assert result["orchestrator_action"].target_node == "risk_analysis"


async def test_nodes_without_transfers_append_nothing() -> None:
    node = make_risk_analysis_node(_FakeAgent([HumanMessage(content="plain")]))

    result = await node(_state())  # type: ignore[arg-type]

    assert result["messages"] == []
