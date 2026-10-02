"""Node-boundary delegation pairs (agent-message-standard 2.1)."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.shared.langgraph_layer.agent_saul.nodes import (
    delegation_pairs_from,
    make_compliance_node,
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


async def test_nodes_without_transfers_append_nothing() -> None:
    node = make_risk_analysis_node(_FakeAgent([HumanMessage(content="plain")]))

    result = await node(_state())  # type: ignore[arg-type]

    assert result["messages"] == []
