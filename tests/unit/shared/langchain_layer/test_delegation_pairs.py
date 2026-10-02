"""Delegation pair construction and pair-aware trimming (agent-message-standard 1.1, 1.2)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.shared.langchain_layer.messages import (
    delete_tool_messages,
    filter_tool_messages,
    is_delegation_answer,
    is_delegation_request,
    make_delegation_pair,
)


def test_pair_links_request_to_answer_by_call_id() -> None:
    request, answer = make_delegation_pair("risk", "clause needs risk review")

    assert isinstance(request, AIMessage)
    assert isinstance(answer, ToolMessage)
    assert len(request.tool_calls) == 1
    assert request.tool_calls[0]["name"] == "transfer_to_risk"
    assert answer.tool_call_id == request.tool_calls[0]["id"]
    assert answer.name == "transfer_to_risk"


def test_pair_reason_has_a_single_source() -> None:
    request, answer = make_delegation_pair("risk", "same reason")

    assert request.content == "same reason"
    assert "same reason" in str(answer.content)


def test_pair_survives_model_dump_round_trip() -> None:
    request, answer = make_delegation_pair("compliance", "statute check")

    assert request.model_dump()["tool_calls"][0]["id"] == answer.model_dump()["tool_call_id"]


def test_pair_ids_are_fresh_per_invocation() -> None:
    first, _ = make_delegation_pair("risk", "r")
    second, _ = make_delegation_pair("risk", "r")

    assert first.tool_calls[0]["id"] != second.tool_calls[0]["id"]


def test_predicates_recognize_only_pairs() -> None:
    request, answer = make_delegation_pair("risk", "r")
    ordinary_ai = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {}, "id": "x", "type": "tool_call"}],
    )

    assert is_delegation_request(request)
    assert not is_delegation_request(answer)
    assert not is_delegation_request(ordinary_ai)
    assert is_delegation_answer(answer)
    assert not is_delegation_answer(request)


def test_delete_removes_pairs_atomically() -> None:
    request, answer = make_delegation_pair("risk", "r")
    history = [HumanMessage(content="hi"), request, answer]

    assert delete_tool_messages(history) == [history[0]]


def test_filter_names_delegations_in_summary() -> None:
    request, answer = make_delegation_pair("risk", "check limits")

    filtered = filter_tool_messages([request, answer])

    assert len(filtered) == 1
    summary = filtered[0]
    assert isinstance(summary, HumanMessage)
    assert "Delegation to risk" in summary.content
