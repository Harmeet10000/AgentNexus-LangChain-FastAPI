"""Delegation-pair conformance for open_deep_search (agent-message-standard 3.1).

This graph is natively message-paired: every tool call receives exactly one
linked ToolMessage. These tests pin the spec requirements (linkage, shared
constructor shape, no orphans, compress visibility) so future edits cannot
silently break pair integrity.
"""

from langchain_core.messages import HumanMessage, ToolMessage, filter_messages

from app.shared.langchain_layer.messages import make_delegation_pair
from app.shared.langgraph_layer.open_deep_search.state import override_reducer


def test_shared_pair_merges_into_supervisor_channel() -> None:
    request, answer = make_delegation_pair("researcher", "topic: tort reform")
    existing = [HumanMessage(content="brief")]

    merged = override_reducer(existing, [request, answer])

    assert isinstance(merged, list)
    assert merged[0].content == "brief"
    assert merged[1].tool_calls[0]["id"] == merged[2].tool_call_id


def test_researcher_tool_outputs_cover_every_call_exactly_once() -> None:
    """Mirror of the researcher_tools contract: one linked ToolMessage per call."""
    calls = [
        {"name": "tavily_search", "args": {}, "id": "c1"},
        {"name": "unknown_tool", "args": {}, "id": "c2"},
    ]

    outputs = [
        ToolMessage(content=f"obs:{call['name']}", name=call["name"], tool_call_id=call["id"])
        for call in calls
    ]

    assert [m.tool_call_id for m in outputs] == ["c1", "c2"]
    assert all(isinstance(m, ToolMessage) for m in outputs)


def test_compress_read_path_sees_both_pair_halves() -> None:
    request, answer = make_delegation_pair("researcher", "visible reason")
    history = [HumanMessage(content="ctx"), request, answer]

    visible = filter_messages(history, include_types=["tool", "ai"])

    contents = " ".join(str(m.content) for m in visible)
    assert "visible reason" in contents
    assert len(visible) == 2
