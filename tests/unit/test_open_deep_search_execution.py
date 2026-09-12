"""Concurrency and failure tests for immediate deep research."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from returns.result import Failure, Success

from app.shared.langgraph_layer.open_deep_search import graph as open_deep_search_graph
from app.shared.langgraph_layer.open_deep_search import utils as open_deep_search_utils
from app.shared.langgraph_layer.open_deep_search.execution import (
    ResearchExecutionGate,
    gather_limited,
)
from app.shared.langgraph_layer.open_deep_search.graph import (
    execute_tool_safely,
    researcher_tools,
    route_researcher,
)
from app.shared.services.errors import TavilyExternalError
from app.shared.services.tavily import SearchResponse
from app.utils import ExternalServiceException


@pytest.mark.asyncio
async def test_gather_limited_respects_per_invocation_limit() -> None:
    active = 0
    peak = 0

    async def operation(value: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return value

    values = await gather_limited(
        (lambda value=value: operation(value) for value in range(8)),
        limit=2,
    )

    assert values == list(range(8))
    assert peak == 2


@pytest.mark.asyncio
async def test_gather_limited_cancels_siblings_after_failure() -> None:
    sibling_cancelled = False

    async def failing_operation() -> None:
        await asyncio.sleep(0)
        msg = "provider failed"
        raise RuntimeError(msg)

    async def long_operation() -> None:
        nonlocal sibling_cancelled
        try:
            await asyncio.sleep(60)
        finally:
            sibling_cancelled = True

    with pytest.raises(RuntimeError, match="provider failed"):
        await gather_limited(
            (failing_operation, long_operation),
            limit=2,
        )

    assert sibling_cancelled is True


@pytest.mark.asyncio
async def test_tavily_queries_are_deduplicated_and_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_search(query: str, **kwargs: object):
        _ = kwargs
        calls.append(query)
        return Success(SearchResponse(query=query, results=[], answer=None, total_results=0))

    monkeypatch.setattr(open_deep_search_utils, "search", fake_search)
    config = {
        "configurable": {
            "max_search_queries": 2,
            "max_concurrent_search_requests": 1,
            "research_execution_gate": ResearchExecutionGate(
                max_concurrent_search_requests=1,
                max_concurrent_summaries=1,
            ),
        }
    }

    responses = await open_deep_search_utils.tavily_search_async(
        [" first ", "", "second", "first", "third"],
        config=config,
    )

    assert calls == ["first", "second"]
    assert len(responses) == 2


@pytest.mark.asyncio
async def test_tavily_search_preserves_successes_when_one_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search(query: str, **kwargs: object):
        _ = kwargs
        if query == "bad":
            return Failure(TavilyExternalError(message="upstream failure"))
        return Success(SearchResponse(query=query, results=[], answer=None, total_results=0))

    monkeypatch.setattr(open_deep_search_utils, "search", fake_search)
    responses = await open_deep_search_utils.tavily_search_async(
        ["bad", "good"],
        config={"configurable": {"max_concurrent_search_requests": 2}},
    )

    assert [response.query for response in responses] == ["good"]


@pytest.mark.asyncio
async def test_execute_tool_safely_turns_provider_failure_into_observation() -> None:
    class FailingTool:
        name = "tavily_search"

        async def ainvoke(self, args: dict[str, object], config: object) -> str:
            _ = (args, config)
            raise ExternalServiceException(service="Tavily", detail="temporarily unavailable")

    observation = await execute_tool_safely(FailingTool(), {}, {})

    assert observation.startswith("Error executing tool:")
    assert "temporarily unavailable" in observation


@pytest.mark.asyncio
async def test_researcher_tools_caps_recognized_calls_per_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []

    class FakeTool:
        name = "tavily_search"

        async def ainvoke(self, args: dict[str, object], config: object) -> str:
            _ = config
            executed.append(str(args["query"]))
            return f"observed {args['query']}"

    async def fake_get_all_tools(config: object) -> list[FakeTool]:
        _ = config
        return [FakeTool()]

    monkeypatch.setattr(open_deep_search_graph, "get_all_tools", fake_get_all_tools)
    state = {
        "researcher_messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tavily_search", "args": {"query": "one"}, "id": "call-1"},
                    {"name": "tavily_search", "args": {"query": "two"}, "id": "call-2"},
                ],
            )
        ]
    }

    command = await researcher_tools(
        state,
        {
            "configurable": {
                "max_tool_calls_per_turn": 1,
                "max_concurrent_research_tools": 1,
            }
        },
    )

    outputs = command.update["researcher_messages"]
    assert executed == ["one"]
    assert [message.content for message in outputs] == [
        "observed one",
        "Error: maximum tool calls per turn exceeded. Retry with 1 or fewer tool calls.",
    ]


@pytest.mark.asyncio
async def test_supervisor_tools_cancels_researcher_siblings_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling_cancelled = False

    class FakeResearcherSubgraph:
        async def ainvoke(self, state: dict[str, object], config: object) -> dict[str, object]:
            nonlocal sibling_cancelled
            _ = config
            if state["research_topic"] == "bad":
                await asyncio.sleep(0)
                raise RuntimeError("subgraph failed")
            try:
                await asyncio.sleep(60)
            finally:
                sibling_cancelled = True
            return {}

    monkeypatch.setattr(open_deep_search_graph, "researcher_subgraph", FakeResearcherSubgraph())
    state = {
        "supervisor_messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ConductResearch",
                        "args": {"research_topic": "bad"},
                        "id": "call-1",
                    },
                    {
                        "name": "ConductResearch",
                        "args": {"research_topic": "long"},
                        "id": "call-2",
                    },
                ],
            )
        ],
        "research_iterations": 1,
    }

    command = await open_deep_search_graph.supervisor_tools(
        state,
        {"configurable": {"max_concurrent_research_units": 2}},
    )

    assert command.goto == "__end__"
    assert sibling_cancelled is True


def test_unknown_crawl_call_is_handled_by_immediate_tool_executor() -> None:
    state = {
        "researcher_messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "crawl_webpage", "args": {}, "id": "call-1"}],
            )
        ]
    }

    assert route_researcher(state) == "researcher_tools"
