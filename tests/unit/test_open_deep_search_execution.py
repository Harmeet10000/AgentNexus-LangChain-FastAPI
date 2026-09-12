"""Concurrency and failure tests for immediate deep research."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from returns.result import Failure, Success

from app.shared.langgraph_layer.open_deep_search import utils as open_deep_search_utils
from app.shared.langgraph_layer.open_deep_search.execution import (
    ResearchExecutionGate,
    gather_limited,
)
from app.shared.langgraph_layer.open_deep_search.graph import (
    execute_tool_safely,
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
