"""Verification: Agent Saul runs carry an explicit step budget (D-10, D-5).

Proven against the D-10 authorised vehicle — a throwaway two-node `StateGraph`
built inside this file with an `InMemorySaver` — never against the application
graph. This module imports only the budget constant and the invoke-config
helper from `agent_saul.graph`; it must not reach the application graph
builder, and no proof here may require a provisioned checkpointer,
a live database, or a mounted route.
"""

from __future__ import annotations

import operator
from typing import (  # noqa: TC003 -- StateGraph resolves state-schema annotations via get_type_hints at compile
    Annotated,
    TypedDict,
)

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from app.shared.langgraph_layer.agent_saul.graph import (
    SAUL_STEP_BUDGET,
    saul_runtime_config,
)


class _LoopState(TypedDict):
    steps: Annotated[list[int], operator.add]


def _build_loop_graph() -> object:
    """A graph that never terminates on its own: the budget is its only exit."""

    async def spin(state: _LoopState) -> dict[str, list[int]]:
        return {"steps": [1]}

    def _keep_spinning(_state: _LoopState) -> str:
        return "spin"

    builder = StateGraph(_LoopState)
    builder.add_node("spin", spin)
    builder.add_edge(START, "spin")
    builder.add_conditional_edges("spin", _keep_spinning, {"spin": "spin", END: END})
    return builder.compile(checkpointer=InMemorySaver())


def test_the_step_budget_is_pinned_explicitly() -> None:
    assert isinstance(SAUL_STEP_BUDGET, int)
    assert SAUL_STEP_BUDGET > 0


def test_the_runtime_config_carries_the_budget() -> None:
    config = saul_runtime_config("thread-budget")
    assert config["recursion_limit"] == SAUL_STEP_BUDGET
    assert config["configurable"] == {"thread_id": "thread-budget"}


async def test_a_run_exceeding_its_step_budget_terminates_explicitly() -> None:
    graph = _build_loop_graph()
    config = saul_runtime_config("thread-over-budget")
    with pytest.raises(GraphRecursionError):
        await graph.ainvoke({"steps": []}, config)


async def test_a_run_within_budget_completes() -> None:
    async def once(_state: _LoopState) -> dict[str, list[int]]:
        return {"steps": [1]}

    builder = StateGraph(_LoopState)
    builder.add_node("once", once)
    builder.add_edge(START, "once")
    builder.add_edge("once", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    result = await graph.ainvoke({"steps": []}, saul_runtime_config("thread-in-budget"))
    assert result["steps"] == [1]
