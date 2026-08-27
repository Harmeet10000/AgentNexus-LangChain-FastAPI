"""Band: agent-tools-unification 9.5 — runtime resilience on a throwaway graph (D-10).

A minimal two-node StateGraph (node_a -> node_b) built inside this file with an
InMemorySaver checkpointer proves the six runtime scenarios from
agent-runtime-resilience without reaching the application graph.

The designated seam is `_make_tool_seam` below: it is the ONLY place in this
module where retry behaviour exists. It wraps tool invocation through langchain's
ToolRetryMiddleware and shields human-in-the-loop pauses — a GraphBubbleUp must
never enter the retry loop, because the middleware would both count it as an
attempt and then convert it into an error ToolMessage (its broad
`except Exception` plus `on_failure="continue"` path), which D-10 forbids.
"""

from __future__ import annotations

import ast
import inspect
import json
import operator
import re
from typing import (  # noqa: TC003 -- StateGraph resolves state-schema annotations via get_type_hints at compile
    Annotated,
    Any,
    TypedDict,
)

from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import (  # noqa: TC002 -- same runtime-introspection constraint as above
    AnyMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool  # noqa: TC002
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.shared.langchain_layer.agents.tools.idempotency import ToolResult


class TransientBackendError(Exception):
    """A transient fault that a retry can plausibly fix."""


class PermanentConfigError(Exception):
    """A fault that cannot succeed on retry."""


class ThrowawayState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    trail: Annotated[list[str], operator.add]


def _make_tool_seam(*, max_retries: int, retry_on: Any = (Exception,)) -> Any:
    """Build the single retry seam used by every scenario."""
    middleware = ToolRetryMiddleware(
        max_retries=max_retries,
        retry_on=retry_on,
        backoff_factor=0.0,
        initial_delay=0.0,
        jitter=False,
    )

    async def invoke_tool(
        tool_fn: BaseTool,
        *,
        tool_call_id: str = "call-1",
        args: dict[str, Any] | None = None,
    ) -> ToolMessage:
        pauses: list[GraphBubbleUp] = []
        request = ToolCallRequest(
            tool_call={"name": tool_fn.name, "args": args or {}, "id": tool_call_id},
            tool=tool_fn,
            state={},
            runtime=None,
        )

        async def handler(req: ToolCallRequest) -> ToolMessage:
            try:
                raw: Any = await req.tool.ainvoke(req.tool_call["args"])
            except GraphBubbleUp as exc:
                pauses.append(exc)
                return ToolMessage(
                    content="__pause__",
                    tool_call_id=req.tool_call["id"],
                    name=req.tool_call["name"],
                )
            content = json.dumps(raw) if isinstance(raw, dict) else str(raw)
            return ToolMessage(
                content=content,
                tool_call_id=req.tool_call["id"],
                name=req.tool_call["name"],
            )

        result = await middleware.awrap_tool_call(request, handler)
        if pauses:
            raise pauses[0]
        return result

    return invoke_tool


def _make_nodes(seam: Any, tool_fn: BaseTool) -> tuple[Any, Any]:
    """node_a invokes a tool through the seam; node_b proves the run went on."""

    async def node_a(state: dict[str, Any]) -> dict[str, Any]:
        message = await seam(tool_fn, args={"query": "ping"})
        return {"messages": [message]}

    async def node_b(state: dict[str, Any]) -> dict[str, Any]:
        return {"trail": ["node_b_ran"]}

    return node_a, node_b


def _build_graph(seam: Any, tool_fn: BaseTool) -> Any:
    node_a, node_b = _make_nodes(seam, tool_fn)
    builder = StateGraph(ThrowawayState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder.compile(checkpointer=InMemorySaver())


def _config(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


async def test_transient_failure_is_retried_to_the_bound_then_surfaced() -> None:
    executions: list[int] = []

    @tool
    def flaky_statute_lookup(query: str) -> str:
        """Look up a statute."""
        executions.append(1)
        msg = "backend timed out"
        raise TransientBackendError(msg)

    bound = 2
    seam = _make_tool_seam(max_retries=bound)
    graph = _build_graph(seam, flaky_statute_lookup)

    result = await graph.ainvoke({"messages": [], "trail": []}, _config("transient"))

    assert len(executions) == bound + 1, "initial attempt plus exactly the bounded retries"
    final = result["messages"][-1]
    assert isinstance(final, ToolMessage)
    assert final.status == "error"
    assert "failed after 3 attempts" in final.content
    assert "TransientBackendError" in final.content


async def test_permanent_failure_is_not_retried() -> None:
    executions: list[int] = []

    @tool
    def broken_config(query: str) -> str:
        """Run a permanently broken call."""
        executions.append(1)
        msg = "misconfigured index"
        raise PermanentConfigError(msg)

    seam = _make_tool_seam(
        max_retries=5,
        retry_on=lambda exc: not isinstance(exc, PermanentConfigError),
    )
    graph = _build_graph(seam, broken_config)

    result = await graph.ainvoke({"messages": [], "trail": []}, _config("permanent"))

    assert len(executions) == 1, "a permanent failure must not be retried"
    final = result["messages"][-1]
    assert isinstance(final, ToolMessage)
    assert final.status == "error"
    assert "PermanentConfigError" in final.content


async def test_raising_tool_does_not_terminate_the_run() -> None:
    @tool
    def always_raises(query: str) -> str:
        """Always raise."""
        msg = "boom"
        raise TransientBackendError(msg)

    seam = _make_tool_seam(max_retries=1)
    graph = _build_graph(seam, always_raises)

    result = await graph.ainvoke({"messages": [], "trail": []}, _config("raising"))

    assert result["trail"] == ["node_b_ran"], "run continued past the raising tool"
    final = result["messages"][-1]
    assert isinstance(final, ToolMessage)
    assert final.status == "error"


async def test_backend_unavailability_reaches_the_model_as_unavailability() -> None:
    @tool
    def precedent_search(query: str) -> dict[str, Any]:
        """Search precedents."""
        envelope = ToolResult.unavailable_result(
            reason="vector store unreachable: no precedent layer available: pgvector",
            layer="pgvector",
        )
        return envelope.model_dump()

    seam = _make_tool_seam(max_retries=2)
    graph = _build_graph(seam, precedent_search)

    result = await graph.ainvoke({"messages": [], "trail": []}, _config("unavailable"))

    final = result["messages"][-1]
    assert isinstance(final, ToolMessage)
    delivered = json.loads(final.content)
    assert delivered["unavailable"] is True
    assert delivered["success"] is False
    assert "pgvector" in delivered["error"]
    assert result["trail"] == ["node_b_ran"]


async def test_hitl_pause_propagates_without_retry_or_counting() -> None:
    executions: list[int] = []

    @tool
    def plan_approval(query: str) -> str:
        """Ask the human to approve."""
        executions.append(1)
        answer = interrupt({"question": "approve the plan?"})
        return f"approved:{answer}"

    seam = _make_tool_seam(max_retries=3)
    graph = _build_graph(seam, plan_approval)
    config = _config("hitl")

    paused = await graph.ainvoke({"messages": [], "trail": []}, config)

    assert "__interrupt__" in paused, "the run paused instead of failing or continuing"
    assert paused["__interrupt__"][0].value == {"question": "approve the plan?"}
    assert len(executions) == 1, "the pause was neither retried nor counted as an attempt"

    resumed = await graph.ainvoke(Command(resume="yes"), config)

    assert resumed["trail"] == ["node_b_ran"], "resume completes the run"
    final = resumed["messages"][-1]
    assert isinstance(final, ToolMessage)
    assert final.content == "approved:yes"


async def test_pause_shield_survives_transient_noise_before_it() -> None:
    """A pause raised after prior retries still escapes unconverted."""
    calls: list[int] = []

    @tool
    def flaky_then_pausing(query: str) -> str:
        """Fail once, then pause."""
        calls.append(1)
        if len(calls) < 2:
            msg = "first attempt fails"
            raise TransientBackendError(msg)
        interrupt({"question": "still there?"})
        return "unreachable"

    seam = _make_tool_seam(max_retries=3)
    graph = _build_graph(seam, flaky_then_pausing)

    paused = await graph.ainvoke({"messages": [], "trail": []}, _config("mixed"))

    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value == {"question": "still there?"}
    assert len(calls) == 2, "one real retry for the transient fault, none for the pause"


def test_retry_behaviour_lives_at_the_seam_not_in_node_bodies() -> None:
    seam = _make_tool_seam(max_retries=1)

    @tool
    def anything(query: str) -> str:
        """Do nothing."""
        return query

    node_a, node_b = _make_nodes(seam, anything)

    forbidden = re.compile(r"\bfor\b|\bwhile\b|range\(|max_retries|ToolRetryMiddleware")
    for name, fn in (("node_a", node_a), ("node_b", node_b)):
        source = inspect.getsource(fn)
        assert not forbidden.search(source), (
            f"{name} carries retry behaviour; retries belong at the seam"
        )

    module = inspect.getmodule(_make_tool_seam)
    tree = ast.parse(inspect.getsource(module))
    seam_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_tool_seam"
    )

    def _is_middleware_construction(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Name) and func.id == "ToolRetryMiddleware") or (
            isinstance(func, ast.Attribute) and func.attr == "ToolRetryMiddleware"
        )

    constructions = [
        node.lineno for node in ast.walk(tree) if _is_middleware_construction(node)
    ]
    assert constructions, "the seam must construct ToolRetryMiddleware"
    assert all(
        seam_def.lineno <= line <= seam_def.end_lineno for line in constructions
    ), "retry construction must exist only inside the seam factory"
