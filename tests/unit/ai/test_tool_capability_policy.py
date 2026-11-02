"""The tool-capability catalog: structured authority, policy, selector scoping.

Covers openspec ``tool-capability-catalog`` tasks 1.1 (metadata authoritative,
tags are views), 2.1 (policy loading rules), 3.1 (selector shrinks and never
widens), and 3.3 (web-search egress gate). The projection-consistency walk
over the factory's role bindings lives in
``tests/unit/features/agent_saul/graph/test_policy_consistency.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse, Runtime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from returns.result import Success

from app.ai import guardrails
from app.ai.tools import web_search as web_search_module
from app.ai.tools.policy import ROLE_CAPABILITY_GRANTS, grants_for, validate_policy
from app.ai.tools.registry import (
    DEEP_RESEARCH_TOOL_NAME,
    PORTFOLIO_TRAVERSAL_TOOL_NAMES,
    get_tool_registry,
    register_deep_research_tool,
    register_default_tools,
    register_portfolio_traversal_tools,
)
from app.ai.tools.web_search import CAP_WEB_SEARCH, WebSearchInput, web_search
from app.features.documents.application.tools.portfolio import make_portfolio_traversal_tools
from app.resilience.idempotency import ToolResult

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any


@pytest.fixture(autouse=True)
def _populated_registry() -> Iterator[None]:
    """Register the default tools plus the portfolio catalog seats, then leave
    the singleton registry as found.

    The registry never imports ``app.features`` (shared-runtime lint
    contract), so the backend-less portfolio seats — exactly as the graph
    factory does — are registered here from the test side via the intake
    function with validation deferred to the stub deep-research registration.
    """
    register_default_tools()
    register_portfolio_traversal_tools(make_portfolio_traversal_tools(None), validate=False)
    yield
    r = get_tool_registry()
    for attr in ("_tools", "_tags", "_authority"):
        for name in (*PORTFOLIO_TRAVERSAL_TOOL_NAMES, DEEP_RESEARCH_TOOL_NAME):
            getattr(r, attr).pop(name, None)


def _register_stub_deep_research() -> object:
    """Register a stand-in deep-research tool (the D5 wrapper's catalog slot)."""

    @tool
    async def deep_research(objective: str, focus: str | None = None) -> dict[str, str]:
        """Stub deep-research wrapper for catalog tests."""

        return {}

    return register_deep_research_tool(deep_research)


# ---------------------------------------------------------------------------
# Task 1.1 — structured authority metadata is authoritative; tags are views
# ---------------------------------------------------------------------------


def test_registrations_carry_structured_authority_matching_tags() -> None:
    """Every default registration's metadata agrees with its derived tags."""
    registry = get_tool_registry()

    expected: dict[str, tuple[str | None, str | None, bool]] = {
        "web_search": (CAP_WEB_SEARCH, "egress", True),
        "shell_tool": ("system.shell.execute", "write", False),
        "read_file": ("filesystem.read", "read", False),
        "write_file": ("filesystem.write", "write", False),
        "list_directory": ("filesystem.read", "read", False),
        "file_search": ("filesystem.read", "read", False),
        # Portfolio traversal (openspec agent-portfolio-traversal-tools):
        # read-only, non-egress graph traversal.
        "cross_document_obligations": ("portfolio.traversal", "read", False),
        "find_conflicting_clauses": ("portfolio.traversal", "read", False),
        "party_obligation_map": ("portfolio.traversal", "read", False),
    }
    for name, (capability, side_effect, egress) in expected.items():
        authority = registry.authority(name)
        assert (authority.capability, authority.side_effect, authority.egress) == (
            capability,
            side_effect,
            egress,
        ), name
        # Tags are derived views over the metadata: the capability tag exists
        # and by_tags still selects through it (AND semantics).
        if capability is not None:
            assert registry.by_tags(f"capability:{capability}", f"side-effect:{side_effect}")

    # Handoff tools declare no authority at all.
    for handoff in registry.by_tags("handoff"):
        authority = registry.authority(handoff.name)
        assert authority.capability is None
        assert authority.side_effect is None
        assert authority.egress is False

    assert registry.declared_capabilities() == {
        CAP_WEB_SEARCH,
        "system.shell.execute",
        "filesystem.read",
        "filesystem.write",
        "portfolio.traversal",
    }


def test_authority_tag_disagreement_is_a_registration_error() -> None:
    """Tag/metadata drift is a registration error, not a silent inconsistency."""

    @tool
    async def _dummy(value: str) -> str:
        """Stub for registration-error tests."""

        return value

    registry = get_tool_registry()
    with pytest.raises(ValueError, match="disagree"):
        registry.register(_dummy, tags=["capability:filesystem.read"], capability="web.search")
    with pytest.raises(ValueError, match="disagree"):
        registry.register(_dummy, tags=["capability:system.shell.execute"])
    with pytest.raises(ValueError, match="disagree"):
        registry.register(_dummy, tags=[], capability="web.search")
    with pytest.raises(ValueError, match="egress"):
        registry.register(_dummy, tags=["side-effect:read"], egress=True)
    with pytest.raises(ValueError, match="disagree"):
        registry.register(
            _dummy,
            tags=["capability:web.search", "side-effect:read"],
            capability="web.search",
            side_effect="egress",
            egress=True,
        )
    # The failed registrations mutated nothing.
    assert "_dummy" not in registry.names()


# ---------------------------------------------------------------------------
# Task 2.1 — policy loading rules
# ---------------------------------------------------------------------------


def test_policy_unknown_role_and_unbound_context_grant_nothing() -> None:
    """Deny-by-default at the policy layer: unknown role → empty grant set."""
    assert grants_for("no-such-role") == frozenset()
    assert grants_for(None) == ROLE_CAPABILITY_GRANTS["unbound"] == frozenset()


def test_initial_policy_encodes_current_effective_grants() -> None:
    """No role holds a system/filesystem capability; web.search is ungranted.

    The deep-research role reaches its own catalog entry and the risk/
    compliance specialists share the read-only portfolio traversal
    capability (PLAN-agent-tools.md § D4 effective sets).
    """
    for role, grants in ROLE_CAPABILITY_GRANTS.items():
        assert "system.shell.execute" not in grants, role
        assert "filesystem.read" not in grants, role
        assert "filesystem.write" not in grants, role
        assert "web.search" not in grants, role
    assert ROLE_CAPABILITY_GRANTS["deep_research"] == frozenset({"deep-research"})
    # The specialists' only non-operation capability is the read-only,
    # non-egress portfolio traversal (openspec agent-portfolio-traversal-tools
    # task 3.2).
    assert ROLE_CAPABILITY_GRANTS["risk"] == frozenset({"portfolio.traversal"})
    assert ROLE_CAPABILITY_GRANTS["compliance"] == frozenset({"portfolio.traversal"})


def test_policy_unknown_capability_identifier_fails_startup() -> None:
    """A grant naming a capability no registered tool declares is a config bug."""
    with pytest.raises(ValueError, match="deep-research"):
        validate_policy(get_tool_registry())

    _register_stub_deep_research()
    validate_policy(get_tool_registry())  # full catalog: no error


def test_policy_typo_fails_startup() -> None:
    """A typo'd capability identifier cannot silently grant nothing."""
    import app.ai.tools.policy as policy_module

    original = policy_module.ROLE_CAPABILITY_GRANTS
    _register_stub_deep_research()
    policy_module.ROLE_CAPABILITY_GRANTS = {
        **original,
        "risk": frozenset({"web.serch"}),  # deliberate typo
    }
    try:
        with pytest.raises(ValueError, match=r"web\.serch"):
            validate_policy(get_tool_registry())
    finally:
        policy_module.ROLE_CAPABILITY_GRANTS = original


# ---------------------------------------------------------------------------
# Task 3.1 — the selector shrinks and never widens
# ---------------------------------------------------------------------------


def _scope_request(tools: list[Any], context: Any) -> ModelRequest[Any]:
    messages = [HumanMessage(content="research the indemnification clauses")]
    return ModelRequest(
        model=object(),
        messages=messages,
        system_message=None,
        tools=tools,
        state={"messages": messages},
        runtime=Runtime(context=context),
    )


async def _scoped_tools(request: ModelRequest[Any]) -> list[Any]:
    middleware = guardrails.PolicyToolScopeMiddleware().build()
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    await middleware.awrap_model_call(request, handler)
    return handler.await_args.args[0].tools


async def test_high_relevance_unauthorized_tool_is_absent_from_projection() -> None:
    """Relevance cannot resurrect a tool the policy did not authorize.

    The scoping runs before the selector ranks anything, so an unauthorized
    tool is absent at any relevance score — the selector only ever sees the
    intersection.
    """
    registry = get_tool_registry()
    tools = [registry.get("shell_tool"), registry.get("web_search")]
    tools += registry.by_tags("handoff")[:1]

    # web.search is granted; system.shell.execute is not — even if the
    # selection model would rank the shell tool first, it is already gone.
    scoped = await _scoped_tools(_scope_request(tools, {"capabilities": ("web.search",)}))

    assert [t.name for t in scoped] == ["web_search", "transfer_to_orchestrator"]


async def test_scoping_denies_by_default_without_grants() -> None:
    """No capability tuple in the context → capability-bearing tools vanish."""
    registry = get_tool_registry()
    tools = [registry.get("web_search"), registry.get("read_file")]
    tools += registry.by_tags("handoff")[:1]

    scoped = await _scoped_tools(_scope_request(tools, {"user_id": "u"}))
    assert [t.name for t in scoped] == ["transfer_to_orchestrator"]

    scoped = await _scoped_tools(_scope_request(tools, None))
    assert [t.name for t in scoped] == ["transfer_to_orchestrator"]


async def test_scoping_preserves_provider_tool_dicts() -> None:
    """Provider-specific tool dicts carry no registry identity; pass through."""
    registry = get_tool_registry()
    provider_dict = {"type": "function", "function": {"name": "custom"}}
    scoped = await _scoped_tools(_scope_request([registry.get("shell_tool"), provider_dict], None))
    assert scoped == [provider_dict]


def test_default_stack_scopes_tools_before_the_selector_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scoper wraps the selector: intersection precedes ranking (ADR-3)."""
    monkeypatch.setattr(guardrails, "SummarizationMiddleware", lambda **_kwargs: object())
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")  # model construction only
    stack = guardrails.build_default_middleware_stack(
        enable_guardrails=False,
        enable_tool_selector=True,
    )
    assert isinstance(stack[2], guardrails.LLMToolSelectorMiddleware)
    # The item before the selector is the policy scoper's built middleware.
    assert stack[1].name == "scope_tools"


# ---------------------------------------------------------------------------
# Task 3.3 — web search egress gate
# ---------------------------------------------------------------------------


class _FakeTavily:
    async def search(self, *, query: str, max_results: int, include_answer: bool) -> Any:
        return Success(
            SimpleNamespace(
                answer="AI answer",
                results=[SimpleNamespace(title="T1", url="https://x/1", score=0.9, content="c1")],
            )
        )


@pytest.fixture
def _tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _client() -> _FakeTavily:
        return _FakeTavily()

    monkeypatch.setattr(web_search_module, "get_tavily_client", _client)


async def test_web_search_refuses_without_grant() -> None:
    """No-grant refusal is typed, names the capability, and sends no query."""
    result = await web_search.coroutine(query="confidential clause text")

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "web.search" in (result.error or "")


async def test_web_search_refuses_wrong_grant() -> None:
    runtime = SimpleNamespace(context={"capabilities": ("filesystem.read",)})
    result = await web_search.coroutine(query="q", runtime=runtime)
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "web.search" in (result.error or "")


@pytest.mark.usefixtures("_tavily")
async def test_web_search_granted_path_unchanged() -> None:
    """With the grant, the formatted result is exactly the pre-gate shape."""
    runtime = SimpleNamespace(context={"capabilities": (CAP_WEB_SEARCH,)})
    output = await web_search.coroutine(query="current information", runtime=runtime)

    assert isinstance(output, str)
    assert "# Answer" in output
    assert "# Search Results" in output
    assert "https://x/1" in output


@pytest.mark.usefixtures("_tavily")
async def test_web_search_runs_through_a_tool_node() -> None:
    """The tool executes through a real ToolNode invocation.

    ToolNode injects the ``InjectedToolArg``-marked ``runtime`` kwarg into the
    call args before validating them against the args schema, so a schema
    with ``extra="forbid"`` rejected every ToolNode invocation
    (langchain-core 1.7.0). ``WebSearchInput`` therefore permits extras —
    this regression pins that the search actually runs end to end, with the
    egress grant flowing through the graph context into the injected runtime.
    """
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([web_search]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
                "args": {"query": "current information"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    result = await graph.ainvoke(
        {"messages": [ai_message]},
        context={"capabilities": (CAP_WEB_SEARCH,)},
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "success"
    assert tool_message.tool_call_id == "call-1"
    assert "# Search Results" in tool_message.content
    assert "https://x/1" in tool_message.content


def test_web_search_schema_stays_model_clean() -> None:
    """The runtime context is injected, never a model-visible argument."""
    registry = get_tool_registry()
    tool = registry.get("web_search")
    assert tool.args_schema is WebSearchInput
    assert set(WebSearchInput.model_fields) == {"query", "max_results"}
    # The tool's model-facing call schema exposes exactly the declared
    # arguments; the injected runtime never appears in it.
    assert set(tool.tool_call_schema.model_fields) == {"query", "max_results"}
    assert "runtime" not in tool.args


def test_web_search_registration_is_egress_classified(tmp_path: Path) -> None:
    """The egress classification lives in structured metadata, not just tags."""
    registry = get_tool_registry()
    authority = registry.authority("web_search")
    assert authority.egress is True
    assert authority.capability == CAP_WEB_SEARCH
