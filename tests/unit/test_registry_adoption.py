"""Band: agent-tools-unification group 3 — registry adoption in the factory.

The factory's string branch was unreachable before explicit registration existed:
``get_tool_registry().get(name)`` raised ``KeyError`` on an empty registry. These
tests pin both halves of the contract — string specs resolve through the
registry, tool-object specs pass through untouched.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.shared.langchain_layer.agents.tools import (
    get_tool_registry,
    register_default_tools,
)


@pytest.fixture(autouse=True)
def populated_registry() -> Any:
    register_default_tools()
    yield
    # The registry is a process-wide singleton; leave it as found.
    r = get_tool_registry()
    for name in list(r.names()):
        if name not in {"web_search", "crawl_url"}:
            del r._tools[name]  # noqa: SLF001 — test cleanup of names this module added


class _FakeModel:
    """Minimal stand-in: the factory only binds tools to it."""

    def bind_tools(self, _tools: Any, **_kw: Any) -> "_FakeModel":
        return self


def _spec(tool: Any) -> Any:
    from app.shared.langchain_layer.agents.factory import AgentSpec

    return AgentSpec(
        name="test-agent",
        tools=[tool],
        # The middleware stack builds provider models; this environment has no
        # vertex package (the D13 finding). Tool resolution happens before any
        # of it and is the only thing under test here.
        enable_guardrails=False,
        enable_tool_selector=False,
    )


def test_a_string_tool_name_resolves_through_the_factory(monkeypatch: Any) -> None:
    """The factory's string branch resolves through the registry — proven at the
    ``create_agent`` boundary, because constructing LangChain's full agent stack
    needs provider packages this environment does not have (the D13 finding).
    """
    import app.shared.langchain_layer.agents.factory as factory

    captured: dict[str, Any] = {}

    def _fake_create_agent(_model: Any, *, tools: Any, **_kw: Any) -> Any:
        captured["tools"] = tools

        class _Compiled:
            def astream(self, *_a: Any, **_k: Any) -> Any:
                return iter(())

        return _Compiled()

    monkeypatch.setattr(factory, "create_agent", _fake_create_agent)
    # The middleware stack and model construction both need provider packages
    # this environment lacks (the D13 finding) — irrelevant to resolution.
    monkeypatch.setattr(factory, "build_default_middleware_stack", lambda **_kw: [])
    monkeypatch.setattr(factory, "_build_chat_model", lambda **_kw: _FakeModel())
    from app.shared.langchain_layer.agents.factory import create_production_agent

    create_production_agent(_spec("web_search"))
    (resolved_tool,) = captured["tools"]
    assert getattr(resolved_tool, "name", None) == "web_search", (
        "the factory must resolve the string to the registered tool object"
    )


def test_resolution_returns_a_tool_object_not_a_string() -> None:
    resolved = get_tool_registry().get("web_search")
    assert hasattr(resolved, "name")
    assert resolved.name == "web_search"


def test_an_unknown_string_name_fails_loudly() -> None:
    with pytest.raises(KeyError, match="no_such_tool"):
        get_tool_registry().get("no_such_tool")


def test_a_tool_object_spec_passes_through_untouched() -> None:
    direct = get_tool_registry().get("web_search")
    resolved = get_tool_registry().by_names(["web_search"])[0]
    assert resolved is direct
