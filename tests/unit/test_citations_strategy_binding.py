"""Band: agent-tools-unification groups 8–9 — citations, output strategy, binding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from typing import Any

# --- 8.1 citations non-empty ---


def test_citation_fields_pre_exist_and_are_not_redefined() -> None:
    from app.shared.langgraph_layer.agent_saul.state import Citation

    fields = set(Citation.model_fields)
    assert {"claim", "source", "confidence"} <= fields, sorted(fields)


def test_a_report_with_no_citations_fails_validation() -> None:
    from app.shared.langgraph_layer.agent_saul.state import (
        Citation,
        FinalReport,
    )

    kwargs: dict[str, Any] = {
        "document_id": "doc-1",
        "summary": "s",
        "risk_findings": [],
        "compliance_findings": [],
        "human_overrides": [],
        "suggested_actions": [],
    }
    # A populated citation list validates…
    FinalReport(**kwargs, citations=[Citation(claim="c", source="s", confidence=0.9)])
    # …and an empty one is a hard error, not a silent pass-through.
    with pytest.raises(ValidationError, match="citation"):
        FinalReport(**kwargs, citations=[])


# --- 8.2 declared output strategy ---


async def test_the_structured_output_strategy_is_explicit_and_pinned() -> None:
    """Assert WHICH strategy the seam selects — recorded in the failure message.

    Q3: the configured gemini-3.1 models are absent from provider profile tables,
    so native/`AutoStrategy` paths silently degrade to tool-calling. Our seam
    pins `method="function_calling"` explicitly; this test fails loudly if that
    pin moves, so a future model change surfaces as a diff rather than an
    invisible upgrade.
    """
    from app.shared.langchain_layer import models as lc_models

    captured: dict[str, Any] = {}

    class _FakeLLM:
        def with_structured_output(self, schema: Any, *, method: str = "<default>") -> str:
            captured["schema"] = schema
            captured["method"] = method
            return "bound"

    class _Answer:
        pass

    result = await lc_models.awith_structured_output(_Answer, model=_FakeLLM())
    assert result == "bound"
    assert captured["method"] == "function_calling", (
        f"structured-output method drifted: {captured['method']!r} "
        "(was 'function_calling' — check Q3 before accepting)"
    )
    assert captured["schema"] is _Answer


# --- 9.2 handoff tools ---


def test_handoff_tools_are_registered_and_tagged() -> None:
    from app.shared.langchain_layer.agents.tools import register_default_tools

    registry = register_default_tools()
    names = {t.name for t in registry.by_tags("handoff")}
    assert any(n.startswith("transfer_to_") for n in names), sorted(names)


async def test_a_transfer_tool_returns_a_routing_payload() -> None:
    from app.shared.langchain_layer.agents.tools import get_tool_registry, register_default_tools

    register_default_tools()
    tool = get_tool_registry().get("transfer_to_risk")
    payload = await tool.ainvoke({"reason": "needs risk analysis"})
    assert payload["transfer_to"] == "risk"
    assert payload["reason"] == "needs risk analysis"


# --- 9.3 saul agents carry tools and middleware ---


def test_saul_agents_have_zero_empty_tool_lists() -> None:
    from pathlib import Path

    source = Path("src/app/shared/langgraph_layer/agent_saul/factory.py").read_text(
        encoding="utf-8"
    )
    assert "tools=[]" not in source, "agents must bind their tools explicitly"


def test_saul_factory_installs_the_retry_middleware() -> None:
    from pathlib import Path

    source = Path("src/app/shared/langgraph_layer/agent_saul/factory.py").read_text(
        encoding="utf-8"
    )
    assert "ToolRetryMiddleware" in source
    assert "handle_tool_errors" not in source
