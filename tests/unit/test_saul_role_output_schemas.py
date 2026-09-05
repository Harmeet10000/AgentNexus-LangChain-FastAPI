"""Verification: agent_saul role agents declare their structured output schemas.

The risk and compliance `create_agent` calls bind tools and the retry
middleware; without `response_format` their outputs are free-form prose the
citation validators downstream cannot hold to account. This pins the
declaration at the construction boundary — the provider-backed agent assembly
itself is stubbed (the D13 finding: provider packages are absent here), and
only the kwargs the factory passes are asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.langchain_layer.agents.tools import register_default_tools
from app.shared.langgraph_layer.agent_saul.state import (
    ComplianceOutput,
    RiskAnalysisOutput,
)

if TYPE_CHECKING:
    from typing import Any


class _StubLLM:
    """Minimal stand-in: the factory only calls `with_structured_output` on it."""

    def with_structured_output(self, schema: Any) -> Any:
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda value: (schema, value))


def _build_with_captured_create_agent(monkeypatch: Any) -> dict[str, Any]:
    from app.shared.langgraph_layer.agent_saul import factory as saul_factory

    captured: dict[str, Any] = {}

    def _fake_create_agent(*_args: Any, **kwargs: Any) -> Any:
        captured[kwargs["system_prompt"][:24]] = kwargs
        return object()

    monkeypatch.setattr(saul_factory, "create_agent", _fake_create_agent)
    register_default_tools()
    saul_factory.build_agent_registry(_StubLLM(), _StubLLM())  # type: ignore[arg-type]
    return captured


def test_risk_agent_declares_its_output_schema(monkeypatch: Any) -> None:
    from app.shared.langgraph_layer.agent_saul.prompts import (
        _RISK_ANALYSIS_SYSTEM_PROMPT,
    )

    captured = _build_with_captured_create_agent(monkeypatch)
    risk_kwargs = captured[_RISK_ANALYSIS_SYSTEM_PROMPT[:24]]
    assert risk_kwargs["response_format"] is RiskAnalysisOutput, (
        "the risk agent must declare RiskAnalysisOutput as its response_format"
    )


def test_compliance_agent_declares_its_output_schema(monkeypatch: Any) -> None:
    from app.shared.langgraph_layer.agent_saul.prompts import (
        _COMPLIANCE_SYSTEM_PROMPT,
    )

    captured = _build_with_captured_create_agent(monkeypatch)
    compliance_kwargs = captured[_COMPLIANCE_SYSTEM_PROMPT[:24]]
    assert compliance_kwargs["response_format"] is ComplianceOutput, (
        "the compliance agent must declare ComplianceOutput as its response_format"
    )
