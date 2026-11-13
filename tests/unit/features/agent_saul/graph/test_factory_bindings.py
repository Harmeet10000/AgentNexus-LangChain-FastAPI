"""Factory bindings: risk's readiness parity (D3) and the D5 wrapper wiring."""

from __future__ import annotations

from typing import Any  # noqa: TC003 — used in runtime stub signatures

import pytest
from langchain_core.tools import tool

from app.ai.tools import register_default_tools
from app.ai.tools.registry import deep_research_role_projection, get_tool_registry
from app.features.agent_saul.graph import factory as saul_factory
from app.features.documents.application.evidence.service import EvidenceService
from app.features.documents.application.tools.bundle import AgentToolBundle

pytestmark = pytest.mark.unit


class _StubLLM:
    """Minimal stand-in: the factory only calls `with_structured_output` on it."""

    def with_structured_output(self, schema: Any) -> Any:
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda value: (schema, value))


def _stub_tool(name: str) -> Any:
    @tool(name)
    async def _stub() -> dict[str, Any]:
        """Stub tool stand-in for factory binding capture."""

    return _stub


def _bundle() -> AgentToolBundle:
    return AgentToolBundle(
        search_legal_precedents=_stub_tool("search_legal_precedents"),
        retrieve_statute_section=_stub_tool("retrieve_statute_section"),
        query_knowledge_graph=_stub_tool("query_knowledge_graph"),
        get_obligation_chain=_stub_tool("get_obligation_chain"),
        search_evidence=_stub_tool("search_evidence"),
        read_evidence_by_id=_stub_tool("read_evidence_by_id"),
        get_ingestion_state=_stub_tool("get_ingestion_state"),
        evidence_service=EvidenceService(repo=object()),
    )


def _build_with_captured_create_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def _fake_create_agent(*_args: Any, **kwargs: Any) -> Any:
        captured[kwargs["system_prompt"][:24]] = kwargs
        return object()

    monkeypatch.setattr(saul_factory, "create_agent", _fake_create_agent)
    register_default_tools()
    return captured


def test_risk_binding_gains_get_ingestion_state_for_readiness_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D3: risk conclusions disclose incomplete documents exactly as compliance."""
    from app.features.agent_saul.graph.prompts import (
        _COMPLIANCE_SYSTEM_PROMPT,
        _RISK_ANALYSIS_SYSTEM_PROMPT,
    )

    captured = _build_with_captured_create_agent(monkeypatch)
    saul_factory.build_agent_registry(_StubLLM(), _StubLLM(), tools=_bundle())  # type: ignore[arg-type]

    risk_tools = [t.name for t in captured[_RISK_ANALYSIS_SYSTEM_PROMPT[:24]]["tools"]]
    compliance_tools = [t.name for t in captured[_COMPLIANCE_SYSTEM_PROMPT[:24]]["tools"]]

    assert "get_ingestion_state" in risk_tools
    assert "get_ingestion_state" in compliance_tools
    # The risk prompt carries the disclosure instruction (the parity wording).
    assert "get_ingestion_state" in _RISK_ANALYSIS_SYSTEM_PROMPT
    assert "not ingestion-complete" in _RISK_ANALYSIS_SYSTEM_PROMPT


def test_deep_research_wrapper_is_built_and_registered_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D5: the compiled subgraph's wrapper is registered with read-only tags."""
    from app.features.agent_saul.graph.prompts import (
        _RISK_ANALYSIS_SYSTEM_PROMPT,
    )

    captured = _build_with_captured_create_agent(monkeypatch)
    registry = saul_factory.build_agent_registry(_StubLLM(), _StubLLM(), tools=_bundle())  # type: ignore[arg-type]

    tool = registry.deep_research_tool
    assert tool is not None
    assert tool.name == "deep_research"
    # One registration, in the single registry, discoverable by authority tag.
    assert get_tool_registry().get("deep_research") is tool
    # The role's model-visible projection is exactly transfer + deep_research.
    assert [t.name for t in deep_research_role_projection()] == [
        "transfer_to_deep_research",
        "deep_research",
    ]
    # The specialists' bound toolsets never gained the wrapper.
    specialist_tools = [t.name for t in captured[_RISK_ANALYSIS_SYSTEM_PROMPT[:24]]["tools"]]
    assert "deep_research" not in specialist_tools


def test_registry_rejects_unknown_tools_and_write_tools() -> None:
    """Harden 5.8: drift fails loudly — unknown, write, and conflict tools."""
    from app.features.documents.application.tools.registry import (
        ToolBindingError,
        assert_role_bindings,
    )

    # Unknown tool.
    with pytest.raises(ToolBindingError, match="outside the registry"):
        assert_role_bindings(
            {"risk": [_stub_tool("mystery_tool")], "compliance": [], "orchestrator": []}
        )
    # Graphiti write tools stay unwired.
    with pytest.raises(ToolBindingError, match="Graphiti write"):
        assert_role_bindings(
            {
                "risk": [_stub_tool("write_clause_episodes_to_graphiti")],
                "compliance": [],
                "orchestrator": [],
            },
            evidence_seam_provisioned=False,
        )
    # The experiment-pending conflict tool stays unbound too.
    with pytest.raises(ToolBindingError, match="outside the registry"):
        assert_role_bindings(
            {"risk": [_stub_tool("detect_graph_conflicts")], "compliance": [], "orchestrator": []},
            evidence_seam_provisioned=False,
        )
    # Missing required tools under a provisioned seam.
    with pytest.raises(ToolBindingError, match="missing required"):
        assert_role_bindings(
            {"risk": [], "compliance": [], "orchestrator": []},
        )


def test_orchestrator_projection_cannot_read_evidence() -> None:
    """Harden 5.8: the orchestrator never carries an evidence read/search tool."""
    from app.features.documents.application.tools.registry import (
        EVIDENCE_READ_TOOLS,
        ToolBindingError,
        assert_role_bindings,
    )

    for name in sorted(EVIDENCE_READ_TOOLS):
        with pytest.raises(ToolBindingError, match="must never read legal evidence"):
            assert_role_bindings(
                {"risk": [], "compliance": [], "orchestrator": [_stub_tool(name)]},
                evidence_seam_provisioned=False,
            )
    # The documented orchestrator binding: delegation only (typed action);
    # ingestion state rides with the specialists it schedules.
    assert_role_bindings(
        {"risk": [], "compliance": [], "orchestrator": []},
        evidence_seam_provisioned=False,
    )
