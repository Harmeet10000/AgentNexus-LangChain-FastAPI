"""Projection consistency: the policy is the only author of grants.

Covers openspec ``tool-capability-catalog`` tasks 2.1 (context capability set
== policy entry per role) and 3.2 (model-visible toolset ⊆ policy-authorized,
≤7 projection cap, failing with the policy row named).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.tools import tool

from app.ai.tools.policy import ROLE_CAPABILITY_GRANTS, grants_for
from app.ai.tools.registry import (
    DEEP_RESEARCH_TOOL_NAME,
    PORTFOLIO_TRAVERSAL_TOOL_NAMES,
    deep_research_role_projection,
    get_tool_registry,
    register_default_tools,
)
from app.features.agent_saul.graph import factory as saul_factory
from app.features.agent_saul.graph.nodes import _analysis_tool_context
from app.features.documents.application.evidence.service import EvidenceService
from app.features.documents.application.tools.bundle import AgentToolBundle
from app.features.documents.application.tools.portfolio import (
    RelationshipEdgeProbe,
    RelationshipEdgeVerdict,
    TraversalGateError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

pytestmark = pytest.mark.unit


class _StubLLM:
    def with_structured_output(self, schema: Any) -> Any:
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda value: (schema, value))


def _stub_tool(name: str) -> Any:
    @tool(name)
    async def _stub() -> dict[str, str]:
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


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Build the registry with the real factory, capturing per-role toolsets."""
    captured: dict[str, Any] = {}

    def _fake_create_agent(*_args: Any, **kwargs: Any) -> Any:
        captured[kwargs["system_prompt"][:24]] = kwargs
        return object()

    monkeypatch.setattr(saul_factory, "create_agent", _fake_create_agent)
    register_default_tools()
    saul_factory.build_agent_registry(_StubLLM(), _StubLLM(), tools=_bundle())  # type: ignore[arg-type]
    return captured


@pytest.fixture(autouse=True)
def _leave_registry_as_found() -> Iterator[None]:
    yield
    r = get_tool_registry()
    for attr in ("_tools", "_tags", "_authority"):
        getattr(r, attr).pop(DEEP_RESEARCH_TOOL_NAME, None)
        for name in PORTFOLIO_TRAVERSAL_TOOL_NAMES:
            getattr(r, attr).pop(name, None)


def _role_toolsets(bound: dict[str, Any]) -> dict[str, list[str]]:
    """Each policy role's model-visible toolset, mirroring the factory bindings."""
    from app.features.agent_saul.graph.prompts import (
        _COMPLIANCE_SYSTEM_PROMPT,
        _RISK_ANALYSIS_SYSTEM_PROMPT,
    )

    return {
        "risk": [t.name for t in bound[_RISK_ANALYSIS_SYSTEM_PROMPT[:24]]["tools"]],
        "compliance": [t.name for t in bound[_COMPLIANCE_SYSTEM_PROMPT[:24]]["tools"]],
        "deep_research": [t.name for t in deep_research_role_projection()],
        "orchestrator": [],  # structured-output nodes — no tools by design
        "qna": [],
    }


def test_context_capability_set_equals_policy_entry_per_role() -> None:
    """Every context-build site populates capabilities from the policy only."""
    state: dict[str, Any] = {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "capabilities": ("document.read",),  # state channel must NOT leak in
    }
    for role in ROLE_CAPABILITY_GRANTS:
        context = _analysis_tool_context(state, role=role)  # type: ignore[arg-type]
        assert context["capabilities"] == tuple(sorted(grants_for(role))), role


def test_every_role_projection_is_authorized_and_within_cap(
    bound: dict[str, Any],
) -> None:
    """Model-visible ⊆ policy-authorized and the ≤7 projection cap, per role.

    A violation fails naming the policy row, so a future toolset edit that
    breaks either invariant points straight at the offending grants entry.
    """
    registry = get_tool_registry()
    for role, visible in _role_toolsets(bound).items():
        grants = grants_for(role)
        assert len(visible) <= 7, (
            f"policy row {role!r} (grants {sorted(grants)}) projects "
            f"{len(visible)} tools (> 7): {visible}"
        )
        for name in visible:
            try:
                capability = registry.authority(name).capability
            except KeyError:
                continue  # bundle tool: no declared authority in the catalog
            assert capability is None or capability in grants, (
                f"policy row {role!r} (grants {sorted(grants)}) does not authorize "
                f"visible tool {name!r} declaring capability {capability!r}"
            )


def test_unknown_role_context_grants_nothing() -> None:
    """A context built for a role the policy does not name carries no grants."""
    state: dict[str, Any] = {"user_id": "user-1", "thread_id": "thread-1"}
    context = _analysis_tool_context(state, role="no-such-role")  # type: ignore[arg-type]
    assert context["capabilities"] == ()


# ---------------------------------------------------------------------------
# Portfolio traversal binding (openspec agent-portfolio-traversal-tools 3.2)
# ---------------------------------------------------------------------------


def _open_gate_probe() -> RelationshipEdgeProbe:
    return RelationshipEdgeProbe(
        verdict=RelationshipEdgeVerdict.EDGES_PRESENT,
        edges_observed=1,
        documents_probed=2,
    )


def test_traversal_binding_keeps_projections_within_cap_and_authorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the gate, both specialists' projections stay ≤7 and authorized."""
    captured = _build_with_traversal(monkeypatch, _open_gate_probe())
    toolsets = _role_toolsets(captured)

    assert toolsets["risk"] == [
        "transfer_to_orchestrator",
        "query_knowledge_graph",
        "get_obligation_chain",
        "search_evidence",
        "read_evidence_by_id",
        "get_ingestion_state",
        "cross_document_obligations",
    ]
    assert toolsets["compliance"] == [
        "transfer_to_orchestrator",
        "transfer_to_risk",
        "search_evidence",
        "read_evidence_by_id",
        "get_ingestion_state",
        "find_conflicting_clauses",
        "party_obligation_map",
    ]

    registry = get_tool_registry()
    for role, visible in toolsets.items():
        grants = grants_for(role)
        assert len(visible) <= 7, f"policy row {role!r} exceeds the ≤7 cap: {visible}"
        for name in visible:
            try:
                capability = registry.authority(name).capability
            except KeyError:
                continue  # bundle tool: no declared authority in the catalog
            assert capability is None or capability in grants, (
                f"policy row {role!r} (grants {sorted(grants)}) does not authorize "
                f"visible tool {name!r} declaring capability {capability!r}"
            )


def test_traversal_binding_without_edges_fails_the_build_naming_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A graphless corpus fails the build loudly; no traversal tool binds."""
    with pytest.raises(TraversalGateError, match="relationship-edges"):
        _build_with_traversal(
            monkeypatch,
            RelationshipEdgeProbe(
                verdict=RelationshipEdgeVerdict.NO_EDGES,
                edges_observed=0,
                documents_probed=2,
            ),
        )


def test_traversal_binding_without_a_probe_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No probe means a scopeless corpus: fail closed, never bind."""
    with pytest.raises(TraversalGateError, match="verdict no_scope"):
        _build_with_traversal(monkeypatch, None)


def test_backendless_build_binds_no_traversal_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a graph backend the build behaves exactly as before the change."""
    captured = _build_with_traversal(monkeypatch, _open_gate_probe(), with_backend=False)
    toolsets = _role_toolsets(captured)
    for name in (*PORTFOLIO_TRAVERSAL_TOOL_NAMES,):
        for visible in toolsets.values():
            assert name not in visible
        # The catalog still carries the (backend-less) tool so the policy's
        # portfolio.traversal grants name a declared capability.
        assert name in get_tool_registry().names()


def _build_with_traversal(
    monkeypatch: pytest.MonkeyPatch,
    probe: RelationshipEdgeProbe | None,
    *,
    with_backend: bool = True,
) -> dict[str, Any]:
    """Build the registry with the real factory, capturing per-role toolsets."""
    captured: dict[str, Any] = {}

    def _fake_create_agent(*_args: Any, **kwargs: Any) -> Any:
        captured[kwargs["system_prompt"][:24]] = kwargs
        return object()

    monkeypatch.setattr(saul_factory, "create_agent", _fake_create_agent)
    register_default_tools()
    saul_factory.build_agent_registry(
        _StubLLM(),
        _StubLLM(),  # type: ignore[arg-type]
        tools=_bundle(),  # type: ignore[arg-type]
        traversal_graphiti_service=object() if with_backend else None,
        traversal_gate_probe=probe,
    )
    return captured
