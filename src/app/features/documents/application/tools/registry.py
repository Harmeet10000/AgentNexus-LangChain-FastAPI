"""The authoritative per-node tool-binding registry (harden 5.8).

The spec (agent-tool-registry) fixes the documented role bindings: risk gets
evidence search/read, the obligation chain, the typed knowledge-graph query,
and handoff; compliance gets evidence search/read, statute retrieval, and
handoff; the orchestrator delegates and observes ingestion state — it never
reads legal evidence directly. Graphiti write tools stay unwired to any node,
and `detect_graph_conflicts` stays unbound pending its recorded experiment
decision (no conflicts-family evidence exists in `evals/reports` yet).

`build_agent_registry` binds from this table and calls
:func:`assert_role_bindings` at construction — drift fails the build loudly
(the design's "tool registry drift" risk), and the same assertion runs in CI
through `tests/unit/features/agent_saul/graph/test_factory_bindings.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class ToolBindingError(Exception):
    """A role projection drifted from the authoritative registry."""


#: Tools every role projection MUST carry, per the documented bindings
#: (agent-tool-registry). The evidence seam (search/read/readiness) is
#: required only when an EvidenceService is provisioned — see
#: :func:`assert_role_bindings`.
BASE_REQUIRED_ROLE_TOOLS: Mapping[str, frozenset[str]] = {
    "risk": frozenset(
        {
            "transfer_to_orchestrator",
            "query_knowledge_graph",
            "get_obligation_chain",
        }
    ),
    "compliance": frozenset(
        {
            "transfer_to_orchestrator",
            "transfer_to_risk",
            "retrieve_statute_section",
        }
    ),
    # The orchestrator is a structured-output router: delegation is the typed
    # OrchestratorAction, never a model-callable tool. It carries no evidence
    # read/search tool — the assertion below enforces exactly that.
    "orchestrator": frozenset(),
}

#: The scoped evidence seam both specialists share (D3 readiness parity).
EVIDENCE_SEAM_TOOLS = frozenset({"search_evidence", "read_evidence_by_id", "get_ingestion_state"})

#: Legacy specialist tools that stay bound only in the transitional build
#: (no provisioned EvidenceService). Once the seam is provisioned the scoped
#: contract replaces them, exactly as the bundle's docstring records.
TRANSITIONAL_TOOLS: Mapping[str, frozenset[str]] = {
    "risk": frozenset(),
    "compliance": frozenset({"search_legal_precedents"}),
}

#: Optional seats granted only behind the binding-time traversal gate
#: (agent-portfolio-traversal-tools, task 3.3); the projections stay within
#: the ≤7 cap exactly as that change's design addendum recorded.
TRAVERSAL_SEATS: Mapping[str, frozenset[str]] = {
    "risk": frozenset({"cross_document_obligations"}),
    "compliance": frozenset({"find_conflicting_clauses", "party_obligation_map"}),
}

#: Any tool that reads legal evidence — the orchestrator must carry none.
EVIDENCE_READ_TOOLS = frozenset(
    {
        "search_evidence",
        "read_evidence_by_id",
        "search_legal_precedents",
        "retrieve_statute_section",
        "query_knowledge_graph",
        "get_obligation_chain",
        "deep_research",
    }
)

#: Graphiti write tools are implemented but deliberately unwired (harden 5.8):
#: persistence runs only after authorized human approval, outside any node's
#: toolset. A binding that names one fails the build.
GRAPHITI_WRITE_TOOLS = frozenset(
    {
        "write_clause_episodes_to_graphiti",
        "write_final_report_to_memory",
    }
)


def tool_name(bound: object) -> str:
    return getattr(bound, "name", str(bound))


def assert_role_bindings(
    bound_by_role: Mapping[str, Iterable[object]],
    *,
    evidence_seam_provisioned: bool = True,
) -> None:
    """Assert every role projection against the authoritative table.

    - every bound tool is documented (required or a traversal seat);
    - every required tool is bound (evidence tools only when the evidence
      seam is provisioned — the transitional no-`EvidenceService` startup);
    - no Graphiti write tool and no unbound-pending experiment tool appears;
    - the orchestrator carries no evidence read/search tool.
    """
    for role, tools in bound_by_role.items():
        if role not in BASE_REQUIRED_ROLE_TOOLS:
            message = f"unknown role in the tool registry: {role!r}"
            raise ToolBindingError(message)
        names = {tool_name(tool) for tool in tools}
        allowed = (
            BASE_REQUIRED_ROLE_TOOLS[role]
            | TRAVERSAL_SEATS.get(role, frozenset())
            | (EVIDENCE_SEAM_TOOLS if evidence_seam_provisioned else frozenset())
            | (
                frozenset()
                if evidence_seam_provisioned
                else TRANSITIONAL_TOOLS.get(role, frozenset())
            )
        )
        # Authority violations are checked before the unknown-tool rule so the
        # message names the worse violation: persistence never becomes a node
        # tool, and the orchestrator never reads legal evidence.
        write = names & GRAPHITI_WRITE_TOOLS
        if write:
            message = f"{role} binds Graphiti write tools: {sorted(write)}"
            raise ToolBindingError(message)
        forbidden = names & EVIDENCE_READ_TOOLS if role == "orchestrator" else frozenset()
        if forbidden:
            message = f"{role} must never read legal evidence directly: {sorted(forbidden)}"
            raise ToolBindingError(message)
        unknown = names - allowed
        if unknown:
            message = f"{role} binds tools outside the registry: {sorted(unknown)}"
            raise ToolBindingError(message)
    if evidence_seam_provisioned:
        for role in ("risk", "compliance"):
            names = {tool_name(tool) for tool in bound_by_role[role]}
            required = BASE_REQUIRED_ROLE_TOOLS[role] | EVIDENCE_SEAM_TOOLS
            missing = required - names
            if missing:
                message = f"{role} is missing required tools: {sorted(missing)}"
                raise ToolBindingError(message)
