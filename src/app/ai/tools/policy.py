"""The role → capability grants policy (openspec: tool-capability-catalog, ADR-1).

This module is the *only* author of capability grants: one static, versioned
declaration mapping each agent role (and the unbound/system context) to the
capability identifiers it may exercise. Runtime ``ToolRuntimeContext.capabilities``
tuples are populated from here — no call site assembles a grant ad hoc.

Loading rules:
  * An unknown role is tolerated and grants nothing (deny-by-default at the
    policy layer — a role that grants no policy receives an empty set).
  * An unknown *capability identifier* — a grant naming a capability no
    registered tool declares — is a config bug, not a no-op: it fails fast at
    startup via :func:`validate_policy`.

The initial rows encode each role's *current effective* grants exactly
(PLAN-agent-tools.md § D4 migration plan): no role holds any
system/filesystem capability, nothing grants shell or filesystem access,
``web.search`` is ungranted, and the only capabilities reachable beyond the
evidence-bundle operation grants are the deep-research subgraph's own catalog
entry and the read-only portfolio traversal capability shared by the risk and
compliance specialists. Narrowing or widening is a reviewed diff to this file
alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .base import ToolRegistry

#: Graph roles and the unbound/system context. Capability identifiers here
#: MUST match the ``capability`` a tool declares at registration; use
#: :func:`validate_policy` to keep the two honest.
ROLE_CAPABILITY_GRANTS: Final[Mapping[str, frozenset[str]]] = {
    # The orchestrator runs structured-output nodes only — no tools by design.
    "orchestrator": frozenset(),
    # Specialists bind handoff + evidence-bundle tools. Both gain the
    # portfolio traversal capability (openspec agent-portfolio-traversal-tools
    # task 3.2): read-only, non-egress graph traversal whose executor gates are
    # the server-owned document scope and the runtime relationship-edge probe.
    # No system/filesystem/web capability is granted to either (P5: web search
    # sends model-chosen text to a third party — ungranted until a reviewed
    # policy diff says otherwise).
    "risk": frozenset({"portfolio.traversal"}),
    "compliance": frozenset({"portfolio.traversal"}),
    # The bounded deep-research subgraph is the deep_research role's single
    # catalog tool (D5). Its registration declares ``deep-research``.
    "deep_research": frozenset({"deep-research"}),
    # Schema-locked Q&A — no tools by design.
    "qna": frozenset(),
    # Contexts built outside any named role (unbound/system): nothing.
    "unbound": frozenset(),
}


def grants_for(role: str | None) -> frozenset[str]:
    """The granted capability identifiers for ``role`` (unknown → empty set)."""
    if role is None:
        return ROLE_CAPABILITY_GRANTS["unbound"]
    return ROLE_CAPABILITY_GRANTS.get(role, frozenset())


def validate_policy(registry: ToolRegistry) -> None:
    """Fail fast when a grant names a capability no registered tool declares.

    Called once the startup catalog is fully populated (the deep-research
    registration is the last one), so a typo'd capability identifier in this
    file is a startup error rather than a silent no-op grant.
    """
    declared = registry.declared_capabilities()
    unknown = sorted(
        {capability for grants in ROLE_CAPABILITY_GRANTS.values() for capability in grants}
        - declared
    )
    if unknown:
        msg = (
            "tool capability policy grants capability identifiers no registered "
            f"tool declares: {unknown}; fix ROLE_CAPABILITY_GRANTS in "
            "app.ai.tools.policy"
        )
        raise ValueError(msg)


__all__ = ["ROLE_CAPABILITY_GRANTS", "grants_for", "validate_policy"]
