"""Explicit tool registration (band: agent-tools-unification, group 3).

Import registers nothing — the registry is populated only when a consumer calls
:data:`register_default_tools`, so "what tools exist" is one readable call, not
an import-order accident. Registration is idempotent; resolving an unknown name
raises :class:`KeyError` instead of returning ``None`` and failing three frames
later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ToolRegistry
from .handoff import make_handoff_tools
from .policy import validate_policy
from .web_search import CAP_WEB_SEARCH, get_web_search_tool

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """The process-wide registry. Empty until :func:`register_default_tools` runs."""
    global _registry  # noqa: PLW0603
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_default_tools() -> ToolRegistry:
    """Register the default tool set. Idempotent by construction.

    Web tools carry the ``web`` tag; every role gets a ``transfer_to_<role>``
    tool tagged ``handoff`` so the orchestrator is genuinely tool-using and
    tag-based selection is real, not decorative. The five shell/filesystem
    tools are registered here explicitly (D-1(c)) — importing this module
    registers nothing; only calling this entry point populates the registry —
    and they enforce their capability grants at execution time regardless of
    how they are bound (PLAN-agent-tools.md D0). The ``capability:*`` /
    ``side-effect:*`` tags are derived views over the structured authority
    metadata (openspec ``tool-capability-catalog``): the metadata is the
    contract, the tags must agree with it, and web search's egress
    classification is enforced by the same executor contract (D4).
    """
    r = get_tool_registry()
    r.register(
        get_web_search_tool(),
        tags=["web", "search", "side-effect:egress", f"capability:{CAP_WEB_SEARCH}"],
        capability=CAP_WEB_SEARCH,
        side_effect="egress",
        egress=True,
    )
    for tool in make_handoff_tools():
        r.register(tool, tags=["handoff", tool.name.removeprefix("transfer_to_")])
    from . import (  # noqa: PLC0415 — D-1(c): entry-point import keeps package import side-effect free
        shell as _shell,
    )

    r.register(
        _shell.shell_tool,
        tags=["system", "shell", "side-effect:write", "capability:system.shell.execute"],
        capability=_shell.CAP_SHELL_EXECUTE,
        side_effect="write",
    )
    r.register(
        _shell.read_file,
        tags=["filesystem", "read", "side-effect:read", "capability:filesystem.read"],
        capability=_shell.CAP_FS_READ,
        side_effect="read",
    )
    r.register(
        _shell.write_file,
        tags=["filesystem", "write", "side-effect:write", "capability:filesystem.write"],
        capability=_shell.CAP_FS_WRITE,
        side_effect="write",
    )
    r.register(
        _shell.list_directory,
        tags=["filesystem", "list", "side-effect:read", "capability:filesystem.read"],
        capability=_shell.CAP_FS_READ,
        side_effect="read",
    )
    r.register(
        _shell.file_search,
        tags=["filesystem", "search", "side-effect:read", "capability:filesystem.read"],
        capability=_shell.CAP_FS_READ,
        side_effect="read",
    )
    return r


def get_all_tools() -> list[BaseTool]:
    """Convenience: register the defaults, then return everything."""
    register_default_tools()
    return get_tool_registry().all()


def get_web_tools() -> list[BaseTool]:
    """Convenience: the ``web``-tagged tools."""
    register_default_tools()
    return get_tool_registry().by_tags("web")


#: The deep-research subgraph wrapper's registry name (one tool per D5).
DEEP_RESEARCH_TOOL_NAME = "deep_research"
#: Authority tags for the read-only deep-research registration.
_DEEP_RESEARCH_TAGS = [
    "deep-research",
    "search",
    "side-effect:read",
    "capability:deep-research",
]
#: Capability identifier the deep-research registration declares (D5).
CAP_DEEP_RESEARCH = "deep-research"


def register_deep_research_tool(tool: BaseTool) -> ToolRegistry:
    """Register the deep-research subgraph wrapper (PLAN-agent-tools.md D5).

    The registration carries read-only structured authority: the wrapper only
    issues and dereferences run-scoped evidence handles, never writes. The
    wrapper does not enforce the ``deep-research`` capability identifier at
    execution — its deny-by-default gate is the server-owned document scope —
    so the identifier classifies the tool for the catalog and the selector
    rather than gating the executor. Registration is idempotent (the registry
    keys by tool name), so graph rebuilds reuse one catalog entry.

    This is the last registration of the startup catalog, so the role → grants
    policy is validated fail-fast here: a grant naming a capability no
    registered tool declares is a config bug, not a no-op (ADR-1).
    """
    r = get_tool_registry()
    r.register(
        tool,
        tags=_DEEP_RESEARCH_TAGS,
        capability=CAP_DEEP_RESEARCH,
        side_effect="read",
    )
    validate_policy(r)
    return r


def deep_research_role_projection() -> list[BaseTool]:
    """The deep_research role's model-visible toolset: exactly two tools.

    Its ``transfer_to_deep_research`` handoff plus the single subgraph
    wrapper — rebuilding the subgraph's internals must never change this
    projection (openspec ``deep-research-subgraph``, one-visible-tool).
    """
    return get_tool_registry().by_names(["transfer_to_deep_research", DEEP_RESEARCH_TOOL_NAME])


__all__ = [
    "deep_research_role_projection",
    "get_all_tools",
    "get_tool_registry",
    "get_web_tools",
    "register_deep_research_tool",
    "register_default_tools",
]
