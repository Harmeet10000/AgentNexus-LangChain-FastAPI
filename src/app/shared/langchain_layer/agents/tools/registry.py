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
from .crawl import get_crawl_url_tool
from .handoff import make_handoff_tools
from .web_search import get_web_search_tool

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
    registers nothing; only calling this entry point populates the registry.
    """
    r = get_tool_registry()
    r.register(get_web_search_tool(), tags=["web", "search"])
    r.register(get_crawl_url_tool(), tags=["web", "crawl"])
    for tool in make_handoff_tools():
        r.register(tool, tags=["handoff", tool.name.removeprefix("transfer_to_")])
    from . import (  # noqa: PLC0415 — D-1(c): entry-point import keeps package import side-effect free
        shell as _shell,
    )

    r.register(_shell.shell_tool, tags=["system", "shell"])
    r.register(_shell.read_file, tags=["filesystem", "read"])
    r.register(_shell.write_file, tags=["filesystem", "write"])
    r.register(_shell.list_directory, tags=["filesystem", "list"])
    r.register(_shell.file_search, tags=["filesystem", "search"])
    return r


def get_all_tools() -> list[BaseTool]:
    """Convenience: register the defaults, then return everything."""
    register_default_tools()
    return get_tool_registry().all()


def get_web_tools() -> list[BaseTool]:
    """Convenience: the ``web``-tagged tools."""
    register_default_tools()
    return get_tool_registry().by_tags("web")


__all__ = [
    "get_all_tools",
    "get_tool_registry",
    "get_web_tools",
    "register_default_tools",
]
