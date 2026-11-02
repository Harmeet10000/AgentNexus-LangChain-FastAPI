"""Deny-by-default capability enforcement for the system/filesystem tools.

``ToolRuntimeContext`` is a ``total=False`` TypedDict and LangChain injects the
runtime invisibly to the type checker, so these tests are the gate that holds
the line: every tool refuses without an explicit server-side grant, and every
path-taking tool confines itself to the configured workspace jail — including
through symlinks. See PLAN-agent-tools.md D0.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from app.ai.tools import shell
from app.ai.tools.registry import register_default_tools
from app.ai.tools.web_search import WebSearchInput

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Final

    import pytest


def _runtime(*capabilities: str) -> SimpleNamespace:
    return SimpleNamespace(context={"capabilities": capabilities})


def _jail(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    monkeypatch.setattr(
        shell, "get_settings", lambda: SimpleNamespace(TOOL_WORKSPACE_ROOTS=(root,))
    )


def _no_jail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "get_settings", lambda: SimpleNamespace(TOOL_WORKSPACE_ROOTS=()))


async def test_shell_tool_denies_without_runtime(tmp_path: Path) -> None:
    result = await shell.shell_tool.coroutine(command="echo hi", cwd=str(tmp_path))
    assert result.success is False
    assert "system.shell.execute" in (result.error or "")


async def test_shell_tool_denies_wrong_grant(tmp_path, monkeypatch) -> None:
    _jail(monkeypatch, str(tmp_path))

    result = await shell.shell_tool.coroutine(
        command="echo hi",
        cwd=str(tmp_path),
        runtime=_runtime("filesystem.read"),
    )
    assert result.success is False
    assert "system.shell.execute" in (result.error or "")


async def test_shell_tool_denies_missing_cwd() -> None:
    result = await shell.shell_tool.coroutine(
        command="echo hi",
        runtime=_runtime("system.shell.execute"),
    )
    assert result.success is False
    assert "cwd" in (result.error or "")


async def test_shell_tool_runs_inside_jail(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _jail(monkeypatch, str(tmp_path))

    result = await shell.shell_tool.coroutine(
        command="echo hi",
        cwd=str(tmp_path),
        runtime=_runtime("system.shell.execute"),
    )
    assert result.success is True
    assert result.data["stdout"].strip() == "hi"


async def test_shell_tool_denies_cwd_outside_jail(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _jail(monkeypatch, str(tmp_path))

    result = await shell.shell_tool.coroutine(
        command="echo hi",
        cwd="/etc",
        runtime=_runtime("system.shell.execute"),
    )
    assert result.success is False
    assert "workspace jail" in (result.error or "")


async def test_read_file_denies_without_jail_configured(tmp_path, monkeypatch) -> None:
    _no_jail(monkeypatch)

    result = await shell.read_file.coroutine(
        path=str(tmp_path / "x.txt"), runtime=_runtime("filesystem.read")
    )
    assert result.success is False
    assert "TOOL_WORKSPACE_ROOTS" in (result.error or "")


async def test_read_file_serves_inside_jail(tmp_path, monkeypatch) -> None:
    _jail(monkeypatch, str(tmp_path))
    target = tmp_path / "x.txt"
    target.write_text("legal text", encoding="utf-8")

    result = await shell.read_file.coroutine(path=str(target), runtime=_runtime("filesystem.read"))
    assert result.success is True
    assert result.data["content"] == "legal text"


async def test_read_file_denies_symlink_escape(tmp_path, monkeypatch) -> None:
    _jail(monkeypatch, str(tmp_path))
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "innocent.md"
    link.symlink_to(outside)
    try:
        result = await shell.read_file.coroutine(
            path=str(link), runtime=_runtime("filesystem.read")
        )
        assert result.success is False
        assert "workspace jail" in (result.error or "")
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


async def test_write_file_denies_without_grant(tmp_path, monkeypatch) -> None:
    _jail(monkeypatch, str(tmp_path))

    result = await shell.write_file.coroutine(
        path=str(tmp_path / "x.txt"), content="hi", runtime=_runtime("filesystem.read")
    )
    assert result.success is False
    assert "filesystem.write" in (result.error or "")


async def test_write_file_writes_inside_jail(tmp_path, monkeypatch) -> None:
    _jail(monkeypatch, str(tmp_path))
    target = tmp_path / "out" / "x.txt"

    result = await shell.write_file.coroutine(
        path=str(target), content="body", runtime=_runtime("filesystem.write")
    )
    assert result.success is True
    assert target.read_text(encoding="utf-8") == "body"


async def test_list_directory_requires_grant(tmp_path, monkeypatch) -> None:
    _jail(monkeypatch, str(tmp_path))

    result = await shell.list_directory.coroutine(
        path=str(tmp_path), runtime=_runtime("filesystem.write")
    )
    assert result.success is False
    assert "filesystem.read" in (result.error or "")


async def test_file_search_serves_inside_jail(tmp_path, monkeypatch) -> None:
    _jail(monkeypatch, str(tmp_path))
    (tmp_path / "a.md").write_text("indemnification clause", encoding="utf-8")

    result = await shell.file_search.coroutine(
        query="indemnification",
        directory=str(tmp_path),
        runtime=_runtime("filesystem.read"),
    )
    assert result.success is True
    assert result.data["results"][0]["file"].endswith("a.md")


async def test_file_search_rejects_absolute_and_parent_patterns(tmp_path, monkeypatch) -> None:
    _jail(monkeypatch, str(tmp_path))

    for file_pattern in ("/etc/*", "../*.md", "nested/../../secret"):
        result = await shell.file_search.coroutine(
            query="secret",
            directory=str(tmp_path),
            file_pattern=file_pattern,
            runtime=_runtime("filesystem.read"),
        )
        assert result.success is False
        assert "relative pattern" in (result.error or "")


async def test_file_search_skips_symlinked_file_outside_jail(tmp_path, monkeypatch) -> None:
    _jail(monkeypatch, str(tmp_path))
    outside = tmp_path.parent / "outside-search-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "linked.txt"
    link.symlink_to(outside)
    try:
        result = await shell.file_search.coroutine(
            query="secret",
            directory=str(tmp_path),
            file_pattern="*",
            runtime=_runtime("filesystem.read"),
        )
        assert result.success is True
        assert result.data["results"] == []
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Registry authority metadata (openspec: agent-tool-capability-enforcement §3.1)
#
# The ``capability:*`` and ``side-effect:*`` tags are the authority catalog:
# what a registration advertises must be discoverable per tool, and the legacy
# category tags used for selection must behave exactly as before the metadata
# landed.
# ---------------------------------------------------------------------------

#: tool name → (enforced capability tag, side-effect class tag, legacy category tag)
_AUTHORITY: Final[dict[str, tuple[str, str, str]]] = {
    "shell_tool": ("capability:system.shell.execute", "side-effect:write", "system"),
    "read_file": ("capability:filesystem.read", "side-effect:read", "filesystem"),
    "write_file": ("capability:filesystem.write", "side-effect:write", "filesystem"),
    "list_directory": ("capability:filesystem.read", "side-effect:read", "filesystem"),
    "file_search": ("capability:filesystem.read", "side-effect:read", "filesystem"),
}


def test_default_registrations_carry_authority_metadata() -> None:
    """Each system/filesystem registration names its capability and side-effect class."""
    registry = register_default_tools()

    for name, (capability, side_effect, _legacy) in _AUTHORITY.items():
        # by_tags is AND-semantics, so a hit proves the same registration
        # carries both the capability and the side-effect tag.
        matches = {t.name for t in registry.by_tags(capability, side_effect)}
        assert name in matches, (name, capability, side_effect)

    # The capability identifiers are real, discoverable registry tags.
    assert {cap for cap, _, _ in _AUTHORITY.values()} <= registry.tags()


async def test_authority_metadata_matches_enforced_capability() -> None:
    """A registration's advertised capability is the one the executor enforces."""
    registry = register_default_tools()

    # Arguments are irrelevant: the grant check runs before anything else, so
    # the no-runtime refusal names the tool's enforced capability and none of
    # these paths is ever touched.
    args_by_tool: Final[dict[str, dict[str, str]]] = {
        "shell_tool": {"command": "echo hi"},
        "read_file": {"path": "unused/denied.md"},
        "write_file": {"path": "unused/denied.md", "content": "x"},
        "list_directory": {"path": "unused"},
        "file_search": {"query": "x", "directory": "unused"},
    }

    for name, (capability, _side_effect, _legacy) in _AUTHORITY.items():
        advertised = capability.removeprefix("capability:")
        result = await registry.get(name).coroutine(**args_by_tool[name])
        assert result.success is False, name
        assert advertised in (result.error or ""), (name, advertised, result.error)


def test_legacy_tag_selection_unchanged() -> None:
    """Selection by the legacy category tags returns the same tools as before."""
    registry = register_default_tools()

    assert {t.name for t in registry.by_tags("filesystem")} == {
        "read_file",
        "write_file",
        "list_directory",
        "file_search",
    }
    assert {t.name for t in registry.by_tags("web")} == {"web_search"}
    assert {t.name for t in registry.by_tags("system")} == {"shell_tool"}


# ---------------------------------------------------------------------------
# Argument schemas are plain schema models (openspec §3.2)
# ---------------------------------------------------------------------------


def test_web_search_args_schema_is_schema_only() -> None:
    """The web-search tool's argument schema is a plain schema model, not a tool."""
    registry = register_default_tools()

    schema = registry.get("web_search").args_schema
    assert schema is WebSearchInput
    # Schema-only: it must not be a tool type, so it cannot be registered or
    # invoked in its own right.
    assert not issubclass(schema, BaseTool)
    # It exposes exactly its declared, model-visible fields — no injected
    # runtime context leaks into the schema.
    assert set(schema.model_fields) == {"query", "max_results"}
