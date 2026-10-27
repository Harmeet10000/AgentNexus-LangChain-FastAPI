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

from app.ai.tools import shell

if TYPE_CHECKING:
    from pathlib import Path

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
