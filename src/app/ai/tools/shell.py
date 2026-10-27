"""
System tools: shell execution, filesystem operations, file search.

All tools return the shared ToolResult envelope — structured, never self-rendered.

Every tool in this module is **deny-by-default**. It executes only when the
server-owned ``ToolRuntime`` context grants the tool's capability, and every
path-taking tool confines itself to the configured workspace jail
(``TOOL_WORKSPACE_ROOTS``). Neither the grant nor the jail is ever a
model-visible argument: LangChain injects ``ToolRuntime`` and hides it from the
tool schema, so the model chooses *what* to run but never *whether* it may.
Refusal messages name the missing grant or setting so the operator's next step
is a grep, not a guessing game.
"""

from __future__ import annotations

import asyncio
import fnmatch
import subprocess  # noqa: S404 — this module IS the sandboxed shell tool; the import names SubprocessError for the specific catch below
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.resilience.idempotency import ToolResult

from .base import register_tool

if TYPE_CHECKING:
    from typing import Any

# ---------------------------------------------------------------------------
# Capability model — the grants live in server-side policy, never in
# model-visible arguments. See PLAN-agent-tools.md §3.
# ---------------------------------------------------------------------------

CAP_SHELL_EXECUTE = "system.shell.execute"
CAP_FS_READ = "filesystem.read"
CAP_FS_WRITE = "filesystem.write"


def _granted(runtime: Annotated[ToolRuntime | None, InjectedToolArg], capability: str) -> bool:
    """True only when the injected runtime context carries ``capability``."""
    context = getattr(runtime, "context", None)
    granted = context.get("capabilities", ()) if isinstance(context, Mapping) else ()
    if isinstance(granted, str):
        granted = (granted,)
    return isinstance(granted, (list, tuple, set, frozenset)) and capability in granted


def _require_capability(
    runtime: Annotated[ToolRuntime | None, InjectedToolArg], capability: str
) -> None:
    if not _granted(runtime, capability):
        msg = (
            f"capability {capability!r} is not granted in the tool runtime context; "
            "grant it in server-side policy — it can never come from model arguments"
        )
        raise PermissionError(msg)


def _resolve_in_jail(raw: str, operation: str) -> Path:
    """Resolve ``raw`` and require it to sit inside a configured workspace root.

    The check runs on the *resolved* path, so a symlink from inside the jail to
    anywhere outside it fails containment rather than passing the check and
    escaping through the link.
    """
    roots = get_settings().TOOL_WORKSPACE_ROOTS
    if not roots:
        msg = (
            "no workspace jail is configured; set TOOL_WORKSPACE_ROOTS before "
            "binding filesystem or shell tools"
        )
        raise PermissionError(msg)
    resolved = Path(raw).resolve()
    if not any(resolved.is_relative_to(Path(root).resolve()) for root in roots):
        msg = f"{operation} path {raw!r} resolves outside the workspace jail"
        raise PermissionError(msg)
    return resolved


def _validate_file_pattern(file_pattern: str) -> None:
    """Reject glob patterns that can select outside the requested directory."""
    candidate = Path(file_pattern)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        msg = f"search file_pattern {file_pattern!r} must be a relative pattern without '..'"
        raise PermissionError(msg)


# ---------------------------------------------------------------------------
# Shell tool
# ---------------------------------------------------------------------------


class ShellInput(BaseModel):
    command: str = Field(..., description="The shell command to execute.")
    cwd: str | None = Field(None, description="Working directory for the command.")
    timeout: int = Field(30, description="Timeout in seconds.", ge=1, le=120)


class ShellOutput(BaseModel):
    stdout: str
    stderr: str
    returncode: int
    success: bool


@register_tool("system", "shell")
@tool(args_schema=ShellInput)
async def shell_tool(
    command: str,
    cwd: str | None = None,
    timeout: int = 30,  # noqa: ASYNC109 — name must match ShellInput.timeout; the tool runtime passes timeout= as a kwarg
    runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
) -> ToolResult:
    """
    Execute a shell command and return stdout/stderr.

    Requires the ``system.shell.execute`` grant and an explicit ``cwd`` inside
    the workspace jail — a shell without a jailed working directory is an
    unconfined process, so a missing ``cwd`` is a refusal, not a default.
    """
    try:
        _require_capability(runtime, CAP_SHELL_EXECUTE)
        if cwd is None:
            return ToolResult.fail("shell_tool requires an explicit cwd inside the workspace jail")
        jailed_cwd = _resolve_in_jail(cwd, "shell cwd")
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(jailed_cwd),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        result = ShellOutput(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            returncode=proc.returncode or 0,
            success=proc.returncode == 0,
        )
        return ToolResult.ok(data=result.model_dump())
    except PermissionError as exc:
        return ToolResult.fail(str(exc))
    except TimeoutError:
        return ToolResult.fail(f"Command timed out after {timeout}s")
    except (OSError, subprocess.SubprocessError) as exc:
        exc.add_note(f"command={command}, operation=execute_shell")
        return ToolResult.fail(f"Shell execution failed: {exc}")
    except Exception as exc:  # noqa: BLE001 — subprocess can raise varied exceptions
        exc.add_note(f"command={command}, operation=execute_shell")
        return ToolResult.fail(str(exc))


# ---------------------------------------------------------------------------
# Filesystem tools
# ---------------------------------------------------------------------------


class ReadFileInput(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file.")
    encoding: str = Field("utf-8", description="File encoding.")


class WriteFileInput(BaseModel):
    path: str = Field(..., description="Path to write the file.")
    content: str = Field(..., description="Content to write.")
    mode: str = Field("w", description="Write mode: 'w' (overwrite) or 'a' (append).")


class ListDirInput(BaseModel):
    path: str = Field(".", description="Directory path to list.")
    recursive: bool = Field(False, description="List recursively.")
    pattern: str | None = Field(None, description="Glob pattern filter, e.g. '*.py'.")


@register_tool("filesystem", "read")
@tool(args_schema=ReadFileInput)
async def read_file(
    path: str,
    encoding: str = "utf-8",
    runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
) -> ToolResult:
    """Read the contents of a file and return them as a string."""
    try:
        _require_capability(runtime, CAP_FS_READ)
        target = _resolve_in_jail(path, "read")
    except PermissionError as exc:
        return ToolResult.fail(str(exc))
    try:
        content = target.read_text(encoding=encoding)
        return ToolResult.ok(data={"content": content})
    except FileNotFoundError:
        return ToolResult.fail(f"File not found: {path}")
    except PermissionError as exc:
        exc.add_note(f"path={path}, operation=read_file")
        return ToolResult.fail(f"Permission denied reading file: {path}")
    except OSError as exc:
        exc.add_note(f"path={path}, operation=read_file")
        return ToolResult.fail(f"OS error reading file {path}: {exc}")
    except Exception as exc:  # noqa: BLE001 — file read, unknown encoding/permission errors
        exc.add_note(f"path={path}, operation=read_file")
        return ToolResult.fail(str(exc))


@register_tool("filesystem", "write")
@tool(args_schema=WriteFileInput)
async def write_file(
    path: str,
    content: str,
    mode: str = "w",
    runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
) -> ToolResult:
    """Write content to a file. Creates parent directories if needed."""
    try:
        _require_capability(runtime, CAP_FS_WRITE)
        target = _resolve_in_jail(path, "write")
    except PermissionError as exc:
        return ToolResult.fail(str(exc))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "a":
            with target.open(mode="a", encoding="utf-8") as handle:
                handle.write(content)
        else:
            target.write_text(content, encoding="utf-8")
        return ToolResult.ok(data={"message": f"Written {len(content)} bytes to {path}"})
    except PermissionError as exc:
        exc.add_note(f"path={path}, operation=write_file")
        return ToolResult.fail(f"Permission denied writing file: {path}")
    except OSError as exc:
        exc.add_note(f"path={path}, operation=write_file")
        return ToolResult.fail(f"OS error writing file {path}: {exc}")
    except Exception as exc:  # noqa: BLE001 — file write, unknown permission/disk errors
        exc.add_note(f"path={path}, operation=write_file")
        return ToolResult.fail(str(exc))


@register_tool("filesystem", "list")
@tool(args_schema=ListDirInput)
async def list_directory(
    path: str = ".",
    recursive: bool = False,
    pattern: str | None = None,
    runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
) -> ToolResult:
    """List files in a directory, optionally filtering by glob pattern."""
    try:
        _require_capability(runtime, CAP_FS_READ)
        base = _resolve_in_jail(path, "list")
    except PermissionError as exc:
        return ToolResult.fail(str(exc))
    try:
        if not base.exists():
            return ToolResult.fail(f"Path does not exist: {path}")

        if recursive:
            all_files = [str(p) for p in base.rglob("*") if p.is_file()]
        else:
            all_files = [str(p) for p in base.iterdir()]

        if pattern:
            all_files = [f for f in all_files if fnmatch.fnmatch(Path(f).name, pattern)]

        return ToolResult.ok(data={"files": sorted(all_files)})
    except PermissionError as exc:
        exc.add_note(f"path={path}, operation=list_dir")
        return ToolResult.fail(f"Permission denied listing directory: {path}")
    except OSError as exc:
        exc.add_note(f"path={path}, operation=list_dir")
        return ToolResult.fail(f"OS error listing directory {path}: {exc}")
    except Exception as exc:  # noqa: BLE001 — directory listing, unknown permission errors
        exc.add_note(f"path={path}, operation=list_dir")
        return ToolResult.fail(str(exc))


# ---------------------------------------------------------------------------
# File search (grep-style)
# ---------------------------------------------------------------------------


class FileSearchInput(BaseModel):
    query: str = Field(..., description="Search string or regex pattern.")
    directory: str = Field(".", description="Directory to search in.")
    file_pattern: str = Field("*", description="Glob pattern for files to include.")
    max_results: int = Field(20, description="Maximum number of matching lines to return.")
    use_regex: bool = Field(False, description="Treat query as a regex pattern.")


@register_tool("filesystem", "search")
@tool(args_schema=FileSearchInput)
async def file_search(
    query: str,
    *,
    directory: str = ".",
    file_pattern: str = "*",
    max_results: int = 20,
    use_regex: bool = False,
    runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
) -> ToolResult:
    """Search for a string or regex pattern across files in a directory."""
    try:
        _require_capability(runtime, CAP_FS_READ)
        base = _resolve_in_jail(directory, "search")
        _validate_file_pattern(file_pattern)
    except PermissionError as exc:
        return ToolResult.fail(str(exc))

    import re as _re

    results: list[dict[str, Any]] = []
    pattern = _re.compile(query) if use_regex else None

    for file_path in base.rglob(file_pattern):
        if not file_path.is_file():
            continue
        # ``rglob`` can yield symlinked files.  Resolve every candidate before
        # reading it; checking only ``directory`` would let a link inside the
        # jail disclose an outside file.
        try:
            _resolve_in_jail(str(file_path), "search result")
        except PermissionError:
            continue
        try:
            lines = file_path.read_text(errors="replace").splitlines()
        except Exception:  # noqa: BLE001, S112 — file read per-file in search, skip unreadable
            continue
        for lineno, line in enumerate(lines, 1):
            matched = bool(pattern.search(line)) if pattern else query in line
            if matched:
                results.append(
                    {
                        "file": str(file_path),
                        "line": lineno,
                        "content": line.strip(),
                    }
                )
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break

    return ToolResult.ok(data={"results": results}, total=len(results))
