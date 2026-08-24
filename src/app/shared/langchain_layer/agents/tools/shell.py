"""
System tools: shell execution, filesystem operations, file search.

All tools return the shared ToolResult envelope — structured, never self-rendered.
Shell and filesystem tools require explicit capability grants via context.
"""

from __future__ import annotations

import asyncio
import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .base import build_validation_error_handler, register_tool
from .idempotency import ToolResult

if TYPE_CHECKING:
    from typing import Any

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
@tool(
    args_schema=ShellInput,
    handle_tool_error=True,
    handle_validation_error=build_validation_error_handler(ShellInput),
)  # ty:ignore[no-matching-overload]
async def shell_tool(command: str, cwd: str | None = None, _timeout: int = 30) -> ToolResult:
    """
    Execute a shell command and return stdout/stderr.
    Use carefully — only for trusted, sandboxed environments.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_timeout)
        result = ShellOutput(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            returncode=proc.returncode or 0,
            success=proc.returncode == 0,
        )
        return ToolResult.ok(data=result.model_dump())
    except TimeoutError:
        return ToolResult.fail(f"Command timed out after {_timeout}s")
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
@tool(
    args_schema=ReadFileInput,
    handle_tool_error=True,
    handle_validation_error=build_validation_error_handler(ReadFileInput),
)  # ty:ignore[no-matching-overload]
async def read_file(path: str, encoding: str = "utf-8") -> ToolResult:
    """Read the contents of a file and return them as a string."""
    try:
        content = Path(path).read_text(encoding=encoding)
        return ToolResult.ok(data={"content": content})
    except FileNotFoundError:
        return ToolResult.fail(f"File not found: {path}")
    except Exception as exc:  # noqa: BLE001 — file read, unknown encoding/permission errors
        exc.add_note(f"path={path}, operation=read_file")
        return ToolResult.fail(str(exc))


@register_tool("filesystem", "write")
@tool(
    args_schema=WriteFileInput,
    handle_tool_error=True,
    handle_validation_error=build_validation_error_handler(WriteFileInput),
)  # ty:ignore[no-matching-overload]
async def write_file(path: str, content: str, _mode: str = "w") -> ToolResult:
    """Write content to a file. Creates parent directories if needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult.ok(data={"message": f"Written {len(content)} bytes to {path}"})
    except Exception as exc:  # noqa: BLE001 — file write, unknown permission/disk errors
        exc.add_note(f"path={path}, operation=write_file")
        return ToolResult.fail(str(exc))


@register_tool("filesystem", "list")
@tool(
    args_schema=ListDirInput,
    handle_tool_error=True,
    handle_validation_error=build_validation_error_handler(ListDirInput),
)  # ty:ignore[no-matching-overload]
async def list_directory(
    path: str = ".", recursive: bool = False, pattern: str | None = None
) -> ToolResult:
    """List files in a directory, optionally filtering by glob pattern."""
    try:
        base = Path(path)
        if not base.exists():
            return ToolResult.fail(f"Path does not exist: {path}")

        if recursive:
            all_files = [str(p) for p in base.rglob("*") if p.is_file()]
        else:
            all_files = [str(p) for p in base.iterdir()]

        if pattern:
            all_files = [f for f in all_files if fnmatch.fnmatch(Path(f).name, pattern)]

        return ToolResult.ok(data={"files": sorted(all_files)})
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
@tool(
    args_schema=FileSearchInput,
    handle_tool_error=True,
    handle_validation_error=build_validation_error_handler(FileSearchInput),
)  # ty:ignore[no-matching-overload]
async def file_search(
    query: str,
    directory: str = ".",
    file_pattern: str = "*",
    max_results: int = 20,
    use_regex: bool = False,
) -> ToolResult:
    """Search for a string or regex pattern across files in a directory."""
    import re as _re

    base = Path(directory)
    results: list[dict[str, Any]] = []
    pattern = _re.compile(query) if use_regex else None

    for file_path in base.rglob(file_pattern):
        if not file_path.is_file():
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
