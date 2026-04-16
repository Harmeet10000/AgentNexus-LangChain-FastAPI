"""
System tools: shell execution, filesystem operations, file search.

All tools use ToolOutput for structured, parseable responses.
Shell and filesystem tools require explicit capability grants via context.
"""

from __future__ import annotations

import asyncio
import fnmatch
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .base import ToolOutput, build_validation_error_handler, register_tool

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
async def shell_tool(command: str, cwd: str | None = None, timeout: int = 30) -> str:
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
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        result = ShellOutput(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            returncode=proc.returncode or 0,
            success=proc.returncode == 0,
        )
        return result.model_dump_json()
    except TimeoutError:
        return ToolOutput.fail(f"Command timed out after {timeout}s").to_agent_string()
    except Exception as exc:
        return ToolOutput.fail(str(exc)).to_agent_string()


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
async def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read the contents of a file and return them as a string."""
    try:
        content = Path(path).read_text(encoding=encoding)
        return ToolOutput.ok(content).to_agent_string()
    except FileNotFoundError:
        return ToolOutput.fail(f"File not found: {path}").to_agent_string()
    except Exception as exc:
        return ToolOutput.fail(str(exc)).to_agent_string()


@register_tool("filesystem", "write")
@tool(
    args_schema=WriteFileInput,
    handle_tool_error=True,
    handle_validation_error=build_validation_error_handler(WriteFileInput),
)  # ty:ignore[no-matching-overload]
async def write_file(path: str, content: str, mode: str = "w") -> str:
    """Write content to a file. Creates parent directories if needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolOutput.ok(f"Written {len(content)} bytes to {path}").to_agent_string()
    except Exception as exc:
        return ToolOutput.fail(str(exc)).to_agent_string()


@register_tool("filesystem", "list")
@tool(
    args_schema=ListDirInput,
    handle_tool_error=True,
    handle_validation_error=build_validation_error_handler(ListDirInput),
)  # ty:ignore[no-matching-overload]
async def list_directory(
    path: str = ".", recursive: bool = False, pattern: str | None = None
) -> str:
    """List files in a directory, optionally filtering by glob pattern."""
    try:
        base = Path(path)
        if not base.exists():
            return ToolOutput.fail(f"Path does not exist: {path}").to_agent_string()

        if recursive:
            all_files = [str(p) for p in base.rglob("*") if p.is_file()]
        else:
            all_files = [str(p) for p in base.iterdir()]

        if pattern:
            all_files = [f for f in all_files if fnmatch.fnmatch(Path(f).name, pattern)]

        return ToolOutput.ok(sorted(all_files)).to_agent_string()
    except Exception as exc:
        return ToolOutput.fail(str(exc)).to_agent_string()


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
) -> str:
    """Search for a string or regex pattern across files in a directory."""
    import re as _re

    try:
        base = Path(directory)
        results: list[dict] = []
        pattern = _re.compile(query) if use_regex else None

        for file_path in base.rglob(file_pattern):
            if not file_path.is_file():
                continue
            try:
                lines = file_path.read_text(errors="replace").splitlines()
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
            except Exception:
                continue
            if len(results) >= max_results:
                break

        return ToolOutput.ok(results, total=len(results)).to_agent_string()
    except Exception as exc:
        return ToolOutput.fail(str(exc)).to_agent_string()
