"""Build provenance resolution used by health HTTP payloads."""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404 — fixed-argv boot-time provenance read

from app.config import get_settings

__all__ = ["resolve_git_sha"]

_GIT_TIMEOUT_SECONDS = 5


def _read_git_sha() -> str | None:
    """Read one short SHA at import time; failures degrade to ``None``."""
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 — absolute executable, literal argv
            [git_executable, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


_GIT_SHA_CACHE: dict[str, str | None] = {"sha": _read_git_sha()}


def _cached_git_sha() -> str | None:
    """Return the import-time cache; kept monkeypatchable for tests."""
    return _GIT_SHA_CACHE.get("sha")


def resolve_git_sha() -> str:
    """Prefer baked metadata, then the cached working-tree SHA."""
    baked = get_settings().GIT_SHA
    if baked and baked != "unknown":
        return baked
    return _cached_git_sha() or "unknown"
