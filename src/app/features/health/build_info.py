"""Build provenance resolution for the health payloads (smoke F3/F6, git_sha).

A docker build bakes ``GIT_SHA`` via ``--build-arg`` (prod.Dockerfile, Makefile
``image-build*`` targets). Native processes — the systemd-run deployments the
2026-11-15 smoke ran — have no build pipeline, and every health payload then
reported ``git_sha: "unknown"``, making the served commit unreconstructable
while review agents were switching branches underneath the running service.

Resolution order, evaluated once per process:

1. baked build metadata (``settings.GIT_SHA`` from the environment);
2. the working tree's short SHA, read from git **at import time** and cached —
   never inside the request path;
3. ``"unknown"`` when neither source is available (git missing, not a repo).
"""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404 — one fixed-argv boot-time provenance read; no shell

from app.config import get_settings

__all__ = ["resolve_git_sha"]

_GIT_TIMEOUT_SECONDS = 5


def _read_git_sha() -> str | None:
    """One git read per process; any failure degrades to ``None``."""
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


_GIT_SHA_CACHE: dict[str, str | None] = {}
# Warmed at import: the subprocess runs once at boot, never in the request path.
_GIT_SHA_CACHE["sha"] = _read_git_sha()


def _cached_git_sha() -> str | None:
    """Import-time cache seam, kept monkeypatchable for tests."""
    return _GIT_SHA_CACHE.get("sha")


def resolve_git_sha() -> str:
    """Baked build metadata first, then the working tree, then ``unknown``."""
    baked = get_settings().GIT_SHA
    if baked and baked != "unknown":
        return baked
    return _cached_git_sha() or "unknown"
