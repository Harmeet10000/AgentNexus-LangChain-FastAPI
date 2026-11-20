"""Compatibility import for the HTTP adapter's build provenance resolver."""

from __future__ import annotations

from app.features.health.adapters.inbound.http.build_info import (
    _GIT_SHA_CACHE,
    _cached_git_sha,
    resolve_git_sha,
)

__all__ = ["_GIT_SHA_CACHE", "_cached_git_sha", "resolve_git_sha"]
