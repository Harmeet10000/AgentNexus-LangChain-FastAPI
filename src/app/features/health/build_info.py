"""Compatibility import for the HTTP adapter's build provenance resolver."""

from __future__ import annotations

import importlib

_implementation = importlib.import_module("app.features.health.adapters.in.http.build_info")
_GIT_SHA_CACHE = _implementation._GIT_SHA_CACHE
_cached_git_sha = _implementation._cached_git_sha
resolve_git_sha = _implementation.resolve_git_sha
