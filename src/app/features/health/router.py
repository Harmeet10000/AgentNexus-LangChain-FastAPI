"""Compatibility import for routes now owned by the HTTP adapter."""

from __future__ import annotations

import importlib

_implementation = importlib.import_module("app.features.health.adapters.in.http.router")
router = _implementation.router
deep_health_router = _implementation.deep_health_router
ALL_PROBES = _implementation.ALL_PROBES
