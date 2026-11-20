"""Compatibility import for routes now owned by the HTTP adapter."""

from __future__ import annotations

from app.features.health.adapters.inbound.http.router import (
    ALL_PROBES,
    deep_health_router,
    router,
)

__all__ = ["ALL_PROBES", "deep_health_router", "router"]
