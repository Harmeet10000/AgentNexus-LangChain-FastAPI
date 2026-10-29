"""Tenancy primitives that must stay importable from the pure kernel layer.

``events`` relay code runs below the features layer (connections → events →
relay), so tenant-id construction cannot live behind a feature import without
breaking the layering contracts. The canonical definition lives here;
``features.auth.domain.tenancy`` re-exports it for feature-internal callers.
"""

from __future__ import annotations

_PERSONAL_TENANT_PREFIX = "personal:"


def personal_tenant_id(user_id: str) -> str:
    """Return the stable tenant used by the one-user backfill.

    The prefix avoids confusing a legacy user id with a tenant id when both
    appear in logs, cache keys, Graphiti groups, or migration diagnostics.
    """
    return f"{_PERSONAL_TENANT_PREFIX}{user_id}"
