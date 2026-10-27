"""Tenant and legal-role primitives shared by auth and document boundaries.

Platform roles (``admin``, ``moderator``, ``user``) are intentionally kept
separate from legal-workflow roles.  A tenant is the ownership boundary;
``user_id`` remains an audit/uploader identity.
"""

from __future__ import annotations

from enum import StrEnum


class LegalRole(StrEnum):
    """Tenant-scoped legal workflow role."""

    ANALYST = "analyst"
    REVIEWER = "reviewer"
    CORPUS_ADMIN = "corpus_admin"
    # Source compatibility for clients that imported LegalRole.ADMIN.  The
    # value deliberately names the tenant-scoped role so it cannot be
    # confused with ``UserRole.ADMIN`` (platform administration).
    ADMIN = "corpus_admin"


_LEGACY_ROLE_VALUES = {"admin": LegalRole.CORPUS_ADMIN}


def normalize_legal_role(value: LegalRole | str) -> LegalRole:
    """Parse current roles while accepting legacy Mongo ``admin`` values.

    Legacy values are accepted only at this explicit compatibility boundary;
    all SQL writes use the canonical ``corpus_admin`` value.
    """

    if isinstance(value, LegalRole):
        return value
    legacy = _LEGACY_ROLE_VALUES.get(value)
    return legacy if legacy is not None else LegalRole(value)


def personal_tenant_id(user_id: str) -> str:
    """Return the stable tenant used by the one-user backfill.

    The prefix avoids confusing a legacy user id with a tenant id when both
    appear in logs, cache keys, Graphiti groups, or migration diagnostics.
    """

    if not user_id:
        message = "user_id must not be empty"
        raise ValueError(message)
    return f"personal:{user_id}"
