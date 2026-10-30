"""Tenant and legal-role primitives shared by auth and document boundaries.

Platform roles (``admin``, ``moderator``, ``user``) are intentionally kept
separate from legal-workflow roles.  A tenant is the ownership boundary;
``user_id`` remains an audit/uploader identity.
"""

from __future__ import annotations

from enum import StrEnum

from app.kernel.tenancy import (
    personal_tenant_id,  # re-export: single canonical definition lives in the kernel
)


class LegalRole(StrEnum):
    """Tenant-scoped legal workflow role."""

    ANALYST = "analyst"
    REVIEWER = "reviewer"
    CORPUS_ADMIN = "corpus_admin"


# Note: the former ``ADMIN = "corpus_admin"`` enum alias is deliberately gone.
# It made Pydantic emit "corpus_admin" twice in the OpenAPI schema, and no
# caller ever imported it — the legacy wire value "admin" is handled by
# ``normalize_legal_role`` below.  Platform administration is ``UserRole.ADMIN``.


_LEGACY_ROLE_VALUES = {"admin": LegalRole.CORPUS_ADMIN}


class ProvisioningState(StrEnum):
    """Account lifecycle for personal-tenant provisioning.

    ``PENDING`` is written explicitly when the Mongo account is created; SQL is
    the provisioning authority, and the state only flips to ``COMPLETED`` when a
    revision-monotonic mirror sync confirms the SQL membership.  Legacy accounts
    default to ``COMPLETED``: the tenancy migration is their repair mechanism.
    """

    PENDING = "pending"
    COMPLETED = "completed"


def normalize_legal_role(value: LegalRole | str) -> LegalRole:
    """Parse current roles while accepting legacy Mongo ``admin`` values.

    Legacy values are accepted only at this explicit compatibility boundary;
    all SQL writes use the canonical ``corpus_admin`` value.
    """

    if isinstance(value, LegalRole):
        return value
    legacy = _LEGACY_ROLE_VALUES.get(value)
    return legacy if legacy is not None else LegalRole(value)


__all__ = [
    "LegalRole",
    "ProvisioningState",
    "normalize_legal_role",
    "personal_tenant_id",
]
