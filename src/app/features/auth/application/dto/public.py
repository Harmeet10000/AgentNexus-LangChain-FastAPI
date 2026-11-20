"""Immutable DTOs for auth operations consumed by other feature slices.

These models deliberately describe an authentication account without exposing
the persistence representation.  In particular, a password hash is never a
field in this module.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AccountRead(BaseModel):
    """The auth-owned read model for an account."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    email: str
    display_name: str | None = None
    avatar_url: str | None = None
    locale: str | None = None
    role: str
    is_verified: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    permissions: frozenset[str] = frozenset()
    legal_roles: tuple[str, ...] = ()
    role_revision: int = Field(default=0, ge=0)


class ProfileMutation(BaseModel):
    """Fields that may be changed through the auth profile seam.

    ``None`` means that a field was not requested for display name and locale.
    An explicitly supplied ``avatar_url=None`` is retained in
    ``model_fields_set`` so callers can clear an avatar through the port.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    avatar_url: str | None = None
    locale: str | None = Field(default=None, min_length=1, max_length=32)


class PasswordChange(BaseModel):
    """Validated input for changing a password."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)
    revoke_other_sessions: bool = True

    @field_validator("new_password")
    @classmethod
    def password_strong_enough(cls, value: str) -> str:
        if not any(character.isupper() for character in value):
            msg = "Must contain at least one uppercase letter"
            raise ValueError(msg)
        if not any(character.isdigit() for character in value):
            msg = "Must contain at least one digit"
            raise ValueError(msg)
        return value


class SessionRevocation(BaseModel):
    """The sessions invalidated by a revocation operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_ids: tuple[str, ...]


class RoleAdministration(BaseModel):
    """Input for an administrator changing an account role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    account_id: str = Field(min_length=1)
    role: str = Field(min_length=1, max_length=32)
    requesting_admin_id: str = Field(min_length=1)


class StatusAdministration(BaseModel):
    """Input for an administrator changing account activation status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    account_id: str = Field(min_length=1)
    is_active: bool
    requesting_admin_id: str = Field(min_length=1)


class ImpersonationToken(BaseModel):
    """A short-lived access token issued for an administrative impersonation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    access_token: str
    token_type: str = "bearer"  # noqa: S105 — protocol value, not a secret
    expires_in: int = Field(gt=0)
    impersonating_account_id: str


__all__ = [
    "AccountRead",
    "ImpersonationToken",
    "PasswordChange",
    "ProfileMutation",
    "RoleAdministration",
    "SessionRevocation",
    "StatusAdministration",
]
