from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, ClassVar

from beanie import Document, Indexed
from pydantic import BaseModel, Field
from pymongo import ASCENDING, IndexModel

from .tenancy import LegalRole, ProvisioningState


class UserRole(StrEnum):
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"


class Permission(StrEnum):
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"
    CONTENT_READ = "content:read"
    CONTENT_WRITE = "content:write"
    CONTENT_DELETE = "content:delete"
    ADMIN_PANEL = "admin:panel"
    PROFILE_READ = "profile:read"
    PROFILE_WRITE = "profile:write"


# Single source of truth for role → permission mapping.
# ADMIN uses frozenset(Permission) — automatically includes any Permission added later.
ROLE_PERMISSIONS: dict[UserRole, frozenset[Permission]] = {
    UserRole.ADMIN: frozenset(Permission),
    UserRole.MODERATOR: frozenset(
        {
            Permission.USERS_READ,
            Permission.CONTENT_READ,
            Permission.CONTENT_WRITE,
            Permission.CONTENT_DELETE,
            Permission.PROFILE_READ,
            Permission.PROFILE_WRITE,
        }
    ),
    UserRole.USER: frozenset(
        {
            Permission.CONTENT_READ,
            Permission.PROFILE_READ,
            Permission.PROFILE_WRITE,
        }
    ),
}


class OAuthAccount(BaseModel):
    """Embedded OAuth provider account. One record per provider per user."""

    provider: str  # "google" | "github"
    provider_user_id: str
    provider_email: str | None = None
    linked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class User(Document):
    """Primary user document. hashed_password is None for OAuth-only accounts."""

    email: Annotated[str, Indexed(unique=True)]
    hashed_password: str | None = None
    full_name: str | None = None
    avatar_url: str | None = None
    is_verified: bool = Field(default=False)
    is_active: bool = Field(default=True)
    role: UserRole = Field(default=UserRole.USER)
    # Tenant ownership is separate from the platform role.  Existing users
    # are backfilled to ``personal:<user_id>`` by the tenancy migration; the
    # adapter keeps the fallback for documents created during rolling deploys.
    tenant_id: str | None = Field(default=None, min_length=1, max_length=255)
    legal_roles: tuple[LegalRole, ...] = (LegalRole.ANALYST,)
    # Monotonic platform authorization/session generation.  This is deliberately
    # independent from the SQL tenant-membership revision below; a tenant role
    # edit must not silently change the meaning of platform session tokens.
    role_revision: int = Field(default=0, ge=0)
    # SQL ``app_tenant_memberships.role_revision`` is authoritative.  ``None``
    # keeps rolling deploys compatible with Mongo documents written before the
    # mirror field existed; the adapter fills it from the SQL read on sync.
    tenant_role_revision: int | None = Field(default=None, ge=0)
    # Personal-tenant provisioning lifecycle.  New accounts are created with an
    # explicit ``pending`` value; only a successful revision-monotonic mirror
    # sync of the SQL-authoritative membership flips it to ``completed``.
    # Legacy documents default to ``completed`` — the tenancy migration is
    # their repair mechanism, not registration.
    provisioning_state: ProvisioningState = Field(default=ProvisioningState.COMPLETED)
    session_revocation_pending: bool = Field(default=False)
    # Set once when a role-change invalidation begins. It is intentionally
    # separate from updated_at, which profile/account edits may advance while
    # a Redis outage leaves this marker pending.
    session_revocation_pending_at: datetime | None = None
    oauth_accounts: list[OAuthAccount] = Field(default_factory=list)

    # Raw tokens are transmitted via email only; only SHA-256 hashes live here.
    verification_token_hash: str | None = None
    reset_token_hash: str | None = None
    reset_token_expires_at: datetime | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    class Settings:
        name = "users"
        indexes: ClassVar[list[IndexModel]] = [
            IndexModel([("email", ASCENDING)], unique=True),
            IndexModel(
                [
                    ("oauth_accounts.provider", ASCENDING),
                    ("oauth_accounts.provider_user_id", ASCENDING),
                ],
                unique=True,
                sparse=True,
            ),
            IndexModel([("reset_token_hash", ASCENDING)], sparse=True),
            IndexModel([("verification_token_hash", ASCENDING)], sparse=True),
        ]

    def get_permissions(self) -> frozenset[Permission]:
        return ROLE_PERMISSIONS.get(self.role, frozenset())

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.get_permissions()

    def has_role(self, *roles: UserRole) -> bool:
        return self.role in roles
