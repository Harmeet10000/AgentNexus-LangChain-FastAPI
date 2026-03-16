from datetime import datetime
from enum import StrEnum
from typing import Annotated

from beanie import Document, Indexed
from pydantic import BaseModel, Field
from pymongo import ASCENDING, IndexModel


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
    linked_at: datetime = Field(default_factory=datetime.utcnow)


class User(Document):
    """Primary user document. hashed_password is None for OAuth-only accounts."""

    email: Annotated[str, Indexed(unique=True)]
    hashed_password: str | None = None
    full_name: str | None = None
    is_verified: bool = Field(default=False)
    is_active: bool = Field(default=True)
    role: UserRole = Field(default=UserRole.USER)
    oauth_accounts: list[OAuthAccount] = Field(default_factory=list)

    # Raw tokens are transmitted via email only; only SHA-256 hashes live here.
    verification_token_hash: str | None = None
    reset_token_hash: str | None = None
    reset_token_expires_at: datetime | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "users"
        indexes = [
            IndexModel([("email", ASCENDING)], unique=True),
            IndexModel([("reset_token_hash", ASCENDING)], sparse=True),
            IndexModel([("verification_token_hash", ASCENDING)], sparse=True),
        ]

    def get_permissions(self) -> frozenset[Permission]:
        return ROLE_PERMISSIONS.get(self.role, frozenset())

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.get_permissions()

    def has_role(self, *roles: UserRole) -> bool:
        return self.role in roles
