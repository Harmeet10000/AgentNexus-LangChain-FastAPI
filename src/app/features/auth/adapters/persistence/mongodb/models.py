"""Beanie persistence models for auth-owned Mongo collections."""

from datetime import UTC, datetime
from typing import Annotated, ClassVar

from beanie import Document, Indexed
from pydantic import Field
from pymongo import ASCENDING, IndexModel

from app.features.auth.domain.tenancy import LegalRole, ProvisioningState
from app.features.auth.domain.user import OAuthAccount, User, UserRole


class UserDocument(Document):
    """Mongo representation of an auth account; never leaves this adapter."""

    email: Annotated[str, Indexed(unique=True)]
    hashed_password: str | None = None
    full_name: str | None = None
    avatar_url: str | None = None
    is_verified: bool = Field(default=False)
    is_active: bool = Field(default=True)
    role: UserRole = Field(default=UserRole.USER)
    tenant_id: str | None = Field(default=None, min_length=1, max_length=255)
    legal_roles: tuple[LegalRole, ...] = (LegalRole.ANALYST,)
    role_revision: int = Field(default=0, ge=0)
    tenant_role_revision: int | None = Field(default=None, ge=0)
    provisioning_state: ProvisioningState = Field(default=ProvisioningState.COMPLETED)
    session_revocation_pending: bool = Field(default=False)
    session_revocation_pending_at: datetime | None = None
    oauth_accounts: list[OAuthAccount] = Field(default_factory=list)
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


def to_domain_user(document: UserDocument) -> User:
    """Detach a Mongo document from Beanie before crossing the adapter seam."""
    return User(
        id=str(document.id),
        email=document.email,
        hashed_password=document.hashed_password,
        full_name=document.full_name,
        avatar_url=document.avatar_url,
        is_verified=document.is_verified,
        is_active=document.is_active,
        role=document.role,
        tenant_id=document.tenant_id,
        legal_roles=document.legal_roles,
        role_revision=document.role_revision,
        tenant_role_revision=document.tenant_role_revision,
        provisioning_state=document.provisioning_state,
        session_revocation_pending=document.session_revocation_pending,
        session_revocation_pending_at=document.session_revocation_pending_at,
        oauth_accounts=[account.model_copy(deep=True) for account in document.oauth_accounts],
        verification_token_hash=document.verification_token_hash,
        reset_token_hash=document.reset_token_hash,
        reset_token_expires_at=document.reset_token_expires_at,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )
