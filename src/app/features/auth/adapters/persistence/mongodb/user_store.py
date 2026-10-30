"""Auth account persistence port and adapters.

This module defines the real persistence seam for authentication:
- AuthAccount: immutable domain model (no Beanie Document)
- AuthAccountStore: Protocol defining the contract
- BeanieAuthAccountStore: production MongoDB adapter
- InMemoryAuthAccountStore: fast test adapter
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from bson import ObjectId as BsonObjectId
from pydantic import BaseModel, ConfigDict
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError
from returns.result import Failure, Success

if TYPE_CHECKING:
    from returns.result import Result

    from app.features.auth.domain.errors import AuthError

from app.features.auth.domain.errors import (
    AuthConflictError,
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthValidationError,
)
from app.features.auth.domain.tenancy import LegalRole, ProvisioningState
from app.features.auth.domain.user import (
    ROLE_PERMISSIONS,
    OAuthAccount,
    User,
    UserRole,
)


class OAuthAccountInput(BaseModel):
    """Provider identity to attach while creating or linking an account."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    provider_user_id: str
    provider_email: str | None = None


class AuthAccount(BaseModel):
    """Immutable authentication-owned account state.

    Returned by the account store; never a Beanie Document.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    email: str
    full_name: str | None
    password_hash: str | None
    is_active: bool
    is_verified: bool
    role: UserRole
    tenant_id: str = ""
    legal_roles: tuple[LegalRole, ...] = (LegalRole.ANALYST,)
    permissions: frozenset[str]
    # Platform/session authorization generation.  Kept separate from the SQL
    # tenant membership generation below.
    role_revision: int = 0
    tenant_role_revision: int = 0
    # Personal-tenant provisioning lifecycle.  New accounts are written with an
    # explicit ``pending``; only a revision-monotonic mirror sync completes it.
    provisioning_state: ProvisioningState = ProvisioningState.COMPLETED
    session_revocation_pending: bool = False
    session_revocation_pending_at: datetime | None = None
    avatar_url: str | None = None
    oauth_accounts: tuple[OAuthAccountInput, ...] = ()
    oauth_providers: tuple[str, ...] = ()
    verification_token_hash: str | None = None
    reset_token_hash: str | None = None
    reset_token_expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class NewAuthAccount(BaseModel):
    """Input for creating a new account."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    email: str
    password_hash: str | None
    full_name: str | None = None
    role: UserRole = UserRole.USER
    tenant_id: str | None = None
    legal_roles: tuple[LegalRole, ...] = (LegalRole.ANALYST,)
    tenant_role_revision: int = 0
    # Explicit at every account-creation call site: registration and OAuth
    # first-login pass ``pending``; defaults keep legacy helpers truthful.
    provisioning_state: ProvisioningState = ProvisioningState.COMPLETED
    is_verified: bool = False
    is_active: bool = True
    verification_token_hash: str | None = None
    oauth_account: OAuthAccountInput | None = None


class AuthAccountStore(Protocol):
    """Real persistence seam for authentication accounts.

    Two adapters implement this:
    - BeanieAuthAccountStore: production MongoDB
    - InMemoryAuthAccountStore: fast behavioral tests
    """

    async def find_by_email(self, email: str) -> Result[AuthAccount | None, AuthError]:
        """Find account by email.

        Returns:
            - Success(account) if found
            - Success(None) if not found (maps to "invalid credentials")
            - Failure(AuthInfrastructureError) on DB error
        """
        ...

    async def find_by_id(self, account_id: str) -> Result[AuthAccount | None, AuthError]:
        """Find account by ID.

        Returns:
            - Success(account) if found
            - Success(None) if not found
            - Failure(AuthInfrastructureError) on DB error
        """
        ...

    async def find_by_verification_token_hash(
        self, token_hash: str
    ) -> Result[AuthAccount | None, AuthError]:
        """Find account by verification token hash."""
        ...

    async def find_by_reset_token_hash(
        self, token_hash: str
    ) -> Result[AuthAccount | None, AuthError]:
        """Find account by password reset token hash."""
        ...

    async def create(self, account: NewAuthAccount) -> Result[AuthAccount, AuthError]:
        """Create a new account.

        Returns:
            - Success(AuthAccount) on success
            - Failure(AuthConflictError) if email exists
            - Failure(AuthInfrastructureError) on DB error
        """
        ...

    async def replace_password_hash(
        self, account_id: str, password_hash: str
    ) -> Result[None, AuthError]:
        """Replace password hash (transparent rehash on login)."""
        ...

    async def mark_email_verified(self, account_id: str) -> Result[None, AuthError]:
        """Mark email as verified and clear verification token."""
        ...

    async def set_verification_token_hash(
        self, account_id: str, token_hash: str
    ) -> Result[None, AuthError]:
        """Set new verification token hash (resend verification)."""
        ...

    async def set_reset_token(
        self,
        account_id: str,
        token_hash: str,
        expires_at: datetime,
    ) -> Result[None, AuthError]:
        """Set password reset token with expiry."""
        ...

    async def clear_reset_token(self, account_id: str) -> Result[None, AuthError]:
        """Clear password reset token after successful reset."""
        ...

    async def consume_reset_token_and_replace_password(
        self,
        account_id: str,
        expected_token_hash: str,
        new_password_hash: str,
    ) -> Result[None, AuthError]:
        """Atomically consume a valid reset token and replace the password.

        Single atomic operation: validates token hash and expiry, then
        replaces password hash and clears reset token state.
        """
        ...

    async def email_exists(self, email: str) -> Result[bool, AuthError]:
        """Check if email is already registered."""
        ...

    async def update_profile(
        self, account_id: str, *, full_name: str
    ) -> Result[AuthAccount, AuthError]:
        """Persist profile-owned fields and return the refreshed account."""
        ...

    async def set_avatar_url(
        self, account_id: str, avatar_url: str | None
    ) -> Result[None, AuthError]:
        """Persist the public avatar URL for an account."""
        ...

    async def link_oauth_account(
        self,
        account_id: str,
        provider: str,
        provider_user_id: str,
        provider_email: str | None,
    ) -> Result[None, AuthError]:
        """Link one provider identity to an existing account.

        A provider may occur at most once per account. Repeating the exact
        provider/user-id pair is idempotent; attempting to replace that
        provider's identity returns ``AuthConflictError``.
        """
        ...

    async def find_by_oauth_identity(
        self, provider: str, provider_user_id: str
    ) -> Result[AuthAccount | None, AuthError]:
        """Find the account that owns one provider subject, if any."""
        ...

    async def unlink_oauth_account(self, account_id: str, provider: str) -> Result[None, AuthError]:
        """Remove a provider identity when another sign-in method remains."""
        ...

    async def set_tenant_binding(
        self,
        account_id: str,
        *,
        tenant_id: str,
        legal_roles: tuple[LegalRole, ...],
        tenant_role_revision: int | None = None,
        role_revision: int | None = None,
    ) -> Result[None, AuthError]:
        """Synchronize the SQL-authoritative tenant membership mirror.

        Revision-monotonic compare-and-set: a mirror write whose tenant
        revision is older than the currently mirrored one is rejected so a
        stale writer can never overwrite newer authorization state.  A
        successful write also completes the account's provisioning lifecycle.
        """
        ...


# Implementation below (not in TYPE_CHECKING to avoid circular imports)
# -----------------------------------------------------------------------------


class BeanieAuthAccountStore:
    """Production MongoDB/Beanie adapter for AuthAccountStore."""

    def __init__(self) -> None:
        self._User = User

    @staticmethod
    def _to_auth_account(user: User) -> AuthAccount:
        """Convert Beanie User document to immutable AuthAccount."""
        tenant_role_revision = getattr(user, "tenant_role_revision", None)
        if tenant_role_revision is None:
            # Rolling-deploy compatibility for documents without the mirror
            # field.  The next SQL-authoritative sync replaces this value.
            tenant_role_revision = user.role_revision
        return AuthAccount(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            avatar_url=user.avatar_url,
            password_hash=user.hashed_password,
            is_active=user.is_active,
            is_verified=user.is_verified,
            role=user.role,
            # A missing tenant is an authorization failure until the explicit
            # tenant migration/reconciliation adapter repairs it.  Never mint
            # a production tenant from an auth read.
            tenant_id=user.tenant_id or "",
            legal_roles=user.legal_roles,
            permissions=frozenset(p.value for p in user.get_permissions()),
            role_revision=user.role_revision,
            # ``None`` means this Mongo document predates the mirror field.
            # Preserve the legacy value until SQL reconciliation supplies the
            # authoritative tenant revision; do not use it to advance the
            # platform/session revision on a tenant-binding write.
            tenant_role_revision=tenant_role_revision,
            provisioning_state=user.provisioning_state,
            session_revocation_pending=user.session_revocation_pending,
            session_revocation_pending_at=user.session_revocation_pending_at,
            verification_token_hash=user.verification_token_hash,
            reset_token_hash=user.reset_token_hash,
            reset_token_expires_at=user.reset_token_expires_at,
            created_at=user.created_at,
            updated_at=user.updated_at,
            oauth_providers=tuple(account.provider for account in user.oauth_accounts),
            oauth_accounts=tuple(
                OAuthAccountInput(
                    provider=account.provider,
                    provider_user_id=account.provider_user_id,
                    provider_email=account.provider_email,
                )
                for account in user.oauth_accounts
            ),
        )

    async def find_by_email(self, email: str) -> Result[AuthAccount | None, AuthError]:
        try:
            user = await self._User.find_one(self._User.email == email.lower())
            if user is None:
                return Success(None)
            return Success(self._to_auth_account(user))
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=find_by_email")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by email",
                    details={"email": email, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def find_by_id(self, account_id: str) -> Result[AuthAccount | None, AuthError]:
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        try:
            user = await self._User.get(BsonObjectId(account_id))
            if user is None:
                return Success(None)
            return Success(self._to_auth_account(user))
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=find_by_id")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by ID",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def find_by_verification_token_hash(
        self, token_hash: str
    ) -> Result[AuthAccount | None, AuthError]:
        try:
            user = await self._User.find_one(self._User.verification_token_hash == token_hash)
            if user is None:
                return Success(None)
            return Success(self._to_auth_account(user))
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=find_by_verification_token_hash")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by verification token",
                    details={"token_hash": token_hash, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def find_by_reset_token_hash(
        self, token_hash: str
    ) -> Result[AuthAccount | None, AuthError]:
        try:
            user = await self._User.find_one(self._User.reset_token_hash == token_hash)
            if user is None:
                return Success(None)
            return Success(self._to_auth_account(user))
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=find_by_reset_token_hash")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user by reset token",
                    details={"token_hash": token_hash, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def create(self, account: NewAuthAccount) -> Result[AuthAccount, AuthError]:
        try:
            user = self._User(
                email=account.email.lower(),
                hashed_password=account.password_hash,
                full_name=account.full_name,
                role=account.role,
                tenant_id=account.tenant_id,
                legal_roles=account.legal_roles,
                tenant_role_revision=account.tenant_role_revision,
                provisioning_state=account.provisioning_state,
                is_verified=account.is_verified,
                is_active=account.is_active,
                verification_token_hash=account.verification_token_hash,
                oauth_accounts=(
                    [
                        OAuthAccount(
                            provider=account.oauth_account.provider,
                            provider_user_id=account.oauth_account.provider_user_id,
                            provider_email=account.oauth_account.provider_email,
                        )
                    ]
                    if account.oauth_account is not None
                    else []
                ),
            )
            created = await user.insert()
            return Success(self._to_auth_account(created))
        except DuplicateKeyError as exc:
            exc.add_note("auth_account_store operation=create duplicate_email")
            return Failure(
                AuthConflictError(
                    message="User with this email already exists",
                    details={"email": account.email, "error": str(exc)},
                    source="auth_account_store",
                )
            )
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=create")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while creating user",
                    details={"email": account.email, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def replace_password_hash(
        self, account_id: str, password_hash: str
    ) -> Result[None, AuthError]:
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        try:
            user = await self._User.get(BsonObjectId(account_id))
            if user is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found",
                        details={"user_id": account_id},
                        source="auth_account_store",
                    )
                )
            user.hashed_password = password_hash
            user.updated_at = datetime.now(UTC)
            await user.save()
            return Success(None)
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=replace_password_hash")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while updating password hash",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def mark_email_verified(self, account_id: str) -> Result[None, AuthError]:
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        try:
            user = await self._User.get(BsonObjectId(account_id))
            if user is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found",
                        details={"user_id": account_id},
                        source="auth_account_store",
                    )
                )
            user.is_verified = True
            user.verification_token_hash = None
            user.updated_at = datetime.now(UTC)
            await user.save()
            return Success(None)
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=mark_email_verified")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while marking email verified",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def set_verification_token_hash(
        self, account_id: str, token_hash: str
    ) -> Result[None, AuthError]:
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        try:
            user = await self._User.get(BsonObjectId(account_id))
            if user is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found",
                        details={"user_id": account_id},
                        source="auth_account_store",
                    )
                )
            user.verification_token_hash = token_hash
            user.updated_at = datetime.now(UTC)
            await user.save()
            return Success(None)
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=set_verification_token_hash")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while setting verification token",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def set_reset_token(
        self,
        account_id: str,
        token_hash: str,
        expires_at: datetime,
    ) -> Result[None, AuthError]:
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        try:
            user = await self._User.get(BsonObjectId(account_id))
            if user is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found",
                        details={"user_id": account_id},
                        source="auth_account_store",
                    )
                )
            user.reset_token_hash = token_hash
            user.reset_token_expires_at = expires_at
            user.updated_at = datetime.now(UTC)
            await user.save()
            return Success(None)
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=set_reset_token")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while setting reset token",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def clear_reset_token(self, account_id: str) -> Result[None, AuthError]:
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        try:
            user = await self._User.get(BsonObjectId(account_id))
            if user is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found",
                        details={"user_id": account_id},
                        source="auth_account_store",
                    )
                )
            user.reset_token_hash = None
            user.reset_token_expires_at = None
            user.updated_at = datetime.now(UTC)
            await user.save()
            return Success(None)
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=clear_reset_token")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while clearing reset token",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def consume_reset_token_and_replace_password(
        self,
        account_id: str,
        expected_token_hash: str,
        new_password_hash: str,
    ) -> Result[None, AuthError]:
        """Atomically consume a valid reset token and replace the password.

        Single atomic operation: validates token hash and expiry, then
        replaces password hash and clears reset token state.
        """
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        try:
            now = datetime.now(UTC)
            updated = await self._User.get_pymongo_collection().find_one_and_update(
                {
                    "_id": BsonObjectId(account_id),
                    "reset_token_hash": expected_token_hash,
                    "reset_token_expires_at": {"$gt": now},
                },
                {
                    "$set": {
                        "hashed_password": new_password_hash,
                        "reset_token_hash": None,
                        "reset_token_expires_at": None,
                        "updated_at": now,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if updated is None:
                return Failure(
                    AuthNotFoundError(
                        message="Invalid or expired reset token",
                        details={"user_id": account_id},
                        source="auth_account_store",
                    )
                )
            return Success(None)
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=consume_reset_token_and_replace_password")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while consuming reset token and replacing password",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def email_exists(self, email: str) -> Result[bool, AuthError]:
        try:
            count = await self._User.find(self._User.email == email.lower()).count()
            return Success(count > 0)
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=email_exists")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while checking email existence",
                    details={"email": email, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def update_profile(
        self, account_id: str, *, full_name: str
    ) -> Result[AuthAccount, AuthError]:
        user_result = await self._get_user(account_id)
        if isinstance(user_result, Failure):
            return Failure(user_result.failure())
        user = user_result.unwrap()
        user.full_name = full_name
        user.updated_at = datetime.now(UTC)
        try:
            await user.save()
            return Success(self._to_auth_account(user))
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=update_profile")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while updating profile",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def set_avatar_url(
        self, account_id: str, avatar_url: str | None
    ) -> Result[None, AuthError]:
        user_result = await self._get_user(account_id)
        if isinstance(user_result, Failure):
            return Failure(user_result.failure())
        user = user_result.unwrap()
        user.avatar_url = avatar_url
        user.updated_at = datetime.now(UTC)
        try:
            await user.save()
            return Success(None)
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=set_avatar_url")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while updating avatar",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def set_tenant_binding(
        self,
        account_id: str,
        *,
        tenant_id: str,
        legal_roles: tuple[LegalRole, ...],
        tenant_role_revision: int | None = None,
        role_revision: int | None = None,
    ) -> Result[None, AuthError]:
        revision = tenant_role_revision if tenant_role_revision is not None else role_revision
        if revision is None:
            message = "tenant_role_revision is required"
            raise ValueError(message)
        # Revision-monotonic compare-and-set executed atomically by MongoDB.
        # Documents predating the mirror field (missing/None revision) always
        # accept their first authoritative write.  A stale writer loses.
        now = datetime.now(UTC)
        try:
            updated = await self._User.get_pymongo_collection().find_one_and_update(
                {
                    "_id": BsonObjectId(account_id),
                    "$or": [
                        {"tenant_role_revision": {"$exists": False}},
                        {"tenant_role_revision": None},
                        {"tenant_role_revision": {"$lte": revision}},
                    ],
                },
                {
                    "$set": {
                        "tenant_id": tenant_id,
                        "legal_roles": [role.value for role in legal_roles],
                        # SQL membership revision is authoritative for tenant
                        # roles.  It must not be folded into ``role_revision``,
                        # which fences platform authorization and session tokens.
                        "tenant_role_revision": revision,
                        "provisioning_state": ProvisioningState.COMPLETED.value,
                        "updated_at": now,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=set_tenant_binding")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while synchronizing tenant membership",
                    details={"user_id": account_id, "operation": "set_tenant_binding"},
                    source="auth_account_store",
                )
            )
        if updated is not None:
            return Success(None)

        # The conditional update matched nothing: either the account is gone or
        # a newer mirror revision already won.  Disambiguate with one read.
        user_result = await self._get_user(account_id)
        if isinstance(user_result, Failure):
            return Failure(user_result.failure())
        current = user_result.unwrap()
        mirrored = getattr(current, "tenant_role_revision", None)
        if mirrored is not None and mirrored > revision:
            return Failure(
                AuthConflictError(
                    message="Stale tenant mirror revision rejected by compare-and-set",
                    details={
                        "user_id": account_id,
                        "operation": "set_tenant_binding",
                        "expected_revision": revision,
                        "current_revision": mirrored,
                    },
                    source="auth_account_store",
                )
            )
        return Success(None)

    async def link_oauth_account(
        self,
        account_id: str,
        provider: str,
        provider_user_id: str,
        provider_email: str | None,
    ) -> Result[None, AuthError]:
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )

        try:
            user = await self._User.get(BsonObjectId(account_id))
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=link_oauth_account_lookup")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user for OAuth link",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )
        if user is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )

        linked_for_provider = next(
            (linked for linked in user.oauth_accounts if linked.provider == provider),
            None,
        )
        if linked_for_provider is not None:
            if linked_for_provider.provider_user_id != provider_user_id:
                return Failure(
                    AuthConflictError(
                        message="OAuth provider is already linked to another identity",
                        details={"provider": provider},
                        source="auth_account_store",
                    )
                )
            return Success(None)

        now = datetime.now(UTC)
        try:
            updated = await self._User.get_pymongo_collection().find_one_and_update(
                {
                    "_id": BsonObjectId(account_id),
                    "oauth_accounts": {
                        "$not": {"$elemMatch": {"provider": provider}},
                    },
                },
                {
                    "$push": {
                        "oauth_accounts": OAuthAccount(
                            provider=provider,
                            provider_user_id=provider_user_id,
                            provider_email=provider_email,
                        ).model_dump(mode="python")
                    },
                    "$set": {"updated_at": now},
                },
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError as exc:
            exc.add_note("auth_account_store operation=link_oauth_account duplicate_identity")
            return Failure(
                AuthConflictError(
                    message="OAuth identity is already linked to another account",
                    details={"provider": provider},
                    source="auth_account_store",
                )
            )
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=link_oauth_account")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while linking OAuth account",
                    details={"user_id": account_id, "provider": provider, "error": str(exc)},
                    source="auth_account_store",
                )
            )

        if updated is None:
            # Another request won the conditional update. Resolve whether it
            # linked this exact identity (idempotent) or a conflicting one.
            raced_result = await self._get_user(account_id)
            if isinstance(raced_result, Failure):
                return Failure(raced_result.failure())
            raced_user = raced_result.unwrap()
            raced_link = next(
                (linked for linked in raced_user.oauth_accounts if linked.provider == provider),
                None,
            )
            if raced_link is not None and raced_link.provider_user_id == provider_user_id:
                return Success(None)
            return Failure(
                AuthConflictError(
                    message="OAuth provider is already linked to another identity",
                    details={"provider": provider},
                    source="auth_account_store",
                )
            )
        return Success(None)

    async def find_by_oauth_identity(
        self, provider: str, provider_user_id: str
    ) -> Result[AuthAccount | None, AuthError]:
        try:
            user = await self._User.find_one(
                {
                    "oauth_accounts": {
                        "$elemMatch": {
                            "provider": provider,
                            "provider_user_id": provider_user_id,
                        }
                    }
                }
            )
            return Success(None if user is None else self._to_auth_account(user))
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=find_by_oauth_identity")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding OAuth identity",
                    details={"provider": provider, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def unlink_oauth_account(self, account_id: str, provider: str) -> Result[None, AuthError]:
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        try:
            user = await self._User.get(BsonObjectId(account_id))
            if user is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found",
                        details={"user_id": account_id},
                        source="auth_account_store",
                    )
                )
            updated = await self._User.get_pymongo_collection().find_one_and_update(
                {
                    "_id": BsonObjectId(account_id),
                    "oauth_accounts": {
                        "$elemMatch": {"provider": provider},
                    },
                    "$or": [
                        {"hashed_password": {"$ne": None}},
                        {"oauth_accounts": {"$elemMatch": {"provider": {"$ne": provider}}}},
                    ],
                },
                {
                    "$pull": {"oauth_accounts": {"provider": provider}},
                    "$set": {"updated_at": datetime.now(UTC)},
                },
                return_document=ReturnDocument.AFTER,
            )
            if updated is not None:
                return Success(None)

            # The conditional update is the ownership/last-method guard.  A
            # second read only classifies a failed update; it is deliberately
            # not used as a precondition, so concurrent unlink requests cannot
            # both pass a read-then-write check.
            current = await self._User.get(BsonObjectId(account_id))
            if current is None:
                return Failure(
                    AuthNotFoundError(
                        message="User not found",
                        details={"user_id": account_id},
                        source="auth_account_store",
                    )
                )
            if not any(linked.provider == provider for linked in current.oauth_accounts):
                return Success(None)
            return Failure(
                AuthConflictError(
                    message="Cannot remove the last sign-in method",
                    details={"provider": provider},
                    source="auth_account_store",
                )
            )
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=unlink_oauth_account")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while unlinking OAuth account",
                    details={"user_id": account_id, "provider": provider, "error": str(exc)},
                    source="auth_account_store",
                )
            )

    async def _get_user(self, account_id: str) -> Result[User, AuthError]:
        if not BsonObjectId.is_valid(account_id):
            return Failure(
                AuthValidationError(
                    message="Invalid user identifier",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        try:
            user = await self._User.get(BsonObjectId(account_id))
        except PyMongoError as exc:
            exc.add_note("auth_account_store operation=find_user")
            return Failure(
                AuthInfrastructureError(
                    message="Database error while finding user",
                    details={"user_id": account_id, "error": str(exc)},
                    source="auth_account_store",
                )
            )
        if user is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        return Success(user)


class InMemoryAuthAccountStore:
    """In-memory adapter for fast behavioral tests.

    No MongoDB required. All operations are synchronous-speed async.
    """

    def __init__(self) -> None:
        self._accounts: dict[str, AuthAccount] = {}  # id -> account
        self._by_email: dict[str, str] = {}  # email.lower() -> id
        self._by_verification_token: dict[str, str] = {}  # token_hash -> id
        self._by_reset_token: dict[str, str] = {}  # token_hash -> id

    @staticmethod
    def _next_id() -> str:
        return str(BsonObjectId())

    def _index_account(self, account: AuthAccount) -> None:
        self._accounts[account.id] = account
        self._by_email[account.email.lower()] = account.id
        if account.verification_token_hash:
            self._by_verification_token[account.verification_token_hash] = account.id
        if account.reset_token_hash:
            self._by_reset_token[account.reset_token_hash] = account.id

    def _unindex_account(self, account: AuthAccount) -> None:
        self._accounts.pop(account.id, None)
        self._by_email.pop(account.email.lower(), None)
        if account.verification_token_hash:
            self._by_verification_token.pop(account.verification_token_hash, None)
        if account.reset_token_hash:
            self._by_reset_token.pop(account.reset_token_hash, None)

    async def find_by_email(self, email: str) -> Result[AuthAccount | None, AuthError]:
        account_id = self._by_email.get(email.lower())
        if account_id is None:
            return Success(None)
        return Success(self._accounts[account_id])

    async def find_by_id(self, account_id: str) -> Result[AuthAccount | None, AuthError]:
        account = self._accounts.get(account_id)
        return Success(account)

    async def find_by_verification_token_hash(
        self, token_hash: str
    ) -> Result[AuthAccount | None, AuthError]:
        account_id = self._by_verification_token.get(token_hash)
        if account_id is None:
            return Success(None)
        return Success(self._accounts[account_id])

    async def find_by_reset_token_hash(
        self, token_hash: str
    ) -> Result[AuthAccount | None, AuthError]:
        account_id = self._by_reset_token.get(token_hash)
        if account_id is None:
            return Success(None)
        return Success(self._accounts[account_id])

    async def create(self, account: NewAuthAccount) -> Result[AuthAccount, AuthError]:
        if account.email.lower() in self._by_email:
            return Failure(
                AuthConflictError(
                    message="User with this email already exists",
                    details={"email": account.email},
                    source="auth_account_store",
                )
            )

        now = datetime.now(UTC)
        account_id = self._next_id()
        new_account = AuthAccount(
            id=account_id,
            email=account.email.lower(),
            full_name=account.full_name,
            password_hash=account.password_hash,
            is_active=account.is_active,
            is_verified=account.is_verified,
            role=account.role,
            tenant_id=account.tenant_id or "",
            legal_roles=account.legal_roles,
            tenant_role_revision=account.tenant_role_revision,
            provisioning_state=account.provisioning_state,
            permissions=frozenset(p.value for p in ROLE_PERMISSIONS[account.role]),
            oauth_accounts=((account.oauth_account,) if account.oauth_account is not None else ()),
            oauth_providers=(
                (account.oauth_account.provider,) if account.oauth_account is not None else ()
            ),
            verification_token_hash=account.verification_token_hash,
            created_at=now,
            updated_at=now,
        )
        self._index_account(new_account)
        return Success(new_account)

    async def replace_password_hash(
        self, account_id: str, password_hash: str
    ) -> Result[None, AuthError]:
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        self._unindex_account(account)
        data = account.model_dump()
        data.pop("password_hash", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            password_hash=password_hash,
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)

    async def find_by_oauth_identity(
        self, provider: str, provider_user_id: str
    ) -> Result[AuthAccount | None, AuthError]:
        for account in self._accounts.values():
            if any(
                linked.provider == provider and linked.provider_user_id == provider_user_id
                for linked in account.oauth_accounts
            ):
                return Success(account)
        return Success(None)

    async def unlink_oauth_account(self, account_id: str, provider: str) -> Result[None, AuthError]:
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        if account.password_hash is None and not any(
            linked.provider != provider for linked in account.oauth_accounts
        ):
            return Failure(
                AuthConflictError(
                    message="Cannot remove the last sign-in method",
                    details={"provider": provider},
                    source="auth_account_store",
                )
            )
        if not any(linked.provider == provider for linked in account.oauth_accounts):
            return Success(None)
        self._unindex_account(account)
        data = account.model_dump()
        data.pop("oauth_accounts", None)
        data.pop("oauth_providers", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            oauth_providers=tuple(p for p in account.oauth_providers if p != provider),
            oauth_accounts=tuple(
                linked for linked in account.oauth_accounts if linked.provider != provider
            ),
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)

    async def mark_email_verified(self, account_id: str) -> Result[None, AuthError]:
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        self._unindex_account(account)
        data = account.model_dump()
        data.pop("is_verified", None)
        data.pop("verification_token_hash", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            is_verified=True,
            verification_token_hash=None,
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)

    async def set_verification_token_hash(
        self, account_id: str, token_hash: str
    ) -> Result[None, AuthError]:
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        self._unindex_account(account)
        updated = AuthAccount(
            **account.model_dump(),
            verification_token_hash=token_hash,
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)

    async def set_reset_token(
        self,
        account_id: str,
        token_hash: str,
        expires_at: datetime,
    ) -> Result[None, AuthError]:
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        self._unindex_account(account)
        data = account.model_dump()
        # Remove fields that will be explicitly set
        data.pop("reset_token_hash", None)
        data.pop("reset_token_expires_at", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            reset_token_hash=token_hash,
            reset_token_expires_at=expires_at,
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)

    async def clear_reset_token(self, account_id: str) -> Result[None, AuthError]:
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        self._unindex_account(account)
        data = account.model_dump()
        # Remove fields that will be explicitly set
        data.pop("reset_token_hash", None)
        data.pop("reset_token_expires_at", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            reset_token_hash=None,
            reset_token_expires_at=None,
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)

    async def consume_reset_token_and_replace_password(
        self,
        account_id: str,
        expected_token_hash: str,
        new_password_hash: str,
    ) -> Result[None, AuthError]:
        """Atomically consume a valid reset token and replace the password."""
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        if account.reset_token_hash != expected_token_hash:
            return Failure(
                AuthNotFoundError(
                    message="Invalid or expired reset token",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        if account.reset_token_expires_at is None or account.reset_token_expires_at <= datetime.now(
            UTC
        ):
            return Failure(
                AuthNotFoundError(
                    message="Invalid or expired reset token",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )

        self._unindex_account(account)
        data = account.model_dump()
        data.pop("password_hash", None)
        data.pop("reset_token_hash", None)
        data.pop("reset_token_expires_at", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            password_hash=new_password_hash,
            reset_token_hash=None,
            reset_token_expires_at=None,
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)

    async def email_exists(self, email: str) -> Result[bool, AuthError]:
        return Success(email.lower() in self._by_email)

    async def update_profile(
        self, account_id: str, *, full_name: str
    ) -> Result[AuthAccount, AuthError]:
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        self._unindex_account(account)
        data = account.model_dump()
        data.pop("full_name", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            full_name=full_name,
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(updated)

    async def set_avatar_url(
        self, account_id: str, avatar_url: str | None
    ) -> Result[None, AuthError]:
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        self._unindex_account(account)
        data = account.model_dump()
        data.pop("avatar_url", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            avatar_url=avatar_url,
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)

    async def set_tenant_binding(
        self,
        account_id: str,
        *,
        tenant_id: str,
        legal_roles: tuple[LegalRole, ...],
        tenant_role_revision: int | None = None,
        role_revision: int | None = None,
    ) -> Result[None, AuthError]:
        """Mirror write guarded by a revision-monotonic compare-and-set."""
        revision = tenant_role_revision if tenant_role_revision is not None else role_revision
        if revision is None:
            message = "tenant_role_revision is required"
            raise ValueError(message)
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        mirrored = account.tenant_role_revision
        if mirrored > revision:
            return Failure(
                AuthConflictError(
                    message="Stale tenant mirror revision rejected by compare-and-set",
                    details={
                        "user_id": account_id,
                        "operation": "set_tenant_binding",
                        "expected_revision": revision,
                        "current_revision": mirrored,
                    },
                    source="auth_account_store",
                )
            )
        self._unindex_account(account)
        data = account.model_dump()
        data.pop("tenant_id", None)
        data.pop("legal_roles", None)
        data.pop("tenant_role_revision", None)
        data.pop("provisioning_state", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            tenant_id=tenant_id,
            legal_roles=legal_roles,
            tenant_role_revision=revision,
            provisioning_state=ProvisioningState.COMPLETED,
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)

    async def link_oauth_account(
        self,
        account_id: str,
        provider: str,
        provider_user_id: str,
        provider_email: str | None,
    ) -> Result[None, AuthError]:
        account = self._accounts.get(account_id)
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": account_id},
                    source="auth_account_store",
                )
            )
        owner = next(
            (
                existing
                for existing in self._accounts.values()
                if any(
                    linked.provider == provider and linked.provider_user_id == provider_user_id
                    for linked in existing.oauth_accounts
                )
            ),
            None,
        )
        if owner is not None and owner.id != account_id:
            return Failure(
                AuthConflictError(
                    message="OAuth identity is already linked to another account",
                    source="auth_account_store",
                )
            )
        linked_for_provider = next(
            (linked for linked in account.oauth_accounts if linked.provider == provider),
            None,
        )
        if linked_for_provider is not None:
            if linked_for_provider.provider_user_id != provider_user_id:
                return Failure(
                    AuthConflictError(
                        message="OAuth provider is already linked to another identity",
                        details={"provider": provider},
                        source="auth_account_store",
                    )
                )
            return Success(None)
        self._unindex_account(account)
        data = account.model_dump()
        data.pop("oauth_accounts", None)
        data.pop("oauth_providers", None)
        data.pop("updated_at", None)
        updated = AuthAccount(
            **data,
            oauth_providers=(*account.oauth_providers, provider),
            oauth_accounts=(
                *account.oauth_accounts,
                OAuthAccountInput(
                    provider=provider,
                    provider_user_id=provider_user_id,
                    provider_email=provider_email,
                ),
            ),
            updated_at=datetime.now(UTC),
        )
        self._index_account(updated)
        return Success(None)
