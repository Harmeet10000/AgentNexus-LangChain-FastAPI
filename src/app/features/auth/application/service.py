from __future__ import annotations

import secrets
import time
import traceback
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from asyncpg.exceptions import PostgresError
from authlib.integrations.base_client.errors import AuthlibBaseError
from authlib.integrations.httpx_client import AsyncOAuth2Client
from httpx import HTTPError
from opentelemetry import metrics
from returns.result import Failure, Success
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.kernel.result import log_expected_failure
from app.observability.logging import logger, trace_layer
from app.utils import ValidationException

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from sqlalchemy.ext.asyncio.engine import AsyncEngine

    from app.features.auth.adapters.persistence.mongodb.account_repository import (
        OAuthLinkIntent,
        RefreshTokenRepository,
    )
    from app.features.auth.adapters.persistence.mongodb.user_store import (
        AuthAccount,
        AuthAccountStore,
    )
    from app.features.auth.api.http.dto import LoginRequest, RegisterRequest
    from app.features.auth.application.personal_tenant_provisioning import (
        PersonalTenantProvisioner,
    )
    from app.features.auth.domain.errors import AuthResult

from app.features.auth.adapters.persistence.mongodb.account_repository import SessionData
from app.features.auth.adapters.persistence.mongodb.user_store import (
    NewAuthAccount,
    OAuthAccountInput,
)
from app.features.auth.api.http.dto import (
    OAuthLinkConfirmationResponse,
    SessionResponse,
    TokenResponse,
    UserResponse,
)
from app.features.auth.domain.errors import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthProvisioningPendingError,
    AuthValidationError,
)
from app.features.auth.domain.tenancy import LegalRole, ProvisioningState
from app.security.passwords import (
    generate_token,
    hash_password,
    hash_token,
    needs_rehash,
    verify_password,
)
from app.security.tokens import (
    create_access_token,
    create_refresh_token,
    decode_token,
    derive_refresh_token_material,
    sign_oauth_state,
    verify_oauth_state,
)

from .oauth import fetch_oauth_userinfo, get_oauth_config

_OAUTH_LINK_RECENT_AUTH_SECONDS = 15 * 60
_REFRESH_FAMILY_GRACE_SECONDS = 10
_REFRESH_METER = metrics.get_meter("app.auth.refresh")
_REFRESH_ROTATIONS = _REFRESH_METER.create_counter("auth.refresh.rotations")
_REFRESH_REPLAYS = _REFRESH_METER.create_counter("auth.refresh.replays")
_REFRESH_GRACE_RETRIES = _REFRESH_METER.create_counter("auth.refresh.grace_retries")
_REFRESH_FAILURES = _REFRESH_METER.create_counter("auth.refresh.failures")
_REFRESH_FAMILY_REVOKES = _REFRESH_METER.create_counter("auth.refresh.family_revocations")

# Dummy hash used in the constant-time negative path during login.
# Prevents timing attacks that would reveal whether an email is registered.
_DUMMY_HASH = "$argon2id$v=19$m=65536,t=2,p=2$c29tZXNhbHRzb21lc2FsdA$dGVzdGhhc2h0ZXN0aGFzaA"


class WebSocketConnectionCloser(Protocol):
    """Structural contract required of the collaborator passed to
    :meth:`AuthService.revoke_session_and_close_connections`.

    Declared structurally rather than importing ``WebSocketSecurityService`` for two
    reasons: only these two members are used, and ``AuthService`` must not acquire an
    import edge onto the WebSocket transport layer to obtain a parameter annotation.

    ``WebSocketSecurityService`` satisfies this protocol: it exposes a ``redis``
    property (``Redis | None`` — closure is best-effort and skipped when Redis is
    unavailable) and ``close_connection(connection_id, *, reason)``, which resolves
    the live WebSocket from the service's in-process registry.
    """

    @property
    def redis(self) -> Redis | None: ...

    async def close_connection(self, connection_id: str, /, *, reason: str) -> None: ...


def _to_user_response(account: AuthAccount) -> UserResponse:
    return UserResponse(
        id=account.id,
        email=account.email,
        full_name=account.full_name,
        avatar_url=account.avatar_url,
        role=account.role.value,
        is_verified=account.is_verified,
        is_active=account.is_active,
        created_at=account.created_at,
    )


class AuthService:
    def __init__(
        self,
        account_store: AuthAccountStore,
        token_repo: RefreshTokenRepository,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        *,
        provisioner: PersonalTenantProvisioner | None = None,
    ) -> None:
        self._account_store = account_store
        self._token_repo = token_repo
        self._session_factory = session_factory
        # Deep provisioning collaborator owning SQL transactions, idempotency,
        # lifecycle state and mirror sync.  Production wiring always supplies
        # it; without it the session-issuance invariant fails closed.
        self._provisioner = provisioner

    @trace_layer("service")
    async def register(self, dto: RegisterRequest) -> AuthResult[UserResponse]:
        email_exists_result = await self._account_store.email_exists(dto.email)
        if isinstance(email_exists_result, Failure):
            error = email_exists_result.failure()
            log_expected_failure(error, operation="email_exists")
            return Failure(error)
        if email_exists_result.unwrap():
            return Failure(
                AuthConflictError(message="Email already registered", source="auth_service")
            )

        verification_token = generate_token()
        new_account = NewAuthAccount(
            email=dto.email.lower(),
            password_hash=hash_password(dto.password),
            full_name=dto.full_name,
            verification_token_hash=hash_token(verification_token),
            # Registration is not complete until the authoritative SQL personal
            # tenant and membership exist; the account starts explicitly pending.
            provisioning_state=ProvisioningState.PENDING,
        )
        create_result = await self._account_store.create(new_account)
        if isinstance(create_result, Failure):
            error = create_result.failure()
            log_expected_failure(error, operation="create_user")
            return Failure(error)
        resolved = create_result.unwrap()

        provision_result = await self._provision_personal_tenant(resolved)
        if isinstance(provision_result, Failure):
            # The Mongo account exists but stays pending with no usable state
            # claimed; SQL provisioning retries converge idempotently.
            return Failure(provision_result.failure())

        logger.bind(user_id=resolved.id).info("User registered")
        return Success(_to_user_response(resolved))

    async def _provision_personal_tenant(
        self, account: AuthAccount
    ) -> AuthResult[None]:
        """Run the one provisioning seam for a freshly created account.

        Registration/OAuth completion requires the authoritative SQL personal
        tenant and membership, so a provisioning failure fails the whole
        account-creation result.  The Mongo account remains in ``pending`` and
        a later login (or migration) repairs it idempotently.
        """
        if self._provisioner is None:
            logger.bind(user_id=account.id).error(
                "auth_provisioner_unavailable_postgres_not_configured"
            )
            return Failure(
                AuthInfrastructureError(
                    message="PostgreSQL is unavailable; account provisioning cannot complete",
                    details={"operation": "provision_personal_tenant"},
                    source="auth_service",
                )
            )
        result = await self._provisioner.provision(
            account.id,
            display_name=account.full_name or account.email,
            legal_roles=account.legal_roles or (LegalRole.ANALYST,),
        )
        if isinstance(result, Failure):
            log_expected_failure(result.failure(), operation="provision_personal_tenant")
            return Failure(result.failure())
        return Success(None)

    @trace_layer("service")
    async def login(
        self,
        dto: LoginRequest,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> AuthResult[TokenResponse]:
        find_result = await self._account_store.find_by_email(dto.email)
        if isinstance(find_result, Failure):
            error = find_result.failure()
            log_expected_failure(error, operation="find_user")
            return Failure(error)
        account = find_result.unwrap()
        if account is None or account.password_hash is None:
            # Use dummy hash for constant-time comparison to prevent timing attacks
            verify_password(_DUMMY_HASH, dto.password)
            return Failure(
                AuthAuthenticationError(message="Invalid credentials", source="auth_service")
            )

        if not verify_password(account.password_hash, dto.password):
            return Failure(
                AuthAuthenticationError(message="Invalid credentials", source="auth_service")
            )

        if not account.is_active:
            return Failure(
                AuthAuthenticationError(message="Account is disabled", source="auth_service")
            )

        if not account.is_verified:
            return Failure(
                AuthAuthenticationError(
                    message="Email not verified. Check your inbox.", source="auth_service"
                )
            )

        # Transparent rehash: argon2 params may have been upgraded since this hash was created
        if needs_rehash(account.password_hash):
            rehash_result = await self._account_store.replace_password_hash(
                account.id, hash_password(dto.password)
            )
            if isinstance(rehash_result, Failure):
                error = rehash_result.failure()
                log_expected_failure(error, operation="rehash_password")
                return Failure(error)

        return await self._create_session(
            account=account,
            role=account.role,
            permissions=account.permissions,
            role_revision=account.role_revision,
            device_name=dto.device_name,
            ip=ip,
            user_agent=user_agent,
        )

    @trace_layer("service")
    async def logout(self, refresh_token: str) -> AuthResult[None]:
        claims = decode_token(refresh_token)
        if claims.token_type != "refresh":
            return Failure(
                AuthAuthenticationError(message="Not a refresh token", source="auth_service")
            )
        session_id = claims.sid or claims.jti
        if claims.family_id is not None:
            family_result = await self._token_repo.revoke_refresh_family(
                claims.family_id,
                reason="logout",
                user_id=claims.sub,
                session_id=session_id,
            )
            if isinstance(family_result, Failure):
                return Failure(family_result.failure())
            _REFRESH_FAMILY_REVOKES.add(1, {"reason": "logout"})
        revoke_result = await self._token_repo.revoke_session(
            session_id=session_id,
            user_id=claims.sub,
            reason="logout",
        )
        if isinstance(revoke_result, Failure):
            log_expected_failure(revoke_result.failure(), operation="revoke_session")
            return Failure(revoke_result.failure())
        logger.bind(user_id=claims.sub, session_id=session_id).info("Session revoked")
        return Success(None)

    @trace_layer("service")
    async def refresh(  # noqa: PLR0912, PLR0914, PLR0915
        self, refresh_token: str
    ) -> AuthResult[TokenResponse]:
        claims = decode_token(refresh_token)
        if claims.token_type != "refresh":
            return Failure(
                AuthAuthenticationError(message="Not a refresh token", source="auth_service")
            )

        session_id = claims.sid or claims.jti
        # Redis lookup is the revocation gate — if the session was deleted, deny.
        session_result = await self._token_repo.get_session(session_id)
        if isinstance(session_result, Failure):
            return Failure(session_result.failure())
        session = session_result.unwrap()
        if session is None or session.user_id != claims.sub:
            return Failure(
                AuthAuthenticationError(message="Session expired or revoked", source="auth_service")
            )

        user_result = await self._account_store.find_by_id(claims.sub)
        if isinstance(user_result, Failure):
            error = user_result.failure()
            log_expected_failure(error, operation="refresh_user_lookup")
            return Failure(error)
        account = user_result.unwrap()
        if account is None or not account.is_active:
            if claims.family_id:
                await self._token_repo.revoke_refresh_family(
                    family_id=claims.family_id,
                    reason="account_disabled",
                    user_id=claims.sub,
                    session_id=session_id,
                )
            return Failure(
                AuthAuthenticationError(message="User not found or disabled", source="auth_service")
            )
        if session.authz_revision != account.role_revision:
            if claims.family_id:
                await self._token_repo.revoke_refresh_family(
                    family_id=claims.family_id,
                    reason="authz_revision_changed",
                    user_id=claims.sub,
                    session_id=session_id,
                )
            return Failure(
                AuthAuthenticationError(
                    message="Session authorization is stale",
                    source="auth_service",
                )
            )

        # Invariant before minting: the authoritative SQL membership must exist.
        refresh_gate = await self._ensure_sql_membership_for_session(account)
        if isinstance(refresh_gate, Failure):
            return Failure(refresh_gate.failure())

        family_id = claims.family_id or f"legacy-{sha256(session_id.encode()).hexdigest()[:32]}"
        token_secret = claims.refresh_secret or claims.jti
        generation = claims.refresh_generation if claims.refresh_generation is not None else 0
        if claims.device_id is not None and claims.device_id != session.device_id:
            return Failure(
                AuthAuthenticationError(
                    message="Refresh token binding is invalid", source="auth_service"
                )
            )
        if claims.family_id is None:
            family_result = await self._token_repo.create_refresh_family(
                family_id=family_id,
                user_id=account.id,
                session_id=session_id,
                device_id=session.device_id,
                authz_revision=account.role_revision,
                token_id=claims.jti,
                token_secret=token_secret,
                issued_at=session.created_at,
                expires_at=session.expires_at,
                legacy=True,
            )
            if isinstance(family_result, Failure):
                return Failure(family_result.failure())
            attach_result = await self._token_repo.attach_refresh_family(
                session_id=session_id,
                family_id=family_id,
            )
            if isinstance(attach_result, Failure):
                await self._token_repo.revoke_refresh_family(
                    family_id=family_id,
                    reason="refresh_migration_failed",
                    user_id=account.id,
                    session_id=session_id,
                )
                return Failure(attach_result.failure())
        child_id, child_secret = derive_refresh_token_material(family_id, generation + 1)
        rotation_result = await self._token_repo.rotate_refresh_token(
            family_id=family_id,
            token_id=claims.jti,
            token_secret=token_secret,
            child_token_id=child_id,
            child_token_secret=child_secret,
            child_generation=generation + 1,
            expected_authz_revision=account.role_revision,
            now=datetime.now(UTC),
            grace_seconds=get_settings().REFRESH_TOKEN_ROTATION_GRACE_SECONDS,
            presented_generation=generation,
        )
        if isinstance(rotation_result, Failure):
            _REFRESH_FAILURES.add(1, {"reason": "infrastructure"})
            return Failure(rotation_result.failure())
        rotation = rotation_result.unwrap()
        if rotation.status == "reused":
            _REFRESH_REPLAYS.add(1, {"reason": "family_reuse"})
            _REFRESH_FAMILY_REVOKES.add(1, {"reason": "refresh_reuse"})
            await self._emit_security_event(
                "auth.refresh_token_reuse_detected",
                account.id,
                {"family_id": family_id, "session_id": session_id},
            )
            return Failure(
                AuthAuthenticationError(
                    message="Refresh token reuse detected; session revoked",
                    source="auth_service",
                )
            )
        if rotation.status in {"missing", "revoked", "stale", "invalid"}:
            if rotation.status == "stale":
                await self._token_repo.revoke_refresh_family(
                    family_id=family_id,
                    reason="authz_revision_changed",
                    user_id=account.id,
                    session_id=session_id,
                )
            _REFRESH_FAILURES.add(1, {"reason": rotation.status})
            return Failure(
                AuthAuthenticationError(
                    message="Refresh token is invalid or revoked", source="auth_service"
                )
            )
        if rotation.generation is None:
            return Failure(
                AuthInfrastructureError(
                    message="Refresh rotation returned no successor generation",
                    source="auth_service",
                )
            )
        child_id, child_secret = derive_refresh_token_material(family_id, rotation.generation)
        remaining_seconds = max(1, int(session.expires_at.timestamp() - time.time()))
        refresh_token, _ = create_refresh_token(
            user_id=account.id,
            session_id=session_id,
            device_id=session.device_id,
            family_id=family_id,
            token_id=child_id,
            token_secret=child_secret,
            generation=rotation.generation,
            parent_jti=claims.jti,
            expire_seconds=remaining_seconds,
        )
        if rotation.status == "grace":
            _REFRESH_GRACE_RETRIES.add(1, {"reason": "concurrent_retry"})
        else:
            _REFRESH_ROTATIONS.add(1, {"outcome": "rotated"})
        access_token, expires_in = create_access_token(
            user_id=account.id,
            session_id=session_id,
            role=account.role,
            permissions=account.permissions,
            authz_revision=account.role_revision,
        )
        return Success(
            TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=expires_in,
            )
        )

    @trace_layer("service")
    async def verify_email(self, token: str) -> AuthResult[None]:
        verify_result = await self._account_store.find_by_verification_token_hash(hash_token(token))
        if isinstance(verify_result, Failure):
            error = verify_result.failure()
            log_expected_failure(error, operation="verify_email")
            return Failure(error)
        account = verify_result.unwrap()
        if account is None:
            return Failure(
                AuthNotFoundError(
                    message="Invalid or expired verification token", source="auth_service"
                )
            )
        mark_result = await self._account_store.mark_email_verified(account.id)
        if isinstance(mark_result, Failure):
            error = mark_result.failure()
            log_expected_failure(error, operation="mark_email_verified")
            return Failure(error)
        logger.bind(user_id=account.id).info("Email verified")
        return Success(None)

    @trace_layer("service")
    async def resend_verification(self, email: str) -> AuthResult[None]:
        """Issue a fresh verification intent and stage its email delivery.

        The account/token mutation is committed by MongoDB and the email intent
        is committed by PostgreSQL; those stores do not participate in one
        transaction.  If PostgreSQL staging fails, return that typed failure so
        the caller can retry.  A retry creates a fresh/current token intent, and
        worker-side delivery remains idempotent for duplicate intents.
        """
        find_result = await self._account_store.find_by_email(email)
        if isinstance(find_result, Failure):
            return Success(None)  # silent — don't reveal email existence
        account = find_result.unwrap()
        if account is None or account.is_verified:
            return (
                Success(None)
                if account is None
                else Failure(
                    AuthConflictError(message="Email already verified", source="auth_service")
                )
            )

        new_token = generate_token()
        token_hash = hash_token(new_token)
        set_result = await self._account_store.set_verification_token_hash(account.id, token_hash)
        if isinstance(set_result, Failure):
            log_expected_failure(set_result.failure(), operation="set_verification_token")
            return Failure(set_result.failure())

        publish_result = await self._publish_outbox_event(
            aggregate_type="auth_email",
            aggregate_id=account.id,
            event_type="auth.send_verification_email",
            payload={
                "user_id": account.id,
                "email": account.email,
                "token": new_token,
                "idempotency_key": f"email-verify:{account.id}:{token_hash}",
            },
        )
        if isinstance(publish_result, Failure):
            return Failure(publish_result.failure())
        return Success(None)

    @trace_layer("service")
    async def forgot_password(self, email: str) -> AuthResult[None]:
        """Issue a fresh password-reset intent and stage its email delivery.

        MongoDB token state and the PostgreSQL outbox are separate stores, so a
        successful token update does not make email staging atomic.  A staging
        failure is returned to the caller for retry; the next retry replaces the
        token with a fresh/current intent.  Duplicate delivery is handled by the
        worker's idempotency contract rather than by this cross-store boundary.
        """
        find_result = await self._account_store.find_by_email(email)
        if isinstance(find_result, Failure):
            return Success(None)  # silent — identical response regardless of outcome
        account = find_result.unwrap()
        if account is None or not account.is_verified:
            return Success(None)  # silent — identical response regardless of outcome

        settings = get_settings()
        reset_token = generate_token()
        token_hash = hash_token(reset_token)
        expires_at = datetime.now(UTC) + timedelta(
            minutes=settings.PASSWORD_RESET_EXPIRE_MINUTES,
        )
        set_result = await self._account_store.set_reset_token(account.id, token_hash, expires_at)
        if isinstance(set_result, Failure):
            log_expected_failure(set_result.failure(), operation="set_reset_token")
            return Failure(set_result.failure())

        publish_result = await self._publish_outbox_event(
            aggregate_type="auth_email",
            aggregate_id=account.id,
            event_type="auth.send_password_reset_email",
            payload={
                "user_id": account.id,
                "email": account.email,
                "token": reset_token,
                "idempotency_key": f"email-reset:{account.id}:{token_hash}",
            },
        )
        if isinstance(publish_result, Failure):
            return Failure(publish_result.failure())
        return Success(None)

    @trace_layer("service")
    async def reset_password(self, token: str, new_password: str) -> AuthResult[None]:
        token_hash = hash_token(token)
        reset_result = await self._account_store.find_by_reset_token_hash(token_hash)
        if isinstance(reset_result, Failure):
            error = reset_result.failure()
            log_expected_failure(error, operation="reset_password")
            return Failure(error)
        account = reset_result.unwrap()
        if account is None:
            return Failure(
                AuthNotFoundError(message="Invalid or expired reset token", source="auth_service")
            )

        if account.reset_token_expires_at is None or account.reset_token_expires_at < datetime.now(
            UTC
        ):
            return Failure(
                AuthAuthenticationError(message="Reset token has expired", source="auth_service")
            )

        # Atomic operation: consume reset token and replace password
        consume_result = await self._account_store.consume_reset_token_and_replace_password(
            account_id=account.id,
            expected_token_hash=token_hash,
            new_password_hash=hash_password(new_password),
        )
        if isinstance(consume_result, Failure):
            error = consume_result.failure()
            log_expected_failure(error, operation="consume_reset_token_and_replace_password")
            return Failure(error)

        # Force all sessions offline after a password reset
        revoke_result = await self._token_repo.revoke_all_user_sessions(
            user_id=account.id,
            reason="password_reset",
        )
        if isinstance(revoke_result, Failure):
            log_expected_failure(revoke_result.failure(), operation="revoke_all_user_sessions")
            return Failure(revoke_result.failure())
        logger.bind(user_id=account.id).info("Password reset — all sessions revoked")
        return Success(None)

    @staticmethod
    @trace_layer("service")
    async def oauth_get_authorization_url(provider: str) -> AuthResult[tuple[str, str]]:
        """Return (authorization_url, signed_state_for_cookie)."""

        config = get_oauth_config(provider)
        state = secrets.token_urlsafe(32)

        async with AsyncOAuth2Client(client_id=config.client_id) as client:
            url, _ = client.create_authorization_url(
                config.authorization_endpoint,
                redirect_uri=config.redirect_uri,
                state=state,
                scope=config.scope,
            )

        return Success((str(url), sign_oauth_state(state, provider)))

    @trace_layer("service")
    async def oauth_link_get_authorization_url(
        self,
        provider: str,
        *,
        user_id: str,
        session_id: str | None,
        authz_revision: int | None,
    ) -> AuthResult[tuple[str, str]]:
        """Start an explicit link transaction for a recently authenticated session."""
        if session_id is None or authz_revision is None:
            return Failure(
                AuthAuthenticationError(
                    message="A stateful authenticated session is required",
                    source="auth_service",
                )
            )
        account_result = await self._account_store.find_by_id(user_id)
        if isinstance(account_result, Failure):
            return Failure(account_result.failure())
        account = account_result.unwrap()
        if account is None or not account.is_active or not account.is_verified:
            return Failure(
                AuthAuthenticationError(
                    message="Account cannot link OAuth identities", source="auth_service"
                )
            )
        session_result = await self._token_repo.get_session(session_id)
        if isinstance(session_result, Failure):
            return Failure(session_result.failure())
        session = session_result.unwrap()
        if (
            session is None
            or session.user_id != user_id
            or session.authz_revision != account.role_revision
            or account.role_revision != authz_revision
            or session.created_at
            < datetime.now(UTC) - timedelta(seconds=_OAUTH_LINK_RECENT_AUTH_SECONDS)
        ):
            return Failure(
                AuthAuthenticationError(
                    message="Recent reauthentication is required before linking an account",
                    source="auth_service",
                )
            )

        config = get_oauth_config(provider)
        state = secrets.token_urlsafe(32)
        intent_result = await self._token_repo.create_oauth_link_intent(
            user_id=user_id,
            session_id=session_id,
            provider=provider,
            authz_revision=account.role_revision,
            state=state,
            account_email=account.email,
        )
        if isinstance(intent_result, Failure):
            return Failure(intent_result.failure())
        async with AsyncOAuth2Client(client_id=config.client_id) as client:
            url, _ = client.create_authorization_url(
                config.authorization_endpoint,
                redirect_uri=config.redirect_uri,
                state=state,
                scope=config.scope,
            )
        return Success((str(url), sign_oauth_state(state, provider)))

    async def _validate_oauth_link_session(
        self, intent: OAuthLinkIntent
    ) -> AuthResult[AuthAccount]:
        account_result = await self._account_store.find_by_id(intent.user_id)
        if isinstance(account_result, Failure):
            return Failure(account_result.failure())
        account = account_result.unwrap()
        if account is None or not account.is_active or not account.is_verified:
            return Failure(
                AuthAuthenticationError(
                    message="Account cannot link OAuth identities", source="auth_service"
                )
            )
        session_result = await self._token_repo.get_session(intent.session_id)
        if isinstance(session_result, Failure):
            return Failure(session_result.failure())
        session = session_result.unwrap()
        email_unchanged = (
            intent.account_email is None or account.email.lower() == intent.account_email.lower()
        )
        session_valid = (
            session is not None
            and session.user_id == intent.user_id
            and email_unchanged
            and session.authz_revision == account.role_revision
            and account.role_revision == intent.authz_revision
            and session.created_at
            >= datetime.now(UTC) - timedelta(seconds=_OAUTH_LINK_RECENT_AUTH_SECONDS)
        )
        if not session_valid:
            return Failure(
                AuthAuthenticationError(
                    message="OAuth link authorization is stale; authenticate again",
                    source="auth_service",
                )
            )
        return Success(account)

    async def oauth_link_preview(
        self, action_token: str, *, user_id: str, session_id: str | None
    ) -> AuthResult[OAuthLinkConfirmationResponse]:
        candidate_result = await self._token_repo.get_oauth_link_candidate(action_token)
        if isinstance(candidate_result, Failure):
            return Failure(candidate_result.failure())
        candidate = candidate_result.unwrap()
        if (
            candidate is None
            or not candidate.ready_for_confirmation
            or candidate.expires_at <= datetime.now(UTC)
        ):
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_replayed",
                user_id,
                reason="missing_or_expired_action",
            )
            return Failure(
                AuthAuthenticationError(
                    message="OAuth link confirmation is invalid or expired", source="auth_service"
                )
            )
        if candidate.user_id != user_id or candidate.session_id != session_id:
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_rejected",
                user_id,
                reason="wrong_user_or_session",
                provider=candidate.provider,
            )
            return Failure(
                AuthAuthorizationError(
                    message="OAuth link confirmation belongs to another session",
                    source="auth_service",
                )
            )
        account_result = await self._validate_oauth_link_session(candidate)
        if isinstance(account_result, Failure):
            return Failure(account_result.failure())
        if candidate.provider_user_id is None or candidate.provider_email is None:
            return Failure(
                AuthAuthenticationError(
                    message="OAuth link confirmation is malformed", source="auth_service"
                )
            )
        return Success(
            OAuthLinkConfirmationResponse(
                action_token=action_token,
                provider=candidate.provider,
                provider_email=candidate.provider_email,
                provider_user_id=candidate.provider_user_id,
                full_name=candidate.full_name,
                expires_at=candidate.expires_at,
            )
        )

    @trace_layer("service")
    async def oauth_link_confirm(
        self,
        action_token: str,
        *,
        user_id: str,
        session_id: str | None,
        authz_revision: int | None,
        confirm: bool,
    ) -> AuthResult[None]:
        if not confirm:
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_rejected", user_id, reason="confirmation_not_explicit"
            )
            return Failure(
                AuthValidationError(
                    message="Explicit confirmation is required", source="auth_service"
                )
            )
        candidate_result = await self._token_repo.get_oauth_link_candidate(action_token)
        if isinstance(candidate_result, Failure):
            return Failure(candidate_result.failure())
        candidate = candidate_result.unwrap()
        if (
            candidate is None
            or not candidate.ready_for_confirmation
            or candidate.expires_at <= datetime.now(UTC)
        ):
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_replayed",
                user_id,
                reason="missing_or_expired_action",
            )
            return Failure(
                AuthAuthenticationError(
                    message="OAuth link confirmation is invalid or expired", source="auth_service"
                )
            )
        if (
            candidate.user_id != user_id
            or candidate.session_id != session_id
            or candidate.authz_revision != authz_revision
        ):
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_rejected",
                user_id,
                reason="wrong_user_session_or_authorization_revision",
                provider=candidate.provider,
            )
            return Failure(
                AuthAuthorizationError(
                    message="OAuth link confirmation belongs to another session",
                    source="auth_service",
                )
            )
        account_result = await self._validate_oauth_link_session(candidate)
        if isinstance(account_result, Failure):
            return Failure(account_result.failure())
        account = account_result.unwrap()
        if candidate.provider_user_id is None or candidate.provider_email is None:
            return Failure(
                AuthAuthenticationError(
                    message="OAuth link confirmation is malformed", source="auth_service"
                )
            )
        owner_result = await self._account_store.find_by_oauth_identity(
            candidate.provider, candidate.provider_user_id
        )
        if isinstance(owner_result, Failure):
            return Failure(owner_result.failure())
        owner = owner_result.unwrap()
        if owner is not None and owner.id != account.id:
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_rejected",
                account.id,
                reason="provider_identity_already_owned",
                provider=candidate.provider,
            )
            return Failure(
                AuthConflictError(
                    message="OAuth identity is already linked to another account",
                    source="auth_service",
                )
            )
        consumed_result = await self._token_repo.consume_oauth_link_candidate(action_token)
        if isinstance(consumed_result, Failure):
            return Failure(consumed_result.failure())
        if consumed_result.unwrap() is None:
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_replayed",
                account.id,
                reason="action_already_consumed",
                provider=candidate.provider,
            )
            return Failure(
                AuthAuthenticationError(
                    message="OAuth link confirmation was already used", source="auth_service"
                )
            )
        link_result = await self._account_store.link_oauth_account(
            account.id,
            candidate.provider,
            candidate.provider_user_id,
            candidate.provider_email,
        )
        if isinstance(link_result, Failure):
            return Failure(link_result.failure())
        await self._emit_security_event(
            "auth.oauth_linked",
            account.id,
            {
                "provider": candidate.provider,
                "provider_user_id": candidate.provider_user_id,
                "provider_email": candidate.provider_email,
            },
        )
        return Success(None)

    @trace_layer("service")
    async def oauth_unlink(
        self, provider: str, *, user_id: str, session_id: str | None, authz_revision: int | None
    ) -> AuthResult[None]:
        account_result = await self._account_store.find_by_id(user_id)
        if isinstance(account_result, Failure):
            return Failure(account_result.failure())
        account = account_result.unwrap()
        if account is None:
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_rejected",
                user_id,
                reason="account_not_found",
                provider=provider,
            )
            return Failure(
                AuthAuthenticationError(message="Account not found", source="auth_service")
            )
        if session_id is None or authz_revision != account.role_revision:
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_rejected",
                user_id,
                reason="reauthentication_required",
                provider=provider,
            )
            return Failure(
                AuthAuthenticationError(
                    message="Recent reauthentication is required", source="auth_service"
                )
            )
        session_result = await self._token_repo.get_session(session_id)
        if isinstance(session_result, Failure):
            return Failure(session_result.failure())
        session = session_result.unwrap()
        if session is None or session.created_at < datetime.now(UTC) - timedelta(
            seconds=_OAUTH_LINK_RECENT_AUTH_SECONDS
        ):
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_rejected",
                user_id,
                reason="stale_or_missing_session",
                provider=provider,
            )
            return Failure(
                AuthAuthenticationError(
                    message="Recent reauthentication is required", source="auth_service"
                )
            )
        result = await self._account_store.unlink_oauth_account(user_id, provider)
        if isinstance(result, Failure):
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_rejected", user_id, reason="unlink_rejected", provider=provider
            )
            return Failure(result.failure())
        await self._emit_security_event("auth.oauth_unlinked", user_id, {"provider": provider})
        return Success(None)

    @trace_layer("service")
    async def oauth_callback(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        provider: str,
        code: str,
        state: str,
        signed_state: str,
        *,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> AuthResult[TokenResponse | OAuthLinkConfirmationResponse]:
        if not verify_oauth_state(signed_state, state, provider):
            return Failure(
                AuthAuthenticationError(
                    message="Invalid OAuth state — possible CSRF attack", source="auth_service"
                )
            )

        config = get_oauth_config(provider)
        try:
            userinfo = await fetch_oauth_userinfo(provider, config, code)
        except (
            AuthlibBaseError,
            HTTPError,
            ValidationException,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            logger.bind(provider=provider, error=str(exc)).warning("OAuth identity lookup failed")
            return Failure(
                AuthAuthenticationError(
                    message="OAuth provider could not verify the account",
                    source="auth_service",
                )
            )

        intent_result = await self._token_repo.get_oauth_link_intent(state)
        if not isinstance(intent_result, (Success, Failure)):
            # Compatibility with older injected test doubles that predate the
            # explicit-link repository contract; production repositories always
            # return a typed Result.
            intent_result = Success(None)
        if isinstance(intent_result, Failure):
            return Failure(intent_result.failure())
        link_intent = intent_result.unwrap()
        if link_intent is not None:
            if link_intent.provider != provider:
                await self._emit_oauth_link_security_event(
                    "auth.oauth_link_rejected",
                    link_intent.user_id,
                    reason="provider_state_mixup",
                    provider=provider,
                )
                return Failure(
                    AuthAuthenticationError(
                        message="OAuth link provider does not match the requested provider",
                        source="auth_service",
                    )
                )
            account_result = await self._validate_oauth_link_session(link_intent)
            if isinstance(account_result, Failure):
                return Failure(account_result.failure())
            account = account_result.unwrap()
            if not userinfo.email_verified:
                await self._emit_oauth_link_security_event(
                    "auth.oauth_link_rejected",
                    account.id,
                    reason="provider_email_unverified",
                    provider=provider,
                )
                return Failure(
                    AuthAuthenticationError(
                        message="OAuth email is not verified", source="auth_service"
                    )
                )
            owner_result = await self._account_store.find_by_oauth_identity(
                provider, userinfo.provider_user_id
            )
            if isinstance(owner_result, Failure):
                return Failure(owner_result.failure())
            owner = owner_result.unwrap()
            if owner is not None and owner.id != account.id:
                await self._emit_oauth_link_security_event(
                    "auth.oauth_link_rejected",
                    account.id,
                    reason="provider_identity_already_owned",
                    provider=provider,
                )
                return Failure(
                    AuthConflictError(
                        message="OAuth identity is already linked to another account",
                        source="auth_service",
                    )
                )
            email_result = await self._account_store.find_by_email(userinfo.email)
            if isinstance(email_result, Failure):
                return Failure(email_result.failure())
            email_owner = email_result.unwrap()
            if email_owner is not None and email_owner.id != account.id:
                await self._emit_oauth_link_security_event(
                    "auth.oauth_link_rejected",
                    account.id,
                    reason="provider_email_belongs_to_another_account",
                    provider=provider,
                )
                return Failure(
                    AuthConflictError(
                        message="OAuth email belongs to another account",
                        source="auth_service",
                    )
                )
            completed_result = await self._token_repo.complete_oauth_link_intent(
                state=state,
                provider_user_id=userinfo.provider_user_id,
                provider_email=userinfo.email.lower(),
                full_name=userinfo.full_name,
            )
            if isinstance(completed_result, Failure):
                await self._emit_oauth_link_security_event(
                    "auth.oauth_link_rejected",
                    account.id,
                    reason="callback_replay_or_expired_state",
                    provider=provider,
                )
                return Failure(completed_result.failure())
            candidate, action_token = completed_result.unwrap()
            if candidate.provider_user_id is None or candidate.provider_email is None:
                return Failure(
                    AuthAuthenticationError(
                        message="OAuth link confirmation is malformed", source="auth_service"
                    )
                )
            return Success(
                OAuthLinkConfirmationResponse(
                    action_token=action_token,
                    provider=provider,
                    provider_email=candidate.provider_email,
                    provider_user_id=candidate.provider_user_id,
                    full_name=candidate.full_name,
                    expires_at=candidate.expires_at,
                )
            )

        # Provider subject is the login identity.  Email is never sufficient to
        # attach a new provider to an existing local account.
        identity_result = await self._account_store.find_by_oauth_identity(
            provider, userinfo.provider_user_id
        )
        if not isinstance(identity_result, (Success, Failure)):
            identity_result = Success(None)
        if isinstance(identity_result, Failure):
            return Failure(identity_result.failure())
        identity_owner = identity_result.unwrap()
        if identity_owner is not None:
            if not identity_owner.is_active or not identity_owner.is_verified:
                return Failure(
                    AuthAuthenticationError(
                        message="Account is disabled or unverified", source="auth_service"
                    )
                )
            if not userinfo.email_verified:
                return Failure(
                    AuthAuthenticationError(
                        message="OAuth email is not verified", source="auth_service"
                    )
                )
            logger.bind(user_id=identity_owner.id, provider=provider, created=False).info(
                "OAuth login"
            )
            return await self._create_session(
                account=identity_owner,
                role=identity_owner.role.value,
                permissions=identity_owner.permissions,
                role_revision=identity_owner.role_revision,
                ip=ip,
                user_agent=user_agent,
            )

        find_result = await self._account_store.find_by_email(userinfo.email)
        if isinstance(find_result, Failure):
            error = find_result.failure()
            log_expected_failure(error, operation="oauth_find_user")
            return Failure(error)

        account = find_result.unwrap()
        if account is not None:
            # User exists - login
            if not account.is_active:
                return Failure(
                    AuthAuthenticationError(message="Account is disabled", source="auth_service")
                )
            if not userinfo.email_verified:
                return Failure(
                    AuthAuthenticationError(
                        message="OAuth email is not verified", source="auth_service"
                    )
                )
            if not account.is_verified:
                return Failure(
                    AuthAuthenticationError(
                        message="Local account email is not verified", source="auth_service"
                    )
                )
            await self._emit_oauth_link_security_event(
                "auth.oauth_link_rejected",
                account.id,
                reason="email_only_auto_link_blocked",
                provider=provider,
            )
            return Failure(
                AuthConflictError(
                    message="OAuth identity is not linked; start an explicit account-link transaction",
                    source="auth_service",
                )
            )

        # Create new OAuth user
        if not userinfo.email_verified:
            return Failure(
                AuthAuthenticationError(
                    message="OAuth email is not verified", source="auth_service"
                )
            )
        new_account = NewAuthAccount(
            email=userinfo.email.lower(),
            password_hash=None,  # OAuth-only account
            full_name=userinfo.full_name,
            is_verified=True,
            verification_token_hash=None,
            # First-login account creation is a registration path: SQL personal
            # tenant and membership must exist before any session is issued.
            provisioning_state=ProvisioningState.PENDING,
            oauth_account=OAuthAccountInput(
                provider=provider,
                provider_user_id=userinfo.provider_user_id,
                provider_email=userinfo.email,
            ),
        )
        create_result = await self._account_store.create(new_account)
        if isinstance(create_result, Failure):
            error = create_result.failure()
            log_expected_failure(error, operation="oauth_create_user")
            return Failure(error)
        account = create_result.unwrap()

        provision_result = await self._provision_personal_tenant(account)
        if isinstance(provision_result, Failure):
            return Failure(provision_result.failure())

        logger.bind(user_id=account.id, provider=provider, created=True).info("OAuth login")
        return await self._create_session(
            account=account,
            role=account.role.value,
            permissions=account.permissions,
            role_revision=account.role_revision,
            ip=ip,
            user_agent=user_agent,
        )

    @trace_layer("service")
    async def list_sessions(
        self,
        user_id: str,
        current_session_id: str | None = None,
    ) -> AuthResult[list[SessionResponse]]:
        sessions_result = await self._token_repo.get_user_sessions(user_id)
        if isinstance(sessions_result, Failure):
            error = sessions_result.failure()
            log_expected_failure(error, operation="get_user_sessions")
            return Failure(error)
        sessions = sessions_result.unwrap()
        return Success(
            [
                SessionResponse(
                    session_id=s.session_id,
                    device_id=s.device_id,
                    device_name=s.device_name,
                    ip_address=s.ip_address,
                    user_agent=s.user_agent,
                    created_at=s.created_at,
                    expires_at=s.expires_at,
                    is_current=s.session_id == current_session_id,
                )
                for s in sessions
            ]
        )

    @trace_layer("service")
    async def revoke_session(
        self,
        session_id: str,
        user_id: str,
    ) -> AuthResult[None]:
        session_result = await self._token_repo.get_session(session_id)
        if isinstance(session_result, Failure):
            error = session_result.failure()
            log_expected_failure(error, operation="get_session")
            return Failure(error)
        resolved_session = session_result.unwrap()
        if resolved_session is None:
            return Failure(AuthNotFoundError(message="Session not found", source="auth_service"))
        if resolved_session.user_id != user_id:
            return Failure(
                AuthAuthorizationError(
                    message="Cannot revoke another user's session", source="auth_service"
                )
            )
        family_id = resolved_session.family_id
        if family_id:
            family_result = await self._token_repo.revoke_refresh_family(
                family_id=family_id,
                reason="manual_revoke",
                user_id=user_id,
                session_id=session_id,
            )
            if isinstance(family_result, Failure):
                log_expected_failure(family_result.failure(), operation="revoke_refresh_family")
                return Failure(family_result.failure())
            _REFRESH_FAMILY_REVOKES.add(1, {"reason": "manual_revoke"})
        revoke_result = await self._token_repo.revoke_session(
            session_id=session_id,
            user_id=user_id,
            reason="manual_revoke",
        )
        if isinstance(revoke_result, Failure):
            log_expected_failure(revoke_result.failure(), operation="revoke_session")
            return Failure(revoke_result.failure())
        return Success(None)

    @trace_layer("service")
    async def revoke_all_sessions(
        self,
        user_id: str,
        except_session_id: str | None = None,
    ) -> AuthResult[None]:
        family_count = 0
        sessions_result = await self._token_repo.get_user_sessions(user_id)
        if isinstance(sessions_result, Success):
            family_count = len(
                {session.family_id for session in sessions_result.unwrap() if session.family_id}
            )
        revoke_result = await self._token_repo.revoke_all_user_sessions(
            user_id=user_id,
            except_session_id=except_session_id,
            reason="revoke_all",
        )
        if isinstance(revoke_result, Failure):
            log_expected_failure(revoke_result.failure(), operation="revoke_all_user_sessions")
            return Failure(revoke_result.failure())
        if family_count:
            _REFRESH_FAMILY_REVOKES.add(family_count, {"reason": "revoke_all"})
        return Success(None)

    @trace_layer("service")
    async def revoke_session_and_close_connections(
        self,
        session_id: str,
        user_id: str,
        ws_security_service: WebSocketConnectionCloser,
        reason: str = "manual_revoke",
    ) -> AuthResult[list[str]]:
        """Revoke a session and close all associated WebSocket connections.

        Task 3.4: Atomic session revocation + connection teardown at service boundary.
        - Revoke the session in Redis token store
        - Look up all connections for this session from Redis sorted set
        - Close each connection and return the list of closed connection IDs

        Args:
            session_id: The session ID to revoke
            user_id: The user ID (for validation)
            ws_security_service: Collaborator satisfying :class:`WebSocketConnectionCloser`
                — a ``redis`` client for the ``ws:session:{id}`` presence set plus a
                ``close_connection()`` coroutine. See that protocol's KNOWN GAP note.
            reason: Reason for revocation (default: "manual_revoke")

        Returns:
            List of connection IDs that were closed

        Raises:
            NotFoundException: If session not found
            UnauthorizedException: If user_id doesn't match session owner
            AppException: If revocation or connection closure fails
        """
        # Verify session exists and belongs to this user
        session_result = await self._token_repo.get_session(session_id)
        if isinstance(session_result, Failure):
            error = session_result.failure()
            log_expected_failure(error, operation="get_session")
            return Failure(error)
        resolved_session = session_result.unwrap()
        if resolved_session is None:
            return Failure(AuthNotFoundError(message="Session not found", source="auth_service"))
        if resolved_session.user_id != user_id:
            return Failure(
                AuthAuthorizationError(
                    message="Cannot revoke another user's session", source="auth_service"
                )
            )

        if resolved_session.family_id:
            family_result = await self._token_repo.revoke_refresh_family(
                family_id=resolved_session.family_id,
                reason=reason,
                user_id=user_id,
                session_id=session_id,
            )
            if isinstance(family_result, Failure):
                log_expected_failure(family_result.failure(), operation="revoke_refresh_family")
                return Failure(family_result.failure())
            _REFRESH_FAMILY_REVOKES.add(1, {"reason": reason})

        # Revoke the session
        revoke_result = await self._token_repo.revoke_session(
            session_id=session_id,
            user_id=user_id,
            reason=reason,
        )
        if isinstance(revoke_result, Failure):
            log_expected_failure(revoke_result.failure(), operation="revoke_session")
            return Failure(revoke_result.failure())

        # Look up active connections for this session from sorted set
        closed_connection_ids: list[str] = []
        try:
            if ws_security_service.redis is None:
                logger.bind(operation="revoke_session", layer="auth").warning(
                    "No Redis client, skipping WebSocket connection closure"
                )
                return Success(closed_connection_ids)
            # Retrieve all connections for this session from the sorted set
            # The sorted set key is ws:session:{session_id}
            connections = await ws_security_service.redis.zrange(
                f"ws:session:{session_id}",
                0,
                -1,
            )
            for member in connections:
                conn_id = member.decode() if isinstance(member, bytes) else str(member)
                # Close the WebSocket connection
                await ws_security_service.close_connection(conn_id, reason=reason)
                closed_connection_ids.append(conn_id)

            logger.bind(
                user_id=user_id,
                session_id=session_id,
                closed_count=len(closed_connection_ids),
            ).info("Session revoked and connections closed")
        except Exception as e:  # noqa: BLE001 — connection closure is best-effort
            logger.bind(user_id=user_id, session_id=session_id, error=str(e)).exception(
                "Error closing connections"
            )
            # Don't fail the entire operation if connection closure has issues
            # The session is already revoked, so new WebSocket auth will fail

        return Success(closed_connection_ids)

    async def _ensure_sql_membership_for_session(self, account: AuthAccount) -> AuthResult[None]:
        """Invariant gate: a session is only issued when SQL membership exists.

        Login and refresh perform this authoritative check before minting; it
        is deliberately not a provisioning mechanism.  Semantics:

        - SQL unavailable while checking → retryable ``AuthInfrastructureError``
          (503), never 401/403.
        - No membership + durably pending account → one idempotent durable
          repair attempt, converging a crash between account creation and the
          SQL commit.  A failing repair keeps the account session-less.
        - No membership on any other account → typed
          ``ACCOUNT_PROVISIONING_PENDING`` (409, retryable) — never invalid
          credentials for a correct password.
        """
        if self._provisioner is None:
            logger.bind(user_id=account.id).error(
                "session_authorization_invariant_unavailable_postgres_not_configured"
            )
            return Failure(
                AuthInfrastructureError(
                    message="SQL membership authority is unavailable; sessions cannot be issued",
                    details={"operation": "session_authorization_invariant"},
                    source="auth_service",
                )
            )
        check = await self._provisioner.has_membership(account.id)
        if isinstance(check, Success) and check.unwrap():
            return Success(None)
        if isinstance(check, Failure):
            return Failure(check.failure())
        if account.provisioning_state == ProvisioningState.PENDING:
            repair = await self._provision_personal_tenant(account)
            if isinstance(repair, Failure):
                return Failure(repair.failure())
            return Success(None)
        return Failure(
            AuthProvisioningPendingError(
                message="Account provisioning is pending; retry shortly",
                details={"user_id": account.id, "retryable": True},
                source="auth_service",
            )
        )

    async def _create_session(
        self,
        *,
        account: AuthAccount,
        role: str,
        permissions: frozenset[str],
        role_revision: int,
        device_name: str | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> AuthResult[TokenResponse]:
        invariant = await self._ensure_sql_membership_for_session(account)
        if isinstance(invariant, Failure):
            return Failure(invariant.failure())
        user_id = account.id
        settings = get_settings()
        session_id = str(uuid4())
        device_id = str(uuid4())
        family_id = str(uuid4())
        now = datetime.now(UTC)
        token_id, token_secret = derive_refresh_token_material(family_id, 0)

        access_token, expires_in = create_access_token(
            user_id=user_id,
            session_id=session_id,
            role=role,
            permissions=permissions,
            authz_revision=role_revision,
        )
        refresh_token, refresh_ttl = create_refresh_token(
            user_id=user_id,
            session_id=session_id,
            device_id=device_id,
            family_id=family_id,
            token_id=token_id,
            token_secret=token_secret,
            generation=0,
            authz_revision=role_revision,
        )
        store_result = await self._token_repo.store_session(
            SessionData(
                session_id=session_id,
                user_id=user_id,
                device_id=device_id,
                device_name=device_name,
                ip_address=ip,
                user_agent=user_agent,
                created_at=now,
                expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
                ttl=refresh_ttl,
                authz_revision=role_revision,
                family_id=family_id,
            )
        )
        if isinstance(store_result, Failure):
            log_expected_failure(store_result.failure(), operation="store_session")
            return Failure(store_result.failure())
        family_result = await self._token_repo.create_refresh_family(
            family_id=family_id,
            user_id=user_id,
            session_id=session_id,
            device_id=device_id,
            authz_revision=role_revision,
            token_id=token_id,
            token_secret=token_secret,
            issued_at=now,
            expires_at=now + timedelta(seconds=refresh_ttl),
        )
        if isinstance(family_result, Failure):
            await self._token_repo.revoke_session(
                session_id, user_id, reason="family_create_failed"
            )
            log_expected_failure(family_result.failure(), operation="create_refresh_family")
            return Failure(family_result.failure())
        return Success(
            TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=expires_in,
            )
        )

    async def _emit_security_event(
        self,
        event_type: str,
        user_id: str,
        payload: dict[str, object],
    ) -> None:
        """Emit an immutable security event through the available event seam."""
        logger.bind(
            event_type=event_type,
            user_id=user_id,
            payload=payload,
            immutable=True,
        ).info("auth_security_event")
        if self._session_factory is None:
            return
        result = await self._publish_outbox_event(
            aggregate_type="auth_security",
            aggregate_id=user_id,
            event_type=event_type,
            payload={"user_id": user_id, **payload},
        )
        if isinstance(result, Failure):
            logger.bind(event_type=event_type, user_id=user_id).error(
                "auth_security_event_persistence_failed"
            )

    async def _emit_oauth_link_security_event(
        self,
        event_type: str,
        user_id: str,
        *,
        reason: str,
        **details: object,
    ) -> None:
        """Record link rejection/replay and hand off notification metadata.

        The current application has no dedicated auth-notification service.  A
        durable outbox event is therefore the notification seam when
        PostgreSQL is configured; the ``notify_user`` marker lets a future
        delivery worker distinguish these events from audit-only records.
        """
        await self._emit_security_event(
            event_type,
            user_id,
            {"reason": reason, "notify_user": True, **details},
        )

    async def _publish_outbox_event(
        self,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> AuthResult[None]:
        """Persist one email intent in an explicit PostgreSQL transaction.

        This is an eventual cross-store submission boundary: MongoDB owns the
        account/token mutation while PostgreSQL owns the durable outbox row.
        There is no distributed transaction between them.  The caller receives
        a typed failure when staging or committing the outbox fails and may
        retry the operation, which generates a new current token intent.
        ``with_outbox`` deliberately only stages SQL and notification work; the
        transaction context below owns commit/rollback.
        """
        from app.events import (
            with_outbox,
        )

        engine: AsyncEngine | None = None
        try:
            if self._session_factory is not None:
                async with self._session_factory() as session, session.begin():
                    await with_outbox(
                        session=session,
                        aggregate_type=aggregate_type,
                        aggregate_id=aggregate_id,
                        event_type=event_type,
                        payload=payload,
                    )
            else:
                from sqlalchemy.ext.asyncio import (
                    AsyncSession,
                    create_async_engine,
                )

                from app.connections.postgres import (
                    get_database_url,
                )

                engine = create_async_engine(get_database_url())
                async with engine.begin() as conn, AsyncSession(bind=conn) as session:
                    await with_outbox(
                        session=session,
                        aggregate_type=aggregate_type,
                        aggregate_id=aggregate_id,
                        event_type=event_type,
                        payload=payload,
                    )
            logger.bind(
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                event_type=event_type,
                operation="publish_outbox_event",
            ).info("Auth email intent committed; cross-store submission remains eventual")
            return Success(None)
        except (PostgresError, SQLAlchemyError) as exc:
            exc.add_note(
                "auth email intent staging failed; MongoDB token mutation and PostgreSQL "
                "outbox commit are separate transactions"
            )
            exc.add_note(traceback.format_exc())
            logger.bind(
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                event_type=event_type,
                operation="publish_outbox_event",
                error=str(exc),
                diagnostic_notes=list(getattr(exc, "__notes__", [])),
            ).exception("Auth email intent could not be committed")
            error = AuthInfrastructureError(
                message="Auth email intent could not be persisted",
                details={"operation": "publish_outbox_event"},
                source="auth_service",
            )
            log_expected_failure(error, operation="publish_outbox_event")
            return Failure(error)
        finally:
            if engine is not None:
                await engine.dispose()
