from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from authlib.integrations.httpx_client import AsyncOAuth2Client
from returns.result import Failure, Success

from app.config import get_settings
from app.shared.result import (
    app_error_to_exception,
    log_expected_failure,
)
from app.utils import (
    ConflictException,
    NotFoundException,
    UnauthorizedException,
    logger,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from .dto import LoginRequest, RegisterRequest
    from .repository import RefreshTokenRepository, UserRepository

from .dto import SessionResponse, TokenResponse, UserResponse
from .model import User
from .repository import SessionData
from .security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    fetch_oauth_userinfo,
    generate_token,
    get_oauth_config,
    hash_password,
    hash_token,
    needs_rehash,
    sign_oauth_state,
    verify_oauth_state,
    verify_password,
)

# Dummy hash used in the constant-time negative path during login.
# Prevents timing attacks that would reveal whether an email is registered.
_DUMMY_HASH = "$argon2id$v=19$m=65536,t=2,p=2$c29tZXNhbHRzb21lc2FsdA$dGVzdGhhc2h0ZXN0aGFzaA"


def _to_user_response(user: User) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        is_verified=user.is_verified,
        is_active=user.is_active,
        created_at=user.created_at,
    )


class AuthService:
    def __init__(
        self,
        user_repo: UserRepository,
        token_repo: RefreshTokenRepository,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._user_repo = user_repo
        self._token_repo = token_repo
        self._session_factory = session_factory

    async def register(self, dto: RegisterRequest) -> UserResponse:
        email_exists_result = await self._user_repo.email_exists(dto.email)
        if isinstance(email_exists_result, Failure):
            error = email_exists_result.failure()
            log_expected_failure(error, operation="email_exists")
            raise app_error_to_exception(error)
        exists = email_exists_result.unwrap()
        if exists:
            msg = "Email already registered"
            raise ConflictException(msg)

        verification_token = generate_token()
        user = User(
            email=dto.email.lower(),
            full_name=dto.full_name,
            hashed_password=hash_password(dto.password),
            verification_token_hash=hash_token(verification_token),
        )
        create_result = await self._user_repo.create(user)
        if isinstance(create_result, Failure):
            error = create_result.failure()
            log_expected_failure(error, operation="create_user")
            raise app_error_to_exception(error)
        resolved = create_result.unwrap()

        # send_verification_email.delay(
        #     user_id=str(resolved.id),
        #     email=resolved.email,
        #     token=verification_token,
        # )
        logger.bind(user_id=str(resolved.id)).info("User registered")
        return _to_user_response(resolved)

    async def login(
        self,
        dto: LoginRequest,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> TokenResponse:
        find_result = await self._user_repo.find_by_email(dto.email)
        if isinstance(find_result, Failure):
            error = find_result.failure()
            log_expected_failure(error, operation="find_user")
            raise app_error_to_exception(error)
        resolved = find_result.unwrap()
        if resolved is None or resolved.hashed_password is None:
            msg = "Invalid credentials"
            raise UnauthorizedException(msg)

        if not verify_password(resolved.hashed_password, dto.password):
            msg = "Invalid credentials"
            raise UnauthorizedException(msg)

        if not resolved.is_active:
            msg = "Account is disabled"
            raise UnauthorizedException(msg)

        if not resolved.is_verified:
            msg = "Email not verified. Check your inbox."
            raise UnauthorizedException(msg)

        # Transparent rehash: argon2 params may have been upgraded since this hash was created
        if needs_rehash(resolved.hashed_password):
            resolved.hashed_password = hash_password(dto.password)
            save_result = await self._user_repo.save(resolved)
            if isinstance(save_result, Failure):
                error = save_result.failure()
                log_expected_failure(error, operation="save_user")
                raise app_error_to_exception(error)

        return await self._create_session(
            user=resolved,
            device_name=dto.device_name,
            ip=ip,
            user_agent=user_agent,
        )

    async def logout(self, refresh_token: str) -> None:
        claims = decode_token(refresh_token)
        if claims.token_type != "refresh":
            msg = "Not a refresh token"
            raise UnauthorizedException(msg)
        revoke_result = await self._token_repo.revoke_session(
            session_id=claims.jti,
            user_id=claims.sub,
            reason="logout",
        )
        if isinstance(revoke_result, Failure):
            log_expected_failure(revoke_result.failure(), operation="revoke_session")
            raise app_error_to_exception(revoke_result.failure())
        logger.bind(user_id=claims.sub, session_id=claims.jti).info("Session revoked")

    async def refresh(self, refresh_token: str) -> TokenResponse:
        claims = decode_token(refresh_token)
        if claims.token_type != "refresh":
            msg = "Not a refresh token"
            raise UnauthorizedException(msg)

        # Redis lookup is the revocation gate — if the session was deleted, deny
        session_result = await self._token_repo.get_session(claims.jti)
        if isinstance(session_result, Success) and session_result.unwrap() is not None:
            pass  # session is valid
        else:
            msg = "Session expired or revoked"
            raise UnauthorizedException(msg)

        user_result = await self._user_repo.find_by_id(claims.sub)
        if isinstance(user_result, Failure):
            error = user_result.failure()
            log_expected_failure(error, operation="refresh_user_lookup")
            raise app_error_to_exception(error)
        resolved_user = user_result.unwrap()
        if resolved_user is None:
            msg = "User not found or disabled"
            raise UnauthorizedException(msg)

        if not resolved_user.is_active:
            msg = "User not found or disabled"
            raise UnauthorizedException(msg)

        access_token, expires_in = create_access_token(
            user_id=str(resolved_user.id),
            session_id=claims.jti,
            role=resolved_user.role,
            permissions=resolved_user.get_permissions(),
        )
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,  # no rotation per config
            token_type="bearer",
            expires_in=expires_in,
        )

    async def verify_email(self, token: str) -> None:
        verify_result = await self._user_repo.find_by_verification_token_hash(hash_token(token))
        if isinstance(verify_result, Failure):
            error = verify_result.failure()
            log_expected_failure(error, operation="verify_email")
            raise app_error_to_exception(error)
        resolved = verify_result.unwrap()
        if resolved is None:
            msg = "Invalid or expired verification token"
            raise NotFoundException(msg)
        resolved.is_verified = True
        resolved.verification_token_hash = None
        save_result = await self._user_repo.save(resolved)
        if isinstance(save_result, Failure):
            error = save_result.failure()
            log_expected_failure(error, operation="save_user")
            raise app_error_to_exception(error)
        logger.bind(user_id=str(resolved.id)).info("Email verified")

    async def resend_verification(self, email: str) -> None:
        find_result = await self._user_repo.find_by_email(email)
        if isinstance(find_result, Success):
            resolved = find_result.unwrap()
        else:
            return  # silent — don't reveal email existence

        if resolved.is_verified:  # ty: ignore[unresolved-attribute]
            msg = "Email already verified"
            raise ConflictException(msg)

        new_token = generate_token()
        resolved.verification_token_hash = hash_token(new_token)  # ty: ignore[invalid-assignment]
        save_result = await self._user_repo.save(resolved)  # ty: ignore[invalid-argument-type]
        if isinstance(save_result, Failure):
            log_expected_failure(save_result.failure(), operation="save_user")
            raise app_error_to_exception(save_result.failure())

        await self._publish_outbox_event(
            aggregate_type="auth_email",
            aggregate_id=str(resolved.id),  # ty: ignore[unresolved-attribute]
            event_type="auth.send_verification_email",
            payload={"user_id": str(resolved.id), "email": resolved.email, "token": new_token},  # ty: ignore[unresolved-attribute]
        )

    async def forgot_password(self, email: str) -> None:
        find_result = await self._user_repo.find_by_email(email)
        if isinstance(find_result, Success):
            resolved = find_result.unwrap()
        else:
            return  # silent — identical response regardless of outcome

        if not resolved or not resolved.is_verified:
            return  # silent — identical response regardless of outcome

        settings = get_settings()
        reset_token = generate_token()
        resolved.reset_token_hash = hash_token(reset_token)
        resolved.reset_token_expires_at = datetime.now(UTC) + timedelta(
            minutes=settings.PASSWORD_RESET_EXPIRE_MINUTES,
        )
        save_result = await self._user_repo.save(resolved)
        if isinstance(save_result, Failure):
            log_expected_failure(save_result.failure(), operation="save_user")
            raise app_error_to_exception(save_result.failure())
        await self._publish_outbox_event(
            aggregate_type="auth_email",
            aggregate_id=str(resolved.id),
            event_type="auth.send_password_reset_email",
            payload={"user_id": str(resolved.id), "email": resolved.email, "token": reset_token},
        )

    async def reset_password(self, token: str, new_password: str) -> None:
        reset_result = await self._user_repo.find_by_reset_token_hash(hash_token(token))
        if isinstance(reset_result, Failure):
            error = reset_result.failure()
            log_expected_failure(error, operation="reset_password")
            raise app_error_to_exception(error)
        resolved = reset_result.unwrap()
        if resolved is None:
            msg = "Invalid or expired reset token"
            raise NotFoundException(msg)

        if (
            resolved.reset_token_expires_at is None
            or resolved.reset_token_expires_at < datetime.now(UTC)
        ):
            msg = "Reset token has expired"
            raise UnauthorizedException(msg)

        resolved.hashed_password = hash_password(new_password)
        resolved.reset_token_hash = None
        resolved.reset_token_expires_at = None
        save_result = await self._user_repo.save(resolved)
        if isinstance(save_result, Failure):
            log_expected_failure(save_result.failure(), operation="save_user")
            raise app_error_to_exception(save_result.failure())

        # Force all sessions offline after a password reset
        revoke_result = await self._token_repo.revoke_all_user_sessions(
            user_id=str(resolved.id),
            reason="password_reset",
        )
        if isinstance(revoke_result, Failure):
            log_expected_failure(revoke_result.failure(), operation="revoke_all_user_sessions")
            raise app_error_to_exception(revoke_result.failure())
        logger.bind(user_id=str(resolved.id)).info("Password reset — all sessions revoked")

    @staticmethod
    async def oauth_get_authorization_url(provider: str) -> tuple[str, str]:
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

        return str(url), sign_oauth_state(state, provider)

    async def oauth_callback(
        self,
        provider: str,
        code: str,
        state: str,
        signed_state: str,
        *,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> TokenResponse:
        if not verify_oauth_state(signed_state, state, provider):
            msg = "Invalid OAuth state — possible CSRF attack"
            raise UnauthorizedException(msg)

        config = get_oauth_config(provider)
        userinfo = await fetch_oauth_userinfo(provider, config, code)

        oauth_result = await self._user_repo.find_or_create_oauth_user(
            email=userinfo.email,
            provider=provider,
            provider_user_id=userinfo.provider_user_id,
            provider_email=userinfo.email,
            full_name=userinfo.full_name,
        )
        if isinstance(oauth_result, Failure):
            error = oauth_result.failure()
            log_expected_failure(error, operation="find_or_create_oauth_user")
            raise app_error_to_exception(error)
        resolved_user, was_created = oauth_result.unwrap()

        if not resolved_user.is_active:
            msg = "Account is disabled"
            raise UnauthorizedException(msg)

        logger.bind(user_id=str(resolved_user.id), provider=provider, created=was_created).info(
            "OAuth login"
        )
        return await self._create_session(user=resolved_user, ip=ip, user_agent=user_agent)

    async def list_sessions(
        self,
        user_id: str,
        current_session_id: str | None = None,
    ) -> list[SessionResponse]:
        sessions_result = await self._token_repo.get_user_sessions(user_id)
        if isinstance(sessions_result, Failure):
            error = sessions_result.failure()
            log_expected_failure(error, operation="get_user_sessions")
            raise app_error_to_exception(error)
        sessions = sessions_result.unwrap()
        return [
            SessionResponse(
                session_id=s.session_id,
                device_id=s.device_id,
                device_name=s.device_name,
                ip_address=s.ip_address,
                created_at=s.created_at,
                expires_at=s.expires_at,
                is_current=s.session_id == current_session_id,
            )
            for s in sessions
        ]

    async def revoke_session(
        self,
        session_id: str,
        user_id: str,
    ) -> None:
        session_result = await self._token_repo.get_session(session_id)
        if isinstance(session_result, Failure):
            error = session_result.failure()
            log_expected_failure(error, operation="get_session")
            raise app_error_to_exception(error)
        resolved_session = session_result.unwrap()
        if resolved_session is None:
            msg = "Session not found"
            raise NotFoundException(msg)
        if resolved_session.user_id != user_id:
            msg = "Cannot revoke another user's session"
            raise UnauthorizedException(msg)
        revoke_result = await self._token_repo.revoke_session(
            session_id=session_id,
            user_id=user_id,
            reason="manual_revoke",
        )
        if isinstance(revoke_result, Failure):
            log_expected_failure(revoke_result.failure(), operation="revoke_session")
            raise app_error_to_exception(revoke_result.failure())

    async def revoke_all_sessions(
        self,
        user_id: str,
        except_session_id: str | None = None,
    ) -> None:
        revoke_result = await self._token_repo.revoke_all_user_sessions(
            user_id=user_id,
            except_session_id=except_session_id,
            reason="revoke_all",
        )
        if isinstance(revoke_result, Failure):
            log_expected_failure(revoke_result.failure(), operation="revoke_all_user_sessions")
            raise app_error_to_exception(revoke_result.failure())

    async def _create_session(
        self,
        user: User,
        device_name: str | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> TokenResponse:
        settings = get_settings()
        session_id = str(uuid4())
        device_id = str(uuid4())
        now = datetime.now(UTC)

        access_token, expires_in = create_access_token(
            user_id=str(user.id),
            session_id=session_id,
            role=user.role,
            permissions=user.get_permissions(),
        )
        refresh_token, refresh_ttl = create_refresh_token(
            user_id=str(user.id),
            session_id=session_id,
            device_id=device_id,
        )
        store_result = await self._token_repo.store_session(
            SessionData(
                session_id=session_id,
                user_id=str(user.id),
                device_id=device_id,
                device_name=device_name,
                ip_address=ip,
                user_agent=user_agent,
                created_at=now,
                expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
                ttl=refresh_ttl,
            )
        )
        if isinstance(store_result, Failure):
            log_expected_failure(store_result.failure(), operation="store_session")
            raise app_error_to_exception(store_result.failure())
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=expires_in,
        )

    async def _publish_outbox_event(
        self,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        from app.shared.outbox import (
            with_outbox,
        )

        if self._session_factory is not None:
            async with self._session_factory() as session:
                await with_outbox(
                    session=session,
                    aggregate_type=aggregate_type,
                    aggregate_id=aggregate_id,
                    event_type=event_type,
                    payload=payload,
                )
            return

        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            create_async_engine,
        )

        from app.connections.postgres import (
            get_database_url,
        )

        engine = create_async_engine(get_database_url())
        try:
            async with engine.begin() as conn:
                session_ = AsyncSession(bind=conn)
                await with_outbox(
                    session=session_,
                    aggregate_type=aggregate_type,
                    aggregate_id=aggregate_id,
                    event_type=event_type,
                    payload=payload,
                )
        finally:
            await engine.dispose()
