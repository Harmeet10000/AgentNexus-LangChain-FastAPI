import secrets
from datetime import datetime, timedelta
from uuid import uuid4

from authlib.integrations.httpx_client import AsyncOAuth2Client

from app.config import get_settings
from app.utils import logger
from app.utils.exceptions import (
    ConflictException,
    NotFoundException,
    UnauthorizedException,
)
from tasks.auth_email_tasks import send_password_reset_email, send_verification_email

from .dto import (
    LoginRequest,
    RegisterRequest,
    SessionResponse,
    TokenResponse,
    UserResponse,
)
from .model import User
from .repository import RefreshTokenRepository, SessionData, UserRepository
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
    ) -> None:
        self._user_repo = user_repo
        self._token_repo = token_repo

    async def register(self, dto: RegisterRequest) -> UserResponse:
        if await self._user_repo.email_exists(dto.email):
            raise ConflictException("Email already registered")

        verification_token = generate_token()
        user = User(
            email=dto.email.lower(),
            full_name=dto.full_name,
            hashed_password=hash_password(dto.password),
            verification_token_hash=hash_token(verification_token),
        )
        user = await self._user_repo.create(user)

        # send_verification_email.delay(
        #     user_id=str(user.id),
        #     email=user.email,
        #     token=verification_token,
        # )
        logger.bind(user_id=str(user.id)).info("User registered")
        return _to_user_response(user)

    async def login(
        self,
        dto: LoginRequest,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> TokenResponse:
        user = await self._user_repo.find_by_email(dto.email)

        # Always run verify_password to normalize response time regardless of outcome.
        # Without this, a missing user returns ~0ms; a found user returns ~100ms — leaking email existence.
        if user is None or user.hashed_password is None:
            verify_password(_DUMMY_HASH, dto.password)
            raise UnauthorizedException("Invalid credentials")

        if not verify_password(user.hashed_password, dto.password):
            raise UnauthorizedException("Invalid credentials")

        if not user.is_active:
            raise UnauthorizedException("Account is disabled")

        if not user.is_verified:
            raise UnauthorizedException("Email not verified. Check your inbox.")

        # Transparent rehash: argon2 params may have been upgraded since this hash was created
        if needs_rehash(user.hashed_password):
            user.hashed_password = hash_password(dto.password)
            await self._user_repo.save(user)

        return await self._create_session(
            user=user,
            device_name=dto.device_name,
            ip=ip,
            user_agent=user_agent,
        )

    async def logout(self, refresh_token: str) -> None:
        claims = decode_token(refresh_token)
        if claims.token_type != "refresh":
            raise UnauthorizedException("Not a refresh token")
        await self._token_repo.revoke_session(
            session_id=claims.jti,
            user_id=claims.sub,
            reason="logout",
        )
        logger.bind(user_id=claims.sub, session_id=claims.jti).info("Session revoked")

    async def refresh(self, refresh_token: str) -> TokenResponse:
        claims = decode_token(refresh_token)
        if claims.token_type != "refresh":
            raise UnauthorizedException("Not a refresh token")

        # Redis lookup is the revocation gate — if the session was deleted, deny
        session = await self._token_repo.get_session(claims.jti)
        if session is None:
            raise UnauthorizedException("Session expired or revoked")

        user = await self._user_repo.find_by_id(claims.sub)
        if user is None or not user.is_active:
            raise UnauthorizedException("User not found or disabled")

        access_token, expires_in = create_access_token(
            user_id=str(user.id),
            session_id=claims.jti,
            role=user.role,
            permissions=user.get_permissions(),
        )
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,  # no rotation per config
            token_type="bearer",
            expires_in=expires_in,
        )

    async def verify_email(self, token: str) -> None:
        user = await self._user_repo.find_by_verification_token_hash(hash_token(token))
        if user is None:
            raise NotFoundException("Invalid or expired verification token")
        user.is_verified = True
        user.verification_token_hash = None
        await self._user_repo.save(user)
        logger.bind(user_id=str(user.id)).info("Email verified")

    async def resend_verification(self, email: str) -> None:
        user = await self._user_repo.find_by_email(email)
        if user is None:
            return  # silent — don't reveal email existence

        if user.is_verified:
            raise ConflictException("Email already verified")

        new_token = generate_token()
        user.verification_token_hash = hash_token(new_token)
        await self._user_repo.save(user)

        send_verification_email.delay(
            user_id=str(user.id),
            email=user.email,
            token=new_token,
        )

    async def forgot_password(self, email: str) -> None:
        user = await self._user_repo.find_by_email(email)
        if user is None or not user.is_verified:
            return  # silent — identical response regardless of outcome

        settings = get_settings()
        reset_token = generate_token()
        user.reset_token_hash = hash_token(reset_token)
        user.reset_token_expires_at = datetime.now(datetime.timezone.utc) + timedelta(
            minutes=settings.PASSWORD_RESET_EXPIRE_MINUTES,
        )
        await self._user_repo.save(user)
        send_password_reset_email.delay(
            user_id=str(user.id),
            email=user.email,
            token=reset_token,
        )

    async def reset_password(self, token: str, new_password: str) -> None:
        user = await self._user_repo.find_by_reset_token_hash(hash_token(token))
        if user is None:
            raise NotFoundException("Invalid or expired reset token")

        if user.reset_token_expires_at is None or user.reset_token_expires_at < datetime.utcnow():
            raise UnauthorizedException("Reset token has expired")

        user.hashed_password = hash_password(new_password)
        user.reset_token_hash = None
        user.reset_token_expires_at = None
        await self._user_repo.save(user)

        # Force all sessions offline after a password reset
        await self._token_repo.revoke_all_user_sessions(
            user_id=str(user.id),
            reason="password_reset",
        )
        logger.bind(user_id=str(user.id)).info("Password reset — all sessions revoked")

    async def oauth_get_authorization_url(self, provider: str) -> tuple[str, str]:
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
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> TokenResponse:
        if not verify_oauth_state(signed_state, state, provider):
            raise UnauthorizedException("Invalid OAuth state — possible CSRF attack")

        config = get_oauth_config(provider)
        userinfo = await fetch_oauth_userinfo(provider, config, code)

        user, created = await self._user_repo.find_or_create_oauth_user(
            email=userinfo.email,
            provider=provider,
            provider_user_id=userinfo.provider_user_id,
            provider_email=userinfo.email,
            full_name=userinfo.full_name,
        )

        if not user.is_active:
            raise UnauthorizedException("Account is disabled")

        logger.bind(user_id=str(user.id), provider=provider, created=created).info("OAuth login")
        return await self._create_session(user=user, ip=ip, user_agent=user_agent)

    async def list_sessions(
        self,
        user_id: str,
        current_session_id: str | None = None,
    ) -> list[SessionResponse]:
        sessions = await self._token_repo.get_user_sessions(user_id)
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
        session = await self._token_repo.get_session(session_id)
        if session is None:
            raise NotFoundException("Session not found")
        if session.user_id != user_id:
            raise UnauthorizedException("Cannot revoke another user's session")
        await self._token_repo.revoke_session(
            session_id=session_id,
            user_id=user_id,
            reason="manual_revoke",
        )

    async def revoke_all_sessions(
        self,
        user_id: str,
        except_session_id: str | None = None,
    ) -> None:
        await self._token_repo.revoke_all_user_sessions(
            user_id=user_id,
            except_session_id=except_session_id,
            reason="revoke_all",
        )

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
        now = datetime.now(datetime.timezone.utc)

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
        await self._token_repo.store_session(
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
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=expires_in,
        )
