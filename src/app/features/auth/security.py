import hashlib
import secrets
import time
from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.jose import jwt
from authlib.jose.errors import ExpiredTokenError, JoseError
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import get_settings
from app.features.auth.model import Permission, UserRole
from app.utils import UnauthorizedException, ValidationException

# ── Password hashing ──────────────────────────────────────────────────────────

_ph = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MB
    parallelism=2,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(hashed: str, plain: str) -> bool:
    try:
        return _ph.verify(hashed, plain)
    except VerifyMismatchError:
        return False
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(hashed: str) -> bool:
    """True when stored hash was created with weaker argon2 params than current config."""
    return _ph.check_needs_rehash(hashed)


# ── One-way token hashing (email / reset tokens) ──────────────────────────────


def generate_token() -> str:
    """Cryptographically secure URL-safe token for email delivery."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """SHA-256 digest stored in DB; the raw token travels only via email."""
    return hashlib.sha256(token.encode()).hexdigest()


# ── JWT ───────────────────────────────────────────────────────────────────────

_JWT_HEADER: dict[str, str] = {"alg": "HS256"}


@dataclass(frozen=True)
class TokenClaims:
    sub: str
    jti: str
    sid: str | None
    role: str
    permissions: list[str]
    token_type: str
    device_id: str | None = None
    impersonated_by: str | None = None  # ← new


def _jwt_key() -> bytes:
    return get_settings().JWT_SECRET_KEY.encode()


def create_access_token(
    user_id: str,
    session_id: str,
    role: UserRole,
    permissions: frozenset[Permission],
    expire_minutes: int | None = None,
) -> tuple[str, int]:
    """Return (encoded_jwt, expires_in_seconds)."""
    settings = get_settings()
    expire_mins = expire_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    now = int(time.time())
    expire_secs = int(timedelta(minutes=expire_mins).total_seconds())

    claims = {
        "sub": user_id,
        "iss": settings.JWT_ISSUER,
        "iat": now,
        "exp": now + expire_secs,
        "jti": str(uuid4()),
        "sid": session_id,  # session_id for /sessions listing without a DB hit
        "role": role.value,
        "permissions": [p.value for p in permissions],
        "type": "access",
    }
    token: bytes = jwt.encode(_JWT_HEADER, claims, _jwt_key())
    return token.decode("utf-8"), expire_secs


def create_refresh_token(
    user_id: str,
    session_id: str,
    device_id: str,
    expire_days: int | None = None,
) -> tuple[str, int]:
    """Return (encoded_jwt, expires_in_seconds). session_id is stored as jti."""
    settings = get_settings()
    expire_d = expire_days or settings.REFRESH_TOKEN_EXPIRE_DAYS
    now = int(time.time())
    expire_secs = int(timedelta(days=expire_d).total_seconds())

    claims = {
        "sub": user_id,
        "iss": settings.JWT_ISSUER,
        "iat": now,
        "exp": now + expire_secs,
        "jti": session_id,
        "device_id": device_id,
        "type": "refresh",
    }
    token: bytes = jwt.encode(_JWT_HEADER, claims, _jwt_key())
    return token.decode("utf-8"), expire_secs

def create_impersonation_token(
    target_user_id: str,
    target_role: UserRole,
    target_permissions: frozenset[Permission],
    admin_user_id: str,
    expire_minutes: int = 15,
) -> tuple[str, int]:
    """Short-lived access token embedding who impersonated whom.

    No refresh token is issued. No session is stored in Redis.
    The 15-minute TTL IS the revocation mechanism — no blacklist needed.
    """
    settings = get_settings()
    now = int(time.time())
    expire_secs = int(timedelta(minutes=expire_minutes).total_seconds())

    claims = {
        "sub": target_user_id,
        "iss": settings.JWT_ISSUER,
        "iat": now,
        "exp": now + expire_secs,
        "jti": str(uuid4()),
        "sid": None,
        "role": target_role.value,
        "permissions": [p.value for p in target_permissions],
        "type": "access",
        "impersonated_by": admin_user_id,
    }
    token: bytes = jwt.encode(_JWT_HEADER, claims, _jwt_key())
    return token.decode("utf-8"), expire_secs


def decode_token(token: str) -> TokenClaims:
    """Decode and validate a JWT. Raises UnauthorizedException on any failure."""
    try:
        claims = jwt.decode(token.encode("utf-8"), _jwt_key())
        claims.validate()
    except ExpiredTokenError as exc:
        raise UnauthorizedException("Token has expired") from exc
    except JoseError as exc:
        raise UnauthorizedException("Invalid token") from exc

    return TokenClaims(
        sub=claims["sub"],
        jti=claims["jti"],
        sid=claims.get("sid"),
        role=claims.get("role", UserRole.USER.value),
        permissions=claims.get("permissions", []),
        token_type=claims.get("type", "access"),
        device_id=claims.get("device_id"),
        impersonated_by=claims.get("impersonated_by"),  # ← new
    )


# ── OAuth2 state signing (stateless signed cookie) ────────────────────────────

OAUTH_STATE_COOKIE = "oauth_state"
_SUPPORTED_PROVIDERS = frozenset({"google", "github"})


def _state_signer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().OAUTH_STATE_SECRET)


def sign_oauth_state(state: str, provider: str) -> str:
    return _state_signer().dumps({"state": state, "provider": provider})


def verify_oauth_state(
    signed: str,
    state: str,
    provider: str,
    max_age: int = 300,
) -> bool:
    try:
        data: dict = _state_signer().loads(signed, max_age=max_age)
        return data.get("state") == state and data.get("provider") == provider
    except (BadSignature, SignatureExpired):
        return False


# ── OAuth2 provider configuration ─────────────────────────────────────────────


@dataclass(frozen=True)
class OAuthProviderConfig:
    client_id: str
    client_secret: str
    authorization_endpoint: str
    token_endpoint: str
    scope: str
    redirect_uri: str


def get_oauth_config(provider: str) -> OAuthProviderConfig:
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValidationException(f"Unsupported OAuth provider: {provider}")

    settings = get_settings()
    base = settings.BACKEND_URL.rstrip("/")

    match provider:
        case "google":
            return OAuthProviderConfig(
                client_id=settings.GOOGLE_CLIENT_ID,
                client_secret=settings.GOOGLE_CLIENT_SECRET,
                authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
                token_endpoint="https://oauth2.googleapis.com/token",
                scope="openid email profile",
                redirect_uri=f"{base}/api/v1/auth/oauth/google/callback",
            )
        case "github":
            return OAuthProviderConfig(
                client_id=settings.GITHUB_CLIENT_ID,
                client_secret=settings.GITHUB_CLIENT_SECRET,
                authorization_endpoint="https://github.com/login/oauth/authorize",
                token_endpoint="https://github.com/login/oauth/access_token",
                scope="read:user user:email",
                redirect_uri=f"{base}/api/v1/auth/oauth/github/callback",
            )
        case _:
            raise ValidationException(f"Unsupported OAuth provider: {provider}")


# ── OAuth2 userinfo normalization ─────────────────────────────────────────────


@dataclass(frozen=True)
class OAuthUserInfo:
    email: str
    provider_user_id: str
    full_name: str | None = None


async def fetch_oauth_userinfo(
    provider: str,
    config: OAuthProviderConfig,
    code: str,
) -> OAuthUserInfo:
    """Exchange authorization code for token, then fetch and normalize userinfo."""
    async with AsyncOAuth2Client(
        client_id=config.client_id,
        client_secret=config.client_secret,
    ) as client:
        fetch_kwargs: dict = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": config.redirect_uri,
        }
        # GitHub returns form-encoded by default; force JSON
        if provider == "github":
            fetch_kwargs["headers"] = {"Accept": "application/json"}

        await client.fetch_token(config.token_endpoint, **fetch_kwargs)

        if provider == "google":
            resp = await client.get("https://openidconnect.googleapis.com/v1/userinfo")
            data: dict = resp.json()
            return OAuthUserInfo(
                email=data["email"],
                provider_user_id=data["sub"],
                full_name=data.get("name"),
            )

        # GitHub: primary email may not be public — requires separate emails endpoint
        resp = await client.get(
            "https://api.github.com/user",
            headers={"Accept": "application/vnd.github+json"},
        )
        user_data: dict = resp.json()
        email: str | None = user_data.get("email")

        if not email:
            emails_resp = await client.get(
                "https://api.github.com/user/emails",
                headers={"Accept": "application/vnd.github+json"},
            )
            email = next(
                (e["email"] for e in emails_resp.json() if e.get("primary") and e.get("verified")),
                None,
            )

        if not email:
            raise ValidationException("GitHub account has no verified primary email")

        return OAuthUserInfo(
            email=email,
            provider_user_id=str(user_data["id"]),
            full_name=user_data.get("name"),
        )
