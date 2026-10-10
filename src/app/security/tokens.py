"""Token claim types and JWT encode/decode. No domain imports.

Role and permissions travel as plain strings (the JWT wire format), so this
module never imports the auth feature's `UserRole`/`Permission` enums —
callers pass `.value`. The default role below is the token contract's default,
not auth policy.
"""

import time
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from joserfc._rfc7518.oct_key import OctKey  # noqa: PLC2701
from joserfc.errors import ExpiredTokenError, JoseError
from joserfc.jwt import JWTClaimsRegistry
from joserfc.jwt import decode as jwt_decode
from joserfc.jwt import encode as jwt_encode
from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.utils import UnauthorizedException

if TYPE_CHECKING:
    from typing import Any

_JWT_HEADER: dict[str, str] = {"alg": "HS256"}

_DEFAULT_ROLE = "user"


class TokenClaims(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub: str
    jti: str
    sid: str | None
    role: str
    permissions: list[str]
    token_type: str
    device_id: str | None = None
    impersonated_by: str | None = None  # ← new


def _jwt_key() -> bytes:
    return get_settings().JWT_SECRET_KEY.get_secret_value().encode()


def _jwt_oct_key() -> OctKey:
    k = _jwt_key()
    return OctKey(k, k)


def create_access_token(
    user_id: str,
    session_id: str,
    role: str,
    permissions: frozenset[str],
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
        "role": role,
        "permissions": list(permissions),
        "type": "access",
    }
    token = jwt_encode(_JWT_HEADER, claims, _jwt_oct_key())
    return token, expire_secs


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
    token = jwt_encode(_JWT_HEADER, claims, _jwt_oct_key())
    return token, expire_secs


def create_impersonation_token(
    target_user_id: str,
    target_role: str,
    target_permissions: frozenset[str],
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
        "role": target_role,
        "permissions": list(target_permissions),
        "type": "access",
        "impersonated_by": admin_user_id,
    }
    token = jwt_encode(_JWT_HEADER, claims, _jwt_oct_key())
    return token, expire_secs


def decode_token(token: str) -> TokenClaims:
    """Decode and validate a JWT. Raises UnauthorizedException on any failure."""
    try:
        token_obj = jwt_decode(token, _jwt_oct_key())
        JWTClaimsRegistry().validate(token_obj.claims)
    except ExpiredTokenError as exc:
        msg = "Token has expired"
        raise UnauthorizedException(msg) from exc
    except JoseError as exc:
        msg = "Invalid token"
        raise UnauthorizedException(msg) from exc

    return TokenClaims(
        sub=token_obj.claims["sub"],
        jti=token_obj.claims["jti"],
        sid=token_obj.claims.get("sid"),
        role=token_obj.claims.get("role", _DEFAULT_ROLE),
        permissions=token_obj.claims.get("permissions", []),
        token_type=token_obj.claims.get("type", "access"),
        device_id=token_obj.claims.get("device_id"),
        impersonated_by=token_obj.claims.get("impersonated_by"),  # ← new
    )


# ── OAuth2 state signing (stateless signed cookie) ────────────────────────────

OAUTH_STATE_COOKIE = "oauth_state"


def _state_signer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().OAUTH_STATE_SECRET.get_secret_value())


def sign_oauth_state(state: str, provider: str) -> str:
    return _state_signer().dumps({"state": state, "provider": provider})


def verify_oauth_state(
    signed: str,
    state: str,
    provider: str,
    max_age: int = 300,
) -> bool:
    try:
        data: dict[str, Any] = _state_signer().loads(signed, max_age=max_age)
        return data.get("state") == state and data.get("provider") == provider
    except (BadSignature, SignatureExpired):
        return False
