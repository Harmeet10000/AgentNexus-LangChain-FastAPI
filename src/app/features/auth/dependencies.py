from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.connections import get_mongodb, get_redis
from app.features.auth.model import Permission, User, UserRole
from app.features.auth.repository import RefreshTokenRepository, UserRepository
from app.features.auth.security import TokenClaims, decode_token
from app.features.auth.service import AuthService
from app.utils.exceptions import ForbiddenException, UnauthorizedException

_http_bearer = HTTPBearer(auto_error=False)


# ── Repository and service wiring ─────────────────────────────────────────────


async def get_user_repository(
    db=Depends(get_mongodb),
) -> UserRepository:
    return UserRepository(db)


async def get_refresh_token_repository(
    redis=Depends(get_redis),
) -> RefreshTokenRepository:
    return RefreshTokenRepository(redis)


async def get_auth_service(
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
) -> AuthService:
    return AuthService(user_repo, token_repo)


# ── Token extraction ──────────────────────────────────────────────────────────


async def _extract_raw_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_http_bearer)],
) -> str:
    """Authorization: Bearer header takes priority; falls back to access_token cookie."""
    if credentials is not None:
        return credentials.credentials
    cookie_token: str | None = request.cookies.get("access_token")
    if cookie_token:
        return cookie_token
    raise UnauthorizedException("Not authenticated")


async def get_token_claims(
    raw_token: Annotated[str, Depends(_extract_raw_token)],
) -> TokenClaims:
    """Decode and validate JWT claims. Zero database round trips."""
    claims: TokenClaims = decode_token(raw_token)
    if claims.token_type != "access":
        raise UnauthorizedException("Expected an access token")
    return claims


async def get_current_user(
    claims: Annotated[TokenClaims, Depends(get_token_claims)],
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
) -> User:
    """Full user hydration from MongoDB. Use when the handler needs live user state."""
    user: User | None = await user_repo.find_by_id(claims.sub)
    if user is None:
        raise UnauthorizedException("User not found")
    return user


async def get_current_active_user(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not user.is_active:
        raise UnauthorizedException("Account is disabled")
    return user


async def get_current_verified_user(
    user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    if not user.is_verified:
        raise UnauthorizedException("Email not verified")
    return user


# ── RBAC guards (claims-based — no DB hit) ────────────────────────────────────


def require_permission(*permissions: Permission):
    """Dependency factory. Validates permissions from JWT claims without a DB round trip.

    Usage: Depends(require_permission(Permission.USERS_READ, Permission.USERS_WRITE))
    """

    async def _guard(
        claims: Annotated[TokenClaims, Depends(get_token_claims)],
    ) -> TokenClaims:
        user_perms = set(claims.permissions)
        for perm in permissions:
            if perm.value not in user_perms:
                raise ForbiddenException(f"Missing required permission: {perm.value}")
        return claims

    return _guard


def require_role(*roles: UserRole):
    """Dependency factory. Validates role from JWT claims without a DB round trip.

    Usage: Depends(require_role(UserRole.ADMIN, UserRole.MODERATOR))
    """

    async def _guard(
        claims: Annotated[TokenClaims, Depends(get_token_claims)],
    ) -> TokenClaims:
        if claims.role not in {r.value for r in roles}:
            raise ForbiddenException("Insufficient role for this operation")
        return claims

    return _guard


# ── Annotated aliases — import and use directly in router signatures ───────────

CurrentUser = Annotated[User, Depends(get_current_active_user)]
CurrentVerifiedUser = Annotated[User, Depends(get_current_verified_user)]
CurrentClaims = Annotated[TokenClaims, Depends(get_token_claims)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
