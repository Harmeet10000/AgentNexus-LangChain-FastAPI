from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated, assert_never

from fastapi import Depends, Request, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from returns.result import Failure, Success
from starlette.requests import HTTPConnection

from app.connections.redis import get_redis
from app.observability.logging import set_request_actor
from app.security.tokens import TokenClaims, decode_token
from app.utils import (
    ConflictException,
    ForbiddenException,
    InfrastructureException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)

from .account_store import (
    AuthAccount,
    AuthAccountStore,
    BeanieAuthAccountStore,
)
from .exceptions import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthResult,
    AuthValidationError,
)
from .models import Permission, UserRole
from .repository import RefreshTokenRepository
from .service import AuthService

_http_bearer = HTTPBearer(auto_error=False)
_ACCESS_TOKEN_TYPE = "access"  # noqa: S105 — token type constant, not a password


# ── Repository and service wiring ─────────────────────────────────────────────


async def get_account_store() -> AuthAccountStore:
    """Production MongoDB/Beanie adapter for AuthAccountStore."""
    return BeanieAuthAccountStore()


async def get_refresh_token_repository(
    redis=Depends(get_redis),
) -> RefreshTokenRepository:
    return RefreshTokenRepository(redis)


async def get_auth_service(
    account_store: Annotated[AuthAccountStore, Depends(get_account_store)],
    token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
    connection: HTTPConnection,
) -> AuthService:
    session_factory = getattr(connection.app.state, "db_session_local", None)
    return AuthService(account_store, token_repo, session_factory=session_factory)


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
    msg = "Not authenticated"
    raise UnauthorizedException(msg)


def _read_authorization_header_token(connection: HTTPConnection) -> str | None:
    authorization = connection.headers.get("authorization")
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def extract_raw_token_from_connection(connection: HTTPConnection) -> str:
    """Read a bearer token from an HTTP or WebSocket connection."""
    header_token = _read_authorization_header_token(connection)
    if header_token:
        return header_token

    cookie_token: str | None = connection.cookies.get("access_token")
    if cookie_token:
        return cookie_token

    msg = "Not authenticated"
    raise UnauthorizedException(msg)


async def get_token_claims(
    raw_token: Annotated[str, Depends(_extract_raw_token)],
    token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
) -> TokenClaims:
    """Decode claims and verify the Redis-backed session before user hydration.

    ``sid=None`` is reserved for short-lived stateless impersonation tokens,
    which intentionally use their JWT expiry as the revocation boundary.
    """
    claims: TokenClaims = decode_token(raw_token)
    if claims.token_type != _ACCESS_TOKEN_TYPE:
        msg = "Expected an access token"
        raise UnauthorizedException(msg)
    if claims.sid is not None:
        session_result = await token_repo.get_session(claims.sid)
        if isinstance(session_result, Failure):
            error = session_result.failure()
            if isinstance(error, AuthInfrastructureError):
                raise InfrastructureException(
                    detail="Session store unavailable",
                    error_code=error.code,
                    retryable=error.retryable,
                    data=error.details,
                )
            msg = "Unable to verify session"
            raise UnauthorizedException(msg)
        session = session_result.unwrap()
        if (
            session is None
            or session.user_id != claims.sub
            or session.expires_at <= datetime.now(UTC)
            or (
                claims.authz_revision is not None
                and session.authz_revision != claims.authz_revision
            )
        ):
            msg = "Session expired or revoked"
            raise UnauthorizedException(msg)
    set_request_actor(claims.sub)
    return claims


async def get_websocket_token_claims(websocket: WebSocket) -> TokenClaims:
    """Decode access-token claims during WebSocket handshake."""
    raw_token = extract_raw_token_from_connection(websocket)
    claims = decode_token(raw_token)
    if claims.token_type != _ACCESS_TOKEN_TYPE:
        msg = "Expected an access token"
        raise UnauthorizedException(msg)
    set_request_actor(claims.sub)
    return claims


async def get_current_user_result(
    claims: Annotated[TokenClaims, Depends(get_token_claims)],
    account_store: Annotated[AuthAccountStore, Depends(get_account_store)],
) -> AuthResult[AuthAccount]:
    """Hydrate the current user while preserving repository failures as typed Results."""
    result = await account_store.find_by_id(claims.sub)
    if isinstance(result, Failure):
        error = result.failure()
        if isinstance(error, AuthNotFoundError | AuthValidationError):
            return Failure(
                AuthAuthenticationError(
                    message="User not found",
                    details={"user_id": claims.sub},
                    source="auth_dependency",
                )
            )
        return Failure(error)
    account = result.unwrap()
    if account is None:
        return Failure(
            AuthAuthenticationError(
                message="User not found",
                details={"user_id": claims.sub},
                source="auth_dependency",
            )
        )
    return Success(account)


async def get_current_user(
    result: Annotated[AuthResult[AuthAccount], Depends(get_current_user_result)],
) -> AuthAccount:
    """Resolve the typed user result at FastAPI's exception-native dependency boundary."""
    if isinstance(result, Failure):
        error = result.failure()
        match error:
            case AuthValidationError():
                raise ValidationException(
                    detail=error.message, error_code=error.code, data=error.details
                )
            case AuthNotFoundError():
                raise NotFoundException(
                    resource="Auth resource", identifier=None, error_code=error.code
                )
            case AuthConflictError():
                raise ConflictException(
                    detail=error.message, error_code=error.code, data=error.details
                )
            case AuthAuthenticationError():
                raise UnauthorizedException(detail=error.message, error_code=error.code)
            case AuthAuthorizationError():
                raise ForbiddenException(detail=error.message, error_code=error.code)
            case AuthInfrastructureError():
                raise InfrastructureException(
                    detail=error.message,
                    error_code=error.code,
                    retryable=error.retryable,
                    data=error.details,
                )
            case _ as unreachable:
                assert_never(unreachable)
    return result.unwrap()


async def get_current_active_user(
    account: Annotated[AuthAccount, Depends(get_current_user)],
) -> AuthAccount:
    if not account.is_active:
        msg = "Account is disabled"
        raise UnauthorizedException(msg)
    return account


async def get_current_verified_user(
    account: Annotated[AuthAccount, Depends(get_current_active_user)],
) -> AuthAccount:
    if not account.is_verified:
        msg = "Email not verified"
        raise UnauthorizedException(msg)
    return account


# ── RBAC guards (live account state) ─────────────────────────────────────────


def require_permission(*permissions: Permission) -> Callable[..., Awaitable[TokenClaims]]:
    """Dependency factory that checks live permissions after session validation.

    Usage: Depends(require_permission(Permission.USERS_READ, Permission.USERS_WRITE))
    """

    async def _guard(
        claims: Annotated[TokenClaims, Depends(get_token_claims)],
        account: Annotated[AuthAccount, Depends(get_current_active_user)],
    ) -> TokenClaims:
        if claims.authz_revision is not None and account.role_revision != claims.authz_revision:
            msg = "Authorization changed; please authenticate again"
            raise ForbiddenException(msg)
        for perm in permissions:
            if perm.value not in account.permissions:
                msg = f"Missing required permission: {perm.value}"
                raise ForbiddenException(msg)
        return claims

    return _guard


def require_role(*roles: UserRole) -> Callable[..., Awaitable[TokenClaims]]:
    """Dependency factory that checks the live account role after session validation.

    Usage: Depends(require_role(UserRole.ADMIN, UserRole.MODERATOR))
    """

    async def _guard(
        claims: Annotated[TokenClaims, Depends(get_token_claims)],
        account: Annotated[AuthAccount, Depends(get_current_active_user)],
    ) -> TokenClaims:
        if claims.authz_revision is not None and account.role_revision != claims.authz_revision:
            msg = "Authorization changed; please authenticate again"
            raise ForbiddenException(msg)
        if account.role not in roles:
            msg = "Insufficient role for this operation"
            raise ForbiddenException(msg)
        return claims

    return _guard


# ── Annotated aliases — import and use directly in router signatures ───────────

CurrentUser = Annotated[AuthAccount, Depends(get_current_active_user)]
CurrentVerifiedUser = Annotated[AuthAccount, Depends(get_current_verified_user)]
CurrentClaims = Annotated[TokenClaims, Depends(get_token_claims)]
WebSocketClaims = Annotated[TokenClaims, Depends(get_websocket_token_claims)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
