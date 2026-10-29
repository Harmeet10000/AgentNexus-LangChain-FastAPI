import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Annotated, NoReturn, assert_never

from fastapi import Depends, Path, Request, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from returns.result import Failure, Success
from starlette.requests import HTTPConnection

from app.connections.redis import get_redis
from app.features.auth.adapters.persistence.mongodb.account_repository import RefreshTokenRepository
from app.features.auth.adapters.persistence.mongodb.user_store import (
    AuthAccount,
    AuthAccountStore,
    BeanieAuthAccountStore,
)
from app.features.auth.adapters.persistence.postgres.tenancy_repository import (
    TenantMembershipService,
)
from app.features.auth.application.service import AuthService
from app.features.auth.domain.errors import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthResult,
    AuthValidationError,
)
from app.features.auth.domain.tenancy import LegalRole
from app.features.auth.domain.user import Permission, UserRole
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

_http_bearer = HTTPBearer(auto_error=False)
_ACCESS_TOKEN_TYPE = "access"  # noqa: S105 — token type constant, not a password
_TENANT_SELECTOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_TENANT_SUBPROTOCOL_PREFIXES = (
    "tenant.",
    "tenant:",
    "tenant-id.",
    "tenant-id:",
    "tenant_id=",
    "tenant-id=",
)


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    """Live authorization state paired with the authenticated token.

    ``TokenClaims`` remains useful for identity and session correlation, but
    its role/permission fields are historical token context.  Privileged
    dependencies return this wrapper so handlers cannot accidentally treat
    those stale fields as current authority.
    """

    claims: TokenClaims
    account: AuthAccount
    authoritative_tenant_id: str
    authoritative_legal_roles: tuple[LegalRole, ...]
    authoritative_self_review_allowed: bool
    authoritative_role_revision: int
    # These are capability fields, never request payload fields.  The
    # document dependency fills the tenant document allowlist from SQL; auth
    # dependencies provide the operation capability and thread binding.
    allowed_doc_ids: tuple[str, ...] = ()
    allowed_operations: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    thread_id: str = ""

    @property
    def user_id(self) -> str:
        return self.account.id

    @property
    def tenant_id(self) -> str:
        """Tenant resolved from the SQL membership authority."""
        return self.authoritative_tenant_id

    @property
    def legal_roles(self) -> tuple[LegalRole, ...]:
        return self.authoritative_legal_roles

    @property
    def self_review_allowed(self) -> bool:
        return self.authoritative_self_review_allowed

    @property
    def role_revision(self) -> int:
        return self.authoritative_role_revision

    def has_legal_role(self, *roles: LegalRole) -> bool:
        return any(role in self.legal_roles for role in roles)

    @property
    def session_id(self) -> str | None:
        return self.claims.sid

    def narrow_document_ids(self, requested: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Intersect client filters with the server-owned document capability."""
        requested_ids = tuple(dict.fromkeys(str(value) for value in requested if str(value)))
        if not requested_ids:
            return self.allowed_doc_ids
        allowed = set(self.allowed_doc_ids)
        return tuple(document_id for document_id in requested_ids if document_id in allowed)

    def for_thread(self, thread_id: str) -> "AuthorizationContext":
        """Bind this immutable capability to a server-created thread identity."""
        if not thread_id:
            message = "thread_id must not be empty"
            raise ValueError(message)
        return replace(self, thread_id=thread_id)


def _allowed_operations(account: AuthAccount, roles: tuple[LegalRole, ...]) -> tuple[str, ...]:
    """Derive document/evidence capabilities from live account + SQL roles."""
    operations = {
        "document.read",
        "document.search",
        "evidence.search",
        "evidence.read",
        "ingestion.status",
    }
    if Permission.CONTENT_WRITE.value in account.permissions:
        operations.update({"document.upload", "document.manage"})
    if Permission.CONTENT_DELETE.value in account.permissions:
        operations.add("document.delete")
    if any(role in roles for role in (LegalRole.REVIEWER, LegalRole.CORPUS_ADMIN)):
        operations.update({"report.approve", "report.reject", "report.override"})
    if LegalRole.CORPUS_ADMIN in roles:
        operations.add("tenant.manage")
    return tuple(sorted(operations))


def _build_authorization_context(
    *,
    claims: TokenClaims,
    account: AuthAccount,
    tenant_id: str,
    roles: tuple[LegalRole, ...],
    self_review_allowed: bool,
    role_revision: int,
    thread_id: str = "",
) -> AuthorizationContext:
    allowed_operations = _allowed_operations(account, roles)
    return AuthorizationContext(
        claims=claims,
        account=account,
        authoritative_tenant_id=tenant_id,
        authoritative_legal_roles=roles,
        authoritative_self_review_allowed=self_review_allowed,
        authoritative_role_revision=role_revision,
        allowed_operations=allowed_operations,
        capabilities=allowed_operations,
        thread_id=thread_id,
    )


# ── Repository and service wiring ─────────────────────────────────────────────


async def get_account_store() -> AuthAccountStore:
    """Production MongoDB/Beanie adapter for AuthAccountStore."""
    return BeanieAuthAccountStore()


async def get_refresh_token_repository(
    redis=Depends(get_redis),
) -> RefreshTokenRepository:
    return RefreshTokenRepository(redis)


async def get_optional_refresh_token_repository(
    connection: HTTPConnection,
) -> RefreshTokenRepository | None:
    """Build the session repository when Redis is available.

    Token claims decode before deciding whether a Redis session lookup is
    needed.  This preserves the intentional stateless ``sid=None``
    impersonation path during a Redis startup outage while stateful tokens
    still fail closed in ``get_token_claims``.
    """
    redis = getattr(connection.app.state, "redis", None)
    return RefreshTokenRepository(redis) if redis is not None else None


async def get_auth_service(
    account_store: Annotated[AuthAccountStore, Depends(get_account_store)],
    token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
    connection: HTTPConnection,
) -> AuthService:
    session_factory = getattr(connection.app.state, "db_session_local", None)
    return AuthService(account_store, token_repo, session_factory=session_factory)


async def get_tenant_membership_service(
    connection: HTTPConnection,
    account_store: Annotated[AuthAccountStore, Depends(get_account_store)],
) -> TenantMembershipService:
    """Build the authoritative SQL membership service for tenant APIs."""
    session_factory = getattr(connection.app.state, "db_session_local", None)
    if session_factory is None:
        message = "PostgreSQL is unavailable"
        raise InfrastructureException(message)
    return TenantMembershipService(session_factory, account_store=account_store)


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
    token_repo: Annotated[
        RefreshTokenRepository | None, Depends(get_optional_refresh_token_repository)
    ],
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
        if token_repo is None:
            raise InfrastructureException(
                detail="Session store unavailable",
                retryable=True,
                data={"operation": "session_verification"},
            )
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


def _websocket_tenant_selector(websocket: WebSocket) -> str | None:
    """Read the browser-safe tenant selector without widening authority.

    Browsers cannot set arbitrary handshake headers, so ``tenant_id`` in the
    WebSocket URL complements ``x-tenant-id``.  If both are present they must
    agree; the selected value is still passed through SQL membership
    resolution below and is never treated as authority by itself.
    """
    candidates: list[str] = []

    def add_candidate(value: object) -> None:
        if not isinstance(value, str):
            message = "Invalid tenant selector"
            raise ForbiddenException(message)
        if not _TENANT_SELECTOR_PATTERN.fullmatch(value):
            message = "Invalid tenant selector"
            raise ForbiddenException(message)
        candidates.append(value)

    header_values = websocket.headers.getlist("x-tenant-id")
    query_values = websocket.query_params.getlist("tenant_id")
    for value in (*header_values, *query_values):
        add_candidate(value)

    # A browser may offer subprotocols during the handshake but cannot set a
    # custom header.  Only an explicitly namespaced protocol is interpreted as
    # a selector; ordinary application protocols (``json``, ``graphql-ws``)
    # remain transport negotiation and never become authority input.
    scope = getattr(websocket, "scope", {})
    offered = scope.get("subprotocols", ()) if isinstance(scope, dict) else ()
    if not isinstance(offered, (list, tuple)):
        message = "Invalid tenant selector"
        raise ForbiddenException(message)
    for protocol in offered:
        if not isinstance(protocol, str):
            message = "Invalid tenant selector"
            raise ForbiddenException(message)
        for prefix in _TENANT_SUBPROTOCOL_PREFIXES:
            if protocol.startswith(prefix):
                add_candidate(protocol[len(prefix) :])
                break

    if not candidates:
        return None
    selected = candidates[0]
    if any(value != selected for value in candidates[1:]):
        message = "Conflicting tenant selectors"
        raise ForbiddenException(message)
    return selected


async def get_current_websocket_authorization_context(
    websocket: WebSocket,
    claims: Annotated[TokenClaims, Depends(get_websocket_token_claims)],
    account_store: Annotated[AuthAccountStore, Depends(get_account_store)],
    membership_service: Annotated[TenantMembershipService, Depends(get_tenant_membership_service)],
) -> AuthorizationContext:
    """Resolve live tenant/legal authority for a WebSocket session.

    A WebSocket can remain open across a role change, so the JWT is only an
    identity/session hint here.  The account store is consulted at admission;
    Agent Saul re-resolves it again before each HITL resume.
    """
    result = await account_store.find_by_id(claims.sub)
    if isinstance(result, Failure):
        error = result.failure()
        if isinstance(error, AuthInfrastructureError):
            raise InfrastructureException(
                detail="Unable to resolve live authorization",
                error_code=error.code,
                retryable=error.retryable,
                data=error.details,
            )
        message = "User not found"
        raise UnauthorizedException(message)
    account = result.unwrap()
    if account is None or not account.is_active:
        message = "Account is disabled or unavailable"
        raise UnauthorizedException(message)
    tenant_id = _websocket_tenant_selector(websocket)
    selected = await membership_service.resolve_tenant(
        user_id=account.id,
        requested_tenant_id=tenant_id,
    )
    if isinstance(selected, Failure):
        _raise_tenant_selection_failure(selected.failure())
    tenant_id = selected.unwrap()
    live = await membership_service.get_live_authorization(
        tenant_id=tenant_id,
        user_id=account.id,
    )
    if isinstance(live, Failure):
        error = live.failure()
        if isinstance(error, AuthInfrastructureError):
            raise InfrastructureException(detail=error.message, error_code=error.code)
        message = "Tenant authorization is unavailable"
        raise ForbiddenException(message)
    legal_roles, self_review_allowed, role_revision = live.unwrap()
    return _build_authorization_context(
        claims=claims,
        account=account,
        tenant_id=tenant_id,
        roles=legal_roles,
        self_review_allowed=self_review_allowed,
        role_revision=role_revision,
    )


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


async def get_current_authorization_context(
    claims: Annotated[TokenClaims, Depends(get_token_claims)],
    account: Annotated[AuthAccount, Depends(get_current_active_user)],
    membership_service: Annotated[TenantMembershipService, Depends(get_tenant_membership_service)],
    connection: HTTPConnection,
) -> AuthorizationContext:
    """Resolve the live tenant and legal roles for the current principal.

    Token claims identify a session, but they are deliberately not the source
    of current tenant authorization.  SQL membership and tenant policy are
    loaded on every request, so this context remains safe after legal-role
    changes or a long-lived WebSocket resume.  Platform session revisions are
    validated separately against the Mongo account/session store.
    """
    # ``x-tenant-id`` is a selector, not authority: the SQL membership query
    # below validates it.  With one membership SQL selects it automatically;
    # with multiple memberships omitting the selector fails closed.
    selected = await membership_service.resolve_tenant(
        user_id=account.id,
        requested_tenant_id=connection.headers.get("x-tenant-id"),
    )
    if isinstance(selected, Failure):
        _raise_tenant_selection_failure(selected.failure())
    tenant_id = selected.unwrap()
    live = await membership_service.get_live_authorization(
        tenant_id=tenant_id,
        user_id=account.id,
    )
    if isinstance(live, Failure):
        error = live.failure()
        if isinstance(error, AuthInfrastructureError):
            raise InfrastructureException(detail=error.message, error_code=error.code)
        message = "Tenant authorization is unavailable"
        raise ForbiddenException(message)
    legal_roles, self_review_allowed, role_revision = live.unwrap()
    return _build_authorization_context(
        claims=claims,
        account=account,
        tenant_id=tenant_id,
        roles=legal_roles,
        self_review_allowed=self_review_allowed,
        role_revision=role_revision,
    )


def _raise_tenant_selection_failure(error: object) -> NoReturn:
    """Translate SQL tenant-selection failures at the HTTP/WS boundary."""
    if isinstance(error, AuthInfrastructureError):
        raise InfrastructureException(
            detail=error.message,
            error_code=error.code,
            retryable=error.retryable,
            data=error.details,
        )
    if isinstance(error, AuthAuthorizationError | AuthConflictError):
        raise ForbiddenException(detail=error.message, error_code=error.code)
    raise ForbiddenException(detail="Tenant authorization is unavailable")


async def get_current_verified_user(
    account: Annotated[AuthAccount, Depends(get_current_active_user)],
) -> AuthAccount:
    if not account.is_verified:
        msg = "Email not verified"
        raise UnauthorizedException(msg)
    return account


# ── RBAC guards (live account state) ─────────────────────────────────────────


def require_permission(*permissions: Permission) -> Callable[..., Awaitable[AuthorizationContext]]:
    """Dependency factory that checks live permissions after session validation.

    The returned context deliberately exposes live account state separately
    from token claims. Handlers needing identity should use ``context.user_id``
    rather than reaching into claim role/permission fields.

    Usage: Depends(require_permission(Permission.USERS_READ, Permission.USERS_WRITE))
    """

    async def _guard(
        claims: Annotated[TokenClaims, Depends(get_token_claims)],
        account: Annotated[AuthAccount, Depends(get_current_active_user)],
        authorization: Annotated[
            AuthorizationContext | None, Depends(get_current_authorization_context)
        ] = None,
    ) -> AuthorizationContext:
        if authorization is None:
            # Direct invocation compatibility for non-FastAPI callers.  The
            # production dependency graph always supplies live SQL context.
            if claims.authz_revision is not None and account.role_revision != claims.authz_revision:
                msg = "Authorization changed; please authenticate again"
                raise ForbiddenException(msg)
            for perm in permissions:
                if perm.value not in account.permissions:
                    msg = f"Missing required permission: {perm.value}"
                    raise ForbiddenException(msg)
            return AuthorizationContext(
                claims=claims,
                account=account,
                authoritative_tenant_id=getattr(account, "tenant_id", ""),
                authoritative_legal_roles=getattr(account, "legal_roles", (LegalRole.ANALYST,)),
                authoritative_self_review_allowed=False,
                authoritative_role_revision=account.role_revision,
            )
        claims = authorization.claims
        account = authorization.account
        if claims.authz_revision is not None and account.role_revision != claims.authz_revision:
            msg = "Authorization changed; please authenticate again"
            raise ForbiddenException(msg)
        for perm in permissions:
            if perm.value not in account.permissions:
                msg = f"Missing required permission: {perm.value}"
                raise ForbiddenException(msg)
        return authorization

    return _guard


def require_role(*roles: UserRole) -> Callable[..., Awaitable[AuthorizationContext]]:
    """Dependency factory that checks the live account role after session validation.

    Usage: Depends(require_role(UserRole.ADMIN, UserRole.MODERATOR))
    """

    async def _guard(
        claims: Annotated[TokenClaims, Depends(get_token_claims)],
        account: Annotated[AuthAccount, Depends(get_current_active_user)],
        authorization: Annotated[
            AuthorizationContext | None, Depends(get_current_authorization_context)
        ] = None,
    ) -> AuthorizationContext:
        if authorization is None:
            if claims.authz_revision is not None and account.role_revision != claims.authz_revision:
                msg = "Authorization changed; please authenticate again"
                raise ForbiddenException(msg)
            if account.role not in roles:
                msg = "Insufficient role for this operation"
                raise ForbiddenException(msg)
            return AuthorizationContext(
                claims=claims,
                account=account,
                authoritative_tenant_id=getattr(account, "tenant_id", ""),
                authoritative_legal_roles=getattr(account, "legal_roles", (LegalRole.ANALYST,)),
                authoritative_self_review_allowed=False,
                authoritative_role_revision=account.role_revision,
            )
        claims = authorization.claims
        account = authorization.account
        if claims.authz_revision is not None and account.role_revision != claims.authz_revision:
            msg = "Authorization changed; please authenticate again"
            raise ForbiddenException(msg)
        if account.role not in roles:
            msg = "Insufficient role for this operation"
            raise ForbiddenException(msg)
        return authorization

    return _guard


def require_legal_role(*roles: LegalRole) -> Callable[..., Awaitable[AuthorizationContext]]:
    """Require a tenant-scoped legal role, independent of platform RBAC."""

    async def _guard(
        claims: Annotated[TokenClaims, Depends(get_token_claims)],
        account: Annotated[AuthAccount, Depends(get_current_active_user)],
        authorization: Annotated[
            AuthorizationContext | None, Depends(get_current_authorization_context)
        ] = None,
    ) -> AuthorizationContext:
        if authorization is None:
            if claims.authz_revision is not None and account.role_revision != claims.authz_revision:
                msg = "Authorization changed; please authenticate again"
                raise ForbiddenException(msg)
            if not account.legal_roles or not any(
                legal_role in account.legal_roles for legal_role in roles
            ):
                msg = "Insufficient legal role for this operation"
                raise ForbiddenException(msg)
            return AuthorizationContext(
                claims=claims,
                account=account,
                authoritative_tenant_id=getattr(account, "tenant_id", ""),
                authoritative_legal_roles=getattr(account, "legal_roles", (LegalRole.ANALYST,)),
                authoritative_self_review_allowed=False,
                authoritative_role_revision=account.role_revision,
            )
        claims = authorization.claims
        account = authorization.account
        if claims.authz_revision is not None and account.role_revision != claims.authz_revision:
            msg = "Authorization changed; please authenticate again"
            raise ForbiddenException(msg)
        if not authorization.legal_roles or not any(
            legal_role in authorization.legal_roles for legal_role in roles
        ):
            msg = "Insufficient legal role for this operation"
            raise ForbiddenException(msg)
        return authorization

    return _guard


def require_platform_admin() -> Callable[..., Awaitable[AuthorizationContext]]:
    """Explicit platform-administration guard (never a legal-role alias)."""
    return require_role(UserRole.ADMIN)


def require_tenant_corpus_admin() -> Callable[..., Awaitable[AuthorizationContext]]:
    """Require corpus-admin authority for the route's exact tenant."""

    async def _guard(
        tenant_id: Annotated[str, Path(min_length=1, max_length=255)],
        authorization: Annotated[AuthorizationContext, Depends(get_current_authorization_context)],
    ) -> AuthorizationContext:
        if authorization.tenant_id != tenant_id:
            message = "Tenant membership does not grant access to this tenant"
            raise ForbiddenException(message)
        if not authorization.has_legal_role(LegalRole.CORPUS_ADMIN):
            message = "Corpus-admin authority is required"
            raise ForbiddenException(message)
        return authorization

    return _guard


def require_tenant_membership_admin() -> Callable[..., Awaitable[AuthorizationContext]]:
    """Allow tenant corpus admins, with an explicit platform-admin override.

    The override is intentionally based on ``UserRole.ADMIN`` (platform RBAC),
    never on the tenant-scoped ``corpus_admin`` legal role.
    """

    async def _guard(
        tenant_id: Annotated[str, Path(min_length=1, max_length=255)],
        authorization: Annotated[AuthorizationContext, Depends(get_current_authorization_context)],
    ) -> AuthorizationContext:
        if authorization.account.role == UserRole.ADMIN:
            return authorization
        if authorization.tenant_id != tenant_id:
            message = "Tenant membership does not grant access to this tenant"
            raise ForbiddenException(message)
        if not authorization.has_legal_role(LegalRole.CORPUS_ADMIN):
            message = "Corpus-admin authority is required"
            raise ForbiddenException(message)
        return authorization

    return _guard


# ── Annotated aliases — import and use directly in router signatures ───────────

CurrentUser = Annotated[AuthAccount, Depends(get_current_active_user)]
CurrentVerifiedUser = Annotated[AuthAccount, Depends(get_current_verified_user)]
CurrentClaims = Annotated[TokenClaims, Depends(get_token_claims)]
CurrentAuthorizationContext = Annotated[
    AuthorizationContext, Depends(get_current_authorization_context)
]
WebSocketClaims = Annotated[TokenClaims, Depends(get_websocket_token_claims)]
WebSocketAuthorizationContext = Annotated[
    AuthorizationContext, Depends(get_current_websocket_authorization_context)
]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
TenantMembershipServiceDep = Annotated[
    TenantMembershipService, Depends(get_tenant_membership_service)
]
