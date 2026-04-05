from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import RedirectResponse, Response

from app.config import get_settings
from app.connections import get_mongodb, get_redis
from app.features.auth.dependencies import (
    AuthServiceDep,
    CurrentClaims,
    CurrentVerifiedUser,
)
from app.features.auth.dto import (
    ForgotPasswordRequest,
    LoginRequest,
    LogoutRequest,
    OAuthAuthorizeResponse,
    RefreshRequest,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    SessionResponse,
    TokenResponse,
    UserResponse,
    VerifyEmailRequest,
)
from app.features.auth.repository import RefreshTokenRepository, UserRepository
from app.features.auth.security import OAUTH_STATE_COOKIE
from app.features.auth.service import AuthService
from app.shared.response_type import APIResponse
from app.utils import UnauthorizedException, ValidationException, http_response
from app.utils.rate_limit.dependencies import get_rate_limiter

router = APIRouter(prefix="/auth", tags=["auth"])

_ACCESS_TOKEN_COOKIE = "access_token"
limit_register = get_rate_limiter(burst=3, rate=3, period=60)
limit_login = get_rate_limiter(burst=5, rate=5, period=60)
limit_forgot_password = get_rate_limiter(burst=3, rate=3, period=300)
limit_resend_verification = get_rate_limiter(burst=2, rate=2, period=300)


# Read this before you read the code: The auth router is the most complex part of the codebase, so it's worth a high-level overview before diving in. The main goals here are:
# 1. Handle all authentication-related routes: registration, login, logout, token refresh, email verification, password reset, and OAuth2.
# 2. Implement robust RBAC guards using JWT claims to avoid unnecessary database hits on protected routes.
# 3. Apply rate limiting to sensitive endpoints to mitigate abuse (e.g. login, registration, password reset).
# The code is organized into three main sections:
# 1. Dependencies: Wiring up repositories and services, plus token extraction and claims decoding.
# 2. Public routes: Registration, login, email verification, password reset, and OAuth2 endpoints that don't require authentication.
# 3. Protected routes: Endpoints that require a valid JWT and appropriate permissions, with the guards implemented as dependencies that decode claims without hitting the

# Protected Routes
# Everything chains through FastAPI's dependency injection. The base chain is:
# _extract_raw_token          # Bearer header OR cookie fallback
#     → get_token_claims      # decode + validate JWT (zero DB hits)
#         → get_current_user  # MongoDB lookup for live user state
#             → get_current_active_user
#                 → get_current_verified_user
# Each step is a dependency that calls the one above it. You pick the right entry point for how much validation a route needs:
# python# Only needs a valid JWT — no DB hit
# async def endpoint(claims: CurrentClaims): ...

# # Needs live user object + active check
# async def endpoint(user: CurrentVerifiedUser): ...
# CurrentClaims, CurrentVerifiedUser etc. are just Annotated aliases — they make the chain invisible at the call site.

# RBAC
# Two dependency factories — require_permission and require_role. Both validate from JWT claims only, no DB round trip:
# def require_permission(*permissions: Permission):
#     async def _guard(claims: Annotated[TokenClaims, Depends(get_token_claims)]) -> TokenClaims:
#         user_perms = set(claims.permissions)
#         for perm in permissions:
#             if perm.value not in user_perms:
#                 raise ForbiddenException(...)
#         return claims
#     return _guard
# require_role does the same but checks claims.role. You attach them at the route level in two ways:
# python# Via dependencies= (no return value needed in handler)
# @router.get("/", dependencies=[Depends(require_permission(Permission.USERS_READ))])

# # Via parameter (when handler needs the claims object)
# async def update_role(
#     claims: Annotated[TokenClaims, Depends(require_permission(Permission.USERS_WRITE))]
# ):
#     await service.update_role(..., requesting_admin_id=claims.sub)
# The permissions themselves are embedded into the JWT at login time via ROLE_PERMISSIONS[user.role] — so the token is self-describing. No database lookup needed to answer "can this user do X."


# ⬛ FOR THE CHOSEN ONES
# The dual-attach pattern on admin routes is intentional redundancy, not a mistake. On routes like update_role you'll notice require_permission appears both in dependencies=[] and as a typed parameter. This is because FastAPI treats dependencies=[] as fire-and-forget — the return value is discarded. If you need claims.sub inside the handler, you must declare it as a parameter too. Two calls to the same dependency with the same arguments hit FastAPI's dependency cache within a single request — the guard runs exactly once, not twice. This is FastAPI's use_cache=True default behavior at work.
# Why claims-based RBAC instead of DB-based. The alternative — fetching the user on every request and checking user.role live — costs one MongoDB round trip per request. At 1000 req/s that's 1000 extra DB reads per second for pure authorization logic. Claims-based shifts that cost entirely to login time. The tradeoff is stale permissions: if you revoke a role, the user's existing access tokens remain valid until expiry (max 15 min with your config). For most SaaS products that's acceptable. If it's not — say, you're building a financial platform — the fix is a Redis permission cache keyed by user_id with an explicit invalidation call on role change, not a raw DB hit on every request.


# ── Registration and login ─────────────────────────────────────────────────────


@router.post(
    "/register",
    response_model=APIResponse[UserResponse],
    status_code=201,
    dependencies=[Depends(limit_register)],
)
async def register(
    body: RegisterRequest,
    service: AuthServiceDep,
) -> APIResponse[UserResponse]:
    result = await service.register(body)
    return http_response(
        "Registration successful. Check your email to verify your account.",
        data=result,
        status_code=201,
    )


@router.post("/login", response_model=APIResponse[TokenResponse], dependencies=[Depends(limit_login)])
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    service: AuthServiceDep,
) -> APIResponse[TokenResponse]:
    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    tokens = await service.login(body, ip=ip, user_agent=user_agent)
    response.set_cookie(
        _ACCESS_TOKEN_COOKIE,
        tokens.access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=tokens.expires_in,
    )
    return http_response("Login successful", data=tokens)


@router.post("/logout", response_model=APIResponse[None])
async def logout(
    body: LogoutRequest,
    response: Response,
    service: AuthServiceDep,
) -> APIResponse[None]:
    await service.logout(body.refresh_token)
    response.delete_cookie(_ACCESS_TOKEN_COOKIE)
    return http_response("Logged out successfully")


@router.post("/refresh", response_model=APIResponse[TokenResponse])
async def refresh_token(
    body: RefreshRequest,
    response: Response,
    service: AuthServiceDep,
) -> APIResponse[TokenResponse]:
    tokens = await service.refresh(body.refresh_token)
    response.set_cookie(
        _ACCESS_TOKEN_COOKIE,
        tokens.access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=tokens.expires_in,
    )
    return http_response("Token refreshed", data=tokens)


# ── Email verification ─────────────────────────────────────────────────────────


@router.post("/verify-email", response_model=APIResponse[None])
async def verify_email(
    body: VerifyEmailRequest,
    service: AuthServiceDep,
) -> APIResponse[None]:
    await service.verify_email(body.token)
    return http_response("Email verified successfully")


@router.post(
    "/resend-verification",
    response_model=APIResponse[None],
    dependencies=[Depends(limit_resend_verification)],
)
async def resend_verification(
    body: ResendVerificationRequest,
    service: AuthServiceDep,
) -> APIResponse[None]:
    await service.resend_verification(body.email)
    return http_response("If that email is registered, a verification link has been sent")


# ── Password reset ─────────────────────────────────────────────────────────────


@router.post(
    "/forgot-password",
    response_model=APIResponse[None],
    dependencies=[Depends(limit_forgot_password)],
)
async def forgot_password(
    body: ForgotPasswordRequest,
    service: AuthServiceDep,
) -> APIResponse[None]:
    await service.forgot_password(body.email)
    return http_response("If that email is registered, a reset link has been sent")


@router.post("/reset-password", response_model=APIResponse[None])
async def reset_password(
    body: ResetPasswordRequest,
    service: AuthServiceDep,
) -> APIResponse[None]:
    await service.reset_password(body.token, body.new_password)
    return http_response("Password reset successfully. All sessions have been revoked.")


# ── OAuth2 ─────────────────────────────────────────────────────────────────────


@router.get("/oauth/{provider}/authorize", response_model=APIResponse[OAuthAuthorizeResponse])
async def oauth_authorize(
    provider: Annotated[str, Path()],
    response: Response,
    service: AuthServiceDep,
) -> APIResponse[OAuthAuthorizeResponse]:
    url, signed_state = await service.oauth_get_authorization_url(provider)
    # samesite="lax" is intentional — "strict" blocks cookies on cross-site redirects,
    # which is exactly the OAuth callback redirect from the provider.
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        signed_state,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=300,
    )
    return http_response(
        "Authorization URL generated",
        data=OAuthAuthorizeResponse(authorization_url=url, provider=provider),
    )


@router.get("/oauth/{provider}/callback")
async def oauth_callback(
    provider: Annotated[str, Path()],
    request: Request,
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
) -> Response:
    settings = get_settings()
    frontend_url = settings.FRONTEND_URL.rstrip("/")

    if error:
        return RedirectResponse(url=f"{frontend_url}/auth/error?reason={error}")

    if not code or not state:
        raise ValidationException("Missing code or state in OAuth callback")

    signed_state: str | None = request.cookies.get(OAUTH_STATE_COOKIE)
    if not signed_state:
        raise UnauthorizedException("Missing OAuth state cookie")

    # Resolve service manually — can't use AuthServiceDep with mixed Response return types

    user_repo = UserRepository(await get_mongodb(request))
    token_repo = RefreshTokenRepository(await get_redis(request))
    service = AuthService(user_repo, token_repo)

    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    tokens = await service.oauth_callback(
        provider=provider,
        code=code,
        state=state,
        signed_state=signed_state,
        ip=ip,
        user_agent=user_agent,
    )

    # Refresh token travels in the URL fragment — never hits the server, stays in browser memory
    redirect = RedirectResponse(
        url=f"{frontend_url}/auth/callback#refresh_token={tokens.refresh_token}",
        status_code=302,
    )
    redirect.delete_cookie(OAUTH_STATE_COOKIE)
    redirect.set_cookie(
        _ACCESS_TOKEN_COOKIE,
        tokens.access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=tokens.expires_in,
    )
    return redirect


# ── Protected endpoints ────────────────────────────────────────────────────────


@router.get("/me", response_model=APIResponse[UserResponse])
async def get_me(user: CurrentVerifiedUser) -> APIResponse[UserResponse]:
    result = UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        is_verified=user.is_verified,
        is_active=user.is_active,
        created_at=user.created_at,
    )
    return http_response("User profile", data=result)


@router.get("/sessions", response_model=APIResponse[list[SessionResponse]])
async def list_sessions(
    claims: CurrentClaims,
    service: AuthServiceDep,
) -> APIResponse[list[SessionResponse]]:
    sessions = await service.list_sessions(
        user_id=claims.sub,
        current_session_id=claims.sid,
    )
    return http_response("Active sessions", data=sessions)


@router.delete("/sessions/{session_id}", response_model=APIResponse[None])
async def revoke_session(
    session_id: Annotated[str, Path()],
    claims: CurrentClaims,
    service: AuthServiceDep,
) -> APIResponse[None]:
    await service.revoke_session(session_id=session_id, user_id=claims.sub)
    return http_response("Session revoked")


@router.delete("/sessions", response_model=APIResponse[None])
async def revoke_all_sessions(
    claims: CurrentClaims,
    service: AuthServiceDep,
    keep_current: Annotated[bool, Query()] = True,
) -> APIResponse[None]:
    await service.revoke_all_sessions(
        user_id=claims.sub,
        except_session_id=claims.sid if keep_current else None,
    )
    return http_response("All other sessions revoked" if keep_current else "All sessions revoked")
