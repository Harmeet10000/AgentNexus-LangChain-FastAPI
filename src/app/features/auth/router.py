from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import RedirectResponse, Response
from returns.result import Failure

from app.config import get_settings
from app.connections.redis import get_redis
from app.kernel.result import render_result
from app.resilience.rate_limit import get_rate_limiter
from app.security.tokens import OAUTH_STATE_COOKIE
from app.utils import APIResponse, http_response

from .dependencies import (
    AuthServiceDep,
    CurrentClaims,
    CurrentVerifiedUser,
    get_account_store,
)
from .dto import (
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
from .exceptions import AuthAuthenticationError, AuthValidationError
from .repository import RefreshTokenRepository
from .service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])

_ACCESS_TOKEN_COOKIE = "access_token"
limit_register = get_rate_limiter(burst=3, rate=3, period=60)
limit_login = get_rate_limiter(burst=5, rate=5, period=60)
limit_forgot_password = get_rate_limiter(burst=3, rate=3, period=300)
limit_resend_verification = get_rate_limiter(burst=2, rate=2, period=300)


# Read this before you read the code: The auth router is the most complex part of the codebase, so it's worth a high-level overview before diving in. The main goals here are:
# 1. Handle all authentication-related routes: registration, login, logout, token refresh, email verification, password reset, and OAuth2.
# 2. Implement robust RBAC guards against live account state on protected routes.
# 3. Apply rate limiting to sensitive endpoints to mitigate abuse (e.g. login, registration, password reset).
# The code is organized into three main sections:
# 1. Dependencies: Wiring up repositories and services, plus token extraction and claims decoding.
# 2. Public routes: Registration, login, email verification, password reset, and OAuth2 endpoints that don't require authentication.
# 3. Protected routes: Endpoints that require a valid JWT and appropriate permissions, with the guards implemented as dependencies that hydrate live account state.

# Protected Routes
# Everything chains through FastAPI's dependency injection. The base chain is:
# _extract_raw_token          # Bearer header OR cookie fallback
#     → get_token_claims      # decode + validate JWT + Redis session
#         → get_current_user  # MongoDB lookup for live user state
#             → get_current_active_user
#                 → get_current_verified_user
# Each step is a dependency that calls the one above it. You pick the right entry point for how much validation a route needs:
# python# Only needs a valid JWT and live Redis session
# async def endpoint(claims: CurrentClaims): ...

# # Needs live user object + active check
# async def endpoint(user: CurrentVerifiedUser): ...
# CurrentClaims, CurrentVerifiedUser etc. are just Annotated aliases — they make the chain invisible at the call site.

# RBAC
# Two dependency factories — require_permission and require_role. Both validate live account state after JWT/session checks and return AuthorizationContext:
# def require_permission(*permissions: Permission):
#     async def _guard(...) -> AuthorizationContext:
#         user_perms = set(account.permissions)
#         for perm in permissions:
#             if perm.value not in user_perms:
#                 raise ForbiddenException(...)
#         return AuthorizationContext(claims=claims, account=account)
#     return _guard
# require_role does the same but checks account.role. Use a typed context parameter when the handler needs the authenticated subject:
# python# Via dependencies= (no return value needed in handler)
# @router.get("/", dependencies=[Depends(require_permission(Permission.USERS_READ))])

# # Via parameter (when handler needs the live-authorized subject)
# async def update_role(
#     authorization: Annotated[AuthorizationContext, Depends(require_permission(Permission.USERS_WRITE))]
# ):
#     await service.update_role(..., requesting_admin_id=authorization.user_id)
# Permissions remain embedded in the JWT for context, but live account state is authoritative for privileged routes. This ensures role changes and deactivation take effect after session invalidation.


# Live RBAC costs a MongoDB account lookup, but closes the stale-claims window for privileged routes. FastAPI dependency caching avoids repeating that chain within one request. Keep route-level dependencies for handlers that do not need the returned context; use the typed context alone when the handler needs the requester identity.


# ── Registration and login ─────────────────────────────────────────────────────


@router.post(
    "/register",
    status_code=201,
    dependencies=[Depends(limit_register)],
)
async def register(
    body: RegisterRequest,
    service: AuthServiceDep,
    response: Response,
) -> APIResponse[UserResponse]:
    result = await service.register(body)
    return render_result(
        result,
        response,
        message="Registration successful. Check your email to verify your account.",
        success_status=201,
    )


@router.post("/login", dependencies=[Depends(limit_login)])
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    service: AuthServiceDep,
) -> APIResponse[TokenResponse]:
    ip: str | None = request.client.host if request.client else None
    user_agent: str | None = request.headers.get("user-agent")

    result = await service.login(body, ip=ip, user_agent=user_agent)
    if isinstance(result, Failure):
        return render_result(result, response, message="Login successful")
    tokens = result.unwrap()
    response.set_cookie(
        _ACCESS_TOKEN_COOKIE,
        tokens.access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=tokens.expires_in,
    )
    return render_result(result, response, message="Login successful")


@router.post("/logout", response_model=APIResponse[None])
async def logout(
    body: LogoutRequest,
    response: Response,
    service: AuthServiceDep,
) -> APIResponse[None]:
    result = await service.logout(body.refresh_token)
    if not isinstance(result, Failure):
        response.delete_cookie(_ACCESS_TOKEN_COOKIE)
    return render_result(result, response, message="Logged out successfully")


@router.post("/refresh")
async def refresh_token(
    body: RefreshRequest,
    response: Response,
    service: AuthServiceDep,
) -> APIResponse[TokenResponse]:
    result = await service.refresh(body.refresh_token)
    if isinstance(result, Failure):
        return render_result(result, response, message="Token refreshed")
    tokens = result.unwrap()
    response.set_cookie(
        _ACCESS_TOKEN_COOKIE,
        tokens.access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=tokens.expires_in,
    )
    return render_result(result, response, message="Token refreshed")


# ── Email verification ─────────────────────────────────────────────────────────


@router.post("/verify-email", response_model=APIResponse[None])
async def verify_email(
    body: VerifyEmailRequest,
    service: AuthServiceDep,
    response: Response,
) -> APIResponse[None]:
    result = await service.verify_email(body.token)
    return render_result(result, response, message="Email verified successfully")


@router.post(
    "/resend-verification",
    response_model=APIResponse[None],
    dependencies=[Depends(limit_resend_verification)],
)
async def resend_verification(
    body: ResendVerificationRequest,
    service: AuthServiceDep,
    response: Response,
) -> APIResponse[None]:
    result = await service.resend_verification(body.email)
    return render_result(
        result, response, message="If that email is registered, a verification link has been sent"
    )


# ── Password reset ─────────────────────────────────────────────────────────────


@router.post(
    "/forgot-password",
    response_model=APIResponse[None],
    dependencies=[Depends(limit_forgot_password)],
)
async def forgot_password(
    body: ForgotPasswordRequest,
    service: AuthServiceDep,
    response: Response,
) -> APIResponse[None]:
    result = await service.forgot_password(body.email)
    return render_result(
        result, response, message="If that email is registered, a reset link has been sent"
    )


@router.post("/reset-password", response_model=APIResponse[None])
async def reset_password(
    body: ResetPasswordRequest,
    service: AuthServiceDep,
    response: Response,
) -> APIResponse[None]:
    result = await service.reset_password(body.token, body.new_password)
    return render_result(
        result, response, message="Password reset successfully. All sessions have been revoked."
    )


# ── OAuth2 ─────────────────────────────────────────────────────────────────────


@router.get("/oauth/{provider}/authorize")
async def oauth_authorize(
    provider: Annotated[str, Path()],
    response: Response,
    service: AuthServiceDep,
) -> APIResponse[OAuthAuthorizeResponse] | APIResponse[tuple[str, str]]:
    result = await service.oauth_get_authorization_url(provider)
    if isinstance(result, Failure):
        return render_result(result, response, message="Authorization URL generation failed")
    url, signed_state = result.unwrap()
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


@router.get("/oauth/{provider}/callback", response_model=None)
async def oauth_callback(
    provider: Annotated[str, Path()],
    request: Request,
    response: Response,
    *,
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
) -> Response | APIResponse[TokenResponse | None]:
    settings = get_settings()
    frontend_url = settings.FRONTEND_URL.rstrip("/")

    if error:
        return RedirectResponse(url=f"{frontend_url}/auth/error?reason={error}")

    if not code or not state:
        return render_result(
            Failure(
                AuthValidationError(
                    message="Missing code or state in OAuth callback", source="auth_router"
                )
            ),
            response,
            message="OAuth callback failed",
        )

    signed_state: str | None = request.cookies.get(OAUTH_STATE_COOKIE)
    if not signed_state:
        return render_result(
            Failure(
                AuthAuthenticationError(message="Missing OAuth state cookie", source="auth_router")
            ),
            response,
            message="OAuth callback failed",
        )

    # Resolve service manually — can't use AuthServiceDep with mixed Response return types

    account_store = await get_account_store()
    token_repo = RefreshTokenRepository(await get_redis(request))
    service = AuthService(account_store, token_repo)

    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    result = await service.oauth_callback(
        provider=provider,
        code=code,
        state=state,
        signed_state=signed_state,
        ip=ip,
        user_agent=user_agent,
    )
    if isinstance(result, Failure):
        return render_result(result, response, message="OAuth callback failed")
    tokens = result.unwrap()

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


@router.get("/me")
async def get_me(user: CurrentVerifiedUser) -> APIResponse[UserResponse]:
    result = UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
        role=user.role.value,
        is_verified=user.is_verified,
        is_active=user.is_active,
        created_at=user.created_at,
    )
    return http_response("User profile", data=result)


@router.get("/sessions")
async def list_sessions(
    claims: CurrentClaims,
    service: AuthServiceDep,
    response: Response,
) -> APIResponse[list[SessionResponse]]:
    result = await service.list_sessions(
        user_id=claims.sub,
        current_session_id=claims.sid,
    )
    return render_result(result, response, message="Active sessions")


@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: Annotated[str, Path()],
    claims: CurrentClaims,
    service: AuthServiceDep,
    request: Request,
    response: Response,
) -> APIResponse[list[str]]:
    # Task 3.4: revoke and close any live WebSocket connections for the session
    result = await service.revoke_session_and_close_connections(
        session_id=session_id,
        user_id=claims.sub,
        ws_security_service=request.app.state.websocket_security,
    )
    return render_result(result, response, message="Session revoked")


@router.delete("/sessions", response_model=APIResponse[None])
async def revoke_all_sessions(
    claims: CurrentClaims,
    service: AuthServiceDep,
    response: Response,
    keep_current: Annotated[bool, Query()] = True,
) -> APIResponse[None]:
    result = await service.revoke_all_sessions(
        user_id=claims.sub,
        except_session_id=claims.sid if keep_current else None,
    )
    return render_result(
        result,
        response,
        message="All other sessions revoked" if keep_current else "All sessions revoked",
    )
