from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import ORJSONResponse, RedirectResponse, Response

from app.config import get_settings
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
from app.features.auth.security import OAUTH_STATE_COOKIE
from app.shared.response_type import APIResponse
from app.utils import UnauthorizedException, ValidationException, http_response
from app.utils.rate_limit import (
    FORGOT_PASSWORD_RATE_LIMIT,
    LOGIN_RATE_LIMIT,
    REGISTER_RATE_LIMIT,
    RESEND_VERIFICATION_RATE_LIMIT,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_ACCESS_TOKEN_COOKIE = "access_token"


# ── Registration and login ─────────────────────────────────────────────────────


@router.post(
    "/register",
    response_model=APIResponse[UserResponse],
    status_code=201,
    dependencies=[Depends(REGISTER_RATE_LIMIT)],
)
async def register(
    body: RegisterRequest,
    service: AuthServiceDep,
) -> ORJSONResponse:
    result = await service.register(body)
    return http_response(
        "Registration successful. Check your email to verify your account.",
        data=result,
        status_code=201,
    )


@router.post("/login", response_model=APIResponse[TokenResponse], dependencies=[Depends(LOGIN_RATE_LIMIT)])
async def login(
    body: LoginRequest,
    request: Request,
    service: AuthServiceDep,
) -> ORJSONResponse:
    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    tokens = await service.login(body, ip=ip, user_agent=user_agent)
    resp = http_response("Login successful", data=tokens)
    resp.set_cookie(
        _ACCESS_TOKEN_COOKIE,
        tokens.access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=tokens.expires_in,
    )
    return resp


@router.post("/logout", response_model=APIResponse[None])
async def logout(
    body: LogoutRequest,
    service: AuthServiceDep,
) -> ORJSONResponse:
    await service.logout(body.refresh_token)
    resp = http_response("Logged out successfully")
    resp.delete_cookie(_ACCESS_TOKEN_COOKIE)
    return resp


@router.post("/refresh", response_model=APIResponse[TokenResponse])
async def refresh_token(
    body: RefreshRequest,
    service: AuthServiceDep,
) -> ORJSONResponse:
    tokens = await service.refresh(body.refresh_token)
    resp = http_response("Token refreshed", data=tokens)
    resp.set_cookie(
        _ACCESS_TOKEN_COOKIE,
        tokens.access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=tokens.expires_in,
    )
    return resp


# ── Email verification ─────────────────────────────────────────────────────────


@router.post("/verify-email", response_model=APIResponse[None])
async def verify_email(
    body: VerifyEmailRequest,
    service: AuthServiceDep,
) -> ORJSONResponse:
    await service.verify_email(body.token)
    return http_response("Email verified successfully")


@router.post("/resend-verification", response_model=APIResponse[None])
async def resend_verification(
    body: ResendVerificationRequest,
    service: AuthServiceDep,
) -> ORJSONResponse:
    await service.resend_verification(body.email)
    return http_response("If that email is registered, a verification link has been sent")


# ── Password reset ─────────────────────────────────────────────────────────────


@router.post("/forgot-password", response_model=APIResponse[None], dependencies=[Depends(FORGOT_PASSWORD_RATE_LIMIT)])
async def forgot_password(
    body: ForgotPasswordRequest,
    service: AuthServiceDep,
) -> ORJSONResponse:
    await service.forgot_password(body.email)
    return http_response("If that email is registered, a reset link has been sent")


@router.post("/reset-password", response_model=APIResponse[None])
async def reset_password(
    body: ResetPasswordRequest,
    service: AuthServiceDep,
) -> ORJSONResponse:
    await service.reset_password(body.token, body.new_password)
    return http_response("Password reset successfully. All sessions have been revoked.")


# ── OAuth2 ─────────────────────────────────────────────────────────────────────


@router.get("/oauth/{provider}/authorize", response_model=APIResponse[OAuthAuthorizeResponse])
async def oauth_authorize(
    provider: Annotated[str, Path()],
    service: AuthServiceDep,
) -> ORJSONResponse:
    url, signed_state = await service.oauth_get_authorization_url(provider)
    resp = http_response(
        "Authorization URL generated",
        data=OAuthAuthorizeResponse(authorization_url=url, provider=provider),
    )
    # samesite="lax" is intentional — "strict" blocks cookies on cross-site redirects,
    # which is exactly the OAuth callback redirect from the provider.
    resp.set_cookie(
        OAUTH_STATE_COOKIE,
        signed_state,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=300,
    )
    return resp


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
    from app.connections import get_mongodb, get_redis
    from app.features.auth.repository import RefreshTokenRepository, UserRepository
    from app.features.auth.service import AuthService

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
async def get_me(user: CurrentVerifiedUser) -> ORJSONResponse:
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
) -> ORJSONResponse:
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
) -> ORJSONResponse:
    await service.revoke_session(session_id=session_id, user_id=claims.sub)
    return http_response("Session revoked")


@router.delete("/sessions", response_model=APIResponse[None])
async def revoke_all_sessions(
    claims: CurrentClaims,
    service: AuthServiceDep,
    keep_current: Annotated[bool, Query()] = True,
) -> ORJSONResponse:
    await service.revoke_all_sessions(
        user_id=claims.sub,
        except_session_id=claims.sid if keep_current else None,
    )
    return http_response("All other sessions revoked" if keep_current else "All sessions revoked")
