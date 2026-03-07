# app/features/auth/router.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_limiter.depends import RateLimiter
from pyrate_limiter import Duration, Limiter, Rate

from app.features.auth.dependencies import get_auth_service, get_current_user
from app.features.auth.dto import (
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
    TokenResponse,
)
from app.features.auth.service import AuthService
from app.utils import logger

router: APIRouter = APIRouter(prefix="/auth", tags=["Auth"])
security: HTTPBearer = HTTPBearer()


@router.post(
    path="/register",
    dependencies=[Depends(RateLimiter(limiter=Limiter(argument=Rate(5, Duration.SECOND * 60))))],
)
async def register(
    data: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
) -> dict[str, str]:
    try:
        user = await service.register(data=data)
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(object=e))


@router.post(
    path="/login",
    response_model=TokenResponse,
    dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(5, Duration.SECOND * 60))))],
)
async def login(
    data: LoginRequest,
    service: AuthService = Depends(get_auth_service),
):
    return await service.login(email=data.email, password=data.password)


@router.post(path="/refresh", response_model=TokenResponse)
async def refresh(
    creds: HTTPAuthorizationCredentials = Depends(security),
    service: AuthService = Depends(get_auth_service),
):
    return await service.refresh(refresh_token=creds.credentials)


@router.post(path="/logout", response_model=LogoutResponse)
async def logout(
    creds: HTTPAuthorizationCredentials = Depends(security),
    service: AuthService = Depends(get_auth_service),
):
    await service.logout(refresh_token=creds.credentials)
    return {"detail": "Logged out successfully"}


@router.get(path="/me")
async def me(user=Depends(get_current_user)):
    logger.info(f"User info accessed for user_id: {user.id}")
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
    }
