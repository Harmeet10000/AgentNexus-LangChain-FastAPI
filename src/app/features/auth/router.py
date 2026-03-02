# app/features/auth/router.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.features.auth.dependency import get_auth_service, get_current_user
from app.features.auth.dto import (
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
    TokenResponse,
)
from app.features.auth.service import AuthService

limiter = Limiter(key_func=get_remote_address)

router: APIRouter = APIRouter(prefix="/auth", tags=["Auth"])
security: HTTPBearer = HTTPBearer()


@router.post(path="/register")
@limiter.limit(limit_value="5/minute")
async def register(
    request: Request,
    data: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
):
    try:
        user = await service.register(data=data)
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(object=e))


@router.post(path="/login", response_model=TokenResponse)
@limiter.limit(limit_value="5/minute")
async def login(
    request: Request,
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
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
    }
