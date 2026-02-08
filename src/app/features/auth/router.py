# app/features/auth/router.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.features.auth.dependency import get_auth_service, get_current_user
from app.features.auth.dto import (
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
    TokenResponse,
)
from app.features.auth.service import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])
security = HTTPBearer()


@router.post("/register")
async def register(
    data: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
):
    try:
        user = await service.register(data)
        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    service: AuthService = Depends(get_auth_service),
):
    return await service.login(data.email, data.password)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    creds: HTTPAuthorizationCredentials = Depends(security),
    service: AuthService = Depends(get_auth_service),
):
    return await service.refresh(creds.credentials)


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    creds: HTTPAuthorizationCredentials = Depends(security),
    service: AuthService = Depends(get_auth_service),
):
    await service.logout(creds.credentials)
    return {"detail": "Logged out successfully"}


@router.get("/me")
async def me(user=Depends(get_current_user)):
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
    }
