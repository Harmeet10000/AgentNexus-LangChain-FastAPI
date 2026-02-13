"""Authentication feature module."""

from app.features.auth.dependency import (
    get_auth_service,
    get_current_user,
    get_refresh_token_repository,
    get_user_repository,
)
from app.features.auth.dto import (
    LoginRequest,
    LogoutResponse,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.features.auth.model import User
from app.features.auth.router import router
from app.features.auth.service import AuthService

__all__ = [
    "get_auth_service",
    "get_current_user",
    "get_refresh_token_repository",
    "get_user_repository",
    "LoginRequest",
    "LogoutResponse",
    "RegisterRequest",
    "TokenResponse",
    "UserResponse",
    "User",
    "router",
    "AuthService",
]
