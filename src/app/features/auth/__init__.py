"""Authentication feature exports.

Keep this module lightweight to avoid circular imports.
"""
from .dependencies import (
    CurrentVerifiedUser,
    get_refresh_token_repository,
    get_user_repository,
    require_permission,
    require_role,
)
from .dto import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from .model import Permission, User, UserRole
from .repository import RefreshTokenRepository, UserRepository
from .router import router
from .security import TokenClaims, create_impersonation_token, hash_password, verify_password
from .token_audit_log import TokenAuditLog
from .websocket_security import build_websocket_security_service

__all__ = [
    "CurrentVerifiedUser",
    "LoginRequest",
    "LogoutResponse",
    "Permission",
    "RefreshTokenRepository",
    "TokenAuditLog",
    "TokenClaims",
    "TokenResponse",
    "User",
    "UserRepository",
    "UserResponse",
    "UserRole",
    "build_websocket_security_service",
    "create_impersonation_token",
    "get_refresh_token_repository",
    "get_user_repository",
    "get_user_repository",
    "hash_password",
    "require_permission",
    "require_role",
    "verify_password",
]

