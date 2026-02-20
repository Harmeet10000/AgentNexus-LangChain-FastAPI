# exceptions.py
from typing import Any

from fastapi import HTTPException, status


class APIException(HTTPException):
    """
    Base exception for all **handled/known** API errors.

    All user-facing error responses should inherit from this class
    (or use its subclasses) so they get consistent formatting.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str | None = None,
        data: Any = None,
        headers: dict[str, str] | None = None,
    ):
        self.error_code = (
            error_code or self.__class__.__name__.replace("Exception", "").upper()
        )
        self.data = data

        # We put everything inside 'detail' so the frontend gets a rich object
        rich_detail = {
            "message": detail,
            "error_code": self.error_code,
        }
        if data is not None:
            rich_detail["data"] = data

        super().__init__(
            status_code=status_code,
            detail=rich_detail,
            headers=headers,
        )


# ────────────────────────────────────────
#              4xx Client Errors
# ────────────────────────────────────────


class ValidationException(APIException):
    """Input validation failed."""

    def __init__(
        self,
        detail: str = "Validation error",
        error_code: str = "VALIDATION_ERROR",
        data: dict | None = None,  # ← very useful: field → error messages
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code=error_code,
            data=data,
        )


class NotFoundException(APIException):
    """Resource not found."""

    def __init__(
        self,
        resource: str,
        identifier: str | int | None = None,
        error_code: str = "NOT_FOUND",
    ):
        if identifier is not None:
            msg = f"{resource} with ID '{identifier}' not found"
        else:
            msg = f"{resource} not found"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=msg,
            error_code=error_code,
        )


class UnauthorizedException(APIException):
    """Authentication failed / missing credentials."""

    def __init__(
        self,
        detail: str = "Authentication failed",
        error_code: str = "UNAUTHORIZED",
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
            headers={"WWW-Authenticate": "Bearer"},  # recommended for OAuth/JWT
        )


class ForbiddenException(APIException):
    """Authenticated but not authorized."""

    def __init__(
        self,
        detail: str = "Insufficient permissions",
        error_code: str = "FORBIDDEN",
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code=error_code,
        )


class ConflictException(APIException):
    """Resource already exists / invalid state transition."""

    def __init__(
        self,
        detail: str,
        error_code: str = "CONFLICT",
        data: dict | None = None,
    ):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code=error_code,
            data=data,
        )


# ────────────────────────────────────────
#              5xx Server Errors
# ────────────────────────────────────────


class DatabaseException(APIException):
    """Database operation failed (caught & handled)."""

    def __init__(
        self,
        detail: str = "Database operation failed",
        error_code: str = "DATABASE_ERROR",
        original_exc: Exception | None = None,
    ):
        data = None
        if original_exc:
            data = {"original_error": str(original_exc)}
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code,
            data=data,
        )


class ExternalServiceException(APIException):
    """Call to third-party service failed."""

    def __init__(
        self,
        service: str,
        detail: str,
        error_code: str = "EXTERNAL_SERVICE_ERROR",
        status_code: int = status.HTTP_502_BAD_GATEWAY,
    ):
        msg = f"{service} request failed: {detail}"
        super().__init__(
            status_code=status_code,
            detail=msg,
            error_code=error_code,
        )


# Optional: more specialized auth exceptions (very common)
class InvalidTokenException(UnauthorizedException):
    """Token is malformed, invalid signature, etc."""

    def __init__(self, reason: str | None = None):
        msg = "Invalid token"
        if reason:
            msg += f": {reason}"
        super().__init__(detail=msg, error_code="INVALID_TOKEN")


class ExpiredTokenException(UnauthorizedException):
    """Token has expired."""

    def __init__(self):
        super().__init__(detail="Token has expired", error_code="TOKEN_EXPIRED")


class InvalidRefreshTokenException(UnauthorizedException):
    """Refresh token is invalid/revoked/blacklisted."""

    def __init__(self, reason: str | None = None):
        msg = "Invalid refresh token"
        if reason:
            msg += f" — {reason}"
        super().__init__(detail=msg, error_code="REFRESH_TOKEN_INVALID")


# from fastapi import Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer
# from sqlalchemy.ext.asyncio import AsyncSession
# from jose import (
#     JWTError,
#     ExpiredSignatureError,
#     InvalidTokenError,
#     InvalidSignatureError,
# )
# import logging

# from . import schemas, crud, models
# from .dependencies import get_db
# from .exceptions import (
#     ExpiredTokenException,
#     InvalidRefreshTokenException,
#     UnauthorizedException,
#     APIException,
#     ConflictException,
# )
# from .utils import create_access_token, verify_password  # assuming you have these
# from .config import settings

# logger = logging.getLogger(__name__)

# oauth2_scheme = OAuth2PasswordBearer(
#     tokenUrl="token"
# )  # not used here, just for reference


# async def refresh_token(
#     refresh_token: str,
#     db: AsyncSession = Depends(get_db),
# ):
#     """
#     Refresh access token using a valid refresh token.
#     Returns new access + (optionally) new refresh token.
#     """
#     credentials_exception = UnauthorizedException(
#         detail="Could not validate credentials",
#     )

#     try:
#         # ────────────────────────────────────────────────
#         # Step 1: Decode and validate the refresh token
#         # ────────────────────────────────────────────────
#         payload = jwt.decode(
#             refresh_token,
#             key=settings.JWT_REFRESH_SECRET_KEY,  # ← usually different secret for refresh tokens
#             algorithms=[settings.JWT_ALGORITHM],
#         )

#         user_id: str = payload.get("sub")
#         token_type: str = payload.get("type")

#         if user_id is None:
#             raise InvalidRefreshTokenException(reason="Missing subject (sub) claim")

#         if token_type != "refresh":
#             raise InvalidRefreshTokenException(reason="Token is not a refresh token")

#         # Optional: check token version / jti / family if you implement token rotation
#         # jti = payload.get("jti")
#         # if await crud.is_token_blacklisted(db, jti):
#         #     raise InvalidRefreshTokenException(reason="Token has been revoked")

#     except ExpiredSignatureError:
#         raise ExpiredTokenException()

#     except InvalidSignatureError:
#         raise InvalidRefreshTokenException(reason="Invalid signature")

#     except InvalidTokenError as exc:
#         raise InvalidRefreshTokenException(
#             reason=f"Invalid token structure: {str(exc)}"
#         )

#     except JWTError as exc:
#         logger.warning(
#             f"JWT decode failed: {str(exc)}",
#             extra={"token": refresh_token[:16] + "..."},
#         )
#         raise InvalidRefreshTokenException(reason="Malformed or tampered token")

#     except Exception as exc:
#         logger.exception(
#             "Unexpected error while decoding refresh token",
#             extra={"token_prefix": refresh_token[:16] + "..."},
#         )
#         raise APIException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             message="Internal error during token validation",
#             error_code="AUTH_REFRESH_DECODE_FAILURE",
#         )

#     # ────────────────────────────────────────────────
#     # Step 2: Fetch the user
#     # ────────────────────────────────────────────────
#     user = await crud.get_user_by_id(db, user_id=int(user_id))

#     if user is None:
#         raise NotFoundException(resource="User", identifier=user_id)

#     if not user.is_active:
#         raise ForbiddenException(detail="User account is deactivated")

#     # Optional: check last password change date vs token issued at (iat)
#     # if user.password_changed_at and user.password_changed_at > payload.get("iat"):
#     #     raise InvalidRefreshTokenException(reason="Password was changed after token issuance")

#     # ────────────────────────────────────────────────
#     # Step 3: Create new tokens
#     # ────────────────────────────────────────────────
#     try:
#         new_access_token = create_access_token(
#             data={"sub": str(user.id), "scopes": user.scopes or []}
#         )

#         # Optional: rotate refresh token (security best practice)
#         new_refresh_token = create_refresh_token(
#             data={"sub": str(user.id), "type": "refresh"}
#         )

#         # Optional: blacklist old refresh token if doing family / rotation
#         # await crud.blacklist_token(db, refresh_token, expires_in=...)

#     except Exception as exc:
#         logger.exception("Failed to generate new tokens")
#         raise APIException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             message="Failed to generate new authentication tokens",
#             error_code="TOKEN_GENERATION_FAILED",
#         )

#     # ────────────────────────────────────────────────
#     # Step 4: Return result
#     # ────────────────────────────────────────────────
#     return schemas.TokenRefreshResponse(
#         access_token=new_access_token,
#         refresh_token=new_refresh_token,  # omit if not rotating
#         token_type="bearer",
#         expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
#     )
