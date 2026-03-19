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


class TooManyRequestsException(APIException):
    """Request was rejected because the caller exceeded a rate limit."""

    def __init__(
        self,
        detail: str = "Too many requests",
        error_code: str = "TOO_MANY_REQUESTS",
        headers: dict[str, str] | None = None,
        data: dict | None = None,
    ):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code=error_code,
            headers=headers,
            data=data,
        )


class ServiceUnavailableException(APIException):
    """Service is temporarily unavailable."""

    def __init__(
        self,
        detail: str = "Service temporarily unavailable",
        error_code: str = "SERVICE_UNAVAILABLE",
        headers: dict[str, str] | None = None,
        data: dict | None = None,
    ):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code=error_code,
            headers=headers,
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
