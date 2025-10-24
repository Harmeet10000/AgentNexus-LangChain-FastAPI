"""Global exception handlers."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.middleware.server_middleware import get_correlation_id


class AppException(Exception):
    """Base application exception."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ValidationException(AppException):
    """Validation error."""

    def __init__(self, message: str):
        super().__init__(message, status_code=422)


class NotFoundException(AppException):
    """Resource not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class UnauthorizedException(AppException):
    """Unauthorized access."""

    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, status_code=401)


class ForbiddenException(AppException):
    """Forbidden access."""

    def __init__(self, message: str = "Forbidden"):
        super().__init__(message, status_code=403)


async def app_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle custom app exceptions."""
    correlation_id = get_correlation_id()
    if isinstance(exc, AppException):
        logger.error(
            f"AppException: {exc.message}",
            extra={"correlation_id": correlation_id, "path": request.url.path},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "correlation_id": correlation_id},
        )
    logger.error(
        f"Unexpected exception type in app_exception_handler: {type(exc).__name__}",
        extra={"correlation_id": correlation_id, "path": request.url.path},
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "correlation_id": correlation_id},
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    correlation_id = get_correlation_id()
    logger.warning(
        f"HTTPException: {exc.detail}",
        extra={
            "correlation_id": correlation_id,
            "status": exc.status_code,
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "correlation_id": correlation_id},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    correlation_id = get_correlation_id()
    logger.exception(
        "Unhandled exception",
        extra={"correlation_id": correlation_id, "path": request.url.path},
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "correlation_id": correlation_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers."""
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
