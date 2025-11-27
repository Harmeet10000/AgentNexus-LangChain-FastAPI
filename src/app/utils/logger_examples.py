"""
Structured Logging Examples with Loguru

This file demonstrates how to use loguru for structured logging
with the patterns you mentioned.
"""

from src.app.utils.logger import logger


# ============================================================================
# BASIC STRUCTURED LOGGING PATTERNS
# ============================================================================

def example_controller_response() -> None:
    """Log controller responses with structured data."""
    response = {"status": 200, "data": {"user_id": 123, "name": "John"}}

    # Log with context
    logger.info(
        "CONTROLLER_RESPONSE",
        response=response,
        status_code=200,
        endpoint="/api/users/123",
    )
    # Output: INFO [timestamp] CONTROLLER_RESPONSE | response={...} status_code=200 endpoint='/api/users/123'


def example_database_query() -> None:
    """Log database operations with metrics."""
    logger.info(
        "DATABASE_QUERY",
        query="SELECT * FROM users WHERE id = ?",
        duration_ms=125,
        rows_affected=5,
        table="users",
    )


def example_authentication() -> None:
    """Log auth events with relevant context."""
    logger.info(
        "AUTH_SUCCESS",
        user_id="user-123",
        username="john@example.com",
        ip_address="192.168.1.1",
        method="oauth2",
    )


def example_error_logging() -> None:
    """Log errors with structured context."""
    logger.error(
        "AUTH_FAILED",
        username="john@example.com",
        reason="invalid_token",
        attempt_count=3,
        ip_address="192.168.1.1",
    )


# ============================================================================
# FASTAPI CONTROLLER EXAMPLE
# ============================================================================

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/api/v1/users/{user_id}")
async def get_user(user_id: int, request: Request) -> JSONResponse:
    """Example FastAPI endpoint with structured logging."""
    # Get user (pseudo-code)
    user = {"id": user_id, "name": "John Doe", "email": "john@example.com"}

    # Log controller response with structured context
    logger.info(
        "CONTROLLER_RESPONSE",
        endpoint=str(request.url),
        method=request.method,
        status_code=200,
        response=user,
        user_id=user_id,
        response_time_ms=12.5,
    )

    return JSONResponse(status_code=200, content=user)


@router.post("/api/v1/auth/login")
async def login(username: str, password: str) -> JSONResponse:
    """Example login endpoint with structured logging."""
    # Attempt authentication (pseudo-code)
    try:
        user = {"username": username, "token": "token-xyz"}

        logger.info(
            "AUTH_SUCCESS",
            username=username,
            method="email_password",
            user_id=123,
        )

        return JSONResponse(status_code=200, content=user)

    except Exception as e:
        logger.error(
            "AUTH_FAILED",
            username=username,
            reason=str(e),
            error_type=type(e).__name__,
        )
        return JSONResponse(status_code=401, content={"error": "Invalid credentials"})


# ============================================================================
# SERVICE LAYER LOGGING
# ============================================================================

class UserService:
    """Example service with structured logging."""

    async def create_user(self, username: str, email: str) -> dict:
        """Create a new user with logging."""
        logger.info("USER_CREATE_STARTED", username=username, email=email)

        try:
            # Pseudo-code: create user in database
            user = {"id": 123, "username": username, "email": email}

            logger.info(
                "USER_CREATE_SUCCESS",
                user_id=123,
                username=username,
                duration_ms=45,
            )

            return user

        except Exception as e:
            logger.error(
                "USER_CREATE_FAILED",
                username=username,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def update_user(self, user_id: int, updates: dict) -> dict:
        """Update user with structured logging."""
        logger.info(
            "USER_UPDATE_STARTED",
            user_id=user_id,
            fields=list(updates.keys()),
        )

        try:
            # Pseudo-code: update in database
            updated_user = {"id": user_id, **updates}

            logger.info(
                "USER_UPDATE_SUCCESS",
                user_id=user_id,
                fields=list(updates.keys()),
                duration_ms=30,
            )

            return updated_user

        except Exception as e:
            logger.error(
                "USER_UPDATE_FAILED",
                user_id=user_id,
                error=str(e),
            )
            raise


# ============================================================================
# MIDDLEWARE LOGGING EXAMPLE
# ============================================================================

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from time import time

app = FastAPI()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""

    async def dispatch(self, request: Request, call_next):
        """Log all HTTP requests and responses."""
        start_time = time()
        path = request.url.path
        method = request.method

        # Log request
        logger.info(
            "HTTP_REQUEST_START",
            method=method,
            path=path,
            client_ip=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
            duration = (time() - start_time) * 1000  # Convert to ms

            # Log response
            logger.info(
                "HTTP_REQUEST_SUCCESS",
                method=method,
                path=path,
                status_code=response.status_code,
                duration_ms=round(duration, 2),
            )

            return response

        except Exception as e:
            duration = (time() - start_time) * 1000

            logger.error(
                "HTTP_REQUEST_FAILED",
                method=method,
                path=path,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(duration, 2),
            )
            raise


app.add_middleware(LoggingMiddleware)


# ============================================================================
# USAGE WITH CONTEXT BINDING
# ============================================================================

def example_with_context_binding() -> None:
    """Example using loguru's context binding for correlation IDs."""
    from loguru import logger as log

    # Bind correlation ID to all subsequent logs
    correlation_id = "corr-abc123"

    def nested_function():
        # This will automatically include the correlation ID
        log.info("NESTED_OPERATION", operation="query_database")

    # Use context manager or bind method
    log.bind(correlation_id=correlation_id).info(
        "REQUEST_STARTED",
        endpoint="/api/users",
    )

    nested_function()

    log.bind(correlation_id=correlation_id).info(
        "REQUEST_COMPLETED",
        status_code=200,
    )


# ============================================================================
# CONSOLE OUTPUT EXAMPLES
# ============================================================================

"""
With the enhanced logger configuration, your logs will look like:

INFO [2025-11-25T10:30:45.123Z] CONTROLLER_RESPONSE | response={'status': 200} status_code=200 endpoint='/api/users/123'

INFO [2025-11-25T10:30:46.456Z] DATABASE_QUERY | query='SELECT * FROM users WHERE id = ?' duration_ms=125 rows_affected=5

ERROR [2025-11-25T10:30:47.789Z] AUTH_FAILED | username='john@example.com' reason='invalid_token' attempt_count=3

And in JSON logs (files):
{"timestamp": "2025-11-25T10:30:45.123000Z", "level": "INFO", "logger": "src.app.main", "message": "CONTROLLER_RESPONSE", "response": {...}, "status_code": 200}
"""
