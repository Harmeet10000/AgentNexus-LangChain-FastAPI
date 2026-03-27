import traceback

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.utils import APIException, execution_path, http_error, logger


async def global_exception_handler(request: Request, exc: Exception) -> ORJSONResponse:
    settings = get_settings()

    # Extract the current function chain from our ContextVar
    current_flow = " -> ".join(execution_path.get())

    # ────────────────────────────────────────────────
    # 1. Custom APIException family
    # ────────────────────────────────────────────────
    if isinstance(exc, APIException):
        status_code = exc.status_code
        error_code = exc.error_code
        message = (
            exc.detail.get("message", str(exc.detail))
            if isinstance(exc.detail, dict)
            else str(exc.detail)
        )
        data = exc.detail.get("data") if isinstance(exc.detail, dict) else None

        return http_error(
            message=message,
            status_code=status_code,
            data=data,
            error_code=error_code,
            flow=current_flow,
        )

    # ────────────────────────────────────────────────
    # 2. Pydantic / FastAPI validation errors (422)
    # ────────────────────────────────────────────────
    if isinstance(exc, RequestValidationError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        error_code = "VALIDATION_ERROR"
        message = "Request validation failed"

        validation_errors = [
            {
                "field": " → ".join(map(str, err["loc"])),
                "message": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors()
        ]

        logger.bind(status_code=status_code, validation_errors=validation_errors[:3]).warning(
            message,
        )

        return http_error(
            message=message,
            status_code=status_code,
            data={"errors": validation_errors},
            error_code=error_code,
            flow=current_flow,
        )

    # ────────────────────────────────────────────────
    # 3. Plain HTTPException / Starlette exceptions
    # ────────────────────────────────────────────────
    if isinstance(exc, StarletteHTTPException):
        status_code = exc.status_code
        error_code = f"HTTP_{status_code}"
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error"

        log_call = logger.bind(status_code=status_code)
        if status_code < 500:
            log_call.warning(message)
        else:
            log_call.error(message)

        response = http_error(
            message=message,
            status_code=status_code,
            error_code=error_code,
            flow=current_flow,
        )

        if exc.headers:
            response.headers.update(exc.headers)
        return response

    # ────────────────────────────────────────────────
    # 4. Catch-all — unexpected server errors (500)
    # ────────────────────────────────────────────────
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "INTERNAL_SERVER_ERROR"
    message = "An unexpected error occurred"

    trace = traceback.format_exc() if settings.ENVIRONMENT != "production" else None
    exc_type = type(exc).__name__
    last_function = execution_path.get()[-1] if execution_path.get() else "unknown_layer"
    dynamic_message = f"Unhandled {exc_type} crashed in {last_function}"

    logger.bind(
        status_code=status_code, error_code=error_code, crashed_at_flow=current_flow
    ).exception(dynamic_message)

    return http_error(
        message=message,
        status_code=status_code,
        error_code=error_code,
        trace=trace,
        flow=current_flow,
    )
