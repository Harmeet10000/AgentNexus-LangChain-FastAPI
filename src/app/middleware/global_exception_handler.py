import traceback
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.utils import is_body_allowed_for_status_code
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.utils import APIException, APIResponse, ErrorCode, execution_path, http_error, logger


def _json_error_response(
    payload: APIResponse[Any],
    status_code: int,
    headers: Mapping[str, str] | None = None,
) -> Response:
    # `headers` is `Mapping`, not `dict`, and that is not incidental widening. Starlette
    # annotates `HTTPException.headers` as `Mapping[str, str] | None` and `Response.__init__`
    # accepts the same, so `Mapping` is what both the source and the sink actually use.
    # Annotating it `dict` here made the two `headers=exc.headers` call sites below fail
    # `ty` — a Mapping is not assignable to a dict — and the original fix was a per-line
    # invalid-argument-type suppression on each. Both are gone: they silenced a correct
    # diagnostic about a needlessly narrow annotation, and each would have gone on masking any
    # *real* argument-type error introduced at that line later.
    #
    # Do not spell that directive out in a comment here, in backticks or otherwise. `ty`
    # matches the pattern anywhere in comment text and will read the mention as a live
    # declaration, then report it as an unused one — a self-inflicted diagnostic. Describe it.
    # 204/304/1xx forbid a message body. FastAPI's own `http_exception_handler` — which
    # `register_exception_handlers` below displaces for every plain HTTPException — guarded
    # this with the same helper. Displacing a handler means inheriting its obligations, so
    # the guard moves here rather than being dropped: without it a `raise
    # HTTPException(status_code=204)` would emit a JSON envelope plus a Content-Length on a
    # response the HTTP spec says has neither, and a real ASGI server (unlike TestClient)
    # rejects that as a protocol error.
    if not is_body_allowed_for_status_code(status_code):
        return Response(status_code=status_code, headers=headers)

    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


async def global_exception_handler(_request: Request, exc: Exception) -> Response:
    settings = get_settings()

    # Extract the current function chain from our ContextVar.
    #
    # The `[]` default is load-bearing, not defensive padding. `execution_path` is set by
    # `RequestStateLoggingMiddleware`, which resets it in a `finally` block on the way out.
    # An exception that escapes `ExceptionMiddleware` therefore passes *through* that reset
    # before `ServerErrorMiddleware` invokes this handler, so by the time branch 4 runs the
    # ContextVar is unset again and a bare `.get()` raises `LookupError` — inside the very
    # handler `ServerErrorMiddleware` called to render the error. Starlette does not catch
    # that, so the client received a bodiless 500: the catch-all branch was registered but
    # could never emit its envelope. Same reasoning covers `request_state` in
    # `app.utils.http_response._build_request_meta`.
    current_flow = " -> ".join(execution_path.get([]))

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

        payload = http_error(
            message=message,
            status_code=status_code,
            data=data,
            error_code=error_code,
            flow=current_flow,
        )
        return _json_error_response(payload, status_code, headers=exc.headers)

    # ────────────────────────────────────────────────
    # 2. Pydantic / FastAPI validation errors (422)
    # ────────────────────────────────────────────────
    if isinstance(exc, RequestValidationError):
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        error_code = ErrorCode.VALIDATION_ERROR
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

        payload: APIResponse[Any] = http_error(
            message=message,
            status_code=status_code,
            data={"errors": validation_errors},
            error_code=error_code,
            flow=current_flow,
        )
        return _json_error_response(payload, status_code)

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

        payload = http_error(
            message=message,
            status_code=status_code,
            error_code=error_code,
            flow=current_flow,
        )
        return _json_error_response(payload, status_code, headers=exc.headers)

    # ────────────────────────────────────────────────
    # 4. Catch-all — unexpected server errors (500)
    # ────────────────────────────────────────────────
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = ErrorCode.INTERNAL_SERVER_ERROR
    message = "An unexpected error occurred"

    trace = traceback.format_exc() if settings.ENVIRONMENT != "production" else None
    exc_type = type(exc).__name__
    path = execution_path.get([])
    last_function = path[-1] if path else "unknown_layer"
    dynamic_message = f"Unhandled {exc_type} crashed in {last_function}"

    logger.bind(
        status_code=status_code, error_code=error_code, crashed_at_flow=current_flow
    ).exception(dynamic_message)

    payload = http_error(
        message=message,
        status_code=status_code,
        error_code=error_code,
        trace=trace,
        flow=current_flow,
    )
    return _json_error_response(payload, status_code)


def register_exception_handlers(app: FastAPI) -> None:
    """Install :func:`global_exception_handler` for every error class the app can raise.

    **Load-bearing. Deleting a line here silently deletes a branch of the handler above —
    no exception, no warning, no failing import, no lint finding.** The application shipped
    for months with three of that handler's four branches unreachable for exactly this
    reason. Do not "simplify" this to the single ``Exception`` registration it replaced.

    Starlette splits ``add_exception_handler`` across two middlewares, and they are not
    interchangeable:

    * ``Exception`` — that exact key, nothing else — is pulled out and given to
      ``ServerErrorMiddleware``, the *outermost* wrapper. It is a last-resort 500 net for
      whatever escaped everything below it, and it re-raises after responding.
    * Every other key goes to ``ExceptionMiddleware``, the *innermost* wrapper. When an
      exception reaches it, it walks ``type(exc).__mro__`` and dispatches to the handler
      registered for the first class it finds in its registry.

    That MRO walk is why ``Exception`` alone was not enough. ``FastAPI.__init__`` pre-seeds
    the registry with ``setdefault(HTTPException, http_exception_handler)`` and
    ``setdefault(RequestValidationError, request_validation_exception_handler)``. Since
    ``APIException`` inherits from ``HTTPException``, the walk for e.g.
    ``ServiceUnavailableException`` hit FastAPI's entry three classes early and returned
    ``{"detail": ...}``; the elaborate ``APIException`` branch never executed for any
    request. And because ``setdefault`` only fills a gap, *omitting* a registration is
    silently accepted — there is nothing to notice.

    **The key object matters more than the class name.** FastAPI keys its default entry on
    ``starlette.exceptions.HTTPException``, *not* ``fastapi.exceptions.HTTPException``. Those
    are two distinct class objects and both sit in the MRO. Registering Starlette's class
    replaces FastAPI's entry (same dict key) and so covers FastAPI's subclass, the
    ``APIException`` family, and code that raises Starlette's class directly. Registering
    FastAPI's subclass instead would leave FastAPI's entry in place, and a bare
    ``starlette.HTTPException`` — which Starlette's own router raises — would still bypass
    us. Verify with ``FastAPI().exception_handlers`` keys' ``__module__`` before changing
    this line.

    ``APIException`` is registered explicitly even though the ``HTTPException`` key already
    covers the family by inheritance. Being the most-derived key, it wins whatever else is
    later registered against either ``HTTPException`` class: this family is the project's
    primary error channel and its envelope must not be contingent on a broader registration
    staying put.

    ``WebSocketRequestValidationError`` is deliberately left with FastAPI's own handler. It
    closes the socket with a code; a websocket scope cannot send an HTTP JSON envelope.
    """
    # Order is irrelevant — resolution is by MRO specificity, not insertion — but is kept
    # broad-to-narrow to read as the fallback chain it behaves like.
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(StarletteHTTPException, global_exception_handler)
    app.add_exception_handler(RequestValidationError, global_exception_handler)
    app.add_exception_handler(APIException, global_exception_handler)
