import functools
import logging
import sys
import time
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import UTC
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from _contextvars import Token

import opentelemetry.trace as otel_trace
from loguru import logger as loguru_logger
from opentelemetry.trace import Status, StatusCode
from returns.result import Failure

from app.config import get_settings

# 1. Context Variables
request_state: ContextVar[dict[str, Any]] = ContextVar("request_state")
execution_path: ContextVar[list[str]] = ContextVar("execution_path")


def set_request_actor(user_id: str | None) -> None:
    """Attach a validated authenticated actor to the active request context."""
    state = request_state.get({})
    state["user_id"] = user_id


# 2. Console Formatter (Unchanged - Your logic here is perfect)
def console_format(record: dict[str, Any]) -> str:
    """Format logs for console with INFO/META structure."""
    level = record["level"].name
    time_utc = record["time"].astimezone(UTC)
    time_str = time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    message = record["message"]
    # Escape braces so loguru's format_map pass does not interpret
    # user content as format fields (KeyError -> lost record).
    message_escaped = message.replace("{", "{{").replace("}", "}}")

    colors: dict[str, str] = {
        "DEBUG": "<cyan>",
        "INFO": "<green>",
        "WARNING": "<yellow>",
        "ERROR": "<red>",
        "CRITICAL": "<red><bold>",
    }
    color = colors.get(level, "<white>")
    end_color = "</>"

    fmt = f"{color}{level}{end_color} <dim>[{time_str}]</dim> {message_escaped}"

    extra_data = {k: v for k, v in record["extra"].items() if not k.startswith("_")}

    trace_id_str = record.get("extra", {}).get("trace_id", "")
    if trace_id_str:
        fmt += f" <green><b>trace_id</b>=<cyan>{trace_id_str[:16]}...</cyan></green>"

    if extra_data:
        meta_parts = [f"<cyan>{k}</>={v!r}" for k, v in extra_data.items() if k != "trace_id"]
        meta_str = " ".join(meta_parts)
        # Escape braces in rendered extra values for the same reason as message.
        meta_str = meta_str.replace("{", "{{").replace("}", "}}")
        fmt += f" <dim>|</dim> {meta_str}"

    if record["exception"]:
        fmt += "\n{exception}"

    return fmt + "\n"


def setup_logging(*, level: str | None = None) -> None:
    """Configure the single stdout/stderr Loguru pipeline from application settings."""
    settings = get_settings() if level is None else None
    configured_level = level or (settings.LOG_LEVEL if settings else "INFO")
    serialize = bool(settings and getattr(settings, "LOG_FORMAT", "text").lower() == "json")
    loguru_logger.remove()

    loguru_logger.add(
        sink=sys.stderr,
        format=console_format,
        level=configured_level,
        colorize=not serialize,
        serialize=serialize,
    )  # ty:ignore[no-matching-overload]
    _install_stdlib_bridge()


_REDACTED = "*** REDACTED ***"
_SENSITIVE_KEY_PARTS = (
    "access_key",
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "credit_card",
    "password",
    "private_key",
    "secret",
    "session_token",
    "token",
)


def _is_sensitive_key(key: object) -> bool:
    normalized_key = str(key).lower().replace("-", "_")
    return any(part in normalized_key for part in _SENSITIVE_KEY_PARTS)


def _redact_value(value: Any, *, depth: int = 0) -> Any:
    """Return a bounded redacted copy of structured log metadata."""
    if depth >= 8:
        return "<nested value omitted>"
    if isinstance(value, Mapping):
        return {
            key: _REDACTED if _is_sensitive_key(key) else _redact_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, depth=depth + 1) for item in value)
    if isinstance(value, set):
        return {_redact_value(item, depth=depth + 1) for item in value}
    return value


def redact_sensitive_data(record: Any) -> None:
    """Redact credential-bearing keys throughout structured log metadata."""
    context = request_state.get({})
    record["extra"].update({key: value for key, value in context.items() if value is not None})
    record["extra"] = _redact_value(record["extra"])


class _InterceptHandler(logging.Handler):
    """Forward stdlib records into the configured Loguru sinks."""

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        loguru_logger.opt(exception=record.exc_info).log(level, record.getMessage())


def _install_stdlib_bridge() -> None:
    handler = _InterceptHandler()
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.NOTSET)
    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "starlette",
        "celery",
        "sqlalchemy",
    ):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [handler]
        stdlib_logger.propagate = False
        stdlib_logger.setLevel(logging.NOTSET)


setup_logging(level="INFO")
logger = loguru_logger.patch(patcher=redact_sensitive_data)


# 3. The Trace Decorator (With Timing & State Isolation)
def trace_layer(layer_name: str) -> Any:
    """Decorator to track function execution flow and timing."""

    def decorator(func) -> Any:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            tracer = otel_trace.get_tracer(__name__)

            # 1. Update Breadcrumbs (Copy to avoid mutating parent state)
            # .get([]) — not bare .get(): execution_path is set only in
            # middleware/server_middleware.py, so outside an HTTP request
            # (Celery tasks, LangGraph nodes, CLI, unit tests) a defaultless
            # read raises LookupError. Same load-bearing default the global
            # exception handler already carries; an empty flow degrades to a
            # single-function breadcrumb.
            current_flow = execution_path.get([]).copy()
            current_flow.append(func.__name__)

            # VERY IMPORTANT: Save the token to reset later
            token: Token[list[str]] = execution_path.set(current_flow)
            flow_str = " -> ".join(current_flow)

            span_name = f"layer.{layer_name}.{func.__module__}.{func.__name__}"
            attrs = {"layer.name": layer_name, "function.name": func.__name__}
            with (
                tracer.start_as_current_span(span_name, attributes=attrs) as span,
                logger.contextualize(layer=layer_name, flow=flow_str),
            ):
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    span.set_attribute("layer.duration_ms", duration_ms)
                    if isinstance(result, Failure):
                        span.set_attribute("result.outcome", "failure")

                    logger.bind(layer_duration_ms=duration_ms, function_name=func.__name__).debug(
                        "Exiting layer"
                    )
                    return result  # noqa: TRY300 — return must be inside try for trace layer span recording

                except Exception as e:
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("layer.duration_ms", duration_ms)
                    logger.bind(
                        layer_duration_ms=duration_ms,
                        function_name=func.__name__,
                        error=str(e),
                    ).exception("Layer failed")
                    raise

                finally:
                    execution_path.reset(token)

        return wrapper

    return decorator
