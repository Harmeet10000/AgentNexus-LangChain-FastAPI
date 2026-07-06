import functools
import sys
import time
from contextvars import ContextVar
from datetime import UTC
from typing import TYPE_CHECKING, Any

import opentelemetry.trace as otel_trace
from loguru import Logger
from loguru import logger as loguru_logger

if TYPE_CHECKING:
    from _contextvars import Token

# Assuming you have these from your existing codebase
# from app.config.settings import get_settings
# from string_utils import generate  # Wherever your generate() comes from

# 1. Context Variables
request_state: ContextVar[dict[str, Any]] = ContextVar("request_state", default={})  # noqa: B039
execution_path: ContextVar[list[str]] = ContextVar("execution_path", default=[])  # noqa: B039


# 2. Console Formatter (Unchanged - Your logic here is perfect)
def console_format(record: dict[str, Any]) -> str:
    """Format logs for console with INFO/META structure."""
    level = record["level"].name
    time_utc = record["time"].astimezone(UTC)
    time_str = time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    message = record["message"]

    colors: dict[str, str] = {
        "DEBUG": "<cyan>",
        "INFO": "<green>",
        "WARNING": "<yellow>",
        "ERROR": "<red>",
        "CRITICAL": "<red><bold>",
    }
    color = colors.get(level, "<white>")
    end_color = "</>"

    fmt = f"{color}{level}{end_color} <dim>[{time_str}]</dim> {message}"

    extra_data = {k: v for k, v in record["extra"].items() if not k.startswith("_")}

    trace_id_str = record.get("extra", {}).get("trace_id", "")
    if trace_id_str:
        fmt += f" <green><b>trace_id</b>=<cyan>{trace_id_str[:16]}...</cyan></green>"

    if extra_data:
        meta_parts = [f"<cyan>{k}</>={v!r}" for k, v in extra_data.items() if k != "trace_id"]
        meta_str = " ".join(meta_parts)
        fmt += f" <dim>|</dim> {meta_str}"

    if record["exception"]:
        fmt += "\n{exception}"

    return fmt + "\n"


def setup_logging() -> None:
    """Configure loguru logger with console and file handlers."""
    # settings = get_settings()
    loguru_logger.remove()

    loguru_logger.add(
        sink=sys.stderr,
        format=console_format,
        level="DEBUG",  # Set to debug to see the layer timings
        colorize=True,
    )  # ty:ignore[no-matching-overload]
    # File handler with JSON serialization
    # settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    # loguru_logger.add(
    #     sink=settings.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    #     format="{message}",
    #     level=settings.LOG_LEVEL,
    #     rotation=settings.LOG_ROTATION,
    #     retention=settings.LOG_RETENTION,
    #     compression=settings.LOG_COMPRESSION,
    #     serialize=True,
    #     backtrace=settings.LOG_BACKTRACE,
    #     diagnose=settings.LOG_DIAGNOSE,
    # )


def redact_sensitive_data(record) -> None:
    """Intercepts the log record and blanks out dangerous keys."""
    sensitive_keys = {"password", "token", "credit_card", "secret"}

    # We iterate through the extra data bound to the log
    for key, value in list(record["extra"].items()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            record["extra"][key] = "*** REDACTED ***"

        # If a whole dictionary is passed (like payment_data), we can scrub inside it too
        elif isinstance(value, dict):
            for sub_key in value:
                if any(sensitive in sub_key.lower() for sensitive in sensitive_keys):
                    record["extra"][key][sub_key] = "*** REDACTED ***"


setup_logging()
logger: Logger = loguru_logger.patch(patcher=redact_sensitive_data)


# 3. The Trace Decorator (With Timing & State Isolation)
def trace_layer(layer_name: str) -> Any:
    """Decorator to track function execution flow and timing."""

    def decorator(func) -> Any:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            _tracer = otel_trace.get_tracer(__name__)

            # 1. Update Breadcrumbs (Copy to avoid mutating parent state)
            current_flow = execution_path.get().copy()
            current_flow.append(func.__name__)

            # VERY IMPORTANT: Save the token to reset later
            token: Token[list[str]] = execution_path.set(current_flow)
            flow_str = " -> ".join(current_flow)

            span_name = f"layer.{layer_name}"
            attrs = {"layer.name": layer_name, "function.name": func.__name__}
            with _tracer.start_as_current_span(span_name, attributes=attrs) as span, \
                    logger.contextualize(layer=layer_name, flow=flow_str):
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    span.set_attribute("layer.duration_ms", duration_ms)

                    logger.bind(layer_duration_ms=duration_ms).debug(f"Exiting {func.__name__}")
                    return result  # noqa: TRY300

                except Exception as e:
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    span.record_exception(e)
                    span.set_attribute("layer.duration_ms", duration_ms)
                    logger.bind(layer_duration_ms=duration_ms).error(
                        f"Failed in {func.__name__} with error: {e}"
                    )
                    raise

                finally:
                    execution_path.reset(token)

        return wrapper

    return decorator
