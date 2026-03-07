import functools
import sys
import time
from contextvars import ContextVar
from datetime import UTC
from typing import Any

from loguru import logger as loguru_logger

# Assuming you have these from your existing codebase
# from app.config.settings import get_settings
# from string_utils import generate  # Wherever your generate() comes from

# 1. Context Variables
request_state: ContextVar[dict[str, Any]] = ContextVar("request_state", default={})
execution_path: ContextVar[list[str]] = ContextVar("execution_path", default=[])


# 2. Console Formatter (Unchanged - Your logic here is perfect)
def console_format(record: dict[str, Any]) -> str:
    """Format logs for console with INFO/META structure."""
    level = record["level"].name
    time_utc = record["time"].astimezone(UTC)
    time_str = time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    message = record["message"]

    colors = {
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

    if extra_data:
        meta_parts = [f"<cyan>{k}</>={v!r}" for k, v in extra_data.items()]
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
logger = loguru_logger.patch(patcher=redact_sensitive_data)


# 3. The Trace Decorator (With Timing & State Isolation)
def trace_layer(layer_name: str) -> Any:
    """Decorator to track function execution flow and timing."""

    def decorator(func) -> Any:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()

            # 1. Update Breadcrumbs (Copy to avoid mutating parent state)
            current_flow = execution_path.get().copy()
            current_flow.append(func.__name__)

            # VERY IMPORTANT: Save the token to reset later
            token = execution_path.set(current_flow)
            flow_str = " -> ".join(current_flow)

            # 2. Execute with Context
            with logger.contextualize(layer=layer_name, flow=flow_str):
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

                    # Log successful completion of the layer with timing
                    logger.bind(layer_duration_ms=duration_ms).debug(f"Exiting {func.__name__}")
                    return result

                except Exception as e:
                    # Log failure timing before raising
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.bind(layer_duration_ms=duration_ms).error(f"Failed in {func.__name__}")
                    raise

                finally:
                    # 3. Reset the context so sibling functions don't inherit this path
                    execution_path.reset(token)

        return wrapper

    return decorator


