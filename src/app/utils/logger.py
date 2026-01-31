import sys
from datetime import datetime, timezone
from typing import Any

from loguru import logger as loguru_logger

from app.config.settings import get_settings


def console_format(record: dict[str, Any]) -> str:
    """Format logs for console with INFO/META structure."""
    level = record["level"].name
    time_utc = record["time"].astimezone(timezone.utc)
    time = time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    message = record["message"]

    # Color mapping
    colors = {
        "DEBUG": "<cyan>",
        "INFO": "<green>",
        "WARNING": "<yellow>",
        "ERROR": "<red>",
        "CRITICAL": "<red><bold>",
    }
    color = colors.get(level, "<white>")
    end_color = "</>"

    # Base format
    fmt = f"{color}{level}{end_color} <dim>[{time}]</dim> {message}"

    # Add extra data (filter out internal loguru keys)
    extra_data = {k: v for k, v in record["extra"].items() if not k.startswith("_")}

    if extra_data:
        meta_parts = [f"<cyan>{k}</>={v!r}" for k, v in extra_data.items()]
        meta_str = " ".join(meta_parts)
        fmt += f" <dim>|</dim> {meta_str}"

    # Add exception if present
    if record["exception"]:
        fmt += "\n{exception}"

    return fmt + "\n"


def setup_logging() -> None:
    """Configure loguru logger with console and file handlers."""
    settings = get_settings()

    # Remove default handler
    loguru_logger.remove()

    # Console handler with colors and custom format
    loguru_logger.add(
        sys.stderr,
        format=console_format,  # type: ignore[arg-type]
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=settings.LOG_BACKTRACE,
        diagnose=settings.LOG_DIAGNOSE,
    )

    # File handler with JSON serialization
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    loguru_logger.add(
        settings.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        format="{message}",
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression=settings.LOG_COMPRESSION,
        serialize=True,
        backtrace=settings.LOG_BACKTRACE,
        diagnose=settings.LOG_DIAGNOSE,
    )


# Initialize logger with setup
setup_logging()

# Export configured logger
logger = loguru_logger
