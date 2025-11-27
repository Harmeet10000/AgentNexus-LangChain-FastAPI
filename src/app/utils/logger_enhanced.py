"""
Enhanced loguru configuration with structured logging patterns.

Usage examples:
    logger.info("CONTROLLER_RESPONSE", response=response_data)
    logger.info("DATABASE_QUERY", query=sql, duration_ms=125, rows=42)
    logger.error("AUTH_FAILED", username=user, reason="invalid_token")
"""

import json
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

from loguru import logger as loguru_logger
from pydantic_settings import BaseSettings

from app.shared.enums import Environment


class LogConfig(BaseSettings):
    """Loguru logging configuration."""

    ENVIRONMENT: str = Environment.DEVELOPMENT
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: Path = Path("logs/")
    LOG_ROTATION: str = "5 MB"
    LOG_RETENTION: str = "30 days"
    LOG_COMPRESSION: str = "zip"
    LOG_BACKTRACE: bool = True
    LOG_DIAGNOSE: bool = False
    LOG_SERIALIZE: bool = True  # JSON serialization for file logs

    class Config:
        env_file = ".env.development"
        extra = "ignore"


def console_format(record: dict[str, Any]) -> str:
    """
    Format logs for console with structured data.

    Format: LEVEL [time] message | key=value key=value

    Example output:
        INFO [2025-11-25T10:30:45.123Z] CONTROLLER_RESPONSE | response={'status': 200}
    """
    level_name = record["level"].name
    message = record["message"]

    # ISO 8601 UTC timestamp
    time_utc = record["time"].astimezone(UTC)
    time_str = time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # Color mapping for console
    colors = {
        "DEBUG": "<cyan>",
        "INFO": "<green>",
        "WARNING": "<yellow>",
        "ERROR": "<red>",
        "CRITICAL": "<magenta><bold>",
    }
    color = colors.get(level_name, "<white>")
    end_color = "</>"

    # Base format: LEVEL [timestamp] message
    fmt = f"{color}{level_name}{end_color} <dim>[{time_str}]</dim> {message}"

    # Append structured data from extra (skip internal keys starting with _)
    extra_data = {k: v for k, v in record["extra"].items() if not k.startswith("_")}

    if extra_data:
        # Format: key=value key=value
        meta_parts = []
        for k, v in extra_data.items():
            # Pretty-print values for readability
            if isinstance(v, (dict, list)):
                v_str = json.dumps(v, default=str)
            else:
                v_str = repr(v)
            meta_parts.append(f"<cyan>{k}</>={v_str}")

        meta_str = " ".join(meta_parts)
        fmt += f" <dim>|</dim> {meta_str}"

    # Add exception traceback if present
    if record["exception"]:
        fmt += "\n{exception}"

    return fmt + "\n"


def json_format(record: dict[str, Any]) -> str:
    """
    Format logs as JSON for structured logging (file output).

    Useful for log aggregation tools (ELK, Datadog, CloudWatch).
    """
    # Extract structured data from extra
    extra_data = {k: v for k, v in record["extra"].items() if not k.startswith("_")}

    # Build JSON payload
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "message": record["message"],
        "function": record["function"],
        "line": record["line"],
        **extra_data,  # Spread structured data
    }

    # Add exception info if present
    if record["exception"]:
        exc_type, exc_value, _ = record["exception"]
        log_entry["exception"] = {
            "type": exc_type.__name__ if exc_type else None,
            "value": str(exc_value),
            "traceback": record["extra"].get("_traceback"),
        }

    return json.dumps(log_entry, default=str) + "\n"


def setup_logging(config: LogConfig | None = None) -> None:
    """
    Configure loguru with console and file handlers.
    Args:
        config: LogConfig instance. If None, creates default config.
    """
    if config is None:
        try:
            config = LogConfig()
        except Exception:
            config = LogConfig(
                ENVIRONMENT=Environment.DEVELOPMENT,
                LOG_LEVEL="DEBUG",
                LOG_DIR=Path("logs/"),
                LOG_ROTATION="5 MB",
                LOG_RETENTION="30 days",
                LOG_COMPRESSION="zip",
                LOG_BACKTRACE=True,
                LOG_DIAGNOSE=False,
                LOG_SERIALIZE=True,
            )

    # Remove default handler
    loguru_logger.remove()

    # Console handler: Pretty colors + structured data
    loguru_logger.add(
        sys.stderr,
        format=console_format,
        level=config.LOG_LEVEL,
        colorize=True,
        backtrace=config.LOG_BACKTRACE,
        diagnose=config.LOG_DIAGNOSE,
    )

    # File handler: JSON format for aggregation tools
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    loguru_logger.add(
        config.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        format=json_format if config.LOG_SERIALIZE else console_format,
        level=config.LOG_LEVEL,
        rotation=config.LOG_ROTATION,
        retention=config.LOG_RETENTION,
        compression=config.LOG_COMPRESSION,
        backtrace=config.LOG_BACKTRACE,
        diagnose=config.LOG_DIAGNOSE,
        serialize=False,  # We handle serialization in json_format
    )


# Initialize logger
logger = loguru_logger
