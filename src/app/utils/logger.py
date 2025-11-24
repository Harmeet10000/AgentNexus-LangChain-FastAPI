import sys
from pathlib import Path
from typing import Any

from loguru import logger as loguru_logger
from pydantic_settings import BaseSettings

from app.shared.enums import Environment


class LogConfig(BaseSettings):
    """Logging configuration."""

    ENVIRONMENT: str = Environment.DEVELOPMENT
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: Path = Path("logs/")
    LOG_ROTATION: str = "5 MB"
    LOG_RETENTION: str = "30 days"
    LOG_COMPRESSION: str = "zip"
    LOG_BACKTRACE: bool = True
    LOG_DIAGNOSE: bool = False

    class Config:
        env_file = ".env.development"
        extra = "ignore"  # Ignore extra fields from .env file


def console_format(record: dict[str, Any]) -> str:
    """Format logs for console with INFO/META structure."""
    level = record["level"].name
    time_utc = (
        record["time"].astimezone().astimezone(tz=__import__("datetime").timezone.utc)
    )
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


def setup_logging(config: LogConfig | None = None) -> None:
    """Configure loguru logger with console and file handlers."""
    if config is None:
        try:
            config = LogConfig()
        except Exception:
            # Fallback to defaults if config fails
            config = LogConfig(
                ENVIRONMENT=Environment.DEVELOPMENT,
                LOG_LEVEL="DEBUG",
                LOG_DIR=Path("logs/"),
                LOG_ROTATION="5 MB",
                LOG_RETENTION="30 days",
                LOG_COMPRESSION="zip",
                LOG_BACKTRACE=True,
                LOG_DIAGNOSE=False,
            )

    # Remove default handler
    loguru_logger.remove()

    # Console handler with colors and custom format
    loguru_logger.add(
        sys.stderr,
        format=console_format,  # type: ignore[arg-type]
        level=config.LOG_LEVEL,
        colorize=True,
        backtrace=config.LOG_BACKTRACE,
        diagnose=config.LOG_DIAGNOSE,
    )

    # File handler with JSON serialization
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    loguru_logger.add(
        config.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        format="{message}",
        level=config.LOG_LEVEL,
        rotation=config.LOG_ROTATION,
        retention=config.LOG_RETENTION,
        compression=config.LOG_COMPRESSION,
        serialize=True,
        backtrace=config.LOG_BACKTRACE,
        diagnose=config.LOG_DIAGNOSE,
    )


# Export configured logger
logger = loguru_logger
