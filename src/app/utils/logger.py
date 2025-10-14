"""Production-grade Loguru logging configuration."""
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger as loguru_logger
from pydantic_settings import BaseSettings
from app.shared.enums import Environment


class LogConfig(BaseSettings):
    """Logging configuration."""

    ENVIRONMENT: str = Environment.DEVELOPMENT
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: Path = Path("logs")
    LOG_ROTATION: str = "500 MB"
    LOG_RETENTION: str = "30 days"
    LOG_COMPRESSION: str = "zip"
    LOG_BACKTRACE: bool = True
    LOG_DIAGNOSE: bool = False

    class Config:
        env_file = ".env"


def serialize_record(record: Dict[str, Any]) -> str:
    """Custom JSON serialization for file logs."""
    import json

    log_meta = {}
    if record.get("extra", {}).get("meta"):
        meta = record["extra"]["meta"]
        if isinstance(meta, dict):
            for key, value in meta.items():
                if isinstance(value, Exception):
                    log_meta[key] = {
                        "name": value.__class__.__name__,
                        "message": str(value),
                        "trace": getattr(value, "__traceback__", "")
                    }
                else:
                    log_meta[key] = value

    log_data = {
        "level": record["level"].name,
        "message": record["message"],
        "timestamp": record["time"].isoformat(),
        "meta": log_meta
    }

    return json.dumps(log_data, indent=4)


def console_format(record: Dict[str, Any]) -> str:
    """Custom console format with colors."""
    level = record["level"].name
    time = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    message = record["message"]

    # Color mapping
    level_colors = {
        "DEBUG": "<cyan>",
        "INFO": "<blue>",
        "WARNING": "<yellow>",
        "ERROR": "<red>",
        "CRITICAL": "<red><bold>"
    }

    color = level_colors.get(level, "")
    meta = record.get("extra", {}).get("meta", {})
    meta_str = f"\n<magenta>META</magenta> {meta}" if meta else ""

    return f"{color}{level}</>  [<green>{time}</green>] {message}{meta_str}\n"


def setup_logging(config: Optional[LogConfig] = None) -> None:
    """Configure Loguru with Winston-like features."""
    if config is None:
        config = LogConfig()

    loguru_logger.remove()

    # Determine log level based on environment
    log_level = "DEBUG" if config.ENVIRONMENT == Environment.DEVELOPMENT else "INFO"

    # Console handler with custom format
    loguru_logger.add(
        sys.stdout,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[request_id]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        backtrace=config.LOG_BACKTRACE,
        diagnose=config.LOG_DIAGNOSE,
        enqueue=True,
    )

    # Create logs directory
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # File handler with JSON format (like Winston)
    loguru_logger.add(
        config.LOG_DIR / f"{config.ENVIRONMENT}.log",
        format=serialize_record,
        level="DEBUG",
        rotation=config.LOG_ROTATION,
        retention=config.LOG_RETENTION,
        compression=config.LOG_COMPRESSION,
        backtrace=config.LOG_BACKTRACE,
        diagnose=config.LOG_DIAGNOSE,
        enqueue=True,
    )

    loguru_logger.info("Logging configured")


# Async logger wrapper (like Winston setImmediate)
class Logger:
    """Custom logger wrapper."""

    @staticmethod
    def info(message: str, meta: Optional[Dict[str, Any]] = None) -> None:
        loguru_logger.bind(meta=meta or {}).info(message)

    @staticmethod
    def error(message: str, meta: Optional[Dict[str, Any]] = None) -> None:
        loguru_logger.bind(meta=meta or {}).error(message)

    @staticmethod
    def warn(message: str, meta: Optional[Dict[str, Any]] = None) -> None:
        loguru_logger.bind(meta=meta or {}).warning(message)

    @staticmethod
    def debug(message: str, meta: Optional[Dict[str, Any]] = None) -> None:
        loguru_logger.bind(meta=meta or {}).debug(message)


# Export logger instance
logger = Logger()

__all__ = ["logger", "setup_logging", "LogConfig"]
