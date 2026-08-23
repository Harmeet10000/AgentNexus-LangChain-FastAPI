"""API middleware for error handling and request processing."""

from .api_versioning import ApiDeprecationMiddleware
from .global_exception_handler import global_exception_handler, register_exception_handlers
from .health_check import ALL_PROBES
from .otel import default_span_details
from .server_middleware import (
    RequestStateLoggingMiddleware,
    build_fastapi_guard_config,
    get_metrics,
    initialize_fastapi_guard,
)

__all__ = [
    "ALL_PROBES",
    "RequestStateLoggingMiddleware",
    "build_fastapi_guard_config",
    "default_span_details",
    "get_metrics",
    "global_exception_handler",
    "initialize_fastapi_guard"  # noqa: F822
    "ApiDeprecationMiddleware",
    "register_exception_handlers",
]
