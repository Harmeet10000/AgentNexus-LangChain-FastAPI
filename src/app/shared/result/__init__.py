"""Internal Result conventions for expected recoverable failures."""

# render imports app.utils.logger; keep it ahead of logging to avoid package-init recursion.
# ruff: noqa: I001

from .errors import (
    STATUS_BY_KIND,
    ErrorKind,
    FeatureError,
    NotFoundError,
    http_status_for_kind,
)
from .render import (
    APIResponse,
    DependencyHealth,
    ErrorDetail,
    HealthResponse,
    HealthStatus,
    RequestMeta,
    http_error,
    http_response,
    render_exception,
    render_result,
)
from .logging import log_expected_failure

__all__ = [
    "STATUS_BY_KIND",
    "APIResponse",
    "DependencyHealth",
    "ErrorDetail",
    "ErrorKind",
    "FeatureError",
    "HealthResponse",
    "HealthStatus",
    "NotFoundError",
    "RequestMeta",
    "http_error",
    "http_response",
    "http_status_for_kind",
    "log_expected_failure",
    "render_exception",
    "render_result",
]
