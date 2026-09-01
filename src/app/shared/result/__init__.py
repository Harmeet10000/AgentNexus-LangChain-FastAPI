"""Internal Result conventions for expected recoverable failures."""

from .errors import (
    STATUS_BY_KIND,
    ErrorKind,
    FeatureError,
    http_status_for_kind,
)
from .logging import log_expected_failure
from .render import render_result

__all__ = [
    "STATUS_BY_KIND",
    "ErrorKind",
    "FeatureError",
    "http_status_for_kind",
    "log_expected_failure",
    "render_result",
]
