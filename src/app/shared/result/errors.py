"""Shared failure conveniences: generic leaf + kind-to-HTTP-status mapping.

The pure spine (`ErrorDetails`, `ErrorKind`, `FeatureError`) lives in
`app.kernel.errors`. This module holds what the render path needs on top.
"""

from enum import StrEnum
from typing import ClassVar

from app.kernel.errors import ErrorKind, FeatureError

class NotFoundCode(StrEnum):
    """Owned code vocabulary for the generic not-found failure."""

    NOT_FOUND = "NOT_FOUND"


class NotFoundError(FeatureError):
    """Generic not-found failure for boundaries without a feature owner."""

    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[NotFoundCode] = NotFoundCode.NOT_FOUND
    retryable: ClassVar[bool] = False


# Maps ErrorKind to HTTP status; INFRASTRUCTURE is refined by retryable (500 dead / 503 transient).
STATUS_BY_KIND: dict[ErrorKind, int] = {
    ErrorKind.VALIDATION: 422,
    ErrorKind.NOT_FOUND: 404,
    ErrorKind.CONFLICT: 409,
    ErrorKind.AUTHENTICATION: 401,
    ErrorKind.AUTHORIZATION: 403,
    ErrorKind.EXTERNAL_SERVICE: 502,
    ErrorKind.INFRASTRUCTURE: 500,
}


def http_status_for_kind(kind: ErrorKind, *, retryable: bool = False) -> int:
    """Return HTTP status for a kind, with INFRASTRUCTURE refined by retryable."""
    if kind is ErrorKind.INFRASTRUCTURE:
        return 503 if retryable else 500
    return STATUS_BY_KIND[kind]
