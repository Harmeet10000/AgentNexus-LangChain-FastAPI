"""Typed payloads for expected internal failures."""

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

type ErrorDetails = dict[str, object]


class ErrorKind(StrEnum):
    """Shared classification vocabulary — the only cross-feature error vocabulary."""

    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_SERVICE = "external_service"


class FeatureError(BaseModel):
    """Base for per-feature typed errors — classification is a ClassVar, never a field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str
    details: ErrorDetails | None = None
    source: str | None = None

    kind: ClassVar[ErrorKind]
    code: ClassVar[StrEnum]
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
