"""Typed payloads for expected internal failures."""

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.utils.codes import ErrorCode

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


class AppError(BaseModel):
    """Base expected failure payload for internal Result values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    details: ErrorDetails | None = None
    retryable: bool = False
    source: str | None = None


class ValidationAppError(AppError):
    """Expected validation or normalization failure."""

    kind: Literal["validation"] = "validation"
    code: str = ErrorCode.VALIDATION_ERROR


class NotFoundAppError(AppError):
    """Expected missing resource failure."""

    kind: Literal["not_found"] = "not_found"
    code: str = ErrorCode.NOT_FOUND
    resource: str = "Resource"
    identifier: str | int | None = None


class ConflictAppError(AppError):
    """Expected conflict or invalid state transition."""

    kind: Literal["conflict"] = "conflict"
    code: str = ErrorCode.CONFLICT


class InfrastructureAppError(AppError):
    """Expected infrastructure failure normalized at an adapter boundary."""

    kind: Literal["infrastructure"] = "infrastructure"
    code: str = ErrorCode.INFRASTRUCTURE_ERROR
    retryable: bool = True


class ExternalServiceAppError(AppError):
    """Expected upstream service failure normalized at an adapter boundary."""

    kind: Literal["external_service"] = "external_service"
    code: str = ErrorCode.EXTERNAL_SERVICE_ERROR
    retryable: bool = True
    service: str = Field(min_length=1)
