"""Typed payloads for expected internal failures."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type ErrorDetails = dict[str, object]


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
    code: str = "VALIDATION_ERROR"


class NotFoundAppError(AppError):
    """Expected missing resource failure."""

    kind: Literal["not_found"] = "not_found"
    code: str = "NOT_FOUND"
    resource: str = "Resource"
    identifier: str | int | None = None


class ConflictAppError(AppError):
    """Expected conflict or invalid state transition."""

    kind: Literal["conflict"] = "conflict"
    code: str = "CONFLICT"


class InfrastructureAppError(AppError):
    """Expected infrastructure failure normalized at an adapter boundary."""

    kind: Literal["infrastructure"] = "infrastructure"
    code: str = "INFRASTRUCTURE_ERROR"
    retryable: bool = True


class ExternalServiceAppError(AppError):
    """Expected upstream service failure normalized at an adapter boundary."""

    kind: Literal["external_service"] = "external_service"
    code: str = "EXTERNAL_SERVICE_ERROR"
    retryable: bool = True
    service: str = Field(min_length=1)
