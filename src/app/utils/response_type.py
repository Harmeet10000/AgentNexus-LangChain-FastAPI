from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class RequestMeta(BaseModel):
    """Request context echoed back in API responses."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    ip: str | None = Field(default=None)
    method: str | None = Field(default=None)
    url: str | None = Field(default=None)
    correlation_id: str | None = Field(default=None, serialization_alias="correlationId")


class ErrorDetail(BaseModel):
    """Normalized error payload for non-success responses."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    code: str
    message: str
    data: dict[str, object] | list[object] | str | None = Field(default=None)
    trace: str | None = Field(default=None)
    inner_error: str | None = Field(default=None, serialization_alias="innerError")
    flow: str | None = Field(default=None)


class APIResponse[T](BaseModel):
    """Default API response envelope for all HTTP handlers."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    success: bool = Field(default=True)
    status_code: int = Field(default=200, serialization_alias="statusCode")
    request: RequestMeta
    message: str = Field(default="Success")
    data: T | None = Field(default=None)
    error: ErrorDetail | None = Field(default=None)


# ---------------------------------------------------------------------------
# Health check models
# ---------------------------------------------------------------------------


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyHealth(BaseModel):
    """Health status of a single dependency."""

    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    name: str
    status: HealthStatus
    latency_ms: float = Field(default=0.0)
    message: str | None = None

    @classmethod
    def ok(cls, name: str, latency_ms: float = 0.0) -> Self:
        return cls(name=name, status=HealthStatus.HEALTHY, latency_ms=latency_ms)

    @classmethod
    def fail(cls, name: str, message: str, latency_ms: float = 0.0) -> Self:
        return cls(name=name, status=HealthStatus.UNHEALTHY, latency_ms=latency_ms, message=message)

    @classmethod
    def degraded(cls, name: str, message: str, latency_ms: float = 0.0) -> Self:
        return cls(name=name, status=HealthStatus.DEGRADED, latency_ms=latency_ms, message=message)


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    status: HealthStatus
    version: str = Field(default="1.0.0")
    dependencies: list[DependencyHealth] = Field(default_factory=list)


APIResponse.model_rebuild(force=True)
