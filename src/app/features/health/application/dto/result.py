"""Typed results produced by outbound health probes."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from typing import Self


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    WARNING = "warning"
    UNKNOWN = "unknown"


class DependencyHealth(BaseModel):
    """Health status returned by one outward dependency probe."""

    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    name: str
    status: HealthStatus
    latency_ms: float = 0.0
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
    """Response body for the deep health endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    status: HealthStatus
    version: str = "1.0.0"
    git_sha: str = "unknown"
    build_date: str = "unknown"
    dependencies: list[DependencyHealth] = Field(default_factory=list)


class ComponentCheck(BaseModel):
    """One dependency check in its explicit state.

    The status/state pair is deliberately stable. Adapter-specific telemetry
    belongs in ``details`` so adding a driver version or latency cannot change
    the health contract consumed by callers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: HealthStatus
    state: str
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def healthy(cls, state: str = "connected", **details: Any) -> Self:
        return cls(status=HealthStatus.HEALTHY, state=state, details=dict(details))

    @classmethod
    def degraded(cls, state: str, message: str | None = None, **details: Any) -> Self:
        return cls(
            status=HealthStatus.DEGRADED,
            state=state,
            error=message,
            details=dict(details),
        )

    @classmethod
    def unhealthy(cls, error: str, state: str = "disconnected", **details: Any) -> Self:
        return cls(status=HealthStatus.UNHEALTHY, state=state, error=error, details=dict(details))

    @classmethod
    def warning(cls, state: str, **details: Any) -> Self:
        return cls(status=HealthStatus.WARNING, state=state, details=dict(details))

    @classmethod
    def timeout(cls, component: str, timeout_seconds: float) -> Self:
        return cls(
            status=HealthStatus.UNHEALTHY,
            state="timeout",
            error="timeout",
            details={"component": component, "timeout_seconds": timeout_seconds},
        )

    @classmethod
    def not_configured(cls) -> Self:
        return cls(status=HealthStatus.UNKNOWN, state="not_configured")


class AgentMemoryCheck(ComponentCheck):
    """Cognee's graph-procedure precondition as named telemetry."""

    graph_procedures_available: bool = False
    graph_reachable: bool = False
    embedding_dimension: int | None = None


type ProbeResult = ComponentCheck
