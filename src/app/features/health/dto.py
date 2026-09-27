"""DTOs for health feature responses."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from app.shared.result import HealthStatus


class SelfInfoDTO(BaseModel):
    """Basic service metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    server: str
    version: str
    client: str
    timestamp: float


class ComponentCheck(BaseModel):
    """One dependency check in its explicit state.

    Replaces stringly-typed ``dict`` bags: every check answers with a status
    from the shared enum, a machine-readable state, and an optional error.
    Telemetry that is not state (latencies, versions, counters) lives in
    ``details`` so the status/state contract stays stable.
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
    """Agent-memory (cognee) check with its named procedure precondition.

    Absent graph procedures mean consolidation will refuse to run — reported
    as a named sub-field, never as a whole-check failure.
    """

    graph_procedures_available: bool = False
    graph_reachable: bool = False
    embedding_dimension: int | None = None


class HealthChecksDTO(BaseModel):
    """Per-component health checks."""

    model_config = ConfigDict(extra="forbid")

    database: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    redis: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    postgres: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    neo4j: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    graphiti: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    celery: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    memory: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    disk: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    # Agent memory (cognee) — distinct from `memory`, which is psutil RAM (N6).
    agent_memory: AgentMemoryCheck = Field(
        default_factory=AgentMemoryCheck.not_configured,
        serialization_alias="agentMemory",
    )


class HealthDataDTO(BaseModel):
    """Aggregated health payload."""

    model_config = ConfigDict(extra="forbid")

    status: HealthStatus
    timestamp: float
    application: dict[str, Any]
    system: dict[str, Any]
    checks: HealthChecksDTO


class HealthResultDTO(BaseModel):
    """Service result consumed by router response wrapper."""

    model_config = ConfigDict(extra="forbid")

    message: str
    status_code: int
    data: HealthDataDTO
