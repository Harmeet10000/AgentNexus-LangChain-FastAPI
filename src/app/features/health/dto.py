"""DTOs for health feature responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class SelfInfoDTO(BaseModel):
    """Basic service metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, slots=True)  # ty:ignore[invalid-key]

    server: str
    version: str
    client: str
    timestamp: float


class HealthChecksDTO(BaseModel):
    """Per-component health checks."""

    model_config = ConfigDict(extra="forbid")

    database: dict[str, Any]
    redis: dict[str, Any]
    postgres: dict[str, Any]
    neo4j: dict[str, Any]
    celery: dict[str, Any]
    memory: dict[str, Any]
    disk: dict[str, Any]


class HealthDataDTO(BaseModel):
    """Aggregated health payload."""

    model_config = ConfigDict(extra="forbid")

    status: str
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
