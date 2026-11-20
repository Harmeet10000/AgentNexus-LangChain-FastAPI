"""Aggregates returned by the health application service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.kernel.result import HealthStatus

from .result import AgentMemoryCheck, ComponentCheck


class HealthChecksDTO(BaseModel):
    """Per-component application health checks."""

    model_config = ConfigDict(extra="forbid")

    database: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    redis: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    postgres: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    neo4j: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    graphiti: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    celery: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    memory: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
    disk: ComponentCheck = Field(default_factory=ComponentCheck.not_configured)
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
    """Application result consumed by the HTTP presenter."""

    model_config = ConfigDict(extra="forbid")

    message: str
    status_code: int
    data: HealthDataDTO
