"""Compatibility imports for health application and HTTP DTOs."""

from app.features.health.adapters.inbound.http.dto import SelfInfoDTO
from app.features.health.application.dto.aggregate import (
    HealthChecksDTO,
    HealthDataDTO,
    HealthResultDTO,
)
from app.features.health.application.dto.result import AgentMemoryCheck, ComponentCheck

__all__ = [
    "AgentMemoryCheck",
    "ComponentCheck",
    "HealthChecksDTO",
    "HealthDataDTO",
    "HealthResultDTO",
    "SelfInfoDTO",
]
