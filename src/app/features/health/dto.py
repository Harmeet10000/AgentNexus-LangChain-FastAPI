"""Compatibility imports for health application and HTTP DTOs."""

import importlib

from app.features.health.application.dto.aggregate import (
    HealthChecksDTO,
    HealthDataDTO,
    HealthResultDTO,
)
from app.features.health.application.dto.result import AgentMemoryCheck, ComponentCheck

SelfInfoDTO = importlib.import_module("app.features.health.adapters.in.http.dto").SelfInfoDTO

__all__ = [
    "AgentMemoryCheck",
    "ComponentCheck",
    "HealthChecksDTO",
    "HealthDataDTO",
    "HealthResultDTO",
    "SelfInfoDTO",
]
