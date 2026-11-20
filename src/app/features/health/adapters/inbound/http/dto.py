"""HTTP-only DTOs for the health feature."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SelfInfoDTO(BaseModel):
    """Basic service metadata returned by the self/live endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    server: str
    version: str
    client: str
    timestamp: float
