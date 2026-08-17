"""Dunning request/response DTOs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime


class DunningConfigDTO(BaseModel):
    """Configure dunning retry intervals and maximum attempts.

    Applied to new subscriptions only (Requirement 21.3 / 47.3).
    """

    model_config = ConfigDict(extra="forbid")

    retry_delay_days: list[int] = Field(min_length=1, max_length=10)
    max_retries: int = Field(ge=1, le=10)


class DunningConfigResponse(BaseModel):
    """Current dunning strategy."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    retry_delay_days: list[int] = Field(serialization_alias="retryDelayDays")
    max_retries: int = Field(serialization_alias="maxRetries")


class RetryAttemptResponse(BaseModel):
    """A single retry attempt (executed or scheduled)."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    attempt: int
    scheduled_at: datetime = Field(serialization_alias="scheduledAt")
    executed_at: datetime | None = Field(default=None, serialization_alias="executedAt")
    status: str
    error_code: str | None = Field(default=None, serialization_alias="errorCode")
