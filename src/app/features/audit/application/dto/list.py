"""Immutable query and result DTOs for listing audit entries."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .record import RecordAuditEntryOutput


class ListAuditEntriesInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_type: str | None = None
    entity_id: str | None = None
    action: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    limit: int = Field(default=100, gt=0, le=1000)
    offset: int = Field(default=0, ge=0)


class ListAuditEntriesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[RecordAuditEntryOutput, ...]
    total: int
