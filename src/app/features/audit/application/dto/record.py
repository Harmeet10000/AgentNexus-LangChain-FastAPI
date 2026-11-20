"""Immutable command and result DTOs for recording audit entries."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RecordAuditEntryInput(BaseModel):
    """Data accepted by the audit recorder facade."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_type: str
    entity_id: str
    action: str
    user_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    changes: dict[str, object] = Field(default_factory=dict)
    metadata_: dict[str, object] = Field(default_factory=dict)


class RecordAuditEntryOutput(BaseModel):
    """Immutable representation returned after an entry is flushed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    entity_type: str
    entity_id: str
    action: str
    user_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    changes: dict[str, object] = Field(default_factory=dict)
    metadata_: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
