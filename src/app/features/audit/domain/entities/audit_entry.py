"""Framework-free audit entry entity."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """The append-only audit record exchanged between use cases and ports."""

    entity_type: str
    entity_id: str
    action: str
    user_id: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    changes: dict[str, object] = field(default_factory=dict)
    metadata_: dict[str, object] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
