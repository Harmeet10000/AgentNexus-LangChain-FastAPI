from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import ASCENDING, IndexModel


class TokenAuditLog(Document):
    """Persistent session audit trail.

    Redis is the authoritative active-session store.
    This collection provides history, revocation tracing, and per-device listings.
    The TTL index on expires_at lets MongoDB auto-purge stale documents — no cron needed.
    """

    session_id: str
    user_id: PydanticObjectId
    device_id: str
    device_name: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: datetime = Field(default_factory=datetime.now(tz=datetime.timezone.utc))
    expires_at: datetime
    revoked_at: datetime | None = None
    is_revoked: bool = Field(default=False)
    revoke_reason: str | None = None  # "logout" | "password_reset" | "manual_revoke" | "revoke_all"

    class Settings:
        name = "token_audit_logs"
        indexes = [
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("session_id", ASCENDING)], unique=True),
            IndexModel(
                [("expires_at", ASCENDING)],
                expireAfterSeconds=0,  # MongoDB TTL index — auto-purge expired docs
            ),
        ]
