"""Pure message schemas; importing them never loads a task implementation."""

from pydantic import BaseModel, Field


class CeleryTaskPayload(BaseModel):
    """Base for all typed Celery task payloads."""

    model_config = {"extra": "forbid", "frozen": True}


class NoKwargsPayload(CeleryTaskPayload):
    """Payload for a task that takes no keyword arguments."""


class AgentMemoryConsolidationPayload(CeleryTaskPayload):
    """Tenant ids to consolidate; empty means the scheduler's default sweep."""

    tenant_ids: list[str] = Field(default_factory=list)


class VerificationEmailPayload(CeleryTaskPayload):
    """Typed payload for the email-verification delivery task."""

    user_id: str
    email: str
    token: str
    idempotency_key: str


class PasswordResetEmailPayload(CeleryTaskPayload):
    """Typed payload for the password-reset delivery task."""

    user_id: str
    email: str
    token: str
    idempotency_key: str


class DocumentIngestPayload(CeleryTaskPayload):
    """Typed payload for the unified document ingestion task."""

    document_id: str
    user_id: str
    filename: str
    content_type: str
    object_uri: str


class CrawlerJobPayload(CeleryTaskPayload):
    """Validated wire payload for a durable crawler task."""

    crawl_id: str
    request_payload: dict[str, object]
