"""Pure message schemas; importing them never loads a task implementation."""

from typing import Any

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
)


class CeleryTaskPayload(BaseModel):
    """Base for all typed Celery task payloads."""

    model_config = {"extra": "forbid", "frozen": True}


class NoKwargsPayload(CeleryTaskPayload):
    """Payload for a task that takes no keyword arguments."""


class AgentMemoryConsolidationPayload(CeleryTaskPayload):
    """Tenant ids to consolidate; empty means the scheduler's default sweep."""

    tenant_ids: list[str] = Field(default_factory=list)
    session_ids: list[str] | None = None


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
    tenant_id: str = ""
    document_version: int = Field(default=1, ge=1)

    @model_serializer(mode="wrap")
    def _serialize_compatibly(
        self, handler: SerializerFunctionWrapHandler, _info: SerializationInfo
    ) -> dict[str, Any]:
        """Keep legacy wire payloads stable when new fields were not supplied.

        The ingestion worker owns defaults for the tenant and document
        revision, so old outbox rows and producers may continue sending the
        original five-field shape.  Explicit values remain on the wire; this
        only omits defaults that were never part of the caller's payload.
        ``TypedJob.wire_payload`` performs the same field-set filtering, but
        registry callers also inspect ``model_dump`` directly.
        """
        data = handler(self)
        if "tenant_id" not in self.model_fields_set and not data.get("tenant_id"):
            data.pop("tenant_id", None)
        if "document_version" not in self.model_fields_set and data.get("document_version") == 1:
            data.pop("document_version", None)
        return data


class DocumentIngestTerminalPayload(CeleryTaskPayload):
    """Exactly-once terminal notification for a document ingestion."""

    document_id: str
    user_id: str
    tenant_id: str
    status: str
    document_version: int = 1
    # A Celery retry keeps the same task id but changes its retry count.  Both
    # belong in the terminal event identity so a new ingestion/retry cannot
    # accidentally suppress a distinct terminal notification.
    attempt: int = 0
    task_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class CrawlerJobPayload(CeleryTaskPayload):
    """Validated wire payload for a durable crawler task."""

    crawl_id: str
    request_payload: dict[str, object]
