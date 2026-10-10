"""Producer-safe dispatch validation; no consumer imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from celery.exceptions import CeleryError
from pydantic import ValidationError

from app.utils import logger

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from app.shared.jobs.payloads import CeleryTaskPayload


from app.shared.jobs.names import (
    AGENT_MEMORY_CONSOLIDATION,
    BILLING_DUNNING,
    BILLING_INVOICE_GENERATION,
    BILLING_PAUSE_RESUME,
    BILLING_RECEIPT_GENERATION,
    BILLING_RECONCILIATION,
    BILLING_RENEWAL,
    CRAWLER_CRAWL,
    CREDITS_EXPIRE,
    CREDITS_RECONCILE,
    DOCUMENTS_INGEST,
    SEND_PASSWORD_RESET_EMAIL,
    SEND_VERIFICATION_EMAIL,
)
from app.shared.jobs.payloads import (
    AgentMemoryConsolidationPayload,
    CrawlerJobPayload,
    DocumentIngestPayload,
    NoKwargsPayload,
    PasswordResetEmailPayload,
    VerificationEmailPayload,
)


class TaskDispatchError(CeleryError):
    """Base for the refusals the typed registry raises before a send."""


class UnregisteredTaskError(TaskDispatchError):
    """A dispatch named a task that has no registered payload model."""

    def __init__(self, task_name: str, *, known_names: frozenset[str]) -> None:
        self.task_name = task_name
        self.known_names = known_names
        message = (
            f"Celery task {task_name!r} has no registered payload model, so the dispatch was "
            f"refused rather than sent to a name no consumer may answer to. Register a "
            f"CeleryTaskPayload subclass for it in app.shared.jobs.payloads and the shared registry. "
            f"Registered names: {sorted(known_names)}"
        )
        super().__init__(message)


class TaskPayloadValidationError(TaskDispatchError):
    """A dispatched payload did not match the model its task declares."""

    def __init__(self, task_name: str, validation_error: ValidationError) -> None:
        self.task_name = task_name
        self.validation_error = validation_error
        message = (
            f"Payload for Celery task {task_name!r} does not match its registered model, so the "
            f"dispatch was refused rather than enqueued for a consumer that cannot accept it: "
            f"{validation_error}"
        )
        super().__init__(message)


class CeleryTaskRegistry:
    """Maps task names → Pydantic payload models for validation."""

    _registry: ClassVar[dict[str, type[CeleryTaskPayload]]] = {
        DOCUMENTS_INGEST: DocumentIngestPayload,
        CRAWLER_CRAWL: CrawlerJobPayload,
        SEND_VERIFICATION_EMAIL: VerificationEmailPayload,
        SEND_PASSWORD_RESET_EMAIL: PasswordResetEmailPayload,
        BILLING_RENEWAL: NoKwargsPayload,
        BILLING_DUNNING: NoKwargsPayload,
        BILLING_INVOICE_GENERATION: NoKwargsPayload,
        BILLING_RECEIPT_GENERATION: NoKwargsPayload,
        BILLING_PAUSE_RESUME: NoKwargsPayload,
        BILLING_RECONCILIATION: NoKwargsPayload,
        CREDITS_EXPIRE: NoKwargsPayload,
        CREDITS_RECONCILE: NoKwargsPayload,
        AGENT_MEMORY_CONSOLIDATION: AgentMemoryConsolidationPayload,
    }

    @classmethod
    def register(cls, task_name: str, payload_model: type[CeleryTaskPayload]) -> None:
        cls._registry[task_name] = payload_model

    @classmethod
    def get(cls, task_name: str) -> type[CeleryTaskPayload] | None:
        return cls._registry.get(task_name)

    @classmethod
    def registered_names(cls) -> frozenset[str]:
        return frozenset(cls._registry)

    @classmethod
    def typed_send(
        cls, task_name: str, kwargs: dict[str, object], **send_task_opts: object
    ) -> object:
        from app.connections.celery import celery_app  # noqa: PLC0415 — load only for publication

        cls.validate(task_name, kwargs)
        return celery_app.send_task(task_name, kwargs=kwargs, **send_task_opts)

    @classmethod
    def validate(cls, task_name: str, kwargs: dict[str, Any]) -> CeleryTaskPayload:
        model = cls._registry.get(task_name)
        if model is None:
            logger.bind(task=task_name, registered=sorted(cls._registry)).error(
                "Task name is not registered"
            )
            raise UnregisteredTaskError(task_name, known_names=cls.registered_names())
        try:
            return model.model_validate(kwargs)
        except ValidationError as exc:
            logger.bind(task=task_name, errors=exc.errors()).exception(
                "Task payload validation failed"
            )
            raise TaskPayloadValidationError(task_name, exc) from exc
