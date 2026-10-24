"""Immutable, producer-safe catalog of the Celery job wire contracts.

This module is deliberately boring at import time.  It imports only stable wire
names and Pydantic payload models; importing it must never discover a consumer,
construct a Celery application, or initialise a connection.  The catalog is the
single producer-side source of truth for the payload and routing metadata that
travels with each stable task name.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

from app.shared.jobs.names import (
    AGENT_MEMORY_CONSOLIDATION,
    AGENT_SAUL_TRANSCRIPT_SWEEP,
    AUTH_PENDING_REVOCATION_RECONCILE,
    BILLING_DUNNING,
    BILLING_INVOICE_GENERATION,
    BILLING_PAUSE_RESUME,
    BILLING_PLAN_RECOVERY,
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
    CeleryTaskPayload,
    CrawlerJobPayload,
    DocumentIngestPayload,
    NoKwargsPayload,
    PasswordResetEmailPayload,
    VerificationEmailPayload,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Final

    from app.shared.jobs.bus import TypedJob


class JobQueue(StrEnum):
    """Logical queue role; deployment settings resolve it to a broker queue."""

    DEFAULT = "default"
    INGESTION = "ingestion"
    CRAWLER = "crawler"


class JobCapability(StrEnum):
    """Capability owning a job, independent of its Celery task implementation."""

    AGENT_MEMORY = "agent_memory"
    BILLING = "billing"
    CRAWLER = "crawler"
    CREDITS = "credits"
    DOCUMENT_INGESTION = "document_ingestion"
    EMAIL = "email"
    TRANSCRIPT_SWEEP = "transcript_sweep"
    AUTH_SECURITY = "auth_security"


class JobDefinition[T: CeleryTaskPayload](BaseModel):
    """The immutable contract needed to validate and route one job."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    name: str
    payload_model: type[T]
    queue: JobQueue
    capability: JobCapability
    declaring_module: str

    @property
    def payload(self) -> type[T]:
        """Compatibility-friendly spelling for callers that say ``payload``."""

        return self.payload_model

    @property
    def task_name(self) -> str:
        """Compatibility-friendly spelling for the stable wire name."""

        return self.name

    @property
    def model(self) -> type[T]:
        """Compatibility-friendly spelling for the payload model."""

        return self.payload_model

    @property
    def queue_name(self) -> str:
        """The default deployment spelling of this logical queue role."""

        return self.queue.value

    @property
    def module(self) -> str:
        """Compatibility-friendly spelling for the declaring module."""

        return self.declaring_module

    @property
    def queue_role(self) -> JobQueue:
        """The logical queue role used by topology adapters."""

        return self.queue

    def prepare(self, payload: object) -> TypedJob[T]:
        """Create a validated job owned by this catalog definition.

        The local import keeps the catalog producer-safe: importing the catalog
        never imports an adapter, a broker, or a database client.  Callers should
        use this method instead of constructing a job from a raw wire name.
        """
        from app.shared.jobs.bus import TypedJob  # noqa: PLC0415

        # Specializing at runtime keeps the Pydantic field annotation precise
        # for introspection while the generic return type carries that same
        # precision to static callers.
        specialize = cast(
            "Callable[[object], object]",
            TypedJob.__class_getitem__,
        )
        job_type = cast("type[TypedJob[T]]", specialize(self.payload_model))
        return job_type.from_definition(self, payload)

    # ``make`` is intentionally a small spelling alias.  It keeps call sites
    # readable when the catalog entry is already the object in hand while
    # retaining one implementation of validation and construction.
    make = prepare


def _job[T: CeleryTaskPayload](
    name: str,
    payload_model: type[T],
    *,
    queue: JobQueue,
    capability: JobCapability,
    declaring_module: str,
) -> JobDefinition[T]:
    specialize = cast(
        "Callable[[object], object]",
        JobDefinition.__class_getitem__,
    )
    definition_type = cast("type[JobDefinition[T]]", specialize(payload_model))
    return definition_type(
        name=name,
        payload_model=payload_model,
        queue=queue,
        capability=capability,
        declaring_module=declaring_module,
    )


_CATALOG: dict[str, JobDefinition[CeleryTaskPayload]] = {
    AGENT_MEMORY_CONSOLIDATION: _job(
        AGENT_MEMORY_CONSOLIDATION,
        AgentMemoryConsolidationPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.AGENT_MEMORY,
        declaring_module="app.worker.jobs.agent_saul",
    ),
    AGENT_SAUL_TRANSCRIPT_SWEEP: _job(
        AGENT_SAUL_TRANSCRIPT_SWEEP,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.TRANSCRIPT_SWEEP,
        declaring_module="app.worker.jobs.agent_saul",
    ),
    DOCUMENTS_INGEST: _job(
        DOCUMENTS_INGEST,
        DocumentIngestPayload,
        queue=JobQueue.INGESTION,
        capability=JobCapability.DOCUMENT_INGESTION,
        declaring_module="app.worker.jobs.documents",
    ),
    CRAWLER_CRAWL: _job(
        CRAWLER_CRAWL,
        CrawlerJobPayload,
        queue=JobQueue.CRAWLER,
        capability=JobCapability.CRAWLER,
        declaring_module="app.worker.jobs.crawler",
    ),
    SEND_VERIFICATION_EMAIL: _job(
        SEND_VERIFICATION_EMAIL,
        VerificationEmailPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.EMAIL,
        declaring_module="app.worker.jobs.auth",
    ),
    SEND_PASSWORD_RESET_EMAIL: _job(
        SEND_PASSWORD_RESET_EMAIL,
        PasswordResetEmailPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.EMAIL,
        declaring_module="app.worker.jobs.auth",
    ),
    AUTH_PENDING_REVOCATION_RECONCILE: _job(
        AUTH_PENDING_REVOCATION_RECONCILE,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.AUTH_SECURITY,
        declaring_module="app.worker.jobs.auth",
    ),
    BILLING_RENEWAL: _job(
        BILLING_RENEWAL,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.BILLING,
        declaring_module="app.worker.jobs.billing",
    ),
    BILLING_DUNNING: _job(
        BILLING_DUNNING,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.BILLING,
        declaring_module="app.worker.jobs.billing",
    ),
    BILLING_INVOICE_GENERATION: _job(
        BILLING_INVOICE_GENERATION,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.BILLING,
        declaring_module="app.worker.jobs.billing",
    ),
    BILLING_RECEIPT_GENERATION: _job(
        BILLING_RECEIPT_GENERATION,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.BILLING,
        declaring_module="app.worker.jobs.billing",
    ),
    BILLING_PAUSE_RESUME: _job(
        BILLING_PAUSE_RESUME,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.BILLING,
        declaring_module="app.worker.jobs.billing",
    ),
    BILLING_RECONCILIATION: _job(
        BILLING_RECONCILIATION,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.BILLING,
        declaring_module="app.worker.jobs.billing",
    ),
    BILLING_PLAN_RECOVERY: _job(
        BILLING_PLAN_RECOVERY,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.BILLING,
        declaring_module="app.worker.jobs.billing",
    ),
    CREDITS_EXPIRE: _job(
        CREDITS_EXPIRE,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.CREDITS,
        declaring_module="app.worker.jobs.credits",
    ),
    CREDITS_RECONCILE: _job(
        CREDITS_RECONCILE,
        NoKwargsPayload,
        queue=JobQueue.DEFAULT,
        capability=JobCapability.CREDITS,
        declaring_module="app.worker.jobs.credits",
    ),
}

# MappingProxyType prevents accidental producer-side registration or mutation;
# the JobDefinition values are frozen Pydantic models as well.
JOB_CATALOG: Final[Mapping[str, JobDefinition[CeleryTaskPayload]]] = MappingProxyType(_CATALOG)
TASK_CATALOG: Final[Mapping[str, JobDefinition[CeleryTaskPayload]]] = JOB_CATALOG
JOB_DEFINITIONS: Final[Mapping[str, JobDefinition[CeleryTaskPayload]]] = JOB_CATALOG
JobCatalogEntry = JobDefinition[CeleryTaskPayload]
TaskCatalogEntry = JobDefinition[CeleryTaskPayload]


def get_job(task_name: str) -> JobDefinition[CeleryTaskPayload] | None:
    """Return the immutable definition for ``task_name``, if it is known."""

    return JOB_CATALOG.get(task_name)


def registered_names() -> frozenset[str]:
    """Return all known stable wire names as an immutable snapshot."""

    return frozenset(JOB_CATALOG)


get_task_definition = get_job
task_catalog = JOB_CATALOG


def payload_models() -> Mapping[str, type[CeleryTaskPayload]]:
    """Return the read-only name-to-payload view used by compatibility code."""

    return MappingProxyType(
        {name: definition.payload_model for name, definition in JOB_CATALOG.items()}
    )


__all__ = [
    "JOB_CATALOG",
    "JOB_DEFINITIONS",
    "TASK_CATALOG",
    "JobCapability",
    "JobCatalogEntry",
    "JobDefinition",
    "JobQueue",
    "TaskCatalogEntry",
    "get_job",
    "get_task_definition",
    "payload_models",
    "registered_names",
    "task_catalog",
]
