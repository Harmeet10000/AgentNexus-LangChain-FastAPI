"""Explicit consumer registration; never imported by producers."""

from collections.abc import Mapping
from typing import Final

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

_AGENT_MEMORY_TASKS: Final = "app.worker.tasks.agent_saul"
_AUTH_EMAIL_TASKS: Final = "app.worker.tasks.auth"
_BILLING_TASKS: Final = "app.worker.tasks.billing"
_CREDIT_TASKS: Final = "app.worker.tasks.credits"
_DOCUMENT_TASKS: Final = "app.worker.tasks.documents"


TASK_DECLARING_MODULES: Final[Mapping[str, str]] = {
    AGENT_MEMORY_CONSOLIDATION: _AGENT_MEMORY_TASKS,
    DOCUMENTS_INGEST: _DOCUMENT_TASKS,
    CRAWLER_CRAWL: "app.worker.tasks.crawler",
    SEND_VERIFICATION_EMAIL: _AUTH_EMAIL_TASKS,
    SEND_PASSWORD_RESET_EMAIL: _AUTH_EMAIL_TASKS,
    BILLING_RENEWAL: _BILLING_TASKS,
    BILLING_DUNNING: _BILLING_TASKS,
    BILLING_INVOICE_GENERATION: _BILLING_TASKS,
    BILLING_RECEIPT_GENERATION: _BILLING_TASKS,
    BILLING_PAUSE_RESUME: _BILLING_TASKS,
    BILLING_RECONCILIATION: _BILLING_TASKS,
    CREDITS_EXPIRE: _CREDIT_TASKS,
    CREDITS_RECONCILE: _CREDIT_TASKS,
}


TASK_MODULES: Final = (
    "app.worker.tasks.agent_saul",
    "app.worker.tasks.auth",
    "app.worker.tasks.billing",
    "app.worker.tasks.credits",
    "app.worker.tasks.documents",
    "app.worker.tasks.crawler",
)
