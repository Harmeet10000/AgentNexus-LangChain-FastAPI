"""Scheduled billing jobs: renewal, dunning, invoice/receipt, pause-resume, reconciliation."""

from __future__ import annotations

from app.features.billing.jobs import (
    dunning_job,
    invoice_backfill,
    pause_resume_job,
    receipt_backfill,
    reconciliation_job,
    renewal_job,
    run_billing_job,
)
from app.shared.jobs.names import (
    BILLING_DUNNING,
    BILLING_INVOICE_GENERATION,
    BILLING_PAUSE_RESUME,
    BILLING_RECEIPT_GENERATION,
    BILLING_RECONCILIATION,
    BILLING_RENEWAL,
)
from app.worker.celery_app import celery_app
from app.worker.runtime import run
from app.worker.task import ResilientTask


@celery_app.task(name=BILLING_RENEWAL, base=ResilientTask)
def billing_renewal() -> dict[str, int]:
    return run(run_billing_job(renewal_job))


@celery_app.task(name=BILLING_DUNNING, base=ResilientTask)
def billing_dunning() -> dict[str, int]:
    return run(run_billing_job(dunning_job))


@celery_app.task(name=BILLING_INVOICE_GENERATION, base=ResilientTask)
def billing_invoice_generation() -> dict[str, int]:
    return run(run_billing_job(invoice_backfill))


@celery_app.task(name=BILLING_RECEIPT_GENERATION, base=ResilientTask)
def billing_receipt_generation() -> dict[str, int]:
    return run(run_billing_job(receipt_backfill))


@celery_app.task(name=BILLING_PAUSE_RESUME, base=ResilientTask)
def billing_pause_resume() -> dict[str, int]:
    return run(run_billing_job(pause_resume_job))


@celery_app.task(name=BILLING_RECONCILIATION, base=ResilientTask)
def billing_reconciliation() -> dict[str, int]:
    return run(run_billing_job(reconciliation_job))
