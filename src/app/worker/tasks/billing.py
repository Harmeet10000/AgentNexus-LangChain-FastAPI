"""Scheduled billing jobs: renewal, dunning, invoice/receipt, pause-resume, reconciliation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.features.billing.jobs import (
    dunning_job,
    invoice_backfill,
    pause_resume_job,
    plan_recovery_job,
    receipt_backfill,
    reconciliation_job,
    renewal_job,
    run_billing_job,
)
from app.features.billing.providers.factory import build_billing_provider
from app.shared.jobs.names import (
    BILLING_DUNNING,
    BILLING_INVOICE_GENERATION,
    BILLING_PAUSE_RESUME,
    BILLING_PLAN_RECOVERY,
    BILLING_RECEIPT_GENERATION,
    BILLING_RECONCILIATION,
    BILLING_RENEWAL,
)
from app.worker.app import celery_app
from app.worker.runtime import run
from app.worker.task import ResilientTask

if TYPE_CHECKING:
    from typing import Any


@celery_app.task(name=BILLING_RENEWAL, bind=True, base=ResilientTask)
def billing_renewal(_self: Any) -> dict[str, int]:
    provider = build_billing_provider()
    return run(
        run_billing_job(
            lambda session, session_factory: renewal_job(
                session, session_factory, provider=provider
            )
        )
    )


@celery_app.task(name=BILLING_DUNNING, bind=True, base=ResilientTask)
def billing_dunning(_self: Any) -> dict[str, int]:
    provider = build_billing_provider()
    return run(
        run_billing_job(
            lambda session, session_factory: dunning_job(
                session, session_factory, provider=provider
            )
        )
    )


@celery_app.task(name=BILLING_INVOICE_GENERATION, bind=True, base=ResilientTask)
def billing_invoice_generation(_self: Any) -> dict[str, int]:
    return run(run_billing_job(invoice_backfill))


@celery_app.task(name=BILLING_RECEIPT_GENERATION, bind=True, base=ResilientTask)
def billing_receipt_generation(_self: Any) -> dict[str, int]:
    return run(run_billing_job(receipt_backfill))


@celery_app.task(name=BILLING_PAUSE_RESUME, bind=True, base=ResilientTask)
def billing_pause_resume(_self: Any) -> dict[str, int]:
    provider = build_billing_provider()
    return run(
        run_billing_job(
            lambda session, session_factory: pause_resume_job(
                session, session_factory, provider=provider
            )
        )
    )


@celery_app.task(name=BILLING_RECONCILIATION, bind=True, base=ResilientTask)
def billing_reconciliation(_self: Any) -> dict[str, int]:
    provider = build_billing_provider()
    return run(
        run_billing_job(
            lambda session, session_factory: reconciliation_job(
                session, session_factory, provider=provider
            )
        )
    )


@celery_app.task(name=BILLING_PLAN_RECOVERY, bind=True, base=ResilientTask)
def billing_plan_recovery(_self: Any) -> dict[str, int]:
    provider = build_billing_provider()
    return run(
        run_billing_job(
            lambda session, session_factory: plan_recovery_job(
                session, session_factory, provider=provider
            )
        )
    )
