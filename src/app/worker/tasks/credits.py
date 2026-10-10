"""Scheduled credit jobs: daily expiration, weekly reconciliation."""

from __future__ import annotations

from app.features.billing.credits.jobs import expire_credits_job, reconcile_credits_job
from app.observability.logging import logger
from app.shared.jobs.names import CREDITS_EXPIRE, CREDITS_RECONCILE
from app.worker.app import celery_app
from app.worker.runtime import run
from app.worker.task import ResilientTask


@celery_app.task(name=CREDITS_EXPIRE, base=ResilientTask)
def credits_expire() -> dict[str, int]:
    """Daily job to expire past-due credits (Requirement 51)."""
    logger.bind(operation="credits.expire").info("Starting credit expiration job")
    result = run(expire_credits_job())
    logger.bind(operation="credits.expire", expired=result["expired"]).info(
        "Credit expiration job completed"
    )
    return result


@celery_app.task(name=CREDITS_RECONCILE, base=ResilientTask)
def credits_reconcile() -> dict[str, int]:
    """Weekly job to verify credit ledger integrity (Requirement 53.1)."""
    logger.bind(operation="credits.reconcile").info("Starting credit reconciliation job")
    result = run(reconcile_credits_job())
    logger.bind(
        operation="credits.reconcile",
        checked=result["checked"],
        discrepancies=result["discrepancies"],
    ).info("Credit reconciliation job completed")
    return result
