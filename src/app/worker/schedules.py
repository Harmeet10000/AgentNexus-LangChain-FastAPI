"""Periodic work; all payloads follow the shared wire contracts."""

from __future__ import annotations

from celery.schedules import crontab

from app.config import get_settings
from app.shared.jobs.names import (
    AGENT_MEMORY_CONSOLIDATION,
    BILLING_DUNNING,
    BILLING_INVOICE_GENERATION,
    BILLING_RECEIPT_GENERATION,
    BILLING_RECONCILIATION,
    CREDITS_EXPIRE,
    CREDITS_RECONCILE,
)

settings = get_settings()
BEAT_SCHEDULE = {
    "billing-invoice-daily": {
        "task": BILLING_INVOICE_GENERATION,
        "schedule": crontab(hour=0, minute=15),
    },
    "billing-dunning-daily": {
        "task": BILLING_DUNNING,
        "schedule": crontab(hour=1, minute=0),
    },
    "billing-receipt-daily": {
        "task": BILLING_RECEIPT_GENERATION,
        "schedule": crontab(hour=1, minute=45),
    },
    "billing-reconciliation-daily": {
        "task": BILLING_RECONCILIATION,
        "schedule": crontab(hour=2, minute=0),
    },
    "credits-expire-daily": {
        "task": CREDITS_EXPIRE,
        "schedule": crontab(
            hour=settings.CREDIT_EXPIRATION_CRON_HOUR,
            minute=settings.CREDIT_EXPIRATION_CRON_MINUTE,
        ),
    },
    "credits-reconcile-weekly": {
        "task": CREDITS_RECONCILE,
        "schedule": crontab(
            hour=settings.CREDIT_RECONCILIATION_CRON_HOUR,
            minute=settings.CREDIT_RECONCILIATION_CRON_MINUTE,
            day_of_week=settings.CREDIT_RECONCILIATION_CRON_DAY_OF_WEEK,
        ),
    },
    "agent-memory-consolidation-nightly": {
        "task": AGENT_MEMORY_CONSOLIDATION,
        "schedule": crontab(hour=3, minute=30),
        "kwargs": {"tenant_ids": []},
    },
}
