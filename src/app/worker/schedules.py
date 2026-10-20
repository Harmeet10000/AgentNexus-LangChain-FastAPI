"""Periodic work; all payloads follow the shared wire contracts."""

from __future__ import annotations

from celery.schedules import crontab

from app.config import get_settings
from app.shared.jobs.names import (
    AGENT_MEMORY_CONSOLIDATION,
    AGENT_SAUL_TRANSCRIPT_SWEEP,
    BILLING_DUNNING,
    BILLING_INVOICE_GENERATION,
    BILLING_PAUSE_RESUME,
    BILLING_PLAN_RECOVERY,
    BILLING_RECEIPT_GENERATION,
    BILLING_RECONCILIATION,
    BILLING_RENEWAL,
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
    "billing-renewal-daily": {
        "task": BILLING_RENEWAL,
        "schedule": crontab(hour=2, minute=15),
    },
    "billing-pause-resume-daily": {
        "task": BILLING_PAUSE_RESUME,
        "schedule": crontab(hour=2, minute=30),
    },
    "billing-plan-recovery-daily": {
        "task": BILLING_PLAN_RECOVERY,
        "schedule": crontab(hour=2, minute=45),
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
    "agent-saul-transcript-sweep-periodic": {
        "task": AGENT_SAUL_TRANSCRIPT_SWEEP,
        # §6 trigger 3: bounded beat sweep (hints + stale non-terminal + GC);
        # [v3] a Redis outage skips derivation entirely on that beat.
        "schedule": crontab(minute="*/15"),
        "options": {"expires": 60},
    },
}
