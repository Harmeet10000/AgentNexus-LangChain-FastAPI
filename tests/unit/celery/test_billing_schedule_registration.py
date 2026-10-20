"""Billing lifecycle tasks are both declared and periodically scheduled."""

from app.shared.jobs.names import (
    BILLING_DUNNING,
    BILLING_INVOICE_GENERATION,
    BILLING_PAUSE_RESUME,
    BILLING_PLAN_RECOVERY,
    BILLING_RECEIPT_GENERATION,
    BILLING_RECONCILIATION,
    BILLING_RENEWAL,
)
from app.worker.schedules import BEAT_SCHEDULE
from app.worker.task_modules import TASK_DECLARING_MODULES


def test_lifecycle_sweeps_have_task_declarations_and_beat_entries() -> None:
    lifecycle_tasks = {
        BILLING_DUNNING,
        BILLING_INVOICE_GENERATION,
        BILLING_RENEWAL,
        BILLING_PAUSE_RESUME,
        BILLING_PLAN_RECOVERY,
        BILLING_RECEIPT_GENERATION,
        BILLING_RECONCILIATION,
    }

    assert lifecycle_tasks <= TASK_DECLARING_MODULES.keys()
    scheduled_tasks = {entry["task"] for entry in BEAT_SCHEDULE.values()}
    assert lifecycle_tasks <= scheduled_tasks


def test_reconciliation_schedule_is_retained_alongside_new_sweeps() -> None:
    names = {
        name
        for name, entry in BEAT_SCHEDULE.items()
        if entry["task"]
        in {
            BILLING_RENEWAL,
            BILLING_PAUSE_RESUME,
            BILLING_RECONCILIATION,
        }
    }

    assert names == {
        "billing-renewal-daily",
        "billing-pause-resume-daily",
        "billing-reconciliation-daily",
    }
