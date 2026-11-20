"""Import-safe composition descriptor for the auth slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.module import Module

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any


def build_consumers() -> Sequence[Any]:
    """Bind auth-owned Celery consumers only in a worker process."""
    from .adapters.inbound.worker.tasks import (  # noqa: PLC0415
        reconcile_pending_revocations,
        send_password_reset_email,
        send_verification_email,
    )

    return (
        send_verification_email,
        send_password_reset_email,
        reconcile_pending_revocations,
    )


def build_schedules() -> Mapping[str, Any]:
    """Build the auth-owned periodic schedule only in a scheduler process."""
    from celery.schedules import crontab  # noqa: PLC0415

    from app.shared.jobs.names import AUTH_PENDING_REVOCATION_RECONCILE  # noqa: PLC0415

    return {
        "auth-pending-revocation-reconcile": {
            "task": AUTH_PENDING_REVOCATION_RECONCILE,
            "schedule": crontab(minute="*/1"),
            "options": {"expires": 55},
        }
    }


MODULE = Module(
    name="auth",
    consumers=(build_consumers,),
    schedules=(build_schedules,),
)
