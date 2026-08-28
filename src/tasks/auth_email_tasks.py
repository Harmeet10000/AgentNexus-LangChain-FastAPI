"""Transactional email tasks.

The payload models below are the declaration's own contract, restated where the
registry can see it: the two task bodies take four required keyword arguments, so
that is what a dispatch for these names must carry. They were previously declared
only in a parallel reference-implementation module that nothing imports, which
left both names unregistered in every process that dispatches them — so the
dispatch-time check had nothing to check against.

Recording these makes a producer/consumer gap visible that was invisible before:
the auth service's outbox payload for both names carries three of the four
arguments and omits the idempotency key these bodies require positionally for
their lock. That gap belongs to the producer, not here, and stating the
consumer's real contract is what surfaces it — declaring the key optional to make
the dispatch pass would hide the defect again and hand the worker a lock key of
``None``.
"""

from functools import partial

from app.config import get_settings
from app.connections.celery import CeleryTaskPayload, CeleryTaskRegistry, ResilientTask, celery_app
from app.connections.celery_task_names import (
    SEND_PASSWORD_RESET_EMAIL,
    SEND_VERIFICATION_EMAIL,
)
from app.shared.services.mailer import config_from_settings, send_template
from app.utils import logger

settings = get_settings()


class VerificationEmailPayload(CeleryTaskPayload):
    """Typed payload for the email-verification delivery task."""

    user_id: str
    email: str
    token: str
    idempotency_key: str


class PasswordResetEmailPayload(CeleryTaskPayload):
    """Typed payload for the password-reset delivery task."""

    user_id: str
    email: str
    token: str
    idempotency_key: str


CeleryTaskRegistry.register(SEND_VERIFICATION_EMAIL, VerificationEmailPayload)
CeleryTaskRegistry.register(SEND_PASSWORD_RESET_EMAIL, PasswordResetEmailPayload)


def _send_verification_email(email: str, token: str) -> dict[str, str]:
    url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
    send_template(
        config_from_settings(settings),
        to=email,
        template_id=settings.RESEND_VERIFICATION_TEMPLATE_ID,
        variables={"verification_url": url, "email": email},
    )
    logger.bind(email=email, url=url).info("Verification email dispatched")
    return {"status": "sent", "email": email}


def _send_password_reset_email(email: str, token: str) -> dict[str, str]:
    url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
    send_template(
        config_from_settings(settings),
        to=email,
        template_id=settings.RESEND_PASSWORD_RESET_TEMPLATE_ID,
        variables={"reset_url": url, "email": email},
    )
    logger.bind(email=email, url=url).info("Password reset email dispatched")
    return {"status": "sent", "email": email}


@celery_app.task(
    name=SEND_VERIFICATION_EMAIL,
    bind=True,
    base=ResilientTask,
)
def send_verification_email(
    self: ResilientTask,
    *,
    user_id: str,
    email: str,
    token: str,
    idempotency_key: str,
) -> dict[str, str]:
    """Deliver email verification link. Wire your mailer of choice here."""
    if not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"user_id": user_id, "email": email},
    ):
        return {"status": "duplicate-skipped", "user_id": user_id}

    try:
        result = self.run_with_circuit_breaker(
            "email-provider",
            partial(_send_verification_email, email=email, token=token),
        )
        self.mark_idempotency_completed(idempotency_key, metadata={"user_id": user_id})
    except ValueError:
        self.mark_idempotency_failed_permanently(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        raise
    except Exception:
        self.release_idempotency_processing_lock(idempotency_key)
        raise
    else:
        return result


@celery_app.task(
    name=SEND_PASSWORD_RESET_EMAIL,
    bind=True,
    base=ResilientTask,
)
def send_password_reset_email(
    self: ResilientTask,
    *,
    user_id: str,
    email: str,
    token: str,
    idempotency_key: str,
) -> dict[str, str]:
    """Deliver password reset link. Wire your mailer of choice here."""
    if not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"user_id": user_id, "email": email},
    ):
        return {"status": "duplicate-skipped", "user_id": user_id}

    try:
        result = self.run_with_circuit_breaker(
            "email-provider",
            partial(_send_password_reset_email, email=email, token=token),
        )
        self.mark_idempotency_completed(idempotency_key, metadata={"user_id": user_id})
    except ValueError:
        self.mark_idempotency_failed_permanently(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        raise
    except Exception:
        self.release_idempotency_processing_lock(idempotency_key)
        raise
    else:
        return result
