"""Transactional email tasks (typed registry pattern).

The single live declaration of the two email task names. An earlier duplicate
module carrying the same handlers was removed (9.13); this file is the survivor,
so exactly one set of handlers is registered and no import order decides which
implementation production sends email through.

The payload models below are the declaration's own contract, restated where the
registry can see it: the two task bodies take four required keyword arguments, so
that is what a dispatch for these names must carry.
"""

from __future__ import annotations

from functools import partial

from returns.result import Failure

from app.config import get_settings
from app.connections.celery import (
    CeleryTaskPayload,
    CeleryTaskRegistry,
    CircuitBreakerOpenError,
    IdempotencyLockError,
    ResilientTask,
    celery_app,
)
from app.connections.celery_task_names import (
    SEND_PASSWORD_RESET_EMAIL,
    SEND_VERIFICATION_EMAIL,
)
from app.shared.services.mailer import config_from_settings, send_template
from app.utils import ExternalServiceException, logger

settings = get_settings()


def _send_verification_email(email: str, token: str) -> dict[str, str]:
    url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
    result = send_template(
        config_from_settings(settings),
        to=email,
        template_id=settings.RESEND_VERIFICATION_TEMPLATE_ID,
        variables={"verification_url": url, "email": email},
    )
    if isinstance(result, Failure):
        error = result.failure()
        raise ExternalServiceException(
            service="resend", detail=error.message, error_code=error.code.value
        )
    logger.bind(email=email, url=url).info("Verification email dispatched")
    return {"status": "sent", "email": email}


def _send_password_reset_email(email: str, token: str) -> dict[str, str]:
    url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
    result = send_template(
        config_from_settings(settings),
        to=email,
        template_id=settings.RESEND_PASSWORD_RESET_TEMPLATE_ID,
        variables={"reset_url": url, "email": email},
    )
    if isinstance(result, Failure):
        error = result.failure()
        raise ExternalServiceException(
            service="resend", detail=error.message, error_code=error.code.value
        )
    logger.bind(email=email, url=url).info("Password reset email dispatched")
    return {"status": "sent", "email": email}


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


# Register typed payloads
CeleryTaskRegistry.register(SEND_VERIFICATION_EMAIL, VerificationEmailPayload)
CeleryTaskRegistry.register(SEND_PASSWORD_RESET_EMAIL, PasswordResetEmailPayload)


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
    # Validate typed payload (catches bad kwargs at task start)
    VerificationEmailPayload(
        user_id=user_id, email=email, token=token, idempotency_key=idempotency_key
    )

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
    except CircuitBreakerOpenError as exc:
        # Transient downstream outage: release the lock so a retry can
        # re-acquire it; the retry itself keeps the failure visible.
        exc.add_note("task=send_verification_email")
        logger.bind(user_id=user_id, error=str(exc)).warning("email_delivery_breaker_open")
        self.release_idempotency_processing_lock(idempotency_key)
        raise
    except IdempotencyLockError as exc:
        # Lock contention: another worker owns this delivery — skip loudly.
        exc.add_note("task=send_verification_email")
        logger.bind(user_id=user_id, error=str(exc)).warning("email_delivery_duplicate")
        return {"status": "duplicate-skipped", "user_id": user_id}
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
    PasswordResetEmailPayload(
        user_id=user_id, email=email, token=token, idempotency_key=idempotency_key
    )

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
    except CircuitBreakerOpenError as exc:
        # Transient downstream outage: release the lock so a retry can
        # re-acquire it; the retry itself keeps the failure visible.
        exc.add_note("task=send_password_reset_email")
        logger.bind(user_id=user_id, error=str(exc)).warning("email_delivery_breaker_open")
        self.release_idempotency_processing_lock(idempotency_key)
        raise
    except IdempotencyLockError as exc:
        # Lock contention: another worker owns this delivery — skip loudly.
        exc.add_note("task=send_password_reset_email")
        logger.bind(user_id=user_id, error=str(exc)).warning("email_delivery_duplicate")
        return {"status": "duplicate-skipped", "user_id": user_id}
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
