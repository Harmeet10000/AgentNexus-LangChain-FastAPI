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
from typing import TYPE_CHECKING, assert_never

from celery.exceptions import Reject
from returns.result import Failure

from app.config import get_settings
from app.connections.mailer import (
    MailerDeliveryError,
    MailerUnavailableError,
    config_from_settings,
    send_template,
)
from app.features.auth.application.revocation.pending import build_production_reconciler
from app.observability.logging import logger
from app.shared.jobs.names import (
    AUTH_PENDING_REVOCATION_RECONCILE,
    SEND_PASSWORD_RESET_EMAIL,
    SEND_VERIFICATION_EMAIL,
)
from app.shared.jobs.payloads import PasswordResetEmailPayload, VerificationEmailPayload
from app.utils import ExternalServiceException
from app.worker.celery.app import get_worker_celery_app
from app.worker.celery.task import ResilientTask
from app.worker.execution.reliability import CircuitBreakerOpenError, IdempotencyLockError
from app.worker.execution.runtime import run

celery_app = get_worker_celery_app()

if TYPE_CHECKING:
    from app.connections.mailer import MailerError

_EMAIL_RETRYABLE: tuple[type[BaseException], ...] = (
    *ResilientTask.autoretry_for,
    ExternalServiceException,
    CircuitBreakerOpenError,
)

# Resend statuses worth another attempt; every other 4xx is a rejection that
# retrying cannot fix. Unknown shapes retry: attempts are bounded and the
# lock is released on each one.
_TRANSIENT_EMAIL_STATUS_CODES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})


class PermanentEmailError(ValueError):
    """Resend rejected the message: retrying cannot succeed."""


def _email_failure_error(*, email: str, error: MailerError) -> Exception:
    """Map a typed mailer failure to a retryable or permanent task error."""
    if isinstance(error, MailerUnavailableError):
        return ExternalServiceException(
            service="resend", detail=error.message, error_code=error.code.value
        )
    if isinstance(error, MailerDeliveryError):
        status = (error.details or {}).get("status_code")
        if not isinstance(status, int) or status in _TRANSIENT_EMAIL_STATUS_CODES:
            return ExternalServiceException(
                service="resend", detail=error.message, error_code=error.code.value
            )
        return PermanentEmailError(f"resend rejected email to {email}: {error.message}")
    assert_never(error)


def _send_verification_email(settings, email: str, token: str) -> dict[str, str]:
    url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
    result = send_template(
        config_from_settings(settings),
        to=email,
        template_id=settings.RESEND_VERIFICATION_TEMPLATE_ID,
        variables={"verification_url": url, "email": email},
    )
    if isinstance(result, Failure):
        raise _email_failure_error(email=email, error=result.failure())
    logger.bind(email=email).info("Verification email dispatched")
    return {"status": "sent", "email": email}


def _send_password_reset_email(settings, email: str, token: str) -> dict[str, str]:
    url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
    result = send_template(
        config_from_settings(settings),
        to=email,
        template_id=settings.RESEND_PASSWORD_RESET_TEMPLATE_ID,
        variables={"reset_url": url, "email": email},
    )
    if isinstance(result, Failure):
        raise _email_failure_error(email=email, error=result.failure())
    logger.bind(email=email).info("Password reset email dispatched")
    return {"status": "sent", "email": email}


# Register typed payloads


@celery_app.task(
    name=SEND_VERIFICATION_EMAIL,
    bind=True,
    base=ResilientTask,
    autoretry_for=_EMAIL_RETRYABLE,
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
    settings = get_settings()

    if not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"user_id": user_id, "email": email},
    ):
        return {"status": "duplicate-skipped", "user_id": user_id}

    try:
        result = self.run_with_circuit_breaker(
            "email-provider",
            partial(_send_verification_email, settings, email=email, token=token),
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
    except ValueError as exc:
        exc.add_note("task=send_verification_email, stage=payload_validation")
        logger.bind(user_id=user_id, error=str(exc)).warning("email_delivery_invalid")
        self.mark_idempotency_failed_permanently(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        raise Reject(exc, requeue=False) from exc
    except Exception as exc:
        exc.add_note("task=send_verification_email")
        logger.bind(user_id=user_id, error=str(exc)).warning("email_delivery_failed")
        self.release_idempotency_processing_lock(idempotency_key)
        raise
    else:
        return result


@celery_app.task(
    name=SEND_PASSWORD_RESET_EMAIL,
    bind=True,
    base=ResilientTask,
    autoretry_for=_EMAIL_RETRYABLE,
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
    settings = get_settings()

    if not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"user_id": user_id, "email": email},
    ):
        return {"status": "duplicate-skipped", "user_id": user_id}

    try:
        result = self.run_with_circuit_breaker(
            "email-provider",
            partial(_send_password_reset_email, settings, email=email, token=token),
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
    except ValueError as exc:
        exc.add_note("task=send_password_reset_email, stage=payload_validation")
        logger.bind(user_id=user_id, error=str(exc)).warning("email_delivery_invalid")
        self.mark_idempotency_failed_permanently(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        raise Reject(exc, requeue=False) from exc
    except Exception as exc:
        exc.add_note("task=send_password_reset_email")
        logger.bind(user_id=user_id, error=str(exc)).warning("email_delivery_failed")
        self.release_idempotency_processing_lock(idempotency_key)
        raise
    else:
        return result


@celery_app.task(name=AUTH_PENDING_REVOCATION_RECONCILE, bind=True, base=ResilientTask)
def reconcile_pending_revocations(_self: ResilientTask) -> dict[str, object]:
    """Run one bounded pass; the next beat retries after outages or poison records."""
    settings = get_settings()
    if not settings.AUTH_PENDING_REVOCATION_ENABLED:
        logger.info("auth_pending_revocation_reconciliation_disabled")
        return {"status": "disabled"}

    async def _run() -> dict[str, object]:
        reconciler = await build_production_reconciler(_self.get_redis_client(), settings)
        return {"status": "ok", **await reconciler.run()}

    return run(_run())
