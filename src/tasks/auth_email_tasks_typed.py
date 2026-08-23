"""Reference implementation of the typed registry pattern for the two email tasks.

**This module declares the same two task names as the live email task module and
must not be added to the task application's ``include`` list.** Both declarations
bind the same names on the same application, so whichever module is imported last
wins, and listing both would let import order decide whether production sends
email through the live implementation or through this demonstration of one. The
payload models it introduced now live with the live declarations, which is where
the registry can see them; what is left here is the pattern, not a second
implementation, and it is a deletion candidate rather than something to wire up.
"""

from __future__ import annotations

from functools import partial

from app.config import get_settings
from app.connections.celery import ResilientTask, celery_app
from app.connections.celery_registry import CeleryTaskPayload, CeleryTaskRegistry
from app.connections.celery_task_names import (
    SEND_PASSWORD_RESET_EMAIL,
    SEND_VERIFICATION_EMAIL,
)
from app.shared.services.mailer import config_from_settings, send_template
from app.utils import logger

settings = get_settings()


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
