"""Example typed Celery task — send_verification_email.

Demonstrates the typed registry pattern.  Existing tasks are migrated
incrementally (one per PR) using this as the reference implementation.
"""

from __future__ import annotations

from app.config import get_settings
from app.connections import celery_app
from app.connections.celery import ResilientTask
from app.connections.celery_registry import CeleryTaskPayload, CeleryTaskRegistry
from app.shared.services.mailer import config_from_settings, send_template
from app.utils import logger

settings = get_settings()


class VerificationEmailPayload(CeleryTaskPayload):
    """Typed payload for auth.send_verification_email."""

    user_id: str
    email: str
    token: str
    idempotency_key: str


class PasswordResetEmailPayload(CeleryTaskPayload):
    """Typed payload for auth.send_password_reset_email."""

    user_id: str
    email: str
    token: str
    idempotency_key: str


# Register typed payloads
CeleryTaskRegistry.register("auth.send_verification_email", VerificationEmailPayload)
CeleryTaskRegistry.register("auth.send_password_reset_email", PasswordResetEmailPayload)


@celery_app.task(
    name="auth.send_verification_email",
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

        def send_email() -> dict[str, str]:
            url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
            send_template(
                config_from_settings(settings),
                to=email,
                template_id=settings.RESEND_VERIFICATION_TEMPLATE_ID,
                variables={"verification_url": url, "email": email},
            )
            logger.bind(user_id=user_id, email=email, url=url).info("Verification email dispatched")
            return {"status": "sent", "user_id": user_id}

        result = self.run_with_circuit_breaker("email-provider", send_email)
        self.mark_idempotency_completed(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        return result  # noqa: TRY300 — return in try is idiomatic for task patterns
    except ValueError:
        self.mark_idempotency_failed_permanently(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        raise
    except Exception:
        self.release_idempotency_processing_lock(idempotency_key)
        raise


@celery_app.task(
    name="auth.send_password_reset_email",
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

        def send_email() -> dict[str, str]:
            url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
            send_template(
                config_from_settings(settings),
                to=email,
                template_id=settings.RESEND_PASSWORD_RESET_TEMPLATE_ID,
                variables={"reset_url": url, "email": email},
            )
            logger.bind(user_id=user_id, email=email, url=url).info(
                "Password reset email dispatched"
            )
            return {"status": "sent", "user_id": user_id}

        result = self.run_with_circuit_breaker("email-provider", send_email)
        self.mark_idempotency_completed(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        return result  # noqa: TRY300 — return in try is idiomatic for task patterns
    except ValueError:
        self.mark_idempotency_failed_permanently(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        raise
    except Exception:
        self.release_idempotency_processing_lock(idempotency_key)
        raise
