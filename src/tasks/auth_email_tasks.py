from functools import partial

from app.config import get_settings
from app.connections import ResilientTask, celery_app
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
