from app.config import get_settings
from app.connections import ResilientTask, celery_app
from app.shared.services import MailerService
from app.utils import logger

settings = get_settings()


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

        def send_email() -> dict[str, str]:
            url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
            MailerService.from_settings().send_template(
                to=email,
                template_id=settings.RESEND_VERIFICATION_TEMPLATE_ID,
                variables={"verification_url": url, "email": email},
            )
            logger.bind(user_id=user_id, email=email, url=url).info(
                "Verification email dispatched"
            )
            return {"status": "sent", "user_id": user_id}

        result = self.run_with_circuit_breaker("email-provider", send_email)
        self.mark_idempotency_completed(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        return result
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
    if not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"user_id": user_id, "email": email},
    ):
        return {"status": "duplicate-skipped", "user_id": user_id}

    try:

        def send_email() -> dict[str, str]:
            url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
            MailerService.from_settings().send_template(
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
        return result
    except ValueError:
        self.mark_idempotency_failed_permanently(
            idempotency_key,
            metadata={"user_id": user_id},
        )
        raise
    except Exception:
        self.release_idempotency_processing_lock(idempotency_key)
        raise
