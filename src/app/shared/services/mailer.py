"""Email dispatch via Resend API — sync client for Celery worker context.

Uses httpx sync client because Celery workers run on a standard thread pool,
not an asyncio event loop. Wrapping async in asyncio.run() inside a worker
creates a new event loop per task invocation — expensive and error-prone.
Sync httpx is the right tool here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Success

from app.config import get_settings
from app.utils import logger

from .errors import MailerDeliveryError, MailerUnavailableError

if TYPE_CHECKING:
    from httpx._models import Response

    from .errors import MailerResult

settings = get_settings()


class MailerConfig(BaseModel):
    """Resend API credentials for dispatching transactional emails."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: str = Field(default=settings.RESEND_API_KEY.get_secret_value())
    from_email: str = Field(default=settings.RESEND_FROM_EMAIL)


def config_from_settings(settings_obj: object | None = None) -> MailerConfig:
    """Build a MailerConfig from the current application settings."""
    _ = settings_obj
    return MailerConfig()


def send_template(
    config: MailerConfig,
    *,
    to: str,
    template_id: str,
    variables: dict[str, str],
) -> MailerResult[None]:
    """Send a Resend hosted template email with typed delivery failures."""
    try:
        with httpx.Client(timeout=10) as client:
            resp: Response = client.post(
                settings.RESEND_SEND_URL,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": config.from_email,
                    "to": [to],
                    "template": template_id,
                    "variables": variables,
                },
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.bind(to=to, template_id=template_id).error(
            "email_send_failed",
            status_code=exc.response.status_code,
            error=str(exc),
        )
        return Failure(
            MailerDeliveryError(
                message="Email dispatch failed",
                details={"status_code": exc.response.status_code},
            )
        )
    except httpx.RequestError as exc:
        logger.bind(to=to, template_id=template_id).error(
            "email_send_request_failed",
            error=str(exc),
        )
        return Failure(MailerUnavailableError(message="Email service unreachable"))
    logger.bind(to=to, template_id=template_id).debug("Email dispatched via Resend")
    return Success(None)
