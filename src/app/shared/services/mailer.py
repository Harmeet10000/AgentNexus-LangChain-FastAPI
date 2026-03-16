from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.utils.logger import logger

_RESEND_SEND_URL = "https://api.resend.com/emails"


@dataclass(frozen=True)
class MailerService:
    """Sync Resend client — correct for Celery worker context.

    Uses httpx sync client because Celery workers run on a standard thread pool,
    not an asyncio event loop. Wrapping async in asyncio.run() inside a worker
    creates a new event loop per task invocation — expensive and error-prone.
    Sync httpx is the right tool here.
    """

    api_key: str
    from_email: str

    @classmethod
    def from_settings(cls) -> "MailerService":
        s = get_settings()
        return cls(api_key=s.RESEND_API_KEY, from_email=s.RESEND_FROM_EMAIL)

    def send_template(
        self,
        to: str,
        template_id: str,
        variables: dict[str, str],
    ) -> None:
        """Send a Resend hosted template email. Raises httpx.HTTPStatusError on failure."""
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                _RESEND_SEND_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": self.from_email,
                    "to": [to],
                    "template": template_id,
                    "variables": variables,
                },
            )
            resp.raise_for_status()
        logger.bind(to=to, template_id=template_id).debug("Email dispatched via Resend")
