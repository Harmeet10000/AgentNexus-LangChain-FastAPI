"""Webhook feature typed exceptions."""

from __future__ import annotations

from app.utils import UnauthorizedException


class WebhookVerificationException(UnauthorizedException):
    """Webhook HMAC signature verification failed (HTTP 401)."""

    def __init__(self, detail: str = "Webhook signature verification failed") -> None:
        super().__init__(detail=detail)
