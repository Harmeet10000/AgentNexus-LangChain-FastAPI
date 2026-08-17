"""Payment feature typed exceptions."""

from __future__ import annotations

from app.utils import ExternalServiceException


class RazorpayRetryableError(ExternalServiceException):
    """Transient Razorpay failure — safe to retry (503/504/429/timeout)."""

    def __init__(self, service: str, detail: str) -> None:
        super().__init__(service=service, detail=detail)
