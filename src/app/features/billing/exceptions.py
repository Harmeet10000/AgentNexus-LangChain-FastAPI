"""Billing-specific typed exceptions."""

from __future__ import annotations

from app.utils import (
    ConflictException,
    ExternalServiceException,
    UnauthorizedException,
    ValidationException,
)


class InvalidStateTransitionException(ValidationException):
    """Subscription status transition violates the state machine rules."""

    def __init__(self, *, current: str, target: str) -> None:
        msg = f"Invalid subscription state transition: {current} -> {target}"
        super().__init__(
            detail=msg,
            data={"current": current, "target": target},
        )


class ProrationCalculationException(ValidationException):
    """Proration calculation failed validation or arithmetic constraints."""

    def __init__(self, detail: str, data: dict[str, object] | None = None) -> None:
        super().__init__(detail=detail, data=data)


class WebhookVerificationException(UnauthorizedException):
    """Webhook HMAC signature verification failed (HTTP 401)."""

    def __init__(self, detail: str = "Webhook signature verification failed") -> None:
        super().__init__(detail=detail)


class InvoiceGenerationException(ValidationException):
    """Invoice generation failed validation or tax-consistency checks."""

    def __init__(self, detail: str, data: dict[str, object] | None = None) -> None:
        super().__init__(detail=detail, data=data)


class RazorpayRetryableError(ExternalServiceException):
    """Transient Razorpay failure — safe to retry (503/504/429/timeout)."""

    def __init__(self, service: str, detail: str) -> None:
        super().__init__(service=service, detail=detail)


class VersionConflictException(ConflictException):
    """Optimistic-lock version mismatch during a subscription update."""

    def __init__(self, subscription_id: str) -> None:
        super().__init__(
            detail="Subscription was modified concurrently; refetch and retry",
            data={"subscription_id": subscription_id},
        )
