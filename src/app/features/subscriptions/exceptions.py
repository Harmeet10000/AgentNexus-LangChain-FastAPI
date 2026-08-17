"""Subscription feature typed exceptions."""

from __future__ import annotations

from app.utils import ConflictException, ValidationException


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


class VersionConflictException(ConflictException):
    """Optimistic-lock version mismatch during a subscription update."""

    def __init__(self, subscription_id: str) -> None:
        super().__init__(
            detail="Subscription was modified concurrently; refetch and retry",
            data={"subscription_id": subscription_id},
        )
