"""Fixture: violations the no-raise-in-result-service rule must flag."""

from app.features.billing.subscriptions.errors import SubscriptionValidationError


async def change_plan(is_active: bool) -> object:
    if not is_active:
        raise SubscriptionValidationError(
            message="Proration is only valid for active subscriptions",
            source="fixture",
        )
    return None
