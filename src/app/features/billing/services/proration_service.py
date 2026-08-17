"""Proration calculations with exact decimal arithmetic."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING

from app.features.billing.dto import ProrationCalculation, ProrationDirection
from app.features.billing.exceptions import ProrationCalculationException
from app.features.billing.models import SubscriptionStatus
from app.features.billing.tax import split_tax_inclusive

if TYPE_CHECKING:
    from app.features.billing.models import Plan, Subscription


def calculate_proration_fraction(
    *, effective_date: datetime, period_start: datetime, period_end: datetime
) -> Decimal:
    """Fraction of the billing period remaining after ``effective_date``.

    Uses integer microsecond arithmetic (Requirement 33) so no precision is
    lost converting between datetimes and Decimal.
    """
    total_microseconds = int((period_end - period_start).total_seconds() * 1_000_000)
    if total_microseconds <= 0:
        msg = "Billing period is empty or inverted"
        raise ProrationCalculationException(
            msg,
            data={"period_start": period_start.isoformat(), "period_end": period_end.isoformat()},
        )
    remaining_microseconds = int((period_end - effective_date).total_seconds() * 1_000_000)
    if remaining_microseconds < 0:
        msg = "Effective date is after the current billing period end"
        raise ProrationCalculationException(
            msg,
            data={
                "effective_date": effective_date.isoformat(),
                "period_end": period_end.isoformat(),
            },
        )
    return Decimal(remaining_microseconds) / Decimal(total_microseconds)


def _round_paisa(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def calculate_plan_change_proration(
    subscription: Subscription,
    current_plan: Plan,
    new_plan: Plan,
    *,
    effective_date: datetime | None = None,
) -> ProrationCalculation:
    """Compute the prorated charge/credit for a mid-cycle plan change."""
    if current_plan.interval != new_plan.interval:
        msg = "Cannot change between different billing intervals"
        raise ProrationCalculationException(
            msg,
            data={"current_interval": current_plan.interval, "new_interval": new_plan.interval},
        )
    if subscription.status != SubscriptionStatus.ACTIVE:
        msg = "Proration is only valid for active subscriptions"
        raise ProrationCalculationException(
            msg,
            data={"subscription_id": str(subscription.id), "status": subscription.status},
        )
    if subscription.current_period_start is None or subscription.current_period_end is None:
        msg = "Subscription has no active billing period"
        raise ProrationCalculationException(
            msg,
            data={"subscription_id": str(subscription.id)},
        )

    now = effective_date or datetime.now(tz=UTC)
    fraction = calculate_proration_fraction(
        effective_date=now,
        period_start=subscription.current_period_start,
        period_end=subscription.current_period_end,
    )

    current_prorated = _round_paisa(Decimal(current_plan.amount) * fraction)
    new_prorated = _round_paisa(Decimal(new_plan.amount) * fraction)
    proration_amount = new_prorated - current_prorated

    if proration_amount > 0:
        direction = ProrationDirection.UPGRADE
        _, tax_amount = split_tax_inclusive(proration_amount, new_plan.tax_rate)
    elif proration_amount < 0:
        direction = ProrationDirection.DOWNGRADE
        _, tax_amount = split_tax_inclusive(abs(proration_amount), new_plan.tax_rate)
    else:
        direction = ProrationDirection.NO_CHANGE
        tax_amount = 0

    return ProrationCalculation(
        subscription_id=str(subscription.id),
        current_plan_id=str(current_plan.id),
        new_plan_id=str(new_plan.id),
        effective_date=now,
        remaining_fraction=fraction,
        current_plan_prorated=current_prorated,
        new_plan_prorated=new_prorated,
        proration_amount=proration_amount,
        tax_amount=tax_amount,
        total_amount=abs(proration_amount),
        direction=direction,
    )
