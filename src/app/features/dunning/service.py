"""Dunning: retry scheduling for past-due subscriptions with jitter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from secrets import randbelow
from typing import TYPE_CHECKING, cast

from returns.result import Failure
from sqlalchemy import select

from app.config import get_settings
from app.features.audit.model import AuditLog
from app.features.payments.clients.razorpay_client import RazorpayClient
from app.features.subscriptions.model import Subscription, SubscriptionStatus
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.features.audit.repository import AuditLogRepository
    from app.features.plans.repository import PlanRepository
    from app.features.subscriptions.model import Subscription as SubscriptionModel
    from app.features.subscriptions.repository import SubscriptionRepository


class DunningService:
    """Identify and execute payment retries for past-due subscriptions.

    The daily job calls ``find_due_for_retry`` then ``execute_retry`` for
    each result. Retry scheduling uses the configured delay ladder plus a
    cryptographically-random jitter so a bulk failure does not hammer
    Razorpay in lockstep (Requirement 47).
    """

    def __init__(
        self,
        session: AsyncSession,
        subscriptions: SubscriptionRepository,
        plans: PlanRepository,
        audit: AuditLogRepository,
        razorpay: RazorpayClient | None = None,
    ) -> None:
        self.session = session
        self.subscriptions = subscriptions
        self.plans = plans
        self.audit = audit
        self.razorpay = razorpay or RazorpayClient()

    @staticmethod
    def _retry_delay_ladder() -> list[int]:
        return list(get_settings().BILLING_DUNNING_RETRY_DAYS)

    @staticmethod
    def _jittered_delay(delay_days: int) -> timedelta:
        # CSPRNG jitter in [-0.5, +0.5] around the ladder delay (Requirement 30).
        jitter = 1.0 + (randbelow(1001) - 500) / 1000
        return timedelta(days=delay_days * jitter)

    def _next_retry_at(self, subscription: SubscriptionModel, *, now: datetime) -> datetime:
        ladder = self._retry_delay_ladder()
        delay_days = ladder[min(subscription.retry_count + 1, len(ladder) - 1)]
        return now + self._jittered_delay(delay_days)

    def _last_attempt_at(self, subscription: SubscriptionModel) -> datetime | None:
        raw = subscription.metadata_.get("dunning_attempts", [])
        attempts = raw if isinstance(raw, list) else []
        if not attempts:
            return None
        last = cast("dict[str, Any]", attempts[-1])
        return self._parse_dt(last.get("executed_at"))

    def is_due_for_retry(self, subscription: SubscriptionModel, *, now: datetime) -> bool:
        if subscription.status != SubscriptionStatus.PAST_DUE.value:
            return False
        if subscription.retry_count >= subscription.max_retries:
            return False
        ladder = self._retry_delay_ladder()
        delay_days = ladder[min(subscription.retry_count, len(ladder) - 1)]
        last_at = self._last_attempt_at(subscription) or subscription.updated_at or now
        return last_at + self._jittered_delay(delay_days) <= now

    async def find_due_for_retry(self, *, limit: int = 200) -> list[Subscription]:
        try:
            statement = (
                select(Subscription)
                .where(
                    Subscription.status == SubscriptionStatus.PAST_DUE.value,
                    Subscription.deleted_at.is_(None),
                )
                .limit(limit)
            )
            result = await self.session.execute(statement)
            candidates = list(result.scalars().all())
        except Exception as exc:  # noqa: BLE001 — orchestration boundary
            logger.bind(operation="dunning").error("find_due_for_retry failed", error=str(exc))
            return []

        now = datetime.now(tz=UTC)
        return [sub for sub in candidates if self.is_due_for_retry(sub, now=now)]

    async def _attempt_charge(self, subscription: Subscription) -> dict[str, Any]:
        """Issue a Razorpay payment link for the retry (R9.5/R40)."""
        if not get_settings().RAZORPAY_KEY_ID:
            return {"status": "skipped", "reason": "razorpay not configured"}
        try:
            plan_result = await self.plans.find_by_id(subscription.plan_id)
            if isinstance(plan_result, Failure):
                return {"status": "charge_failed", "reason": "plan lookup failed"}
            plan = plan_result.unwrap()
            if plan is None:
                return {"status": "charge_failed", "reason": "plan not found"}
            link = await self.razorpay.create_payment_link(
                amount=plan.amount,
                currency=plan.currency,
                description=f"Dunning retry for subscription {str(subscription.id)[:8]}",
                customer_id=subscription.razorpay_customer_id,
                notes={"subscription_id": str(subscription.id), "dunning": "retry"},
            )
            return {"status": "charge_attempted", "payment_link_id": link.get("id")}
        except Exception as exc:  # noqa: BLE001  -- recorded, not re-raised
            logger.bind(operation="dunning", subscription_id=str(subscription.id)).warning(
                f"Charge attempt failed: {exc}"
            )
            return {"status": "charge_failed", "reason": str(exc)[:500]}

    async def execute_retry(self, subscription: Subscription) -> Subscription:
        """Attempt a charge, record the attempt, and maybe halt (R9.5/R40)."""
        now = datetime.now(tz=UTC)
        raw_attempts = subscription.metadata_.get("dunning_attempts", [])
        attempts: list[dict[str, Any]] = (
            [cast("dict[str, Any]", a) for a in raw_attempts]
            if isinstance(raw_attempts, list)
            else []
        )
        charge: dict[str, Any] = await self._attempt_charge(subscription)

        next_retry_at = self._next_retry_at(subscription, now=now)
        attempts.append(
            {
                "attempt": subscription.retry_count + 1,
                "scheduled_at": now.isoformat(),
                "executed_at": now.isoformat(),
                "status": "executed",
                **charge,
                "next_retry_at": next_retry_at.isoformat(),
            }
        )
        values: dict[str, object] = {
            "metadata_": {**subscription.metadata_, "dunning_attempts": attempts}
        }

        if subscription.retry_count + 1 >= subscription.max_retries:
            update = await self.subscriptions.update_status(
                subscription,
                SubscriptionStatus.HALTED,
                expected_version=subscription.version,
                extra_values=values,
            )
        else:
            update = await self.subscriptions.update_with_lock(
                subscription,
                subscription.version,
                values={
                    **values,
                    "retry_count": subscription.retry_count + 1,
                },
            )
        if isinstance(update, Failure):
            logger.bind(operation="dunning", subscription_id=str(subscription.id)).error(
                update.failure().message
            )
            raise RuntimeError(update.failure().message)  # noqa: TRY004

        updated = update.unwrap()
        logger.bind(
            operation="dunning",
            subscription_id=str(updated.id),
            status=updated.status,
            retry_count=updated.retry_count,
            next_retry_at=next_retry_at.isoformat(),
        ).info("Dunning retry executed")
        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(updated.id),
                action="dunning.retry",
                changes={
                    "retry_count": updated.retry_count,
                    "status": updated.status,
                    "attempts": len(attempts),
                    "charge": charge,
                    "next_retry_at": next_retry_at.isoformat(),
                },
            )
        )
        return updated

    @staticmethod
    def _parse_dt(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None
