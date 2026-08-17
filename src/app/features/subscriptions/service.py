"""Subscription lifecycle service: create, cancel, pause, resume, change plan."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from returns.result import Failure

from app.config import get_settings
from app.features.audit.model import AuditAction, AuditLog
from app.features.payments.clients.razorpay_client import RazorpayClient
from app.features.payments.dto import RefundRequestDTO
from app.features.payments.model import PaymentStatus
from app.shared.result import app_error_to_exception, log_expected_failure
from app.utils import NotFoundException, ValidationException, logger

from .dto import (
    SubscriptionListResponse,
    SubscriptionResponse,
)
from .exceptions import InvalidStateTransitionException
from .model import Subscription, SubscriptionStatus
from .proration import calculate_plan_change_proration
from .trial_extension import TrialExtension

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.features.audit.repository import AuditLogRepository
    from app.features.payments.repository import PaymentRepository
    from app.features.payments.service import PaymentService
    from app.features.plans.model import Plan
    from app.features.plans.repository import PlanRepository
    from app.features.subscriptions.repository import SubscriptionRepository
    from app.shared.result import AppError

    from .dto import (
        PlanChangeDTO,
        ProrationCalculation,
        SubscriptionCancelDTO,
        SubscriptionCreateDTO,
        SubscriptionPauseDTO,
    )


def _subscription_to_response(
    subscription: Subscription, plan: Plan | None = None, payment_url: str | None = None
) -> SubscriptionResponse:
    return SubscriptionResponse(
        id=str(subscription.id),
        user_id=subscription.user_id,
        plan_id=str(subscription.plan_id),
        plan=(
            {
                "id": str(plan.id),
                "name": plan.name,
                "amount": plan.amount,
                "currency": plan.currency,
                "interval": plan.interval,
            }
            if plan is not None
            else None
        ),
        razorpay_subscription_id=subscription.razorpay_subscription_id,
        status=subscription.status,
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end,
        trial_end=subscription.trial_end,
        cancel_at_period_end=subscription.cancel_at_period_end,
        pause_start=subscription.pause_start,
        pause_end=subscription.pause_end,
        retry_count=subscription.retry_count,
        currency_display=subscription.currency_display,
        version=subscription.version,
        payment_url=payment_url,
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
    )


def _repo_failure(error: AppError, operation: str) -> None:
    log_expected_failure(error, operation=operation)
    raise app_error_to_exception(error)


class SubscriptionService:
    """Manage the subscription lifecycle for a user."""

    def __init__(  # noqa: PLR0917
        self,
        session: AsyncSession,
        subscriptions: SubscriptionRepository,
        plans: PlanRepository,
        payments: PaymentRepository,
        audit: AuditLogRepository,
        payment_service: PaymentService,
        razorpay: RazorpayClient | None = None,
    ) -> None:
        self.session = session
        self.subscriptions = subscriptions
        self.plans = plans
        self.payments = payments
        self.audit = audit
        self.payment_service = payment_service
        self.razorpay = razorpay or RazorpayClient()

    async def _get_owned_subscription(
        self, user_id: str, subscription_id: str | UUID
    ) -> Subscription:
        result = await self.subscriptions.find_by_id(subscription_id)
        if isinstance(result, Failure):
            _repo_failure(result.failure(), "get_owned_subscription")
        subscription = result.unwrap()
        if subscription is None:
            msg = "Subscription"
            raise NotFoundException(msg, str(subscription_id))
        if subscription.user_id != user_id:
            msg = "Subscription does not belong to this user"
            raise ValidationException(msg)
        return subscription

    async def _load_plan(self, plan_id: str | UUID) -> Plan:
        result = await self.plans.find_by_id(plan_id)
        if isinstance(result, Failure):
            _repo_failure(result.failure(), "load_plan")
        plan = result.unwrap()
        if plan is None:
            msg = "Plan"
            raise NotFoundException(msg, str(plan_id))
        return plan

    async def _refund_on_cancel(self, subscription: Subscription, plan: Plan, user_id: str) -> int:
        """Apply the plan's refund policy on immediate cancellation (Requirement 41).

        Returns the refund amount in paisa actually issued, else 0.
        """
        if not self._razorpay_enabled() or plan.refund_policy == "NONE":
            return 0
        result = await self.payments.find_by_subscription(subscription.id, limit=1)
        if isinstance(result, Failure) or not result.unwrap():
            return 0
        payment = result.unwrap()[0]
        if payment.status != PaymentStatus.CAPTURED.value:
            return 0

        refund_paisa = payment.amount
        if plan.refund_policy == "PRO_RATA":
            now = datetime.now(tz=UTC)
            start = subscription.current_period_start
            end = subscription.current_period_end
            if start is None or end is None or end <= now or end <= start:
                return 0
            total = (end - start).total_seconds()
            used = max((now - start).total_seconds(), 0.0)
            refund_paisa = max(int(payment.amount * (1.0 - used / total)), 0)
        if refund_paisa <= 0:
            return 0

        try:
            await self.payment_service.refund(
                str(payment.id),
                RefundRequestDTO(amount=refund_paisa, reason="Immediate cancellation refund"),
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001 — refund is best-effort on cancel
            logger.bind(operation="cancel_subscription").warning(
                "Refund on cancel failed", error=str(exc)
            )
            return 0
        return refund_paisa

    async def create_subscription(
        self, user_id: str, dto: SubscriptionCreateDTO
    ) -> SubscriptionResponse:
        plan = await self._load_plan(dto.plan_id)
        if not plan.is_active:
            msg = "Plan is not active"
            raise ValidationException(msg)

        existing = await self.subscriptions.find_by_user_and_plan(user_id, plan.id)
        if isinstance(existing, Failure):
            _repo_failure(existing.failure(), "create_subscription")
        if existing.unwrap() is not None:
            msg = "An active subscription already exists for this plan"
            raise ValidationException(msg)

        trial_days = (
            dto.trial_period_days if dto.trial_period_days is not None else plan.trial_period_days
        )
        now = datetime.now(tz=UTC)
        subscription = Subscription(
            user_id=user_id,
            plan_id=plan.id,
            status=SubscriptionStatus.CREATED.value,
            trial_end=now + timedelta(days=trial_days) if trial_days > 0 else None,
            currency_display=plan.currency,
        )
        result = await self.subscriptions.create(subscription)
        if isinstance(result, Failure):
            _repo_failure(result.failure(), "create_subscription")
        created = result.unwrap()

        payment_url: str | None = None
        if self._razorpay_enabled():
            try:  # noqa: PLW0717
                customer_id = await self._find_or_create_customer(
                    dto.customer_email, dto.customer_phone
                )
                razorpay_sub = await self.razorpay.create_subscription(
                    plan_id=plan.razorpay_plan_id or "",
                    customer_id=customer_id,
                    total_count=0,
                    quantity=1,
                    customer_notify=dto.customer_notify,
                    notes={"subscription_id": str(created.id), "user_id": user_id},
                )
                payment_url = razorpay_sub.get("short_url")
                update = await self.subscriptions.update_with_lock(
                    created,
                    created.version,
                    values={
                        "razorpay_subscription_id": razorpay_sub.get("id"),
                        "razorpay_customer_id": customer_id,
                    },
                )
                if isinstance(update, Failure):
                    _repo_failure(update.failure(), "create_subscription")
                created = update.unwrap()
            except Exception as exc:  # noqa: BLE001 — Razorpay is optional in dev
                logger.bind(operation="create_subscription").warning(
                    "Razorpay subscription creation failed", error=str(exc)
                )

        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(created.id),
                action=AuditAction.SUBSCRIPTION_CREATED.value,
                user_id=user_id,
                changes={"plan_id": str(plan.id), "trial_end": str(created.trial_end)},
            )
        )
        return _subscription_to_response(created, plan=plan, payment_url=payment_url)

    async def list_subscriptions(
        self,
        user_id: str,
        *,
        status: SubscriptionStatus | None = None,
        plan_id: str | UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SubscriptionListResponse:
        result = await self.subscriptions.list_by_user(
            user_id, status=status, plan_id=plan_id, limit=limit, offset=offset
        )
        if isinstance(result, Failure):
            _repo_failure(result.failure(), "list_subscriptions")
        items, total = result.unwrap()
        plan_cache: dict[str, Plan] = {}
        responses: list[SubscriptionResponse] = []
        for subscription in items:
            plan = plan_cache.get(str(subscription.plan_id))
            if plan is None:
                try:
                    plan = await self._load_plan(subscription.plan_id)
                except ValidationException:
                    plan = None
                if plan is not None:
                    plan_cache[str(subscription.plan_id)] = plan
            responses.append(_subscription_to_response(subscription, plan=plan))
        return SubscriptionListResponse(items=responses, total=total, limit=limit, offset=offset)

    async def get_subscription(
        self, user_id: str, subscription_id: str
    ) -> SubscriptionResponse:
        result = await self.subscriptions.find_by_id(subscription_id)
        if isinstance(result, Failure):
            _repo_failure(result.failure(), "get_subscription")
        subscription = result.unwrap()
        if subscription is None:
            msg = "Subscription"
            raise NotFoundException(msg, subscription_id)
        if subscription.user_id != user_id:
            msg = "Subscription does not belong to this user"
            raise ValidationException(msg)
        plan = await self._load_plan(subscription.plan_id)
        return _subscription_to_response(subscription, plan=plan)

    async def cancel_subscription(
        self, user_id: str, subscription_id: str, dto: SubscriptionCancelDTO
    ) -> SubscriptionResponse:
        subscription = await self._get_owned_subscription(user_id, subscription_id)
        if subscription.status not in {
            SubscriptionStatus.ACTIVE.value,
            SubscriptionStatus.PAUSED.value,
            SubscriptionStatus.PAST_DUE.value,
            SubscriptionStatus.HALTED.value,
        }:
            raise InvalidStateTransitionException(
                current=subscription.status, target=SubscriptionStatus.CANCELLED.value
            )

        if dto.cancel_at_period_end:
            if subscription.cancel_at_period_end:
                msg = "Subscription is already scheduled to cancel"
                raise ValidationException(msg)
            update = await self.subscriptions.update_with_lock(
                subscription,
                subscription.version,
                values={"cancel_at_period_end": True},
            )
            if isinstance(update, Failure):
                _repo_failure(update.failure(), "cancel_subscription")
            subscription = update.unwrap()
        else:
            plan = await self._load_plan(subscription.plan_id)
            if self._razorpay_enabled() and subscription.razorpay_subscription_id:
                try:
                    await self.razorpay.cancel_subscription(
                        subscription.razorpay_subscription_id, cancel_at_cycle_end=False
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.bind(operation="cancel_subscription").warning(
                        "Razorpay cancel failed", error=str(exc)
                    )
            refund_paisa = await self._refund_on_cancel(subscription, plan, user_id)
            update = await self.subscriptions.update_status(
                subscription,
                SubscriptionStatus.CANCELLED,
                expected_version=subscription.version,
                extra_values={
                    "cancelled_at": datetime.now(tz=UTC),
                    "ended_at": datetime.now(tz=UTC),
                    "cancel_at_period_end": False,
                },
            )
            if isinstance(update, Failure):
                _repo_failure(update.failure(), "cancel_subscription")
            subscription = update.unwrap()
            if refund_paisa > 0:
                await self.audit.create(
                    AuditLog(
                        entity_type="subscription",
                        entity_id=str(subscription.id),
                        action="subscription.refunded_on_cancel",
                        user_id=user_id,
                        changes={
                            "refund_paisa": refund_paisa,
                            "policy": plan.refund_policy,
                        },
                    )
                )

        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(subscription.id),
                action=AuditAction.SUBSCRIPTION_CANCELLED.value,
                user_id=user_id,
                changes={"cancel_at_period_end": dto.cancel_at_period_end, "reason": dto.reason},
            )
        )
        plan = await self._load_plan(subscription.plan_id)
        return _subscription_to_response(subscription, plan=plan)

    async def pause_subscription(
        self, user_id: str, subscription_id: str, dto: SubscriptionPauseDTO
    ) -> SubscriptionResponse:
        subscription = await self._get_owned_subscription(user_id, subscription_id)
        if subscription.status != SubscriptionStatus.ACTIVE.value:
            raise InvalidStateTransitionException(
                current=subscription.status, target=SubscriptionStatus.PAUSED.value
            )

        now = datetime.now(tz=UTC)
        values: dict[str, object] = {
            "pause_start": now,
            "pause_end": now + timedelta(days=dto.pause_duration_days)
            if dto.pause_duration_days
            else None,
        }
        if self._razorpay_enabled() and subscription.razorpay_subscription_id:
            try:
                await self.razorpay.update_subscription(
                    subscription.razorpay_subscription_id,
                    values={"pause_at": int(now.timestamp())},
                )
            except Exception as exc:  # noqa: BLE001
                logger.bind(operation="pause_subscription").warning(
                    "Razorpay pause failed", error=str(exc)
                )
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.PAUSED,
            expected_version=subscription.version,
            extra_values=values,
        )
        if isinstance(update, Failure):
            _repo_failure(update.failure(), "pause_subscription")
        subscription = update.unwrap()

        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(subscription.id),
                action="subscription.paused",
                user_id=user_id,
                changes=values,
            )
        )
        plan = await self._load_plan(subscription.plan_id)
        return _subscription_to_response(subscription, plan=plan)

    async def resume_subscription(
        self, user_id: str, subscription_id: str
    ) -> SubscriptionResponse:
        subscription = await self._get_owned_subscription(user_id, subscription_id)
        if subscription.status != SubscriptionStatus.PAUSED.value:
            raise InvalidStateTransitionException(
                current=subscription.status, target=SubscriptionStatus.ACTIVE.value
            )

        if self._razorpay_enabled() and subscription.razorpay_subscription_id:
            try:
                await self.razorpay.update_subscription(
                    subscription.razorpay_subscription_id,
                    values={"resume_at": int(datetime.now(tz=UTC).timestamp())},
                )
            except Exception as exc:  # noqa: BLE001
                logger.bind(operation="resume_subscription").warning(
                    "Razorpay resume failed", error=str(exc)
                )
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={"pause_start": None, "pause_end": None},
        )
        if isinstance(update, Failure):
            _repo_failure(update.failure(), "resume_subscription")
        subscription = update.unwrap()

        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(subscription.id),
                action="subscription.resumed",
                user_id=user_id,
                changes={"status": SubscriptionStatus.ACTIVE.value},
            )
        )
        plan = await self._load_plan(subscription.plan_id)
        return _subscription_to_response(subscription, plan=plan)

    async def change_plan(
        self,
        user_id: str,
        subscription_id: str,
        dto: PlanChangeDTO,
    ) -> SubscriptionResponse:
        subscription = await self._get_owned_subscription(user_id, subscription_id)
        current_plan = await self._load_plan(subscription.plan_id)
        new_plan = await self._load_plan(dto.new_plan_id)
        if not new_plan.is_active:
            msg = "New plan is not active"
            raise ValidationException(msg)

        proration = calculate_plan_change_proration(
            subscription, current_plan, new_plan, effective_date=dto.effective_date
        )
        payment_url: str | None = None
        if proration.direction.value == "upgrade" and proration.proration_amount > 0:
            if self._razorpay_enabled():
                try:
                    link = await self.razorpay.create_payment_link(
                        amount=proration.proration_amount,
                        currency=current_plan.currency,
                        description=f"Prorated upgrade to {new_plan.name}",
                        notes={"subscription_id": str(subscription.id)},
                    )
                    payment_url = link.get("short_url")
                except Exception as exc:  # noqa: BLE001
                    logger.bind(operation="change_plan").warning(
                        "Proration payment link creation failed", error=str(exc)
                    )
            else:
                msg = "Razorpay is required to charge the prorated upgrade amount"
                raise ValidationException(msg)

        update = await self.subscriptions.update_with_lock(
            subscription,
            subscription.version,
            values={"plan_id": new_plan.id},
        )
        if isinstance(update, Failure):
            _repo_failure(update.failure(), "change_plan")
        subscription = update.unwrap()

        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(subscription.id),
                action=AuditAction.PLAN_CHANGED.value,
                user_id=user_id,
                changes={
                    "old_plan_id": str(current_plan.id),
                    "new_plan_id": str(new_plan.id),
                    "proration_amount": proration.proration_amount,
                    "direction": proration.direction.value,
                },
            )
        )
        return _subscription_to_response(subscription, plan=new_plan, payment_url=payment_url)

    async def get_change_preview(
        self, user_id: str, subscription_id: str, new_plan_id: str
    ) -> ProrationCalculation:
        subscription = await self._get_owned_subscription(user_id, subscription_id)
        current_plan = await self._load_plan(subscription.plan_id)
        new_plan = await self._load_plan(new_plan_id)
        return calculate_plan_change_proration(subscription, current_plan, new_plan)

    async def request_trial_extension(
        self, user_id: str, subscription_id: str, *, days: int, reason: str | None = None
    ) -> dict[str, object]:
        subscription = await self._get_owned_subscription(user_id, subscription_id)
        if subscription.trial_end is None:
            msg = "Subscription is not in trial"
            raise ValidationException(msg)
        now = datetime.now(tz=UTC)
        extension = TrialExtension(
            subscription_id=subscription.id,
            requested_days=days,
            approved_days=0,
            status="pending",
            requested_by_user_id=user_id,
            requested_at=now,
            rejection_reason=reason,
            original_trial_end=subscription.trial_end,
        )
        self.session.add(extension)
        await self.session.flush()
        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(subscription.id),
                action=AuditAction.TRIAL_EXTENSION_REQUESTED.value,
                user_id=user_id,
                changes={"requested_days": days, "reason": reason},
            )
        )
        return {"status": "pending", "requested_days": days}

    @staticmethod
    def _razorpay_enabled() -> bool:
        return bool(get_settings().RAZORPAY_KEY_ID)

    async def _find_or_create_customer(self, email: str, phone: str | None) -> str:
        customer = await self.razorpay.find_customer_by_email(email)
        if customer is not None:
            return str(customer.get("id"))
        created = await self.razorpay.create_customer(
            email=email,
            contact=phone,
            name=email.split("@", maxsplit=1)[0],
        )
        return str(created.get("id"))
