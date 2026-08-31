"""Subscription lifecycle service: create, cancel, pause, resume, change plan."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, assert_never

from returns.result import Failure, Success

from app.config import get_settings
from app.features.audit.model import AuditAction, AuditLog
from app.features.payments.clients.razorpay_client import RazorpayClient
from app.features.payments.dto import RefundRequestDTO
from app.features.payments.model import PaymentStatus
from app.shared.result.errors import (
    ConflictAppError,
    ErrorKind,
    ExternalServiceAppError,
    InfrastructureAppError,
    NotFoundAppError,
    ValidationAppError,
    http_status_for_kind,
)
from app.utils import logger

from .dto import (
    SubscriptionListResponse,
    SubscriptionResponse,
)
from .errors import (
    SubscriptionDuplicateError,
    SubscriptionInfrastructureError,
    SubscriptionInvalidTransitionError,
    SubscriptionNotFoundError,
    SubscriptionPlanNotFoundError,
    SubscriptionTransientInfrastructureError,
    SubscriptionValidationError,
    SubscriptionVersionConflictError,
)
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

    from .dto import (
        PlanChangeDTO,
        ProrationCalculation,
        SubscriptionCancelDTO,
        SubscriptionCreateDTO,
        SubscriptionPauseDTO,
    )
    from .errors import SubscriptionError, SubscriptionResult


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


def subscription_error_to_http_status(error: SubscriptionError) -> int:
    """Exhaustive dispatch over SubscriptionError — task 5.4.

    Adding a member to SubscriptionError without an arm fails ty with
    type-assertion-failure naming the missing type via assert_never.
    """
    match error:
        case SubscriptionNotFoundError():
            return http_status_for_kind(ErrorKind.NOT_FOUND)
        case SubscriptionDuplicateError():
            return http_status_for_kind(ErrorKind.CONFLICT)
        case SubscriptionVersionConflictError():
            return http_status_for_kind(ErrorKind.CONFLICT)
        case SubscriptionInvalidTransitionError():
            return http_status_for_kind(ErrorKind.VALIDATION)
        case SubscriptionPlanNotFoundError():
            return http_status_for_kind(ErrorKind.NOT_FOUND)
        case SubscriptionInfrastructureError():
            return http_status_for_kind(ErrorKind.INFRASTRUCTURE, retryable=error.retryable)
        case SubscriptionTransientInfrastructureError():
            return http_status_for_kind(ErrorKind.INFRASTRUCTURE, retryable=error.retryable)
        case SubscriptionValidationError():
            return http_status_for_kind(ErrorKind.VALIDATION)
        case _ as unreachable:
            assert_never(unreachable)


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
    ) -> SubscriptionResult[Subscription]:
        result = await self.subscriptions.find_by_id(subscription_id)
        if isinstance(result, Failure):
            return result
        subscription = result.unwrap()
        if subscription is None:
            return Failure(
                SubscriptionNotFoundError(
                    message="Subscription not found",
                    details={"subscription_id": str(subscription_id)},
                    source="subscription_service",
                    subscription_id=str(subscription_id),
                )
            )
        if subscription.user_id != user_id:
            return Failure(
                SubscriptionValidationError(
                    message="Subscription does not belong to this user",
                    details={"subscription_id": str(subscription_id), "user_id": user_id},
                    source="subscription_service",
                )
            )
        return Success(subscription)

    async def _load_plan(self, plan_id: str | UUID) -> SubscriptionResult[Plan]:
        result = await self.plans.find_by_id(plan_id)
        if isinstance(result, Failure):
            err = result.failure()
            # Preserve plan infrastructure/validation semantics — do not collapse to not-found
            if isinstance(err, InfrastructureAppError):
                error_type = (
                    SubscriptionTransientInfrastructureError
                    if err.retryable
                    else SubscriptionInfrastructureError
                )
                return Failure(
                    error_type(
                        message=err.message,
                        details=err.details or {"plan_id": str(plan_id)},
                        source="subscription_service",
                        operation="load_plan",
                    )
                )
            if isinstance(err, NotFoundAppError):
                return Failure(
                    SubscriptionPlanNotFoundError(
                        message=err.message,
                        details=err.details or {"plan_id": str(plan_id)},
                        source="subscription_service",
                        plan_id=str(plan_id),
                    )
                )
            if isinstance(err, ValidationAppError):
                return Failure(
                    SubscriptionValidationError(
                        message=err.message,
                        details=err.details or {"plan_id": str(plan_id)},
                        source="subscription_service",
                    )
                )
            if isinstance(err, ConflictAppError):
                return Failure(
                    SubscriptionValidationError(
                        message=err.message,
                        details=err.details or {"plan_id": str(plan_id)},
                        source="subscription_service",
                    )
                )
            if isinstance(err, ExternalServiceAppError):
                return Failure(
                    SubscriptionTransientInfrastructureError(
                        message=err.message,
                        details=err.details or {"plan_id": str(plan_id)},
                        source="subscription_service",
                        operation="load_plan",
                    )
                )
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message=getattr(err, "message", "Plan not found"),
                    details={"plan_id": str(plan_id), "error": str(err)},
                    source="subscription_service",
                    operation="load_plan",
                )
            )
        plan = result.unwrap()
        if plan is None:
            return Failure(
                SubscriptionPlanNotFoundError(
                    message="Plan not found",
                    details={"plan_id": str(plan_id)},
                    source="subscription_service",
                    plan_id=str(plan_id),
                )
            )
        return Success(plan)

    async def _refund_on_cancel(self, subscription: Subscription, plan: Plan, user_id: str) -> int:
        """Apply the plan's refund policy on immediate cancellation (Requirement 41)."""
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
    ) -> SubscriptionResult[SubscriptionResponse]:
        plan_result = await self._load_plan(dto.plan_id)
        if isinstance(plan_result, Failure):
            return plan_result
        plan = plan_result.unwrap()
        if not plan.is_active:
            return Failure(
                SubscriptionValidationError(
                    message="Plan is not active",
                    details={"plan_id": str(plan.id)},
                    source="subscription_service",
                )
            )

        existing = await self.subscriptions.find_by_user_and_plan(user_id, plan.id)
        if isinstance(existing, Failure):
            return existing
        if existing.unwrap() is not None:
            return Failure(
                SubscriptionValidationError(
                    message="An active subscription already exists for this plan",
                    details={"user_id": user_id, "plan_id": str(plan.id)},
                    source="subscription_service",
                )
            )

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
            return result
        created = result.unwrap()

        payment_url: str | None = None
        if self._razorpay_enabled():
            try:
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
                    return update
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
        return Success(_subscription_to_response(created, plan=plan, payment_url=payment_url))

    async def list_subscriptions(
        self,
        user_id: str,
        *,
        status: SubscriptionStatus | None = None,
        plan_id: str | UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SubscriptionResult[SubscriptionListResponse]:
        result = await self.subscriptions.list_by_user(
            user_id, status=status, plan_id=plan_id, limit=limit, offset=offset
        )
        if isinstance(result, Failure):
            return result
        items, total = result.unwrap()
        plan_cache: dict[str, Plan] = {}
        responses: list[SubscriptionResponse] = []
        for subscription in items:
            plan = plan_cache.get(str(subscription.plan_id))
            if plan is None:
                plan_result = await self._load_plan(subscription.plan_id)
                if isinstance(plan_result, Failure):
                    # If plan not found, skip caching but still return subscription without plan
                    plan = None
                else:
                    plan = plan_result.unwrap()
                    if plan is not None:
                        plan_cache[str(subscription.plan_id)] = plan
            responses.append(_subscription_to_response(subscription, plan=plan))
        return Success(SubscriptionListResponse(items=responses, total=total, limit=limit, offset=offset))

    async def get_subscription(self, user_id: str, subscription_id: str) -> SubscriptionResult[SubscriptionResponse]:
        result = await self.subscriptions.find_by_id(subscription_id)
        if isinstance(result, Failure):
            return result
        subscription = result.unwrap()
        if subscription is None:
            return Failure(
                SubscriptionNotFoundError(
                    message="Subscription not found",
                    details={"subscription_id": subscription_id},
                    source="subscription_service",
                    subscription_id=subscription_id,
                )
            )
        if subscription.user_id != user_id:
            return Failure(
                SubscriptionValidationError(
                    message="Subscription does not belong to this user",
                    details={"subscription_id": subscription_id, "user_id": user_id},
                    source="subscription_service",
                )
            )
        plan_result = await self._load_plan(subscription.plan_id)
        if isinstance(plan_result, Failure):
            return plan_result
        plan = plan_result.unwrap()
        return Success(_subscription_to_response(subscription, plan=plan))

    async def cancel_subscription(
        self, user_id: str, subscription_id: str, dto: SubscriptionCancelDTO
    ) -> SubscriptionResult[SubscriptionResponse]:
        owned = await self._get_owned_subscription(user_id, subscription_id)
        if isinstance(owned, Failure):
            return owned
        subscription = owned.unwrap()
        if subscription.status not in {
            SubscriptionStatus.ACTIVE.value,
            SubscriptionStatus.PAUSED.value,
            SubscriptionStatus.PAST_DUE.value,
            SubscriptionStatus.HALTED.value,
        }:
            return Failure(
                SubscriptionInvalidTransitionError(
                    message=f"Invalid state transition: {subscription.status} -> {SubscriptionStatus.CANCELLED.value}",
                    details={"current": subscription.status, "target": SubscriptionStatus.CANCELLED.value},
                    source="subscription_service",
                    subscription_id=str(subscription.id),
                    current=subscription.status,
                    target=SubscriptionStatus.CANCELLED.value,
                )
            )

        if dto.cancel_at_period_end:
            if subscription.cancel_at_period_end:
                return Failure(
                    SubscriptionValidationError(
                        message="Subscription is already scheduled to cancel",
                        details={"subscription_id": str(subscription.id)},
                        source="subscription_service",
                    )
                )
            update = await self.subscriptions.update_with_lock(
                subscription,
                subscription.version,
                values={"cancel_at_period_end": True},
            )
            if isinstance(update, Failure):
                return update
            subscription = update.unwrap()
        else:
            plan_result = await self._load_plan(subscription.plan_id)
            if isinstance(plan_result, Failure):
                return plan_result
            plan = plan_result.unwrap()
            if self._razorpay_enabled() and subscription.razorpay_subscription_id:
                try:
                    await self.razorpay.cancel_subscription(
                        subscription.razorpay_subscription_id, cancel_at_cycle_end=False
                    )
                except Exception as exc:  # noqa: BLE001 -- subscription op best-effort
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
                return update
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
        plan_result = await self._load_plan(subscription.plan_id)
        if isinstance(plan_result, Failure):
            # Return with subscription but without plan details if plan load fails
            return Success(_subscription_to_response(subscription, plan=None))
        plan = plan_result.unwrap()
        return Success(_subscription_to_response(subscription, plan=plan))

    async def pause_subscription(
        self, user_id: str, subscription_id: str, dto: SubscriptionPauseDTO
    ) -> SubscriptionResult[SubscriptionResponse]:
        owned = await self._get_owned_subscription(user_id, subscription_id)
        if isinstance(owned, Failure):
            return owned
        subscription = owned.unwrap()
        if subscription.status != SubscriptionStatus.ACTIVE.value:
            return Failure(
                SubscriptionInvalidTransitionError(
                    message=f"Invalid state transition: {subscription.status} -> {SubscriptionStatus.PAUSED.value}",
                    details={"current": subscription.status, "target": SubscriptionStatus.PAUSED.value},
                    source="subscription_service",
                    subscription_id=str(subscription.id),
                    current=subscription.status,
                    target=SubscriptionStatus.PAUSED.value,
                )
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
            except Exception as exc:  # noqa: BLE001 -- subscription op best-effort
                logger.bind(operation="pause_subscription").warning(
                    "Razorpay pause failed", error=str(exc)
                )
        update = await self.subscriptions.update_status(
            subscription,
            new_status=SubscriptionStatus.PAUSED,
            expected_version=subscription.version,
            extra_values=values,
        )
        if isinstance(update, Failure):
            return update
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
        plan_result = await self._load_plan(subscription.plan_id)
        if isinstance(plan_result, Failure):
            return Success(_subscription_to_response(subscription, plan=None))
        plan = plan_result.unwrap()
        return Success(_subscription_to_response(subscription, plan=plan))

    async def resume_subscription(self, user_id: str, subscription_id: str) -> SubscriptionResult[SubscriptionResponse]:
        owned = await self._get_owned_subscription(user_id, subscription_id)
        if isinstance(owned, Failure):
            return owned
        subscription = owned.unwrap()
        if subscription.status != SubscriptionStatus.PAUSED.value:
            return Failure(
                SubscriptionInvalidTransitionError(
                    message=f"Invalid state transition: {subscription.status} -> {SubscriptionStatus.ACTIVE.value}",
                    details={"current": subscription.status, "target": SubscriptionStatus.ACTIVE.value},
                    source="subscription_service",
                    subscription_id=str(subscription.id),
                    current=subscription.status,
                    target=SubscriptionStatus.ACTIVE.value,
                )
            )

        if self._razorpay_enabled() and subscription.razorpay_subscription_id:
            try:
                await self.razorpay.update_subscription(
                    subscription.razorpay_subscription_id,
                    values={"resume_at": int(datetime.now(tz=UTC).timestamp())},
                )
            except Exception as exc:  # noqa: BLE001 -- subscription op best-effort
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
            return update
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
        plan_result = await self._load_plan(subscription.plan_id)
        if isinstance(plan_result, Failure):
            return Success(_subscription_to_response(subscription, plan=None))
        plan = plan_result.unwrap()
        return Success(_subscription_to_response(subscription, plan=plan))

    async def change_plan(
        self,
        user_id: str,
        subscription_id: str,
        dto: PlanChangeDTO,
    ) -> SubscriptionResult[SubscriptionResponse]:
        owned = await self._get_owned_subscription(user_id, subscription_id)
        if isinstance(owned, Failure):
            return owned
        subscription = owned.unwrap()
        current_plan_result = await self._load_plan(subscription.plan_id)
        if isinstance(current_plan_result, Failure):
            return current_plan_result
        current_plan = current_plan_result.unwrap()
        new_plan_result = await self._load_plan(dto.new_plan_id)
        if isinstance(new_plan_result, Failure):
            return new_plan_result
        new_plan = new_plan_result.unwrap()
        if not new_plan.is_active:
            return Failure(
                SubscriptionValidationError(
                    message="New plan is not active",
                    details={"new_plan_id": str(new_plan.id)},
                    source="subscription_service",
                )
            )

        proration: ProrationCalculation = calculate_plan_change_proration(
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
                except Exception as exc:  # noqa: BLE001 -- subscription op best-effort
                    logger.bind(operation="change_plan").warning(
                        "Proration payment link creation failed", error=str(exc)
                    )
            else:
                return Failure(
                    SubscriptionValidationError(
                        message="Razorpay is required to charge the prorated upgrade amount",
                        details={"proration_amount": proration.proration_amount},
                        source="subscription_service",
                    )
                )

        update = await self.subscriptions.update_with_lock(
            subscription,
            subscription.version,
            values={"plan_id": new_plan.id},
        )
        if isinstance(update, Failure):
            return update
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
        return Success(_subscription_to_response(subscription, plan=new_plan, payment_url=payment_url))

    async def get_change_preview(
        self, user_id: str, subscription_id: str, new_plan_id: str
    ) -> SubscriptionResult[ProrationCalculation]:
        owned = await self._get_owned_subscription(user_id, subscription_id)
        if isinstance(owned, Failure):
            return owned
        subscription = owned.unwrap()
        current_plan_result = await self._load_plan(subscription.plan_id)
        if isinstance(current_plan_result, Failure):
            return current_plan_result
        current_plan = current_plan_result.unwrap()
        new_plan_result = await self._load_plan(new_plan_id)
        if isinstance(new_plan_result, Failure):
            return new_plan_result
        new_plan = new_plan_result.unwrap()
        return Success(calculate_plan_change_proration(subscription, current_plan, new_plan))

    async def request_trial_extension(
        self, user_id: str, subscription_id: str, *, days: int, reason: str | None = None
    ) -> SubscriptionResult[dict[str, object]]:
        owned = await self._get_owned_subscription(user_id, subscription_id)
        if isinstance(owned, Failure):
            return owned
        subscription = owned.unwrap()
        if subscription.trial_end is None:
            return Failure(
                SubscriptionValidationError(
                    message="Subscription is not in trial",
                    details={"subscription_id": str(subscription.id)},
                    source="subscription_service",
                )
            )
        now: datetime = datetime.now(tz=UTC)
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
        try:
            self.session.add(extension)
            await self.session.flush()
        except Exception as exc:  # noqa: BLE001 — trial extension flush may fail
            await self.session.rollback()
            return Failure(
                SubscriptionTransientInfrastructureError(
                    message="Database error while creating trial extension",
                    details={"subscription_id": str(subscription.id), "error": str(exc)},
                    source="subscription_service",
                    operation="request_trial_extension",
                )
            )
        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(subscription.id),
                action=AuditAction.TRIAL_EXTENSION_REQUESTED.value,
                user_id=user_id,
                changes={"requested_days": days, "reason": reason},
            )
        )
        return Success({"status": "pending", "requested_days": days})

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
