"""Subscription lifecycle service: create, cancel, pause, resume, change plan."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

from returns.result import Failure, Success

from app.features.audit.models import AuditAction, AuditLog
from app.features.billing.exceptions import (
    LifecycleCoordinatorVersionConflictError,
)
from app.features.billing.payments.models import PaymentStatus
from app.features.billing.providers import CreateRefundOperation, ProviderReceipt
from app.kernel.errors import ErrorKind
from app.observability.logging import logger, trace_layer

from .dto import (
    SubscriptionListResponse,
    SubscriptionResponse,
)
from .exceptions import (
    SubscriptionInfrastructureError,
    SubscriptionInvalidTransitionError,
    SubscriptionNotFoundError,
    SubscriptionPlanNotFoundError,
    SubscriptionRefundError,
    SubscriptionTransientInfrastructureError,
    SubscriptionValidationError,
    SubscriptionVersionConflictError,
)
from .lifecycle import (
    CancelSubscription,
    ChangeSubscriptionPlan,
    CommandMetadata,
    CreateSubscription,
    PauseSubscription,
    PolicyContext,
    RemoteSyncState,
    ResumeSubscription,
)
from .models import Subscription, SubscriptionStatus
from .proration import calculate_plan_change_proration
from .trial_extension import TrialExtension

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.features.audit.repository import AuditLogRepository
    from app.features.billing.credits.service import CreditService
    from app.features.billing.exceptions import LifecycleCoordinatorError
    from app.features.billing.payments.clients.razorpay_client import RazorpayClient
    from app.features.billing.payments.repository import PaymentRepository
    from app.features.billing.payments.service import PaymentService
    from app.features.billing.plans.models import Plan
    from app.features.billing.plans.repository import PlanRepository
    from app.features.billing.subscriptions.repository import SubscriptionRepository
    from app.kernel.errors import FeatureError

    from .coordinator import SubscriptionLifecycleCoordinator
    from .dto import (
        PlanChangeDTO,
        ProrationCalculation,
        SubscriptionCancelDTO,
        SubscriptionCreateDTO,
        SubscriptionPauseDTO,
    )
    from .exceptions import SubscriptionError, SubscriptionResult
    from .lifecycle import SubscriptionCommand


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
        remote_sync=subscription.remote_sync,
        observed_plan_id=(
            str(subscription.observed_plan_id)
            if subscription.observed_plan_id is not None
            else None
        ),
        entitlement_until=subscription.entitlement_until,
        cancellation_requested_at=subscription.cancellation_requested_at,
        payment_url=payment_url,
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
    )


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
        lifecycle_coordinator: SubscriptionLifecycleCoordinator,
        credit_service: CreditService | None = None,
        razorpay: RazorpayClient | None = None,
    ) -> None:
        self.session = session
        self.subscriptions = subscriptions
        self.plans = plans
        self.payments = payments
        self.audit = audit
        self.payment_service = payment_service
        # External clients/providers are lifespan-owned dependencies.  The
        # lifecycle coordinator is mandatory for all subscription mutations;
        # the legacy client is retained only for constructor compatibility.
        self.razorpay = razorpay
        self.lifecycle_coordinator = lifecycle_coordinator
        self.credit_service = credit_service

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
            return Failure(self._translate_plan_error(result.failure(), plan_id))
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

    async def _authoritative_source_invoice_id(
        self, subscription: Subscription
    ) -> SubscriptionResult[UUID | None]:
        """Resolve the invoice that economically funded this subscription.

        Downgrade credit notes must point at the original invoice; selecting a
        latest invoice is unsafe when renewals race with a plan change.  Prefer
        an explicit subscription binding, then the oldest captured payment
        carrying an invoice foreign key.
        """
        raw = (subscription.metadata_ or {}).get("source_invoice_id")
        if isinstance(raw, str):
            try:
                return Success(UUID(raw))
            except ValueError:
                return Failure(
                    SubscriptionValidationError(
                        message="Subscription source invoice binding is invalid",
                        details={"subscription_id": str(subscription.id), "manual_review": True},
                        source="subscription_service",
                    )
                )
        direct_lookup = getattr(self.payments, "find_oldest_captured_invoice_by_subscription", None)
        if callable(direct_lookup):
            invoice_result = await direct_lookup(subscription.id)
            if isinstance(invoice_result, Failure):
                return Failure(
                    SubscriptionInfrastructureError(
                        message=invoice_result.failure().message,
                        details=invoice_result.failure().details,
                        source="subscription_service",
                        operation="resolve_source_invoice",
                    )
                )
            return Success(invoice_result.unwrap())

        # Compatibility for lightweight pre-ledger doubles.  Production uses
        # the direct repository query above; this fallback still requests the
        # complete result and never uses a newest-page heuristic.
        payments = await self.payments.find_by_subscription(subscription.id, limit=None)  # type: ignore[arg-type]
        if isinstance(payments, Failure):
            return Failure(
                SubscriptionInfrastructureError(
                    message=payments.failure().message,
                    details=payments.failure().details,
                    source="subscription_service",
                    operation="resolve_source_invoice",
                )
            )
        captured = [
            payment
            for payment in payments.unwrap()
            if payment.status
            in {
                PaymentStatus.CAPTURED.value,
                PaymentStatus.PARTIALLY_REFUNDED.value,
                PaymentStatus.REFUNDED.value,
            }
            and payment.invoice_id is not None
        ]
        if not captured:
            return Success(None)
        captured.sort(
            key=lambda payment: (
                getattr(payment, "captured_at", None)
                or getattr(payment, "created_at", datetime.min.replace(tzinfo=UTC))
            )
        )
        source = captured[0].invoice_id
        return Success(source if isinstance(source, UUID) else None)

    @staticmethod
    def _translate_plan_error(error: FeatureError, plan_id: str | UUID) -> SubscriptionError:
        details = error.details or {"plan_id": str(plan_id)}
        if error.kind is ErrorKind.NOT_FOUND:
            return SubscriptionPlanNotFoundError(
                message=error.message,
                details=details,
                source="subscription_service",
                plan_id=str(plan_id),
            )
        if error.kind is ErrorKind.INFRASTRUCTURE:
            return SubscriptionInfrastructureError(
                message=error.message,
                details=details,
                source="subscription_service",
                operation="load_plan",
            )
        return SubscriptionValidationError(
            message=error.message,
            details=details,
            source="subscription_service",
        )

    async def _refund_on_cancel(  # noqa: PLR0914, PLR0915
        self,
        subscription: Subscription,
        plan: Plan,
        user_id: str,
        *,
        idempotency_key: str,
    ) -> SubscriptionResult[int]:
        """Apply the plan's refund policy on immediate cancellation (Requirement 41)."""
        # This request key belongs to the cancellation command.  The refund
        # operation key below is intentionally derived from the economic
        # subscription/payment identity instead.
        del idempotency_key
        if plan.refund_policy == "NONE":
            return Success(0)
        direct_lookup = getattr(self.payments, "find_refundable_captured_by_subscription", None)
        result = (
            await direct_lookup(subscription.id)
            if callable(direct_lookup)
            else await self.payments.find_by_subscription(  # type: ignore[arg-type]
                subscription.id, limit=None
            )
        )
        if isinstance(result, Failure):
            error = result.failure()
            return Failure(
                SubscriptionRefundError(
                    message="Unable to resolve a refundable captured payment",
                    details={
                        "subscription_id": str(subscription.id),
                        "manual_review": True,
                        "operation": "resolve_refundable_payment",
                        "cause": error.message,
                        **(error.details or {}),
                    },
                    source="subscription_service",
                )
            )
        refundable_payments = [
            candidate
            for candidate in result.unwrap()
            if candidate.status
            in {
                PaymentStatus.CAPTURED.value,
                PaymentStatus.PARTIALLY_REFUNDED.value,
            }
            and candidate.amount > int((getattr(candidate, "refund_amount", 0) or 0) * 100)
        ]
        if not refundable_payments:
            return Success(0)

        def _recency_key(candidate: object) -> tuple[bool, datetime, datetime, str]:
            minimum = datetime.min.replace(tzinfo=UTC)

            def _timestamp(value: object) -> datetime:
                if not isinstance(value, datetime):
                    return minimum
                return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

            return (
                getattr(candidate, "captured_at", None) is not None,
                _timestamp(getattr(candidate, "captured_at", None)),
                _timestamp(getattr(candidate, "created_at", None)),
                str(getattr(candidate, "id", "")),
            )

        # Keep the service authoritative for lightweight repository doubles as
        # well as the production query: failed rows are excluded before the
        # latest/current captured payment is selected.
        payment = max(refundable_payments, key=_recency_key)

        previously_refunded = int((getattr(payment, "refund_amount", 0) or 0) * 100)
        refundable_paisa = max(payment.amount - previously_refunded, 0)
        refund_paisa = refundable_paisa
        if plan.refund_policy == "PRO_RATA":
            now = datetime.now(tz=UTC)
            start = subscription.current_period_start
            end = subscription.current_period_end
            if start is None or end is None or end <= now or end <= start:
                return Success(0)
            total = (end - start).total_seconds()
            used = max((now - start).total_seconds(), 0.0)
            refund_paisa = min(
                refundable_paisa,
                max(int(payment.amount * (1.0 - used / total)), 0),
            )
        if refund_paisa <= 0:
            return Success(0)

        coordinator = self.lifecycle_coordinator
        operation = CreateRefundOperation(
            payment_id=payment.razorpay_payment_id,
            amount=refund_paisa,
            correlation_id=f"subscription:{subscription.id}:refund:{payment.id}",
            notes={"subscription_id": str(subscription.id), "reason": "immediate_cancellation"},
        )
        # The HTTP request key identifies the cancellation command, not the
        # economic refund.  A retry can arrive with a different request key
        # (for example, a new correlation id after a worker crash), so using
        # it here would let two workers issue the same cash refund.  The
        # subscription/payment pair is the durable identity of this one
        # immediate-cancellation refund; the coordinator claims this key
        # before the provider mutation and reconciles a claimed retry.
        refund_operation_key = f"billing:cancel-refund:{subscription.id}:{payment.id}"
        try:
            refund_result = await coordinator.execute_provider_operation(
                operation,
                subscription_id=str(subscription.id),
                idempotency_key=refund_operation_key,
            )
        except Exception as exc:  # noqa: BLE001 — refund is best-effort on cancel
            exc.add_note(f"subscription_id={subscription.id}, operation=cancel_refund")
            logger.bind(operation="cancel_subscription", error=str(exc)).exception(
                "Refund on cancel failed"
            )
            return await self._refund_failure(
                subscription,
                refund_paisa=refund_paisa,
                status="unknown",
                message="Cancellation refund outcome is unknown; reconciliation is required",
                details={"exception_type": type(exc).__name__},
            )
        if isinstance(refund_result, Failure):
            error = refund_result.failure()
            logger.bind(operation="cancel_subscription", error=error.message).warning(
                "Refund on cancel returned failure"
            )
            provider_code = str((error.details or {}).get("provider_code", ""))
            refund_status = (
                "unknown"
                if provider_code.endswith("outcome_unknown")
                else "manual_review"
                if provider_code.endswith("manual_review_required")
                else "pending"
                if error.retryable
                else "failed"
            )
            return await self._refund_failure(
                subscription,
                refund_paisa=refund_paisa,
                status=refund_status,
                message=error.message,
                details=dict(error.details or {}),
            )
        receipt = refund_result.unwrap().value
        if not isinstance(receipt, ProviderReceipt):
            logger.bind(operation="cancel_subscription").warning(
                "Provider refund did not return a receipt"
            )
            return await self._refund_failure(
                subscription,
                refund_paisa=refund_paisa,
                status="manual_review",
                message="Provider refund returned no durable receipt",
                details={"manual_review": True},
            )
        if receipt.payment_id != payment.razorpay_payment_id or receipt.amount != refund_paisa:
            logger.bind(
                operation="cancel_subscription",
                expected_payment_id=payment.razorpay_payment_id,
                received_payment_id=receipt.payment_id,
                expected_amount=refund_paisa,
                received_amount=receipt.amount,
            ).warning("Provider refund receipt requires reconciliation")
            return Failure(
                SubscriptionRefundError(
                    message="Provider refund receipt does not match the selected payment",
                    details={
                        "manual_review": True,
                        "requires_reconciliation": True,
                        "expected_payment_id": payment.razorpay_payment_id,
                        "provider_payment_id": receipt.payment_id,
                        "expected_amount_paisa": refund_paisa,
                        "provider_amount_paisa": receipt.amount,
                        "received_amount_paisa": receipt.amount,
                        "provider_refund_id": receipt.provider_refund_id,
                    },
                    source="subscription_service",
                )
            )
        previous = int((getattr(payment, "refund_amount", 0) or 0) * 100)
        recorded_ids = {
            item
            for item in (payment.metadata_ or {}).get("processed_refund_ids", [])
            if isinstance(item, str)
        }
        if receipt.provider_refund_id in recorded_ids:
            logger.bind(operation="cancel_subscription", refund_id=receipt.provider_refund_id).info(
                "Immediate-cancellation refund already applied"
            )
            return Success(0)
        total = previous + receipt.amount
        payment_update = await self.payments.update_status(
            payment,
            status=(
                PaymentStatus.REFUNDED.value
                if total >= payment.amount
                else PaymentStatus.PARTIALLY_REFUNDED.value
            ),
            extra_values={
                "refund_amount": Decimal(total) / Decimal(100),
                "metadata_": {
                    **(payment.metadata_ or {}),
                    "processed_refund_ids": [*recorded_ids, receipt.provider_refund_id],
                },
            },
        )
        if isinstance(payment_update, Failure):
            logger.bind(
                operation="cancel_subscription", error=payment_update.failure().message
            ).warning("Refund provider receipt was durable but payment update failed")
            return await self._refund_failure(
                subscription,
                refund_paisa=receipt.amount,
                status="manual_review",
                message="Refund receipt is durable but local payment projection failed",
                details={"manual_review": True, "provider_refund_id": receipt.provider_refund_id},
            )
        await self.audit.create(
            AuditLog(
                entity_type="payment",
                entity_id=str(payment.id),
                action=AuditAction.REFUND_ISSUED.value,
                user_id=user_id,
                changes={
                    "razorpay_refund_id": receipt.provider_refund_id,
                    "amount": receipt.amount,
                },
            )
        )
        return Success(receipt.amount)

    async def _refund_failure(
        self,
        subscription: Subscription,
        *,
        refund_paisa: int,
        status: str,
        message: str,
        details: dict[str, object],
    ) -> SubscriptionResult[int]:
        """Persist unresolved refund sync state without undoing cancellation."""
        remote_sync = (
            RemoteSyncState.RETRYABLE_FAILURE
            if status == "pending"
            else RemoteSyncState.PENDING
            if status in {"unknown", "manual_review"}
            else RemoteSyncState.PERMANENT_FAILURE
        )
        update = getattr(self.subscriptions, "update_with_lock", None)
        if callable(update):
            updated = await update(
                subscription,
                subscription.version,
                values={"remote_sync": remote_sync.value},
            )
            if isinstance(updated, Success):
                subscription = updated.unwrap()
            else:
                logger.bind(
                    operation="cancel_subscription",
                    error=updated.failure().message,
                ).warning("Could not persist refund remote sync state")
        failure_details = {
            **details,
            "refund_status": status,
            "operation_status": status,
            "refundable_amount_paisa": refund_paisa,
            "cancelled": subscription.status == SubscriptionStatus.CANCELLED.value,
            "remote_sync": remote_sync.value,
            "manual_review": status in {"manual_review", "unknown"},
        }
        return Failure(
            SubscriptionRefundError(
                message=message,
                details=failure_details,
                source="subscription_service",
            )
        )

    async def _apply_lifecycle(
        self,
        subscription: Subscription,
        command: object,
        *,
        context: PolicyContext,
        idempotency_key: str,
    ) -> SubscriptionResult[Any]:
        """Apply one user command through the durable coordinator seam."""

        coordinator = self.lifecycle_coordinator
        result = await coordinator.apply(
            cast("SubscriptionCommand", command),
            idempotency_key,
            subscription_id=str(subscription.id),
            context=context,
        )
        if isinstance(result, Failure):
            error = result.failure()
            return Failure(self._translate_lifecycle_error(error))
        return Success(result.unwrap())

    @staticmethod
    def _request_idempotency_key(value: str | None, operation: str) -> str:
        """Use the HTTP key verbatim; only keyless legacy calls get a nonce."""
        if value:
            return value
        # Backward-compatible callers predate Idempotency-Key.  Their old
        # behavior was one fresh operation per invocation; HTTP retries should
        # supply the header to obtain durable replay semantics.
        return f"legacy:{operation}:{uuid4()}"

    @staticmethod
    def _translate_lifecycle_error(error: LifecycleCoordinatorError) -> SubscriptionError:
        details = dict(error.details or {})
        details["lifecycle_code"] = str(error.code)
        if isinstance(error, LifecycleCoordinatorVersionConflictError):
            return SubscriptionVersionConflictError(
                message=error.message,
                details=details,
                source="subscription_service",
                subscription_id=error.subscription_id,
                expected_version=error.expected_version,
            )
        if error.kind is ErrorKind.VALIDATION:
            return SubscriptionValidationError(
                message=error.message,
                details=details,
                source="subscription_service",
            )
        if getattr(error, "retryable", False):
            return SubscriptionTransientInfrastructureError(
                message=error.message,
                details=details,
                source="subscription_service",
                operation="lifecycle_apply",
            )
        return SubscriptionInfrastructureError(
            message=error.message,
            details=details,
            source="subscription_service",
            operation="lifecycle_apply",
        )

    @staticmethod
    def _provider_payment_url(outcome: object) -> str | None:
        for result in getattr(outcome, "provider_results", ()):
            value = getattr(result, "value", None)
            # ProviderPaymentLink deliberately mirrors Razorpay's ``short_url``
            # field; the domain response exposes it as paymentUrl.
            url = getattr(value, "short_url", None)
            if not isinstance(url, str) or not url:
                url = getattr(value, "payment_url", None)
            if isinstance(url, str) and url:
                return url
        return None

    @staticmethod
    def _ensure_plan_binding(subscription: Subscription, plan: Plan) -> None:
        metadata = dict(subscription.metadata_ or {})
        raw_bindings = metadata.get("billing_provider_plan_ids")
        bindings = (
            dict(cast("dict[str, str]", raw_bindings)) if isinstance(raw_bindings, dict) else {}
        )
        if not plan.razorpay_plan_id:
            return
        bindings[str(plan.id)] = plan.razorpay_plan_id
        metadata["billing_provider_plan_ids"] = bindings
        subscription.metadata_ = metadata

    @trace_layer("service")
    async def create_subscription(
        self, user_id: str, dto: SubscriptionCreateDTO, *, idempotency_key: str | None = None
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
        if not plan.razorpay_plan_id:
            return Failure(
                SubscriptionValidationError(
                    message="Plan is missing its provider billing binding",
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
            metadata_={
                "billing_customer_email": dto.customer_email,
                "billing_customer_phone": dto.customer_phone,
                "billing_customer_name": dto.customer_email.split("@", maxsplit=1)[0],
                "billing_provider_plan_ids": {str(plan.id): plan.razorpay_plan_id},
            },
        )
        result = await self.subscriptions.create(subscription)
        if isinstance(result, Failure):
            return result
        created = result.unwrap()

        payment_url: str | None = None
        operation_key = self._request_idempotency_key(idempotency_key, "create")
        applied = await self._apply_lifecycle(
            created,
            CreateSubscription(
                metadata=CommandMetadata(
                    request_id=operation_key, expected_version=created.version
                ),
                plan_id=str(plan.id),
            ),
            context=PolicyContext(now=now),
            idempotency_key=operation_key,
        )
        if isinstance(applied, Failure):
            return applied
        payment_url = self._provider_payment_url(applied.unwrap())
        refreshed = await self.subscriptions.find_by_id(created.id)
        if isinstance(refreshed, Failure):
            return refreshed
        if refreshed.unwrap() is not None:
            created = refreshed.unwrap()

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

    @trace_layer("service")
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
        return Success(
            SubscriptionListResponse(items=responses, total=total, limit=limit, offset=offset)
        )

    @trace_layer("service")
    async def get_subscription(
        self, user_id: str, subscription_id: str
    ) -> SubscriptionResult[SubscriptionResponse]:
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

    @trace_layer("service")
    async def cancel_subscription(  # noqa: PLR0912 -- refund reconciliation branches are explicit
        self,
        user_id: str,
        subscription_id: str,
        dto: SubscriptionCancelDTO,
        *,
        idempotency_key: str | None = None,
    ) -> SubscriptionResult[SubscriptionResponse]:
        owned = await self._get_owned_subscription(user_id, subscription_id)
        if isinstance(owned, Failure):
            return owned
        subscription = owned.unwrap()
        operation_key = idempotency_key or (
            f"billing:cancel:{subscription.id}:"
            f"{'period-end' if dto.cancel_at_period_end else 'immediate'}"
        )
        # Immediate cancellation stages the local terminal state before the
        # refund provider call.  If the process dies after the remote receipt
        # but before the HTTP response, a keyless retry must enter the refund
        # ledger again instead of failing the state transition first.
        if subscription.status == SubscriptionStatus.CANCELLED.value:
            if dto.cancel_at_period_end:
                plan_result = await self._load_plan(subscription.plan_id)
                if isinstance(plan_result, Failure):
                    return Success(_subscription_to_response(subscription, plan=None))
                return Success(_subscription_to_response(subscription, plan=plan_result.unwrap()))
            plan_result = await self._load_plan(subscription.plan_id)
            if isinstance(plan_result, Failure):
                return plan_result
            refund = await self._refund_on_cancel(
                subscription,
                plan_result.unwrap(),
                user_id,
                idempotency_key=operation_key,
            )
            if isinstance(refund, Failure):
                return refund
            refreshed = await self.subscriptions.find_by_id(subscription.id)
            if isinstance(refreshed, Failure):
                return refreshed
            recovered = refreshed.unwrap() or subscription
            return Success(_subscription_to_response(recovered, plan=plan_result.unwrap()))
        if subscription.status not in {
            SubscriptionStatus.ACTIVE.value,
            SubscriptionStatus.PAUSED.value,
            SubscriptionStatus.PAST_DUE.value,
            SubscriptionStatus.HALTED.value,
        }:
            return Failure(
                SubscriptionInvalidTransitionError(
                    message=f"Invalid state transition: {subscription.status} -> {SubscriptionStatus.CANCELLED.value}",
                    details={
                        "current": subscription.status,
                        "target": SubscriptionStatus.CANCELLED.value,
                    },
                    source="subscription_service",
                    subscription_id=str(subscription.id),
                    current=subscription.status,
                    target=SubscriptionStatus.CANCELLED.value,
                )
            )

        now = datetime.now(tz=UTC)
        applied = await self._apply_lifecycle(
            subscription,
            CancelSubscription(
                metadata=CommandMetadata(
                    request_id=operation_key, expected_version=subscription.version
                ),
                cancel_at_period_end=dto.cancel_at_period_end,
            ),
            context=PolicyContext(now=now),
            idempotency_key=operation_key,
        )
        if isinstance(applied, Failure):
            return applied
        if not dto.cancel_at_period_end:
            plan_result = await self._load_plan(subscription.plan_id)
            if isinstance(plan_result, Failure):
                return plan_result
            # The coordinator committed the local cancellation before the
            # independent refund mutation.  Refresh first so a refund-sync
            # failure can update remote_sync against the new lifecycle
            # version without touching/undoing the terminal cancellation.
            current_result = await self.subscriptions.find_by_id(subscription.id)
            if isinstance(current_result, Failure):
                return current_result
            current = current_result.unwrap() or subscription
            refund = await self._refund_on_cancel(
                current,
                plan_result.unwrap(),
                user_id,
                idempotency_key=operation_key,
            )
            if isinstance(refund, Failure):
                return refund
        refreshed = await self.subscriptions.find_by_id(subscription.id)
        if isinstance(refreshed, Failure):
            return refreshed
        subscription = refreshed.unwrap() or subscription
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

    @trace_layer("service")
    async def pause_subscription(
        self,
        user_id: str,
        subscription_id: str,
        dto: SubscriptionPauseDTO,
        *,
        idempotency_key: str | None = None,
    ) -> SubscriptionResult[SubscriptionResponse]:
        owned = await self._get_owned_subscription(user_id, subscription_id)
        if isinstance(owned, Failure):
            return owned
        subscription = owned.unwrap()
        if subscription.status != SubscriptionStatus.ACTIVE.value:
            return Failure(
                SubscriptionInvalidTransitionError(
                    message=f"Invalid state transition: {subscription.status} -> {SubscriptionStatus.PAUSED.value}",
                    details={
                        "current": subscription.status,
                        "target": SubscriptionStatus.PAUSED.value,
                    },
                    source="subscription_service",
                    subscription_id=str(subscription.id),
                    current=subscription.status,
                    target=SubscriptionStatus.PAUSED.value,
                )
            )

        operation_key = self._request_idempotency_key(idempotency_key, "pause")
        applied = await self._apply_lifecycle(
            subscription,
            PauseSubscription(
                metadata=CommandMetadata(
                    request_id=operation_key, expected_version=subscription.version
                ),
                duration_days=dto.pause_duration_days,
            ),
            context=PolicyContext(now=datetime.now(tz=UTC)),
            idempotency_key=operation_key,
        )
        if isinstance(applied, Failure):
            return applied
        refreshed = await self.subscriptions.find_by_id(subscription.id)
        if isinstance(refreshed, Failure):
            return refreshed
        subscription = refreshed.unwrap() or subscription
        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(subscription.id),
                action="subscription.paused",
                user_id=user_id,
                changes={
                    "pause_start": subscription.pause_start,
                    "pause_end": subscription.pause_end,
                },
            )
        )
        plan_result = await self._load_plan(subscription.plan_id)
        if isinstance(plan_result, Failure):
            return Success(_subscription_to_response(subscription, plan=None))
        plan = plan_result.unwrap()
        return Success(_subscription_to_response(subscription, plan=plan))

    @trace_layer("service")
    async def resume_subscription(
        self,
        user_id: str,
        subscription_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> SubscriptionResult[SubscriptionResponse]:
        owned = await self._get_owned_subscription(user_id, subscription_id)
        if isinstance(owned, Failure):
            return owned
        subscription = owned.unwrap()
        if subscription.status != SubscriptionStatus.PAUSED.value:
            return Failure(
                SubscriptionInvalidTransitionError(
                    message=f"Invalid state transition: {subscription.status} -> {SubscriptionStatus.ACTIVE.value}",
                    details={
                        "current": subscription.status,
                        "target": SubscriptionStatus.ACTIVE.value,
                    },
                    source="subscription_service",
                    subscription_id=str(subscription.id),
                    current=subscription.status,
                    target=SubscriptionStatus.ACTIVE.value,
                )
            )

        operation_key = self._request_idempotency_key(idempotency_key, "resume")
        applied = await self._apply_lifecycle(
            subscription,
            ResumeSubscription(
                metadata=CommandMetadata(
                    request_id=operation_key, expected_version=subscription.version
                )
            ),
            context=PolicyContext(now=datetime.now(tz=UTC)),
            idempotency_key=operation_key,
        )
        if isinstance(applied, Failure):
            return applied
        refreshed = await self.subscriptions.find_by_id(subscription.id)
        if isinstance(refreshed, Failure):
            return refreshed
        subscription = refreshed.unwrap() or subscription
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

    @trace_layer("service")
    async def change_plan(  # noqa: PLR0912 -- lifecycle and economic branches are explicit
        self,
        user_id: str,
        subscription_id: str,
        dto: PlanChangeDTO,
        *,
        idempotency_key: str | None = None,
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

        if not new_plan.razorpay_plan_id:
            return Failure(
                SubscriptionValidationError(
                    message="Plan is missing its provider billing binding",
                    details={"new_plan_id": str(new_plan.id)},
                    source="subscription_service",
                )
            )
        self._ensure_plan_binding(subscription, new_plan)
        operation_key = self._request_idempotency_key(idempotency_key, "change-plan")
        effective_at = dto.effective_date or datetime.now(tz=UTC)
        proration_value = None
        if (
            current_plan.interval == new_plan.interval
            and subscription.current_period_start is not None
            and subscription.current_period_end is not None
        ):
            proration = calculate_plan_change_proration(
                subscription, current_plan, new_plan, effective_date=effective_at
            )
            if isinstance(proration, Failure):
                return proration
            proration_value = proration.unwrap()
        applied = await self._apply_lifecycle(
            subscription,
            ChangeSubscriptionPlan(
                metadata=CommandMetadata(
                    request_id=operation_key, expected_version=subscription.version
                ),
                new_plan_id=str(new_plan.id),
                effective_at=effective_at,
            ),
            context=PolicyContext(
                now=datetime.now(tz=UTC),
                current_plan_amount=int(current_plan.amount),
                target_plan_amount=int(new_plan.amount),
                current_interval=current_plan.interval,
                target_interval=new_plan.interval,
                target_currency=new_plan.currency,
                target_provider_plan_id=new_plan.razorpay_plan_id,
            ),
            idempotency_key=operation_key,
        )
        if isinstance(applied, Failure):
            return applied
        # The provider plan mutation must succeed before issuing the economic
        # compensation.  If the request crashes here, retrying the same
        # idempotency key reuses the lifecycle outcome and the operation-keyed
        # credit grant below, so reconciliation cannot orphan or duplicate it.
        if (
            proration_value is not None
            and proration_value.direction.value == "downgrade"
            and self.credit_service is not None
            and subscription.current_period_end is not None
        ):
            source_invoice = await self._authoritative_source_invoice_id(subscription)
            if isinstance(source_invoice, Failure):
                return source_invoice
            source_invoice_id = source_invoice.unwrap()
            if source_invoice_id is None:
                return Failure(
                    SubscriptionValidationError(
                        message="Downgrade credit requires the authoritative source invoice",
                        details={
                            "subscription_id": str(subscription.id),
                            "manual_review": True,
                        },
                        source="subscription_service",
                    )
                )
            credit = await self.credit_service.grant_credit_on_downgrade(
                user_id,
                subscription.id,
                proration_value.total_amount,
                billing_cycle_end=subscription.current_period_end,
                currency=new_plan.currency,
                operation_key=f"{operation_key}:downgrade-credit",
                source_invoice_id=source_invoice_id,
            )
            if isinstance(credit, Failure):
                return Failure(
                    SubscriptionInfrastructureError(
                        message=credit.failure().message,
                        details=credit.failure().details,
                        source="subscription_service",
                        operation="grant_downgrade_credit",
                    )
                )
        refreshed = await self.subscriptions.find_by_id(subscription.id)
        if isinstance(refreshed, Failure):
            return refreshed
        subscription = refreshed.unwrap() or subscription
        await self.audit.create(
            AuditLog(
                entity_type="subscription",
                entity_id=str(subscription.id),
                action=AuditAction.PLAN_CHANGED.value,
                user_id=user_id,
                changes={"old_plan_id": str(current_plan.id), "new_plan_id": str(new_plan.id)},
            )
        )
        return Success(
            _subscription_to_response(
                subscription,
                plan=new_plan,
                payment_url=self._provider_payment_url(applied.unwrap()),
            )
        )

    @trace_layer("service")
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
        return calculate_plan_change_proration(subscription, current_plan, new_plan)

    @trace_layer("service")
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
