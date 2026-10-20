"""Scheduled billing jobs: renewal, dunning, invoice/receipt, pause-resume, reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from returns.result import Failure
from sqlalchemy import select

from app.config import get_settings
from app.connections.postgres import independent_session, init_db
from app.features.audit.models import AuditAction, AuditLog
from app.features.audit.repository import AuditLogRepository
from app.features.billing.dunning.service import DunningService
from app.features.billing.invoices.receipt import PaymentReceipt
from app.features.billing.invoices.repository import InvoiceRepository
from app.features.billing.invoices.service import InvoiceService
from app.features.billing.payments.currency import CurrencyCode
from app.features.billing.payments.dto import PaymentRecordDTO
from app.features.billing.payments.models import Payment, PaymentMethod
from app.features.billing.payments.repository import PaymentRepository
from app.features.billing.payments.service import PaymentService
from app.features.billing.plans.repository import PlanRepository
from app.features.billing.plans.service import PlanService
from app.features.billing.providers import (
    ListPaymentsOperation,
    ObserveSubscriptionOperation,
    ProviderPaymentBatch,
    ProviderSubscriptionState,
)
from app.features.billing.providers.factory import build_billing_provider
from app.features.billing.subscriptions.coordinator import SubscriptionLifecycleCoordinator
from app.features.billing.subscriptions.lifecycle import (
    CommandMetadata,
    EventMetadata,
    PaymentCaptured,
    PolicyContext,
    ResumeSubscription,
    SubscriptionCharged,
    SubscriptionResumed,
)
from app.features.billing.subscriptions.models import Subscription, SubscriptionStatus
from app.features.billing.subscriptions.repository import (
    SqlAlchemyLifecyclePersistence,
    SubscriptionRepository,
)
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.features.billing.providers import BillingProvider

type SessionFactory = async_sessionmaker[AsyncSession]
type BillingOperation = Callable[[AsyncSession, SessionFactory], Awaitable[dict[str, int]]]


def _subscription_repo(session) -> SubscriptionRepository:
    return SubscriptionRepository(session)


def _plan_repo(session) -> PlanRepository:
    return PlanRepository(session)


def _payment_repo(session) -> PaymentRepository:
    return PaymentRepository(session)


def _invoice_repo(session) -> InvoiceRepository:
    return InvoiceRepository(session)


def _audit_repo(session) -> AuditLogRepository:
    return AuditLogRepository(session)


def _invoice_service(session) -> InvoiceService:
    return InvoiceService(
        session,
        InvoiceRepository(session),
        SubscriptionRepository(session),
        PlanRepository(session),
        PaymentRepository(session),
        AuditLogRepository(session),
    )


async def run_billing_job(operation: BillingOperation) -> dict[str, int]:
    engine, session_local = await init_db()
    try:
        async with session_local() as session:
            try:
                result = await operation(session, session_local)
                await session.commit()
            except Exception as exc:
                exc.add_note("operation=billing.scheduled_run, stage=commit")
                await session.rollback()
                raise
            else:
                return result
    finally:
        await engine.dispose()


def _current_utc() -> datetime:
    return datetime.now(tz=UTC)


async def renewal_job(
    session: AsyncSession,
    session_factory: SessionFactory,
    *,
    provider: BillingProvider | None = None,
) -> dict[str, int]:
    """Reconcile-only renewal pass: no Razorpay charges are initiated here."""
    audit = _audit_repo(session)
    now = _current_utc()
    statement = (
        select(Subscription)
        .where(
            Subscription.status == SubscriptionStatus.ACTIVE.value,
            Subscription.current_period_end.is_not(None),
            Subscription.current_period_end < now,
            Subscription.deleted_at.is_(None),
        )
        .limit(500)
    )
    result = await session.execute(statement)
    subscription_rows = list(result.scalars().all())

    provider = provider or build_billing_provider()
    renewed = 0
    for subscription in subscription_rows:
        if not subscription.razorpay_subscription_id:
            continue
        try:
            observed = await provider.execute(
                ObserveSubscriptionOperation(
                    provider_subscription_id=subscription.razorpay_subscription_id,
                    correlation_id=f"billing:renewal:{subscription.id}:{subscription.version}",
                )
            )
            if isinstance(observed, Failure):
                error = observed.failure()
                logger.bind(
                    operation="billing.renewal",
                    subscription_id=str(subscription.id),
                    error=error.message,
                    details=error,
                ).warning("Renewal provider observation failed")
                continue
            value = observed.unwrap().value
            if not isinstance(value, ProviderSubscriptionState):
                logger.bind(
                    operation="billing.renewal",
                    subscription_id=str(subscription.id),
                ).warning("Renewal provider observation had an unexpected value")
                continue
            current_start = value.current_period_start
            current_end = value.current_period_end
            if current_start is None and current_end is None:
                continue
            async with independent_session(session_factory) as item_session:
                coordinator = SubscriptionLifecycleCoordinator(
                    SqlAlchemyLifecyclePersistence(item_session), provider
                )
                lifecycle_result = await coordinator.observe(
                    SubscriptionCharged(
                        metadata=EventMetadata(
                            event_id=(f"billing:renewal:{subscription.id}:{subscription.version}"),
                            occurred_at=now,
                            expected_version=subscription.version,
                        ),
                        period_start=current_start,
                        period_end=current_end,
                    ),
                    idempotency_key=f"billing:renewal:{subscription.id}:{subscription.version}",
                    subscription_id=str(subscription.id),
                    context=PolicyContext(now=now),
                )
                if isinstance(lifecycle_result, Failure):
                    renewal_error = lifecycle_result.failure()
                    logger.bind(
                        operation="billing.renewal",
                        error=renewal_error.message,
                        details=renewal_error.details,
                    ).warning("Renewal update failed")
                    continue
                if lifecycle_result.unwrap().status.value != "skipped":
                    renewed += 1
        except Exception as exc:  # noqa: BLE001 — one bad subscription must not kill the run
            exc.add_note("operation=billing.renewal, stage=reconcile_subscription")
            logger.bind(
                operation="billing.renewal",
                subscription_id=str(subscription.id),
                error=str(exc),
            ).warning("renewal reconcile failed")

    await audit.create(
        AuditLog(
            entity_type="system",
            entity_id="billing",
            action=AuditAction.RECONCILIATION_RUN.value,
            changes={"job": "renewal", "checked": len(subscription_rows), "renewed": renewed},
        )
    )
    return {"checked": len(subscription_rows), "renewed": renewed}


async def dunning_job(
    session: AsyncSession,
    session_factory: SessionFactory,
    *,
    provider: BillingProvider | None = None,
) -> dict[str, int]:
    provider = provider or build_billing_provider()
    service = DunningService(
        session,
        SubscriptionRepository(session),
        PlanRepository(session),
        AuditLogRepository(session),
        provider=provider,
    )
    due_result = await service.find_due_for_retry(limit=200)
    if isinstance(due_result, Failure):
        dunning_error = due_result.failure()
        logger.bind(
            operation="billing.dunning",
            error=dunning_error.message,
            details=dunning_error.details,
        ).error("Dunning query failed")
        return {"due": 0, "retried": 0, "halted": 0}
    due = due_result.unwrap()
    retried = 0
    halted = 0
    for subscription in due:
        async with independent_session(session_factory) as item_session:
            item_service = DunningService(
                item_session,
                subscriptions=SubscriptionRepository(item_session),
                plans=PlanRepository(item_session),
                audit=AuditLogRepository(item_session),
                provider=provider,
            )
            updated_result = await item_service.execute_retry(subscription)
            if isinstance(updated_result, Failure):
                retry_error = updated_result.failure()
                logger.bind(
                    operation="billing.dunning",
                    subscription_id=str(subscription.id),
                    error=retry_error.message,
                    details=retry_error.details,
                ).warning("Dunning retry failed")
                continue
            updated = updated_result.unwrap()
            if updated.status == SubscriptionStatus.HALTED.value:
                halted += 1
            retried += 1
    return {"due": len(due), "retried": retried, "halted": halted}


async def plan_recovery_job(
    session: AsyncSession,
    _session_factory: SessionFactory,
    *,
    provider: BillingProvider | None = None,
) -> dict[str, int]:
    """Admin-safe sweep for plan operations stranded before local binding."""
    provider = provider or build_billing_provider()
    coordinator = SubscriptionLifecycleCoordinator(
        SqlAlchemyLifecyclePersistence(session), provider
    )
    intent_recovery = await coordinator.recover_pending()
    recovered_intents = 0
    if isinstance(intent_recovery, Failure):
        logger.bind(
            operation="billing.plan_recovery",
            error=intent_recovery.failure().message,
            details=intent_recovery.failure().details,
        ).warning("Provider intent recovery sweep failed")
    else:
        recovered_intents = intent_recovery.unwrap()
    service = PlanService(
        session,
        plans=PlanRepository(session),
        audit=AuditLogRepository(session),
        lifecycle_coordinator=coordinator,
    )
    result = await service.recover_pending_provider_plans()
    if isinstance(result, Failure):
        logger.bind(
            operation="billing.plan_recovery",
            error=result.failure().message,
            details=result.failure().details,
        ).warning("Provider plan recovery sweep failed")
        return {"recovered": 0, "intents_recovered": recovered_intents}
    return {"recovered": result.unwrap(), "intents_recovered": recovered_intents}


async def invoice_backfill(
    session: AsyncSession, session_factory: SessionFactory
) -> dict[str, int]:
    subscriptions = _subscription_repo(session)
    plans = _plan_repo(session)
    statement = (
        select(Payment)
        .where(
            Payment.status == "captured",
            # A historical invoice may already point at this payment while
            # the inverse payment relation is still NULL.  Include both that
            # repair case and genuinely missing invoices; the invoice service
            # is idempotent for the former.
            Payment.invoice_id.is_(None),
        )
        .limit(200)
    )
    result = await session.execute(statement)
    payments = list(result.scalars().all())

    generated = 0
    for payment in payments:
        sub_result = await subscriptions.find_by_id(payment.subscription_id)
        if isinstance(sub_result, Failure):
            continue
        subscription = sub_result.unwrap()
        if subscription is None:
            continue
        plan_result = await plans.find_by_id(subscription.plan_id)
        if isinstance(plan_result, Failure):
            continue
        plan = plan_result.unwrap()
        if plan is None:
            continue
        try:
            async with independent_session(session_factory) as item_session:
                await _invoice_service(item_session).generate_for_payment(
                    payment, subscription, plan
                )
                generated += 1
        except Exception as exc:  # noqa: BLE001 -- one bad payment must not kill the run
            exc.add_note("operation=billing.invoice_backfill, stage=generate_for_payment")
            logger.bind(
                operation="billing.invoice_backfill",
                payment_id=str(payment.id),
                error=str(exc),
            ).warning("invoice generation failed")
    return {"checked": len(payments), "generated": generated}


async def receipt_backfill(
    session: AsyncSession, session_factory: SessionFactory
) -> dict[str, int]:
    subscriptions = _subscription_repo(session)
    plans = _plan_repo(session)
    existing = select(PaymentReceipt.payment_id)
    statement = (
        select(Payment)
        .where(
            Payment.status == "captured",
            Payment.id.not_in(existing),
        )
        .limit(200)
    )
    result = await session.execute(statement)
    payments = list(result.scalars().all())

    generated = 0
    for payment in payments:
        sub_result = await subscriptions.find_by_id(payment.subscription_id)
        if isinstance(sub_result, Failure):
            continue
        subscription = sub_result.unwrap()
        if subscription is None:
            continue
        plan_result = await plans.find_by_id(subscription.plan_id)
        if isinstance(plan_result, Failure):
            continue
        plan = plan_result.unwrap()
        if plan is None:
            continue
        try:
            async with independent_session(session_factory) as item_session:
                await _invoice_service(item_session).generate_receipt_for_payment(
                    payment, subscription, plan
                )
                generated += 1
        except Exception as exc:  # noqa: BLE001 -- one bad payment must not kill the run
            exc.add_note("operation=billing.receipt_backfill, stage=generate_receipt")
            logger.bind(
                operation="billing.receipt_backfill",
                payment_id=str(payment.id),
                error=str(exc),
            ).warning("receipt generation failed")
    return {"checked": len(payments), "generated": generated}


async def pause_resume_job(
    session: AsyncSession,
    session_factory: SessionFactory,
    *,
    provider: BillingProvider | None = None,
) -> dict[str, int]:
    provider = provider or build_billing_provider()
    now = _current_utc()
    statement = (
        select(Subscription)
        .where(
            Subscription.status == SubscriptionStatus.PAUSED.value,
            Subscription.pause_end.is_not(None),
            Subscription.pause_end <= now,
            Subscription.deleted_at.is_(None),
        )
        .limit(500)
    )
    result = await session.execute(statement)
    subscription_rows = list(result.scalars().all())

    resumed = 0
    for subscription in subscription_rows:
        if not subscription.razorpay_subscription_id:
            continue
        observed = await provider.execute(
            ObserveSubscriptionOperation(
                provider_subscription_id=subscription.razorpay_subscription_id,
                correlation_id=f"billing:pause-resume:{subscription.id}:{subscription.version}",
            )
        )
        if isinstance(observed, Failure):
            error = observed.failure()
            logger.bind(
                operation="billing.pause_resume",
                subscription_id=str(subscription.id),
                error=error.message,
                details=error,
            ).warning("Pause resume provider observation failed")
            continue
        value = observed.unwrap().value
        if not isinstance(value, ProviderSubscriptionState) or value.status not in {
            "active",
            "paused",
        }:
            continue
        async with independent_session(session_factory) as item_session:
            coordinator = SubscriptionLifecycleCoordinator(
                SqlAlchemyLifecyclePersistence(item_session), provider
            )
            key = f"billing:pause-resume:{subscription.id}:{subscription.version}"
            if value.status == "active":
                lifecycle_input = SubscriptionResumed(
                    metadata=EventMetadata(
                        event_id=key,
                        occurred_at=now,
                        expected_version=subscription.version,
                    )
                )
                lifecycle_result = await coordinator.observe(
                    lifecycle_input,
                    idempotency_key=key,
                    subscription_id=str(subscription.id),
                    context=PolicyContext(now=now),
                )
            else:
                lifecycle_result = await coordinator.apply(
                    ResumeSubscription(
                        metadata=CommandMetadata(
                            request_id=key,
                            expected_version=subscription.version,
                        )
                    ),
                    idempotency_key=key,
                    subscription_id=str(subscription.id),
                    context=PolicyContext(now=now),
                )
            if isinstance(lifecycle_result, Failure):
                pause_error = lifecycle_result.failure()
                logger.bind(
                    operation="billing.pause_resume",
                    subscription_id=str(subscription.id),
                    error=pause_error.message,
                    details=pause_error.details,
                ).warning("Pause resume update failed")
                continue
            if lifecycle_result.unwrap().status.value != "skipped":
                resumed += 1
    return {"checked": len(subscription_rows), "resumed": resumed}


async def reconciliation_job(  # noqa: PLR0912, PLR0914, PLR0915 — typed facts and persistence stages stay local
    session: AsyncSession,
    session_factory: SessionFactory,
    *,
    provider: BillingProvider | None = None,
) -> dict[str, int]:
    """Daily Razorpay reconciliation: fetch captured payments and align local records."""
    service = PaymentService(PaymentRepository(session), AuditLogRepository(session))
    subscriptions = _subscription_repo(session)
    provider = provider or build_billing_provider()
    lookback_days = get_settings().BILLING_RECONCILIATION_LOOKBACK_DAYS

    since = _current_utc() - timedelta(days=lookback_days)
    reconciled = 0
    missing = 0
    try:
        observed = await provider.execute(
            ListPaymentsOperation(
                correlation_id=f"billing:reconciliation:{since.date().isoformat()}",
                from_timestamp=int(since.timestamp()),
                to_timestamp=int(_current_utc().timestamp()),
                count=100,
            )
        )
        if isinstance(observed, Failure):
            error = observed.failure()
            logger.bind(
                operation="billing.reconciliation",
                error=error.message,
                details=error,
            ).warning("Billing provider payment observation failed")
            return {"reconciled": 0, "missing": 0}
        value = observed.unwrap().value
        if not isinstance(value, ProviderPaymentBatch):
            logger.bind(operation="billing.reconciliation").warning(
                "Billing provider returned an unexpected payment observation"
            )
            return {"reconciled": 0, "missing": 0}
    except Exception as exc:  # noqa: BLE001 -- upstream failure degrades this scheduled run
        exc.add_note("operation=billing.reconciliation, stage=fetch_payments")
        logger.bind(operation="billing.reconciliation", error=str(exc)).exception(
            "Razorpay payment fetch failed"
        )
        return {"reconciled": 0, "missing": 0}

    for payment in value.payments:
        rz_id = payment.provider_payment_id
        if (
            not isinstance(rz_id, str)
            or not rz_id.strip()
            or isinstance(payment.amount, bool)
            or payment.amount <= 0
            or payment.currency not in {code.value for code in CurrencyCode}
        ):
            missing += 1
            continue
        existing = await service.payments.find_by_razorpay_id(rz_id)
        if isinstance(existing, Failure):
            continue
        existing_payment = existing.unwrap()
        if existing_payment is not None:
            # A payment row can be durable before a pending-upgrade capture
            # gate succeeds.  Keep retrying the lifecycle dependency while
            # the subscription still advertises that exact pending upgrade;
            # ordinary already-processed payments remain a cheap no-op.
            sub_result = await subscriptions.find_by_id(existing_payment.subscription_id)
            if isinstance(sub_result, Failure) or sub_result.unwrap() is None:
                missing += 1
                continue
            subscription = sub_result.unwrap()
            pending_upgrade = (getattr(subscription, "metadata_", None) or {}).get(
                "pending_upgrade"
            )
            if not isinstance(pending_upgrade, dict):
                reconciled += 1
                continue
        else:
            # New payment not created via webhook: create if subscription resolvable.
            sub_result = await subscriptions.find_by_razorpay_id(payment.subscription_id or "")
            if isinstance(sub_result, Failure) or sub_result.unwrap() is None:
                missing += 1
                continue
            subscription = sub_result.unwrap()
        if subscription is None:
            missing += 1
            continue
        async with independent_session(session_factory) as item_session:
            try:
                method = PaymentMethod(payment.method) if payment.method is not None else None
            except ValueError:
                method = None
            item_service = PaymentService(
                PaymentRepository(item_session), AuditLogRepository(item_session)
            )
            if existing_payment is None:
                recorded = await item_service.record_payment(
                    PaymentRecordDTO(
                        razorpay_payment_id=rz_id,
                        subscription_id=str(subscription.id),
                        amount=payment.amount,
                        currency=payment.currency,
                        method=method,
                        captured_at=payment.captured_at or payment.created_at or _current_utc(),
                    ),
                    subscription=subscription,
                )
                if isinstance(recorded, Failure):
                    missing += 1
                    continue
            lifecycle = SubscriptionLifecycleCoordinator(
                SqlAlchemyLifecyclePersistence(item_session), provider
            )
            payment_key = f"billing:payment-captured:{subscription.id}:{rz_id}"
            lifecycle_result = await lifecycle.observe(
                PaymentCaptured(
                    metadata=EventMetadata(
                        event_id=payment_key,
                        occurred_at=payment.captured_at or payment.created_at or _current_utc(),
                        expected_version=subscription.version,
                    ),
                    payment_id=rz_id,
                    amount_paisa=payment.amount,
                    currency=payment.currency,
                ),
                idempotency_key=payment_key,
                subscription_id=str(subscription.id),
                context=PolicyContext(now=_current_utc()),
            )
            if isinstance(lifecycle_result, Failure):
                logger.bind(
                    operation="billing.reconciliation",
                    subscription_id=str(subscription.id),
                    payment_id=rz_id,
                    error=lifecycle_result.failure().message,
                ).warning("Payment lifecycle observation failed")
                missing += 1
                continue
            reconciled += 1

    await _audit_repo(session).create(
        AuditLog(
            entity_type="system",
            entity_id="billing",
            action=AuditAction.RECONCILIATION_RUN.value,
            changes={"job": "daily", "reconciled": reconciled, "missing": missing},
        )
    )
    return {"reconciled": reconciled, "missing": missing}


# These are dispatched by the scheduler with no keyword arguments. Registering a
# field-less payload states that as the contract, so a scheduler entry that starts
# passing arguments is refused at dispatch rather than reaching a body that cannot
# accept them — and so no dispatchable name is left with nothing registered
# against it, which is the state the dispatch helper now refuses outright.
