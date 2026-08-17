"""Scheduled billing jobs: renewal, dunning, invoice/receipt, pause-resume, reconciliation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from returns.result import Failure
from sqlalchemy import select

from app.config import get_settings
from app.connections import ResilientTask, celery_app, init_db
from app.features.audit.model import AuditAction, AuditLog
from app.features.audit.repository import AuditLogRepository
from app.features.dunning.service import DunningService
from app.features.invoices.model import Invoice
from app.features.invoices.receipt import PaymentReceipt
from app.features.invoices.repository import InvoiceRepository
from app.features.invoices.service import InvoiceService
from app.features.payments.clients.razorpay_client import RazorpayClient
from app.features.payments.dto import PaymentRecordDTO
from app.features.payments.model import Payment
from app.features.payments.repository import PaymentRepository
from app.features.payments.service import PaymentService
from app.features.plans.repository import PlanRepository
from app.features.subscriptions.model import Subscription, SubscriptionStatus
from app.features.subscriptions.repository import SubscriptionRepository
from app.utils import logger

settings = get_settings()


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


async def _run(operation) -> dict[str, int]:
    engine, session_local = await init_db()
    try:
        async with session_local() as session:
            try:
                result = await operation(session)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            else:
                return result
    finally:
        await engine.dispose()


def _current_utc() -> datetime:
    return datetime.now(tz=UTC)


async def _renewal_job(session) -> dict[str, int]:
    """Reconcile-only renewal pass: no Razorpay charges are initiated here."""
    subscriptions = _subscription_repo(session)
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

    razorpay = RazorpayClient()
    renewed = 0
    for subscription in subscription_rows:
        if not subscription.razorpay_subscription_id:
            continue
        try:  # noqa: PLW0717
            data = await razorpay.fetch_subscription(subscription.razorpay_subscription_id)
            current_start = data.get("current_start")
            current_end = data.get("current_end")
            values: dict[str, object] = {}
            if current_start:
                values["current_period_start"] = datetime.fromtimestamp(int(current_start), tz=UTC)
            if current_end:
                values["current_period_end"] = datetime.fromtimestamp(int(current_end), tz=UTC)
            if not values:
                continue
            update = await subscriptions.update_with_lock(
                subscription, subscription.version, values=values
            )
            if isinstance(update, Failure):
                logger.bind(operation="billing.renewal").warning(update.failure().message)
                continue
            renewed += 1
        except Exception as exc:  # noqa: BLE001 — one bad subscription must not kill the run
            logger.bind(
                operation="billing.renewal",
                subscription_id=str(subscription.id),
            ).warning("renewal reconcile failed", error=str(exc))

    await audit.create(
        AuditLog(
            entity_type="system",
            entity_id="billing",
            action=AuditAction.RECONCILIATION_RUN.value,
            changes={"job": "renewal", "checked": len(subscription_rows), "renewed": renewed},
        )
    )
    return {"checked": len(subscription_rows), "renewed": renewed}


async def _dunning_job(session) -> dict[str, int]:
    service = DunningService(
        session,
        SubscriptionRepository(session),
        PlanRepository(session),
        AuditLogRepository(session),
    )
    due = await service.find_due_for_retry(limit=200)
    retried = 0
    halted = 0
    for subscription in due:
        updated = await service.execute_retry(subscription)
        if updated.status == SubscriptionStatus.HALTED.value:
            halted += 1
        retried += 1
    return {"due": len(due), "retried": retried, "halted": halted}


async def _invoice_backfill(session) -> dict[str, int]:
    service = _invoice_service(session)
    subscriptions = _subscription_repo(session)
    plans = _plan_repo(session)
    existing = select(Invoice.payment_id).where(Invoice.payment_id.is_not(None))
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
            await service.generate_for_payment(payment, subscription, plan)
            generated += 1
        except Exception as exc:  # noqa: BLE001
            logger.bind(operation="billing.invoice_backfill", payment_id=str(payment.id)).warning(
                "invoice generation failed", error=str(exc)
            )
    return {"checked": len(payments), "generated": generated}


async def _receipt_backfill(session) -> dict[str, int]:
    service = _invoice_service(session)
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
            await service.generate_receipt_for_payment(payment, subscription, plan)
            generated += 1
        except Exception as exc:  # noqa: BLE001
            logger.bind(operation="billing.receipt_backfill", payment_id=str(payment.id)).warning(
                "receipt generation failed", error=str(exc)
            )
    return {"checked": len(payments), "generated": generated}


async def _pause_resume_job(session) -> dict[str, int]:
    subscriptions = _subscription_repo(session)
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
        update = await subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={"pause_start": None, "pause_end": None},
        )
        if isinstance(update, Failure):
            logger.bind(
                operation="billing.pause_resume", subscription_id=str(subscription.id)
            ).warning(update.failure().message)
            continue
        resumed += 1
    return {"checked": len(subscription_rows), "resumed": resumed}


async def _reconciliation_job(session) -> dict[str, int]:
    """Daily Razorpay reconciliation: fetch captured payments and align local records."""
    service = PaymentService(PaymentRepository(session), AuditLogRepository(session))
    subscriptions = _subscription_repo(session)
    razorpay = RazorpayClient()

    since = _current_utc() - timedelta(days=settings.BILLING_RECONCILIATION_LOOKBACK_DAYS)
    reconciled = 0
    missing = 0
    try:
        payments = await razorpay.list_payments(
            params={
                "from": int(since.timestamp()),
                "to": int(_current_utc().timestamp()),
                "count": 100,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.bind(operation="billing.reconciliation").error(
            "Razorpay payment fetch failed", error=str(exc)
        )
        return {"reconciled": 0, "missing": 0}

    for payment in payments:
        rz_id = payment.get("id")
        if not isinstance(rz_id, str):
            continue
        existing = await service.payments.find_by_razorpay_id(rz_id)
        if isinstance(existing, Failure):
            continue
        if existing.unwrap() is not None:
            reconciled += 1
            continue
        # New payment not created via webhook: create if subscription resolvable.
        sub_result = await subscriptions.find_by_razorpay_id(
            str(payment.get("subscription_id") or "")
        )
        if isinstance(sub_result, Failure) or sub_result.unwrap() is None:
            missing += 1
            continue
        subscription = sub_result.unwrap()
        if subscription is None:
            missing += 1
            continue
        await service.record_payment(
            PaymentRecordDTO(
                razorpay_payment_id=rz_id,
                subscription_id=str(subscription.id),
                amount=int(payment.get("amount") or 0),
                currency=str(payment.get("currency") or "INR"),
                method=payment.get("method"),
                captured_at=datetime.fromtimestamp(int(payment.get("created_at") or 0), tz=UTC),
            ),
            subscription=subscription,
        )
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


@celery_app.task(name="billing.renewal", base=ResilientTask)
def billing_renewal() -> dict[str, int]:
    return asyncio.run(_run(_renewal_job))


@celery_app.task(name="billing.dunning", base=ResilientTask)
def billing_dunning() -> dict[str, int]:
    return asyncio.run(_run(_dunning_job))


@celery_app.task(name="billing.invoice_generation", base=ResilientTask)
def billing_invoice_generation() -> dict[str, int]:
    return asyncio.run(_run(_invoice_backfill))


@celery_app.task(name="billing.receipt_generation", base=ResilientTask)
def billing_receipt_generation() -> dict[str, int]:
    return asyncio.run(_run(_receipt_backfill))


@celery_app.task(name="billing.pause_resume", base=ResilientTask)
def billing_pause_resume() -> dict[str, int]:
    return asyncio.run(_run(_pause_resume_job))


@celery_app.task(name="billing.reconciliation", base=ResilientTask)
def billing_reconciliation() -> dict[str, int]:
    return asyncio.run(_run(_reconciliation_job))
