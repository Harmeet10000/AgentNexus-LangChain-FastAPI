"""Razorpay webhook processing: signature verification + idempotent dispatch."""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from returns.result import Failure

from app.config import get_settings
from app.features.billing.exceptions import WebhookVerificationException
from app.features.billing.models import (
    AuditAction,
    AuditLog,
    SubscriptionStatus,
    WebhookEvent,
    WebhookEventStatus,
    WebhookEventType,
)
from app.features.billing.services.invoice_service import InvoiceService
from app.features.billing.services.payment_service import PaymentService
from app.utils import logger

from ..dto import PaymentRecordDTO

if TYPE_CHECKING:
    from typing import Any

    from app.features.billing.models import Subscription
    from app.features.billing.repositories import BillingRepositories

_IGNORED_EVENTS = frozenset(
    {
        WebhookEventType.PAYMENT_AUTHORIZED.value,
        WebhookEventType.REFUND_CREATED.value,
    }
)

_TERMINAL_PAYMENT_STATES = frozenset({SubscriptionStatus.CANCELLED.value})


class WebhookService:
    """Verify, log, and dispatch Razorpay webhooks with idempotency."""

    def __init__(self, repos: BillingRepositories) -> None:
        self.repos = repos
        self.payments = PaymentService(repos)
        self.invoices = InvoiceService(repos)

    @staticmethod
    def verify_signature(*, raw_body: str, signature: str) -> None:
        secret = get_settings().RAZORPAY_WEBHOOK_SECRET.get_secret_value()
        if not secret:
            msg = "Razorpay webhook secret is not configured"
            raise WebhookVerificationException(msg)
        expected = hmac.new(secret.encode(), raw_body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            msg = "Signature mismatch"
            raise WebhookVerificationException(msg)

    async def process(  # noqa: PLR0912
        self, *, event_id: str, event_type: str, payload: dict[str, object]
    ) -> bool:
        """Idempotently process a verified webhook event. Returns True if handled."""
        existing = await self.repos.webhooks.find_by_razorpay_event_id(event_id)
        if isinstance(existing, Failure):
            raise WebhookVerificationException(existing.failure().message)
        previous = existing.unwrap()
        if previous is not None:
            if previous.status in {
                WebhookEventStatus.PROCESSED.value,
                WebhookEventStatus.SKIPPED.value,
            }:
                return True
            if previous.status == WebhookEventStatus.FAILED.value:
                # A previous attempt failed; allow re-processing of the delivery.
                pass
            else:
                msg = "Webhook event already in flight"
                raise WebhookVerificationException(msg)

        if event_type in _IGNORED_EVENTS:
            if previous is None:
                event = WebhookEvent(
                    razorpay_event_id=event_id,
                    event_type=event_type,
                    status=WebhookEventStatus.SKIPPED.value,
                    payload=payload,
                )
                await self.repos.webhooks.create(event)
            return False

        if previous is None:
            event = WebhookEvent(
                razorpay_event_id=event_id,
                event_type=event_type,
                status=WebhookEventStatus.PENDING.value,
                payload=payload,
            )
            created = await self.repos.webhooks.create(event)
            if isinstance(created, Failure):
                raise WebhookVerificationException(created.failure().message)
            event = created.unwrap()
        else:
            event = previous

        processing = await self.repos.webhooks.update_status(
            event, status=WebhookEventStatus.PROCESSING.value
        )
        if isinstance(processing, Failure):
            raise WebhookVerificationException(processing.failure().message)
        event = processing.unwrap()

        try:
            result = await self._dispatch(event_type, payload)
        except Exception as exc:  # noqa: BLE001  -- recorded, not re-raised
            update = await self.repos.webhooks.update_status(
                event,
                status=WebhookEventStatus.FAILED.value,
                extra_values={
                    "failed_at": datetime.now(tz=UTC),
                    "error_message": str(exc)[:1000],
                    "retry_count": event.retry_count + 1,
                },
            )
            if isinstance(update, Failure):
                logger.bind(operation="webhook").error(update.failure().message)
            logger.bind(operation="webhook", event_type=event_type, event_id=event_id).error(
                f"Webhook processing failed: {exc}"
            )
            return False

        status = (
            WebhookEventStatus.SKIPPED.value
            if result == "skipped"
            else WebhookEventStatus.PROCESSED.value
        )
        update = await self.repos.webhooks.update_status(
            event,
            status=status,
            extra_values={"processed_at": datetime.now(tz=UTC)},
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").error(update.failure().message)
        return True

    async def replay(self, event_id: str) -> WebhookEvent:
        """Re-process a FAILED webhook event (Requirement 22/31)."""
        result = await self.repos.webhooks.find_by_id(event_id)
        if isinstance(result, Failure):
            raise WebhookVerificationException(result.failure().message)
        event = result.unwrap()
        if event is None:
            msg = "Webhook event not found"
            raise WebhookVerificationException(msg)
        if event.status != WebhookEventStatus.FAILED.value:
            msg = f"Cannot replay event in status '{event.status}'"
            raise WebhookVerificationException(msg)
        if event.event_type in _IGNORED_EVENTS:
            msg = "Event type is intentionally ignored"
            raise WebhookVerificationException(msg)

        processing = await self.repos.webhooks.update_status(
            event, status=WebhookEventStatus.PROCESSING.value
        )
        if isinstance(processing, Failure):
            raise WebhookVerificationException(processing.failure().message)
        event = processing.unwrap()

        try:
            result = await self._dispatch(event.event_type, event.payload, replay=True)
        except Exception as exc:
            update = await self.repos.webhooks.update_status(
                event,
                status=WebhookEventStatus.FAILED.value,
                extra_values={
                    "failed_at": datetime.now(tz=UTC),
                    "error_message": str(exc)[:1000],
                    "retry_count": event.retry_count + 1,
                },
            )
            if isinstance(update, Failure):
                logger.bind(operation="webhook").error(update.failure().message)
            raise
        status = (
            WebhookEventStatus.SKIPPED.value
            if result == "skipped"
            else WebhookEventStatus.PROCESSED.value
        )
        updated = await self.repos.webhooks.update_status(
            event,
            status=status,
            extra_values={"processed_at": datetime.now(tz=UTC)},
        )
        if isinstance(updated, Failure):
            raise WebhookVerificationException(updated.failure().message)
        return updated.unwrap()

    async def _dispatch(  # noqa: PLR0912
        self, event_type: str, payload: dict[str, object], *, replay: bool = False
    ) -> str:
        match event_type:
            case WebhookEventType.SUBSCRIPTION_AUTHENTICATED.value:
                return await self._handle_subscription_authenticated(payload, replay=replay)
            case WebhookEventType.SUBSCRIPTION_ACTIVATED.value:
                return await self._handle_subscription_activated(payload, replay=replay)
            case WebhookEventType.SUBSCRIPTION_CHARGED.value:
                return await self._handle_subscription_charged(payload)
            case WebhookEventType.SUBSCRIPTION_PENDING.value:
                return await self._handle_subscription_pending(payload)
            case WebhookEventType.SUBSCRIPTION_HALTED.value:
                return await self._handle_subscription_halted(payload)
            case WebhookEventType.SUBSCRIPTION_CANCELLED.value:
                return await self._handle_subscription_cancelled(payload)
            case WebhookEventType.SUBSCRIPTION_PAUSED.value:
                return await self._handle_subscription_paused(payload)
            case WebhookEventType.SUBSCRIPTION_RESUMED.value:
                return await self._handle_subscription_resumed(payload)
            case WebhookEventType.PAYMENT_CAPTURED.value:
                return await self._handle_payment_captured(payload, replay=replay)
            case WebhookEventType.PAYMENT_FAILED.value:
                return await self._handle_payment_failed(payload)
            case WebhookEventType.REFUND_PROCESSED.value:
                return await self._handle_refund_processed(payload)
            case WebhookEventType.DISPUTE_CREATED.value:
                return await self._handle_dispute_created(payload)
            case _:
                logger.bind(operation="webhook", event_type=event_type).warning(
                    "Unhandled webhook event type"
                )
                return "skipped"

    @staticmethod
    def _entity(payload: dict[str, object], key: str) -> dict[str, Any]:
        section = payload.get(key)
        if isinstance(section, dict):
            entity = section.get("entity")
            if isinstance(entity, dict):
                return cast("dict[str, Any]", entity)
        return {}

    async def _find_subscription_by_entity(self, entity: dict[str, Any]) -> Subscription | None:
        rz_id = entity.get("subscription_id") or entity.get("id")
        if not isinstance(rz_id, str):
            notes = entity.get("notes")
            if isinstance(notes, dict):
                rz_id = notes.get("subscription_id")
        if not isinstance(rz_id, str):
            return None
        result = await self.repos.subscriptions.find_by_razorpay_id(rz_id)
        if isinstance(result, Failure):
            return None
        return result.unwrap()

    @staticmethod
    def _skipped(current: str, expected: set[str]) -> str | None:
        """Replay guard: return 'skipped' when the effect is already applied."""
        return "skipped" if current in expected else None

    async def _handle_subscription_authenticated(
        self, payload: dict[str, object], *, replay: bool
    ) -> str:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return "skipped"
        if replay:
            skip = self._skipped(
                subscription.status,
                {SubscriptionStatus.AUTHENTICATED.value, SubscriptionStatus.ACTIVE.value},
            )
            if skip is not None:
                return skip
        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.AUTHENTICATED,
            expected_version=subscription.version,
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)
        else:
            await self._audit(update.unwrap(), AuditAction.SUBSCRIPTION_AUTHENTICATED.value)
        return "processed"

    async def _handle_subscription_activated(
        self, payload: dict[str, object], *, replay: bool
    ) -> str:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return "skipped"
        if replay:
            skip = self._skipped(
                subscription.status,
                {SubscriptionStatus.ACTIVE.value, SubscriptionStatus.CANCELLED.value},
            )
            if skip is not None:
                return skip
        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={
                "current_period_start": self._parse_datetime(entity.get("current_start")),
                "current_period_end": self._parse_datetime(entity.get("current_end")),
            },
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)
        else:
            await self._audit(update.unwrap(), AuditAction.SUBSCRIPTION_ACTIVATED.value)
        return "processed"

    async def _handle_subscription_charged(self, payload: dict[str, object]) -> str:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return "skipped"
        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={
                "current_period_start": self._parse_datetime(entity.get("current_start")),
                "current_period_end": self._parse_datetime(entity.get("current_end")),
            },
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)
        else:
            await self._audit(update.unwrap(), AuditAction.SUBSCRIPTION_ACTIVATED.value)
        return "processed"

    async def _handle_subscription_pending(self, payload: dict[str, object]) -> str:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return "skipped"
        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.PAST_DUE,
            expected_version=subscription.version,
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)
        return "processed"

    async def _handle_subscription_halted(self, payload: dict[str, object]) -> str:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return "skipped"
        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.HALTED,
            expected_version=subscription.version,
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)
        else:
            await self._audit(update.unwrap(), AuditAction.SUBSCRIPTION_HALTED.value)
        return "processed"

    async def _handle_subscription_cancelled(self, payload: dict[str, object]) -> str:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return "skipped"
        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.CANCELLED,
            expected_version=subscription.version,
            extra_values={
                "cancelled_at": datetime.now(tz=UTC),
                "ended_at": datetime.now(tz=UTC),
            },
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)
        else:
            await self._audit(update.unwrap(), AuditAction.SUBSCRIPTION_CANCELLED.value)
        return "processed"

    async def _handle_subscription_paused(self, payload: dict[str, object]) -> str:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return "skipped"
        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.PAUSED,
            expected_version=subscription.version,
            extra_values={"pause_start": datetime.now(tz=UTC)},
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)
        return "processed"

    async def _handle_subscription_resumed(self, payload: dict[str, object]) -> str:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return "skipped"
        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={"pause_start": None, "pause_end": None},
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)
        return "processed"

    async def _handle_payment_captured(
        self, payload: dict[str, object], *, replay: bool
    ) -> str:
        entity = self._entity(payload, "payment")
        rz_payment_id = entity.get("id")
        subscription = await self._find_subscription_by_entity(entity)
        if not isinstance(rz_payment_id, str) or subscription is None:
            logger.bind(operation="webhook").warning(
                "payment.captured without resolvable subscription",
                payment_id=rz_payment_id,
            )
            return "skipped"
        if replay and subscription.status in _TERMINAL_PAYMENT_STATES:
            return "skipped"

        payment = await self.payments.record_payment(
            PaymentRecordDTO(
                razorpay_payment_id=rz_payment_id,
                subscription_id=str(subscription.id),
                amount=int(entity.get("amount") or 0),
                currency=str(entity.get("currency") or "INR"),
                method=entity.get("method"),
                razorpay_order_id=entity.get("order_id"),
                captured_at=self._parse_datetime(entity.get("created_at")),
            ),
            subscription=subscription,
        )
        plan_result = await self.repos.plans.find_by_id(subscription.plan_id)
        if isinstance(plan_result, Failure):
            logger.bind(operation="webhook").warning(plan_result.failure().message)
            return "processed"
        plan = plan_result.unwrap()
        if plan is None:
            logger.bind(operation="webhook").warning("Plan not found for subscription")
            return "processed"

        await self.invoices.generate_for_payment(payment, subscription, plan)
        await self.invoices.generate_receipt_for_payment(payment, subscription, plan)

        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={"retry_count": 0},
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)

        await self._audit(payment, AuditAction.PAYMENT_CAPTURED.value)
        return "processed"

    async def _handle_payment_failed(self, payload: dict[str, object]) -> str:
        entity = self._entity(payload, "payment")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return "skipped"
        rz_payment_id = entity.get("id")
        await self.payments.record_failed_payment(
            razorpay_payment_id=str(rz_payment_id or ""),
            subscription_id=str(subscription.id),
            error_code=str(entity.get("error_code") or ""),
            error_description=str(entity.get("error_description") or ""),
        )
        update = await self.repos.subscriptions.update_status(
            subscription,
            SubscriptionStatus.PAST_DUE,
            expected_version=subscription.version,
        )
        if isinstance(update, Failure):
            logger.bind(operation="webhook").warning(update.failure().message)
        else:
            await self._audit(update.unwrap(), AuditAction.PAYMENT_FAILED.value)
        return "processed"

    async def _handle_refund_processed(self, payload: dict[str, object]) -> str:
        entity = self._entity(payload, "refund")
        rz_payment_id = entity.get("payment_id")
        amount = entity.get("amount")
        if not isinstance(rz_payment_id, str) or not isinstance(amount, (int, float)):
            return "skipped"
        await self.payments.handle_refund_processed(
            razorpay_payment_id=rz_payment_id, refund_paisa=int(amount)
        )
        return "processed"

    async def _handle_dispute_created(self, payload: dict[str, object]) -> str:
        entity = self._entity(payload, "dispute")
        rz_payment_id = entity.get("payment_id")
        dispute_id = entity.get("id")
        reason = entity.get("reason")
        if not isinstance(rz_payment_id, str) or not isinstance(dispute_id, str):
            return "skipped"
        await self.payments.handle_chargeback(
            razorpay_payment_id=rz_payment_id,
            dispute_id=dispute_id,
            reason=str(reason or ""),
        )
        return "processed"

    async def _audit(self, entity: object, action: str) -> None:
        entity_type = type(entity).__name__.lower()
        await self.repos.audit.create(
            AuditLog(
                entity_type=entity_type,
                entity_id=str(getattr(entity, "id", "")),
                action=action,
            )
        )

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=UTC)
        return None
