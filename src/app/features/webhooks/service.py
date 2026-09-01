"""Razorpay webhook processing: signature verification + idempotent dispatch."""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from returns.result import Failure, Success

from app.config import get_settings
from app.features.audit.model import AuditAction, AuditLog
from app.features.payments.dto import PaymentRecordDTO
from app.features.subscriptions.model import SubscriptionStatus
from app.features.webhooks.model import (
    WebhookEvent,
    WebhookEventStatus,
    WebhookEventType,
)
from app.utils import logger

from .errors import (
    WebhookCollaboratorError,
    WebhookValidationError,
    WebhookVerificationError,
)

if TYPE_CHECKING:
    from typing import Any

    from app.features.audit.repository import AuditLogRepository
    from app.features.invoices.service import InvoiceService
    from app.features.payments.service import PaymentService
    from app.features.plans.repository import PlanRepository
    from app.features.subscriptions.model import Subscription
    from app.features.subscriptions.repository import SubscriptionRepository
    from app.features.webhooks.repository import WebhookEventRepository
    from app.shared.result.errors import FeatureError

    from .errors import WebhookResult

_IGNORED_EVENTS = frozenset(
    {
        WebhookEventType.PAYMENT_AUTHORIZED.value,
        WebhookEventType.REFUND_CREATED.value,
    }
)

_TERMINAL_PAYMENT_STATES = frozenset({SubscriptionStatus.CANCELLED.value})


def _collaborator_error(error: FeatureError) -> WebhookCollaboratorError:
    return WebhookCollaboratorError(message=error.message, details=error.details)


class WebhookService:
    """Verify, log, and dispatch Razorpay webhooks with idempotency."""

    def __init__(  # noqa: PLR0917
        self,
        webhooks: WebhookEventRepository,
        subscriptions: SubscriptionRepository,
        plans: PlanRepository,
        audit: AuditLogRepository,
        payment_service: PaymentService,
        invoice_service: InvoiceService,
    ) -> None:
        self.webhooks = webhooks
        self.subscriptions = subscriptions
        self.plans = plans
        self.audit = audit
        self.payment_service = payment_service
        self.invoice_service = invoice_service

    @staticmethod
    def verify_signature(*, raw_body: str, signature: str) -> WebhookResult[None]:
        secret = get_settings().RAZORPAY_WEBHOOK_SECRET.get_secret_value()
        if not secret:
            return Failure(
                WebhookVerificationError(message="Razorpay webhook secret is not configured")
            )
        expected = hmac.new(secret.encode(), raw_body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return Failure(WebhookVerificationError(message="Signature mismatch"))
        return Success(None)

    async def process(  # noqa: PLR0912
        self, *, event_id: str, event_type: str, payload: dict[str, object]
    ) -> WebhookResult[bool]:
        """Idempotently process a verified webhook event. Returns True if handled."""
        existing = await self.webhooks.find_by_razorpay_event_id(event_id)
        if isinstance(existing, Failure):
            return existing
        previous = existing.unwrap()
        if previous is not None:
            if previous.status in {
                WebhookEventStatus.PROCESSED.value,
                WebhookEventStatus.SKIPPED.value,
            }:
                return Success(True)
            if previous.status == WebhookEventStatus.FAILED.value:
                # A previous attempt failed; allow re-processing of the delivery.
                pass
            else:
                return Failure(WebhookValidationError(message="Webhook event already in flight"))

        if event_type in _IGNORED_EVENTS:
            if previous is None:
                event = WebhookEvent(
                    razorpay_event_id=event_id,
                    event_type=event_type,
                    status=WebhookEventStatus.SKIPPED.value,
                    payload=payload,
                )
                created = await self.webhooks.create(event)
                if isinstance(created, Failure):
                    return created
            return Success(False)

        if previous is None:
            event = WebhookEvent(
                razorpay_event_id=event_id,
                event_type=event_type,
                status=WebhookEventStatus.PENDING.value,
                payload=payload,
            )
            created = await self.webhooks.create(event)
            if isinstance(created, Failure):
                return created
            event = created.unwrap()
        else:
            event = previous

        processing = await self.webhooks.update_status(
            event, status=WebhookEventStatus.PROCESSING.value
        )
        if isinstance(processing, Failure):
            return processing
        event = processing.unwrap()

        dispatched = await self._dispatch(event_type, payload)

        if isinstance(dispatched, Failure):
            error = dispatched.failure()
            update = await self.webhooks.update_status(
                event,
                status=WebhookEventStatus.FAILED.value,
                extra_values={
                    "failed_at": datetime.now(tz=UTC),
                    "error_message": error.message[:1000],
                    "retry_count": event.retry_count + 1,
                },
            )
            if isinstance(update, Failure):
                return update
            return dispatched
        dispatch_status = dispatched.unwrap()

        status = (
            WebhookEventStatus.SKIPPED.value
            if dispatch_status == "skipped"
            else WebhookEventStatus.PROCESSED.value
        )
        update = await self.webhooks.update_status(
            event,
            status=status,
            extra_values={"processed_at": datetime.now(tz=UTC)},
        )
        if isinstance(update, Failure):
            return update
        return Success(True)

    async def replay(self, event_id: str) -> WebhookResult[WebhookEvent]:
        """Re-process a FAILED webhook event (Requirement 22/31)."""
        result = await self.webhooks.find_by_id(event_id)
        if isinstance(result, Failure):
            return result
        event = result.unwrap()
        if event is None:
            return Failure(WebhookValidationError(message="Webhook event not found"))
        if event.status != WebhookEventStatus.FAILED.value:
            return Failure(
                WebhookValidationError(message=f"Cannot replay event in status '{event.status}'")
            )
        if event.event_type in _IGNORED_EVENTS:
            return Failure(WebhookValidationError(message="Event type is intentionally ignored"))

        processing = await self.webhooks.update_status(
            event, status=WebhookEventStatus.PROCESSING.value
        )
        if isinstance(processing, Failure):
            return processing
        event = processing.unwrap()

        dispatched = await self._dispatch(event.event_type, event.payload, replay=True)
        if isinstance(dispatched, Failure):
            return dispatched
        dispatch_status = dispatched.unwrap()
        status = (
            WebhookEventStatus.SKIPPED.value
            if dispatch_status == "skipped"
            else WebhookEventStatus.PROCESSED.value
        )
        updated = await self.webhooks.update_status(
            event,
            status=status,
            extra_values={"processed_at": datetime.now(tz=UTC)},
        )
        if isinstance(updated, Failure):
            return updated
        return Success(updated.unwrap())

    async def _dispatch(  # noqa: PLR0912
        self, event_type: str, payload: dict[str, object], *, replay: bool = False
    ) -> WebhookResult[str]:
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
                return Success("skipped")

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
        result = await self.subscriptions.find_by_razorpay_id(rz_id)
        if isinstance(result, Failure):
            return None
        return result.unwrap()

    @staticmethod
    def _skipped(current: str, expected: set[str]) -> str | None:
        """Replay guard: return 'skipped' when the effect is already applied."""
        return "skipped" if current in expected else None

    async def _handle_subscription_authenticated(
        self, payload: dict[str, object], *, replay: bool
    ) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        if replay:
            skip = self._skipped(
                subscription.status,
                {SubscriptionStatus.AUTHENTICATED.value, SubscriptionStatus.ACTIVE.value},
            )
            if skip is not None:
                return Success(skip)
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.AUTHENTICATED,
            expected_version=subscription.version,
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))
        audit_result = await self._audit(
            update.unwrap(), AuditAction.SUBSCRIPTION_AUTHENTICATED.value
        )
        if isinstance(audit_result, Failure):
            return audit_result
        return Success("processed")

    async def _handle_subscription_activated(
        self, payload: dict[str, object], *, replay: bool
    ) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        if replay:
            skip = self._skipped(
                subscription.status,
                {SubscriptionStatus.ACTIVE.value, SubscriptionStatus.CANCELLED.value},
            )
            if skip is not None:
                return Success(skip)
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={
                "current_period_start": self._parse_datetime(entity.get("current_start")),
                "current_period_end": self._parse_datetime(entity.get("current_end")),
            },
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))
        audit_result = await self._audit(update.unwrap(), AuditAction.SUBSCRIPTION_ACTIVATED.value)
        if isinstance(audit_result, Failure):
            return audit_result
        return Success("processed")

    async def _handle_subscription_charged(self, payload: dict[str, object]) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={
                "current_period_start": self._parse_datetime(entity.get("current_start")),
                "current_period_end": self._parse_datetime(entity.get("current_end")),
            },
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))
        audit_result = await self._audit(update.unwrap(), AuditAction.SUBSCRIPTION_ACTIVATED.value)
        if isinstance(audit_result, Failure):
            return audit_result
        return Success("processed")

    async def _handle_subscription_pending(self, payload: dict[str, object]) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.PAST_DUE,
            expected_version=subscription.version,
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))
        return Success("processed")

    async def _handle_subscription_halted(self, payload: dict[str, object]) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.HALTED,
            expected_version=subscription.version,
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))
        audit_result = await self._audit(update.unwrap(), AuditAction.SUBSCRIPTION_HALTED.value)
        if isinstance(audit_result, Failure):
            return audit_result
        return Success("processed")

    async def _handle_subscription_cancelled(
        self, payload: dict[str, object]
    ) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.CANCELLED,
            expected_version=subscription.version,
            extra_values={
                "cancelled_at": datetime.now(tz=UTC),
                "ended_at": datetime.now(tz=UTC),
            },
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))
        audit_result = await self._audit(update.unwrap(), AuditAction.SUBSCRIPTION_CANCELLED.value)
        if isinstance(audit_result, Failure):
            return audit_result
        return Success("processed")

    async def _handle_subscription_paused(self, payload: dict[str, object]) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.PAUSED,
            expected_version=subscription.version,
            extra_values={"pause_start": datetime.now(tz=UTC)},
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))
        return Success("processed")

    async def _handle_subscription_resumed(self, payload: dict[str, object]) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={"pause_start": None, "pause_end": None},
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))
        return Success("processed")

    async def _handle_payment_captured(
        self, payload: dict[str, object], *, replay: bool
    ) -> WebhookResult[str]:
        entity = self._entity(payload, "payment")
        rz_payment_id = entity.get("id")
        subscription = await self._find_subscription_by_entity(entity)
        if not isinstance(rz_payment_id, str) or subscription is None:
            logger.bind(operation="webhook").warning(
                "payment.captured without resolvable subscription",
                payment_id=rz_payment_id,
            )
            return Success("skipped")
        if replay and subscription.status in _TERMINAL_PAYMENT_STATES:
            return Success("skipped")

        payment_result = await self.payment_service.record_payment(
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
        if isinstance(payment_result, Failure):
            return Failure(_collaborator_error(payment_result.failure()))
        payment = payment_result.unwrap()
        plan_result = await self.plans.find_by_id(subscription.plan_id)
        if isinstance(plan_result, Failure):
            return Failure(_collaborator_error(plan_result.failure()))
        plan = plan_result.unwrap()
        if plan is None:
            return Failure(WebhookValidationError(message="Plan not found for subscription"))

        invoice_result = await self.invoice_service.generate_for_payment(
            payment, subscription, plan
        )
        if isinstance(invoice_result, Failure):
            return Failure(_collaborator_error(invoice_result.failure()))
        receipt_result = await self.invoice_service.generate_receipt_for_payment(
            payment, subscription, plan
        )
        if isinstance(receipt_result, Failure):
            return Failure(_collaborator_error(receipt_result.failure()))

        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.ACTIVE,
            expected_version=subscription.version,
            extra_values={"retry_count": 0},
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))

        audit_result = await self._audit(payment, AuditAction.PAYMENT_CAPTURED.value)
        if isinstance(audit_result, Failure):
            return audit_result
        return Success("processed")

    async def _handle_payment_failed(self, payload: dict[str, object]) -> WebhookResult[str]:
        entity = self._entity(payload, "payment")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        rz_payment_id = entity.get("id")
        payment_result = await self.payment_service.record_failed_payment(
            razorpay_payment_id=str(rz_payment_id or ""),
            subscription_id=str(subscription.id),
            error_code=str(entity.get("error_code") or ""),
            error_description=str(entity.get("error_description") or ""),
        )
        if isinstance(payment_result, Failure):
            return Failure(_collaborator_error(payment_result.failure()))
        update = await self.subscriptions.update_status(
            subscription,
            SubscriptionStatus.PAST_DUE,
            expected_version=subscription.version,
        )
        if isinstance(update, Failure):
            return Failure(_collaborator_error(update.failure()))
        audit_result = await self._audit(update.unwrap(), AuditAction.PAYMENT_FAILED.value)
        if isinstance(audit_result, Failure):
            return audit_result
        return Success("processed")

    async def _handle_refund_processed(self, payload: dict[str, object]) -> WebhookResult[str]:
        entity = self._entity(payload, "refund")
        rz_payment_id = entity.get("payment_id")
        amount = entity.get("amount")
        if not isinstance(rz_payment_id, str) or not isinstance(amount, (int, float)):
            return Success("skipped")
        result = await self.payment_service.handle_refund_processed(
            razorpay_payment_id=rz_payment_id, refund_paisa=int(amount)
        )
        if isinstance(result, Failure):
            return Failure(_collaborator_error(result.failure()))
        return Success("processed")

    async def _handle_dispute_created(self, payload: dict[str, object]) -> WebhookResult[str]:
        entity = self._entity(payload, "dispute")
        rz_payment_id = entity.get("payment_id")
        dispute_id = entity.get("id")
        reason = entity.get("reason")
        if not isinstance(rz_payment_id, str) or not isinstance(dispute_id, str):
            return Success("skipped")
        result = await self.payment_service.handle_chargeback(
            razorpay_payment_id=rz_payment_id,
            dispute_id=dispute_id,
            reason=str(reason or ""),
        )
        if isinstance(result, Failure):
            return Failure(_collaborator_error(result.failure()))
        return Success("processed")

    async def _audit(self, entity: object, action: str) -> WebhookResult[None]:
        entity_type = type(entity).__name__.lower()
        result = await self.audit.create(
            AuditLog(
                entity_type=entity_type,
                entity_id=str(getattr(entity, "id", "")),
                action=action,
            )
        )
        if isinstance(result, Failure):
            return Failure(_collaborator_error(result.failure()))
        return Success(None)

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=UTC)
        return None
