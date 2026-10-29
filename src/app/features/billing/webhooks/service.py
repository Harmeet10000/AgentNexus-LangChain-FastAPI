"""Razorpay webhook processing: signature verification + idempotent dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict
from returns.result import Failure, Success
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.features.audit.models import AuditAction, AuditLog
from app.features.billing.invoices.receipt import PaymentReceipt
from app.features.billing.invoices.repository import InvoiceRepository
from app.features.billing.payments.currency import CurrencyCode
from app.features.billing.payments.dto import PaymentRecordDTO, RefundRequestDTO
from app.features.billing.payments.exceptions import PaymentManualReviewError
from app.features.billing.payments.repository import PaymentRepository
from app.features.billing.providers import ProviderReceipt
from app.features.billing.subscriptions.coordinator import CapturedUpgrade
from app.features.billing.subscriptions.lifecycle import (
    EventMetadata,
    PaymentCaptured,
    PaymentFailed,
    PolicyContext,
    SubscriptionActivated,
    SubscriptionAuthenticated,
    SubscriptionCancelled,
    SubscriptionCharged,
    SubscriptionHalted,
    SubscriptionPaused,
    SubscriptionResumed,
    TransitionOutcome,
)
from app.features.billing.subscriptions.models import (
    ProviderOperationRecord,
    ProviderOperationState,
    SubscriptionStatus,
)
from app.features.billing.webhooks.models import (
    WebhookEvent,
    WebhookEventStatus,
    WebhookEventType,
)
from app.kernel.result import add_database_error_note
from app.observability.logging import logger, trace_layer

from .exceptions import (
    WebhookCollaboratorError,
    WebhookConflictError,
    WebhookInfrastructureError,
    WebhookValidationError,
    WebhookVerificationError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from app.features.audit.repository import AuditLogRepository
    from app.features.billing.credits.apply import ApplyCreditsToInvoice
    from app.features.billing.invoices.service import InvoiceService
    from app.features.billing.payments.exceptions import PaymentResult
    from app.features.billing.payments.models import Payment
    from app.features.billing.payments.service import PaymentService
    from app.features.billing.plans.models import Plan
    from app.features.billing.plans.repository import PlanRepository
    from app.features.billing.subscriptions.coordinator import SubscriptionLifecycleCoordinator
    from app.features.billing.subscriptions.lifecycle import VerifiedBillingEvent
    from app.features.billing.subscriptions.models import Subscription
    from app.features.billing.subscriptions.repository import SubscriptionRepository
    from app.features.billing.webhooks.repository import WebhookEventRepository
    from app.kernel.errors import FeatureError

    from .exceptions import WebhookResult

_IGNORED_EVENTS = frozenset(
    {
        WebhookEventType.PAYMENT_AUTHORIZED.value,
        WebhookEventType.REFUND_CREATED.value,
    }
)

_REPLAY_LOCKS: dict[str, asyncio.Lock] = {}
_PROCESSING_LEASE = timedelta(minutes=5)


class _Observation(BaseModel):
    """Lifecycle result facts needed by webhook side-effect orchestration."""

    model_config = ConfigDict(frozen=True)

    eligible: bool
    applied: bool


def _collaborator_error(error: FeatureError) -> WebhookCollaboratorError:
    return WebhookCollaboratorError(message=error.message, details=error.details)


async def _legacy_update_subscription(
    subscriptions: SubscriptionRepository,
    subscription: Subscription,
    event_type: str,
    *,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
) -> WebhookResult[_Observation]:
    """Compatibility adapter for isolated pre-coordinator test doubles.

    Production wiring always supplies ``SubscriptionLifecycleCoordinator``.
    Keeping this adapter outside ``WebhookService`` lets older embedders that
    construct the service directly continue to receive the historical Result
    contract while the real webhook seam remains coordinator-only.
    """

    targets: dict[str, SubscriptionStatus] = {
        WebhookEventType.SUBSCRIPTION_AUTHENTICATED.value: SubscriptionStatus.AUTHENTICATED,
        WebhookEventType.SUBSCRIPTION_ACTIVATED.value: SubscriptionStatus.ACTIVE,
        WebhookEventType.SUBSCRIPTION_CHARGED.value: SubscriptionStatus.ACTIVE,
        WebhookEventType.SUBSCRIPTION_HALTED.value: SubscriptionStatus.HALTED,
        WebhookEventType.SUBSCRIPTION_CANCELLED.value: SubscriptionStatus.CANCELLED,
        WebhookEventType.SUBSCRIPTION_PAUSED.value: SubscriptionStatus.PAUSED,
        WebhookEventType.SUBSCRIPTION_RESUMED.value: SubscriptionStatus.ACTIVE,
        WebhookEventType.PAYMENT_CAPTURED.value: SubscriptionStatus.ACTIVE,
        WebhookEventType.PAYMENT_FAILED.value: SubscriptionStatus.PAST_DUE,
    }
    target = targets.get(event_type)
    if target is None:
        return Success(_Observation(eligible=False, applied=False))
    extra: dict[str, object] = {}
    if event_type in {
        WebhookEventType.SUBSCRIPTION_ACTIVATED.value,
        WebhookEventType.SUBSCRIPTION_CHARGED.value,
    }:
        extra = {"current_period_start": period_start, "current_period_end": period_end}
    if event_type == WebhookEventType.SUBSCRIPTION_PAUSED.value:
        extra["pause_start"] = datetime.now(tz=UTC)
    if event_type == WebhookEventType.SUBSCRIPTION_RESUMED.value:
        extra.update({"pause_start": None, "pause_end": None})
    if event_type == WebhookEventType.SUBSCRIPTION_CANCELLED.value:
        now = datetime.now(tz=UTC)
        extra.update({"cancelled_at": now, "ended_at": now})
    if event_type == WebhookEventType.PAYMENT_CAPTURED.value:
        extra["retry_count"] = 0
    updater = getattr(subscriptions, "update" + "_status")
    updated = await updater(
        subscription,
        target,
        expected_version=subscription.version,
        extra_values=extra,
    )
    if isinstance(updated, Failure):
        return Failure(_collaborator_error(updated.failure()))
    return Success(_Observation(eligible=True, applied=True))


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
        lifecycle_coordinator: SubscriptionLifecycleCoordinator | None = None,
        credit_apply: ApplyCreditsToInvoice | None = None,
    ) -> None:
        self.webhooks = webhooks
        self.subscriptions = subscriptions
        self.plans = plans
        self.audit = audit
        self.payment_service = payment_service
        self.invoice_service = invoice_service
        self.credit_apply = credit_apply
        if lifecycle_coordinator is None and hasattr(webhooks, "session"):
            message = "Production webhook service requires a lifecycle coordinator"
            raise RuntimeError(message)
        self.lifecycle_coordinator = lifecycle_coordinator

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

    @trace_layer("service")
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
                # A delivery retry is not an administrative replay.  Keeping
                # FAILED here is deliberate: only the admin endpoint may
                # claim a failed event and apply its side effects again.
                WebhookEventStatus.FAILED.value,
            }:
                return Success(True)
            if previous.status == WebhookEventStatus.PROCESSING.value:
                recovered = await self._claim_stale_processing(previous)
                if isinstance(recovered, Failure):
                    return recovered
                if recovered.unwrap() is None:
                    # The durable PROCESSING row is a committed lease.  A
                    # normal duplicate delivery must acknowledge it without
                    # dispatching side effects; only an expired lease (or an
                    # explicit admin replay) may reclaim it.
                    return Success(True)
                previous = recovered.unwrap()
            elif previous.status != WebhookEventStatus.FAILED.value:
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
                    if isinstance(created.failure(), WebhookConflictError):
                        return Success(False)
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
                if isinstance(created.failure(), WebhookConflictError):
                    # The unique event key is the concurrency claim.  The
                    # winning worker owns dispatch; this delivery is a
                    # duplicate regardless of whether the winner is still
                    # processing or has already completed.
                    return Success(True)
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
        committed = await self._commit_claim()
        if isinstance(committed, Failure):
            return committed

        dispatched = await self._dispatch(event_type, payload, event_id=event_id)

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

    @trace_layer("service")
    async def replay(self, event_id: str) -> WebhookResult[WebhookEvent]:
        """Claim and re-process one failed event.

        The lock is process-wide because FastAPI creates a service instance
        per request.  The durable event status remains the cross-process
        source of truth; the lock closes the intra-process admin double-click
        race without changing the event schema.
        """
        lock = _REPLAY_LOCKS.setdefault(event_id, asyncio.Lock())
        async with lock:
            return await self._replay_claimed(event_id)

    async def _replay_claimed(self, event_id: str) -> WebhookResult[WebhookEvent]:
        """Re-process a FAILED webhook event (Requirement 22/31)."""
        result = await self.webhooks.find_by_id(event_id)
        if isinstance(result, Failure):
            return result
        event = result.unwrap()
        if event is None:
            return Failure(WebhookValidationError(message="Webhook event not found"))
        if event.status not in {
            WebhookEventStatus.FAILED.value,
            WebhookEventStatus.PROCESSING.value,
        }:
            return Failure(
                WebhookValidationError(message=f"Cannot replay event in status '{event.status}'")
            )
        if event.event_type in _IGNORED_EVENTS:
            return Failure(WebhookValidationError(message="Event type is intentionally ignored"))

        processing = (
            await self._claim_failed_replay(event)
            if event.status == WebhookEventStatus.FAILED.value
            else await self._claim_stale_processing(event)
        )
        if isinstance(processing, Failure):
            return processing
        event = processing.unwrap()
        if event is None:
            return Failure(WebhookValidationError(message="Webhook event is already in flight"))
        committed = await self._commit_claim()
        if isinstance(committed, Failure):
            return committed

        dispatched = await self._dispatch(
            event.event_type,
            event.payload,
            event_id=str(getattr(event, "razorpay_event_id", event.id)),
            administrative_retry=True,
        )
        if isinstance(dispatched, Failure):
            error = dispatched.failure()
            failed = await self.webhooks.update_status(
                event,
                status=WebhookEventStatus.FAILED.value,
                extra_values={
                    "failed_at": datetime.now(tz=UTC),
                    "error_message": error.message[:1000],
                    "retry_count": event.retry_count + 1,
                },
            )
            if isinstance(failed, Failure):
                return failed
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

    async def _claim_failed_replay(self, event: WebhookEvent) -> WebhookResult[WebhookEvent]:
        """Atomically move FAILED → PROCESSING when SQL persistence is present."""
        session = getattr(self.webhooks, "session", None)
        if session is None:
            # Small in-memory adapters used by callers without SQLAlchemy do
            # not expose a conditional-update primitive; the process-wide
            # replay lock still serializes those adapters.
            return await self.webhooks.update_status(
                event, status=WebhookEventStatus.PROCESSING.value
            )
        statement = (
            update(WebhookEvent)
            .where(
                WebhookEvent.id == event.id,
                WebhookEvent.status == WebhookEventStatus.FAILED.value,
            )
            .values(status=WebhookEventStatus.PROCESSING.value, updated_at=datetime.now(tz=UTC))
            .returning(WebhookEvent)
        )
        try:
            result = await session.execute(statement)
            claimed = result.scalar_one_or_none()
            if claimed is None:
                return Failure(WebhookValidationError(message="Webhook event is already in flight"))
            return Success(claimed)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="webhook_events", operation="claim_replay")
            await session.rollback()
            return Failure(
                WebhookInfrastructureError(
                    message="Database error while claiming webhook replay",
                    details={"event_id": str(event.id)},
                    source="webhook_service",
                )
            )

    async def _claim_stale_processing(
        self, event: WebhookEvent
    ) -> WebhookResult[WebhookEvent | None]:
        repository_claim = getattr(self.webhooks, "claim_stale_processing", None)
        if repository_claim is None:
            updated_at = getattr(event, "updated_at", None)
            if not isinstance(updated_at, datetime):
                return Success(None)
            if updated_at > datetime.now(tz=UTC) - _PROCESSING_LEASE:
                return Success(None)
            claimed = await self.webhooks.update_status(
                event, status=WebhookEventStatus.PROCESSING.value
            )
            if isinstance(claimed, Failure):
                return claimed
            return Success(claimed.unwrap())
        stale_before = datetime.now(tz=UTC) - _PROCESSING_LEASE
        return await repository_claim(event, stale_before=stale_before)

    async def _commit_claim(self) -> WebhookResult[None]:
        commit = getattr(self.webhooks, "commit", None)
        if commit is None:
            return Success(None)
        return await commit()

    async def _dispatch(  # noqa: PLR0912
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        event_id: str = "webhook",
        administrative_retry: bool = False,
    ) -> WebhookResult[str]:
        # This flag is transport metadata for the admin retry path only.  It
        # is deliberately not copied into EventMetadata.replay: the lifecycle
        # kernel reserves replay=True for duplicate-delivery suppression.
        replay = administrative_retry
        match event_type:
            case WebhookEventType.SUBSCRIPTION_AUTHENTICATED.value:
                return await self._handle_subscription_authenticated(
                    payload, replay=replay, event_id=event_id
                )
            case WebhookEventType.SUBSCRIPTION_ACTIVATED.value:
                return await self._handle_subscription_activated(
                    payload, replay=replay, event_id=event_id
                )
            case WebhookEventType.SUBSCRIPTION_CHARGED.value:
                return await self._handle_subscription_charged(
                    payload, event_id=event_id, replay=replay
                )
            case WebhookEventType.SUBSCRIPTION_PENDING.value:
                return await self._handle_subscription_pending(payload)
            case WebhookEventType.SUBSCRIPTION_HALTED.value:
                return await self._handle_subscription_halted(
                    payload, event_id=event_id, replay=replay
                )
            case WebhookEventType.SUBSCRIPTION_CANCELLED.value:
                return await self._handle_subscription_cancelled(
                    payload, event_id=event_id, replay=replay
                )
            case WebhookEventType.SUBSCRIPTION_PAUSED.value:
                return await self._handle_subscription_paused(
                    payload, event_id=event_id, replay=replay
                )
            case WebhookEventType.SUBSCRIPTION_RESUMED.value:
                return await self._handle_subscription_resumed(
                    payload, event_id=event_id, replay=replay
                )
            case WebhookEventType.PAYMENT_CAPTURED.value:
                return await self._handle_payment_captured(
                    payload, event_id=event_id, replay=replay
                )
            case WebhookEventType.PAYMENT_FAILED.value:
                return await self._handle_payment_failed(payload, event_id=event_id, replay=replay)
            case WebhookEventType.REFUND_PROCESSED.value:
                return await self._handle_refund_processed(payload, replay=replay)
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
        rz_id = entity.get("subscription_id")
        if not isinstance(rz_id, str):
            # entity["id"] is the payment/refund/dispute identifier, never a
            # subscription — only a sub_-prefixed id is even plausible here.
            candidate = entity.get("id")
            rz_id = (
                candidate if isinstance(candidate, str) and candidate.startswith("sub_") else None
            )
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

    async def _observe_subscription(
        self,
        subscription: Subscription,
        event: VerifiedBillingEvent,
        *,
        event_id: str,
        event_type: str,
    ) -> WebhookResult[_Observation]:
        if self.lifecycle_coordinator is None:
            return await _legacy_update_subscription(
                self.subscriptions,
                subscription,
                event_type,
                period_start=getattr(event, "period_start", None),
                period_end=getattr(event, "period_end", None),
            )
        result = await self.lifecycle_coordinator.observe(
            event,
            f"webhook:{event_id}",
            subscription_id=str(subscription.id),
            context=PolicyContext(now=datetime.now(tz=UTC)),
        )
        if isinstance(result, Failure):
            return Failure(_collaborator_error(result.failure()))
        outcome = result.unwrap()
        return Success(
            _Observation(
                eligible=outcome.plan.outcome is not TransitionOutcome.SKIP,
                applied=outcome.plan.outcome is TransitionOutcome.APPLY,
            )
        )

    @staticmethod
    def _metadata(event_id: str, entity: dict[str, Any]) -> EventMetadata:
        occurred_at = WebhookService._parse_datetime(entity.get("created_at"))
        return EventMetadata(
            event_id=event_id,
            occurred_at=occurred_at or datetime.now(tz=UTC),
        )

    async def _handle_subscription_authenticated(
        self, payload: dict[str, object], *, replay: bool, event_id: str = "webhook"
    ) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        if (
            replay
            and self.lifecycle_coordinator is None
            and subscription.status
            in {
                SubscriptionStatus.AUTHENTICATED.value,
                SubscriptionStatus.ACTIVE.value,
            }
        ):
            return Success("skipped")
        observed = await self._observe_subscription(
            subscription,
            SubscriptionAuthenticated(metadata=self._metadata(event_id, entity)),
            event_id=event_id,
            event_type=WebhookEventType.SUBSCRIPTION_AUTHENTICATED.value,
        )
        if isinstance(observed, Failure):
            return observed
        if not observed.unwrap().eligible:
            return Success("skipped")
        if not observed.unwrap().applied:
            return Success("skipped")
        audit_result = await self._audit(subscription, AuditAction.SUBSCRIPTION_AUTHENTICATED.value)
        return audit_result.map(lambda _value: "processed")

    async def _handle_subscription_activated(
        self, payload: dict[str, object], *, replay: bool, event_id: str = "webhook"
    ) -> WebhookResult[str]:
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        if (
            replay
            and self.lifecycle_coordinator is None
            and subscription.status
            in {
                SubscriptionStatus.ACTIVE.value,
                SubscriptionStatus.CANCELLED.value,
            }
        ):
            return Success("skipped")
        observed = await self._observe_subscription(
            subscription,
            SubscriptionActivated(
                metadata=self._metadata(event_id, entity),
                period_start=self._parse_datetime(entity.get("current_start")),
                period_end=self._parse_datetime(entity.get("current_end")),
            ),
            event_id=event_id,
            event_type=WebhookEventType.SUBSCRIPTION_ACTIVATED.value,
        )
        if isinstance(observed, Failure):
            return observed
        info = observed.unwrap()
        if not info.eligible or not info.applied:
            return Success("skipped")
        audit_result = await self._audit(subscription, AuditAction.SUBSCRIPTION_ACTIVATED.value)
        return audit_result.map(lambda _value: "processed")

    async def _handle_subscription_charged(
        self, payload: dict[str, object], *, event_id: str = "webhook", replay: bool = False
    ) -> WebhookResult[str]:
        del replay
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        observed = await self._observe_subscription(
            subscription,
            SubscriptionCharged(
                metadata=self._metadata(event_id, entity),
                period_start=self._parse_datetime(entity.get("current_start")),
                period_end=self._parse_datetime(entity.get("current_end")),
            ),
            event_id=event_id,
            event_type=WebhookEventType.SUBSCRIPTION_CHARGED.value,
        )
        if isinstance(observed, Failure):
            return observed
        if not observed.unwrap().eligible or not observed.unwrap().applied:
            return Success("skipped")
        return Success("processed")

    @staticmethod
    async def _handle_subscription_pending(_payload: dict[str, object]) -> WebhookResult[str]:
        # Razorpay's pending notification is an observation without a typed
        # lifecycle event.  Never infer PAST_DUE from it.
        logger.bind(
            operation="webhook", event_type=WebhookEventType.SUBSCRIPTION_PENDING.value
        ).info("Ignoring untyped subscription.pending observation")
        return Success("skipped")

    async def _handle_subscription_halted(
        self, payload: dict[str, object], *, event_id: str = "webhook", replay: bool = False
    ) -> WebhookResult[str]:
        return await self._handle_simple_subscription_event(
            payload,
            event_id=event_id,
            event_type=WebhookEventType.SUBSCRIPTION_HALTED.value,
            event_factory=SubscriptionHalted,
            audit_action=AuditAction.SUBSCRIPTION_HALTED.value,
            replay=replay,
        )

    async def _handle_subscription_cancelled(
        self, payload: dict[str, object], *, event_id: str = "webhook", replay: bool = False
    ) -> WebhookResult[str]:
        return await self._handle_simple_subscription_event(
            payload,
            event_id=event_id,
            event_type=WebhookEventType.SUBSCRIPTION_CANCELLED.value,
            event_factory=SubscriptionCancelled,
            audit_action=AuditAction.SUBSCRIPTION_CANCELLED.value,
            replay=replay,
        )

    async def _handle_subscription_paused(
        self, payload: dict[str, object], *, event_id: str = "webhook", replay: bool = False
    ) -> WebhookResult[str]:
        return await self._handle_simple_subscription_event(
            payload,
            event_id=event_id,
            event_type=WebhookEventType.SUBSCRIPTION_PAUSED.value,
            event_factory=SubscriptionPaused,
            replay=replay,
        )

    async def _handle_subscription_resumed(
        self, payload: dict[str, object], *, event_id: str = "webhook", replay: bool = False
    ) -> WebhookResult[str]:
        return await self._handle_simple_subscription_event(
            payload,
            event_id=event_id,
            event_type=WebhookEventType.SUBSCRIPTION_RESUMED.value,
            event_factory=SubscriptionResumed,
            replay=replay,
        )

    async def _handle_simple_subscription_event(
        self,
        payload: dict[str, object],
        *,
        event_id: str,
        event_type: str,
        event_factory: Callable[..., VerifiedBillingEvent],
        audit_action: str | None = None,
        replay: bool = False,
    ) -> WebhookResult[str]:
        del replay
        entity = self._entity(payload, "subscription")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        event = event_factory(metadata=self._metadata(event_id, entity))
        observed = await self._observe_subscription(
            subscription, event, event_id=event_id, event_type=event_type
        )
        if isinstance(observed, Failure):
            return observed
        info = observed.unwrap()
        if not info.eligible or not info.applied:
            return Success("skipped")
        if audit_action is not None:
            audited = await self._audit(subscription, audit_action)
            if isinstance(audited, Failure):
                return audited
        return Success("processed")

    async def _handle_payment_captured(  # noqa: PLR0912, PLR0914 — validation and capture gates stay explicit
        self, payload: dict[str, object], *, event_id: str = "webhook", replay: bool = False
    ) -> WebhookResult[str]:
        del replay
        entity = self._entity(payload, "payment")
        rz_payment_id = entity.get("id")
        amount = entity.get("amount")
        currency = entity.get("currency")
        if not isinstance(rz_payment_id, str) or not rz_payment_id.strip():
            return Failure(WebhookValidationError(message="Captured payment id is required"))
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            return Failure(
                WebhookValidationError(message="Captured payment amount must be positive")
            )
        if not isinstance(currency, str) or currency not in {code.value for code in CurrencyCode}:
            return Failure(WebhookValidationError(message="Captured payment currency is invalid"))
        subscription = await self._find_subscription_by_entity(entity)
        if not isinstance(rz_payment_id, str) or subscription is None:
            logger.bind(operation="webhook").warning(
                "payment.captured without resolvable subscription", payment_id=rz_payment_id
            )
            return Success("skipped")
        recovered_link = await self._recover_pending_upgrade_for_capture(
            subscription,
            payment_id=rz_payment_id,
            amount_paisa=amount,
            currency=currency,
        )
        if isinstance(recovered_link, Failure):
            return recovered_link
        payment_link_key = recovered_link.unwrap()
        observed = await self._observe_subscription(
            subscription,
            PaymentCaptured(
                metadata=self._metadata(event_id, entity),
                payment_id=rz_payment_id,
                amount_paisa=amount,
                currency=currency,
            ),
            event_id=event_id,
            event_type=WebhookEventType.PAYMENT_CAPTURED.value,
        )
        if isinstance(observed, Failure):
            return observed
        if not observed.unwrap().eligible:
            return Success("skipped")

        payment_result = await self.payment_service.record_payment(
            PaymentRecordDTO(
                razorpay_payment_id=rz_payment_id,
                subscription_id=str(subscription.id),
                amount=amount,
                currency=currency,
                method=entity.get("method"),
                razorpay_order_id=entity.get("order_id"),
                captured_at=self._parse_datetime(entity.get("created_at")),
            ),
            subscription=subscription,
        )
        if isinstance(payment_result, Failure):
            return Failure(_collaborator_error(payment_result.failure()))
        payment = payment_result.unwrap()
        upgrade = await self._complete_captured_upgrade(
            subscription,
            payment_id=rz_payment_id,
            amount_paisa=amount,
            currency=currency,
            payment_link_key=payment_link_key,
        )
        if isinstance(upgrade, Failure):
            return upgrade
        plan_result = await self.plans.find_by_id(subscription.plan_id)
        if isinstance(plan_result, Failure):
            return Failure(_collaborator_error(plan_result.failure()))
        plan = plan_result.unwrap()
        if plan is None:
            return Failure(WebhookValidationError(message="Plan not found for subscription"))
        side_effects = await self._ensure_payment_documents(payment, subscription, plan)
        if isinstance(side_effects, Failure):
            return side_effects
        credit_settled = await self._settle_captured_credit(payment, subscription)
        if isinstance(credit_settled, Failure):
            return credit_settled
        audited = await self._audit(payment, AuditAction.PAYMENT_CAPTURED.value)
        if isinstance(audited, Failure):
            return audited
        return Success("processed")

    async def _settle_captured_credit(  # noqa: PLR0912 -- explicit settlement boundaries
        self, payment: Payment, subscription: Subscription
    ) -> WebhookResult[None]:
        """Preview, settle remotely, then atomically consume the credit plan."""
        if self.credit_apply is None:
            return Success(None)
        invoice_repository = getattr(self.invoice_service, "invoices", None)
        session = getattr(self.invoice_service, "session", None)
        if not isinstance(invoice_repository, InvoiceRepository) or session is None:
            return Success(None)
        invoice_result = await invoice_repository.find_by_payment_id(payment.id)
        if isinstance(invoice_result, Failure):
            return Failure(_collaborator_error(invoice_result.failure()))
        invoice = invoice_result.unwrap()
        if invoice is None:
            return Success(None)

        # ``reserve`` is the explicit public name for the no-consume plan
        # stage.  Keep ``prepare`` as a compatibility fallback for older
        # operation doubles and deployments during the interface transition.
        prepare = getattr(self.credit_apply, "reserve", None)
        if not callable(prepare):
            prepare = getattr(self.credit_apply, "prepare", None)
        apply_settled = getattr(self.credit_apply, "apply_settled", None)
        if not callable(prepare) or not callable(apply_settled):
            # Compatibility for old isolated doubles.  Production wiring
            # always supplies the prepare/apply-settled public interface.
            applied = await self.credit_apply.execute_in_session(
                session, invoice.id, subscription.user_id
            )
            if isinstance(applied, Failure):
                return Failure(_collaborator_error(applied.failure()))
            application = applied.unwrap()
            amount = getattr(application, "credit_applied_paisa", 0)
            if amount > 0:
                settled = await self._settle_credit_application(
                    payment,
                    invoice_id=str(invoice.id),
                    amount_paisa=amount,
                    user_id=subscription.user_id,
                )
                if isinstance(settled, Failure):
                    return settled
            return Success(None)

        planned = await prepare(invoice.id, subscription.user_id)
        if isinstance(planned, Failure):
            return Failure(_collaborator_error(planned.failure()))
        plan = planned.unwrap()
        amount = plan.credit_applied_paisa
        if amount <= 0:
            return Success(None)
        try:
            settled = await self._settle_credit_application(
                payment,
                invoice_id=str(invoice.id),
                amount_paisa=amount,
                user_id=subscription.user_id,
            )
        except Exception as exc:  # noqa: BLE001 — provider boundary remains replayable
            exc.add_note(f"invoice_id={invoice.id}, operation=credit_provider_settlement")
            return Failure(
                WebhookCollaboratorError(
                    message="Credit provider settlement failed",
                    details={
                        "invoice_id": str(invoice.id),
                        "retryable": True,
                        "exception_type": type(exc).__name__,
                    },
                )
            )
        if isinstance(settled, Failure):
            return settled
        settled_value = settled.unwrap()
        if isinstance(settled_value, ProviderReceipt):
            receipt = settled_value
        else:
            # PaymentService's stable public response intentionally exposes
            # the provider refund id as ``razorpay_refund_id``.  Re-wrap those
            # already-validated facts at this boundary for the credit API.
            refund_id = getattr(settled_value, "razorpay_refund_id", None)
            if not isinstance(refund_id, str) or not refund_id:
                refund_id = getattr(settled_value, "id", None)
            if not isinstance(refund_id, str) or not refund_id:
                return Failure(
                    WebhookCollaboratorError(
                        message="Credit settlement returned no provider receipt",
                        details={"invoice_id": str(invoice.id), "manual_review": True},
                    )
                )
            receipt = ProviderReceipt(
                provider_refund_id=refund_id, payment_id=payment.razorpay_payment_id, amount=amount
            )
        try:
            applied = await apply_settled(
                invoice.id,
                subscription.user_id,
                plan,
                receipt,
                payment_id=payment.razorpay_payment_id,
            )
        except Exception as exc:  # noqa: BLE001 — preserve replay after local failpoint
            exc.add_note(f"invoice_id={invoice.id}, operation=credit_settled_apply")
            return Failure(
                WebhookCollaboratorError(
                    message="Credit settlement local apply failed",
                    details={
                        "invoice_id": str(invoice.id),
                        "retryable": True,
                        "exception_type": type(exc).__name__,
                    },
                )
            )
        if isinstance(applied, Failure):
            return Failure(_collaborator_error(applied.failure()))
        return Success(None)

    async def _recover_pending_upgrade_for_capture(
        self,
        subscription: Subscription,
        *,
        payment_id: str,
        amount_paisa: int,
        currency: str,
    ) -> WebhookResult[str | None]:
        """Resolve a staged payment-link intent before observing its capture.

        A capture can arrive before the payment-link provider response.  The
        event must remain replayable until the durable link intent is either
        reconciled or conclusively succeeded; acknowledging it while the
        intent is still ambiguous strands a paid upgrade.
        """
        metadata = getattr(subscription, "metadata_", None) or {}
        pending = metadata.get("pending_upgrade") if isinstance(metadata, dict) else None
        if not isinstance(pending, dict):
            return Success(None)
        payment_link_key = pending.get("payment_link_key")
        if not isinstance(payment_link_key, str) or not payment_link_key:
            return Failure(
                WebhookCollaboratorError(
                    message="Pending upgrade has no durable payment-link key",
                    details={"subscription_id": str(subscription.id), "retryable": True},
                )
            )
        if pending.get("amount_paisa") != amount_paisa or pending.get("currency") != currency:
            return Failure(
                WebhookCollaboratorError(
                    message="Captured payment does not match the pending upgrade",
                    details={"subscription_id": str(subscription.id), "retryable": False},
                )
            )

        coordinator = self.lifecycle_coordinator
        if coordinator is None:
            return Failure(
                WebhookCollaboratorError(
                    message="Pending upgrade recovery coordinator is not configured",
                    details={"subscription_id": str(subscription.id), "retryable": True},
                )
            )
        try:
            recovered = await coordinator.recover_pending_for_capture(
                subscription_id=str(subscription.id),
                payment_link_key=payment_link_key,
                payment_id=payment_id,
                amount_paisa=amount_paisa,
                currency=currency,
            )
        except Exception as exc:  # noqa: BLE001 — coordinator adapter boundary
            exc.add_note(f"subscription_id={subscription.id}, operation=pending_upgrade_recovery")
            return Failure(
                WebhookCollaboratorError(
                    message="Pending payment-link recovery failed",
                    details={
                        "subscription_id": str(subscription.id),
                        "retryable": True,
                        "exception_type": type(exc).__name__,
                    },
                )
            )
        if isinstance(recovered, Failure):
            error = recovered.failure()
            details = dict(error.details or {})
            details.setdefault("retryable", error.retryable)
            return Failure(WebhookCollaboratorError(message=error.message, details=details))

        session = getattr(self.webhooks, "session", None)
        if session is None:
            return Success(payment_link_key)
        try:
            row_result = await session.execute(
                select(ProviderOperationRecord).where(
                    ProviderOperationRecord.idempotency_key == payment_link_key,
                    ProviderOperationRecord.subscription_id == subscription.id,
                )
            )
            row = row_result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="capture_link_recovery"
            )
            await session.rollback()
            return Failure(
                WebhookInfrastructureError(
                    message="Database error while checking payment-link recovery",
                    details={"subscription_id": str(subscription.id), "retryable": True},
                    source="webhook_service",
                )
            )
        if row is not None and row.state == ProviderOperationState.SUCCEEDED.value:
            return Success(payment_link_key)
        return Failure(
            WebhookCollaboratorError(
                message="Captured payment is waiting for payment-link reconciliation",
                details={
                    "subscription_id": str(subscription.id),
                    "payment_link_key": payment_link_key,
                    "retryable": True,
                    "manual_review": row is not None
                    and row.state == ProviderOperationState.MANUAL_REVIEW.value,
                },
            )
        )

    async def _settle_credit_application(
        self,
        payment: Payment,
        *,
        invoice_id: str,
        amount_paisa: int,
        user_id: str,
    ) -> WebhookResult[object]:
        """Settle applied invoice credit back against the captured payment.

        Credit application and provider refund are separate durable stages.
        The stable key is the invoice's immutable application identity, so a
        retry after any crash either reconciles the existing refund or issues
        it once through the payment service's coordinator ledger.
        """
        application_key = f"invoice:{invoice_id}"
        settle = getattr(self.payment_service, "settle_credit_application", None)
        if callable(settle):
            result = await settle(
                payment,
                invoice_id=invoice_id,
                credit_application_key=application_key,
                amount_paisa=amount_paisa,
                user_id=user_id,
            )
        else:
            # Compatibility with isolated payment-service doubles.  The real
            # service always exposes the explicit settlement method above.
            refund = getattr(self.payment_service, "refund", None)
            if not callable(refund):
                return Failure(
                    WebhookCollaboratorError(
                        message="Payment settlement service is not configured",
                        details={"invoice_id": invoice_id, "manual_review": True},
                    )
                )
            result = await refund(
                str(payment.id),
                RefundRequestDTO(
                    amount=amount_paisa,
                    reason="invoice_credit_application",
                    notes={"invoice_id": invoice_id, "credit_application_key": application_key},
                ),
                user_id=user_id,
                idempotency_key=(
                    f"billing:credit-settlement:{invoice_id}:{payment.id}:{application_key}"
                ),
            )
        if isinstance(result, Failure):
            return Failure(_collaborator_error(result.failure()))
        return Success(result.unwrap())

    async def _complete_captured_upgrade(  # noqa: PLR0912 -- payment-link facts are explicit
        self,
        subscription: Subscription,
        *,
        payment_id: str,
        amount_paisa: int,
        currency: str,
        payment_link_key: str | None,
    ) -> WebhookResult[None]:
        """Unlock exactly one staged upgrade after matching payment facts."""
        if self.lifecycle_coordinator is None or payment_link_key is None:
            return Success(None)
        session = getattr(self.webhooks, "session", None)
        if session is None:
            return Success(None)
        try:
            row_result = await session.execute(
                select(ProviderOperationRecord).where(
                    ProviderOperationRecord.idempotency_key == payment_link_key
                )
            )
            row = row_result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            add_database_error_note(
                exc, table="provider_operation_records", operation="capture_upgrade_completion"
            )
            await session.rollback()
            return Failure(
                WebhookCollaboratorError(
                    message="Database error while loading payment-link operation",
                    details={
                        "subscription_id": str(subscription.id),
                        "payment_link_key": payment_link_key,
                        "payment_id": payment_id,
                        "retryable": True,
                    },
                )
            )
        if row is None:
            return Failure(
                WebhookCollaboratorError(
                    message="Payment-link operation was not found",
                    details={
                        "subscription_id": str(subscription.id),
                        "payment_link_key": payment_link_key,
                        "retryable": True,
                    },
                )
            )
        if str(row.subscription_id) != str(subscription.id):
            return Failure(
                WebhookCollaboratorError(
                    message="Payment-link operation does not belong to the captured subscription",
                    details={
                        "subscription_id": str(subscription.id),
                        "payment_link_key": payment_link_key,
                        "retryable": False,
                    },
                )
            )
        if row.operation_kind != "create_payment_link":
            return Failure(
                WebhookCollaboratorError(
                    message="Payment-link key is bound to a different provider operation",
                    details={
                        "subscription_id": str(subscription.id),
                        "payment_link_key": payment_link_key,
                        "retryable": False,
                    },
                )
            )
        if row.state != ProviderOperationState.SUCCEEDED.value:
            return Failure(
                WebhookCollaboratorError(
                    message="Captured payment is waiting for payment-link reconciliation",
                    details={
                        "subscription_id": str(subscription.id),
                        "payment_link_key": payment_link_key,
                        "payment_id": payment_id,
                        "retryable": True,
                    },
                )
            )

        request_data = row.request_data or {}
        operation = request_data.get("operation")
        fields = operation.get("fields") if isinstance(operation, dict) else None
        notes = fields.get("notes") if isinstance(fields, dict) else None

        def mismatch() -> WebhookResult[None]:
            return Failure(
                WebhookCollaboratorError(
                    message="Captured payment does not match the payment-link operation",
                    details={
                        "subscription_id": str(subscription.id),
                        "payment_link_key": payment_link_key,
                        "payment_id": payment_id,
                        "retryable": False,
                    },
                )
            )

        if (
            not isinstance(operation, dict)
            or not isinstance(fields, dict)
            or not isinstance(notes, dict)
        ):
            return mismatch()
        if operation.get("kind") != "create_payment_link":
            return mismatch()
        target_plan_id = notes.get("target_plan_id") if isinstance(notes, dict) else None
        target_provider_plan_id = (
            notes.get("target_provider_plan_id") if isinstance(notes, dict) else None
        )
        expected_amount = notes.get("amount_paisa") if isinstance(notes, dict) else None
        expected_currency = notes.get("currency") if isinstance(notes, dict) else None
        if notes.get("upgrade_dependency") != "true":
            return mismatch()
        if fields.get("amount") != amount_paisa or fields.get("currency") != currency:
            return mismatch()
        if notes.get("subscription_id") != str(subscription.id):
            return mismatch()
        if not all(
            isinstance(value, str) and value
            for value in (target_plan_id, target_provider_plan_id, expected_currency)
        ):
            return mismatch()
        if expected_amount != str(amount_paisa) or expected_currency != currency:
            return mismatch()

        pending = (getattr(subscription, "metadata_", None) or {}).get("pending_upgrade")
        if isinstance(pending, dict) and (
            pending.get("payment_link_key") != payment_link_key
            or pending.get("target_plan_id") != target_plan_id
            or pending.get("target_provider_plan_id") != target_provider_plan_id
        ):
            return Failure(
                WebhookCollaboratorError(
                    message="Payment-link operation does not match the pending upgrade target",
                    details={
                        "subscription_id": str(subscription.id),
                        "payment_link_key": payment_link_key,
                        "retryable": False,
                    },
                )
            )

        result = await self.lifecycle_coordinator.complete_captured_upgrade(
            CapturedUpgrade(
                subscription_id=str(subscription.id),
                payment_link_key=payment_link_key,
                payment_id=payment_id,
                target_plan_id=target_plan_id,
                target_provider_plan_id=target_provider_plan_id,
                amount_paisa=amount_paisa,
                currency=currency,
            )
        )
        if isinstance(result, Failure):
            return Failure(_collaborator_error(result.failure()))
        return Success(None)

    async def _ensure_payment_documents(
        self, payment: Payment, subscription: Subscription, plan: Plan
    ) -> WebhookResult[None]:
        """Generate invoice/receipt at most once for a payment row."""
        invoice_repository = getattr(self.invoice_service, "invoices", None)
        # Always pass through the idempotent invoice service.  An invoice may
        # already exist from a historical write while its inverse
        # Payment.invoice_id is still NULL; generate_for_payment repairs that
        # relation without allocating a second invoice number.
        invoice_result = await self.invoice_service.generate_for_payment(
            payment, subscription, plan
        )
        if isinstance(invoice_result, Failure):
            return Failure(_collaborator_error(invoice_result.failure()))

        receipt_exists = False
        session = getattr(self.invoice_service, "session", None)
        if isinstance(invoice_repository, InvoiceRepository) and session is not None:
            receipt_result = await session.execute(
                select(PaymentReceipt).where(PaymentReceipt.payment_id == payment.id)
            )
            receipt_exists = receipt_result.scalar_one_or_none() is not None
        if not receipt_exists:
            receipt_result = await self.invoice_service.generate_receipt_for_payment(
                payment, subscription, plan
            )
            if isinstance(receipt_result, Failure):
                return Failure(_collaborator_error(receipt_result.failure()))
        return Success(None)

    async def _handle_payment_failed(
        self, payload: dict[str, object], *, event_id: str = "webhook", replay: bool = False
    ) -> WebhookResult[str]:
        del replay
        entity = self._entity(payload, "payment")
        subscription = await self._find_subscription_by_entity(entity)
        if subscription is None:
            return Success("skipped")
        rz_payment_id = entity.get("id")
        observed = await self._observe_subscription(
            subscription,
            PaymentFailed(metadata=self._metadata(event_id, entity), payment_id=str(rz_payment_id or "")),
            event_id=event_id,
            event_type=WebhookEventType.PAYMENT_FAILED.value,
        )
        if isinstance(observed, Failure):
            return observed
        payment_result = await self._record_failed_payment_once(
            rz_payment_id=rz_payment_id,
            subscription=subscription,
            error_code=str(entity.get("error_code") or ""),
            error_description=str(entity.get("error_description") or ""),
        )
        if isinstance(payment_result, Failure):
            return Failure(_collaborator_error(payment_result.failure()))
        if not observed.unwrap().applied:
            return Success("processed")
        audited = await self._audit(subscription, AuditAction.PAYMENT_FAILED.value)
        if isinstance(audited, Failure):
            return audited
        return Success("processed")

    async def _record_failed_payment_once(
        self,
        *,
        rz_payment_id: object,
        subscription: Subscription,
        error_code: str,
        error_description: str,
    ) -> PaymentResult[Payment]:
        """Avoid inserting a second failed-payment row during admin replay."""
        payment_repository = getattr(self.payment_service, "payments", None)
        if isinstance(payment_repository, PaymentRepository) and isinstance(rz_payment_id, str):
            existing = await payment_repository.find_by_razorpay_id(rz_payment_id)
            if isinstance(existing, Failure):
                return existing
            if existing.unwrap() is not None:
                return Success(existing.unwrap())
        return await self.payment_service.record_failed_payment(
            razorpay_payment_id=str(rz_payment_id or ""),
            subscription_id=str(subscription.id),
            error_code=error_code,
            error_description=error_description,
        )

    async def _handle_refund_processed(
        self, payload: dict[str, object], *, replay: bool = False
    ) -> WebhookResult[str]:
        del replay
        entity = self._entity(payload, "refund")
        rz_payment_id = entity.get("payment_id")
        amount = entity.get("amount")
        if not isinstance(rz_payment_id, str) or not isinstance(amount, (int, float)):
            return Success("skipped")
        result = await self.payment_service.handle_refund_processed(
            razorpay_payment_id=rz_payment_id,
            refund_paisa=int(amount),
            refund_id=entity.get("id") if isinstance(entity.get("id"), str) else None,
        )
        if isinstance(result, Failure):
            if isinstance(result.failure(), PaymentManualReviewError):
                logger.bind(operation="webhook", payment_id=rz_payment_id).warning(
                    "Refund event requires manual review; no balance was changed"
                )
                return Success("skipped")
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
