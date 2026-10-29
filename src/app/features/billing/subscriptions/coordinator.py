"""Transport-neutral subscription lifecycle coordinator.

The state machine is deliberately pure.  This module is the narrow seam that
turns its typed effects into provider operations and gives the two sides a
durable ordering:

1. load and plan against an immutable snapshot;
2. atomically stage the new local snapshot, audit facts, and operation intent;
3. execute the already-recorded provider operation;
4. reconcile an ambiguous mutation through a read, never by retrying it.

The production persistence adapter is intentionally not included here.  The
in-memory adapter is useful both as executable documentation of the port and
as a coordinator test adapter.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, cast, override

from pydantic import BaseModel, ConfigDict
from returns.result import Failure, Result, Success

from ..exceptions import (
    LifecycleCoordinatorError,
    LifecycleCoordinatorPersistenceError,
    LifecycleCoordinatorProviderError,
    LifecycleCoordinatorTransientProviderError,
    LifecycleCoordinatorValidationError,
    LifecycleCoordinatorVersionConflictError,
)
from ..providers import (
    CancelSubscriptionOperation,
    ChangeSubscriptionPlanOperation,
    CreateCustomerOperation,
    CreatePaymentLinkOperation,
    CreatePlanOperation,
    CreateRefundOperation,
    CreateSubscriptionOperation,
    FindCustomerByEmailOperation,
    FindPaymentLinkByCorrelationOperation,
    FindPlanByCorrelationOperation,
    FindRefundByCorrelationOperation,
    FindSubscriptionByCorrelationOperation,
    ObserveSubscriptionOperation,
    PauseSubscriptionOperation,
    ProviderCustomer,
    ProviderError,
    ProviderManualReviewRequiredError,
    ProviderOperation,
    ProviderOutcomeUnknownError,
    ProviderPaymentLink,
    ProviderSubscriptionState,
    ProviderSuccess,
    ProviderValue,
    ResumeSubscriptionOperation,
)
from .lifecycle import (
    EffectKind,
    PaymentCaptured,
    PaymentFailed,
    PolicyContext,
    RemoteSyncState,
    SubscriptionActivated,
    SubscriptionAuthenticated,
    SubscriptionCancelled,
    SubscriptionCharged,
    SubscriptionCommand,
    SubscriptionHalted,
    SubscriptionMachine,
    SubscriptionPaused,
    SubscriptionResumed,
    SubscriptionSnapshot,
    TransitionOutcome,
    TransitionPlan,
    VerifiedBillingEvent,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..providers import BillingProvider, ProviderOperationKind
    from .exceptions import SubscriptionVersionConflictError
    from .lifecycle import PendingUpgrade, TransitionEffect


type LifecycleCoordinatorResult[T] = Result[T, LifecycleCoordinatorError]


class IntentStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    TRANSIENT_FAILURE = "transient_failure"
    UNKNOWN = "unknown"
    RECONCILIATION = "reconciliation"
    MANUAL_REVIEW = "manual_review"
    FAILED = "failed"


class ProviderOperationIntent(BaseModel):
    """A provider mutation/read recorded before it is sent."""

    model_config = ConfigDict(frozen=True)

    key: str
    subscription_id: str | None
    operation: ProviderOperation
    status: IntentStatus = IntentStatus.PENDING
    result: ProviderSuccess[ProviderValue] | None = None
    error: ProviderError | None = None


class CapturedUpgrade(BaseModel):
    """Verified economic fact that unlocks one staged plan upgrade."""

    model_config = ConfigDict(frozen=True)

    subscription_id: str
    payment_link_key: str
    payment_id: str
    target_plan_id: str
    target_provider_plan_id: str
    amount_paisa: int
    currency: str


class LifecycleRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    """The persistence port's aggregate projection.

    Provider IDs and customer facts are deliberately outside
    ``SubscriptionSnapshot``: they are integration facts, not lifecycle
    policy.  A SQL adapter can map this projection to its own tables.
    """

    snapshot: SubscriptionSnapshot
    customer_email: str | None = None
    customer_contact: str | None = None
    customer_name: str | None = None
    provider_customer_id: str | None = None
    provider_subscription_id: str | None = None
    last_payment_id: str | None = None
    provider_plan_id: str | None = None
    desired_provider_plan_id: str | None = None


class LifecycleRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    subscription_id: str
    input: SubscriptionCommand | VerifiedBillingEvent
    context: PolicyContext


class LifecyclePreviewRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    subscription_id: str
    input: SubscriptionCommand | VerifiedBillingEvent
    context: PolicyContext


class CoordinatorStatus(StrEnum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    RECONCILED = "reconciled"


class LifecycleOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot: SubscriptionSnapshot
    plan: TransitionPlan
    status: CoordinatorStatus
    provider_results: tuple[ProviderSuccess[ProviderValue], ...] = ()

    @property
    def grants_paid_entitlement(self) -> bool:
        return self.snapshot.grants_paid_entitlement


class LifecyclePersistence(Protocol):
    """Small persistence port used by the coordinator.

    ``stage`` is the transaction seam: a production adapter must atomically
    persist the optimistic transition, audit facts, and intents.  Provider
    calls happen only after it returns successfully.
    """

    async def load(self, subscription_id: str) -> LifecycleCoordinatorResult[LifecycleRecord]: ...

    async def find_command(
        self, idempotency_key: str
    ) -> LifecycleCoordinatorResult[LifecycleOutcome | None]: ...

    async def find_intents(
        self, idempotency_key: str
    ) -> LifecycleCoordinatorResult[tuple[ProviderOperationIntent, ...]]: ...

    async def pending_intents(
        self, *, limit: int = 500, lease_age_seconds: int = 300
    ) -> LifecycleCoordinatorResult[tuple[ProviderOperationIntent, ...]]: ...

    async def find_plan(
        self, idempotency_key: str
    ) -> LifecycleCoordinatorResult[TransitionPlan | None]: ...

    async def stage(
        self,
        record: LifecycleRecord,
        after: SubscriptionSnapshot,
        *,
        idempotency_key: str,
        expected_version: int,
        intents: tuple[ProviderOperationIntent, ...],
        audit_effects: tuple[TransitionEffect, ...],
        plan: TransitionPlan,
    ) -> LifecycleCoordinatorResult[LifecycleRecord]: ...

    async def intent(
        self, key: str
    ) -> LifecycleCoordinatorResult[ProviderOperationIntent | None]: ...

    async def claim_intent(self, key: str) -> LifecycleCoordinatorResult[bool]:
        """Atomically claim the first provider attempt for a pending intent.

        ``True`` means this caller owns the only mutation attempt.  A durable
        pending row which cannot be claimed has already had an attempt (or is
        concurrently owned), so callers must reconcile it before doing any
        further write.  The lease is represented by the existing
        ``started_at``/``last_attempt_at`` columns; no schema expansion is
        needed and a crashed process leaves an auditable lease behind.
        """
        ...

    async def claim_capture_payment(
        self,
        *,
        payment_link_key: str,
        subscription_id: str,
        payment_id: str,
    ) -> LifecycleCoordinatorResult[bool]:
        """Atomically pair one captured payment with one payment-link intent.

        ``True`` claims a previously unpaired link; ``False`` means this same
        payment already owns it.  A different payment is a typed conflict.
        """
        ...

    async def record_intent(
        self, intent: ProviderOperationIntent
    ) -> LifecycleCoordinatorResult[None]: ...

    async def provider_outcome(
        self,
        intent: ProviderOperationIntent,
    ) -> LifecycleCoordinatorResult[None]: ...

    async def mark_sync_failure(
        self,
        subscription_id: str,
        *,
        expected_version: int,
        retryable: bool,
    ) -> LifecycleCoordinatorResult[None]: ...

    async def finish(
        self,
        subscription_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
        provider_results: tuple[ProviderSuccess[ProviderValue], ...],
        reconciled: bool,
        plan: TransitionPlan,
        sync_provider: bool = True,
    ) -> LifecycleCoordinatorResult[LifecycleOutcome]: ...

    async def mark_observed_plan(
        self,
        subscription_id: str,
        *,
        expected_version: int,
        observed_plan_id: str,
    ) -> LifecycleCoordinatorResult[None]: ...


def _persistence_error(operation: str, message: str) -> Failure[LifecycleCoordinatorError]:
    return Failure(
        LifecycleCoordinatorPersistenceError(
            message=message,
            details={"operation": operation},
            source="subscription_lifecycle_coordinator",
            operation=operation,
        )
    )


def _provider_error(
    error: ProviderError,
    *,
    operation: ProviderOperationKind | None = None,
    correlation_id: str | None = None,
) -> Failure[LifecycleCoordinatorError]:
    error_type = (
        LifecycleCoordinatorTransientProviderError
        if error.retryable
        else LifecycleCoordinatorProviderError
    )
    return Failure(
        error_type(
            message=error.message,
            details={"provider_code": error.code.value},
            source="subscription_lifecycle_coordinator",
            provider_code=error.code,
            operation=operation or getattr(error, "operation", None),
            correlation_id=correlation_id or getattr(error, "correlation_id", None),
            requires_reconciliation=error.requires_reconciliation,
        )
    )


def _as_values(effect: TransitionEffect) -> dict[str, object]:
    return dict(effect.values)


class SubscriptionLifecycleCoordinator:
    """Deep lifecycle module with one command/event seam.

    ``apply`` and ``observe`` accept a ``LifecycleRequest``.  For ergonomic
    integration, callers may instead pass an existing typed command/event and
    provide ``subscription_id`` and ``context`` as keyword arguments.  The
    latter is intentionally an adapter convenience; provider operation types
    never leak into callers.
    """

    def __init__(
        self,
        persistence: LifecyclePersistence,
        provider: BillingProvider,
        *,
        machine: SubscriptionMachine | None = None,
    ) -> None:
        self._persistence = persistence
        self._provider = provider
        self._machine = machine or SubscriptionMachine()

    async def preview(
        self,
        request: LifecyclePreviewRequest | LifecycleRequest,
    ) -> LifecycleCoordinatorResult[TransitionPlan]:
        loaded = await self._persistence.load(request.subscription_id)
        if isinstance(loaded, Failure):
            return loaded
        planned = self._machine.plan(loaded.unwrap().snapshot, request.input, request.context)
        if isinstance(planned, Failure):
            return Failure(
                LifecycleCoordinatorValidationError(
                    message=planned.failure().message,
                    details=planned.failure().details,
                    source="subscription_lifecycle_coordinator",
                )
            )
        return Success(planned.unwrap())

    async def apply(
        self,
        request: LifecycleRequest | SubscriptionCommand,
        idempotency_key: str,
        *,
        subscription_id: str | None = None,
        context: PolicyContext | None = None,
    ) -> LifecycleCoordinatorResult[LifecycleOutcome]:
        resolved = self._resolve_request(request, subscription_id=subscription_id, context=context)
        if isinstance(resolved, Failure):
            return resolved
        return await self._run(resolved.unwrap(), idempotency_key=idempotency_key)

    async def observe(
        self,
        request: LifecycleRequest | VerifiedBillingEvent,
        idempotency_key: str,
        *,
        subscription_id: str | None = None,
        context: PolicyContext | None = None,
    ) -> LifecycleCoordinatorResult[LifecycleOutcome]:
        resolved = self._resolve_request(request, subscription_id=subscription_id, context=context)
        if isinstance(resolved, Failure):
            return resolved
        if not isinstance(resolved.unwrap().input, tuple(self._event_types())):
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="observe requires a verified billing event",
                    source="subscription_lifecycle_coordinator",
                )
            )
        request_value = resolved.unwrap()
        pending_capture = None
        capture = request_value.input
        captured_event: PaymentCaptured | None = None
        if isinstance(capture, PaymentCaptured):
            captured_event = capture
            loaded = await self._persistence.load(request_value.subscription_id)
            if isinstance(loaded, Failure):
                return loaded
            pending_capture = loaded.unwrap().snapshot.pending_upgrade
            if pending_capture is not None:
                amount_mismatch = capture.amount_paisa != pending_capture.amount_paisa
                currency_mismatch = capture.currency != pending_capture.currency
                if amount_mismatch is True or currency_mismatch is True:
                    return Failure(
                        LifecycleCoordinatorValidationError(
                            message="Captured payment does not match the pending upgrade",
                            details={
                                "subscription_id": request_value.subscription_id,
                                "expected_amount_paisa": pending_capture.amount_paisa,
                                "captured_amount_paisa": capture.amount_paisa,
                                "expected_currency": pending_capture.currency,
                                "captured_currency": capture.currency,
                            },
                            source="subscription_lifecycle_coordinator",
                        )
                    )
                recovered = await self._ensure_pending_capture_payment_link(
                    subscription_id=request_value.subscription_id,
                    pending_upgrade=pending_capture,
                    capture=capture,
                )
                if isinstance(recovered, Failure):
                    return recovered
        outcome = await self._run(request_value, idempotency_key=idempotency_key)
        if isinstance(outcome, Failure):
            return outcome
        if pending_capture is None or captured_event is None:
            # A replay may find the command ledger before the follow-up
            # capture/remote stage was observed.  Return the current aggregate
            # projection rather than a stale pre-reconciliation snapshot.
            refreshed = await self._persistence.load(request_value.subscription_id)
            if isinstance(refreshed, Success):
                return Success(
                    outcome.unwrap().model_copy(update={"snapshot": refreshed.unwrap().snapshot})
                )
            return outcome
        completed = await self.complete_captured_upgrade(
            CapturedUpgrade(
                subscription_id=request_value.subscription_id,
                payment_link_key=pending_capture.payment_link_key,
                payment_id=captured_event.payment_id,
                target_plan_id=pending_capture.target_plan_id,
                target_provider_plan_id=pending_capture.target_provider_plan_id,
                amount_paisa=pending_capture.amount_paisa,
                currency=pending_capture.currency,
            )
        )
        if isinstance(completed, Failure):
            return Failure(completed.failure())
        refreshed = await self._persistence.load(request_value.subscription_id)
        if isinstance(refreshed, Success):
            return Success(
                outcome.unwrap().model_copy(
                    update={
                        "snapshot": refreshed.unwrap().snapshot,
                        "status": CoordinatorStatus.RECONCILED,
                    }
                )
            )
        return outcome

    async def _ensure_pending_capture_payment_link(
        self,
        *,
        subscription_id: str,
        pending_upgrade: PendingUpgrade,
        capture: PaymentCaptured,
    ) -> LifecycleCoordinatorResult[None]:
        """Prove the exact payment-link dependency before completing a capture.

        Scheduled reconciliation enters through ``observe`` just like a
        webhook, but it does not have the webhook service's prior recovery
        stage.  The pending snapshot key is therefore the only eligible
        intent.  A succeeded intent is read-only here (the webhook may have
        recovered it already); every other state goes through the same
        lease-aware, correlation-read recovery path used by the webhook.
        """
        if not capture.payment_id:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Captured payment id is required for upgrade completion",
                    details={"subscription_id": subscription_id},
                    source="subscription_lifecycle_coordinator",
                )
            )

        loaded = await self._persistence.intent(pending_upgrade.payment_link_key)
        if isinstance(loaded, Failure):
            return loaded
        intent = loaded.unwrap()
        if intent is None:
            return _persistence_error(
                "observe_payment_captured",
                "Pending upgrade payment-link intent was not recorded",
            )
        operation = intent.operation
        if intent.subscription_id != subscription_id:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Pending upgrade payment-link intent does not belong to the subscription",
                    details={
                        "subscription_id": subscription_id,
                        "intent_subscription_id": intent.subscription_id,
                        "payment_link_key": pending_upgrade.payment_link_key,
                    },
                    source="subscription_lifecycle_coordinator",
                )
            )
        if not isinstance(operation, CreatePaymentLinkOperation):
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Pending upgrade key is bound to a different provider operation",
                    details={
                        "payment_link_key": pending_upgrade.payment_link_key,
                        "operation": operation.kind.value,
                    },
                    source="subscription_lifecycle_coordinator",
                )
            )
        expected_notes = {
            "subscription_id": subscription_id,
            "upgrade_dependency": "true",
            "target_plan_id": pending_upgrade.target_plan_id,
            "target_provider_plan_id": pending_upgrade.target_provider_plan_id,
            "amount_paisa": str(pending_upgrade.amount_paisa),
            "currency": pending_upgrade.currency,
        }
        actual_notes = {key: operation.notes.get(key) for key in expected_notes}
        if (
            operation.amount != pending_upgrade.amount_paisa
            or operation.currency != pending_upgrade.currency
            or actual_notes != expected_notes
        ):
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Pending upgrade payment-link intent does not match the capture dependency",
                    details={
                        "payment_link_key": pending_upgrade.payment_link_key,
                        "expected_amount_paisa": pending_upgrade.amount_paisa,
                        "intent_amount_paisa": operation.amount,
                        "expected_currency": pending_upgrade.currency,
                        "intent_currency": operation.currency,
                        "expected_notes": expected_notes,
                        "intent_notes": actual_notes,
                    },
                    source="subscription_lifecycle_coordinator",
                )
            )

        if intent.status is not IntentStatus.SUCCEEDED:
            recovered = await self.recover_pending_for_capture(
                subscription_id=subscription_id,
                payment_link_key=pending_upgrade.payment_link_key,
                payment_id=capture.payment_id,
                amount_paisa=capture.amount_paisa,
                currency=capture.currency or "",
            )
            if isinstance(recovered, Failure):
                return recovered
            loaded = await self._persistence.intent(pending_upgrade.payment_link_key)
            if isinstance(loaded, Failure):
                return loaded
            intent = loaded.unwrap()
            if intent is None:
                return _persistence_error(
                    "observe_payment_captured",
                    "Recovered payment-link intent disappeared",
                )

        if intent.status is not IntentStatus.SUCCEEDED or intent.result is None:
            return _provider_error(
                ProviderOutcomeUnknownError(
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                    message="Payment-link dependency has not reached a succeeded provider result",
                ),
                operation=operation.kind,
                correlation_id=operation.correlation_id,
            )
        if not isinstance(intent.result.value, ProviderPaymentLink):
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Payment-link dependency result is not a provider payment link",
                    details={"payment_link_key": pending_upgrade.payment_link_key},
                    source="subscription_lifecycle_coordinator",
                )
            )
        return Success(None)

    async def execute_provider_operation(  # noqa: PLR0912 -- explicit ledger states
        self,
        operation: ProviderOperation,
        *,
        subscription_id: str | None,
        idempotency_key: str,
    ) -> LifecycleCoordinatorResult[ProviderSuccess[ProviderValue]]:
        """Execute one already-normalized operation through the durable ledger.

        This is intentionally narrow: lifecycle commands use :meth:`apply`,
        while ancillary provider mutations (currently immediate refunds) use
        this seam so an ambiguous network result is reconciled by correlation
        rather than re-issued by a request handler.
        """
        record_result = (
            await self._persistence.load(subscription_id)
            if subscription_id is not None
            else Success(None)
        )
        if isinstance(record_result, Failure):
            return record_result
        existing_result = await self._persistence.intent(idempotency_key)
        if isinstance(existing_result, Failure):
            return existing_result
        existing = existing_result.unwrap()
        if existing is not None and existing.operation != operation:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Idempotency key is already bound to a different provider operation",
                    details={"idempotency_key": idempotency_key},
                    source="subscription_lifecycle_coordinator",
                )
            )
        if (
            existing is not None
            and existing.status is IntentStatus.SUCCEEDED
            and existing.result is not None
        ):
            return Success(existing.result)
        intent = existing or ProviderOperationIntent(
            key=idempotency_key,
            subscription_id=subscription_id,
            operation=operation,
        )
        if existing is None:
            recorded = await self._persistence.record_intent(intent)
            if isinstance(recorded, Failure):
                return recorded
            persisted = await self._persistence.intent(idempotency_key)
            if isinstance(persisted, Failure):
                return persisted
            intent = persisted.unwrap() or intent
        else:
            intent = existing
        if intent.status is IntentStatus.PENDING:
            claimed = await self._persistence.claim_intent(idempotency_key)
            if isinstance(claimed, Failure):
                return claimed
            if not claimed.unwrap():
                # A persisted PENDING is ambiguous once another caller has
                # claimed it (or a prior process crashed after claiming it).
                # Correlation-read first; never reissue the mutation.
                reconciled = await self._reconcile_unknown(intent, record_result.unwrap())
                if isinstance(reconciled, Failure):
                    return reconciled
                result, _ = reconciled.unwrap()
                if result is None:
                    return _provider_error(
                        ProviderOutcomeUnknownError(
                            operation=operation.kind, correlation_id=operation.correlation_id
                        ),
                        operation=operation.kind,
                        correlation_id=operation.correlation_id,
                    )
                return Success(result)
        if intent.status in {
            IntentStatus.UNKNOWN,
            IntentStatus.RECONCILIATION,
            IntentStatus.TRANSIENT_FAILURE,
        }:
            reconciled = await self._reconcile_unknown(intent, record_result.unwrap())
            if isinstance(reconciled, Failure):
                return reconciled
            result, _ = reconciled.unwrap()
            if result is None:
                return _provider_error(
                    ProviderOutcomeUnknownError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    ),
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                )
            return Success(result)
        if intent.status is IntentStatus.FAILED:
            return _provider_error(
                intent.error
                or ProviderOutcomeUnknownError(
                    operation=operation.kind, correlation_id=operation.correlation_id
                ),
                operation=operation.kind,
                correlation_id=operation.correlation_id,
            )
        if intent.status is IntentStatus.MANUAL_REVIEW:
            return _provider_error(
                intent.error
                or ProviderManualReviewRequiredError(
                    operation=operation.kind, correlation_id=operation.correlation_id
                ),
                operation=operation.kind,
                correlation_id=operation.correlation_id,
            )
        result = await self._provider.execute(operation)
        if isinstance(result, Failure):
            status = (
                IntentStatus.UNKNOWN
                if isinstance(result.failure(), ProviderOutcomeUnknownError)
                else IntentStatus.MANUAL_REVIEW
                if result.failure().requires_reconciliation
                else IntentStatus.TRANSIENT_FAILURE
                if result.failure().retryable
                else IntentStatus.FAILED
            )
            saved = await self._persistence.provider_outcome(
                intent.model_copy(update={"status": status, "error": result.failure()})
            )
            if isinstance(saved, Failure):
                return saved
            if isinstance(result.failure(), ProviderOutcomeUnknownError):
                reconciled = await self._reconcile_unknown(intent, record_result.unwrap())
                if isinstance(reconciled, Failure):
                    return reconciled
                reconciled_result, _ = reconciled.unwrap()
                if reconciled_result is not None:
                    return Success(reconciled_result)
            return _provider_error(
                result.failure(), operation=operation.kind, correlation_id=operation.correlation_id
            )
        success = result.unwrap()
        saved = await self._persistence.provider_outcome(
            intent.model_copy(update={"status": IntentStatus.SUCCEEDED, "result": success})
        )
        if isinstance(saved, Failure):
            return saved
        return Success(success)

    async def recover_pending(
        self, *, limit: int = 500, lease_age_seconds: int = 300
    ) -> LifecycleCoordinatorResult[int]:
        """Sweep durable PENDING intents after a worker crash.

        The sweep intentionally uses the same lease gate as request retries:
        an untouched intent may claim the first attempt, while a row carrying
        a prior lease is correlation-read-only.  This makes the worker a
        liveness mechanism without turning it into a second mutation path.
        """
        pending = await self._persistence.pending_intents(
            limit=limit, lease_age_seconds=lease_age_seconds
        )
        if isinstance(pending, Failure):
            return pending
        recovered = 0
        first_failure: LifecycleCoordinatorError | None = None
        for intent in pending.unwrap():
            record: LifecycleRecord | None = None
            if intent.subscription_id is not None:
                loaded = await self._persistence.load(intent.subscription_id)
                if isinstance(loaded, Failure):
                    first_failure = first_failure or loaded.failure()
                    continue
                record = loaded.unwrap()
            if record is None:
                executed = await self.execute_provider_operation(
                    intent.operation,
                    subscription_id=None,
                    idempotency_key=intent.key,
                )
            else:
                executed = await self._execute_intent(
                    intent,
                    record,
                    expected_version=record.snapshot.version,
                )
            if isinstance(executed, Failure):
                first_failure = first_failure or executed.failure()
                continue
            recovered += 1
        if first_failure is not None:
            return Failure(first_failure)
        return Success(recovered)

    async def recover_intent(
        self,
        key: str,
        *,
        subscription_id: str | None = None,
    ) -> LifecycleCoordinatorResult[ProviderSuccess[ProviderValue]]:
        """Recover exactly one durable provider intent by correlation key.

        This is the safe entry point for workflows that already know the
        operation key (for example a captured-upgrade continuation).  It
        routes through the same claim/read gate as the request and scheduled
        sweeps; a pending row that has a prior lease is reconciled by a read,
        never posted a second time.
        """
        loaded = await self._persistence.intent(key)
        if isinstance(loaded, Failure):
            return loaded
        intent = loaded.unwrap()
        if intent is None:
            return _persistence_error("recover_intent", "Provider intent was not recorded")
        target_subscription_id = subscription_id or intent.subscription_id
        return await self.execute_provider_operation(
            intent.operation,
            subscription_id=target_subscription_id,
            idempotency_key=key,
        )

    async def recover_pending_for_capture(
        self,
        *,
        subscription_id: str,
        payment_link_key: str,
        payment_id: str,
        amount_paisa: int,
        currency: str,
    ) -> LifecycleCoordinatorResult[None]:
        """Recover one staged upgrade payment-link intent for a capture.

        A capture webhook already identifies the durable payment-link key.  Do
        not turn that fact into a pending-intent sweep: only this operation is
        eligible for reconciliation, and its recorded request facts must
        agree with the captured amount/currency before the provider is read.
        """
        if not payment_id:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Captured payment id is required for upgrade recovery",
                    details={"subscription_id": subscription_id},
                    source="subscription_lifecycle_coordinator",
                )
            )

        loaded = await self._persistence.intent(payment_link_key)
        if isinstance(loaded, Failure):
            return loaded
        intent = loaded.unwrap()
        if intent is None:
            return _persistence_error(
                "recover_pending_for_capture",
                "Payment-link provider intent was not recorded",
            )
        if intent.subscription_id != subscription_id:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Payment-link intent does not belong to the captured subscription",
                    details={
                        "subscription_id": subscription_id,
                        "intent_subscription_id": intent.subscription_id,
                        "payment_link_key": payment_link_key,
                    },
                    source="subscription_lifecycle_coordinator",
                )
            )
        operation = intent.operation
        if not isinstance(operation, CreatePaymentLinkOperation):
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Durable payment-link key is bound to a different provider operation",
                    details={
                        "payment_link_key": payment_link_key,
                        "operation": operation.kind.value,
                    },
                    source="subscription_lifecycle_coordinator",
                )
            )
        notes = operation.notes
        facts_mismatch = operation.amount != amount_paisa or operation.currency != currency
        notes_amount = notes.get("amount_paisa")
        if notes_amount is not None and notes_amount != str(amount_paisa):
            facts_mismatch = True
        notes_currency = notes.get("currency")
        if notes_currency is not None and notes_currency != currency:
            facts_mismatch = True
        if facts_mismatch:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Captured payment does not match the payment-link intent",
                    details={
                        "subscription_id": subscription_id,
                        "payment_link_key": payment_link_key,
                        "expected_amount_paisa": operation.amount,
                        "captured_amount_paisa": amount_paisa,
                        "expected_currency": operation.currency,
                        "captured_currency": currency,
                    },
                    source="subscription_lifecycle_coordinator",
                )
            )

        recovered = await self.recover_intent(
            payment_link_key,
            subscription_id=subscription_id,
        )
        if isinstance(recovered, Failure):
            return recovered
        provider_link = recovered.unwrap()
        if not isinstance(provider_link.value, ProviderPaymentLink):
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Recovered provider result is not the expected payment link",
                    details={
                        "payment_link_key": payment_link_key,
                        "operation": provider_link.operation.value,
                    },
                    source="subscription_lifecycle_coordinator",
                )
            )
        return Success(None)

    async def complete_captured_upgrade(
        self, dependency: CapturedUpgrade
    ) -> LifecycleCoordinatorResult[None]:
        """Apply a staged upgrade after its matching payment is captured.

        The payment-link operation owns the economic prerequisite.  This
        method uses a deterministic follow-up ledger key, so retries recover
        an already-sent remote mutation and never compensate by rolling the
        desired local plan back.
        """
        paired = await self._persistence.claim_capture_payment(
            payment_link_key=dependency.payment_link_key,
            subscription_id=dependency.subscription_id,
            payment_id=dependency.payment_id,
        )
        if isinstance(paired, Failure):
            return paired
        loaded = await self._persistence.load(dependency.subscription_id)
        if isinstance(loaded, Failure):
            return loaded
        record = loaded.unwrap()
        if record.provider_subscription_id is None:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Provider subscription binding is required before an upgrade capture",
                    details={"subscription_id": dependency.subscription_id},
                    source="subscription_lifecycle_coordinator",
                )
            )
        remote_key = f"{dependency.payment_link_key}:captured:{dependency.payment_id}:remote"
        operation = ChangeSubscriptionPlanOperation(
            provider_subscription_id=record.provider_subscription_id,
            plan_id=dependency.target_provider_plan_id,
            correlation_id=remote_key,
        )
        remote = await self.execute_provider_operation(
            operation,
            subscription_id=dependency.subscription_id,
            idempotency_key=remote_key,
        )
        if isinstance(remote, Failure):
            return remote
        observed = await self.execute_provider_operation(
            ObserveSubscriptionOperation(
                provider_subscription_id=record.provider_subscription_id,
                correlation_id=f"{remote_key}:observe",
            ),
            subscription_id=dependency.subscription_id,
            idempotency_key=f"{remote_key}:observe",
        )
        if isinstance(observed, Failure):
            return observed
        value = observed.unwrap().value
        if (
            not isinstance(value, ProviderSubscriptionState)
            or value.plan_id != dependency.target_provider_plan_id
        ):
            return _provider_error(
                ProviderOutcomeUnknownError(
                    operation=ObserveSubscriptionOperation(
                        provider_subscription_id=record.provider_subscription_id,
                        correlation_id=f"{remote_key}:observe",
                    ).kind,
                    correlation_id=f"{remote_key}:observe",
                    message="Provider plan observation does not match captured upgrade",
                ),
                operation=ObserveSubscriptionOperation(
                    provider_subscription_id=record.provider_subscription_id,
                    correlation_id=f"{remote_key}:observe",
                ).kind,
                correlation_id=f"{remote_key}:observe",
            )
        return await self._persistence.mark_observed_plan(
            dependency.subscription_id,
            expected_version=record.snapshot.version,
            observed_plan_id=dependency.target_plan_id,
        )

    @staticmethod
    def _event_types() -> tuple[type[object], ...]:
        return (
            SubscriptionAuthenticated,
            SubscriptionActivated,
            SubscriptionCharged,
            PaymentCaptured,
            PaymentFailed,
            SubscriptionPaused,
            SubscriptionResumed,
            SubscriptionHalted,
            SubscriptionCancelled,
        )

    @staticmethod
    def _resolve_request(
        request: LifecycleRequest | SubscriptionCommand | VerifiedBillingEvent,
        *,
        subscription_id: str | None,
        context: PolicyContext | None,
    ) -> LifecycleCoordinatorResult[LifecycleRequest]:
        if isinstance(request, LifecycleRequest):
            return Success(request)
        if subscription_id is None or context is None:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="subscription_id and context are required for a raw lifecycle input",
                    source="subscription_lifecycle_coordinator",
                )
            )
        return Success(
            LifecycleRequest(subscription_id=subscription_id, input=request, context=context)
        )

    async def _run(  # noqa: PLR0912,PLR0914 -- orchestration stages are explicit
        self,
        request: LifecycleRequest,
        *,
        idempotency_key: str,
    ) -> LifecycleCoordinatorResult[LifecycleOutcome]:
        duplicate = await self._persistence.find_command(idempotency_key)
        if isinstance(duplicate, Failure):
            return duplicate
        if duplicate.unwrap() is not None:
            return Success(duplicate.unwrap())

        loaded = await self._persistence.load(request.subscription_id)
        if isinstance(loaded, Failure):
            return loaded
        record = loaded.unwrap()
        # A crash can occur after stage committed but before finish.  Resume
        # the durable intents directly; re-planning against the already
        # advanced local state would either reject a valid retry or issue a
        # second mutation.
        staged_intents = await self._persistence.find_intents(idempotency_key)
        if isinstance(staged_intents, Failure):
            return staged_intents
        if staged_intents.unwrap():
            saved_plan = await self._persistence.find_plan(idempotency_key)
            if isinstance(saved_plan, Failure):
                return saved_plan
            plan = saved_plan.unwrap()
            if plan is None:
                return _persistence_error("find_plan", "Staged intents have no transition plan")
            provider_results: list[ProviderSuccess[ProviderValue]] = []
            reconciled = False
            for intent in staged_intents.unwrap():
                executed = await self._execute_intent(
                    intent, record, expected_version=record.snapshot.version
                )
                if isinstance(executed, Failure):
                    return executed
                result, was_reconciled = executed.unwrap()
                reconciled = reconciled or was_reconciled
                if result is not None:
                    provider_results.append(result)
            return await self._persistence.finish(
                request.subscription_id,
                expected_version=record.snapshot.version,
                idempotency_key=idempotency_key,
                provider_results=tuple(provider_results),
                reconciled=reconciled,
                plan=plan,
            )
        planned = self._machine.plan(record.snapshot, request.input, request.context)
        if isinstance(planned, Failure):
            error = planned.failure()
            return Failure(
                LifecycleCoordinatorValidationError(
                    message=error.message,
                    details=error.details,
                    source="subscription_lifecycle_coordinator",
                )
            )
        plan = planned.unwrap()
        if plan.outcome is TransitionOutcome.CONFLICT:
            lifecycle_error = cast("SubscriptionVersionConflictError", plan.error)
            return Failure(
                LifecycleCoordinatorVersionConflictError(
                    message=lifecycle_error.message,
                    details=lifecycle_error.details,
                    source="subscription_lifecycle_coordinator",
                    subscription_id=record.snapshot.id,
                    expected_version=lifecycle_error.expected_version,
                )
            )

        if plan.outcome in {TransitionOutcome.SKIP, TransitionOutcome.NO_OP}:
            outcome = LifecycleOutcome(
                snapshot=record.snapshot,
                plan=plan,
                status=CoordinatorStatus.SKIPPED,
            )
            return await self._store_noop(outcome, idempotency_key)

        translated = self._translate_effects(record, plan.effects, idempotency_key)
        if isinstance(translated, Failure):
            return translated
        intents = translated.unwrap()
        pending_upgrade = plan.after.pending_upgrade
        payment_link_intents = [
            intent for intent in intents if isinstance(intent.operation, CreatePaymentLinkOperation)
        ]
        if pending_upgrade is not None and any(
            effect.kind is EffectKind.PRORATION for effect in plan.effects
        ):
            if len(payment_link_intents) != 1:
                return Failure(
                    LifecycleCoordinatorValidationError(
                        message="Pending upgrade must have exactly one payment-link intent",
                        details={"subscription_id": request.subscription_id},
                        source="subscription_lifecycle_coordinator",
                    )
                )
            # The child intent key is the durable identity used by recovery.
            # Bind it to the snapshot once, before either is persisted, rather
            # than reconstructing the key in a later webhook path.
            payment_link_key = payment_link_intents[0].key
            plan = plan.model_copy(
                update={
                    "after": plan.after.model_copy(
                        update={
                            "pending_upgrade": pending_upgrade.model_copy(
                                update={"payment_link_key": payment_link_key}
                            )
                        }
                    )
                }
            )
        staged = await self._persistence.stage(
            record,
            plan.after,
            expected_version=plan.expected_version,
            idempotency_key=idempotency_key,
            intents=tuple(intents),
            audit_effects=tuple(
                effect for effect in plan.effects if effect.kind is EffectKind.AUDIT
            ),
            plan=plan,
        )
        if isinstance(staged, Failure):
            return staged

        provider_results: list[ProviderSuccess[ProviderValue]] = []
        reconciled = False
        for intent in intents:
            executed = await self._execute_intent(
                intent, record, expected_version=plan.after.version
            )
            if isinstance(executed, Failure):
                return executed
            result, was_reconciled = executed.unwrap()
            reconciled = reconciled or was_reconciled
            if result is not None:
                provider_results.append(result)

        finished = await self._persistence.finish(
            request.subscription_id,
            expected_version=plan.after.version,
            idempotency_key=idempotency_key,
            provider_results=tuple(provider_results),
            reconciled=reconciled,
            plan=plan,
        )
        if isinstance(finished, Failure):
            return finished
        return finished

    async def _store_noop(
        self,
        outcome: LifecycleOutcome,
        idempotency_key: str,
    ) -> LifecycleCoordinatorResult[LifecycleOutcome]:
        saved = await self._persistence.finish(
            outcome.snapshot.id,
            expected_version=outcome.snapshot.version,
            idempotency_key=idempotency_key,
            provider_results=(),
            reconciled=False,
            plan=outcome.plan,
            sync_provider=False,
        )
        if isinstance(saved, Failure):
            return saved
        return Success(
            saved.unwrap().model_copy(
                update={"status": CoordinatorStatus.SKIPPED, "plan": outcome.plan}
            )
        )

    def _translate_effects(
        self,
        record: LifecycleRecord,
        effects: tuple[TransitionEffect, ...],
        idempotency_key: str,
    ) -> LifecycleCoordinatorResult[list[ProviderOperationIntent]]:
        intents: list[ProviderOperationIntent] = []
        for effect in effects:
            if (
                effect.kind is not EffectKind.REMOTE_SYNC
                and effect.kind is not EffectKind.PRORATION
            ):
                continue
            operation_result = self._operation_for_effect(record, effect, idempotency_key)
            if isinstance(operation_result, Failure):
                return operation_result
            for index, operation in enumerate(operation_result.unwrap()):
                intents.append(
                    ProviderOperationIntent(
                        key=f"{idempotency_key}:{effect.idempotency_key}:{index}",
                        subscription_id=record.snapshot.id,
                        operation=operation,
                    )
                )
        return Success(intents)

    @staticmethod
    def _operation_for_effect(  # noqa: PLR0912 — closed effect dispatch
        record: LifecycleRecord,
        effect: TransitionEffect,
        idempotency_key: str,
    ) -> LifecycleCoordinatorResult[list[ProviderOperation]]:
        values = _as_values(effect)
        correlation = f"{record.snapshot.id}:{idempotency_key}:{effect.operation}"
        provider_id = record.provider_subscription_id
        provider_plan_id = record.desired_provider_plan_id or record.provider_plan_id
        if effect.operation == "create_remote":
            customer_id = record.provider_customer_id
            if customer_id is not None:
                if provider_plan_id is None:
                    return Failure(
                        LifecycleCoordinatorValidationError(
                            message="Resolved provider plan binding is required before creating a subscription",
                            details={"subscription_id": record.snapshot.id},
                            source="subscription_lifecycle_coordinator",
                        )
                    )
                return Success(
                    [
                        CreateSubscriptionOperation(
                            plan_id=provider_plan_id,
                            customer_id=customer_id,
                            total_count=0,
                            quantity=1,
                            customer_notify=True,
                            correlation_id=correlation,
                            notes={"subscription_id": record.snapshot.id},
                        )
                    ]
                )
            if record.customer_email is None:
                return Failure(
                    LifecycleCoordinatorValidationError(
                        message="Customer email is required before creating a provider subscription",
                        details={"subscription_id": record.snapshot.id},
                        source="subscription_lifecycle_coordinator",
                    )
                )
            email = record.customer_email
            return Success(
                [
                    CreateCustomerOperation(
                        email=email,
                        contact=record.customer_contact,
                        name=record.customer_name,
                        correlation_id=f"{correlation}:customer",
                        notes={"subscription_id": record.snapshot.id},
                    ),
                ]
            )
        if provider_id is None:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message=f"Provider subscription binding is required for {effect.operation}",
                    details={"subscription_id": record.snapshot.id},
                    source="subscription_lifecycle_coordinator",
                )
            )
        if effect.operation == "cancel_remote":
            return Success(
                [
                    CancelSubscriptionOperation(
                        provider_subscription_id=provider_id,
                        cancel_at_cycle_end=bool(values.get("cancel_at_period_end", True)),
                        correlation_id=correlation,
                    )
                ]
            )
        if effect.operation == "pause_remote":
            return Success(
                [
                    PauseSubscriptionOperation(
                        provider_subscription_id=provider_id,
                        correlation_id=correlation,
                    )
                ]
            )
        if effect.operation == "resume_remote":
            return Success(
                [
                    ResumeSubscriptionOperation(
                        provider_subscription_id=provider_id,
                        correlation_id=correlation,
                    )
                ]
            )
        if effect.operation == "change_remote_plan":
            target = values.get("plan_id")
            if not isinstance(target, str) or not target:
                return Failure(
                    LifecycleCoordinatorValidationError(
                        message="Plan-change effect is missing its target plan",
                        source="subscription_lifecycle_coordinator",
                    )
                )
            return Success(
                [
                    ChangeSubscriptionPlanOperation(
                        provider_subscription_id=provider_id,
                        # ``target`` is the historical in-memory provider
                        # fixture.  SQL records always carry the explicit
                        # desired provider binding; it is never inferred
                        # from a local UUID in production.
                        plan_id=cast("str", values.get("target_provider_plan_id"))
                        if isinstance(values.get("target_provider_plan_id"), str)
                        else record.desired_provider_plan_id or target,
                        correlation_id=correlation,
                    )
                ]
            )
        if effect.operation == "settle_proration":
            amount = values.get("amount_paisa")
            if not isinstance(amount, int) or amount <= 0:
                return Success([])
            if values.get("direction") == "downgrade":
                # A downgrade is a local credit-note/grant operation owned by
                # SubscriptionService; never create a payment link for it.
                return Success([])
            currency = values.get("currency")
            if not isinstance(currency, str) or len(currency) != 3:
                return Failure(
                    LifecycleCoordinatorValidationError(
                        message="Proration payment currency must come from plan policy context",
                        details={"subscription_id": record.snapshot.id},
                        source="subscription_lifecycle_coordinator",
                    )
                )
            return Success(
                [
                    CreatePaymentLinkOperation(
                        amount=amount,
                        currency=currency,
                        description=f"Proration for subscription {record.snapshot.id}",
                        correlation_id=correlation,
                        customer_id=record.provider_customer_id,
                        notes={
                            "subscription_id": record.snapshot.id,
                            "upgrade_dependency": "true",
                            "target_plan_id": str(values.get("target_plan_id", "")),
                            "target_provider_plan_id": str(
                                values.get("target_provider_plan_id", "")
                            ),
                            "amount_paisa": str(amount),
                            "currency": currency,
                        },
                    )
                ]
            )
        return Failure(
            LifecycleCoordinatorValidationError(
                message=f"Unsupported lifecycle effect operation: {effect.operation}",
                source="subscription_lifecycle_coordinator",
            )
        )

    async def _execute_intent(  # noqa: PLR0912 -- provider outcomes are explicitly typed
        self,
        intent: ProviderOperationIntent,
        record: LifecycleRecord,
        *,
        expected_version: int,
    ) -> LifecycleCoordinatorResult[tuple[ProviderSuccess[ProviderValue] | None, bool]]:
        if intent.subscription_id is None:
            return _provider_error(
                ProviderOutcomeUnknownError(
                    operation=intent.operation.kind, correlation_id=intent.operation.correlation_id
                ),
                operation=intent.operation.kind,
                correlation_id=intent.operation.correlation_id,
            )
        intent_subscription_id = intent.subscription_id
        existing = await self._persistence.intent(intent.key)
        if isinstance(existing, Failure):
            return existing
        saved_intent = existing.unwrap()
        if saved_intent is not None and saved_intent.status is IntentStatus.SUCCEEDED:
            if isinstance(saved_intent.operation, CreateCustomerOperation):
                if saved_intent.result is None:
                    return _provider_error(
                        ProviderOutcomeUnknownError(
                            operation=saved_intent.operation.kind,
                            correlation_id=saved_intent.operation.correlation_id,
                        ),
                        operation=saved_intent.operation.kind,
                        correlation_id=saved_intent.operation.correlation_id,
                    )
                # A crash can happen after the customer outcome commits but
                # before its chained subscription intent is recorded.  A
                # succeeded parent therefore still needs to drive its child;
                # returning early would falsely finish the aggregate.
                return await self._continue_customer_followup(
                    saved_intent,
                    saved_intent.result,
                    record,
                    expected_version=expected_version,
                    reconciled=False,
                )
            return Success((saved_intent.result, False))
        if saved_intent is not None and saved_intent.status in {
            IntentStatus.UNKNOWN,
            IntentStatus.RECONCILIATION,
        }:
            # An unknown mutation may already have committed.  Reconcile its
            # durable correlation before allowing any further provider call;
            # never issue the mutation a second time.
            return await self._reconcile_unknown(saved_intent, record)
        if saved_intent is not None and saved_intent.status is IntentStatus.MANUAL_REVIEW:
            error = saved_intent.error or ProviderManualReviewRequiredError(
                operation=intent.operation.kind,
                correlation_id=intent.operation.correlation_id,
            )
            return _provider_error(
                error,
                operation=intent.operation.kind,
                correlation_id=intent.operation.correlation_id,
            )
        if saved_intent is not None and saved_intent.status is IntentStatus.FAILED:
            return _provider_error(
                saved_intent.error
                or ProviderOutcomeUnknownError(
                    operation=intent.operation.kind,
                    correlation_id=intent.operation.correlation_id,
                ),
                operation=intent.operation.kind,
                correlation_id=intent.operation.correlation_id,
            )
        if saved_intent is None or saved_intent.status in {
            IntentStatus.PENDING,
            IntentStatus.TRANSIENT_FAILURE,
        }:
            claimed = await self._persistence.claim_intent(intent.key)
            if isinstance(claimed, Failure):
                return claimed
            if not claimed.unwrap():
                # The operation was staged by another worker or an earlier
                # process.  A PENDING row is not permission to send again.
                return await self._reconcile_unknown(saved_intent or intent, record)
        result = await self._provider.execute(intent.operation)
        if isinstance(result, Failure):
            recorded = await self._persistence.provider_outcome(
                intent.model_copy(
                    update={
                        "status": (
                            IntentStatus.UNKNOWN
                            if isinstance(result.failure(), ProviderOutcomeUnknownError)
                            else IntentStatus.TRANSIENT_FAILURE
                            if result.failure().retryable
                            else IntentStatus.MANUAL_REVIEW
                            if result.failure().requires_reconciliation
                            else IntentStatus.FAILED
                        ),
                        "error": result.failure(),
                    }
                )
            )
            if isinstance(recorded, Failure):
                return recorded
            if isinstance(result.failure(), ProviderOutcomeUnknownError):
                return await self._reconcile_unknown(intent, record)
            marked = await self._persistence.mark_sync_failure(
                intent_subscription_id,
                expected_version=expected_version,
                retryable=result.failure().retryable,
            )
            if isinstance(marked, Failure):
                return marked
            return _provider_error(
                result.failure(),
                operation=intent.operation.kind,
                correlation_id=intent.operation.correlation_id,
            )
        success = result.unwrap()
        recorded = await self._persistence.provider_outcome(
            intent.model_copy(update={"status": IntentStatus.SUCCEEDED, "result": success})
        )
        if isinstance(recorded, Failure):
            return recorded
        if isinstance(intent.operation, CreateCustomerOperation):
            return await self._continue_customer_followup(
                intent,
                success,
                record,
                expected_version=expected_version,
                reconciled=False,
            )
        return Success((success, False))

    async def _continue_customer_followup(
        self,
        intent: ProviderOperationIntent,
        customer_result: ProviderSuccess[ProviderValue],
        record: LifecycleRecord,
        *,
        expected_version: int,
        reconciled: bool,
    ) -> LifecycleCoordinatorResult[tuple[ProviderSuccess[ProviderValue] | None, bool]]:
        """Resume the subscription child of a customer provisioning intent."""
        customer = customer_result.value
        if not isinstance(customer, ProviderCustomer):
            return _provider_error(
                ProviderOutcomeUnknownError(
                    operation=intent.operation.kind,
                    correlation_id=intent.operation.correlation_id,
                    message="Customer provisioning returned no provider customer",
                ),
                operation=intent.operation.kind,
                correlation_id=intent.operation.correlation_id,
            )
        provider_plan_id = record.desired_provider_plan_id or record.provider_plan_id
        if provider_plan_id is None:
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Resolved provider plan binding is required before creating a subscription",
                    details={"subscription_id": record.snapshot.id},
                    source="subscription_lifecycle_coordinator",
                )
            )
        followup = CreateSubscriptionOperation(
            plan_id=provider_plan_id,
            customer_id=customer.provider_customer_id,
            total_count=0,
            quantity=1,
            customer_notify=True,
            correlation_id=f"{intent.operation.correlation_id}:subscription",
            notes={"subscription_id": record.snapshot.id},
        )
        followup_intent = ProviderOperationIntent(
            key=f"{intent.key}:subscription",
            subscription_id=record.snapshot.id,
            operation=followup,
        )
        recorded = await self._persistence.record_intent(followup_intent)
        if isinstance(recorded, Failure):
            return recorded
        followup_result = await self._execute_intent(
            followup_intent,
            record,
            expected_version=expected_version,
        )
        if isinstance(followup_result, Failure):
            return followup_result
        followup_success, followup_reconciled = followup_result.unwrap()
        return Success((followup_success, reconciled or followup_reconciled))

    async def _reconcile_unknown(  # noqa: PLR0912 -- operation-specific reconciliation branches
        self,
        intent: ProviderOperationIntent,
        record: LifecycleRecord | None,
    ) -> LifecycleCoordinatorResult[tuple[ProviderSuccess[ProviderValue] | None, bool]]:
        operation = intent.operation
        marked = await self._persistence.provider_outcome(
            intent.model_copy(update={"status": IntentStatus.RECONCILIATION})
        )
        if isinstance(marked, Failure):
            return marked
        if isinstance(operation, CreateCustomerOperation):
            if record is None:
                return _provider_error(
                    ProviderOutcomeUnknownError(
                        operation=operation.kind, correlation_id=operation.correlation_id
                    ),
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                )
            customer_record = record
            read: ProviderOperation = FindCustomerByEmailOperation(
                email=operation.email,
                correlation_id=f"{operation.correlation_id}:reconcile",
            )
        elif isinstance(operation, CreateSubscriptionOperation):
            read: ProviderOperation = FindSubscriptionByCorrelationOperation(
                correlation_id=operation.correlation_id
            )
        elif isinstance(operation, CreateRefundOperation):
            read = FindRefundByCorrelationOperation(
                payment_id=operation.payment_id,
                correlation_id=operation.correlation_id,
            )
        elif isinstance(operation, CreatePaymentLinkOperation):
            read = FindPaymentLinkByCorrelationOperation(correlation_id=operation.correlation_id)
        elif isinstance(operation, CreatePlanOperation):
            read = FindPlanByCorrelationOperation(correlation_id=operation.correlation_id)
        elif record is not None and record.provider_subscription_id is not None:
            read = ObserveSubscriptionOperation(
                provider_subscription_id=record.provider_subscription_id,
                correlation_id=f"{operation.correlation_id}:reconcile",
            )
        else:
            return _provider_error(
                ProviderOutcomeUnknownError(
                    operation=operation.kind, correlation_id=operation.correlation_id
                ),
                operation=operation.kind,
                correlation_id=operation.correlation_id,
            )
        observed = await self._provider.execute(read)
        if isinstance(observed, Failure):
            return _provider_error(
                observed.failure(),
                operation=read.kind,
                correlation_id=read.correlation_id,
            )
        success = observed.unwrap()
        if isinstance(operation, ChangeSubscriptionPlanOperation):
            observed_value = success.value
            if not isinstance(observed_value, ProviderSubscriptionState) or (
                observed_value.plan_id != operation.plan_id
            ):
                # Keep the original intent in reconciliation.  The remote
                # mutation is never retried; a later webhook/reconciliation
                # pass can read again and converge once the provider catches
                # up.
                return _provider_error(
                    ProviderOutcomeUnknownError(
                        operation=operation.kind,
                        correlation_id=operation.correlation_id,
                        message="Provider plan has not converged to the requested target",
                    ),
                    operation=operation.kind,
                    correlation_id=operation.correlation_id,
                )
        recorded = await self._persistence.provider_outcome(
            intent.model_copy(update={"status": IntentStatus.SUCCEEDED, "result": success})
        )
        if isinstance(recorded, Failure):
            return recorded
        if isinstance(operation, CreateCustomerOperation):
            # Resolving an ambiguous customer create is only the first half of
            # subscription creation.  The follow-up subscription intent must
            # still be durably recorded and executed; otherwise finish would
            # incorrectly mark the local row converged after finding only a
            # customer.
            return await self._continue_customer_followup(
                intent,
                success,
                customer_record,
                expected_version=customer_record.snapshot.version,
                reconciled=True,
            )
        return Success((success, True))


class InMemoryLifecyclePersistence(LifecyclePersistence):
    """Concurrency-checking persistence adapter for coordinator tests."""

    def __init__(self, records: Mapping[str, LifecycleRecord] | None = None) -> None:
        # The test double historically used ``snapshot.plan_id`` as its
        # provider fixture identifier.  Production SQL rows must provide an
        # explicit binding; retaining this compatibility here keeps the pure
        # coordinator tests focused on state-machine behavior.
        self.records = {
            key: record
            if record.desired_provider_plan_id or record.provider_plan_id
            else record.model_copy(update={"desired_provider_plan_id": record.snapshot.plan_id})
            for key, record in (records or {}).items()
        }
        self.intents: dict[str, ProviderOperationIntent] = {}
        # ``started_at`` is represented by membership here.  It intentionally
        # survives a failed/crashed coordinator call so a retry reconciles
        # instead of issuing a second mutation.
        self._claimed_intents: set[str] = set()
        self.capture_pairs: dict[str, str] = {}
        self.commands: dict[str, LifecycleOutcome] = {}
        self.audit: list[TransitionEffect] = []
        self.plans: dict[str, TransitionPlan] = {}

    def seed(self, record: LifecycleRecord) -> None:
        self.records[record.snapshot.id] = (
            record
            if record.desired_provider_plan_id or record.provider_plan_id
            else record.model_copy(update={"desired_provider_plan_id": record.snapshot.plan_id})
        )

    @override
    async def load(self, subscription_id: str) -> LifecycleCoordinatorResult[LifecycleRecord]:
        record = self.records.get(subscription_id)
        if record is None:
            return _persistence_error("load", f"Subscription {subscription_id} was not found")
        return Success(record)

    @override
    async def find_command(
        self, idempotency_key: str
    ) -> LifecycleCoordinatorResult[LifecycleOutcome | None]:
        return Success(self.commands.get(idempotency_key))

    @override
    async def find_intents(
        self, idempotency_key: str
    ) -> LifecycleCoordinatorResult[tuple[ProviderOperationIntent, ...]]:
        prefix = f"{idempotency_key}:"
        return Success(
            tuple(intent for key, intent in self.intents.items() if key.startswith(prefix))
        )

    @override
    async def pending_intents(
        self, *, limit: int = 500, lease_age_seconds: int = 300
    ) -> LifecycleCoordinatorResult[tuple[ProviderOperationIntent, ...]]:
        del lease_age_seconds
        return Success(
            tuple(
                intent
                for intent in list(self.intents.values())[:limit]
                if intent.status
                in {
                    IntentStatus.PENDING,
                    IntentStatus.UNKNOWN,
                    IntentStatus.RECONCILIATION,
                }
            )
        )

    @override
    async def find_plan(
        self, idempotency_key: str
    ) -> LifecycleCoordinatorResult[TransitionPlan | None]:
        return Success(self.plans.get(idempotency_key))

    @override
    async def stage(
        self,
        record: LifecycleRecord,
        after: SubscriptionSnapshot,
        *,
        idempotency_key: str,
        expected_version: int,
        intents: tuple[ProviderOperationIntent, ...],
        audit_effects: tuple[TransitionEffect, ...],
        plan: TransitionPlan,
    ) -> LifecycleCoordinatorResult[LifecycleRecord]:
        current = self.records.get(record.snapshot.id)
        if current is None:
            return _persistence_error("stage", "Subscription disappeared during staging")
        if current.snapshot.version != expected_version:
            return Failure(
                LifecycleCoordinatorVersionConflictError(
                    message="Subscription changed while lifecycle transition was being staged",
                    source="subscription_lifecycle_coordinator",
                    subscription_id=record.snapshot.id,
                    expected_version=expected_version,
                )
            )
        staged = current.model_copy(update={"snapshot": after})
        self.records[record.snapshot.id] = staged
        self.audit.extend(audit_effects)
        self.plans.setdefault(idempotency_key, plan)
        for intent in intents:
            self.intents.setdefault(intent.key, intent)
        return Success(staged)

    @override
    async def intent(self, key: str) -> LifecycleCoordinatorResult[ProviderOperationIntent | None]:
        return Success(self.intents.get(key))

    @override
    async def claim_intent(self, key: str) -> LifecycleCoordinatorResult[bool]:
        intent = self.intents.get(key)
        if intent is None or intent.status is not IntentStatus.PENDING:
            return Success(False)
        if key in self._claimed_intents:
            return Success(False)
        self._claimed_intents.add(key)
        return Success(True)

    @override
    async def claim_capture_payment(
        self,
        *,
        payment_link_key: str,
        subscription_id: str,
        payment_id: str,
    ) -> LifecycleCoordinatorResult[bool]:
        intent = self.intents.get(payment_link_key)
        if (
            intent is None
            or intent.subscription_id != subscription_id
            or not isinstance(intent.operation, CreatePaymentLinkOperation)
            or intent.status is not IntentStatus.SUCCEEDED
        ):
            return Failure(
                LifecycleCoordinatorValidationError(
                    message="Payment-link intent is not a valid succeeded link for capture",
                    details={
                        "payment_link_key": payment_link_key,
                        "subscription_id": subscription_id,
                        "manual_review": True,
                    },
                    source="subscription_lifecycle_coordinator",
                )
            )
        existing = self.capture_pairs.get(payment_link_key)
        if existing is None:
            self.capture_pairs[payment_link_key] = payment_id
            return Success(True)
        if existing == payment_id:
            return Success(False)
        return Failure(
            LifecycleCoordinatorValidationError(
                message="Payment-link operation is already paired with another capture",
                details={
                    "payment_link_key": payment_link_key,
                    "subscription_id": subscription_id,
                    "captured_payment_id": existing,
                    "conflicting_payment_id": payment_id,
                    "manual_review": True,
                },
                source="subscription_lifecycle_coordinator",
            )
        )

    @override
    async def record_intent(
        self, intent: ProviderOperationIntent
    ) -> LifecycleCoordinatorResult[None]:
        self.intents.setdefault(intent.key, intent)
        return Success(None)

    @override
    async def mark_observed_plan(
        self,
        subscription_id: str,
        *,
        expected_version: int,
        observed_plan_id: str,
    ) -> LifecycleCoordinatorResult[None]:
        record = self.records.get(subscription_id)
        if record is None:
            return _persistence_error("mark_observed_plan", "Subscription disappeared")
        if record.snapshot.version != expected_version:
            return Failure(
                LifecycleCoordinatorVersionConflictError(
                    message="Subscription changed while recording observed plan",
                    source="subscription_lifecycle_coordinator",
                    subscription_id=subscription_id,
                    expected_version=expected_version,
                )
            )
        self.records[subscription_id] = record.model_copy(
            update={
                "snapshot": record.snapshot.model_copy(
                    update={
                        "observed_plan_id": observed_plan_id,
                        "remote_sync": RemoteSyncState.SYNCED,
                        "pending_upgrade": None,
                    }
                ),
                "provider_plan_id": record.desired_provider_plan_id,
            }
        )
        return Success(None)

    @override
    async def provider_outcome(
        self, intent: ProviderOperationIntent
    ) -> LifecycleCoordinatorResult[None]:
        self.intents[intent.key] = intent
        record = self.records.get(intent.subscription_id)
        if record is not None and intent.result is not None:
            value = intent.result.value
            if isinstance(value, ProviderSubscriptionState):
                record = record.model_copy(
                    update={
                        "provider_subscription_id": value.provider_subscription_id,
                        "provider_plan_id": value.plan_id or record.provider_plan_id,
                    }
                )
            elif isinstance(value, ProviderCustomer):
                record = record.model_copy(
                    update={"provider_customer_id": value.provider_customer_id}
                )
            if intent.subscription_id is not None:
                self.records[intent.subscription_id] = record
        return Success(None)

    @override
    async def mark_sync_failure(
        self,
        subscription_id: str,
        *,
        expected_version: int,
        retryable: bool,
    ) -> LifecycleCoordinatorResult[None]:
        record = self.records.get(subscription_id)
        if record is None:
            return _persistence_error("mark_sync_failure", "Subscription disappeared")
        if record.snapshot.version != expected_version:
            return Failure(
                LifecycleCoordinatorVersionConflictError(
                    message="Subscription changed while recording provider failure",
                    source="subscription_lifecycle_coordinator",
                    subscription_id=subscription_id,
                    expected_version=expected_version,
                )
            )
        state = (
            RemoteSyncState.RETRYABLE_FAILURE if retryable else RemoteSyncState.PERMANENT_FAILURE
        )
        self.records[subscription_id] = record.model_copy(
            update={
                "snapshot": record.snapshot.model_copy(update={"remote_sync": state}),
            }
        )
        return Success(None)

    @override
    async def finish(
        self,
        subscription_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
        provider_results: tuple[ProviderSuccess[ProviderValue], ...],
        reconciled: bool,
        plan: TransitionPlan,
        sync_provider: bool = True,
    ) -> LifecycleCoordinatorResult[LifecycleOutcome]:
        record = self.records.get(subscription_id)
        if record is None:
            return _persistence_error("finish", "Subscription disappeared while finalizing")
        if record.snapshot.version != expected_version:
            return Failure(
                LifecycleCoordinatorVersionConflictError(
                    message="Subscription changed while lifecycle transition was finalizing",
                    source="subscription_lifecycle_coordinator",
                    subscription_id=subscription_id,
                    expected_version=expected_version,
                )
            )
        provider_plan = record.provider_plan_id
        if reconciled:
            for result in provider_results:
                if isinstance(result.value, ProviderSubscriptionState):
                    provider_plan = result.value.plan_id or provider_plan
        finalized_snapshot = record.snapshot.model_copy(
            update={
                "remote_sync": (
                    RemoteSyncState.SYNCED
                    if sync_provider and reconciled
                    else record.snapshot.remote_sync
                ),
                "observed_plan_id": (
                    provider_plan or record.snapshot.observed_plan_id
                    if sync_provider and reconciled
                    else record.snapshot.observed_plan_id
                ),
                # Provider acknowledgement is an integration fact on the already
                # staged lifecycle version; it must not create a second logical
                # transition or make optimistic command versions skip numbers.
                "version": record.snapshot.version,
            }
        )
        finalized = record.model_copy(
            update={"snapshot": finalized_snapshot, "provider_plan_id": provider_plan}
        )
        self.records[subscription_id] = finalized
        status = CoordinatorStatus.RECONCILED if reconciled else CoordinatorStatus.APPLIED
        outcome = LifecycleOutcome(
            snapshot=finalized_snapshot,
            plan=plan,
            status=status,
            provider_results=provider_results,
        )
        self.commands[idempotency_key] = outcome
        return Success(outcome)


# Short name for dependency wiring while retaining the explicit class name in
# tracebacks and documentation.
LifecycleCoordinator = SubscriptionLifecycleCoordinator
