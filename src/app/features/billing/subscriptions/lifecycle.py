"""Pure subscription lifecycle policy.

This module is deliberately unaware of SQLAlchemy, FastAPI, and Razorpay.  It
turns a typed snapshot plus a typed command/provider event into a deterministic
plan that an outer transaction executor may persist and dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

from returns.result import Failure, Success

from .exceptions import (
    SubscriptionInvalidTransitionError,
    SubscriptionValidationError,
    SubscriptionVersionConflictError,
)

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Literal

    from .exceptions import SubscriptionResult


class SubscriptionStateKind(StrEnum):
    CREATED = "created"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    HALTED = "halted"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class CreatedState:
    kind: Literal[SubscriptionStateKind.CREATED] = SubscriptionStateKind.CREATED


@dataclass(frozen=True, slots=True)
class AuthenticatedState:
    kind: Literal[SubscriptionStateKind.AUTHENTICATED] = SubscriptionStateKind.AUTHENTICATED


@dataclass(frozen=True, slots=True)
class ActiveState:
    period_start: datetime | None = None
    period_end: datetime | None = None
    kind: Literal[SubscriptionStateKind.ACTIVE] = SubscriptionStateKind.ACTIVE


@dataclass(frozen=True, slots=True)
class PastDueState:
    retry_count: int = 0
    period_start: datetime | None = None
    period_end: datetime | None = None
    kind: Literal[SubscriptionStateKind.PAST_DUE] = SubscriptionStateKind.PAST_DUE


@dataclass(frozen=True, slots=True)
class HaltedState:
    period_start: datetime | None = None
    period_end: datetime | None = None
    kind: Literal[SubscriptionStateKind.HALTED] = SubscriptionStateKind.HALTED


@dataclass(frozen=True, slots=True)
class CancelledState:
    at: datetime | None = None
    at_period_end: bool = False
    kind: Literal[SubscriptionStateKind.CANCELLED] = SubscriptionStateKind.CANCELLED


@dataclass(frozen=True, slots=True)
class PausedState:
    start: datetime | None = None
    end: datetime | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    kind: Literal[SubscriptionStateKind.PAUSED] = SubscriptionStateKind.PAUSED


@dataclass(frozen=True, slots=True)
class ExpiredState:
    kind: Literal[SubscriptionStateKind.EXPIRED] = SubscriptionStateKind.EXPIRED


type SubscriptionState = (
    CreatedState
    | AuthenticatedState
    | ActiveState
    | PastDueState
    | HaltedState
    | CancelledState
    | PausedState
    | ExpiredState
)


class RemoteSyncState(StrEnum):
    SYNCED = "synced"
    PENDING = "pending"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENT_FAILURE = "permanent_failure"


@dataclass(frozen=True, slots=True)
class CancellationIntent:
    """A local cancellation request, independent from observed remote state."""

    requested_at: datetime
    at_period_end: bool


@dataclass(frozen=True, slots=True)
class PendingUpgrade:
    """Economic dependency for a desired upgrade awaiting capture.

    The local plan is intentionally advanced before payment, but this durable
    fact keeps the provider plan and paid entitlement on the old side until a
    matching payment event is verified.
    """

    payment_link_key: str
    target_plan_id: str
    target_provider_plan_id: str
    amount_paisa: int
    currency: str


@dataclass(frozen=True, slots=True)
class SubscriptionSnapshot:
    """Immutable domain snapshot passed to the pure state machine."""

    id: str
    user_id: str
    plan_id: str
    state: SubscriptionState
    version: int = 0
    remote_sync: RemoteSyncState = RemoteSyncState.SYNCED
    cancellation_intent: CancellationIntent | None = None
    observed_plan_id: str | None = None
    pending_upgrade: PendingUpgrade | None = None

    @property
    def grants_paid_entitlement(self) -> bool:
        """Only active or past-due service states grant paid access."""
        return self.pending_upgrade is None and isinstance(self.state, (ActiveState, PastDueState))


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Explicit clock and optional financial facts used by planning."""

    now: datetime
    current_plan_amount: int | None = None
    target_plan_amount: int | None = None
    current_interval: str | None = None
    target_interval: str | None = None
    target_currency: str | None = None
    # Provider binding for the desired target.  The local UUID is not a valid
    # provider plan identifier; keeping this fact in policy context lets the
    # coordinator persist a payment dependency without guessing later.
    target_provider_plan_id: str | None = None
    tax_rate: Decimal = Decimal("0.18")


@dataclass(frozen=True, slots=True)
class CommandMetadata:
    request_id: str
    expected_version: int | None = None


@dataclass(frozen=True, slots=True)
class PauseSubscription:
    metadata: CommandMetadata
    duration_days: int | None = None


@dataclass(frozen=True, slots=True)
class ResumeSubscription:
    metadata: CommandMetadata


@dataclass(frozen=True, slots=True)
class CancelSubscription:
    metadata: CommandMetadata
    cancel_at_period_end: bool = True


@dataclass(frozen=True, slots=True)
class ChangeSubscriptionPlan:
    metadata: CommandMetadata
    new_plan_id: str
    effective_at: datetime


@dataclass(frozen=True, slots=True)
class CreateSubscription:
    metadata: CommandMetadata
    plan_id: str


type SubscriptionCommand = (
    PauseSubscription
    | ResumeSubscription
    | CancelSubscription
    | ChangeSubscriptionPlan
    | CreateSubscription
)


@dataclass(frozen=True, slots=True)
class EventMetadata:
    event_id: str
    occurred_at: datetime
    replay: bool = False
    expected_version: int | None = None


@dataclass(frozen=True, slots=True)
class SubscriptionAuthenticated:
    metadata: EventMetadata


@dataclass(frozen=True, slots=True)
class SubscriptionActivated:
    metadata: EventMetadata
    period_start: datetime | None = None
    period_end: datetime | None = None


@dataclass(frozen=True, slots=True)
class SubscriptionCharged:
    metadata: EventMetadata
    period_start: datetime | None = None
    period_end: datetime | None = None


@dataclass(frozen=True, slots=True)
class PaymentCaptured:
    metadata: EventMetadata
    payment_id: str
    amount_paisa: int
    currency: str | None = None


@dataclass(frozen=True, slots=True)
class PaymentFailed:
    metadata: EventMetadata
    payment_id: str


@dataclass(frozen=True, slots=True)
class SubscriptionPaused:
    metadata: EventMetadata


@dataclass(frozen=True, slots=True)
class SubscriptionResumed:
    metadata: EventMetadata


@dataclass(frozen=True, slots=True)
class SubscriptionHalted:
    metadata: EventMetadata


@dataclass(frozen=True, slots=True)
class SubscriptionCancelled:
    metadata: EventMetadata


type VerifiedBillingEvent = (
    SubscriptionAuthenticated
    | SubscriptionActivated
    | SubscriptionCharged
    | PaymentCaptured
    | PaymentFailed
    | SubscriptionPaused
    | SubscriptionResumed
    | SubscriptionHalted
    | SubscriptionCancelled
)


class TransitionOutcome(StrEnum):
    APPLY = "apply"
    NO_OP = "no_op"
    SKIP = "skip"
    CONFLICT = "conflict"


class EffectKind(StrEnum):
    REMOTE_SYNC = "remote_sync"
    PAYMENT_FACT = "payment_fact"
    PRORATION = "proration"
    REFUND = "refund"
    AUDIT = "audit"


@dataclass(frozen=True, slots=True)
class TransitionEffect:
    kind: EffectKind
    operation: str
    idempotency_key: str
    values: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class TransitionPlan:
    outcome: TransitionOutcome
    before: SubscriptionSnapshot
    after: SubscriptionSnapshot
    expected_version: int
    effects: tuple[TransitionEffect, ...] = ()
    reason: str | None = None
    error: object | None = None


def _error(message: str, *, details: dict[str, object]) -> SubscriptionValidationError:
    return SubscriptionValidationError(
        message=message,
        details=details,
        source="subscription_lifecycle",
    )


def _invalid_transition(
    snapshot: SubscriptionSnapshot, target: SubscriptionStateKind
) -> SubscriptionInvalidTransitionError:
    return SubscriptionInvalidTransitionError(
        message=f"Invalid subscription state transition: {snapshot.state.kind} -> {target}",
        details={"current": snapshot.state.kind.value, "target": target.value},
        source="subscription_lifecycle",
        subscription_id=snapshot.id,
        current=snapshot.state.kind.value,
        target=target.value,
    )


def _effect(
    kind: EffectKind,
    operation: str,
    key: str,
    **values: object,
) -> TransitionEffect:
    return TransitionEffect(kind, operation, key, tuple(values.items()))


def _validate_explicit_time(value: datetime, *, name: str) -> SubscriptionResult[datetime]:
    if value.tzinfo is None or value.utcoffset() is None:
        return Failure(_error(f"{name} must include a timezone offset", details={"field": name}))
    return Success(value)


def _merge_period_facts(
    before_start: datetime | None,
    before_end: datetime | None,
    incoming_start: datetime | None,
    incoming_end: datetime | None,
) -> SubscriptionResult[tuple[datetime | None, datetime | None] | None]:
    """Merge unordered provider facts without erasing known period bounds.

    ``None`` is a successful stale-fact marker.  The caller turns it into a
    zero-effect ``SKIP`` plan, while a ``Failure`` means the payload itself is
    malformed (for example, an inverted period).
    """

    for name, value in (
        ("period_start", incoming_start),
        ("period_end", incoming_end),
    ):
        if value is not None:
            checked = _validate_explicit_time(value, name=name)
            if isinstance(checked, Failure):
                return checked
    start_regressed = (
        before_start is not None and incoming_start is not None and incoming_start < before_start
    )
    end_regressed = (
        before_end is not None and incoming_end is not None and incoming_end < before_end
    )
    if start_regressed or end_regressed:
        return Success(None)
    merged_start = incoming_start if incoming_start is not None else before_start
    merged_end = incoming_end if incoming_end is not None else before_end
    if merged_start is not None and merged_end is not None and merged_end < merged_start:
        return Failure(_error("Billing period is empty or inverted", details={}))
    return Success((merged_start, merged_end))


def _event_facts(
    before: SubscriptionState,
    incoming_start: datetime | None,
    incoming_end: datetime | None,
) -> SubscriptionResult[tuple[datetime | None, datetime | None] | None]:
    if isinstance(before, ActiveState | PastDueState | HaltedState | PausedState):
        return _merge_period_facts(
            before.period_start,
            before.period_end,
            incoming_start,
            incoming_end,
        )
    return _merge_period_facts(None, None, incoming_start, incoming_end)


class SubscriptionMachine:
    """Exhaustive, side-effect-free subscription transition planner."""

    def plan(  # noqa: PLR0912 -- closed union dispatch is intentionally exhaustive
        self,
        snapshot: SubscriptionSnapshot,
        input_: SubscriptionCommand | VerifiedBillingEvent,
        context: PolicyContext,
    ) -> SubscriptionResult[TransitionPlan]:
        now_result = _validate_explicit_time(context.now, name="context.now")
        if isinstance(now_result, Failure):
            return now_result
        metadata = input_.metadata
        if isinstance(
            input_,
            (
                SubscriptionAuthenticated,
                SubscriptionActivated,
                SubscriptionCharged,
                PaymentCaptured,
                PaymentFailed,
                SubscriptionPaused,
                SubscriptionResumed,
                SubscriptionHalted,
                SubscriptionCancelled,
            ),
        ):
            occurred = _validate_explicit_time(
                input_.metadata.occurred_at, name="event.occurred_at"
            )
            if isinstance(occurred, Failure):
                return occurred
            if input_.metadata.replay:
                return self._skip(snapshot, "replayed_event")
        if metadata.expected_version is not None and metadata.expected_version != snapshot.version:
            conflict = SubscriptionVersionConflictError(
                message="Subscription snapshot is stale; refetch and retry",
                details={
                    "subscription_id": snapshot.id,
                    "expected_version": metadata.expected_version,
                    "actual_version": snapshot.version,
                },
                source="subscription_lifecycle",
                subscription_id=snapshot.id,
                expected_version=metadata.expected_version,
            )
            return Success(
                TransitionPlan(
                    TransitionOutcome.CONFLICT,
                    snapshot,
                    snapshot,
                    snapshot.version,
                    reason="expected_version_mismatch",
                    error=conflict,
                )
            )

        match input_:
            case PauseSubscription():
                result = self._pause(snapshot, input_, context)
            case ResumeSubscription():
                result = self._resume(snapshot, input_)
            case CancelSubscription():
                result = self._cancel(snapshot, input_, context)
            case ChangeSubscriptionPlan():
                result = self._change_plan(snapshot, input_, context)
            case CreateSubscription():
                result = self._create(snapshot, input_)
            case SubscriptionAuthenticated():
                result = self._authenticated(snapshot, input_)
            case SubscriptionActivated():
                result = self._activated(snapshot, input_)
            case SubscriptionCharged():
                result = self._charged(snapshot, input_)
            case PaymentCaptured():
                result = self._payment_captured(snapshot, input_)
            case PaymentFailed():
                result = self._payment_failed(snapshot, input_)
            case SubscriptionPaused():
                result = self._paused(snapshot, input_)
            case SubscriptionResumed():
                result = self._resumed(snapshot, input_)
            case SubscriptionHalted():
                result = self._halted(snapshot, input_)
            case SubscriptionCancelled():
                result = self._cancelled(snapshot, input_)
            case _ as unreachable:
                assert_never(unreachable)
        if isinstance(result, Failure):
            return result
        return Success(self._bump(result.unwrap()))

    @staticmethod
    def _bump(plan: TransitionPlan) -> TransitionPlan:
        if plan.outcome is TransitionOutcome.APPLY:
            after = replace(plan.after, version=plan.before.version + 1)
            return replace(plan, after=after)
        return plan

    @staticmethod
    def _base(
        before: SubscriptionSnapshot,
        after: SubscriptionSnapshot,
        *,
        operation: str,
        metadata_key: str,
        effects: tuple[TransitionEffect, ...] = (),
        outcome: TransitionOutcome = TransitionOutcome.APPLY,
        reason: str | None = None,
    ) -> SubscriptionResult[TransitionPlan]:
        return Success(
            TransitionPlan(
                outcome,
                before,
                after,
                before.version,
                effects=(
                    *_effects_for_change(before, after, operation, metadata_key),
                    *effects,
                ),
                reason=reason,
            )
        )

    def _pause(
        self, before: SubscriptionSnapshot, command: PauseSubscription, context: PolicyContext
    ) -> SubscriptionResult[TransitionPlan]:
        if not isinstance(before.state, ActiveState):
            return Failure(_invalid_transition(before, SubscriptionStateKind.PAUSED))
        if command.duration_days is not None and not 1 <= command.duration_days <= 365:
            return Failure(
                _error(
                    "Pause duration must be between 1 and 365 days",
                    details={"duration_days": command.duration_days},
                )
            )
        end = (
            context.now + timedelta(days=command.duration_days)
            if command.duration_days is not None
            else None
        )
        after = replace(
            before,
            state=PausedState(
                start=context.now,
                end=end,
                period_start=before.state.period_start,
                period_end=before.state.period_end,
            ),
            remote_sync=RemoteSyncState.PENDING,
        )
        return self._base(
            before,
            after,
            operation="pause_subscription",
            metadata_key=command.metadata.request_id,
            effects=(
                _effect(
                    EffectKind.REMOTE_SYNC,
                    "pause_remote",
                    f"{command.metadata.request_id}:remote",
                ),
            ),
        )

    def _resume(
        self, before: SubscriptionSnapshot, command: ResumeSubscription
    ) -> SubscriptionResult[TransitionPlan]:
        if not isinstance(before.state, PausedState):
            return Failure(_invalid_transition(before, SubscriptionStateKind.ACTIVE))
        after = replace(
            before,
            state=ActiveState(before.state.period_start, before.state.period_end),
            remote_sync=RemoteSyncState.PENDING,
        )
        return self._base(
            before,
            after,
            operation="resume_subscription",
            metadata_key=command.metadata.request_id,
            effects=(
                _effect(
                    EffectKind.REMOTE_SYNC,
                    "resume_remote",
                    f"{command.metadata.request_id}:remote",
                ),
            ),
        )

    def _cancel(
        self, before: SubscriptionSnapshot, command: CancelSubscription, context: PolicyContext
    ) -> SubscriptionResult[TransitionPlan]:
        if not isinstance(before.state, (ActiveState, PastDueState, HaltedState, PausedState)):
            return Failure(_invalid_transition(before, SubscriptionStateKind.CANCELLED))
        intent = CancellationIntent(
            requested_at=context.now,
            at_period_end=command.cancel_at_period_end,
        )
        if command.cancel_at_period_end:
            after = replace(
                before,
                cancellation_intent=intent,
                remote_sync=RemoteSyncState.PENDING,
            )
        else:
            after = replace(
                before,
                state=CancelledState(at=context.now, at_period_end=False),
                cancellation_intent=intent,
                remote_sync=RemoteSyncState.PENDING,
            )
        return self._base(
            before,
            after,
            operation="cancel_subscription",
            metadata_key=command.metadata.request_id,
            effects=(
                _effect(
                    EffectKind.REMOTE_SYNC,
                    "cancel_remote",
                    f"{command.metadata.request_id}:remote",
                    cancel_at_period_end=command.cancel_at_period_end,
                ),
            ),
        )

    def _change_plan(
        self,
        before: SubscriptionSnapshot,
        command: ChangeSubscriptionPlan,
        context: PolicyContext,
    ) -> SubscriptionResult[TransitionPlan]:
        if not isinstance(before.state, ActiveState):
            return Failure(_invalid_transition(before, SubscriptionStateKind.ACTIVE))
        effective = _validate_explicit_time(command.effective_at, name="effective_at")
        if isinstance(effective, Failure):
            return effective
        if (
            context.current_interval is not None
            and context.target_interval is not None
            and context.current_interval != context.target_interval
        ):
            return Failure(
                _error(
                    "Cannot change between different billing intervals",
                    details={
                        "current_interval": context.current_interval,
                        "target_interval": context.target_interval,
                    },
                )
            )
        if (context.current_plan_amount is None) != (context.target_plan_amount is None):
            return Failure(
                _error(
                    "Current and target plan amounts must be provided together",
                    details={},
                )
            )
        if (
            before.state.period_start is not None
            and command.effective_at < before.state.period_start
        ):
            return Failure(
                _error(
                    "effective_at must not precede the current billing period",
                    details={"effective_at": command.effective_at.isoformat()},
                )
            )
        if before.state.period_end is not None and command.effective_at > before.state.period_end:
            return Failure(
                _error(
                    "effective_at must not follow the current billing period",
                    details={"effective_at": command.effective_at.isoformat()},
                )
            )
        after = replace(before, plan_id=command.new_plan_id, remote_sync=RemoteSyncState.PENDING)
        effects: tuple[TransitionEffect, ...] = ()
        upgrade_requires_capture = False
        if context.current_plan_amount is not None and context.target_plan_amount is not None:
            proration = calculate_proration(
                subscription_id=before.id,
                current_plan_id=before.plan_id,
                new_plan_id=command.new_plan_id,
                current_amount_paisa=context.current_plan_amount,
                target_amount_paisa=context.target_plan_amount,
                period_start=before.state.period_start,
                period_end=before.state.period_end,
                effective_at=command.effective_at,
                tax_rate=context.tax_rate,
            )
            if isinstance(proration, Failure):
                return proration
            effects += (
                _effect(
                    EffectKind.PRORATION,
                    "settle_proration",
                    f"{command.metadata.request_id}:proration",
                    amount_paisa=proration.unwrap().amount_paisa,
                    direction=proration.unwrap().direction,
                    currency=context.target_currency,
                    target_plan_id=command.new_plan_id,
                    target_provider_plan_id=context.target_provider_plan_id,
                ),
            )
            direction = proration.unwrap().direction
            upgrade_requires_capture = (
                direction.value if isinstance(direction, StrEnum) else str(direction)
            ) == "upgrade"
            if (
                upgrade_requires_capture
                and context.target_provider_plan_id
                and context.target_currency
            ):
                after = replace(
                    after,
                    pending_upgrade=PendingUpgrade(
                        payment_link_key=command.metadata.request_id,
                        target_plan_id=command.new_plan_id,
                        target_provider_plan_id=context.target_provider_plan_id,
                        amount_paisa=proration.unwrap().amount_paisa,
                        currency=context.target_currency,
                    ),
                )
        # An upgrade is an economic two-phase transition.  Stage the desired
        # local plan and payment-link intent now, but do not mutate the remote
        # subscription until a matching captured payment is verified.  A
        # downgrade (or no-change) has no such gate and can be synchronized
        # in the same lifecycle command.
        if not upgrade_requires_capture:
            effects += (
                _effect(
                    EffectKind.REMOTE_SYNC,
                    "change_remote_plan",
                    f"{command.metadata.request_id}:remote",
                    plan_id=command.new_plan_id,
                    target_provider_plan_id=context.target_provider_plan_id,
                ),
            )
        return self._base(
            before,
            after,
            operation="change_plan",
            metadata_key=command.metadata.request_id,
            effects=effects,
        )

    def _create(
        self, before: SubscriptionSnapshot, command: CreateSubscription
    ) -> SubscriptionResult[TransitionPlan]:
        if not isinstance(before.state, CreatedState) or before.plan_id != command.plan_id:
            return Failure(_error("Subscription creation requires a CREATED snapshot", details={}))
        after = replace(before, remote_sync=RemoteSyncState.PENDING)
        return self._base(
            before,
            after,
            operation="create_subscription",
            metadata_key=command.metadata.request_id,
            effects=(
                _effect(
                    EffectKind.REMOTE_SYNC,
                    "create_remote",
                    f"{command.metadata.request_id}:remote",
                ),
                _effect(
                    EffectKind.AUDIT,
                    "create_subscription",
                    f"{command.metadata.request_id}:audit",
                    before=before.state.kind,
                    after=after.state.kind,
                ),
            ),
        )

    def _authenticated(
        self, before: SubscriptionSnapshot, event: SubscriptionAuthenticated
    ) -> SubscriptionResult[TransitionPlan]:
        if isinstance(before.state, AuthenticatedState | ActiveState):
            return self._base(
                before,
                before,
                operation="subscription_authenticated",
                metadata_key=event.metadata.event_id,
                outcome=TransitionOutcome.SKIP
                if event.metadata.replay
                else TransitionOutcome.NO_OP,
                reason="already_authenticated",
            )
        if not isinstance(before.state, CreatedState):
            return Failure(_invalid_transition(before, SubscriptionStateKind.AUTHENTICATED))
        return self._base(
            before,
            replace(before, state=AuthenticatedState()),
            operation="subscription_authenticated",
            metadata_key=event.metadata.event_id,
        )

    def _activated(
        self, before: SubscriptionSnapshot, event: SubscriptionActivated
    ) -> SubscriptionResult[TransitionPlan]:
        if isinstance(before.state, CancelledState | ExpiredState):
            return self._skip(before, "terminal_subscription")
        if isinstance(before.state, ActiveState):
            facts = _event_facts(before.state, event.period_start, event.period_end)
            if isinstance(facts, Failure):
                return facts
            merged = facts.unwrap()
            if merged is None:
                return self._skip(before, "stale_period_facts")
            after = replace(
                before,
                state=ActiveState(*merged),
            )
            return self._base(
                before,
                after,
                operation="subscription_activated",
                metadata_key=event.metadata.event_id,
                outcome=TransitionOutcome.NO_OP if after == before else TransitionOutcome.APPLY,
                reason="already_active",
            )
        if not isinstance(before.state, AuthenticatedState):
            return Failure(_invalid_transition(before, SubscriptionStateKind.ACTIVE))
        facts = _event_facts(before.state, event.period_start, event.period_end)
        if isinstance(facts, Failure):
            return facts
        merged = facts.unwrap()
        if merged is None:
            return self._skip(before, "stale_period_facts")
        return self._base(
            before,
            replace(before, state=ActiveState(*merged)),
            operation="subscription_activated",
            metadata_key=event.metadata.event_id,
        )

    def _charged(
        self, before: SubscriptionSnapshot, event: SubscriptionCharged
    ) -> SubscriptionResult[TransitionPlan]:
        if isinstance(before.state, (CancelledState, ExpiredState, PausedState, HaltedState)):
            return self._skip(before, "ineligible_subscription")
        if not isinstance(before.state, (ActiveState, PastDueState)):
            return Failure(_invalid_transition(before, SubscriptionStateKind.ACTIVE))
        facts = _event_facts(before.state, event.period_start, event.period_end)
        if isinstance(facts, Failure):
            return facts
        merged = facts.unwrap()
        if merged is None:
            return self._skip(before, "stale_period_facts")
        after = replace(before, state=ActiveState(*merged))
        return self._base(
            before,
            after,
            operation="subscription_charged",
            metadata_key=event.metadata.event_id,
            outcome=TransitionOutcome.NO_OP if after == before else TransitionOutcome.APPLY,
        )

    def _payment_captured(
        self, before: SubscriptionSnapshot, event: PaymentCaptured
    ) -> SubscriptionResult[TransitionPlan]:
        if isinstance(
            before.state,
            (
                CancelledState,
                ExpiredState,
                CreatedState,
                AuthenticatedState,
                PausedState,
                HaltedState,
            ),
        ):
            return self._skip(before, "payment_without_paid_state")
        if isinstance(before.state, ActiveState):
            state = before.state
        elif isinstance(before.state, PastDueState):
            state = ActiveState(before.state.period_start, before.state.period_end)
        else:
            return self._skip(before, "payment_without_paid_state")
        after = replace(before, state=state)
        return self._base(
            before,
            after,
            operation="payment_captured",
            metadata_key=event.metadata.event_id,
            effects=(
                _effect(
                    EffectKind.PAYMENT_FACT,
                    "record_payment",
                    f"{event.metadata.event_id}:payment",
                    payment_id=event.payment_id,
                    amount_paisa=event.amount_paisa,
                ),
            ),
        )

    def _payment_failed(
        self, before: SubscriptionSnapshot, event: PaymentFailed
    ) -> SubscriptionResult[TransitionPlan]:
        if isinstance(before.state, (CancelledState, ExpiredState, HaltedState)):
            return self._skip(before, "terminal_subscription")
        if isinstance(before.state, PastDueState):
            return self._base(
                before,
                before,
                operation="payment_failed",
                metadata_key=event.metadata.event_id,
                outcome=TransitionOutcome.NO_OP,
                reason="already_past_due",
            )
        if not isinstance(before.state, ActiveState):
            return Failure(_invalid_transition(before, SubscriptionStateKind.PAST_DUE))
        return self._base(
            before,
            replace(
                before,
                state=PastDueState(
                    retry_count=1,
                    period_start=before.state.period_start,
                    period_end=before.state.period_end,
                ),
            ),
            operation="payment_failed",
            metadata_key=event.metadata.event_id,
        )

    def _paused(
        self, before: SubscriptionSnapshot, event: SubscriptionPaused
    ) -> SubscriptionResult[TransitionPlan]:
        if isinstance(before.state, PausedState):
            return self._skip(before, "already_paused")
        if not isinstance(before.state, ActiveState):
            return Failure(_invalid_transition(before, SubscriptionStateKind.PAUSED))
        return self._base(
            before,
            replace(
                before,
                state=PausedState(
                    start=event.metadata.occurred_at,
                    period_start=before.state.period_start,
                    period_end=before.state.period_end,
                ),
            ),
            operation="subscription_paused",
            metadata_key=event.metadata.event_id,
        )

    def _resumed(
        self, before: SubscriptionSnapshot, event: SubscriptionResumed
    ) -> SubscriptionResult[TransitionPlan]:
        if isinstance(before.state, ActiveState):
            return self._skip(before, "already_active")
        if not isinstance(before.state, PausedState):
            return Failure(_invalid_transition(before, SubscriptionStateKind.ACTIVE))
        return self._base(
            before,
            replace(before, state=ActiveState(before.state.period_start, before.state.period_end)),
            operation="subscription_resumed",
            metadata_key=event.metadata.event_id,
        )

    def _halted(
        self, before: SubscriptionSnapshot, event: SubscriptionHalted
    ) -> SubscriptionResult[TransitionPlan]:
        if isinstance(before.state, HaltedState):
            return self._skip(before, "already_halted")
        if not isinstance(before.state, PastDueState):
            return Failure(_invalid_transition(before, SubscriptionStateKind.HALTED))
        return self._base(
            before,
            replace(
                before,
                state=HaltedState(
                    period_start=before.state.period_start,
                    period_end=before.state.period_end,
                ),
            ),
            operation="subscription_halted",
            metadata_key=event.metadata.event_id,
        )

    def _cancelled(
        self, before: SubscriptionSnapshot, event: SubscriptionCancelled
    ) -> SubscriptionResult[TransitionPlan]:
        if isinstance(before.state, CancelledState | ExpiredState):
            return self._skip(before, "already_terminal")
        if not isinstance(before.state, (ActiveState, PastDueState, HaltedState, PausedState)):
            return Failure(_invalid_transition(before, SubscriptionStateKind.CANCELLED))
        return self._base(
            before,
            replace(before, state=CancelledState(at=event.metadata.occurred_at)),
            operation="subscription_cancelled",
            metadata_key=event.metadata.event_id,
        )

    @staticmethod
    def _skip(before: SubscriptionSnapshot, reason: str) -> SubscriptionResult[TransitionPlan]:
        return Success(
            TransitionPlan(
                TransitionOutcome.SKIP,
                before,
                before,
                before.version,
                reason=reason,
            )
        )


def _effects_for_change(
    before: SubscriptionSnapshot,
    after: SubscriptionSnapshot,
    operation: str,
    key: str,
) -> tuple[TransitionEffect, ...]:
    if (
        before.state == after.state
        and before.plan_id == after.plan_id
        and before.cancellation_intent == after.cancellation_intent
        and before.observed_plan_id == after.observed_plan_id
    ):
        return ()
    return (
        _effect(
            EffectKind.AUDIT,
            operation,
            f"{key}:audit",
            before=before.state.kind,
            after=after.state.kind,
        ),
    )


@dataclass(frozen=True, slots=True)
class ProrationDecision:
    remaining_fraction: Decimal
    amount_paisa: int
    tax_paisa: int
    direction: Literal["upgrade", "downgrade", "no_change"]


@dataclass(frozen=True, slots=True)
class CancellationRefund:
    amount_paisa: int
    remaining_fraction: Decimal
    policy: Literal["FULL", "PRO_RATA", "NONE"]


def remaining_fraction(
    *, period_start: datetime, period_end: datetime, effective_at: datetime
) -> SubscriptionResult[Decimal]:
    """Return unused-time fraction with explicit, bounded timestamps."""
    for name, value in (
        ("period_start", period_start),
        ("period_end", period_end),
        ("effective_at", effective_at),
    ):
        checked = _validate_explicit_time(value, name=name)
        if isinstance(checked, Failure):
            return checked
    if period_end <= period_start:
        return Failure(_error("Billing period is empty or inverted", details={}))
    if effective_at < period_start:
        return Failure(
            _error("effective_at must not precede the current billing period", details={})
        )
    if effective_at > period_end:
        return Failure(
            _error("effective_at must not follow the current billing period", details={})
        )
    total = (period_end - period_start) // timedelta(microseconds=1)
    remaining = (period_end - effective_at) // timedelta(microseconds=1)
    return Success(Decimal(remaining) / Decimal(total))


def calculate_proration(
    *,
    subscription_id: str,
    current_plan_id: str,
    new_plan_id: str,
    current_amount_paisa: int,
    target_amount_paisa: int,
    period_start: datetime | None,
    period_end: datetime | None,
    effective_at: datetime,
    tax_rate: Decimal,
) -> SubscriptionResult[ProrationDecision]:
    """Pure plan-change proration policy."""
    del subscription_id, current_plan_id, new_plan_id
    if period_start is None or period_end is None:
        return Failure(_error("Subscription has no active billing period", details={}))
    if current_amount_paisa < 0 or target_amount_paisa < 0:
        return Failure(_error("Plan amounts cannot be negative", details={}))
    fraction_result = remaining_fraction(
        period_start=period_start, period_end=period_end, effective_at=effective_at
    )
    if isinstance(fraction_result, Failure):
        return fraction_result
    fraction = fraction_result.unwrap()
    current = int(
        (Decimal(current_amount_paisa) * fraction).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
    )
    target = int(
        (Decimal(target_amount_paisa) * fraction).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
    )
    amount = target - current
    if amount > 0:
        direction: Literal["upgrade", "downgrade", "no_change"] = "upgrade"
    elif amount < 0:
        direction = "downgrade"
    else:
        direction = "no_change"
    tax = 0
    if amount != 0:
        base = (Decimal(abs(amount)) / (Decimal(1) + tax_rate)).quantize(
            Decimal(1), rounding=ROUND_HALF_EVEN
        )
        tax = abs(amount) - int(base)
    return Success(ProrationDecision(fraction, amount, tax, direction))


def calculate_cancellation_refund(
    *,
    policy: Literal["FULL", "PRO_RATA", "NONE"],
    captured_amount_paisa: int,
    period_start: datetime,
    period_end: datetime,
    effective_at: datetime,
) -> SubscriptionResult[CancellationRefund]:
    """Calculate unused value; FULL never means refunding all captured money."""
    fraction_result = remaining_fraction(
        period_start=period_start, period_end=period_end, effective_at=effective_at
    )
    if isinstance(fraction_result, Failure):
        return fraction_result
    fraction = fraction_result.unwrap()
    if policy not in {"FULL", "PRO_RATA", "NONE"}:
        return Failure(_error("Unknown cancellation refund policy", details={"policy": policy}))
    if policy == "NONE":
        amount = 0
    else:
        amount = int(
            (Decimal(max(captured_amount_paisa, 0)) * fraction).quantize(
                Decimal(1), rounding=ROUND_HALF_EVEN
            )
        )
    return Success(CancellationRefund(amount, fraction, policy))
