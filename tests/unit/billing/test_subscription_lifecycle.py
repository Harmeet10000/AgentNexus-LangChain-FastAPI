"""Behavioral tests for the pure subscription lifecycle kernel."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from returns.result import Failure

from app.features.billing.subscriptions.lifecycle import (
    ActiveState,
    AuthenticatedState,
    CancelledState,
    CancelSubscription,
    ChangeSubscriptionPlan,
    CommandMetadata,
    CreatedState,
    CreateSubscription,
    EffectKind,
    EventMetadata,
    HaltedState,
    PausedState,
    PauseSubscription,
    PaymentCaptured,
    PaymentFailed,
    PolicyContext,
    RemoteSyncState,
    SubscriptionActivated,
    SubscriptionAuthenticated,
    SubscriptionCancelled,
    SubscriptionCharged,
    SubscriptionHalted,
    SubscriptionMachine,
    SubscriptionPaused,
    SubscriptionResumed,
    SubscriptionSnapshot,
    SubscriptionStateKind,
    TransitionOutcome,
    calculate_cancellation_refund,
    calculate_proration,
    remaining_fraction,
)

NOW = datetime(2026, 1, 15, tzinfo=UTC)
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 2, 1, tzinfo=UTC)


def _snapshot(state: object, *, version: int = 0) -> SubscriptionSnapshot:
    return SubscriptionSnapshot(
        id="sub-1",
        user_id="user-1",
        plan_id="plan-1",
        state=state,  # type: ignore[arg-type]
        version=version,
    )


def _command_metadata(
    *, request_id: str = "request-1", version: int | None = None
) -> CommandMetadata:
    return CommandMetadata(request_id=request_id, expected_version=version)


def _event_metadata(*, event_id: str = "event-1", replay: bool = False) -> EventMetadata:
    return EventMetadata(event_id=event_id, occurred_at=NOW, replay=replay)


def test_state_graph_is_enforced_by_the_public_planner() -> None:
    machine = SubscriptionMachine()
    result = machine.plan(
        _snapshot(CreatedState()),
        SubscriptionActivated(metadata=_event_metadata()),
        PolicyContext(now=NOW),
    )

    assert isinstance(result, Failure)
    assert result.failure().code.value == "INVALID_STATE_TRANSITION"


def test_authenticated_then_activated_preserves_paid_entitlement_boundary() -> None:
    machine = SubscriptionMachine()
    authenticated = machine.plan(
        _snapshot(CreatedState()),
        SubscriptionAuthenticated(metadata=_event_metadata()),
        PolicyContext(now=NOW),
    ).unwrap()
    assert isinstance(authenticated.after.state, AuthenticatedState)
    assert authenticated.after.grants_paid_entitlement is False

    activated = machine.plan(
        authenticated.after,
        SubscriptionActivated(
            metadata=_event_metadata(event_id="event-2"), period_start=START, period_end=END
        ),
        PolicyContext(now=NOW),
    ).unwrap()
    assert isinstance(activated.after.state, ActiveState)
    assert activated.after.grants_paid_entitlement is True


def test_active_same_state_payment_event_is_a_fact_update_not_invalid_transition() -> None:
    machine = SubscriptionMachine()
    snapshot = _snapshot(ActiveState(period_start=START, period_end=END))
    result = machine.plan(
        snapshot,
        SubscriptionCharged(
            metadata=_event_metadata(),
            period_start=START + timedelta(days=31),
            period_end=END + timedelta(days=31),
        ),
        PolicyContext(now=NOW),
    ).unwrap()

    assert result.outcome is TransitionOutcome.APPLY
    assert isinstance(result.after.state, ActiveState)
    assert result.after.state.period_start == START + timedelta(days=31)
    assert result.after.version == snapshot.version + 1


def test_replayed_activation_is_deterministic_skip_without_effects() -> None:
    machine = SubscriptionMachine()
    snapshot = _snapshot(ActiveState(period_start=START, period_end=END), version=4)
    result = machine.plan(
        snapshot,
        SubscriptionActivated(
            metadata=_event_metadata(replay=True), period_start=START, period_end=END
        ),
        PolicyContext(now=NOW),
    ).unwrap()

    assert result.outcome is TransitionOutcome.SKIP
    assert result.after == snapshot
    assert result.effects == ()


def test_user_status_change_is_immediate_but_remote_sync_is_pending() -> None:
    machine = SubscriptionMachine()
    result = machine.plan(
        _snapshot(ActiveState(period_start=START, period_end=END)),
        CancelSubscription(metadata=_command_metadata(), cancel_at_period_end=False),
        PolicyContext(now=NOW),
    ).unwrap()

    assert isinstance(result.after.state, CancelledState)
    assert result.after.remote_sync is RemoteSyncState.PENDING
    assert any(effect.kind is EffectKind.REMOTE_SYNC for effect in result.effects)


def test_period_end_cancellation_keeps_active_entitlement_and_records_intent() -> None:
    result = (
        SubscriptionMachine()
        .plan(
            _snapshot(ActiveState(period_start=START, period_end=END)),
            CancelSubscription(metadata=_command_metadata(), cancel_at_period_end=True),
            PolicyContext(now=NOW),
        )
        .unwrap()
    )

    assert isinstance(result.after.state, ActiveState)
    assert result.after.grants_paid_entitlement is True
    assert result.after.cancellation_intent is not None
    assert result.after.cancellation_intent.at_period_end is True
    assert result.after.remote_sync is RemoteSyncState.PENDING
    assert any(effect.kind is EffectKind.AUDIT for effect in result.effects)


def test_create_marks_remote_sync_pending_and_emits_audit() -> None:
    result = (
        SubscriptionMachine()
        .plan(
            _snapshot(CreatedState()),
            CreateSubscription(metadata=_command_metadata(), plan_id="plan-1"),
            PolicyContext(now=NOW),
        )
        .unwrap()
    )

    assert result.after.remote_sync is RemoteSyncState.PENDING
    assert any(effect.kind is EffectKind.REMOTE_SYNC for effect in result.effects)
    assert any(effect.kind is EffectKind.AUDIT for effect in result.effects)


def test_stale_command_returns_conflict_plan_without_effects() -> None:
    machine = SubscriptionMachine()
    snapshot = _snapshot(ActiveState(period_start=START, period_end=END), version=3)
    result = machine.plan(
        snapshot,
        CancelSubscription(metadata=_command_metadata(version=2)),
        PolicyContext(now=NOW),
    ).unwrap()

    assert result.outcome is TransitionOutcome.CONFLICT
    assert result.after == snapshot
    assert result.effects == ()
    assert result.error is not None


@pytest.mark.parametrize(
    "event",
    [
        SubscriptionAuthenticated(metadata=_event_metadata(replay=True)),
        SubscriptionActivated(
            metadata=_event_metadata(replay=True), period_start=START, period_end=END
        ),
        SubscriptionCharged(
            metadata=_event_metadata(replay=True), period_start=START, period_end=END
        ),
        PaymentCaptured(
            metadata=_event_metadata(replay=True), payment_id="pay-1", amount_paisa=1000
        ),
        PaymentFailed(metadata=_event_metadata(replay=True), payment_id="pay-2"),
        SubscriptionPaused(metadata=_event_metadata(replay=True)),
        SubscriptionResumed(metadata=_event_metadata(replay=True)),
        SubscriptionHalted(metadata=_event_metadata(replay=True)),
        SubscriptionCancelled(metadata=_event_metadata(replay=True)),
    ],
)
def test_every_replayed_event_returns_original_snapshot_without_effects(event: object) -> None:
    snapshot = _snapshot(ActiveState(period_start=START, period_end=END), version=7)
    result = SubscriptionMachine().plan(snapshot, event, PolicyContext(now=NOW))  # type: ignore[arg-type]

    assert result.unwrap().outcome is TransitionOutcome.SKIP
    assert result.unwrap().after == snapshot
    assert result.unwrap().effects == ()
    assert result.unwrap().after.version == 7


def test_event_timestamp_must_be_timezone_aware() -> None:
    event = SubscriptionAuthenticated(
        metadata=EventMetadata(event_id="event-1", occurred_at=NOW.replace(tzinfo=None))
    )

    result = SubscriptionMachine().plan(_snapshot(CreatedState()), event, PolicyContext(now=NOW))

    assert isinstance(result, Failure)
    assert "timezone" in result.failure().message


@pytest.mark.parametrize("duration_days", [0, 366])
def test_pause_duration_is_bounded_inside_the_kernel(duration_days: int) -> None:
    result = SubscriptionMachine().plan(
        _snapshot(ActiveState(period_start=START, period_end=END)),
        PauseSubscription(metadata=_command_metadata(), duration_days=duration_days),
        PolicyContext(now=NOW),
    )

    assert isinstance(result, Failure)
    assert "between 1 and 365" in result.failure().message


@pytest.mark.parametrize(
    ("effective_at", "message_fragment"),
    [
        (START - timedelta(seconds=1), "precede"),
        (END + timedelta(seconds=1), "follow"),
    ],
)
def test_plan_change_checks_both_period_bounds_without_financial_context(
    effective_at: datetime, message_fragment: str
) -> None:
    result = SubscriptionMachine().plan(
        _snapshot(ActiveState(period_start=START, period_end=END)),
        ChangeSubscriptionPlan(
            metadata=_command_metadata(), new_plan_id="plan-2", effective_at=effective_at
        ),
        PolicyContext(now=NOW),
    )

    assert isinstance(result, Failure)
    assert message_fragment in result.failure().message


@pytest.mark.parametrize(
    ("current_amount", "target_amount"),
    [(100, None), (None, 200)],
)
def test_plan_change_requires_both_or_neither_financial_amount(
    current_amount: int | None, target_amount: int | None
) -> None:
    result = SubscriptionMachine().plan(
        _snapshot(ActiveState(period_start=START, period_end=END)),
        ChangeSubscriptionPlan(
            metadata=_command_metadata(), new_plan_id="plan-2", effective_at=NOW
        ),
        PolicyContext(
            now=NOW,
            current_plan_amount=current_amount,
            target_plan_amount=target_amount,
        ),
    )

    assert isinstance(result, Failure)
    assert "provided together" in result.failure().message


def test_omitted_period_facts_are_preserved_and_noop_does_not_bump_version() -> None:
    snapshot = _snapshot(ActiveState(period_start=START, period_end=END), version=3)
    result = (
        SubscriptionMachine()
        .plan(
            snapshot,
            SubscriptionCharged(metadata=_event_metadata(), period_start=None, period_end=None),
            PolicyContext(now=NOW),
        )
        .unwrap()
    )

    assert result.outcome is TransitionOutcome.NO_OP
    assert result.after == snapshot
    assert result.after.version == 3
    assert result.effects == ()


def test_regressive_period_facts_are_skipped_without_erasing_current_facts() -> None:
    snapshot = _snapshot(ActiveState(period_start=START, period_end=END), version=3)
    result = (
        SubscriptionMachine()
        .plan(
            snapshot,
            SubscriptionCharged(
                metadata=_event_metadata(),
                period_start=START - timedelta(days=1),
                period_end=END - timedelta(days=1),
            ),
            PolicyContext(now=NOW),
        )
        .unwrap()
    )

    assert result.outcome is TransitionOutcome.SKIP
    assert result.after == snapshot
    assert result.effects == ()


def test_paused_and_halted_subscriptions_do_not_reactivate_on_payment_capture() -> None:
    machine = SubscriptionMachine()
    for state in (PausedState(start=NOW), HaltedState()):
        result = machine.plan(
            _snapshot(state),
            PaymentCaptured(metadata=_event_metadata(), payment_id="pay-1", amount_paisa=1000),
            PolicyContext(now=NOW),
        ).unwrap()
        assert result.outcome is TransitionOutcome.SKIP
        assert result.after.state == state
        assert result.effects == ()


def test_plan_change_updates_desired_plan_without_overwriting_observed_plan() -> None:
    snapshot = SubscriptionSnapshot(
        id="sub-1",
        user_id="user-1",
        plan_id="plan-1",
        state=ActiveState(period_start=START, period_end=END),
        observed_plan_id="plan-1",
    )
    result = (
        SubscriptionMachine()
        .plan(
            snapshot,
            ChangeSubscriptionPlan(
                metadata=_command_metadata(), new_plan_id="plan-2", effective_at=NOW
            ),
            PolicyContext(now=NOW),
        )
        .unwrap()
    )

    assert result.after.plan_id == "plan-2"
    assert result.after.observed_plan_id == "plan-1"
    assert result.after.remote_sync is RemoteSyncState.PENDING


def test_effective_date_before_period_start_is_rejected() -> None:
    result = remaining_fraction(
        period_start=START,
        period_end=END,
        effective_at=START - timedelta(seconds=1),
    )

    assert isinstance(result, Failure)
    assert "precede" in result.failure().message


def test_proration_requires_explicit_timezone_and_uses_decimal_time() -> None:
    result = calculate_proration(
        subscription_id="sub-1",
        current_plan_id="plan-1",
        new_plan_id="plan-2",
        current_amount_paisa=3100,
        target_amount_paisa=6200,
        period_start=START,
        period_end=END,
        effective_at=NOW,
        tax_rate=Decimal("0.18"),
    )

    decision = result.unwrap()
    assert decision.direction == "upgrade"
    assert decision.amount_paisa > 0
    assert decision.remaining_fraction == Decimal(17) / Decimal(31)


def test_cancellation_full_policy_refunds_unused_value_not_full_capture() -> None:
    result = calculate_cancellation_refund(
        policy="FULL",
        captured_amount_paisa=3100,
        period_start=START,
        period_end=END,
        effective_at=NOW,
    ).unwrap()

    assert result.amount_paisa == 1700
    assert result.amount_paisa < 3100


@pytest.mark.parametrize(
    "event",
    [
        PaymentCaptured(metadata=_event_metadata(), payment_id="pay-1", amount_paisa=1000),
        PaymentFailed(metadata=_event_metadata(), payment_id="pay-2"),
        SubscriptionCancelled(metadata=_event_metadata()),
    ],
)
def test_terminal_cancelled_state_never_reactivates_from_provider_events(event: object) -> None:
    machine = SubscriptionMachine()
    result = machine.plan(
        _snapshot(CancelledState()),
        event,  # type: ignore[arg-type]
        PolicyContext(now=NOW),
    ).unwrap()

    assert result.outcome is TransitionOutcome.SKIP
    assert result.after.state.kind is SubscriptionStateKind.CANCELLED
