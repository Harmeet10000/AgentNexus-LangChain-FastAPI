"""Persisted Saul budget semantics: consume, resume, and fail closed."""

from datetime import UTC, datetime, timedelta

import pytest

from app.features.agent_saul.budget import (
    BudgetDimension,
    BudgetExhaustedError,
    BudgetLedger,
    BudgetLimits,
)


def test_ledger_usage_survives_checkpoint_round_trip() -> None:
    ledger = BudgetLedger(limits=BudgetLimits(model_calls=2))
    consumed = ledger.consume(BudgetDimension.MODEL_CALLS)
    restored = BudgetLedger.model_validate_json(consumed.model_dump_json())
    assert restored.model_calls == 1
    assert restored.limits.model_calls == 2
    restored = restored.consume(BudgetDimension.MODEL_CALLS)
    with pytest.raises(BudgetExhaustedError):
        restored.consume(BudgetDimension.MODEL_CALLS)


def test_token_usage_is_checked_when_provider_reports_tokens() -> None:
    ledger = BudgetLedger(limits=BudgetLimits(token_usage=3))
    with pytest.raises(BudgetExhaustedError) as error:
        ledger.consume(BudgetDimension.MODEL_CALLS, input_tokens=2, output_tokens=2)
    assert error.value.dimension is BudgetDimension.TOKEN_USAGE


def test_wall_clock_budget_is_fail_closed() -> None:
    ledger = BudgetLedger(
        limits=BudgetLimits(wall_clock_seconds=1),
        started_at=datetime.now(tz=UTC) - timedelta(seconds=2),
    )
    with pytest.raises(BudgetExhaustedError) as error:
        ledger.consume(BudgetDimension.TOOL_CALLS)
    assert error.value.dimension is BudgetDimension.WALL_CLOCK


def test_hitl_pause_excludes_waiting_time_but_preserves_active_elapsed() -> None:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    ledger = BudgetLedger(
        limits=BudgetLimits(wall_clock_seconds=10),
        started_at=started,
    )
    active = ledger.consume(BudgetDimension.TOOL_CALLS, now=started + timedelta(seconds=4))
    paused = active.pause(now=started + timedelta(seconds=5))
    resumed = paused.resume(now=started + timedelta(hours=8))

    # The eight-hour human review does not count, and resuming does not reset
    # the four seconds already consumed before the interrupt.
    continued = resumed.consume(
        BudgetDimension.TOOL_CALLS,
        now=started + timedelta(hours=8, seconds=4),
    )
    assert continued.active_elapsed_seconds == pytest.approx(9)


def test_suspended_ledger_round_trip_cannot_reset_wall_clock_budget() -> None:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    ledger = BudgetLedger(
        limits=BudgetLimits(wall_clock_seconds=10),
        started_at=started,
    ).consume(BudgetDimension.TOOL_CALLS, now=started + timedelta(seconds=8))
    checkpoint = ledger.pause(now=started + timedelta(seconds=9))
    restored = BudgetLedger.model_validate_json(checkpoint.model_dump_json()).resume(
        now=started + timedelta(days=2)
    )

    with pytest.raises(BudgetExhaustedError) as error:
        restored.consume(BudgetDimension.TOOL_CALLS, now=started + timedelta(days=2, seconds=3))
    assert error.value.dimension is BudgetDimension.WALL_CLOCK
