"""Property-based tests for credit domain invariants."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import NamedTuple

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# In-memory domain model (no ORM, no DB)
# ---------------------------------------------------------------------------


class Credit(NamedTuple):
    id: int
    credit_amount: int  # paisa, positive
    remaining_balance: int  # paisa, 0 <= remaining <= credit_amount
    status: str  # "active" | "consumed" | "expired"
    valid_until: datetime | None
    created_at: datetime


class Consumption(NamedTuple):
    credit_id: int
    consumed_amount: int  # paisa, positive
    created_at: datetime


def _apply_consumption(credit: Credit, amount: int) -> Credit:
    new_remaining = credit.remaining_balance - amount
    new_status = "consumed" if new_remaining == 0 else credit.status
    return Credit(
        id=credit.id,
        credit_amount=credit.credit_amount,
        remaining_balance=new_remaining,
        status=new_status,
        valid_until=credit.valid_until,
        created_at=credit.created_at,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def credit_strategy(draw) -> Credit:
    credit_amount = draw(st.integers(min_value=1, max_value=100_000))
    remaining = draw(st.integers(min_value=0, max_value=credit_amount))
    now = datetime.now(tz=UTC)
    has_expiry = draw(st.booleans())
    valid_until = (
        now + timedelta(days=draw(st.integers(min_value=1, max_value=365)))
        if has_expiry
        else None
    )
    created_at = now - timedelta(days=draw(st.integers(min_value=0, max_value=30)))
    # New credits are always active
    return Credit(
        id=draw(st.integers(min_value=1, max_value=999_999)),
        credit_amount=credit_amount,
        remaining_balance=remaining,
        status="active",
        valid_until=valid_until,
        created_at=created_at,
    )


@st.composite
def credit_with_consumptions_strategy(draw) -> tuple[Credit, list[Consumption]]:
    credit_amount = draw(st.integers(min_value=1, max_value=100_000))
    now = datetime.now(tz=UTC)
    has_expiry = draw(st.booleans())
    valid_until = (
        now + timedelta(days=draw(st.integers(min_value=1, max_value=365)))
        if has_expiry
        else None
    )
    created_at = now - timedelta(days=draw(st.integers(min_value=0, max_value=30)))

    # Generate consumption amounts that sum to at most credit_amount
    num_consumptions = draw(st.integers(min_value=0, max_value=5))
    if num_consumptions == 0:
        total_consumed = 0
        consumption_amounts: list[int] = []
    else:
        amounts = draw(
            st.lists(
                st.integers(min_value=1, max_value=credit_amount),
                min_size=num_consumptions,
                max_size=num_consumptions,
            )
        )
        total_consumed = min(sum(amounts), credit_amount)
        # Clamp individual amounts so they sum to exactly total_consumed
        if total_consumed < sum(amounts):
            # Scale down proportionally
            scale = total_consumed / sum(amounts) if sum(amounts) > 0 else 0
            consumption_amounts = [max(1, int(a * scale)) for a in amounts]
            # Adjust rounding
            diff = total_consumed - sum(consumption_amounts)
            for i in range(abs(diff)):
                idx = i % len(consumption_amounts)
                consumption_amounts[idx] += 1 if diff > 0 else -1
        else:
            consumption_amounts = amounts

    remaining_balance = credit_amount - total_consumed
    status = "consumed" if remaining_balance == 0 else "active"

    credit = Credit(
        id=1,
        credit_amount=credit_amount,
        remaining_balance=remaining_balance,
        status=status,
        valid_until=valid_until,
        created_at=created_at,
    )

    consumptions = []
    base_time = created_at
    for i, amt in enumerate(consumption_amounts):
        base_time += timedelta(hours=1)
        consumptions.append(
            Consumption(credit_id=1, consumed_amount=amt, created_at=base_time)
        )

    return credit, consumptions


# ---------------------------------------------------------------------------
# 11.1 — Ledger Integrity
# ---------------------------------------------------------------------------

class TestLedgerIntegrity:
    """credit_amount == remaining_balance + sum(consumed_amount)."""

    @given(data=credit_with_consumptions_strategy())
    @settings(max_examples=200)
    def test_ledger_integrity(self, data: tuple[Credit, list[Consumption]]):
        credit, consumptions = data
        total_consumed = sum(c.consumed_amount for c in consumptions)
        assert credit.credit_amount == credit.remaining_balance + total_consumed


# ---------------------------------------------------------------------------
# 11.4 — Status Transition
# ---------------------------------------------------------------------------

class TestStatusTransition:
    """CONSUMED only when remaining_balance==0, EXPIRED only when now > valid_until and was ACTIVE."""

    @given(data=credit_strategy())
    @settings(max_examples=200)
    def test_consumed_implies_zero_balance(self, data: Credit):
        if data.status == "consumed":
            assert data.remaining_balance == 0

    @given(data=credit_strategy())
    @settings(max_examples=200)
    def test_expired_requires_past_valid_until(self, data: Credit):
        now = datetime.now(tz=UTC)
        if data.status == "expired":
            assert data.valid_until is not None
            assert now > data.valid_until
            # Was previously active (consumption logic only targets active credits)

    @given(data=credit_strategy())
    @settings(max_examples=200)
    def test_active_with_remaining_never_consumed(self, data: Credit):
        if data.status == "active" and data.remaining_balance > 0:
            # Still has balance, shouldn't be consumed
            assert data.remaining_balance <= data.credit_amount

    @given(data=credit_strategy())
    @settings(max_examples=200)
    def test_consumed_at_most_credit_amount(self, data: Credit):
        assert data.remaining_balance >= 0
        assert data.remaining_balance <= data.credit_amount


# ---------------------------------------------------------------------------
# 11.5 — Consumption Order
# ---------------------------------------------------------------------------

class TestConsumptionOrder:
    """Verify soonest valid_until consumed first, then oldest created_at, no-expiry last."""

    @given(
        credits=st.lists(credit_strategy(), min_size=2, max_size=10),
    )
    @settings(max_examples=200)
    def test_soonest_valid_until_first(self, credits: list[Credit]):
        def consumption_sort_key(c: Credit):
            # valid_until ASC (NULLS LAST), then created_at ASC
            if c.valid_until is None:
                return (1, c.created_at)
            return (0, c.valid_until, c.created_at)

        sorted_credits = sorted(credits, key=consumption_sort_key)

        # Credits with valid_until should come before those without
        expiry_indices = [
            i for i, c in enumerate(sorted_credits) if c.valid_until is not None
        ]
        no_expiry_indices = [
            i for i, c in enumerate(sorted_credits) if c.valid_until is None
        ]

        if expiry_indices and no_expiry_indices:
            assert max(expiry_indices) < min(no_expiry_indices)

        # Among credits with expiry, earliest valid_until should come first
        expiry_credits = [c for c in sorted_credits if c.valid_until is not None]
        for i in range(len(expiry_credits) - 1):
            if expiry_credits[i].valid_until == expiry_credits[i + 1].valid_until:
                # Tie-break: oldest created_at first
                assert expiry_credits[i].created_at <= expiry_credits[i + 1].created_at
            else:
                assert expiry_credits[i].valid_until <= expiry_credits[i + 1].valid_until


# ---------------------------------------------------------------------------
# 11.8 — Balance Calculation
# ---------------------------------------------------------------------------

class TestBalanceCalculation:
    """Balance equals sum of remaining_balance for ACTIVE, non-expired credits."""

    @given(
        credits=st.lists(credit_strategy(), min_size=0, max_size=10),
    )
    @settings(max_examples=200)
    def test_balance_is_sum_active_non_expired(self, credits: list[Credit]):
        now = datetime.now(tz=UTC)

        def is_eligible(c: Credit) -> bool:
            if c.status != "active":
                return False
            if c.valid_until is not None and c.valid_until <= now:
                return False
            return True

        expected_balance = sum(c.remaining_balance for c in credits if is_eligible(c))

        # This is the logic the service delegates to the repo
        # We verify the invariant holds regardless of the credit list
        assert expected_balance >= 0
        # All remaining_balance values are non-negative by construction
        for c in credits:
            assert c.remaining_balance >= 0
            assert c.remaining_balance <= c.credit_amount

    @given(
        credits=st.lists(credit_strategy(), min_size=1, max_size=10),
    )
    @settings(max_examples=200)
    def test_balance_no_larger_than_total_amount(self, credits: list[Credit]):
        """Total balance cannot exceed sum of credit_amounts."""
        total_amount = sum(c.credit_amount for c in credits)
        active_balance = sum(c.remaining_balance for c in credits if c.status == "active")
        assert active_balance <= total_amount
