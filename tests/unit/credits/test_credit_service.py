"""Unit tests for CreditService."""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

# ---------------------------------------------------------------------------
# Shim: prevent circular imports by stubbing problematic modules first
# ---------------------------------------------------------------------------
_PKG = "app.features"
if _PKG not in sys.modules:
    from pathlib import Path

    _project_root = Path(__file__).resolve().parents[3]
    _pkg_mod = types.ModuleType(_PKG)
    _pkg_mod.__path__ = [str(_project_root / "src" / "app" / "features")]
    _pkg_mod.__package__ = _PKG
    sys.modules[_PKG] = _pkg_mod

# Stub heavy connection modules to break circular import chain
# app.connections.__init__ imports celery, crawl4ai, mongodb, etc → all trigger app.utils circular
for _mod in (
    "app.connections.celery",
    "app.connections.crawl4ai",
    "app.connections.httpx_client",
    "app.connections.mongodb",
    "app.connections.neo4j",
    "app.connections.postgres",
    "app.connections.redis",
    "app.connections.tavily",
    "app.utils.cache",
    "app.utils.cache.redis_func",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest  # noqa: E402
from returns.result import Success  # noqa: E402

from app.features.credits.dto.consumption_dto import CreditConsumptionResult  # noqa: E402
from app.features.credits.dto.credit_dto import (  # noqa: E402
    CreditBalanceResponse,
    CreditGrantDTO,
    CreditGrantResponse,
    CreditHistoryResponse,
)
from app.features.credits.models.credit import CreditStatus, CreditType, UserCredit  # noqa: E402
from app.features.credits.services.credit_service import CreditService  # noqa: E402


def _make_credit(
    *,
    remaining_balance: int = 500,
    credit_amount: int = 500,
    status: str = CreditStatus.ACTIVE.value,
    valid_until: datetime | None = None,
    credit_type: str = CreditType.PLAN_CREDIT.value,
) -> UserCredit:
    credit = MagicMock(spec=UserCredit)
    credit.id = uuid4()
    credit.user_id = "user-123"
    credit.credit_type = credit_type
    credit.credit_amount = credit_amount
    credit.remaining_balance = remaining_balance
    credit.valid_from = datetime.now(tz=UTC)
    credit.valid_until = valid_until
    credit.status = status
    credit.metadata_ = {}
    credit.created_at = datetime.now(tz=UTC)
    credit.consumed_at = None
    return credit


def _run(coro):
    """Run an async coroutine synchronously for test convenience."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestGrantCredit:
    """12.1 - Grant operations."""

    def test_grant_credit_valid_data(self):
        """grant_credit with valid data returns CreditGrantResponse."""
        svc = CreditService(
            session=AsyncMock(),
            credit_repo=AsyncMock(),
            consumptions=AsyncMock(),
            audit=AsyncMock(),
        )
        credit = _make_credit()
        svc.credit_repo.create = AsyncMock(return_value=Success(credit))

        dto = CreditGrantDTO(
            credit_type=CreditType.PLAN_CREDIT.value,
            credit_amount=500,
        )

        result = _run(svc.grant_credit("user-123", dto))

        assert isinstance(result, CreditGrantResponse)
        assert result.credit_amount == 500
        assert result.remaining_balance == 500
        assert result.user_id == "user-123"

    def test_grant_credit_admin_grant_missing_metadata(self):
        """ADMIN_GRANT without admin_user_id in metadata raises ValidationException."""
        from app.utils.exceptions import ValidationException

        with pytest.raises(ValidationException):
            CreditGrantDTO(
                credit_type=CreditType.ADMIN_GRANT.value,
                credit_amount=500,
                metadata_={},
            )

    def test_grant_credit_amount_zero(self):
        """credit_amount=0 raises ValidationError (DTO field constraint ge=1)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreditGrantDTO(
                credit_type=CreditType.PROMOTIONAL.value,
                credit_amount=0,
            )

    def test_grant_credit_creates_audit_log(self):
        """grant_credit creates an audit log entry."""
        svc = CreditService(
            session=AsyncMock(),
            credit_repo=AsyncMock(),
            consumptions=AsyncMock(),
            audit=AsyncMock(),
        )
        credit = _make_credit()
        svc.credit_repo.create = AsyncMock(return_value=Success(credit))

        dto = CreditGrantDTO(
            credit_type=CreditType.PLAN_CREDIT.value,
            credit_amount=500,
        )

        _run(svc.grant_credit("user-123", dto))

        svc.audit.create.assert_awaited_once()
        audit_call = svc.audit.create.call_args[0][0]
        assert audit_call.entity_type == "user_credit"
        assert audit_call.action == "credit.granted"
        assert audit_call.user_id == "user-123"
        assert audit_call.changes["credit_amount"] == 500


class TestConsumeCredits:
    """12.2 - Consumption operations."""

    def test_consume_full_coverage(self):
        """Credit covers entire invoice, cash_due=0."""
        svc = CreditService(
            session=AsyncMock(),
            credit_repo=AsyncMock(),
            consumptions=AsyncMock(),
            audit=AsyncMock(),
        )

        credit = _make_credit(remaining_balance=1000, credit_amount=1000)
        svc.credit_repo.find_available_for_consumption = AsyncMock(
            return_value=Success([credit])
        )
        updated_credit = _make_credit(remaining_balance=0)
        updated_credit.status = CreditStatus.CONSUMED.value
        svc.credit_repo.update_balance = AsyncMock(return_value=Success(updated_credit))
        svc.consumptions.create = AsyncMock(return_value=Success(MagicMock()))

        result = _run(
            svc.consume_credits(
                "user-123",
                invoice_id=uuid4(),
                invoice_gross_total=Decimal("10.00"),
                session=AsyncMock(),
            )
        )

        assert isinstance(result, CreditConsumptionResult)
        assert result.credit_applied == 1000
        assert result.cash_due == 0
        assert result.invoice_paid_in_full is True

    def test_consume_partial_coverage(self):
        """Credit covers part of invoice."""
        svc = CreditService(
            session=AsyncMock(),
            credit_repo=AsyncMock(),
            consumptions=AsyncMock(),
            audit=AsyncMock(),
        )

        credit = _make_credit(remaining_balance=300, credit_amount=500)
        svc.credit_repo.find_available_for_consumption = AsyncMock(
            return_value=Success([credit])
        )
        updated_credit = _make_credit(remaining_balance=0, credit_amount=500)
        updated_credit.status = CreditStatus.CONSUMED.value
        svc.credit_repo.update_balance = AsyncMock(return_value=Success(updated_credit))
        svc.consumptions.create = AsyncMock(return_value=Success(MagicMock()))

        result = _run(
            svc.consume_credits(
                "user-123",
                invoice_id=uuid4(),
                invoice_gross_total=Decimal("10.00"),
                session=AsyncMock(),
            )
        )

        assert result.credit_applied == 300
        assert result.cash_due == 700
        assert result.invoice_paid_in_full is False

    def test_consume_no_credits_available(self):
        """No credits available, all cash due."""
        svc = CreditService(
            session=AsyncMock(),
            credit_repo=AsyncMock(),
            consumptions=AsyncMock(),
            audit=AsyncMock(),
        )

        svc.credit_repo.find_available_for_consumption = AsyncMock(
            return_value=Success([])
        )

        result = _run(
            svc.consume_credits(
                "user-123",
                invoice_id=uuid4(),
                invoice_gross_total=Decimal("10.00"),
                session=AsyncMock(),
            )
        )

        assert result.credit_applied == 0
        assert result.cash_due == 1000
        assert result.invoice_paid_in_full is False
        assert result.credits_consumed == []

    def test_consume_respects_order_soonest_expiring(self):
        """Credits consumed in soonest-expiring-first order."""
        svc = CreditService(
            session=AsyncMock(),
            credit_repo=AsyncMock(),
            consumptions=AsyncMock(),
            audit=AsyncMock(),
        )

        now = datetime.now(tz=UTC)
        credit_soon = _make_credit(
            remaining_balance=200,
            valid_until=now + timedelta(days=5),
        )
        credit_later = _make_credit(
            remaining_balance=200,
            valid_until=now + timedelta(days=30),
        )
        credit_no_expiry = _make_credit(
            remaining_balance=200,
            valid_until=None,
        )
        # Order as the repo would return them (soonest first, no-expiry last)
        svc.credit_repo.find_available_for_consumption = AsyncMock(
            return_value=Success([credit_soon, credit_later, credit_no_expiry])
        )

        async def fake_update(credit, *, new_remaining_balance, new_status=None, consumed_at=None):
            updated = _make_credit(
                remaining_balance=new_remaining_balance,
                credit_amount=credit.credit_amount,
            )
            if new_status is not None:
                updated.status = new_status.value
            return Success(updated)

        svc.credit_repo.update_balance = AsyncMock(side_effect=fake_update)
        svc.consumptions.create = AsyncMock(return_value=Success(MagicMock()))

        result = _run(
            svc.consume_credits(
                "user-123",
                invoice_id=uuid4(),
                invoice_gross_total=Decimal("3.00"),
                session=AsyncMock(),
            )
        )

        # 3 credits at 200 each, invoice is 300 paisa → first 2 consumed, third untouched
        assert result.credit_applied == 300
        assert result.cash_due == 0

        # Verify update_balance was called on the soonest-expiring credit first
        first_call_credit = svc.credit_repo.update_balance.call_args_list[0][0][0]
        assert first_call_credit is credit_soon

    def test_consume_creates_audit_log(self):
        """consume_credits creates an audit log entry."""
        svc = CreditService(
            session=AsyncMock(),
            credit_repo=AsyncMock(),
            consumptions=AsyncMock(),
            audit=AsyncMock(),
        )

        credit = _make_credit(remaining_balance=500, credit_amount=500)
        svc.credit_repo.find_available_for_consumption = AsyncMock(
            return_value=Success([credit])
        )
        updated_credit = _make_credit(remaining_balance=0)
        updated_credit.status = CreditStatus.CONSUMED.value
        svc.credit_repo.update_balance = AsyncMock(return_value=Success(updated_credit))
        svc.consumptions.create = AsyncMock(return_value=Success(MagicMock()))

        invoice_id = uuid4()
        _run(
            svc.consume_credits(
                "user-123",
                invoice_id=invoice_id,
                invoice_gross_total=Decimal("5.00"),
                session=AsyncMock(),
            )
        )

        svc.audit.create.assert_awaited_once()
        audit_call = svc.audit.create.call_args[0][0]
        assert audit_call.action == "credit.consumed"
        assert audit_call.changes["invoice_id"] == str(invoice_id)


class TestCreditBalance:
    """12.3 - Balance."""

    def test_get_credit_balance_returns_paisa_and_rupees(self):
        """get_credit_balance returns total_balance in paisa and rupees."""
        svc = CreditService(
            session=AsyncMock(),
            credit_repo=AsyncMock(),
            consumptions=AsyncMock(),
            audit=AsyncMock(),
        )
        svc.credit_repo.get_active_balance = AsyncMock(return_value=Success(5000))

        result = _run(svc.get_credit_balance("user-123"))

        assert isinstance(result, CreditBalanceResponse)
        assert result.total_balance == 5000
        assert result.total_balance_rupees == 50.0


class TestCreditHistory:
    """12.4 - History."""

    def test_get_credit_history_with_pagination(self):
        """get_credit_history returns credits and consumptions with pagination."""
        svc = CreditService(
            session=AsyncMock(),
            credit_repo=AsyncMock(),
            consumptions=AsyncMock(),
            audit=AsyncMock(),
        )

        credit = _make_credit()
        svc.credit_repo.find_by_user = AsyncMock(
            return_value=Success(([credit], 1))
        )

        consumption = MagicMock()
        consumption.id = uuid4()
        consumption.credit_id = uuid4()
        consumption.consumed_amount = 200
        consumption.invoice_id = None
        consumption.razorpay_payment_id = None
        consumption.created_at = datetime.now(tz=UTC)
        svc.consumptions.find_by_user = AsyncMock(
            return_value=Success(([consumption], 1))
        )

        result = _run(
            svc.get_credit_history("user-123", limit=10, offset=0)
        )

        assert isinstance(result, CreditHistoryResponse)
        assert len(result.credits) == 1
        assert len(result.consumptions) == 1
        assert result.total == 2
        assert result.limit == 10
        assert result.offset == 0
