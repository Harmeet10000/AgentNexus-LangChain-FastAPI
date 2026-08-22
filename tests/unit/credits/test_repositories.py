"""Unit tests for CreditRepository and ConsumptionRepository."""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

# Shim to avoid circular imports
_PKG = "app.features"
if _PKG not in sys.modules:
    from pathlib import Path

    _project_root = Path(__file__).resolve().parents[3]
    _pkg_mod = types.ModuleType(_PKG)
    _pkg_mod.__path__ = [str(_project_root / "src" / "app" / "features")]
    _pkg_mod.__package__ = _PKG
    sys.modules[_PKG] = _pkg_mod

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

from returns.result import Success  # noqa: E402

from app.features.credits.models.credit import CreditStatus  # noqa: E402
from app.features.credits.repositories.consumption_repository import (
    ConsumptionRepository,  # noqa: E402
)
from app.features.credits.repositories.credit_repository import CreditRepository  # noqa: E402


def _mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


def _make_row(**overrides):
    """Create a mock ORM row with credit-like attributes."""
    row = MagicMock()
    row.id = overrides.get("id", uuid4())
    row.user_id = overrides.get("user_id", "user-1")
    row.credit_type = overrides.get("credit_type", "plan_credit")
    row.credit_amount = overrides.get("credit_amount", 1000)
    row.remaining_balance = overrides.get("remaining_balance", 1000)
    row.status = overrides.get("status", CreditStatus.ACTIVE.value)
    row.valid_from = overrides.get("valid_from", datetime.now(tz=UTC))
    row.valid_until = overrides.get("valid_until", None)
    row.consumed_at = overrides.get("consumed_at", None)
    row.metadata_ = overrides.get("metadata_", {})
    row.created_at = overrides.get("created_at", datetime.now(tz=UTC))
    return row


def _run(coro):
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestCreditRepository:
    """12.5 - CreditRepository CRUD."""

    def test_create_adds_and_flushes(self):
        session = _mock_session()
        repo = CreditRepository(session)
        credit = _make_row()

        result = _run(repo.create(credit))

        assert isinstance(result, Success)
        session.add.assert_called_once_with(credit)
        session.flush.assert_awaited_once()

    def test_find_by_id_returns_credit(self):
        session = _mock_session()
        credit = _make_row()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = credit
        session.execute.return_value = mock_result

        repo = CreditRepository(session)
        result = _run(repo.find_by_id(credit.id))

        assert isinstance(result, Success)
        assert result.unwrap() is credit

    def test_find_by_id_returns_none(self):
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        repo = CreditRepository(session)
        result = _run(repo.find_by_id(uuid4()))

        assert isinstance(result, Success)
        assert result.unwrap() is None

    def test_get_active_balance_sums_remaining(self):
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 5000
        session.execute.return_value = mock_result

        repo = CreditRepository(session)
        result = _run(repo.get_active_balance("user-1"))

        assert isinstance(result, Success)
        assert result.unwrap() == 5000

    def test_get_active_balance_zero_when_no_credits(self):
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 0
        session.execute.return_value = mock_result

        repo = CreditRepository(session)
        result = _run(repo.get_active_balance("user-1"))

        assert isinstance(result, Success)
        assert result.unwrap() == 0

    def test_find_available_for_consumption_filters_active(self):
        session = _mock_session()
        credit = _make_row(status=CreditStatus.ACTIVE.value)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [credit]
        session.execute.return_value = mock_result

        repo = CreditRepository(session)
        result = _run(repo.find_available_for_consumption("user-1"))

        assert isinstance(result, Success)
        assert len(result.unwrap()) == 1

    def test_expire_credits_past_date_returns_empty_when_none(self):
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute.return_value = mock_result

        repo = CreditRepository(session)
        result = _run(repo.expire_credits_past_date(datetime.now(tz=UTC)))

        assert isinstance(result, Success)
        assert result.unwrap() == []


class TestConsumptionRepository:
    """12.5 - ConsumptionRepository CRUD."""

    def test_create_adds_and_flushes(self):
        session = _mock_session()
        repo = ConsumptionRepository(session)
        consumption = MagicMock()
        consumption.credit_id = uuid4()
        consumption.user_id = "user-1"

        result = _run(repo.create(consumption))

        assert isinstance(result, Success)
        session.add.assert_called_once_with(consumption)

    def test_get_total_consumed_sums_amount(self):
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 750
        session.execute.return_value = mock_result

        repo = ConsumptionRepository(session)
        result = _run(repo.get_total_consumed(uuid4()))

        assert isinstance(result, Success)
        assert result.unwrap() == 750

    def test_find_by_credit_id_returns_list(self):
        session = _mock_session()
        consumption = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [consumption]
        session.execute.return_value = mock_result

        repo = ConsumptionRepository(session)
        result = _run(repo.find_by_credit_id(uuid4()))

        assert isinstance(result, Success)
        assert len(result.unwrap()) == 1

    def test_find_by_invoice_id_returns_record(self):
        session = _mock_session()
        consumption = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = consumption
        session.execute.return_value = mock_result

        repo = ConsumptionRepository(session)
        result = _run(repo.find_by_invoice_id(uuid4()))

        assert isinstance(result, Success)
        assert result.unwrap() is consumption
