"""Refund idempotency ledger: each Razorpay refund counts exactly once."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from returns.result import Failure, Success

from app.features.billing.payments.dto import RefundRequestDTO
from app.features.billing.payments.exceptions import PaymentManualReviewError
from app.features.billing.payments.service import PaymentService
from app.features.billing.providers import ProviderReceipt
from app.features.billing.webhooks.service import WebhookService

if TYPE_CHECKING:
    from typing import Any


def _payment(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "id": "pay-1",
        "subscription_id": "sub-1",
        "razorpay_payment_id": "pay_rz_1",
        "amount": 10000,
        "currency": "INR",
        "status": "captured",
        "refund_amount": Decimal(0),
        "metadata_": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakePayments:
    def __init__(self, payment: SimpleNamespace) -> None:
        self.payment = payment
        self.locked_lookups = 0
        self.unlocked_lookups = 0
        self.update_calls = 0

    async def find_by_id_for_update(self, payment_id: str) -> Any:
        self.locked_lookups += 1
        return Success(self.payment)

    async def find_by_id(self, payment_id: str) -> Any:
        self.unlocked_lookups += 1
        return Success(self.payment)

    async def find_by_razorpay_id_for_update(self, razorpay_payment_id: str) -> Any:
        self.locked_lookups += 1
        return Success(self.payment)

    async def update_status(
        self,
        payment: SimpleNamespace,
        *,
        status: str,
        extra_values: dict[str, object] | None = None,
    ) -> Any:
        self.update_calls += 1
        payment.status = status
        for key, value in (extra_values or {}).items():
            setattr(payment, key, value)
        return Success(payment)


class _FakeAudit:
    def __init__(self) -> None:
        self.create_calls = 0

    async def create(self, entry: object) -> Any:
        self.create_calls += 1
        return Success(entry)


class _FakeRazorpay:
    def __init__(self, refund_id: str = "rfnd_1") -> None:
        self.refund_id = refund_id

    async def create_refund(self, **kwargs: object) -> dict[str, str]:
        return {"id": self.refund_id}


def _service(
    payment: SimpleNamespace, refund_id: str = "rfnd_1"
) -> tuple[PaymentService, _FakePayments]:
    repo = _FakePayments(payment)
    service = PaymentService(
        payments=repo,  # type: ignore[arg-type]
        audit=_FakeAudit(),  # type: ignore[arg-type]
        razorpay=_FakeRazorpay(refund_id),  # type: ignore[arg-type]
    )
    return service, repo


async def test_refund_records_id_and_uses_row_lock() -> None:
    payment = _payment()
    service, repo = _service(payment)

    result = await service.refund("pay-1", RefundRequestDTO(amount=2000), user_id="user-1")

    assert isinstance(result, Success)
    assert repo.locked_lookups == 1
    assert repo.unlocked_lookups == 0
    assert payment.metadata_["processed_refund_ids"] == ["rfnd_1"]
    assert payment.refund_amount == Decimal("20.00")


async def test_duplicate_webhook_delivery_is_a_noop() -> None:
    payment = _payment(
        status="partially_refunded",
        refund_amount=Decimal("20.00"),
        metadata_={"processed_refund_ids": ["rfnd_1"]},
    )
    service, _ = _service(payment)

    result = await service.handle_refund_processed(
        razorpay_payment_id="pay_rz_1", refund_paisa=2000, refund_id="rfnd_1"
    )

    assert isinstance(result, Success)
    assert payment.refund_amount == Decimal("20.00")
    assert payment.metadata_["processed_refund_ids"] == ["rfnd_1"]


@pytest.mark.parametrize(
    ("receipt_payment_id", "receipt_amount"),
    [("pay_rz_other", 2000), ("pay_rz_1", 1999)],
)
async def test_refund_receipt_mismatch_requires_reconciliation_without_local_projection(
    receipt_payment_id: str, receipt_amount: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    payment = _payment()
    service, _ = _service(payment)
    provider_receipt = ProviderReceipt("rfnd_mismatch", receipt_payment_id, receipt_amount)

    async def execute_malformed_provider(*_args: object, **_kwargs: object) -> Success[object]:
        return Success(SimpleNamespace(value=provider_receipt))

    monkeypatch.setattr(service, "_execute_refund_provider", execute_malformed_provider)

    result = await service.refund("pay-1", RefundRequestDTO(amount=2000), user_id="user-1")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), PaymentManualReviewError)
    details = result.failure().details
    assert details["manual_review"] is True
    assert details["requires_reconciliation"] is True
    assert details["provider_refund_id"] == "rfnd_mismatch"
    assert details["provider_payment_id"] == receipt_payment_id
    assert details["provider_amount_paisa"] == receipt_amount
    assert payment.refund_amount == Decimal(0)
    assert payment.status == "captured"
    assert payment.metadata_ == {}


async def test_new_webhook_refund_increments_once() -> None:
    payment = _payment(metadata_={"processed_refund_ids": ["rfnd_1"]})
    service, _ = _service(payment)

    result = await service.handle_refund_processed(
        razorpay_payment_id="pay_rz_1", refund_paisa=2000, refund_id="rfnd_2"
    )

    assert isinstance(result, Success)
    assert payment.metadata_["processed_refund_ids"] == ["rfnd_1", "rfnd_2"]


@pytest.mark.parametrize("refund_paisa", [0, -1, 8_001])
async def test_refund_processed_invalid_amount_requires_reconciliation_without_projection(
    refund_paisa: int,
) -> None:
    payment = _payment(
        status="partially_refunded",
        refund_amount=Decimal("20.00"),
    )
    service, repo = _service(payment)
    audit = _FakeAudit()
    service.audit = audit  # type: ignore[assignment]

    result = await service.handle_refund_processed(
        razorpay_payment_id="pay_rz_1", refund_paisa=refund_paisa, refund_id="rfnd_2"
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), PaymentManualReviewError)
    assert result.failure().details["manual_review"] is True
    assert result.failure().details["requires_reconciliation"] is True
    assert repo.update_calls == 0
    assert audit.create_calls == 0
    assert payment.refund_amount == Decimal("20.00")
    assert payment.status == "partially_refunded"
    assert payment.metadata_ == {}


async def test_duplicate_refund_id_is_noop_before_amount_validation() -> None:
    payment = _payment(
        status="partially_refunded",
        refund_amount=Decimal("20.00"),
        metadata_={"processed_refund_ids": ["rfnd_1"]},
    )
    service, repo = _service(payment)
    audit = _FakeAudit()
    service.audit = audit  # type: ignore[assignment]

    result = await service.handle_refund_processed(
        razorpay_payment_id="pay_rz_1", refund_paisa=99_999, refund_id="rfnd_1"
    )

    assert isinstance(result, Success)
    assert repo.update_calls == 0
    assert audit.create_calls == 0
    assert payment.refund_amount == Decimal("20.00")


async def test_refund_processed_manual_review_is_skipped_by_webhook() -> None:
    payment_service = SimpleNamespace(
        handle_refund_processed=AsyncMock(
            return_value=Failure(
                PaymentManualReviewError(
                    message="refund amount requires reconciliation",
                    details={"manual_review": True, "requires_reconciliation": True},
                )
            )
        )
    )
    service = WebhookService(
        webhooks=SimpleNamespace(),  # type: ignore[arg-type]
        subscriptions=SimpleNamespace(),  # type: ignore[arg-type]
        plans=SimpleNamespace(),  # type: ignore[arg-type]
        audit=SimpleNamespace(),  # type: ignore[arg-type]
        payment_service=payment_service,  # type: ignore[arg-type]
        invoice_service=SimpleNamespace(),  # type: ignore[arg-type]
    )

    result = await service._handle_refund_processed(  # type: ignore[attr-defined]
        {
            "refund": {
                "entity": {
                    "id": "rfnd_2",
                    "payment_id": "pay_rz_1",
                    "amount": 8_001,
                }
            }
        }
    )

    assert result == Success("skipped")
    payment_service.handle_refund_processed.assert_awaited_once_with(
        razorpay_payment_id="pay_rz_1",
        refund_paisa=8_001,
        refund_id="rfnd_2",
    )


async def test_webhook_without_refund_id_requires_manual_review_without_balance_change() -> None:
    payment = _payment()
    service, _ = _service(payment)

    result = await service.handle_refund_processed(
        razorpay_payment_id="pay_rz_1", refund_paisa=2000, refund_id=None
    )

    assert isinstance(result, Failure)
    assert payment.refund_amount == 0
