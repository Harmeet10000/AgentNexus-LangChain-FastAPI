"""Refund idempotency ledger: each Razorpay refund counts exactly once."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING

from returns.result import Success

from app.features.billing.payments.dto import RefundRequestDTO
from app.features.billing.payments.service import PaymentService

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
        self, payment: SimpleNamespace, *, status: str, extra_values: dict | None = None
    ) -> Any:
        payment.status = status
        for key, value in (extra_values or {}).items():
            setattr(payment, key, value)
        return Success(payment)


class _FakeAudit:
    async def create(self, entry: object) -> Any:
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


async def test_new_webhook_refund_increments_once() -> None:
    payment = _payment(metadata_={"processed_refund_ids": ["rfnd_1"]})
    service, _ = _service(payment)

    result = await service.handle_refund_processed(
        razorpay_payment_id="pay_rz_1", refund_paisa=2000, refund_id="rfnd_2"
    )

    assert isinstance(result, Success)
    assert payment.metadata_["processed_refund_ids"] == ["rfnd_1", "rfnd_2"]


async def test_webhook_without_refund_id_keeps_legacy_behavior() -> None:
    payment = _payment()
    service, _ = _service(payment)

    result = await service.handle_refund_processed(
        razorpay_payment_id="pay_rz_1", refund_paisa=2000, refund_id=None
    )

    assert isinstance(result, Success)
    assert payment.refund_amount == Decimal("20.00")
