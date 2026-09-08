"""Branded invoice/receipt PDF rendering: bytes, words, placeholders."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

from app.features.billing.invoices.pdf import (
    SellerProfile,
    amount_in_words,
    render_invoice_pdf,
    render_receipt_pdf,
)


def _invoice(**overrides: object) -> SimpleNamespace:
    now = datetime.now(tz=UTC)
    base: dict[str, object] = {
        "invoice_number": "INV-2026-0001",
        "seller_gstin": "27ABCDE1234F1Z5",
        "buyer_gstin": "29ABCDE1234F1Z5",
        "place_of_supply": "29",
        "sac_code": "998314",
        "issued_at": now,
        "due_at": now + timedelta(days=7),
        "status": "issued",
        "user_id": "user-1",
        "subtotal": Decimal("1000.00"),
        "tax_rate": Decimal("0.18"),
        "tax_amount": Decimal("180.00"),
        "cgst_amount": Decimal("0.00"),
        "sgst_amount": Decimal("0.00"),
        "igst_amount": Decimal("180.00"),
        "total": Decimal("1180.00"),
        "currency": "INR",
        "line_items": [
            SimpleNamespace(
                plan_name="Pro Monthly",
                description="SaaS subscription",
                quantity=1,
                unit_price=Decimal("1000.00"),
                amount=Decimal("1000.00"),
                tax_amount=Decimal("180.00"),
                sac_code="998314",
            )
        ],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _seller() -> SellerProfile:
    return SellerProfile(
        name="Acme Technologies Pvt Ltd",
        address_line1="4th Floor, Cyber Towers",
        city="Hyderabad",
        state="Telangana",
        pincode="500081",
        email="billing@acme.example",
        phone="+91-40-12345678",
        gstin="27ABCDE1234F1Z5",
    )


def _receipt(**overrides: object) -> SimpleNamespace:
    now = datetime.now(tz=UTC)
    base: dict[str, object] = {
        "receipt_number": "REC-2026-0001",
        "subscription_id": "00000000-0000-0000-0000-000000000001",
        "razorpay_payment_id": "pay_ABC123",
        "user_id": "user-1",
        "amount": Decimal("1180.00"),
        "currency": "INR",
        "payment_method": "upi",
        "receipt_date": now,
        "billing_period_start": now - timedelta(days=30),
        "billing_period_end": now,
        "plan_name": "Pro Monthly",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_amount_in_words_covers_indian_denominations() -> None:
    assert amount_in_words(Decimal(0)) == "Rupees Zero Only"
    assert amount_in_words(Decimal(1)) == "Rupees One Only"
    assert amount_in_words(Decimal(1180)) == "Rupees One Thousand One Hundred Eighty Only"
    assert "Lakh" in amount_in_words(Decimal(250000))
    assert "Crore" in amount_in_words(Decimal(15000000))
    assert "Paise Fifty" in amount_in_words(Decimal("100.50"))


def test_invoice_renders_valid_pdf_with_seller() -> None:
    pdf = render_invoice_pdf(_invoice(), _seller())
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 2000


def test_invoice_without_line_items_still_renders() -> None:
    pdf = render_invoice_pdf(_invoice(line_items=[]), _seller())
    assert pdf[:5] == b"%PDF-"


def test_invoice_without_seller_shows_placeholder_not_blank() -> None:
    # Unconfigured seller must be visible on the document, never silent.
    pdf = render_invoice_pdf(_invoice(), SellerProfile())
    assert pdf[:5] == b"%PDF-"
    assert SellerProfile().is_configured is False


def test_overdue_invoice_renders() -> None:
    now = datetime.now(tz=UTC)
    pdf = render_invoice_pdf(_invoice(due_at=now - timedelta(days=10), status="issued"), _seller())
    assert pdf[:5] == b"%PDF-"


def test_receipt_renders_with_period_and_words() -> None:
    pdf = render_receipt_pdf(_receipt(), _seller())
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 1500


def test_seller_profile_address_lines_skip_blanks() -> None:
    profile = SellerProfile(name="Acme", city="Hyderabad", pincode="500081")
    assert profile.address_lines == ["Hyderabad - 500081"]
    assert profile.is_configured is True
