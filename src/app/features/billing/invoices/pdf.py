"""Branded PDF rendering for invoices and receipts (fpdf2, no extra deps).

Layout (A4): navy header band with seller identity, invoice meta box, buyer
block, bordered line-item table, right-aligned totals, amount in words
(Indian numbering), GST breakup, status stamp, and a footer with terms and
page numbers on every page.

Two deliberate constraints: fpdf2 core fonts are latin-1, so the rupee sign
(U+20B9) cannot render — amounts use "Rs." unconditionally. Seller identity
comes from ``SellerProfile`` (``BILLING_SELLER_*`` settings); when
unconfigured the PDF prints an explicit placeholder box instead of silently
omitting the seller, so a misconfigured deploy is visible on the document.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, override

from fpdf import FPDF
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from app.config import Settings

    from .model import Invoice
    from .receipt import PaymentReceipt

_NAVY: tuple[int, int, int] = (23, 58, 94)
_LIGHT_GREY: tuple[int, int, int] = (242, 244, 247)
_MID_GREY: tuple[int, int, int] = (148, 163, 184)
_GREEN: tuple[int, int, int] = (22, 101, 52)
_RED: tuple[int, int, int] = (153, 27, 27)
_AMBER: tuple[int, int, int] = (146, 64, 14)

_TERMS: tuple[str, ...] = (
    "1. Payment is due within 7 days of the invoice date unless agreed otherwise.",
    "2. This is a computer-generated document and needs no physical signature.",
    "3. Disputes are subject to the jurisdiction stated in the service agreement.",
)


class SellerProfile(BaseModel):
    """Display identity printed on every billing document."""

    model_config = ConfigDict(frozen=True)

    name: str = ""
    address_line1: str = ""
    address_line2: str = ""
    city: str = ""
    state: str = ""
    pincode: str = ""
    email: str = ""
    phone: str = ""
    gstin: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.name.strip())

    @property
    def address_lines(self) -> list[str]:
        lines = [self.address_line1, self.address_line2]
        city_line = ", ".join(part for part in (self.city, self.state) if part)
        if self.pincode:
            city_line = f"{city_line} - {self.pincode}" if city_line else self.pincode
        lines.append(city_line)
        return [line for line in lines if line.strip()]

    @classmethod
    def from_settings(cls, settings: Settings) -> SellerProfile:
        return cls(
            name=settings.BILLING_SELLER_NAME,
            address_line1=settings.BILLING_SELLER_ADDRESS_LINE1,
            address_line2=settings.BILLING_SELLER_ADDRESS_LINE2,
            city=settings.BILLING_SELLER_CITY,
            state=settings.BILLING_SELLER_STATE,
            pincode=settings.BILLING_SELLER_PINCODE,
            email=settings.BILLING_SELLER_EMAIL,
            phone=settings.BILLING_SELLER_PHONE,
            gstin=settings.BILLING_SELLER_GSTIN,
        )


_ONES: tuple[str, ...] = (
    "",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    "Thirteen",
    "Fourteen",
    "Fifteen",
    "Sixteen",
    "Seventeen",
    "Eighteen",
    "Nineteen",
)
_TENS: tuple[str, ...] = (
    "",
    "",
    "Twenty",
    "Thirty",
    "Forty",
    "Fifty",
    "Sixty",
    "Seventy",
    "Eighty",
    "Ninety",
)


def _two_digits(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return f"{_TENS[tens]} {_ONES[ones]}".strip()


def _three_digits(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    words = f"{_ONES[hundreds]} Hundred" if hundreds else ""
    if rest:
        words = f"{words} {_two_digits(rest)}".strip()
    return words


def _rupees_in_words(amount: int) -> str:
    """Spell a non-negative rupee amount using the Indian numbering system."""
    if amount == 0:
        return "Zero"
    parts: list[str] = []
    crore, amount = divmod(amount, 10_000_000)
    lakh, amount = divmod(amount, 100_000)
    thousand, amount = divmod(amount, 1_000)
    if crore:
        parts.append(f"{_three_digits(crore)} Crore")
    if lakh:
        parts.append(f"{_two_digits(lakh)} Lakh")
    if thousand:
        parts.append(f"{_two_digits(thousand)} Thousand")
    if amount:
        parts.append(_three_digits(amount))
    return " ".join(parts)


def amount_in_words(total: Decimal, currency: str = "INR") -> str:
    """Render a total as words, e.g. "Rupees Twelve Thousand Only"."""
    quantized = total.quantize(Decimal("0.01"))
    rupees = int(quantized)
    paise = int((quantized - Decimal(rupees)) * 100)
    words = f"Rupees {_rupees_in_words(rupees)}"
    if paise:
        words += f" and Paise {_two_digits(paise)}"
    words += " Only"
    if currency != "INR":
        words = f"{currency} {words}"
    return words


def _money(value: object) -> str:
    amount = value if isinstance(value, Decimal) else Decimal(str(value))
    return f"Rs. {amount:,.2f}"


def _date(value: object) -> str:
    if isinstance(value, datetime):
        return value.strftime("%d %b %Y")
    return "—"


class _BrandedPDF(FPDF):
    """FPDF with a terms + page-number footer on every page."""

    @override
    def footer(self) -> None:
        self.set_y(-18)
        self.set_font("helvetica", "I", 7)
        self.set_text_color(*_MID_GREY)
        self.cell(0, 4, _TERMS[1], align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", align="C")


def _new_doc() -> _BrandedPDF:
    pdf = _BrandedPDF(format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(True, margin=22)
    pdf.alias_nb_pages("{nb}")
    pdf.add_page()
    return pdf


def _header_band(pdf: _BrandedPDF, title: str, seller: SellerProfile) -> None:
    pdf.set_fill_color(*_NAVY)
    pdf.rect(0, 0, 210, 32, style="F")
    pdf.set_xy(12, 7)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 15)
    pdf.cell(120, 9, seller.name if seller.is_configured else "[SELLER NAME]", new_x="END")
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 9, title, align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(12)
    pdf.set_font("helvetica", "", 8)
    contact = "  |  ".join(part for part in (seller.email, seller.phone) if part)
    fallback = (seller.gstin and f"GSTIN: {seller.gstin}") or ""
    pdf.cell(120, 5, contact or fallback, new_x="END")
    pdf.cell(0, 5, "COMPUTER GENERATED", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(12, 35)
    pdf.set_text_color(0, 0, 0)


def _seller_block(pdf: _BrandedPDF, seller: SellerProfile, width: float = 100) -> None:
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(*_NAVY)
    pdf.cell(0, 5, "BILLED BY", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 9)
    if not seller.is_configured:
        pdf.set_font("helvetica", "B", 9)
        pdf.multi_cell(
            width,
            5,
            "[SELLER NOT CONFIGURED - set BILLING_SELLER_*]",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        return
    for line in seller.address_lines:
        pdf.multi_cell(width, 5, line, new_x="LMARGIN", new_y="NEXT")
    if seller.gstin:
        pdf.multi_cell(width, 5, f"GSTIN: {seller.gstin}", new_x="LMARGIN", new_y="NEXT")


def _meta_box(
    pdf: _BrandedPDF, rows: list[tuple[str, str]], x: float, width: float, y: float
) -> float:
    pdf.set_xy(x, y)
    pdf.set_fill_color(*_LIGHT_GREY)
    pdf.set_font("helvetica", "", 9)
    for label, value in rows:
        pdf.set_x(x)
        pdf.set_font("helvetica", "B", 9)
        pdf.cell(28, 6, label, new_x="END", fill=True)
        pdf.set_font("helvetica", "", 9)
        pdf.cell(width - 28, 6, value, new_x="LMARGIN", new_y="NEXT", fill=True)
    return pdf.get_y() + 2


def _status_stamp(pdf: _BrandedPDF, status: str, overdue_days: int | None) -> None:
    normalized = (status or "").upper()
    if normalized == "PAID":
        color, text = _GREEN, "PAID"
    elif normalized == "VOID":
        color, text = _RED, "VOID"
    elif overdue_days:
        color, text = _RED, f"OVERDUE BY {overdue_days} DAYS"
    elif normalized in {"ISSUED", "DRAFT"}:
        color, text = _AMBER, normalized or "DUE"
    else:
        color, text = _AMBER, normalized or "DUE"
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(*color)
    pdf.cell(0, 7, text, align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def _overdue_days(invoice: Invoice) -> int | None:
    if (invoice.status or "").lower() in {"paid", "void"} or invoice.due_at is None:
        return None
    days = (datetime.now(tz=UTC) - invoice.due_at).days
    return days if days > 0 else None


def _line_item_table(pdf: _BrandedPDF, invoice: Invoice) -> None:
    items = list(invoice.line_items or [])
    pdf.set_font("helvetica", "B", 10)
    pdf.set_text_color(*_NAVY)
    pdf.cell(0, 7, "Line Items", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    headings = ("S.No", "Particulars (HSN/SAC)", "Qty", "Rate", "Taxable", "Tax", "Amount")
    widths = (12, 70, 12, 23, 23, 20, 26)
    rows: list[tuple[str, ...]] = []
    for index, item in enumerate(items, start=1):
        particulars = item.plan_name
        if item.description:
            particulars += f"\n{item.description}"
        particulars += f"\nSAC: {item.sac_code}"
        rows.append(
            (
                str(index),
                particulars,
                str(item.quantity),
                _money(item.unit_price),
                _money(item.amount),
                _money(item.tax_amount),
                _money(Decimal(str(item.amount)) + Decimal(str(item.tax_amount))),
            )
        )
    if not rows:
        rows.append(("", "No line items recorded", "", "", "", "", ""))
    with pdf.table(
        col_widths=widths,
        line_height=6,
        first_row_as_headings=True,
        text_align=("CENTER", "LEFT", "CENTER", "RIGHT", "RIGHT", "RIGHT", "RIGHT"),
        width=186,
    ) as table:
        heading = table.row()
        for text in headings:
            heading.cell(text)
        for row_data in rows:
            row = table.row()
            for text in row_data:
                row.cell(text)


def _totals_block(pdf: _BrandedPDF, invoice: Invoice) -> None:
    pdf.ln(3)
    left = 110
    pdf.set_font("helvetica", "", 10)
    totals: list[tuple[str, str]] = [
        ("Subtotal", _money(invoice.subtotal)),
        ("CGST", _money(invoice.cgst_amount)),
        ("SGST", _money(invoice.sgst_amount)),
        ("IGST", _money(invoice.igst_amount)),
        (f"Tax total ({Decimal(str(invoice.tax_rate)):.0%})", _money(invoice.tax_amount)),
    ]
    for label, value in totals:
        pdf.set_x(left)
        pdf.cell(45, 6, label, new_x="END")
        pdf.cell(0, 6, value, align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(left)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(*_NAVY)
    pdf.cell(45, 8, "Total", new_x="END")
    pdf.cell(
        0,
        8,
        f"{_money(invoice.total)} {invoice.currency}",
        align="R",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    pdf.set_font("helvetica", "B", 9)
    pdf.multi_cell(
        0,
        5,
        f"Amount in words: {amount_in_words(invoice.total, invoice.currency)}",
        new_x="LMARGIN",
        new_y="NEXT",
    )


def _terms_block(pdf: _BrandedPDF) -> None:
    pdf.ln(2)
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(*_NAVY)
    pdf.cell(0, 5, "Terms & Notes", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 8)
    for term in _TERMS:
        pdf.multi_cell(0, 4, term, new_x="LMARGIN", new_y="NEXT")


def _buyer_block(pdf: _BrandedPDF, invoice: Invoice, buyer_label: str = "BILLED TO") -> None:
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(*_NAVY)
    pdf.cell(0, 5, buyer_label, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 9)
    pdf.cell(0, 5, f"Customer ID: {invoice.user_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0, 5, f"Buyer GSTIN: {invoice.buyer_gstin or 'Unregistered'}", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(0, 5, f"Place of supply: {invoice.place_of_supply}", new_x="LMARGIN", new_y="NEXT")


def render_invoice_pdf(invoice: Invoice, seller: SellerProfile | None = None) -> bytes:
    """Render a branded GST tax invoice. Seller falls back to placeholders."""
    profile = seller or SellerProfile()
    pdf = _new_doc()
    _header_band(pdf, "TAX INVOICE", profile)
    top = pdf.get_y()
    _seller_block(pdf, profile)
    seller_end = pdf.get_y()
    meta_end = _meta_box(
        pdf,
        [
            ("Number", invoice.invoice_number),
            ("Issued", _date(invoice.issued_at)),
            ("Due", _date(invoice.due_at)),
            ("SAC", invoice.sac_code),
            ("Currency", invoice.currency),
        ],
        x=118,
        width=80,
        y=top,
    )
    pdf.set_y(max(seller_end, meta_end))
    _status_stamp(pdf, invoice.status, _overdue_days(invoice))
    pdf.ln(1)
    _buyer_block(pdf, invoice)
    pdf.ln(2)
    _line_item_table(pdf, invoice)
    _totals_block(pdf, invoice)
    _terms_block(pdf)
    return bytes(pdf.output())


def _kv(pdf: _BrandedPDF, label: str, value: str) -> None:
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(52, 7, label, new_x="END")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")


def render_receipt_pdf(receipt: PaymentReceipt, seller: SellerProfile | None = None) -> bytes:
    """Render a branded payment receipt with amount in words and period."""
    profile = seller or SellerProfile()
    pdf = _new_doc()
    _header_band(pdf, "PAYMENT RECEIPT", profile)
    top = pdf.get_y()
    _seller_block(pdf, profile)
    seller_end = pdf.get_y()
    meta_end = _meta_box(
        pdf,
        [
            ("Number", receipt.receipt_number),
            ("Date", _date(receipt.receipt_date)),
            ("Method", receipt.payment_method or "—"),
            ("Currency", receipt.currency),
        ],
        x=118,
        width=80,
        y=top,
    )
    pdf.set_y(max(seller_end, meta_end))
    pdf.ln(1)
    _kv(pdf, "Received from", receipt.user_id)
    _kv(pdf, "Plan", receipt.plan_name or "—")
    if receipt.billing_period_start and receipt.billing_period_end:
        _kv(
            pdf,
            "Billing period",
            f"{_date(receipt.billing_period_start)} to {_date(receipt.billing_period_end)}",
        )
    _kv(pdf, "Razorpay Payment ID", receipt.razorpay_payment_id)
    _kv(pdf, "Subscription", str(receipt.subscription_id))
    pdf.ln(3)
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(*_NAVY)
    pdf.cell(
        0,
        9,
        f"Amount received: {_money(receipt.amount)} {receipt.currency}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "B", 9)
    pdf.multi_cell(
        0,
        5,
        f"Amount in words: {amount_in_words(receipt.amount, receipt.currency)}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    _terms_block(pdf)
    return bytes(pdf.output())
