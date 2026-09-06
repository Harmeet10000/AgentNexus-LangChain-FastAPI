"""Minimal PDF rendering for invoices and receipts (fpdf2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fpdf import FPDF

if TYPE_CHECKING:
    from .model import Invoice
    from .receipt import PaymentReceipt


def _header(pdf: FPDF, title: str, number: str) -> None:
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Number: {number}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)


def _kv(pdf: FPDF, label: str, value: str) -> None:
    pdf.set_font("helvetica", "", 10)
    pdf.cell(60, 6, label, new_x="END")
    pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")


def _line_items(pdf: FPDF, invoice: Invoice) -> None:
    items = list(invoice.line_items or [])
    if not items:
        return
    pdf.ln(4)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, "Line Items", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 9)
    for item in items:
        pdf.cell(0, 5, item.plan_name, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 8)
        pdf.cell(
            0,
            5,
            (
                f"  {item.description or ''}  qty {item.quantity}  "
                f"@ {item.unit_price:.2f}  tax {item.tax_amount:.2f}  "
                f"total {item.amount:.2f}"
            ),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("helvetica", "", 9)


def render_invoice_pdf(invoice: Invoice) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    _header(pdf, "TAX INVOICE", invoice.invoice_number)
    _kv(pdf, "Seller GSTIN", invoice.seller_gstin)
    _kv(pdf, "Buyer GSTIN", invoice.buyer_gstin or "—")
    _kv(pdf, "Place of supply", invoice.place_of_supply)
    _kv(pdf, "SAC", invoice.sac_code)
    _kv(pdf, "Issued", invoice.issued_at.isoformat() if invoice.issued_at else "—")
    _kv(pdf, "Due", invoice.due_at.isoformat() if invoice.due_at else "—")
    _line_items(pdf, invoice)
    pdf.ln(4)

    pdf.set_font("helvetica", "B", 10)
    pdf.cell(
        0, 7, f"Subtotal: {invoice.subtotal:.2f} {invoice.currency}", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(
        0,
        7,
        f"Tax ({invoice.tax_rate:.2%}): {invoice.tax_amount:.2f}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        7,
        f"CGST: {invoice.cgst_amount:.2f}   SGST: {invoice.sgst_amount:.2f}   IGST: {invoice.igst_amount:.2f}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, f"Total: {invoice.total:.2f} {invoice.currency}", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def render_receipt_pdf(receipt: PaymentReceipt) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    _header(pdf, "PAYMENT RECEIPT", receipt.receipt_number)
    _kv(pdf, "Razorpay Payment ID", receipt.razorpay_payment_id)
    _kv(pdf, "Date", receipt.receipt_date.isoformat())
    _kv(pdf, "Method", receipt.payment_method or "—")
    _kv(pdf, "Plan", receipt.plan_name or "—")
    pdf.ln(4)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(
        0,
        8,
        f"Amount received: {receipt.amount:.2f} {receipt.currency}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    return bytes(pdf.output())
