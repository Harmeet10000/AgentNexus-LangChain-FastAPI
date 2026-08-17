"""GST-inclusive tax helpers computed in integer paisa.

GST-inclusive pricing (Requirements 12, 38):
``subtotal = amount / 1.18``, ``tax = subtotal * 0.18``, computed in integer
paisa so ``subtotal + tax == amount`` holds exactly (Property 1) and
``cgst + sgst == tax`` holds exactly with ROUND_HALF_EVEN (Property 5).
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, Decimal

GST_RATE = Decimal("0.18")
_PAISA = Decimal(1)
_RUPEES = Decimal("0.01")

# Indian state/UT GST codes (Requirement 37). 97 covers "Other Territory".
GST_STATE_CODES = frozenset(f"{code:02d}" for code in range(1, 39)) | {"97"}

_GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]Z[0-9A-Z]$")


def state_code_from_gstin(gstin: str) -> str | None:
    """Return the 2-digit state code from a GSTIN, or None if malformed."""
    if len(gstin) < 2 or not gstin[:2].isdigit():
        return None
    return gstin[:2]


def validate_gstin(gstin: str) -> bool:
    """Validate a 15-char GSTIN and its state code (Requirements 12.6/37)."""
    if not _GSTIN_RE.fullmatch(gstin.upper()):
        return False
    return state_code_from_gstin(gstin) in GST_STATE_CODES


def split_tax_for_gst(
    tax_paisa: int, *, seller_state_code: str, buyer_state_code: str | None
) -> tuple[int, int, int]:
    """Split tax into (cgst, sgst, igst).

    Intra-state supply (seller and buyer in the same state): CGST + SGST.
    Inter-state supply: single IGST (Requirements 12.4/12.5).
    """
    if buyer_state_code is None or buyer_state_code == seller_state_code:
        cgst, sgst = split_cgst_sgst(tax_paisa)
        return cgst, sgst, 0
    return 0, 0, tax_paisa


def split_tax_inclusive(amount_paisa: int, tax_rate: Decimal) -> tuple[int, int]:
    """Return (subtotal_paisa, tax_paisa) for a GST-inclusive amount.

    ``subtotal = amount / (1 + rate)`` rounded half-even; ``tax`` is the
    exact remainder so the two always sum to the original amount.
    """
    rate = tax_rate or GST_RATE
    subtotal = (Decimal(amount_paisa) / (Decimal(1) + rate)).quantize(
        _PAISA, rounding=ROUND_HALF_EVEN
    )
    return int(subtotal), amount_paisa - int(subtotal)


def split_cgst_sgst(tax_paisa: int) -> tuple[int, int]:
    """Split tax into equal CGST/SGST with banker's rounding.

    When the tax amount is odd, CGST gets the half-even rounded share and
    SGST absorbs the remainder so the pair always sums exactly to tax.
    """
    cgst = int((Decimal(tax_paisa) / Decimal(2)).quantize(_PAISA, rounding=ROUND_HALF_EVEN))
    return cgst, tax_paisa - cgst


def paisa_to_rupees(amount_paisa: int) -> Decimal:
    return (Decimal(amount_paisa) / Decimal(100)).quantize(_RUPEES, rounding=ROUND_HALF_EVEN)


def rupees_to_paisa(amount: Decimal) -> int:
    return int((amount * Decimal(100)).quantize(_PAISA, rounding=ROUND_HALF_EVEN))
