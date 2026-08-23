from decimal import Decimal

from app.features.invoices.tax import (
    GST_STATE_CODES,
    paisa_to_rupees,
    rupees_to_paisa,
    split_cgst_sgst,
    split_tax_for_gst,
    split_tax_inclusive,
    state_code_from_gstin,
    validate_gstin,
)


class TestTaxSplits:
    def test_split_tax_inclusive_sums_exactly(self) -> None:
        subtotal, tax = split_tax_inclusive(11800, Decimal("0.18"))
        assert subtotal + tax == 11800
        assert tax == 1800

    def test_split_cgst_sgst_sums_to_tax(self) -> None:
        cgst, sgst = split_cgst_sgst(1801)
        assert cgst + sgst == 1801

    def test_intra_state_uses_cgst_sgst(self) -> None:
        cgst, sgst, igst = split_tax_for_gst(1800, seller_state_code="27", buyer_state_code="27")
        assert cgst + sgst == 1800
        assert igst == 0

    def test_inter_state_uses_igst(self) -> None:
        cgst, sgst, igst = split_tax_for_gst(1800, seller_state_code="27", buyer_state_code="08")
        assert cgst == 0
        assert sgst == 0
        assert igst == 1800

    def test_unknown_buyer_state_treated_as_intra(self) -> None:
        cgst, sgst, igst = split_tax_for_gst(1800, seller_state_code="27", buyer_state_code=None)
        assert cgst + sgst == 1800
        assert igst == 0

    def test_round_trip_rupees_paisa(self) -> None:
        assert rupees_to_paisa(paisa_to_rupees(12345)) == 12345


class TestGstinValidation:
    def test_valid_gstin(self) -> None:
        assert validate_gstin("27AAPFU0939F1ZV")

    def test_invalid_state_code(self) -> None:
        assert not validate_gstin("99AAPFU0939F1ZV")

    def test_bad_format(self) -> None:
        assert not validate_gstin("27AAPFU0939F1Z")
        assert not validate_gstin("not-a-gstin")

    def test_state_code_extraction(self) -> None:
        assert state_code_from_gstin("27AAPFU0939F1ZV") == "27"
        assert state_code_from_gstin("bad") is None

    def test_all_state_codes_present(self) -> None:
        assert len(GST_STATE_CODES) == 39
