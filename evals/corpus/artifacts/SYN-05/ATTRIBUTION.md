# ATTRIBUTION — SYN-05

Source: Corpus agent (Wave 2A) — synthetic fixture authored 2026-10-28; no external source.

Status: SYNTHETIC — SYNTHETIC — authored by the corpus agent for evaluation fixture use only; no third-party rights; India Code/Gazette/CPWD redistribution is prohibited (independently verified 2026-10-28) so statutory text is reproduced from model knowledge with drift-risk labeling.

File: original.pdf (sha256 `38e831feea760d40b122d34710e96872bfddf107b90989148e74c51531c3812d`, 294364 bytes)

Title: SYN-05 Synthetic scanned-style purchase order with operative term in a footnote (PO-2026-04517)

Fixture role: footnote-operative fixture: image-only PDF (no text layer); operative payment term (45 days) and LD term (0.75%/week, 10% cap) appear ONLY in footnote 2 at page bottom; created because SRC-05/06 have no usable text layer and tesseract was unavailable

Permission status: synthetic_owned — redistribution not applicable (synthetic). Document text is untrusted data; the adversarial payload in SYN-04 must never be obeyed.

## Re-render note (Wave 2B-2, case authoring, 2026-10-28)

The Wave 2A render was image-only (zero text layer), so gold-set quotes could not be
verified against extracted text. The same authored PO (PO-2026-04517, same operative
footnote-2 terms: payment 45 days, LD 0.75%/week capped at 10%) was re-rendered as a
single-page text-layer PDF via LibreOffice Writer (txt→pdf; producer LibreOffice 26.2.5.2)
so quotes are mechanically extractable with `pdftotext`.

- New file: original.pdf (sha256 `7e336666e8024b270289dcfec197efbd3ebc87b96ee0774b192dee642da7a0b2`, 27056 bytes)
- `raw_ocr.txt`, `normalized.txt` (rules R1–R7, no repair), and `pages.json` regenerated from the new PDF.
- The acquisition record `evals/corpus/acquisition/SYN-05.json` still records the Wave 2A
  sha256 (`38e831feea76…`) and byte size; the manifest reconciliation step must use the
  new sha256 above for SYN-05 (the acquisition record is owned by Wave 2A and was not
  modified). Visual scan-style effects (JPEG noise, skew) are gone — the OCR-noise axis
  of the corpus is carried by SRC-05/06/07 and SYN-05's role is now purely
  footnote-placement.
