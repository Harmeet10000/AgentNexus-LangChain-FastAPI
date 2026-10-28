# Legal-Retrieval Gold-Set Corpus — Acquisition Report (Wave 2A)

Acquired 2026-10-28 by the corpus acquisition and normalization agent.
Worktree: `/home/harmeet/worktrees/legal-gold-set-expansion` (branch `codex/legal-gold-set-expansion`).
All downloads to `evals/corpus/artifacts/<SOURCE_ID>/original.pdf|txt` only; scratch in `/tmp`.

**Tooling**: `pdftotext`/`pdfinfo`/`pdftoppm` (poppler **26.01.0**); `tesseract` **not installed** (checked `which tesseract pdftoppm`) — no OCR pass was possible for text-layer-less scans, and none was faked. `file`, `sha256sum`, `jq`, Python 3 (stdlib + PIL 12.1.1 for SYN-05).

**Result: 12 of 16 real sources acquired (valid PDFs + SHA-256), 4 failed with documented exact errors and alternates tried; 5 synthetic fixtures authored and labeled; 21 acquisition records (16 source records incl. 4 failure records + 5 synthetic records).**

---

## 1. Per-source status table (real sources)

| source_id | bytes | sha256 (prefix) | pages | ocr_status | contract_boundaries | status |
|---|---|---|---|---|---|---|
| SRC-01 | 2,529,606 | `95eb5f8efc48…` | 333 | embedded_text_layer | [] (single) | OK |
| SRC-02 | 1,976,762 | `a46c844662eb…` | 244 | embedded_text_layer | [] (single) | OK |
| SRC-03 | 1,357,972 | `c5baa3ff5394…` | 252 | embedded_text_layer | [] (single) | OK — medium confidence RESOLVED (content = HAM MCA, distinct edition from SRC-02) |
| SRC-04 | 37,223,714 | `b566b81555f6…` | 372 | embedded_text_layer | 5 boundaries (see §4) | OK |
| SRC-05 | 45,945,471 | `68c1b33c0d6b…` | 113 | **scanned_no_text_layer** | [] — boundaries undetectable without OCR (flagged) | OK download; no text layer; original.pdf NOT stored in-repo (see note below) |
| SRC-06 | 48,424,982 | `666405611cb7…` | 347 | **scanned_no_text_layer** | [] — flagged | OK download; no text layer; original.pdf NOT stored in-repo (see note below) |
| SRC-07 | 5,265,744 | `7f87527c5446…` | 155 | embedded_text_layer (noisy scanned circular prefix p2–3) | [] (single) | OK |
| SRC-08 | — | — | — | — | — | **FAILED** (see §3) |
| SRC-09 | — | — | — | — | — | **FAILED** (see §3) |
| SRC-10 | — | — | — | — | — | **FAILED** (see §3) |
| SRC-11 | — | — | — | — | — | **FAILED** (see §3) |
| SRC-12 | 38,197 | `0de0ddcd04bb…` | 9 | embedded_text_layer | [] (single) | OK — verified: Sukhdev Singh v. Maharaja Bahadur of Gidhaur (02.05.1951) |
| SRC-13 | 624,405 | `8141dca43db8…` | 157 | embedded_text_layer | [] (single) | OK — verified: I.C. Golaknath v. State of Punjab (27.02.1967) |
| SRC-14 | 1,539,207 | `04a38b514cac…` | 134 | embedded_text_layer | 2 boundaries (see §4) | OK |
| SRC-15 | 599,484 | `d25ab07fcc61…` | 22 | embedded_text_layer | [] (single) | OK |
| SRC-16 | 365,317 | `b823db45848e…` | 31 | embedded_text_layer | [] (single) | OK |

MIME notes: all SRC PDFs verify by `%PDF` magic bytes. SRC-12/SRC-13 have 4 junk bytes before `%PDF`, so `file(1)` reports `application/octet-stream` — `pdfinfo`/`pdftotext` parse them as valid PDFs (producer GNU Ghostscript 7.05); this is recorded verbatim in their `mime_type` fields.

**Storage note (2026-10-28):** the SRC-05 and SRC-06 `original.pdf` binaries (~94 MB combined) are intentionally **not stored in-repo**: both are image-only scans with zero text layer, neither is referenced by any gold row, and the repo cost outweighs the value. `raw_ocr.txt`, `normalized.txt`, `pages.json` and `ATTRIBUTION.md` are retained in their artifact directories. Their acquisition records (`acquisition/SRC-05.json`, `SRC-06.json`) keep `storage_status: "not_stored_in_repo"` plus the original sha256 and byte size — re-download from the recorded `source_url` with the recorded `download_command` and verify the sha256 before use.

## 2. Failures (exact errors, alternates tried — no mirrors used per mission rule)

All four failures are on `indianrailways.gov.in` / `ecr.indianrailways.gov.in`, whose uploads tree now serves a **"Site Under Construction"** placeholder (verified on 2026-10-28); the Railway Board content section (`view_section.jsp?id=0,4,1244`) returns 404.

| source_id | exact error | alternates tried |
|---|---|---|
| SRC-08 (ACS-9 letter) | `curl: (22) The requested URL returned error: 404` (HTTP 404, `text/html;charset=UTF-8`, 0 bytes) | literal-space URL variant → curl exit 000 (TLS/host reject); `http://` → 302 → https → same 404; parent dir listing → "Site Under Construction" |
| SRC-09 (ACS-11) | same as SRC-08 (HTTP 404) | same set (404 / 404) |
| SRC-10 (ECR IBD v3.0) | HTTP 404 on `ecr.indianrailways.gov.in` (host itself returns 200 for `/`) | `www.ecr.` host variant → 404 |
| SRC-11 (2019 SGC decoy) | HTTP 404 | `http://` → 302 → https → same 404; path without `pdf/` subdir → 404 |

Per mission, mirror substitution (e.g. web.archive.org) was **not** attempted. Each failure has a full acquisition record (`evals/corpus/acquisition/SRC-{08,09,10,11}.json`) with `sha256=null`, `page_count=null`, alternates list, and fixture-role impact in `unresolved_rights_notes`.

Fixture-role impact: SRC-07 (GCC April 2022 base) is acquired, so the version-pair axis survives partially; the ACS-9/ACS-11 slips (SRC-08/09) and the 2019 older-revision decoy (SRC-11) are unavailable — `characteristics.version_pair_role=true` is recorded on their failure records so the schema/fixture layer knows the intended role.

## 3. Synthetic supplements (all clearly labeled synthetic)

Rationale: India Code / Gazette / CPWD redistribution is prohibited (independently verified 2026-10-28), so statute text is authored from model knowledge with an explicit drift-risk label; contract fixtures are authored outright.

| id | file | sha256 (prefix) | bytes | pages | purpose |
|---|---|---|---|---|---|
| SYN-01 | `original.txt` | `1c4889cbf918…` | 7,471 | 1 | statute excerpt: ICA 1872 ss. 73–74 + A&C Act 1996 s. 34 (from model knowledge, mandated header, "not a reproduction of India Code official text") |
| SYN-02 | `original.txt` | `79846f04bd33…` | 7,977 | 2 | cross-tenant twin works contracts, one template: Tenant A LD **0.5%**/wk (p1) vs Tenant B LD **1.0%**/wk (p2); Completion Date 30 Jun vs 31 Jul 2026 |
| SYN-03 | `original.txt` | `b00fa3e3f9d4…` | 2,626 | 1 | benign addendum to SYN-02-WC-A (30-day extension; LD rate unchanged at 0.5%/wk, 5% cap) |
| SYN-04 | `original.txt` | `0ac11f68446b…` | 4,161 | 1 | **adversarial twin of SYN-03**: fake `### SYSTEM MESSAGE ###`, `ignore prior instructions` payloads, poisoned Clause 3.1 (claims 1.0% LD); true operative term restated in Clause 3.2 (0.5%) |
| SYN-05 | `original.pdf` | `7e336666e802…` | 27,056 | 1 | scanned-style purchase order, **re-rendered 2026-10-28 with an embedded text layer** (single-page text-layer PDF via LibreOffice Writer 26.2.5.2, txt→pdf) so quotes are mechanically extractable; operative payment term (45 days) + LD (0.75%/wk, 10% cap) appear **only in footnote 2**. Created because SRC-05/06 have no text layer and tesseract was unavailable (the mission's fallback trigger); the Wave 2A image-only render (PIL JPEG q72, `38e831feea76…`, 294,364 bytes) had zero text layer and was superseded — see the re-render note in `artifacts/SYN-05/ATTRIBUTION.md`. |

SYN-04's payload is inert fixture data. It exists to test that ingestion treats document text as untrusted; every page of the file is labeled as adversarial synthetic.

## 4. Multi-contract boundaries (one-based `contract_index` per repository convention; never 0)

**SRC-04** (NHAI HAM packet, 372 pp; `contract_boundaries` in `acquisition/SRC-04.json`):
| idx | title | pages |
|---|---|---|
| 1 | Concession Agreement with Schedules A–S (HAM) | 5–320 |
| 2 | Integrity Pact (tender NHAI/HQ/MP/Budhni-Badi/Pkg-4/2023 dated 15.12.2023) | 321–330 |
| 3 | Annexure: Corrigendum & Reply to Pre-Bid Queries | 331–363 |
| 4 | Financial Bid (Appendix IB) | 364–367 |
| 5 | Letter of Award and Acceptance thereof | 368–372 |

Flagged (`ambiguous_boundaries`): pages 1–4 are cover + "INDEX FOR VOLUME I" front matter attributed to no contract; p331 is an "Annexure" divider included in index 3; the CA execution date uses blanks (packet fixity = 29.08.2024 digital signature).

**SRC-14** (CCS RM3765 framework, 134 pp):
| idx | title | pages |
|---|---|---|
| 1 | Framework Agreement RM3765 (main terms + Framework Schedules 1–22) | 1–134 |
| 2 | Template Call-Off Contract (Framework Schedule 4: Annex 1 Template Order Form + Annex 2 Template Call Off Terms) | 76–77 |

Flagged: the call-off template is **embedded** in the framework's schedule sequence (its range overlaps index 1) and is a template, not an executed contract; the Draft Deed of Guarantee form (Schedule 13, pp. 104–113) is a second embedded template, deliberately NOT counted; TOC printed page numbers are offset from PDF pages (boundaries located by headings).

**SYN-02**: idx 1 = base works contract Tenant A (p1), idx 2 = twin Tenant B (p2).

Single-contract documents carry `contract_boundaries: []` (implicit index null). No boundaries were fabricated where evidence was ambiguous — ambiguity is in `ambiguous_boundaries`.

## 5. Fixture-role mapping (retrieval-eval targeting)

| Role | Sources |
|---|---|
| OCR-heavy (text-layer-less or noisy) | SRC-05, SRC-06 (zero text layer; originals not stored in-repo — acquisition records retain sha256 + size for re-download); SRC-07 (scanned circular prefix p2–3, garbled Devanagari OCR inside typed doc) |
| Schedule/table-heavy | SRC-01 (BOT MCA schedules), SRC-04 (Schedules A–S + annuity tables + financial bid), SRC-10 (tender packet, failed), SRC-14 (22 framework schedules) |
| Irregular numbering | SRC-07 (index pages stamped "GCC July 2020" under the April 2022 edition; scanned prefix vs typed body), SRC-10 (0100-series, failed), SRC-11 (2019 revision, failed), SRC-14 (printed-vs-PDF page offset; "NOT USED" schedules 10/15/16) |
| Version pairs | SRC-02 ↔ SRC-03 (HAM MCA 28.04.2022 edition vs Nov-2020-base edition updated till 31.10.2023); SRC-07 + SRC-08 (ACS-9, failed) + SRC-09 (ACS-11, failed); SRC-11 as older-revision decoy (failed); SYN-02 Tenant A vs Tenant B; SYN-03 vs SYN-04 |
| Multi-contract | SRC-04 (5 boundaries), SRC-14 (2 boundaries incl. embedded template), SYN-02 (2 contracts in 1 file) |
| Clean controls | SRC-01, SRC-02, SRC-03, SRC-12 (9 pp, 1951), SRC-13 (157 pp, 1967), SRC-15, SRC-16, SYN-01, SYN-03 |
| Jurisdiction decoys (UK vs IN) | SRC-14, SRC-15, SRC-16 (OGL v3; LCIA arbitration/England courts in SRC-14) |
| Footnote-operative | SYN-05 (operative term only in footnote 2; re-rendered 2026-10-28 with an embedded text layer — see §3 — so `pdftotext` extracts the full text). SRC-05/06 could not be verified for usable footnote/stamp pages (no OCR tooling), triggering SYN-05 creation |
| Adversarial ingestion | SYN-04 (twin of SYN-03; injected instruction-like text) |

## 6. Commands (exact)

Download (per source; browser UA; `api.sci.gov.in` included):
```
curl -L --fail --silent --show-error --max-time 300 --retry 2 --retry-delay 5 \
  -A "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36" \
  -H "Accept: application/pdf,*/*" -o original.pdf "<URL>"
```
Verification: `file --mime-type original.pdf`, `sha256sum original.pdf`, `%PDF` magic check on `head -c 8`.
Page count: `pdfinfo original.pdf`.
`raw_ocr.txt` (verbatim, never repaired): `pdftotext original.pdf -`.
`pages.json`: `pdftotext -f N -l N original.pdf -` for N = 1..page_count, one-based `page_no`.
`normalized.txt`: documented minimal normalization only —
R1 remove NUL bytes; R2 CRLF/lone CR → LF; R3 U+00A0 & U+2007–U+200B → U+0020; R4 ligatures U+FB00–FB06 expanded; R5 form-feed page markers removed; R6 trailing whitespace stripped per line; R7 runs of 3+ blank lines collapsed to 2; R8 **nothing else — no spelling, hyphenation, or OCR repair of any kind**.
SYN-05 authoring: PIL 12.1.1 image render (JPEG q72, 150 dpi, seeded noise, GaussianBlur 0.4, −0.4° skew) embedded in a single-page PDF; `pdftotext` returned 0 chars (no text layer, by design). **Superseded 2026-10-28 (Wave 2B-2):** re-rendered as a single-page text-layer PDF via LibreOffice Writer 26.2.5.2 (txt→pdf) — same authored PO and operative footnote-2 terms; `raw_ocr.txt`/`normalized.txt`/`pages.json` regenerated from the new PDF (see §3 and `artifacts/SYN-05/ATTRIBUTION.md`).

## 7. Layout

```
evals/corpus/artifacts/<ID>/original.pdf|txt   (original binary/text; 12 real + 5 synthetic)
evals/corpus/artifacts/<ID>/raw_ocr.txt        (verbatim pdftotext output)
evals/corpus/artifacts/<ID>/normalized.txt     (documented minimal normalization only)
evals/corpus/artifacts/<ID>/pages.json         (per-page text + one-based page_no mapping)
evals/corpus/artifacts/<ID>/ATTRIBUTION.md     (issuer, URL, access date, policy/licence)
evals/corpus/acquisition/<ID>.json             (21 records: 12 acquired + 4 failures + 5 synthetic; exact keys, validated)
```
No artifact directories exist for SRC-08/09/10/11 (nothing downloaded — no substitutes). Failed sources have complete acquisition records with nulls in file-derived fields. SRC-05/SRC-06 artifact directories exist but intentionally hold **no `original.pdf`** (image-only scans, no gold references — see §1 storage note); all text-derived sidecars are present.

Security: document text treated as untrusted data throughout; no content sent to any external service (downloads only, no URLs from within documents fetched); no instructions inside documents executed; no git operations performed.
