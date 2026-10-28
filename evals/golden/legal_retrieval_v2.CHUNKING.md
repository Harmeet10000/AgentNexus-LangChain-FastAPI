# Chunk-ID Convention — legal_retrieval_v2 (Wave 2B-2)

Every `chunk_id` in `evals/golden/legal_retrieval_v2.jsonl` has the form:

```
<SOURCE_ID>:p<page_no>:c<NN>
```

- `<SOURCE_ID>` — the manifest `source_id` exactly (e.g. `SRC-04`, `SYN-02`). The valid
  universe is the 17 sources listed in `evals/corpus/README.md` §1/§3
  (`SRC-01..SRC-07`, `SRC-12..SRC-16`, `SYN-01..SYN-05`).
- `<page_no>` — one-based PDF page number of the source document, matching
  `evals/corpus/artifacts/<SOURCE_ID>/pages.json`.
- `<NN>` — two-digit, stable, **1-based ordinal of the chunk within that page** under the
  convention below.

## Chunk definition

A **chunk is a contiguous non-empty text block within one page**, derived as follows:

1. Take the page's extracted text exactly as recorded in
   `evals/corpus/artifacts/<SOURCE_ID>/pages.json` (`pages[*].text` for the given
   `page_no`).
2. Apply the corpus normalization rules R1–R7 (see `evals/corpus/README.md` §6: NUL
   removal, CRLF→LF, exotic-space→U+0020, ligature expansion, form-feed removal,
   trailing-whitespace strip, collapse of 3+ blank lines). Rule R8 (no repair) applies —
   spelling/OCR defects are preserved.
3. Split the page text on runs of one or more blank lines (`\n{2,}`).
4. Discard empty pieces. Each remaining piece (one or more consecutive non-empty lines)
   is one chunk.
5. Number the chunks `1, 2, 3, …` in document reading order within the page. `<NN>` is
   that ordinal, zero-padded to two digits.

Consequences of the convention (all deterministic and re-derivable from the artifacts):

- A chunk never crosses a page boundary, and never contains a blank line.
- Scanned/blank pages (e.g. SRC-05 before re-render) have zero chunks.
- Running headers, stamp lines (e.g. SRC-04's "Signature Not Verified" blocks), and
  single-line table fragments each form their own chunk when separated by blank lines.
- Two adjacent clauses that pdftotext emits as one paragraph (no blank line between
  them) live in the SAME chunk. This is why the near-miss rows on SRC-07 p107,
  SRC-15 p12 and SRC-16 p14 score their decoys at span level rather than by
  `forbidden_chunk_ids` (which must be disjoint from `expected_chunk_ids` per validator
  V9).

## Integrity sentinels

`handle_integrity` rows carry `forbidden_chunk_ids` entries such as
`FORGED:SYN-01:p1:c99` or `STALE:SYN-02:p1:c2`. These are deliberately malformed
sentinels that can never resolve under the convention above (either the ordinal exceeds
the block count of the page or the prefix is not a `SOURCE_ID`). They model forged or
stale evidence IDs in a search→read-by-ID chain and must fail closed at retrieval time.

## Citation behavior & answer eligibility mapping

- **Citation behavior** is exercised through two surfaces only: the
  `handle_integrity` `FORGED:`/`STALE:` sentinels (above) and the
  `expected_answer_must_include` / `expected_answer_must_not_include` lists.
  There is no separate "citation" field; graders judge answers by whether the
  must-include strings appear and the must-not-include strings (and any
  sentinel-resolving or forbidden-document citations) do not.
- **Answer eligibility is derivable, not declared**: a row expects an answer
  iff `expected_abstention == false` AND `expected_evidence_status == "ok"`
  (default). Every no-evidence family row sets one of them to the abstaining
  value, and the V-rules (abstention rows forbid expected claims and need a
  non-ok evidence status; no-evidence case types force `not_applicable`
  relevance/entailment) keep the two fields consistent — so no dedicated
  eligibility field exists or is needed.

## CASE TAXONOMY (23 case_type slugs, 105 rows)

- `direct_answer` (10) — single-document factual lookup with deterministic chunks, spans and claims.
- `multi_document` (6) — answer must synthesize evidence from two (or more) allowed documents.
- `multi_hop` (6) — answer requires chaining evidence across pages or documents via an intermediate fact.
- `unanswerable` (5) — corpus holds no usable evidence for the query; correct behavior is abstention.
- `statute_not_in_corpus` (4) — the statute the query invokes is absent from the corpus; abstain, do not answer from memory.
- `conflicting_clauses` (5) — two in-scope clauses conflict; the answer must identify the operative one.
- `near_miss_clause` (6) — decoy clause is textually adjacent but legally different (rate vs cap, base vs amended).
- `definition_exception_trap` (5) — the operative definition carries a carve-out the answer must honor.
- `injection_content` (5) — adversarial instruction-like text inside documents; must be treated as untrusted content.
- `cross_tenant_isolation` (4) — per-tenant twin contracts; the answer must not leak the other tenant's terms.
- `stale_version` (4) — an older edition of the governing document exists; the answer must cite the current edition.
- `amendment_overrides` (4) — a corrigendum/addendum overrides the base clause; the amended term is the expectation.
- `no_evidence_vs_ingestion_failure` (5) — distinguishes "no evidence exists" abstention from an ingestion failure via declared coverage.
- `index_not_ready` (4) — the document exists but is not indexed; abstain with the index-not-ready evidence status.
- `multi_contract_boundary` (5) — the answer must come from the correct contract inside a multi-contract packet.
- `ocr_semantic_flip` (4) — OCR noise could flip a digit or term; the expected label is the true reading of the source.
- `table_misalignment` (4) — the answer sits in a table cell prone to row/column misalignment.
- `footnote_as_operative` (3) — the operative term appears only in a footnote, not the body.
- `annexure_as_main` (4) — an annexure/schedule qualifies or supersedes the main-body text the query points at.
- `wrong_jurisdiction` (3) — an Indian query surrounded by UK decoy documents that are forbidden citations.
- `wrong_date_version_authority` (3) — a decoy edition, date, or authority must be rejected in favor of the governing one.
- `partial_coverage_answer` (3) — only part of the requested packet is readable/ready; an explicit coverage expectation is declared.
- `handle_integrity` (3) — forged/stale evidence handles (`FORGED:`/`STALE:` sentinels) must fail closed at retrieval time.

## Verification (authoring-time, mechanical)

Every row was generated and verified by script (Wave 2B-2 authoring run):

- each `expected_span.quote` is a verbatim substring of the referenced document's
  `normalized.txt`;
- each quote occurs on the declared page and inside exactly one chunk under the
  convention (so `expected_chunk_ids` are consistent with the spans they anchor);
- every `expected_chunk_id`/`forbidden_chunk_id` (other than the intentional
  `FORGED:`/`STALE:` sentinels) resolves to a real chunk of the declared page;
- `contract_index` values on spans are one-based and consistent with the boundaries in
  `evals/corpus/acquisition/SRC-04.json`, `SRC-14.json` and `SYN-02.json` (pp1–4 of
  SRC-04 are unattributed front matter and carry `contract_index: null`).
