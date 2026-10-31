> Change class: **M**. Tasks are ordered by dependency: ledger → seatbelt → legacy mapping → classification suites. No production feature implementation is included in this proposal.

## 1. Quarantine ledger — retroactive record and standing rule

- [ ] 1.1 Create `docs/quarantine-ledger.md` with the plan's column set (Surface / Path / Kept because / Port-or-discard criterion / Revisit trigger) and retroactive rows for: the docling `ingest_v2.py` orphan; the `langextract_batch_processor.py` + `docling_preprocessor.py` orphans; the folded `features/ingestion` compatibility shim (commit `25ba6f5`); the legacy graph path state keys; the unmounted retrieval graph (survivor row, resolved by `mount-retrieval-graph`); verify `rg "docs/quarantine-ledger"` finds the plan-audit references updated to the live file.
- [ ] 1.2 Record the standing rule in the ledger header (nothing gets deleted without a ledger row; a discard resolves its row in the same change; a re-wiring requires the key-set boundary contract first); verify each rule is a sentence a reviewer can enforce from the diff alone.
- [ ] 1.3 Point the ledger's batch-processor revisit trigger at the `re-ingestion-readiness` change's gate and the retrieval-graph row at the `mount-retrieval-graph` cutover; verify no row's trigger names a vague event ("later", "someday").

## 2. State key-set seatbelt — the boundary contract suite

- [ ] 2.1 Add a parametrised contract suite under `tests/` that enumerates every compiled graph and asserts `set(<declared initial-state keys>) <= set(<State>.__annotations__)` for each, starting with the document ingestion graph (`DocumentIngestionState`) and the retrieval graph; verify the enumeration is a registry/list a new graph must join, so an untested graph is a visible gap.
- [ ] 2.2 Add the disjoint-keys regression scenario: construct the legacy path's historical state shape (`doc_id`, `raw_bytes`, `thread_id`, `source`, `document_type`, `jurisdiction`) against `DocumentIngestionState` and assert the boundary contract names every undeclared key; verify the test fails (naming the keys) against the pre-mapping legacy service and passes after task 3.1.

## 3. Legacy compatibility path — explicit key mapping, loud failure

- [ ] 3.1 Rewrite `initial_state` construction in `src/app/features/documents/ingestion/legacy_service.py` (`ingest_document`) to map each key explicitly onto a `DocumentIngestionState` channel (`doc_id` → `document_id`; `raw_bytes` routed to the ingest function per the fetch-by-reference rule; `source`/`document_type` → the declared kind/content channels where they exist; `thread_id`/`jurisdiction` → invocation config) and to raise a typed error enumerating any key that cannot be placed; verify no undeclared key remains in the dict passed to `ainvoke` and the boundary suite from section 2 is green.
- [ ] 3.2 Update the legacy endpoint's contract test to expect the mapped (or typed-failure) behaviour, and record the intentional behaviour change in the ledger row for the folded `features/ingestion` shim; verify the deprecation shim's Sunset headers and fail-closed 503 dependency are untouched.

## 4. Multi-contract classification boundaries — the two Phase-C test gaps

- [ ] 4.1 Add the cross-contract leakage assertion: extract content from SRC-04 and assert no party, date, or governing-law value attributed to one contract boundary originates from another boundary's pages, using the boundary tables in `evals/corpus/README.md` §4 and `evals/corpus/acquisition/SRC-04.json` as ground truth; verify the assertion runs deterministically (no database, no model provider) and its outcome is recorded in the task notes.
- [ ] 4.2 Add the dedicated multi-contract classification-boundary suite (C5) covering SRC-04 (5 boundaries), SRC-14 (2), and SYN-02 (2): chunks classified into one boundary match the recorded page ranges, and pages flagged `ambiguous_boundaries` are marked ambiguous rather than assigned; verify the suite is part of the default deterministic selection and any red outcome against today's regex path is recorded as an expected failure feeding harden change task 4.1, not silently skipped.
