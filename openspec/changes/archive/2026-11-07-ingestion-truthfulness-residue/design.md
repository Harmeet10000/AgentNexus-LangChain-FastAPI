> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

`PLAN-bulletproof-ingestion.md` (retired 2026-10-31, archived at `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md`) closed with Phases A–D delivered; Phase E (quarantine ledger + state key-set seatbelt) and two Phase-C test gaps were recorded as residue. Verified on disk (2026-10-31):

- `src/app/features/documents/ingestion/legacy_service.py:58-67` builds `initial_state` with `doc_id`, `user_id`, `thread_id`, `raw_bytes`, `filename`, `source`, `document_type`, `jurisdiction`. `DocumentIngestionState` (`ingestion/workflow.py:49-67`, `TypedDict(total=False)`) declares `document_id`, `user_id`, `tenant_id`, `filename`, `content_type`, `object_uri`, `document_version`, `attempt`, `task_id`, `status`, `chunk_count`, `verified_chunk_count`, `document_kind`, `error_code`, `error_message`, `error_retryable`. Only `user_id` and `filename` overlap; the other six keys are silently dropped by LangGraph's channel mapping.
- The legacy path is reached through `documents/api/http/legacy_ingest_router.py` (~173-183), which resolves `app.state.ingestion_graph` and wraps it in `IngestionService`.
- Orphans with zero production importers: `src/app/extraction/docling/ingest_v2.py` (tests only), `src/app/extraction/langextract/langextract_batch_processor.py`, `src/app/extraction/langextract/docling_preprocessor.py`. The `features/ingestion` package was folded into `documents/api/http/` + `documents/ingestion/` by commit `25ba6f5` (2026-10-30) — the shim survived as the legacy compatibility surface.
- The golden corpus records multi-contract boundaries in `evals/corpus/README.md` §4: SRC-04 (5 boundaries), SRC-14 (2, incl. an embedded template), SYN-02 (2 contracts in one file); ambiguity is explicitly flagged (`ambiguous_boundaries`) rather than fabricated.

## Goals / Non-Goals

**Goals:**

- The ledger exists before anything else is deleted, ever — retroactive rows first.
- The silent-drop class is dead on the last live edge, and the seatbelt generalises to every compiled graph.
- The two Phase-C test gaps close without touching the pipeline implementation the harden change owns.

**Non-Goals:**

- No deletion of any quarantined surface; no decision on the port-or-discard criteria.
- No boundary-detection implementation, no `contract_index` carrying into Graphiti/evidence (harden change task 4.1).
- No re-ingestion preconditions, no retrieval measurement work (their own changes).
- No change to the live (non-legacy) upload path.

## Decisions

### 1. The ledger is a document capability, not a wiki page

`docs/quarantine-ledger.md` is the artifact; the `quarantine-ledger` spec makes its existence, row shape, and the deletion rule enforceable requirements rather than good intentions. The plan's exact column set (Surface / Path / Kept because / Port-or-discard criterion / Revisit trigger) is adopted verbatim — it was the audited design. Alternative rejected: tracking quarantined surfaces in issue tickets only — not greppable from the tree, not reviewable alongside a deletion diff.

### 2. Map the legacy keys explicitly; do not fail the legacy path today

Six of the eight legacy keys are droppable garbage, but the endpoint is quarantined and deprecation-shimmed. The explicit mapping routes what the declared channels can carry (`doc_id` → `document_id`, `raw_bytes` → the ingest function via object-store reference per `document-ingestion-pipeline`'s fetch-by-reference rule, `source`/`document_type` → `document_kind`-adjacent declared channels where they exist, `thread_id`/`jurisdiction` → invocation config) and fails loudly — typed error naming every key it could not place — for anything unmappable. The boundary test pins whichever choice lands. Alternative rejected: deleting the legacy endpoint — that is the ledger's port-or-discard decision, not this change's.

### 3. The seatbelt is a test contract, not a LangGraph guard

A key-set assertion per compiled graph (`set(initial_state) <= set(State.__annotations__)`), written against the graph's actual input channel, enumerated in one parametrised suite so a new compiled graph without a row is visible. Runtime guards inside LangGraph were rejected: the wrapper would itself be a second mapping layer to maintain, and the failure mode this kills is a *re-wiring* mistake, which a test at the boundary catches exactly where it happens.

### 4. The classification tests pin the contract the harden change will implement

The leakage assertion and the boundary suite run on the current `classify_document` path and record what it does today; when harden task 4.1 lands boundary-aware extraction, the suite is the regression gate it must satisfy. The corpus's recorded boundary tables (`evals/corpus/README.md` §4, `acquisition/SRC-04.json`) are the normative fixture — no new boundary labels are invented here.

## Risks / Trade-offs

- [Explicit legacy mapping changes a (degraded) success into a possible typed failure] → Acceptable and intentional: the endpoint is quarantined; a truthful typed failure outranks a silently empty ingestion. The deprecation shim's Sunset headers are untouched.
- [Key-set tests ossify a state schema that legitimately grows] → The test asserts the *input channel against the declaration*, not a frozen key list; adding a declared channel updates the declaration and the test follows mechanically.
- [Classification suite may fail today's regex path on SRC-04] → That is the point: the suite records the verification outcome the plan's Phase-C discipline demands; a red outcome is a recorded finding feeding harden task 4.1's V8 gate, not a blocker to landing the suite behind the project's expected-failure conventions.

## Migration Plan

1. Write `docs/quarantine-ledger.md` with the retroactive rows; record the standing rule.
2. Land the parametrised key-set boundary suite over the compiled graphs; make it green.
3. Map the legacy path's keys explicitly (or fail loudly per key); boundary test proves it.
4. Add the cross-contract leakage assertion and the multi-contract classification suite; record outcomes.

## Open Questions

None blocking. The ledger's port-or-discard verdicts are deliberately left to their named revisit triggers.
