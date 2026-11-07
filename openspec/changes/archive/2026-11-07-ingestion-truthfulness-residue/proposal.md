> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — contract-level hardening confined to the documents/ingestion subsystem plus one governance document: a quarantine ledger, graph state key-set boundary contracts, an explicit key mapping on the legacy compatibility ingestion path, and two multi-contract classification test suites. No public API change, no migration, no new external dependency.

## Why

`PLAN-bulletproof-ingestion.md` (retired 2026-10-31, now at `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md`) closed with Phases A–D delivered, but its completion audit recorded residues that never landed: the quarantine ledger and state key-set seatbelt (Phase E) were never written, the legacy compatibility ingestion path still sends state keys that are disjoint from the graph's declared channels (the last live instance of the LangGraph silent-drop class), and two small Phase-C test gaps (cross-contract leakage, multi-contract classification boundaries) were left behind. This change is the audit's residue list, executed.

## What Changes

- **Quarantine ledger, created retroactively.** `docs/quarantine-ledger.md` records every known orphaned/quarantined surface with the columns the plan defined (surface, path, kept-because, port-or-discard criterion, revisit trigger): the docling `ingest_v2.py` orphan; the `langextract_batch_processor.py` + `docling_preprocessor.py` orphans; the folded `features/ingestion` deletion in `25ba6f5` (folded into `documents/api/http/` as the legacy compatibility shim); the legacy graph path keys; the unmounted retrieval graph (survivor per the `mount-retrieval-graph` change). A standing rule is recorded in the ledger: nothing gets deleted without a ledger row.
- **State key-set seatbelt (the contract test).** A boundary contract test asserts, for every compiled graph's input channel, `set(initial_state) <= set(<State>.__annotations__)` — because `TypedDict(total=False)` plus LangGraph's silent key-dropping make the type system load-bearing nowhere on that edge.
- **The one known ungated silent-drop edge is gated.** The legacy compatibility path (`legacy_service.py:58-67` building `initial_state` with `doc_id`/`raw_bytes`/`thread_id`/`source`/`document_type`/`jurisdiction`) sends keys disjoint from `DocumentIngestionState`'s declared channels (`ingestion/workflow.py:49-67` — only `user_id` and `filename` survive; the rest are silently dropped). The keys are mapped explicitly onto the declared channels, or the invocation fails loudly; silent dropping is no longer possible on this edge.
- **Cross-contract leakage assertion.** A test asserts that parties, dates, and governing-law extracted from a multi-contract document do not leak across contract boundaries — the corpus has SRC-04 with 5 recorded boundaries (`evals/corpus/README.md` §4) to verify against.
- **Multi-contract classification-boundary test suite (C5).** A dedicated suite pins the `classify_document` / legal-metadata path against multi-contract packets (SRC-04, SRC-14, SYN-02): content from one contract boundary must not be attributed to another.

## Scope / Non-Goals

In scope: the ledger document, the key-set boundary contract tests, the legacy-path key mapping (or loud failure), and the two classification test suites.

Out of scope (explicit): any deletion of a quarantined surface (the ledger's port-or-discard criteria decide later); the full contract-boundary *implementation* (boundary detection, per-contract extraction, `contract_index` carrying into Graphiti/evidence — owned by `harden-agent-saul-evidence-pipeline` task 4.1); re-ingestion preconditions 3–5 (the `re-ingestion-readiness` change); retrieval measurement (the `retrieval-measurement-gate` change); any change to the live upload pipeline's serving behaviour.

## Capabilities

### New Capabilities

- `quarantine-ledger`: the standing governance record for disconnected/orphaned surfaces — every kept surface has a ledger row naming a port-or-discard criterion and a revisit trigger, no surface is deleted without a ledger row, and re-wiring a quarantined surface requires the state key-set boundary contract first.
- `ingestion-state-boundary`: the key-set contract between every compiled graph's input channel and its declared state — an invocation whose initial state names a key the state does not declare is a loud failure or an explicit mapping, never a silent drop.

### Modified Capabilities

- `hierarchical-document-chunking`: classification and legal-metadata extraction respect multi-contract document boundaries — extracted content does not mix parties, dates, or governing law across contract boundaries, verified by the SRC-04 (5-boundary) fixtures.

## Impact

Affected code: `docs/quarantine-ledger.md` (new); `src/app/features/documents/ingestion/legacy_service.py` (explicit key mapping or loud failure at `initial_state`); `src/app/features/documents/ingestion/workflow.py` (no behaviour change — the declared channels it already names are the contract); new contract tests under `tests/` (state key-set boundary; cross-contract leakage; multi-contract classification suite) and possibly a fixture under `evals/`. The legacy compatibility endpoint's externally observable success/failure shape may change from "silently degraded success" to either a correct mapping or a typed failure — that endpoint is itself quarantined and deprecated, so this is intentional, not breaking. Normative input: `openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md` (RETIRED — Phase E and the Phase-C residue rows are the source of this change's scope; its completion audit is the authoritative record).
