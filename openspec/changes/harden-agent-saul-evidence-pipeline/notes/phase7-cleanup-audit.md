# Phase 7 — cleanup audit (tasks 7.4–7.10)

Audit date: 2026-11-14, at HEAD `f36bf5d` (plus concurrent-session edits). Every
"consumers" line is the result of a repo-wide `rg` at audit time; load-bearing
verdicts name the production import chain.

---

## 7.4 — import/consumer audit

### A1. The Phase-3 graph (`ingestion_kb`)

- **Path today:** gone. `rg "ingestion_kb"` over `src/` and `tests/` returns
  zero code hits. Remaining mentions are historical prose only: the retired
  plan (`PLAN-bulletproof-ingestion.md`), `docs/fresh-environment.md`, a deleted
  historical performance-audit snapshot, `tests/performance/todo.md` (a frozen
  checklist), and one stale line in the
  archived spec `openspec/specs/shared-infrastructure-errors/spec.md:252`
  (a historical-error-count table that still cites `ingestion_kb/nodes.py` as
  an `TransientExternalError` raiser — prose residue, not a code path).
- **Deleted by:** commit `3bd67da` ("delete dead trees"), before the quarantine
  ledger existed; the deletion is recorded in the retired plan's Phase E.
- **Load-bearing:** no — the canonical queued pipeline is
  `src/app/features/documents/ingestion/workflow.py` (the `tasks.documents_ingest`
  graph), consumed by `application/commands/upload_document.py` and
  `worker/jobs/documents.py`; the boundary seam is pinned by
  `tests/unit/test_graph_state_boundary.py`.
- **Recommended action:** none in code. Tick 7.8 with the pointers below.

### A2. `/ingestion` legacy shim

- **Paths:** `src/app/features/documents/api/http/legacy_ingest_router.py`
  (mounted in both `src/app/api/v1/__init__.py` and
  `src/app/api/v2/__init__.py`); support modules
  `ingestion/legacy_service.py`, `legacy_dto.py`, `legacy_errors.py`.
- **Consumers:** the two API-version mounts only; no other src importer. The
  service is the last caller of the legacy ingestion-graph mapping (ledger row
  "Legacy graph path state keys").
- **Load-bearing:** yes while the compatibility window runs —
  `API_SUNSET_DATE = 2027-06-15` (`config/settings.py`), Deprecation/Sunset/
  successor-version headers + the `legacy_ingestion_upload_deprecated`
  structured event are emitted on every request
  (`legacy_ingest_router._mark_legacy_upload`/`_log_legacy_usage`), enforced by
  `tests/unit/features/documents/api/test_legacy_upload_deprecation.py` and
  `tests/unit/documents/test_re_ingestion_readiness.py`.
- **Recommended action:** keep the route until the sunset date (ledger discard
  criterion: "Discard when the Sunset date passes with no logged usage; port
  only if a real consumer appears"). Do NOT remove now — tick 7.7 on the
  documented-window evidence (below).

### A3. `token_counter`

- **Path:** `src/app/retrieval/token_counter.py` (moved out of `shared/rag` by
  commit `03f1a7d`).
- **Consumers (production, all live):** `retrieval/edge_expansion.py`,
  `features/search/application/context_assembly.py`,
  `application/queries/documents.py` (fused search token budgeting),
  `documents/retrieval/service.py` (the 5.1 `retrieve` service),
  `documents/retrieval/context_assembly.py`, and the retrieval graph
  (`graph/nodes.py`, `graph/hydration.py`, `graph/state.py`); plus
  `shared/langchain_layer/messages.py`.
- **Load-bearing:** yes — it budgets every fused search and the ask path.
- **Recommended action:** keep. No stale references found.

### A4. `structural_navigator`

- **Path:** `src/app/retrieval/structural_navigator.py`.
- **Consumers:** `application/queries/documents.py` (the ask path's structural
  narrowing) and its unit fixture `tests/unit/shared/rag/test_structural_navigator.py`
  (test path kept from the retired `shared/rag` suite — still green, name is
  historical only).
- **Load-bearing:** yes.
- **Recommended action:** keep. (The `tests/unit/shared/rag/` directory name is
  a historical artifact, not a stale import; renaming it is churn with no
  behavior effect and is left to a later cosmetic pass.)

### A5. `errors` (the shared retrieval errors module)

- **Path:** `src/app/retrieval/errors.py` (`RagValidationError`, `RagResult` —
  relocated from `shared/rag` by `03f1a7d`).
- **Consumers (src):** exactly two —
  `extraction/langextract/docling_preprocessor.py` and
  `extraction/langextract/langextract_to_graph.py`. Both are the
  **quarantine-ledger orphan modules** ("LangExtract batch processor + Docling
  preprocessor orphans" row: zero production importers). No test imports it.
  The kernel-wide error base (`app/kernel/errors.py`, 33 consumers) is the real
  shared error vocabulary; per-feature `*/domain/errors.py` modules own their
  codes.
- **Load-bearing:** no — its only consumers are quarantined orphans. It is
  **chained** to those rows: when the batch-processor/preprocessor row resolves
  (port or discard), this module resolves with them. The ast-grep ADR-006 ignore
  already covers `retrieval/**/errors.py`, so keeping it costs nothing today.
- **Recommended action:** keep under the existing ledger row's trigger. Removing
  it now would break the orphan modules' imports without resolving their row.

### A6. Stale PageIndex / shared-RAG references

- **Code:** zero. `rg -i "pageindex|shared/rag|shared\.rag"` over `src/` finds
  only a historical comment inside migration
  `src/alembic/versions/0013_authoritative_target_shape_outbox_documents_retrieval_indexes.py:101`
  (a migration docstring quoting the pre-move path — migrations are immutable
  history and must not be edited).
- **Docs/tests prose:** `tests/performance/todo.md` (frozen checklist, many
  hits — historical record, not documentation of current reality) and the two
  frozen docs snapshots above. None of these are imported or rendered anywhere.
- **Fixed in this change (7.10):** the live stale prose that DID describe
  current code — `application/tools/bundle.py`'s "not yet wired into any node"
  memory-writer sentence (superseded by 5.8's registry contract and 6.10's
  approval-gated `persist_memory` node) and its pre-registry wiring narrative.
  No PageIndex/shared-RAG prose existed in live `src/`.

### A7. Trivially stale fixes applied by the audit

- `application/tools/bundle.py` module + class docstrings rewritten to the
  current registry-driven reality (5.8) — see 7.10 below. No dead re-exports
  and no moved-module imports were found anywhere else (`rg` for the audited
  surfaces returned only live seams), so no import edits were needed.

---

## 7.5 (H1) — Phase-3 harvest: canonicalize + contextualization prompt

**Verdict: the precondition is moot — both harvest targets already exist in the
live tree; nothing needs a preservation branch.**

- `canonicalize.py` does not exist anywhere (`find src -name "*canonicalize*"`
  is empty; `rg canonicaliz` over `src/` finds only the tenant-mirror migration,
  the docling chunker's ordinary vocabulary, and the like — no Phase-3 entity
  canonicalization implementation survived `3bd67da`). The Phase-3
  canonicalization implementation died with `ingestion_kb`; E5 evidence for
  porting it does not exist (see 7.6).
- The **contextual preamble** (the one good Phase-3 idea, per the retired plan's
  G2 note) was re-implemented in the live pipeline, not harvested from the dead
  tree: `src/app/features/documents/ingestion/chunking/context_summary.py`
  (one context call per document) fills the reserved `preamble` column behind
  `DOCUMENTS_CONTEXTUAL_PREAMBLE_ENABLED`, landed by `deferred-retrieval-roadmap`
  and recorded HALF-RESOLVED (ported) in the quarantine ledger's retroactive
  history (2026-11-08).
- Therefore the "tag the harvest commit and copy to `archive/phase3-harvest`"
  step is moot: there is no file to preserve that is not already either (a)
  deleted with its own recorded decision (`3bd67da` + the retired plan's Phase E
  record) or (b) alive in the canonical tree with a cleaner implementation.
  Creating `archive/phase3-harvest` would archive nothing.

---

## 7.6 (H1/E3–E5) — port-or-discard decisions with evidence links

| Phase-3 idea | Decision | Evidence | Pointer |
| --- | --- | --- | --- |
| Entity canonicalization | **DISCARD — E5 BLOCKED, no evidence, no code** | E5 recorded BLOCKED by the concurrent Phase-8 run (8.6): "canonicalization code does not survive in the tree" — `rg canonicaliz` over `src/` matches only an unrelated alembic tenant-mirror migration; the entity harvest that would feed it is not implemented, and no E5 numbers exist. The experiment is recorded `REQUIRES-HARVEST` (nothing invented). If the harvest+E5 later show a win, canonicalization ports as NEW work in the live pipeline, not as a Phase-3 harvest. | `evals/reports/phase8_experiments.json` (`entity-canonicalization-e5`); this audit §7.5 |
| Contextual preamble (one LLM context call per chunk) | **PORTED — implemented, flag-off, pending live measurement** | E3 EXISTS: `evals/reports/contextual_preamble_legal_retrieval_v2_wave4.comparison.json` — contracts stratum recall@k 0.5602 → 0.7030 (**+0.143**), overall recall +0.123, BUT per-class regressions on filings/judgments (nDCG −0.244/−0.193 offline proxy), and one live generation run is still outstanding. Decision per the task rule ("only if E3 justifies one LLM call/chunk"): the per-DOCUMENT amortized implementation (not per-chunk) stays behind `DOCUMENTS_CONTEXTUAL_PREAMBLE_ENABLED` (default off) until the live run lands; matches the ledger preamble row's HALF-RESOLVED status. | `src/app/features/documents/ingestion/chunking/context_summary.py`; ledger retroactive history 2026-11-08 |
| `bm25_force_merge` | **DISCARD — the E4 gain condition is NOT met** | E4 MEASURED (the concurrent Phase-8 run, 8.6): `evals/reports/phase8_experiments.json` (`bm25-force-merge-e4`) + harness branch sweeps `e4_bm25_only_legal_retrieval_v2.json` (recall@10 0.5806, MRR 0.3828, nDCG 0.4281) vs the fused default (recall 0.5774, MRR 0.3889, nDCG 0.4305) vs fused+graph@0.1 (recall 0.6032 — dominates every configuration). bm25-alone's +0.32 recall points come with −0.61 MRR and −0.24 nDCG; the trigram leg adds ranking precision; the flag "production flag does NOT exist" — recorded `NEEDED-FLAG` with the sweep as the A/B protocol for a future candidate-generation flag. The Phase-3 force-merge semantics are not ported; a flag that cites the sweep would be new work. | `evals/reports/phase8_experiments.json`, `evals/reports/e4_*_legal_retrieval_v2.json`; the fusion registration-point ledger row |
| Parallel per-chunk fan-out | **DISCARD** | Conditional per the task ("only with contextualization"): the only contextualization that landed is per-document amortized (`context_summary.py`), not per-chunk, so the per-chunk parallel fan-out that contextualization justified has no trigger and no surviving implementation. | 7.5 above; ledger preamble row |
| Universal `chunk_kind="clause"` / empty-warning / never-verified semantics | **DISCARDED (verified gone)** | `rg -i 'chunk_kind\s*=\s*"clause"|empty-warning|empty warning' src/app` returns zero hits; warning semantics are the measured fidelity gates (4.3: typed `FIDELITY_*` reasons with versioned thresholds, `ingestion/chunking/fidelity.py`; the 8.4 regeneration added the warning_reasons gap fill), and `graphiti_verified` is persisted coverage metadata, never legal truth (1.7). | `ingestion/chunking/fidelity.py`, `notes/phase0-verification-records.md` §V7/V9 |

---

## 7.7 — `/ingestion` documented-window behavior

**Verdict: route STAYS; removal is scheduled by its ledger row; nothing to do
in code.** Evidence:

- Sunset gate present: `API_SUNSET_DATE: str = Field(default="2027-06-15")`
  (`src/app/config/settings.py`), emitted as the `Sunset` response header with
  `Deprecation: true` and the successor `Link` on every legacy request
  (`legacy_ingest_router._mark_legacy_upload`).
- Usage logging present: `legacy_ingestion_upload_deprecated` structured event
  (`_log_legacy_usage`) is exactly the event the ledger's discard criterion
  watches ("Discard when the Sunset date passes with no logged
  `legacy_ingestion_upload_deprecation` usage; port only if a real consumer
  appears").
- Window enforcement pinned: `tests/unit/features/documents/api/test_legacy_upload_deprecation.py`
  + `tests/unit/documents/test_re_ingestion_readiness.py` (V3/A4 evidence, task
  1.3 recorded in `notes/phase0-verification-records.md` §V3).
- The route is mounted in both API versions (`api/v1/__init__.py`,
  `api/v2/__init__.py`) — removal of route + lifespan wiring + the shim row
  happens together when the trigger fires, per the ledger row. **Not removed
  now** (2026-11-14 is before the 2027-06-15 sunset).

---

## 7.8 — `ingestion_kb` deletion verification

Already deleted (commit `3bd67da`); tree restructures since have left zero
remnants. Verified 2026-11-14:

- `rg "ingestion_kb"` over `src/` and `tests/` → **zero code hits**; no test
  references the surface. Remaining hits are historical prose (retired plan,
  frozen docs snapshots, `tests/performance/todo.md`) and one stale historical
  error-count line in the archived spec
  `openspec/specs/shared-infrastructure-errors/spec.md:252` (prose residue in an
  archived table; not a code path — left untouched, archived specs are
  out of this change's scope).
- Preconditions: I4 fidelity green — `evals/reports/ingestion_fidelity_i4_offline.json`
  (gate=pass, 25/25 gold rows, live-run pending honestly recorded) with the
  CI-enforced gate `tests/unit/shared/evaluation/test_ingestion_fidelity_gold_set.py`;
  the archived measurement-gate change `2026-11-07-retrieval-measurement-gate`
  supplies the measurement baseline; H1 decisions recorded above (7.6);
  canonical queued-ingestion parity green (Phase 4's `test_ingestion_truth_fidelity.py`
  suite + the boundary seam `tests/unit/test_graph_state_boundary.py`).

---

## 7.9 — `hybrid_retrieve_precedents` deletion verification

Already deleted (commit `accf5a2`), verified by Phase-5 task 5.7. Verified
2026-11-14:

- `rg "hybrid_retrieve_precedents"` over `src/` and `tests/` → **zero code
  hits**. The only mentions: the deliberate historical record in
  `application/tools/precedent_tools.py`'s module docstring, the frozen
  `tests/performance/todo.md`, and this change's own tasks.md.
- Parity: the tool was never bound to any agent (5.7's tick records its
  never-bound status); `search_legal_precedents` covers the same Graphiti leg
  plus the shared tenant-scoped fused Postgres leg (2.3/3.3). Dependency scans:
  `lint-imports` 5/5 and the ast-grep lane are clean on the retrieval seam
  (Phase-1/5 records).
