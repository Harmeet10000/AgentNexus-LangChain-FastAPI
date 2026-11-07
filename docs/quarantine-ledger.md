# Quarantine Ledger

The standing record for surfaces in this tree that are deliberately kept
disconnected: orphans with zero production importers, unmounted routes and
graphs, and compatibility shims that exist only until a named decision retires
them. It was created retroactively from the completion audit of the retired
`PLAN-bulletproof-ingestion.md` (archived at
`openspec/changes/archive/2026-10-31-plan-bulletproof-ingestion.md`, Phase E).

A surface is not deleted because it looks dead. It is quarantined here first,
and it leaves the tree only through a recorded discard decision against its row.

## Standing rules

1. **Nothing gets deleted without a ledger row.** A surface discovered to be
   orphaned, unmounted, or disconnected gains a row *before* any deletion of it
   is proposed.
2. **A discard is a ledger update, not just a commit.** The change that deletes
   a quarantined surface resolves its row in the same change: mark it
   `discarded`, name the port-or-discard criterion that fired, and link the
   commit. Deleting the file and leaving the row untouched is a review failure.
3. **Re-wiring requires the boundary contract first.** Reviving any
   graph-driving quarantined surface requires the state key-set boundary
   contract from `openspec/changes/ingestion-truthfulness-residue/specs/ingestion-state-boundary/spec.md`
   (`set(initial_state) <= set(State.__annotations__)`) to exist and pass against
   the new wiring. `TypedDict(total=False)` plus LangGraph's silent key-dropping
   leave no type-system protection on that edge.
4. **A quarantine without a trigger is a landfill with better prose.** A row
   whose revisit trigger is vague ("later", "someday", "when convenient") is
   rejected in review.

## Ledger

| Surface | Path | Kept because | Port-or-discard criterion | Revisit trigger |
|---|---|---|---|---|
| Docling `ingest_v2` orphan | `src/app/extraction/docling/ingest_v2.py` | Hybrid chunker may beat policy chunking for some document classes; the module has zero production importers (tests only) | Port only if a golden-set comparison shows a chunking-quality win over the live policy path; otherwise discard | The harden change's C10 gold set lands (`openspec/changes/harden-agent-saul-evidence-pipeline`) |
| LangExtract batch processor + Docling preprocessor orphans | `src/app/extraction/langextract/langextract_batch_processor.py`, `src/app/extraction/langextract/docling_preprocessor.py` | Batch mode may matter for corpus re-processing; zero production importers today | Port only through the re-ingestion readiness contract as an approval-gated administrative action; otherwise discard | The `re-ingestion-readiness` capability (`openspec/specs/re-ingestion-readiness/spec.md` once the change archives; active at `openspec/changes/re-ingestion-readiness/`) is satisfied **and** an approved first bulk re-index exists — the row's gate is that capability spec, and its CI inspection tests (no re-ingestion route/tool/grant) are the standing enforcement |
| Folded `features/ingestion` compatibility shim | `src/app/features/documents/api/http/legacy_ingest_router.py`, `legacy_ingest_dependencies.py`; `src/app/features/documents/ingestion/legacy_service.py`, `legacy_dto.py`, `legacy_errors.py` | The package was folded into `documents/` by commit `25ba6f5`; the deprecated `POST /ingestion/documents/upload` route and its fail-closed 503 dependency are retained while production consumers are unknown | Discard when the Sunset date passes with no logged `legacy_ingestion_upload_deprecated` usage; port only if a real consumer appears before then | `API_SUNSET_DATE` is reached, or the first logged legacy consumer appears |
| Legacy graph path state keys | `src/app/features/documents/ingestion/legacy_service.py` (`IngestionService.ingest_document`) | The deprecated shim is the last live caller of the ingestion graph outside the canonical command path; its historical state keys were disjoint from `DocumentIngestionState` | Close with the shim's discard (row above); the explicit mapping landed in `ingestion-truthfulness-residue` task 3.1 and may be deleted only with the shim | Same trigger as the folded compatibility shim row (`API_SUNSET_DATE` or first logged consumer) |
| Unmounted retrieval graph | `src/app/features/documents/retrieval/graph/` (`build_retrieval_graph`; called through `DocumentQueryService.ask_via_retrieval_graph`) | The intended survivor and unification target for `/search/ask` and `/legal/ask`, not a duplicate to kill; the retired plan's older path `src/app/features/documents/retrieval_graph/` is this package after the hexagonal move | Survives when `/search/ask` and `/legal/ask` route through it and the inline duplicate in `documents/application/queries/documents.py` is deleted; otherwise discard | The `mount-retrieval-graph` cutover lands (`openspec/changes/mount-retrieval-graph`); its inline-deletion task resolves this row in the same change |
| Empty `preamble` column on chunks | schema (migration `0003`); search text concatenates `clause_type ‖ preamble ‖ content` with `preamble` always `''` | The column is the reserved seat for contextual retrieval (roadmap G2); filling it before a measurement baseline exists would make the retrieval change an unmeasured guess | Port: fill via the `contextual-chunk-retrieval` contract with a measured default-on; discard: drop the column and simplify the search-text concatenation if the baseline shows no win | The `retrieval-measurement-gate` baseline lands (`openspec/changes/retrieval-measurement-gate`); owned by `openspec/changes/deferred-retrieval-roadmap` |
| Reserved fourth-branch registration point at the fusion call site | the weighted-RRF fusion in `documents/application/queries/documents.py` (point reserved by `mount-retrieval-graph`) | Graph-traversal fusion (roadmap G4) is deferred until its weight can be measured; relationship edges exist (harden 4.1 ✅) but no baseline attribution yet | Port: admit the fourth branch with a baseline-cited weight per the `hybrid-retrieval-ranking` delta; discard: remove the reserved parameter if the baseline shows no contribution | The `retrieval-measurement-gate` baseline lands; owned by `openspec/changes/deferred-retrieval-roadmap` |

## Retroactive row history

- The retired plan recorded the first row against `src/app/features/ingestion/`
  before commit `25ba6f5` folded that package into `documents/`. The row above
  is the folded successor; the old path no longer exists.
- The retired plan recorded the retrieval-graph row against
  `src/app/features/documents/retrieval_graph/`; the hexagonal restructure moved
  it to `src/app/features/documents/retrieval/graph/`. Same surface, new path.
- The legacy graph path keys row is new in this ledger: the audit found the
  mismatch after the plan's Phase E table was written. Task 3.1 of
  `ingestion-truthfulness-residue` replaced the silent drop with an explicit
  mapping (or a typed failure naming unmapped keys); the row remains because
  the shim itself remains.

## Referenced by

- `openspec/changes/ingestion-truthfulness-residue` — creates and enforces this
  ledger (`specs/quarantine-ledger/spec.md`).
- `openspec/changes/re-ingestion-readiness` — the batch-processor row's revisit
  trigger points at its contract.
- `openspec/changes/mount-retrieval-graph` — the retrieval-graph row resolves at
  its cutover, and its inline deletion updates this row in the same change.
- `openspec/changes/deferred-retrieval-roadmap` — owns the two roadmap-deferral
  rows (preamble column, fusion registration point) and the gated contracts
  (G2/G4/G5/G6-remainder/G7) whose triggers resolve them.
