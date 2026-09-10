# Repair the source tree and the specification baseline

**Class: L.** It is small in lines and cross-cutting in consequence: nothing else in this cluster can
be measured, linted, or spec-amended until it lands.

## Why

The working tree does not import. `src/app/shared/rag/document_processing/` is deleted-and-staged
while its replacement `src/app/shared/rag/docling/` is untracked and still imports the package it
replaces, so `import app.main` fails and no gate in this repository is currently measuring anything
real. Separately, ten capabilities that later changes in this cluster must amend exist only as deltas
inside archived changes and are absent from `openspec/specs/`, so a `## MODIFIED Requirements` block
in any of them would have no base to modify.

## What Changes

- Track `src/app/shared/rag/docling/` in version control. It is untracked today, so every subsequent
  proof in this cluster would run against files the repository cannot see, and a commit would land a
  tree that does not import.
- Finish the seven-item repoint so no module under `src/` or `tests/` references
  `app.shared.rag.document_processing`, including the five per-file lint exemptions in
  `pyproject.toml` and one exemption for a module the rename did not carry over
  (`document_processing/ingest.py`, which has no counterpart in the new seven-file package).
- Restore `src/app/features/__init__.py` as an **empty** file, clearing 12 pre-existing `INP001`
  diagnostics. It must stay empty: it was deleted in `8e25352` precisely to sever model-import and
  router-import coupling.
- Repair `src/alembic/env.py` until `alembic current` executes, and record the live revision. Three
  sources currently disagree about the head — disk carries `0001`–`0017`, the handover claims
  `b3e7c41d92af` (matching no revision on disk), project memory says the database is stamped at
  `0004`. At least one is wrong, and the chunk-identity migration in `ingestion-chunking` cannot
  choose a `down_revision` until this is settled by measurement.
- **Restore ten archived capabilities into `openspec/specs/`**, verbatim from their archived deltas.
  Evidence that they were never applied is recorded in `review.md`.
- Record measured lint, type, test and migration outputs to durable baseline files that every later
  change in this cluster compares against, rather than against numbers written in prose.

No behaviour of the running application changes. No migration is authored or applied.

## Capabilities

**New Capabilities**

- `source-tree-integrity` — the application tree imports, is fully tracked, carries no stale lint
  exemptions, runs its migration tooling, and has a recorded gate baseline; and every capability this
  cluster amends is present in the specification baseline before it is amended.

**Modified Capabilities**

None. This change *restores* ten capabilities into the baseline but writes no delta against any of
them — the changes that amend them own that, and doing both here would make one change both the
author and the amender of the same requirement.

## Impact

- **Code:** `src/app/shared/rag/docling/` (7 modules), `src/app/features/documents/{parser,classification}.py`,
  `src/app/shared/langgraph_layer/ingestion_kb/nodes.py`, `src/app/examples/policy_examples.py`,
  `src/app/utils/embedding.py`, `src/app/shared/rag/langextract/langextract_to_graph.py`,
  `src/alembic/env.py`, `src/alembic/versions/0013_*.py` (comment only), `pyproject.toml`.
- **Tests:** four test modules import the retired path; one
  (`tests/unit/shared/rag/test_rag_agent_embedder_import.py:38`) references it as a **string**, not an
  import, so a symbol-import probe cannot see it and collection is the only honest check.
- **Specs:** creates ten files under `openspec/specs/`. Blocks all six other changes in this cluster.
- **Not touched:** the three `PLC2701` private-import sites, embedder unification, chunking, and all
  retrieval SQL. Those belong to other changes and are deliberately left red here.
