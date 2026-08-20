> Change class: **L** — cross-cutting. Touches two feature modules, the retrieval graph, the Celery task
> registry and the test conftest; changes a tenant-isolation boundary; and defines the document/chunk schema
> contract that changes 0, 1 and 3 build on.

## Why

`features/search/` is not the future of retrieval — it is the **pre-unified twin that was left behind**.
`features/documents/` already holds a complete retargeted **superset** of search's *retrieval* surface, and the
**mounted** documents router already exposes an equivalent of every read endpoint search's unmounted router
exposes, with one exception that costs nothing: search's task-status read has no documents equivalent keyed by task
(documents exposes a status read keyed by document instead), and the ingest whose task it tracked is deleted by
this change. So this is de-duplication, not a port: we subtract the twin, relocate the schema-free helpers that
`documents/` currently imports *back out of* `search/`, and name one document store and one chunk store as the sole
retrieval truth.

Deleting the twin's write path costs nothing, and that is now a positive proof rather than an absence of one:
its ingest call publishes an outbox event, the outbox table does not exist, and the asynchronous consumer of
that ingest fires **only** from the event that write would have emitted. So even in the counterfactual where the
router had been mounted, the request would have failed on a nonexistent relation and the consumer could never
have run. The write path is not merely unreached — it is unrunnable.

## What Changes

- Declare the unified document/chunk store the **sole retrieval truth**. `clauses`, `search_documents`,
  `search_chunks`, `statutes` and `document_vectors` are not retrieval tables and never become one.
- Relocate the schema-free retrieval helpers (chunking, rank fusion, RAG assembly, retrieval constants) into
  the documents feature, inverting today's `documents → search` import direction.
- Delete the schema-bound twin: the superseded search document/chunk models, their repository, router,
  dependency layer, ingest service path, ingest DTOs and Celery ingest task.
- Retarget the retrieval graph's fused-search call off `clauses` — a table **no migration creates and no
  environment has ever had** — onto the unified chunk store.
- Remove the second, reader-less derived text-search column and its index **from the source of truth only**.
  This change ships **no DDL and drops no table**: the superseded tables were never created in any environment,
  so there is nothing to drop, and all schema authorship belongs to change 0.
- Add a static drift gate asserting that every database index and unique constraint named inside query text is
  created by a migration, on a table a migration creates. Measured under that rule, it is **red on exactly one
  count today** — the clause keyword index, declared on a table no migration creates — and that one identifier has
  two source readers, both of which this change removes.
- Make a failed retrieval branch fail the request instead of quietly fusing a result from whichever branches
  happened to succeed.
- Record chunk modification time so a re-embedding campaign can distinguish a current-generation embedding
  from a carried-over one.
- Give statute identity a home on the chunk row — instrument name, section reference and instrument year, with one
  partial index serving the point lookup and the newest-applicable-year rule — so the legal-corpus capability change
  3 specifies has a schema that can satisfy it, and no `statutes` table is ever created.
- **BREAKING (spec level, not API level):** the `llm-injection` capability's requirement bound to the
  superseded search service is removed, and its dependency-layer requirement is narrowed to the surviving
  document path. **No mounted API request or response contract changes, and no endpoint becomes newly
  reachable** — the deleted router was never mounted.

## Scope / Non-Goals

In scope: source-code subtraction, helper relocation, the graph retarget, the drift gate, and the **schema
specification** this change hands to change 0.

Out of scope, each recorded with its owner in `design.md`: **all DDL and migration authorship** (change 0 owns
the single authoritative create-schema migration); mounting the search router; replacing the deleted raw-text
ingest path; re-ranking and retrieval-quality tuning; the embedding-dimension and normalization decisions; and
the broken owner-resolution dependency on the mounted router.

## Capabilities

### New Capabilities
- `document-retrieval-schema`: one document store and one chunk store as the sole source of retrieval truth —
  per-tenant document identity, mandatory ownership and object provenance, three rank-fused retrieval modes
  over one derived searchable text, tenant-scoped reads, and the guarantee that every database identifier a
  query names actually exists.

### Modified Capabilities
- `llm-injection`: the requirement that the superseded search service take its model by constructor injection
  is **removed** — that service ceases to exist and its graph-backed ask path moves to the document query
  service, which the capability's own document-injection requirement already governs. The dependency-layer
  requirement is **modified** to drop its search half; its surviving documents half keeps the same contract, though
  the wording is restated to describe behaviour rather than name the deleted dependency module and its private
  model-builder function, and it gains a scenario asserting that no provider survives for the dissolved service.

## Impact

- `src/app/features/documents/` — gains `constants.py`, the relocated helpers, the retargeted imports, the
  chunk modification-time column and its write in both the upsert conflict set and the row builder, the three
  nullable statute identity columns on the same terms, and the fail-the-request handling of a failed retrieval
  branch.
- `src/app/features/search/` — reduced to the embedding client and its package init, awaiting change 1.
- `src/app/shared/langgraph_layer/retrieval_kb/` — fused search retargeted off `clauses`.
- `src/app/shared/langgraph_layer/ingestion_kb/` — one string literal only: the keyword-index name in the
  force-merge maintenance call, retargeted off the phantom clause index so the drift gate can be green. The rest of
  that module is change 1's.
- `src/tasks/` and `src/app/connections/celery.py` — the search ingest task and its registration removed.
- `tests/conftest.py`, `tests/integration/test_search.py`, `tests/unit/search/` — repointed; one new drift test.
- Change 0 receives a column-and-index specification; change 1 receives an accepted schema ADR it must build
  its persistence nodes against.

## Risks

- The global test conftest imports twenty-one symbols from the module being deleted, so a mistimed deletion is a
  collection error for the **entire** suite, not just one feature's tests.
- Nothing in the repository executes SQL against a database, and the target tables have never existed in any
  environment — so every column, index and constraint name here ships unverified unless the real-database gate
  lands.
- The superseded and target retrieval indexes are alike **defined but never created**; no reader of this change
  may conclude that keyword, vector or fuzzy retrieval works today.
- Per-tenant document identity makes cross-tenant duplicate storage the normal case. That is the intended
  trade, taken deliberately over a cross-tenant information leak.
