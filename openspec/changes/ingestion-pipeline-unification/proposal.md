> Change class: **L** cross-cutting (multi-module, data-shape dependency, new external dependency, deployment surface)

## Why

Document ingestion has two implementations: a decorative single-stage graph that wraps ten opaque straight-line
stages and is the only reachable path, and a genuine multi-stage graph with fan-out and reducer-based chunk
accumulation that nothing can reach. The reachable one loses paragraph 201 onward of every legal document
without warning, embeds at a width no vector column accepts, blocks the event loop through an entire parse, and
has no recovery unit smaller than "start over".

## What Changes

- **Correctness fixes that are already wrong today**: a diagnostic warning path that raises instead of warning; a
  1536-dimension embedding contract against 768-wide vector columns; a zero-vector substitution that turns a
  failed embedding into a valid row that ranks against nothing; a reference to a module that does not exist; a
  degraded-LLM branch whose own error handling destroys the diagnostic it was written to preserve.
- **One embedding path** replacing four, with the dimension read from configuration, explicit task type on both
  the query and document sides, one cross-process cache, and one normalisation convention per column.
- **Structure-aware chunking for legal documents**, replacing a blank-line regular expression and a silent
  200-block truncation; parse stops blocking the event loop and stops discarding parsed tables.
- **Checkpointing that can actually be constructed**: an application-owned connection pool, a teardown that
  closes what it owns, a driver-shaped connection string carrying credentials, and no import-time alias that
  turns a missing database driver into a `None` returned from a function typed to return a saver.
- **Pointer state**: checkpointed channels carry identifiers, not document bytes, and the failure channel becomes
  a serialisable record instead of an exception instance.
- **Entity canonicalisation before any knowledge-graph write** — the only irreversible requirement in this
  change.
- **A queue consumer**: the deployment gains a worker and a scheduler process. Today nothing consumes the queue,
  so every dispatched ingestion task enqueues forever, and the documented start command names an application
  module that does not exist.
- **Retrieval quality**: the existing lexical and fusion implementations are harvested into one path, index
  identities stop being string literals inside queries, and re-ranking — the one genuinely missing third of the
  hybrid contract — is wired in. Structure extraction moves upstream of persistence and graph writes.
- **The fold**: the multi-stage graph absorbs the three concerns only the live path has (object-store fetch,
  status transitions, per-chunk graph verification), each chunk is written once instead of twice, and the
  single-stage wrapper is deleted.
- **BREAKING (internal)**: the single-stage ingestion graph and its state type are removed. No public HTTP
  contract changes; the upload surface, its request shape, and its status vocabulary are preserved.

## Scope / Non-Goals

In scope: the ingestion graph, document parsing and chunking, the embedding path, the retrieval ranking path,
the checkpointer, the queue deployment, and entity canonicalisation.

Out of scope, each recorded in `design.md` Non-Goals with its reason: any database migration (the foundation
change owns all schema); mounting the retrieval router or the ingestion router; the search-to-unified table
collapse; agent state shape, tool registry, and model-call middleware; a vector-store singleton; a second
document parser; adopting a whole external agentic-retrieval architecture.

## Capabilities

### New Capabilities
- `document-ingestion-pipeline`: an uploaded document becomes retrievable chunks with a reported terminal status, through one recoverable multi-stage pipeline.
- `hierarchical-document-chunking`: structure-aware, token-bounded chunking that never silently discards content and never blocks the event loop.
- `unified-embedding`: one embedding provider, one configured dimension, declared task type, one cache, one normalisation convention.
- `langgraph-checkpointing`: constructible, resumable pipeline persistence with an application-owned pool and reference-only state.
- `celery-worker-deployment`: a process consumes the queue, the documented command starts it, and registration does not rely on an import side effect.
- `graph-entity-canonicalisation`: extracted entities resolve to one stable identity before any knowledge-graph write.
- `hybrid-retrieval-ranking`: lexical, vector, and fuzzy branches fused once and re-ranked, with extension availability as a declared precondition.

### Modified Capabilities
- `typed-exception-handling`: embedding failures raise instead of substituting a placeholder vector, and retry boundaries retry named transient types only while preserving the original exception for callers.

## Impact

- Document ingestion, document parsing and classification, the embedding clients, the retrieval repository and
  fusion, the ingestion graph and its nodes, the checkpointer, the queue configuration, and the deployment
  compose file.
- New external dependency: a working PostgreSQL driver binding for the checkpointer. The dependency is declared
  today but its libpq binding is absent, which is why the import-time alias fallback is the live path.
- Depends on the foundation change for: the target schema (this change ships no migration), the deleted
  syntactically-invalid module that holds the lint baseline at 125 rather than 123, the tidy of the task package
  whose import side effect currently guarantees dispatch, the request-scoped user identity dependency, and the
  health probe that is this change's only observable degradation signal.

## Risks

- The promoted modules have **zero** covering tests, so "it still works" is unfalsifiable from lint and types
  alone. Every step whose defect is invisible to lint carries a mandatory new test.
- The coverage gate fails below eighty per cent against current coverage, so a fully green suite still exits
  non-zero. Verification compares summary counts, never exit codes.
- Entity canonicalisation is irreversible: duplicate party nodes cannot be separated once written, because the
  disambiguating context is the extraction already discarded.
- The pipeline graph's shared wiring stays deliberately disabled, so the end-to-end acceptance check is not
  available inside this change; correctness for those pieces is proven by construction.
- Ingestion dispatch has **four independent breaks stacked on one path**, each of which hides the next: the
  durable outbound event table does not exist, so the insert raises; the relay that would consume it swallows the
  missing table behind a catch-all and emits one warning line; the target task is registered only as an import
  side effect; and no process consumes the queue at all while the documented start command names a module that
  does not exist. Fixing any single layer produces **no observable improvement**, which is precisely the trap a
  future maintainer falls into. The first two layers are the foundation change's to fix; this change owns the last
  two and must not assume the first two work.
- No verification step in this change may be expressed as "a durable outbound event fires" — that check cannot
  pass until the foundation change creates the event tables.
