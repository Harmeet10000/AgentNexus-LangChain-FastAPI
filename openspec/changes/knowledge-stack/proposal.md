# Run extraction before chunking, and reason over the stored tree instead of an external service

**Class: M.** Two complete modules with zero runtime call sites become one wired stage and one
retired dependency. The scope of the second half depends on a measurement taken as its first task.

## Why

Two substantial packages sit in the tree with **no runtime callers at all**.

`shared/rag/langextract/` is a complete structured-extraction module. Nothing calls it, so clause-level
entities are never extracted, and the graphiti episode writer's dependency on them
(`graphiti/write_clause_episodes.py:42`) exists only under `TYPE_CHECKING` — a type-level edge with no
runtime edge beneath it.

`shared/rag/pageindex/` is a complete client for an external reasoning-over-document-tree service. It
is re-exported from `shared/rag/__init__.py:3` and its construction sits commented out at
`lifespan.py:538`. Wiring it would add an external service dependency and send document content to a
third party, for a capability the system can have locally: the document parser already produces a
structural tree, and reasoning over that tree needs no external call.

Graphiti itself is built and wired, and `graph-entity-canonicalisation` already specifies idempotent
canonical graph writes — so the missing piece is not the graph, it is the extraction that should be
feeding it.

## What Changes

- **Extraction runs before chunking.** A structured-extraction stage runs on the parsed document and
  its output is available to the chunk writer, so extracted entities and the chunks they describe are
  produced from the same parse rather than reconciled afterward.
- **Extractions become graph episodes**, written idempotently under re-ingestion — satisfying
  `graph-entity-canonicalisation` rather than restating it. The `TYPE_CHECKING` import at
  `write_clause_episodes.py:42` becomes a real one.
- **Extraction failure is non-fatal.** An unavailable extraction provider leaves ingestion completing,
  with the document flagged extraction-incomplete rather than silently indistinguishable from a
  document that had nothing to extract.
- **Structural navigation reasons over the persisted tree**, exposed as a retrieval branch through the
  repository. No external service is called.
- **The external client's package surface is retired** — the re-export, the commented construction,
  and the package.

## Capabilities

**New Capabilities**

- `knowledge-extraction-stack` — extraction ordering, episode emission, non-fatal extraction failure,
  local tree reasoning, and retirement of the external surface.

**Modified Capabilities**

None. This change **cites** `graph-entity-canonicalisation`, which already requires that canonical
graph writes are idempotent and that re-ingestion does not duplicate entities. This change satisfies
that requirement by routing its episodes through the existing writer; it does not tighten it.

## A measurement gates half of this change

**Whether the parser's structural tree is persisted at all is not established, and is not assumed.**
Task 1.2 settles it before any tree-reasoning work is designed.

- If the tree **is** persisted, task group 4 is a navigator over stored data.
- If it is **not**, task group 2 becomes a persistence task first, and this change grows from wiring
  into a storage build.

This is deliberately the first task, and the proposal states the conditional rather than picking the
likely branch, because the two shapes have materially different costs.

## Impact

- **Code:** `src/app/shared/rag/langextract/` (wired), `src/app/shared/rag/graphiti/write_clause_episodes.py`,
  `src/app/shared/rag/pageindex/` (deleted), `src/app/shared/rag/__init__.py:3`,
  `src/app/lifecycle/lifespan.py:538`, plus a retrieval branch and possibly a document column.
- **Behaviour:** clause entities reach the knowledge graph. Structural navigation becomes available
  without an external call. A document whose extraction failed is identifiable.
- **Dependencies:** one external service dependency and its package leave. One extraction provider key
  is required, with the system degrading when it is absent.
- **Not touched:** chunking policy, chunk identity, the embedder, retrieval SQL, fusion, the reranker,
  graph lifecycle. `graph-lifecycle` deliberately left `app.state.pageindex_client` unwired for this
  change to remove.
