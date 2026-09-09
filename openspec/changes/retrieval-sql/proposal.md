# Collapse retrieval onto one fused path planned over its own indexes

**Class: L.** Two divergent implementations of the same three-branch fusion are reachable from two
different doors, and the one the retrieval graph actually uses cannot reach the indexes it was written
for.

## Why

`DocumentRepository.legal_rrf_search` (`src/app/features/documents/repository.py:596-728`) is what the
retrieval graph calls (`retrieval_kb/nodes.py:237`). It is a single monolithic weighted CTE, and
measurement of its shape shows four distinct defects:

- **It cannot use its own indexes.** All three legs select `FROM candidate_chunks`, a CTE referenced
  **three times**. Postgres inlines a `WITH` clause only when it is referenced exactly once; three
  references force materialisation, which puts a temporary relation between the query and
  `chunks_bm25_idx` / `chunks_embedding_idx`. The indexes the query was written for are unreachable
  from it.
- **It skips the vector tuning the other path sets.** No `SET LOCAL diskann.*` runs on this path.
- **Its trigram leg ranks an arbitrary sample.** `LIMIT 50` with no statement-level `ORDER BY` selects
  fifty rows in whatever order the plan produces them, and then ranks those.
- **It honours fewer filters** than the shared `_FILTER_SQL` — no `chunk_kind`, no jsonb containment,
  no parties.

Meanwhile a second, better implementation exists: three branch methods over base `chunks`
(`bm25_search:405`, `vector_search:452`, `trigram_search:507`) fused in Python by `fusion.py:28`.

So **the same query returns different results depending on which door it entered**, and the door the
agent uses is the worse one. It also hard-codes `60.0` and a `0.15` trigram weight directly beside
`constants.RRF_K`, so the two paths' fusion constants can drift silently.

## What Changes

- **One fused path.** The three branch methods become the single source of truth for each leg's SQL,
  each over base `chunks`, each carrying its own `ORDER BY` and `LIMIT` in the same statement as its
  ranking expression. `legal_rrf_search` is deleted and the retrieval graph routes through the shared
  fusion.
- **Per-leg fusion weights become explicit** named constants, added to `reciprocal_rank_fusion` as a
  keyword-only argument defaulting to unweighted — so the second consumer at
  `search_legal_precedents.py:188` is untouched.
- **Tenant scoping moves onto the chunk relation.** `chunks.user_id` and `ix_chunks_user_document`
  already exist (`model.py:155`), so this needs no migration. Today every leg detours through
  `JOIN documents ... WHERE d.user_id`, which forces filtering *after* the approximate-nearest-neighbour
  scan and lets the candidate pool be exhausted before the tenant's own rows are reached.
- **Every leg gets a deterministic tiebreaker**, so two runs over unchanged data produce byte-identical
  ordering.
- **Exact-phrase search is fixed without `tsvector`.** The keyword extension has no phrase search; the
  vendor-prescribed remedy is over-fetch plus a literal post-filter. The repository already does this
  at `repository.py:694` but unescaped and on only one path.
- **`tsvector` is prohibited in application code** and the prohibition is enforced by a test, not by
  convention.
- **Required extensions are declared in the schema chain** rather than assumed present.

## Capabilities

**New Capabilities**

- `postgres-hybrid-retrieval` — one fused path, per-leg index reachability, the inverted BM25 sign
  convention, deterministic ordering, a uniform filter surface, chunk-relation tenant scoping,
  phrase search without `tsvector`, and declared extensions.

**Modified Capabilities**

- `legal-corpus-retrieval` — the requirement *"Ranked retrieval and fusion have a single
  implementation"* is tightened. It currently binds only agent tools, forbidding *them* from
  introducing a second ranking or fusion implementation. Measurement shows the second implementation
  arrived through the repository instead, which satisfies the letter of the requirement while
  defeating its purpose. The requirement is widened to bind every caller, and to require that two
  callers issuing the same query with the same filters receive the same ranked identifiers in the same
  order. Both existing scenarios are preserved.

This change also **cites** `hybrid-retrieval-ranking` and `document-retrieval-schema` without amending
them.

## Impact

- **Code:** `src/app/features/documents/repository.py` (the three branch methods; `legal_rrf_search`
  deleted), `fusion.py`, `constants.py`, `service.py`, `retrieval_kb/nodes.py`, and the executable stub
  at `src/app/examples/policy_examples.py:142`.
- **Schema:** index and extension DDL only. **No columns are added** — which is what makes a rebase
  against `ingestion-chunking`'s identity migration mechanical rather than a merge conflict.
- **Behaviour:** the retrieval graph and the search endpoint return the same results for the same
  query. Retrieval recall for a tenant owning a small fraction of the corpus improves.
- **Not touched:** the chunk-identity migration, the golden set, lifespan compilation, `shared/rag/**`,
  and the reranker — `agentic-retrieval` owns that.
