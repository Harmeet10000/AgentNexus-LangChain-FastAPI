# Baseline — retrieval-sql measurements (2026-09-14)

## Gate numbers (post-hoc: task 1.1 was checked without recording them, so these are the
first recorded gate counts, not a pre-change baseline)

- `uv run pytest tests/unit/documents tests/unit/shared/langgraph_layer -q`:
  `172 passed` (7.1 re-run; includes this change's guard, parity, phrase, and preparability tests)
- Full `uv run pytest -q` at 7.1: `697 passed, 43 deselected`, **0 failures**
- `uv run ruff check --no-cache src/`: `All checks passed!`
- `uv run ty check src/`: `Found 4 diagnostics` (all `invalid-argument-type` on `configure`
  in `src/alembic/env.py`, another agent's active edit — none in this change's files)

## 5.1 tenant recall (tenant-1pct, 496/59,096 chunks, exact-NN ground truth via numpy)

| Shape | recall@50 | Plan | Time |
|---|---|---|---|
| old JOIN (`d.user_id`) | 50/50 = 1.00 | `ix_documents_user_id` + nested loop + exact sort | 4.13ms |
| new chunk predicate (`c.user_id`) | 50/50 = 1.00 | `ix_chunks_user_document` + Sort | 3.92ms |

No recall gap exists (both evaluate exactly; a 496-row tenant never reaches the approximate
scan). The move's value is structural. Same filter block used for both shapes to isolate the
JOIN variable; the old shape additionally required the new CAST-typed filters because its own
HEAD filter text cannot prepare under asyncpg.

## 1.3 EXPLAIN capture

## Corpus

`scratch_13` schema on the live instance: **59,096 chunks / 24 users** (a superset of the
required ~50k/20 — the extra rows are later probe tenants; harmless for plan shape). All three
retrieval indexes present: `chunks_bm25_idx`, `chunks_embedding_idx` (diskann),
`chunks_search_text_trgm_idx`. Full plans: `docs/relay/explain-13/*.md`.

Probe query: `'termination obligations compensation'`, user `scratch-user-00` (2,500 chunks),
`candidate_limit` 50, `limit` 20, weights 0.4/0.6, query embedding taken from one of the
user's own stored vectors.

## Verdicts — whether each index appears AS AN ACCESS PATH

A name merely mentioned (e.g. as a function argument) does not count; only a scan node counts.

| Statement | bm25 idx | embedding idx | trigram idx |
|---|---|---|---|
| `legal_rrf_search` (HEAD SQL) | absent | absent | absent |
| `bm25_search` (current) | **Index Scan** (65 rows, 22ms) | — | — |
| `vector_search` (current) | — | **Index Scan** (51 rows, ~100ms) | — |
| `trigram_search` (current) | — | — | **Bitmap Index Scan** (0 rows, 6.1s) |

The task's expectation is confirmed: absent for `legal_rrf_search`, present for all three
branch methods — and structurally so, not an artifact of selectivity:

- `legal_rrf_search` materializes `candidate_chunks` once; all three legs are
  `CTE Scan on candidate_chunks`. Legs read CTE output, so no base-table retrieval index
  *can* be used. `'chunks_bm25_idx'` appears only as the `to_bm25query()` tokenizer-statistics
  argument. The monolith pays full CTE materialization, then ranks in Python-shaped SQL.
- `vector_search` uses the diskann index with 596 rows removed by the tenant filter *after*
  the scan — input to 5.1.
- `trigram_search` uses the trigram index but takes 6.1s rechecking 44.5k heap rows to
  return nothing for this query — input to 4.x tuning.

## Measurement substitutions (recorded, not hidden)

- `legal_rrf_search` was deleted by 3.3, so the HEAD-version SQL was measured. Its
  `:x IS NULL` + typed-use dual-context parameters **cannot be prepared by either driver**
  (asyncpg *and* psycopg raise `DatatypeMismatch`) for None *or* concrete bindings, so
  bindings were inlined as literals into a copy (`jurisdiction='India'`,
  `contract_type='services'`, `clause_type='termination'`, one real document UUID, one real
  chunk UUID, `bm25_threshold=-1.0`, `exact_phrase_like='%termination%'`). Plan shape is
  identical to a successfully-bound execution. Corollary finding: the old path appears
  **unexecutable with defaults**, not just unindexed — supporting the collapse.
- `bm25_search` was measured with production-default bindings (threshold/phrase None).
  pg_textsearch hard-requires the planner to use the named index; during measurement the
  planner intermittently chose otherwise and the statement errored (`WrongObjectTypeError`).
  Both outcomes are on record (`bm25_search__default_cost.md` = successful Index Scan;
  the errors were transient across runs). The index *can* serve the full filtered query.
- Tables were schema-qualified (`scratch_13.*`) because `SET search_path` does not survive
  the pool/rollback boundary reliably; identical plans either way.
