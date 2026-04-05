---
name: pg-textsearch-skill
description: Use when working on pg_textsearch in this repository, especially for extension installation, BM25 index creation, ranked text queries, score-based filtering, configuration tuning, or hybrid search with vectors. Trigger this skill for SQL changes, docs, examples, or guidance involving bm25 indexes, BM25 scoring operators, `to_bm25query()`, filtering tradeoffs, or pg_textsearch deployment constraints.
---

# PG Text Search

## Overview

Use this skill to write, review, or migrate pg_textsearch content in this repo. Keep `SKILL.md` as the routing layer and load the bundled reference for the complete migrated documentation.

Read [`references/pg_textsearch.md`](references/pg_textsearch.md) when you need the full source material, all SQL examples, configuration options, installation constraints, filtering guidance, or limitations. That file contains the complete migrated content from the former repo doc.

## Reference Routing

Read the reference file selectively:
- For install and prerequisites, read the opening installation sections.
- For BM25 index creation and ranked queries, read the indexing and query optimization sections.
- For hybrid search, read the reciprocal-rank-fusion examples.
- For explicit `bm25query` handling, filtering, crash recovery, and self-hosted caveats, read the later operational sections.
- For full option and limitation coverage, read the configuration and current limitations sections.

## Core Workflow

1. Confirm whether the target is Tiger Cloud or self-hosted Postgres.
2. Install and verify `pg_textsearch`.
3. Create a single-column BM25 index on the text field to search.
4. Use implicit `<@>` syntax for simple ranked queries in `ORDER BY`.
5. Switch to `to_bm25query()` when filtering in `WHERE`, naming the index explicitly, or working inside PL/pgSQL.
6. Add tuning, monitoring, or hybrid-search patterns only after the base query works.

## Install Pattern

Use:

```sql
CREATE EXTENSION pg_textsearch;
SELECT * FROM pg_extension WHERE extname = 'pg_textsearch';
```

For self-hosted deployments, remember that `shared_preload_libraries = 'pg_textsearch'` is required before restart. Do not suggest that requirement for Tiger Cloud.

## Index Pattern

Create BM25 indexes on a single text column:

```sql
CREATE INDEX products_search_idx ON products
USING bm25(description)
WITH (text_config = 'english');
```

Keep these rules in mind:
- BM25 indexes are single-column only.
- Load data first, then build the index when possible.
- Use language-appropriate `text_config`.
- Tune `k1` and `b` only when the task actually needs custom ranking behavior.

## Query Patterns

Use implicit syntax for simple ranked retrieval:

```sql
SELECT name, description, description <@> 'ergonomic work' AS score
FROM products
ORDER BY score
LIMIT 3;
```

Remember that BM25 scores are negative. Lower, more negative values mean better matches.

Use explicit `to_bm25query()` when filtering or when the context cannot infer the index:

```sql
SELECT name, description <@> to_bm25query('wireless', 'products_search_idx') AS score
FROM products
WHERE description <@> to_bm25query('wireless', 'products_search_idx') < -0.5
ORDER BY score
LIMIT 5;
```

Inside PL/pgSQL functions, DO blocks, or stored procedures, do not use the implicit `text <@> 'query'` form. Use `to_bm25query()` with the index name.

## Filtering Guidance

Explain the tradeoff clearly:
- Pre-filtering reduces rows before BM25 scoring, usually with another index such as B-tree.
- Post-filtering scores top-k first, then filters after ranking.

Use pre-filtering when the non-text filter is selective. Warn that post-filtering can return fewer rows than requested unless the query over-fetches and re-limits later.

## Performance And Operations

Reach for these knobs only when needed:
- `max_parallel_maintenance_workers`
- `maintenance_work_mem`
- `pg_textsearch.memtable_spill_threshold`
- `pg_textsearch.bulk_load_threshold`
- `pg_textsearch.default_limit`
- `pg_textsearch.enable_bmw`
- `pg_textsearch.compress_segments`
- `pg_textsearch.segments_per_level`

Use `bm25_force_merge()` after large batch loads when segment consolidation is useful. Use `bm25_summarize_index()` and `pg_stat_user_indexes` for inspection.

For `EXPLAIN`, call out that small datasets may still show sequential scans, while BM25 corpus statistics are still used internally for scoring.

## Hybrid Search Pattern

When the task combines semantic and keyword search:
- Create the vector index separately with pgvector or pgvectorscale.
- Create the BM25 index on the text column.
- Fuse the ranked result sets with reciprocal rank fusion.
- Add weighting only when the user has a clear relevance preference between vector and keyword results.

Reuse the hybrid SQL pattern from [`references/pg_textsearch.md`](references/pg_textsearch.md) instead of inventing a new scoring formula unless the task explicitly asks for it.

## Current Limits To Mention

- No phrase search. Emulate it with BM25 over-fetch plus a text post-filter.
- No expression index support. Use a generated column when multi-field search text is needed.
- No compressed data support.
- Partition-local statistics mean scores are not directly comparable across partitions.
- Write-heavy workloads and compaction behavior still need careful handling.

## What To Reuse

When answering a pg_textsearch task in this repo:
- Reuse exact SQL forms and operational caveats from [`references/pg_textsearch.md`](references/pg_textsearch.md).
- Keep examples centered on `bm25`, `<@>`, and `to_bm25query()`.
- Prefer compact executable SQL over long narrative explanation.
