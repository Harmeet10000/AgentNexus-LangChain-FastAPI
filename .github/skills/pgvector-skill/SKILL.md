---
name: pgvector-skill
description: Use when working on pgvector or pgvectorscale in this repository, especially for extension setup, embedding-table design, similarity queries, vector distance/operator choices, or ANN index selection and tuning. Trigger this skill for SQL changes, docs, examples, or guidance involving vector columns, cosine/L2/inner-product search, StreamingDiskANN, HNSW, or ivfflat.
---

# PGVector

## Overview

Use this skill to write, review, or migrate pgvector and pgvectorscale content in this repo. Keep `SKILL.md` as the routing layer and load the bundled reference for the full source material.

Read [`references/pgvector.md`](references/pgvector.md) when you need exact SQL, full parameter tables, installation notes, or detailed index guidance. That file contains the complete migrated content from the former repo doc.

## Reference Routing

Read the reference file selectively:
- For extension setup and base table design, read the opening sections of [`references/pgvector.md`](references/pgvector.md).
- For query operators and canonical similarity queries, read the query section.
- For index selection and tuning, read the StreamingDiskANN, HNSW, and ivfflat sections.
- For build-from-source or installation details beyond the standard SQL workflow, read the later installation sections.

## Core Workflow

1. Confirm that the target database should use `vector` only or both `vector` and `vectorscale`.
2. Enable the required extensions first:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS vectorscale;
   ```
3. Model embeddings in a table that keeps the raw text and embedding in the same row.
4. Pick the distance metric before designing indexes or queries.
5. Create the ANN index whose operator class matches the query operator.
6. Tune query-time settings only after the basic query shape is correct.

## Table Pattern

Default to a table that keeps:
- A primary key
- The source row identifier if embeddings belong to a parent document
- Optional metadata
- The embedded text chunk
- The embedding vector

Use a fixed-dimension `VECTOR(n)` column when the embedding model is known and stable. Use plain `VECTOR` only when variable dimensions are intentional.

Keep the stored text and embedding in the same row so similarity search can return the matched content directly.

## Query Pattern

Use the canonical query shape:

```sql
SELECT *
FROM document_embedding
ORDER BY embedding <=> $1
LIMIT 10;
```

Choose the operator from the intended metric:
- Cosine: `<=>`
- Euclidean/L2: `<->`
- Negative inner product: `<#>`

If an index exists, the query operator must match the operator class used at index creation. If they do not match, the index cannot accelerate the query.

## Index Selection

Use this decision order:
- Prefer `diskann` from pgvectorscale for most production workloads. It is the recommended default in the source doc.
- Use `hnsw` when you want pgvector-native graph indexing without pgvectorscale-specific features.
- Use `ivfflat` only when fast index build time matters more than query speed, and the dataset is already populated.

## Index Recipes

Use StreamingDiskANN for cosine search:

```sql
CREATE INDEX document_embedding_cos_idx ON document_embedding
USING diskann (embedding vector_cosine_ops);
```

Use StreamingDiskANN for L2 search:

```sql
CREATE INDEX document_embedding_l2_idx ON document_embedding
USING diskann (embedding vector_l2_ops);
```

Use HNSW for cosine search:

```sql
CREATE INDEX document_embedding_idx ON document_embedding
USING hnsw (embedding vector_cosine_ops);
```

Use ivfflat only after loading data:

```sql
CREATE INDEX document_embedding_idx ON document_embedding
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

Do not create ivfflat on an empty table. Rebuild it periodically if the table changes heavily.

## Tuning Rules

Use these defaults unless the task explicitly asks for tuning:
- StreamingDiskANN build-time defaults are usually good enough.
- Tune `diskann.query_rescore` before deeper algorithm changes.
- Tune `hnsw.ef_search` to trade latency for recall.
- Set `ivfflat.probes` near `sqrt(lists)` as the starting point.

Use `SET LOCAL ...` inside a transaction when showing one-off tuning examples so the setting does not leak to the whole session.

## Gotchas

- Do not add an ANN index and then query with a different distance operator.
- Do not present indexed search as exact search. Approximate indexes trade speed for recall.
- Do not recommend ivfflat for empty tables or write-heavy tables without mentioning rebuild cost.
- Call out that `VECTOR(n)` enforces dimensions while bare `VECTOR` does not.

## What To Reuse

When answering a pgvector task in this repo:
- Reuse examples and exact parameter tables from [`references/pgvector.md`](references/pgvector.md).
- Keep new examples aligned with the table/query/index shapes already documented there.
- Prefer concise SQL snippets over prose when the user needs executable guidance.
