---
title: Optimize full text search with BM25
excerpt: Set up and optimize BM25-based full-text search using the pg_textsearch extension
keywords: [pg_textsearch, BM25, full-text search, text search, ranking, hybrid search]
tags: [search, indexing, performance, BM25]
products: [cloud, self_hosted]
---

import SINCE010 from "versionContent/_partials/_since_0_1_0.mdx";
import SINCE040 from "versionContent/_partials/_since_0_4_0.mdx";
import SINCE050 from "versionContent/_partials/_since_0_5_0.mdx";
import SINCE100 from "versionContent/_partials/_since_1_0_0.mdx";
import IntegrationPrereqs from "versionContent/_partials/_integration-prereqs.mdx";

# Optimize full text search with BM25 

Postgres full-text search at scale consistently hits a wall where performance degrades catastrophically.
Tiger Data's [pg_textsearch][pg_textsearch-github-repo] brings modern [BM25][bm25-wiki]-based full-text search directly into Postgres,
with a memtable architecture for efficient indexing and ranking. `pg_textsearch` integrates seamlessly with SQL and
provides better search quality and performance than the Postgres built-in full-text search. With Block-Max WAND optimization,
`pg_textsearch` delivers up to **4x faster top-k queries** compared to native BM25 implementations. Parallel index builds
reduce indexing times by **4x or more** for large tables. Advanced compression using delta encoding and bitpacking reduces
index sizes by **41%** while improving query performance by 10-20% for shorter queries.

BM25 scores in `pg_textsearch` are returned as negative values, where lower (more negative) numbers indicate better 
matches. `pg_textsearch` implements the following:

* **Corpus-aware ranking**: BM25 uses inverse document frequency to weight rare terms higher
* **Term frequency saturation**: prevents documents with excessive term repetition from dominating results
* **Length normalization**: adjusts scores based on document length relative to corpus average
* **Relative ranking**: focuses on rank order rather than absolute score values

This page shows you how to install `pg_textsearch`, configure BM25 indexes, and optimize your search capabilities using
the following best practices:

* **Parallel indexing**: enable parallel workers for faster index creation on large tables
* **Language configuration**: choose appropriate text search configurations for your data language
* **Hybrid search**: combine with pgvector or pgvectorscale for applications requiring both semantic and keyword search
* **Query optimization**: use score thresholds to filter low-relevance results
* **Index monitoring**: regularly check index usage and memory consumption

`pg_textsearch` v1.0.0 is production ready (March 2026). It supports Postgres 17 and 18.

<Tag variant="hollow">Since [pg_textsearch v1.0.0](https://github.com/timescale/pg_textsearch/releases/tag/v1.0.0)</Tag>

## Prerequisites

To follow the steps on this page:

* Create a target [Tiger Cloud service][create-service] with the Real-time analytics capability.

   You need [your connection details][connection-info]. This procedure also 
   works for [self-hosted TimescaleDB][enable-timescaledb].

## Install pg_textsearch 

To install this Postgres extension: 

<Procedure>

1. **Connect to your Tiger Cloud service**

   In [Tiger Console][services-portal] open an [SQL editor][in-console-editors]. You can also connect to your service using [psql][connect-using-psql].

1. **Enable the extension on your Tiger Cloud service**

   - For new services, simply enable the extension:
      ```sql
      CREATE EXTENSION pg_textsearch;
      ```
   
   - For existing services, update your instance, then enable the extension:

      The extension may not be available until after your next scheduled maintenance window. To pick up the update 
      immediately, manually pause and restart your service.

1. **Verify the installation**

   ```sql
   SELECT * FROM pg_extension WHERE extname = 'pg_textsearch';
   ```

</Procedure>

You have installed `pg_textsearch` on Tiger Cloud.

## Create BM25 indexes on your data

BM25 indexes provide modern relevance ranking that outperforms Postgres's built-in ts_rank functions by using corpus
statistics and better algorithmic design.

To create a BM25 index with `pg_textsearch`:

<Procedure>

1. **Create a table with text content**

   ```sql
   CREATE TABLE products (
       id serial PRIMARY KEY,
       name text,
       description text,
       category text,
       price numeric
   );
   ```

1. **Insert sample data**

   ```sql
   INSERT INTO products (name, description, category, price) VALUES
   ('Mechanical Keyboard', 'Durable mechanical switches with RGB backlighting for gaming and productivity', 'Electronics', 149.99),
   ('Ergonomic Mouse', 'Wireless mouse with ergonomic design to reduce wrist strain during long work sessions', 'Electronics', 79.99),
   ('Standing Desk', 'Adjustable height desk for better posture and productivity throughout the workday', 'Furniture', 599.99);
   ```

1. **Create a BM25 index**

   ```sql
   CREATE INDEX products_search_idx ON products
   USING bm25(description)
   WITH (text_config='english');
   ```

   BM25 supports single-column indexes only. For optimal performance, load your data first, then create the index.

</Procedure>

You have created a BM25 index for full-text search.

## Enable parallel indexing for faster index creation

`pg_textsearch` supports parallel index builds that can significantly reduce indexing times for large tables.
Postgres automatically uses parallel workers based on table size and available resources.

<Procedure>

1. **Configure parallel workers (optional)**

   Postgres uses server defaults, but you can adjust settings for your workload:

   ```sql
   -- Set number of parallel workers (uses CPU count by default)
   SET max_parallel_maintenance_workers = 4;

   -- Set memory for index builds (must be at least 64MB for parallel builds)
   SET maintenance_work_mem = '256MB';
   ```

   **Note**: The planner requires `maintenance_work_mem >= 64MB` to enable parallel index builds. With insufficient
   memory, builds fall back to serial mode silently.

1. **Create index (parallel workers used automatically for large tables)**

   ```sql
   CREATE INDEX products_search_idx ON products
   USING bm25(description)
   WITH (text_config='english');
   ```

   When parallel build is used, you see a notice:

   ```
   NOTICE:  parallel index build: launched 4 of 4 requested workers
   ```

1. **Verify parallel execution in partitioned tables**

   For partitioned tables, each partition builds its index independently with parallel workers if the partition is
   large enough. This allows efficient indexing of very large partitioned datasets.

</Procedure>

<Tag variant="hollow">Since [pg_textsearch v0.5.0](https://github.com/timescale/pg_textsearch/releases/tag/v0.5.0)</Tag>

You have configured parallel index builds for faster indexing.

## Optimize search queries for performance

Use efficient query patterns to leverage BM25 ranking and optimize search performance. The `<@>` operator provides
BM25-based ranking scores as negative values, where lower (more negative) scores indicate better matches. In `ORDER BY`
clauses, the index is automatically detected from the column. For `WHERE` clause filtering, use `to_bm25query()` with
an explicit index name.

<Procedure>

1. **Perform ranked searches using implicit syntax**

   The simplest way to query is with the implicit `<@>` syntax. The BM25 index is automatically detected from the column:

   ```sql
   SELECT name, description, description <@> 'ergonomic work' as score
   FROM products
   ORDER BY score
   LIMIT 3;
   ```

   You see something like:

   ```sql
                name           |                                    description                                    |        score
   ----------------------------+-----------------------------------------------------------------------------------+---------------------
    Ergonomic Mouse            | Wireless mouse with ergonomic design to reduce wrist strain during long work sessions | -1.8132977485656738
    Mechanical Keyboard        | Durable mechanical switches with RGB backlighting for gaming and productivity      |                   0
    Standing Desk              | Adjustable height desk for better posture and productivity throughout the workday  |                   0
   ```

1. **Use explicit index specification with `to_bm25query()`**

   For `WHERE` clause filtering or when you need to specify the index explicitly, use `to_bm25query()`:

   ```sql
   SELECT name, description <@> to_bm25query('ergonomic work', 'products_search_idx') as score
   FROM products
   ORDER BY score
   LIMIT 3;
   ```

   The implicit `text <@> 'query'` syntax does not work inside PL/pgSQL functions or DO blocks. Use
   `to_bm25query()` with an explicit index name in those contexts. See [bm25query data type](#bm25query-data-type)
   for details.

1. **Filter results by score threshold**

   For filtering with WHERE clauses, use explicit index specification with `to_bm25query()`:

   ```sql
   SELECT name, description <@> to_bm25query('wireless', 'products_search_idx') as score
   FROM products
   WHERE description <@> to_bm25query('wireless', 'products_search_idx') < -0.5;
   ```

   You see something like:

   ```sql
        name       |        score
   ----------------+---------------------
    Ergonomic Mouse | -0.9066488742828369
   ```

1. **Combine with standard SQL operations**

   ```sql
   SELECT category, name, description <@> to_bm25query('ergonomic', 'products_search_idx') as score
   FROM products
   WHERE price < 500
     AND description <@> to_bm25query('ergonomic', 'products_search_idx') < -0.5
   ORDER BY score
   LIMIT 5;
   ```

   You see something like:

   ```sql
     category   |      name       |        score
   -------------+-----------------+---------------------
    Electronics | Ergonomic Mouse | -0.9066488742828369
   ```

1. **Verify index usage with EXPLAIN**

   ```sql
   EXPLAIN SELECT * FROM products
   ORDER BY description <@> 'ergonomic'
   LIMIT 5;
   ```

   You see something like:

   ```sql
                                              QUERY PLAN
   --------------------------------------------------------------------------------------------
    Limit  (cost=8.55..8.56 rows=3 width=140)
      ->  Sort  (cost=8.55..8.56 rows=3 width=140)
            Sort Key: ((description <@> 'products_search_idx:ergonomic'::bm25query))
            ->  Seq Scan on products  (cost=0.00..8.53 rows=3 width=140)
   ```

   For small datasets, Postgres may prefer sequential scans over index scans. To force index usage during testing:

   ```sql
   SET enable_seqscan = off;
   ```

   Even when `EXPLAIN` shows a sequential scan, the `<@>` operator always uses the BM25 index internally for
   corpus statistics (document counts, average document length) required for accurate BM25 scoring.

</Procedure>

You have optimized your search queries for BM25 ranking.

## Build hybrid search with semantic and keyword search

Combine `pg_textsearch` with `pgvector` or `pgvectorscale` to build powerful hybrid search systems that use both semantic vector search and keyword BM25 search.

<Procedure>

1. **Enable the [vectorscale][pg-vectorscale] extension on your Tiger Cloud service**
   ```sql
    CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
    ```
1. **Create a table with both text content and vector embeddings**

   ```sql
   CREATE TABLE articles (
       id serial PRIMARY KEY,
       title text,
       content text,
       embedding vector(3)  -- Using 3 dimensions for this example; use 1536 for OpenAI ada-002
   );
   ```

1. **Insert sample data**

   ```sql
   INSERT INTO articles (title, content, embedding) VALUES
   ('Database Query Optimization', 'Learn how to optimize database query performance using indexes and query planning', '[0.1, 0.15, 0.2]'),
   ('Performance Tuning Guide', 'A comprehensive guide to performance tuning in distributed systems and databases', '[0.12, 0.18, 0.25]'),
   ('Introduction to Indexing', 'Understanding how database indexes improve query performance and data retrieval', '[0.09, 0.14, 0.19]'),
   ('Advanced SQL Techniques', 'Master advanced SQL techniques for complex data analysis and reporting', '[0.5, 0.6, 0.7]'),
   ('Data Warehousing Basics', 'Getting started with data warehousing and analytical query processing', '[0.8, 0.9, 0.85]');
   ```

1. **Create indexes for both search types**

   ```sql
   -- Vector index for semantic search
   CREATE INDEX articles_embedding_idx ON articles
   USING hnsw (embedding vector_cosine_ops);

   -- Keyword index for BM25 search
   CREATE INDEX articles_content_idx ON articles
   USING bm25(content)
   WITH (text_config='english');
   ```

1. **Perform hybrid search using [reciprocal rank fusion][recip-rank-fusion]**

   ```sql
   WITH vector_search AS (
     SELECT id,
            ROW_NUMBER() OVER (ORDER BY embedding <=> '[0.1, 0.2, 0.3]'::vector) AS rank
     FROM articles
     ORDER BY embedding <=> '[0.1, 0.2, 0.3]'::vector
     LIMIT 20
   ),
   keyword_search AS (
     SELECT id,
            ROW_NUMBER() OVER (ORDER BY content <@> to_bm25query('query performance', 'articles_content_idx')) AS rank
     FROM articles
     ORDER BY content <@> to_bm25query('query performance', 'articles_content_idx')
     LIMIT 20
   )
   SELECT a.id,
          a.title,
          COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS combined_score
   FROM articles a
   LEFT JOIN vector_search v ON a.id = v.id
   LEFT JOIN keyword_search k ON a.id = k.id
   WHERE v.id IS NOT NULL OR k.id IS NOT NULL
   ORDER BY combined_score DESC
   LIMIT 10;
   ```

   You see something like:

   ```sql
    id |           title            |   combined_score
   ----+----------------------------+--------------------
     3 | Introduction to Indexing   | 0.0325224748810153
     1 | Database Query Optimization| 0.0322664584959667
     2 | Performance Tuning Guide   | 0.0320020481310804
     5 | Data Warehousing Basics    | 0.0310096153846154
     4 | Advanced SQL Techniques    | 0.0310096153846154
   ```

1. **Adjust relative weights for different search types**

   ```sql
     WITH vector_search AS (
     SELECT id,
            ROW_NUMBER() OVER (ORDER BY embedding <=> '[0.1, 0.2, 0.3]'::vector) AS rank
     FROM articles
     ORDER BY embedding <=> '[0.1, 0.2, 0.3]'::vector
     LIMIT 20
   ),
   keyword_search AS (
     SELECT id,
            ROW_NUMBER() OVER (ORDER BY content <@> to_bm25query('query performance', 'articles_content_idx')) AS rank
     FROM articles
     ORDER BY content <@> to_bm25query('query performance', 'articles_content_idx')
     LIMIT 20
   )
   SELECT
       a.id,
       a.title,
       0.7 * COALESCE(1.0 / (60 + v.rank), 0.0) +  -- 70% weight to vectors
       0.3 * COALESCE(1.0 / (60 + k.rank), 0.0)    -- 30% weight to keywords
   AS combined_score
   FROM articles a
   LEFT JOIN vector_search v ON a.id = v.id
   LEFT JOIN keyword_search k ON a.id = k.id
   WHERE v.id IS NOT NULL OR k.id IS NOT NULL
   ORDER BY combined_score DESC
   LIMIT 10;
   ```

   You see something like:

   ```sql
    id |           title            |   combined_score
   ----+----------------------------+--------------------
     3 | Introduction to Indexing   | 0.0163141195134849
     2 | Performance Tuning Guide   | 0.0160522273425499
     1 | Database Query Optimization| 0.0160291438979964
     4 | Advanced SQL Techniques    | 0.0155528846153846
     5 | Data Warehousing Basics    | 0.0154567307692308
   ```

</Procedure>

You have implemented hybrid search combining semantic and keyword search.

## bm25query data type

The `bm25query` type represents queries for BM25 scoring with optional index context. You need this type when using
`to_bm25query()` for explicit index specification, `WHERE` clause filtering, or PL/pgSQL compatibility.

### Constructor functions

| Function | Description |
|---|---|
| `to_bm25query(text)` | Create a bm25query without index name (for `ORDER BY` only) |
| `to_bm25query(text, text)` | Create a bm25query with query text and index name |

```sql
-- Create a bm25query with index name (required for WHERE clause and standalone scoring)
SELECT to_bm25query('search query text', 'products_search_idx');
-- Returns: products_search_idx:search query text

-- Create a bm25query without index name (only works in ORDER BY with index scan)
SELECT to_bm25query('search query text');
-- Returns: search query text
```

### Cast syntax

You can also create a `bm25query` using cast syntax with an embedded index name:

```sql
SELECT 'products_search_idx:search query text'::bm25query;
-- Returns: products_search_idx:search query text
```

### Operators

| Operator | Description |
|---|---|
| `text <@> bm25query` | BM25 scoring operator (returns negative scores; lower is better) |
| `bm25query = bm25query` | Equality comparison |

<Tag variant="hollow">Since [pg_textsearch v1.0.0](https://github.com/timescale/pg_textsearch/releases/tag/v1.0.0)</Tag>

## Configuration options

Customize `pg_textsearch` behavior for your specific use case and data characteristics.

<Procedure>

1. **Configure memory and performance settings**

   To manage memory usage, you control when the in-memory index spills to disk segments. When the memtable reaches the
   threshold, it automatically flushes to a segment at transaction commit.

   ```sql
   -- Set memtable spill threshold (default 32000000 posting entries, ~1M docs/segment)
   SET pg_textsearch.memtable_spill_threshold = 32000000;

   -- Set bulk load spill threshold (default 100000 terms per transaction)
   SET pg_textsearch.bulk_load_threshold = 150000;

   -- Set default query limit when no LIMIT clause is present (default 1000)
   SET pg_textsearch.default_limit = 5000;

   -- Enable Block-Max WAND optimization for faster top-k queries (enabled by default)
   SET pg_textsearch.enable_bmw = true;

   -- Log block skip statistics for debugging query performance (disabled by default)
   SET pg_textsearch.log_bmw_stats = false;
   ```
   <Tag variant="hollow">Since [pg_textsearch v0.1.0](https://github.com/timescale/pg_textsearch/releases/tag/v0.1.0)</Tag>

   ```sql
   -- Enable segment compression using delta encoding and bitpacking (enabled by default)
   -- Reduces index size by ~41% with 10-20% query performance improvement for shorter queries
   SET pg_textsearch.compress_segments = on;
   ```
   <Tag variant="hollow">Since [pg_textsearch v0.4.0](https://github.com/timescale/pg_textsearch/releases/tag/v0.4.0)</Tag>

   ```sql
   -- Control segments per level before automatic compaction (default 8, range 2-64)
   SET pg_textsearch.segments_per_level = 8;

   -- Log BM25 scores during scans for debugging (disabled by default)
   SET pg_textsearch.log_scores = false;
   ```
   <Tag variant="hollow">Since [pg_textsearch v1.0.0](https://github.com/timescale/pg_textsearch/releases/tag/v1.0.0)</Tag>

1. **Configure language-specific text processing**

   You can create multiple BM25 indexes on the same column with different language configurations:

   ```sql
   -- Create an additional index with simple tokenization (no stemming)
   CREATE INDEX products_simple_idx ON products
   USING bm25(description)
   WITH (text_config='simple');

   -- Example: French language configuration for a French products table
   -- CREATE INDEX products_fr_idx ON products_fr
   -- USING bm25(description)
   -- WITH (text_config='french');
   ```

1. **Tune BM25 parameters**

   ```sql
   -- Adjust term frequency saturation (k1) and length normalization (b)
   CREATE INDEX products_custom_idx ON products
   USING bm25(description)
   WITH (text_config='english', k1=1.5, b=0.8);
   ```

1. **Optimize query performance with force merge**

   After bulk loads or sustained incremental inserts, multiple index segments may accumulate. Consolidating
   them into a single segment improves query speed by reducing the number of segments scanned. This is
   analogous to Lucene's `forceMerge(1)`:

   ```sql
   SELECT bm25_force_merge('products_search_idx');
   ```

   Best used after large batch inserts, not during ongoing write traffic. The operation rewrites all segments
   into a single segment and reclaims freed pages.

   <Tag variant="hollow">Since [pg_textsearch v1.0.0](https://github.com/timescale/pg_textsearch/releases/tag/v1.0.0)</Tag>

1. **Monitor index usage and memory consumption**

      - Check index usage statistics
          ```sql
          SELECT schemaname, relname, indexrelname, idx_scan, idx_tup_read
          FROM pg_stat_user_indexes
          WHERE indexrelid::regclass::text ~ 'bm25';
          ```

      - View index summary with corpus statistics and memory usage (requires superuser)
          ```sql
          SELECT bm25_summarize_index('products_search_idx');
          ```

      - View detailed index structure (requires superuser, output is truncated for display)
          ```sql
          SELECT bm25_dump_index('products_search_idx');
          ```

          The two-argument form `bm25_dump_index('idx', '/tmp/dump.txt')` that writes output to a file is
          only available in debug builds (compiled with `-DDEBUG_DUMP_INDEX`). It is not available in
          production builds on Tiger Cloud.

      - Force memtable spill to disk (useful for testing or memory management)
          ```sql
          SELECT bm25_spill_index('products_search_idx');
          ```

</Procedure>

You have configured `pg_textsearch` for optimal performance. For production applications, consider implementing result
caching and pagination to improve user experience with large result sets.

## Filtering guidance

There are two ways filtering interacts with BM25 index scans:

**Pre-filtering** uses a separate index (B-tree, etc.) to reduce rows before scoring:

```sql
-- Create index on filter column
CREATE INDEX ON products (category);

-- Query filters first, then scores matching rows
SELECT * FROM products
WHERE category = 'Electronics'
ORDER BY description <@> 'ergonomic wireless'
LIMIT 10;
```

**Post-filtering** applies the BM25 index scan first, then filters results:

```sql
SELECT * FROM products
WHERE description <@> to_bm25query('ergonomic', 'products_search_idx') < -0.5
ORDER BY description <@> 'ergonomic'
LIMIT 10;
```

**Performance considerations**:

* **Pre-filtering tradeoff**: if the filter matches many rows (for example, 100K+), scoring all of them can be expensive.
  The BM25 index is most efficient when it can use top-k optimization (`ORDER BY` + `LIMIT`) to avoid scoring every
  matching document.
* **Post-filtering tradeoff**: the index returns top-k results *before* filtering. If your `WHERE` clause eliminates
  most results, you may get fewer rows than requested. Increase `LIMIT` to compensate, then re-limit in application code.
* **Best case**: pre-filter with a selective condition (matches <10% of rows), then let BM25 score the reduced set with
  `ORDER BY` + `LIMIT`.

## Crash recovery

The memtable is rebuilt from the heap on startup, so no data is lost if Postgres crashes before spilling to disk.

## Self-hosted installation

For self-hosted installations, `pg_textsearch` must be loaded via `shared_preload_libraries`. Add the following to
`postgresql.conf` and restart the server:

```
shared_preload_libraries = 'pg_textsearch'  # add to existing list if needed
```

This is not required on Tiger Cloud, where the extension is pre-configured.

## Current limitations

Current limitations include:

* **No phrase search**: you cannot search for exact multi-word phrases. You can emulate phrase matching by combining
  BM25 ranking with a post-filter:
  ```sql
  SELECT * FROM (
      SELECT *, content <@> 'database system' AS score
      FROM documents
      ORDER BY score
      LIMIT 100  -- over-fetch to account for post-filter
  ) sub
  WHERE content ILIKE '%database system%'
  ORDER BY score
  LIMIT 10;
  ```
* **No compressed data support**: `pg_textsearch` does not work with compressed data.
* **No expression indexing**: each BM25 index covers a single text column. You cannot create an index on an expression
  like `lower(title) || ' ' || content`. As a workaround, use a generated column:
  ```sql
  ALTER TABLE documents ADD COLUMN search_text text
      GENERATED ALWAYS AS (
          COALESCE(title, '') || ' ' || COALESCE(content, '')
      ) STORED;
  CREATE INDEX ON documents USING bm25(search_text) WITH (text_config = 'english');
  ```
* **No built-in faceted search**: `pg_textsearch` does not provide dedicated faceting operators. Use standard Postgres
  `GROUP BY` for facet counts:
  ```sql
  SELECT category, count(*)
  FROM products
  WHERE description <@> to_bm25query('ergonomic', 'products_search_idx') < -1.0
  GROUP BY category;
  ```
* **Insert/update performance**: sustained write-heavy workloads are not yet fully optimized. For initial data loading,
  create the index after loading data rather than using incremental inserts.
* **No background compaction**: segment compaction runs synchronously during memtable spill operations. Write-heavy
  workloads may observe compaction latency during spills.
* **Partitioned table statistics**: BM25 indexes on partitioned tables use partition-local statistics. Each partition
  maintains its own document count, average document length, and per-term document frequencies. Scores are not directly
  comparable across partitions.
* **Word length limit**: inherits Postgres's tsvector word length limit of 2047 characters. Words exceeding this limit are
  ignored during tokenization.
* **PL/pgSQL limitation**: the implicit `text <@> 'query'` syntax does not work inside PL/pgSQL DO blocks, functions,
  or stored procedures. Use `to_bm25query()` with an explicit index name instead:
  ```sql
  -- Inside PL/pgSQL, use explicit index name:
  SELECT * FROM documents
  ORDER BY content <@> to_bm25query('search terms', 'docs_idx')
  LIMIT 10;
  ```

[connection-info]: /integrations/latest/find-connection-details/
[create-service]: /getting-started/latest/services/
[enable-timescaledb]: /self-hosted/latest/install/

[bm25-wiki]: https://en.wikipedia.org/wiki/Okapi_BM25
[connect-using-psql]: /integrations/latest/psql/#connect-to-your-service
[in-console-editors]: /getting-started/latest/run-queries-from-console/
[pg-vectorscale]: /ai/latest/sql-interface-for-pgvector-and-timescale-vector/#installing-the-pgvector-and-pgvectorscale-extensions
[pg_textsearch-github-repo]: https://github.com/timescale/pg_textsearch
[recip-rank-fusion]: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
[services-portal]: https://console.cloud.timescale.com/dashboard/services

Modern ranked text search for Postgres.

Simple syntax: ORDER BY content <@> 'search terms'
BM25 ranking with configurable parameters (k1, b)
Works with Postgres text search configurations (english, french, german, etc.)
Fast top-k queries via Block-Max WAND optimization
Parallel index builds for large tables
Supports partitioned tables
Best in class performance and scalability
🚀 Status: v1.0.0 - Production ready. See ROADMAP.md for upcoming features.

Tapir and Friends

Historical note
The original name of the project was Tapir - Textual Analysis for Postgres Information Retrieval. We still use the tapir as our mascot and the name occurs in various places in the source code.

PostgreSQL Version Compatibility
pg_textsearch supports PostgreSQL 17 and 18.

Installation
Pre-built Binaries
Download pre-built binaries from the Releases page. Available for Linux and macOS (amd64 and arm64), PostgreSQL 17 and 18.

Build from Source
cd /tmp
git clone https://github.com/timescale/pg_textsearch
cd pg_textsearch
make
make install # may need sudo
Getting Started
pg_textsearch must be loaded via shared_preload_libraries. Add the following to postgresql.conf and restart the server:

shared_preload_libraries = 'pg_textsearch'  # add to existing list if needed
Then enable the extension (once per database):

CREATE EXTENSION pg_textsearch;
Create a table with text content

CREATE TABLE documents (id bigserial PRIMARY KEY, content text);
INSERT INTO documents (content) VALUES
    ('PostgreSQL is a powerful database system'),
    ('BM25 is an effective ranking function'),
    ('Full text search with custom scoring');
Create a pg_textsearch index on the text column

CREATE INDEX docs_idx ON documents USING bm25(content) WITH (text_config='english');
Querying
Get the most relevant documents using the <@> operator

SELECT * FROM documents
ORDER BY content <@> 'database system'
LIMIT 5;
Note: <@> returns the negative BM25 score since Postgres only supports ASC order index scans on operators. Lower scores indicate better matches.

The index is automatically detected from the column. For explicit index specification:

SELECT * FROM documents
WHERE content <@> to_bm25query('database system', 'docs_idx') < -1.0;
Supported operations:

text <@> 'query' - Score text against a query (index auto-detected)
text <@> bm25query - Score text with explicit index specification
Verifying Index Usage
Check query plan with EXPLAIN:

EXPLAIN SELECT * FROM documents
ORDER BY content <@> 'database system'
LIMIT 5;
For small datasets, PostgreSQL may prefer sequential scans. Force index usage:

SET enable_seqscan = off;
Note: Even if EXPLAIN shows a sequential scan, <@> and to_bm25query always use the index for corpus statistics (document counts, average length) required for BM25 scoring.

Filtering with WHERE Clauses
There are two ways filtering interacts with BM25 index scans:

Pre-filtering uses a separate index (B-tree, etc.) to reduce rows before scoring:

-- Create index on filter column
CREATE INDEX ON documents (category_id);

-- Query filters first, then scores matching rows
SELECT * FROM documents
WHERE category_id = 123
ORDER BY content <@> 'search terms'
LIMIT 10;
Post-filtering applies the BM25 index scan first, then filters results:

SELECT * FROM documents
WHERE content <@> to_bm25query('search terms', 'docs_idx') < -5.0
ORDER BY content <@> 'search terms'
LIMIT 10;
Performance considerations:

Pre-filtering tradeoff: If the filter matches many rows (e.g., 100K+), scoring all of them can be expensive. The BM25 index is most efficient when it can use top-k optimization (ORDER BY + LIMIT) to avoid scoring every matching document.

Post-filtering tradeoff: The index returns top-k results before filtering. If your WHERE clause eliminates most results, you may get fewer rows than requested. Increase LIMIT to compensate, then re-limit in application code.

Best case: Pre-filter with a selective condition (matches <10% of rows), then let BM25 score the reduced set with ORDER BY + LIMIT.

This is similar to the filtering behavior in pgvector, where approximate indexes also apply filtering after the index scan.

Indexing
Create a BM25 index on your text columns:

CREATE INDEX ON documents USING bm25(content) WITH (text_config='english');
Index Options
text_config - PostgreSQL text search configuration to use (required)
k1 - term frequency saturation parameter (1.2 by default)
b - length normalization parameter (0.75 by default)
CREATE INDEX ON documents USING bm25(content) WITH (text_config='english', k1=1.5, b=0.8);
Also supports different text search configurations:

-- English documents with stemming
CREATE INDEX docs_en_idx ON documents USING bm25(content) WITH (text_config='english');

-- Simple text processing without stemming
CREATE INDEX docs_simple_idx ON documents USING bm25(content) WITH (text_config='simple');

-- Language-specific configurations
CREATE INDEX docs_fr_idx ON french_docs USING bm25(content) WITH (text_config='french');
CREATE INDEX docs_de_idx ON german_docs USING bm25(content) WITH (text_config='german');
Data Types
bm25query
The bm25query type represents queries for BM25 scoring with optional index context:

-- Create a bm25query with index name (required for WHERE clause and standalone scoring)
SELECT to_bm25query('search query text', 'docs_idx');
-- Returns: docs_idx:search query text

-- Embedded index name syntax (alternative form using cast)
SELECT 'docs_idx:search query text'::bm25query;
-- Returns: docs_idx:search query text

-- Create a bm25query without index name (only works in ORDER BY with index scan)
SELECT to_bm25query('search query text');
-- Returns: search query text
Note: In PostgreSQL 18, the embedded index name syntax using single colon (:) allows the query planner to determine the index name even when evaluating SELECT clause expressions early. This ensures compatibility across different query evaluation strategies.

bm25query Functions
Function	Description
to_bm25query(text) → bm25query	Create bm25query without index name (for ORDER BY only)
to_bm25query(text, text) → bm25query	Create bm25query with query text and index name
text <@> bm25query → double precision	BM25 scoring operator (returns negative scores)
bm25query = bm25query → boolean	Equality comparison
Performance
pg_textsearch indexes use a memtable architecture for efficient writes. Like other index types, it's faster to create an index after loading your data.

-- Load data first
INSERT INTO documents (content) VALUES (...);

-- Then create index
CREATE INDEX docs_idx ON documents USING bm25(content) WITH (text_config='english');
Parallel Index Builds
pg_textsearch supports parallel index builds for faster indexing of large tables. Postgres automatically uses parallel workers based on table size and configuration.

-- Configure parallel workers (optional, uses server defaults otherwise)
SET max_parallel_maintenance_workers = 4;
SET maintenance_work_mem = '256MB';  -- At least 64MB required for parallel builds

-- Create index (parallel workers used automatically for large tables)
CREATE INDEX docs_idx ON documents USING bm25(content) WITH (text_config='english');
Note: The planner requires maintenance_work_mem >= 64MB to enable parallel index builds. With insufficient memory, builds fall back to serial mode silently.

You'll see a notice when parallel build is used:

NOTICE:  parallel index build: launched 4 of 4 requested workers
For partitioned tables, each partition builds its index independently with parallel workers if the partition is large enough. This allows efficient indexing of very large partitioned datasets.

Performance Tuning
Force-merging segments
The index stores data in multiple segments across levels (similar to an LSM tree). After bulk loads or sustained incremental inserts, multiple segments may accumulate; consolidating them into one improves query speed by reducing the number of segments scanned:

SELECT bm25_force_merge('docs_idx');
This is analogous to Lucene's forceMerge(1). It rewrites all segments into a single segment and reclaims the freed pages. Best used after large batch inserts, not during ongoing write traffic.

Use LIMIT with ORDER BY
Top-k queries (ORDER BY ... LIMIT n) enable Block-Max WAND optimization, which skips blocks of postings that cannot contribute to the top results. Without a LIMIT clause, the index falls back to scoring all matching documents up to pg_textsearch.default_limit.

-- Fast: BMW skips non-competitive blocks
SELECT * FROM documents ORDER BY content <@> 'search terms' LIMIT 10;

-- Slower: scores up to default_limit documents
SELECT * FROM documents ORDER BY content <@> 'search terms';
Segment compression
Compression is on by default and generally improves both index size and query performance (fewer pages to read). Disable only if you observe that decompression overhead is a bottleneck for your workload:

SET pg_textsearch.compress_segments = off;
Postgres settings that affect index builds
Setting	Effect
max_parallel_maintenance_workers	Number of parallel workers for CREATE INDEX (default 2)
maintenance_work_mem	Memory per worker; must be >= 64MB for parallel builds
pg_textsearch GUCs
Setting	Default	Description
pg_textsearch.default_limit	1000	Max documents scored when no LIMIT clause is present
pg_textsearch.compress_segments	on	Compress posting blocks in new segments
pg_textsearch.segments_per_level	8	Segments per level before automatic compaction (2-64)
pg_textsearch.bulk_load_threshold	100000	Terms per transaction before auto-spill (0 = disable)
pg_textsearch.memtable_spill_threshold	32000000	Posting entries before auto-spill (0 = disable)
Spill thresholds
The memtable_spill_threshold controls when the in-memory index flushes to a disk segment. When the memtable reaches this many posting entries, it automatically spills at transaction commit. The bulk_load_threshold triggers a spill based on term count within a single transaction. Both keep memory usage bounded while maintaining good query performance.

Crash recovery: The memtable is rebuilt from the heap on startup, so no data is lost if Postgres crashes before spilling to disk.

Monitoring
-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexrelid::regclass::text ~ 'pg_textsearch';
Examples
Basic Search
CREATE TABLE articles (id serial PRIMARY KEY, title text, content text);
CREATE INDEX articles_idx ON articles USING bm25(content) WITH (text_config='english');

INSERT INTO articles (title, content) VALUES
    ('Database Systems', 'PostgreSQL is a powerful relational database system'),
    ('Search Technology', 'Full text search enables finding relevant documents quickly'),
    ('Information Retrieval', 'BM25 is a ranking function used in search engines');

-- Find relevant documents
SELECT title, content <@> 'database search' as score
FROM articles
ORDER BY score;
Also supports different languages and custom parameters:

-- Different languages
CREATE INDEX fr_idx ON french_articles USING bm25(content) WITH (text_config='french');
CREATE INDEX de_idx ON german_articles USING bm25(content) WITH (text_config='german');

-- Custom parameters
CREATE INDEX custom_idx ON documents USING bm25(content)
    WITH (text_config='english', k1=2.0, b=0.9);
Limitations
No Phrase Queries
The BM25 index stores term frequencies but not term positions, so it cannot natively evaluate phrase queries like "database system". You can emulate phrase matching by combining BM25 ranking with a post-filter:

-- BM25 ranks candidates; subquery over-fetches to account for
-- post-filter eliminating non-phrase matches
SELECT * FROM (
    SELECT *, content <@> 'database system' AS score
    FROM documents
    ORDER BY score
    LIMIT 100  -- over-fetch
) sub
WHERE content ILIKE '%database system%'
ORDER BY score
LIMIT 10;
Because the post-filter eliminates some results, the inner LIMIT should be larger than the desired result count.

No Expression Indexing
Each BM25 index covers a single text column. You cannot create an index on an expression like lower(title) || ' ' || content. As a workaround, use a generated column:

ALTER TABLE documents ADD COLUMN search_text text
    GENERATED ALWAYS AS (
        COALESCE(title, '') || ' ' || COALESCE(content, '')
    ) STORED;
CREATE INDEX ON documents USING bm25(search_text)
    WITH (text_config = 'english');
No Built-in Faceted Search
pg_textsearch does not provide dedicated faceting operators, but standard Postgres query machinery handles common faceting patterns:

-- Filter by category (assumes a B-tree index on category)
SELECT * FROM documents
WHERE category = 'engineering'
ORDER BY content <@> 'search terms'
LIMIT 10;

-- Compute facet counts over BM25-matched results
SELECT category, count(*)
FROM documents
WHERE content <@> to_bm25query('search terms', 'docs_idx') < -1.0
GROUP BY category;
Insert/Update Performance
The memtable architecture is designed to support efficient writes, but sustained write-heavy workloads are not yet fully optimized. For initial data loading, creating the index after loading data is faster than incremental inserts. This is an active area of development.

No Background Compaction
Segment compaction currently runs synchronously during memtable spill operations. Write-heavy workloads may observe compaction latency during spills. Background compaction is planned for a future release.

Partitioned Tables
BM25 indexes on partitioned tables use partition-local statistics. Each partition maintains its own:

Document count (total_docs)
Average document length (avg_doc_len)
Per-term document frequencies for IDF calculation
This means:

Queries targeting a single partition compute accurate BM25 scores using that partition's statistics
Queries spanning multiple partitions return scores computed independently per partition, which may not be directly comparable across partitions
Example: If partition A has 1000 documents and partition B has 10 documents, the term "database" would have different IDF values in each partition. Results from both partitions would have scores on different scales.

Recommendations:

For time-partitioned data, query individual partitions when score comparability matters
Use partitioning schemes where queries naturally target single partitions
Consider this behavior when designing partition strategies for search workloads
-- Query single partition (scores are accurate within partition)
SELECT * FROM docs
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01'
ORDER BY content <@> 'search terms'
LIMIT 10;

-- Cross-partition query (scores computed per-partition)
SELECT * FROM docs
ORDER BY content <@> 'search terms'
LIMIT 10;
Word Length Limit
pg_textsearch inherits PostgreSQL's tsvector word length limit of 2047 characters. Words exceeding this limit are ignored during tokenization (with an INFO message). This is defined by MAXSTRLEN in PostgreSQL's text search implementation.

For typical natural language text, this limit is never encountered. It may affect documents containing very long tokens such as base64-encoded data, long URLs, or concatenated identifiers.

This behavior is similar to other search engines:

Elasticsearch: Truncates tokens (configurable via truncate filter, default 10 chars)
Tantivy: Truncates to 255 bytes by default
PL/pgSQL and Stored Procedures
The implicit text <@> 'query' syntax relies on planner hooks to automatically detect the BM25 index. These hooks don't run inside PL/pgSQL DO blocks, functions, or stored procedures.

Inside PL/pgSQL, use explicit index names with to_bm25query():

-- This won't work in PL/pgSQL:
-- SELECT * FROM docs ORDER BY content <@> 'search terms' LIMIT 10;

-- Use explicit index name instead:
SELECT * FROM docs
ORDER BY content <@> to_bm25query('search terms', 'docs_idx')
LIMIT 10;
Regular SQL queries (outside PL/pgSQL) support both forms.

Troubleshooting
-- List available text search configurations
SELECT cfgname FROM pg_ts_config;

-- List BM25 indexes
SELECT indexname FROM pg_indexes WHERE indexdef LIKE '%USING bm25%';
Installation Notes
If your machine has multiple Postgres installations, specify the path to pg_config:

export PG_CONFIG=/Library/PostgreSQL/18/bin/pg_config  # or 17
make clean && make && make install
If you get compilation errors, install Postgres development files:

# Ubuntu/Debian
sudo apt install postgresql-server-dev-17  # for PostgreSQL 17
sudo apt install postgresql-server-dev-18  # for PostgreSQL 18
Reference
Index Options
Option	Type	Default	Description
text_config	string	required	PostgreSQL text search configuration to use
k1	real	1.2	Term frequency saturation parameter (0.1 to 10.0)
b	real	0.75	Length normalization parameter (0.0 to 1.0)
Text Search Configurations
Available configurations depend on your Postgres installation:

# SELECT cfgname FROM pg_ts_config;
  cfgname
------------
 simple
 arabic
 armenian
 basque
 catalan
 danish
 dutch
 english
 finnish
 french
 german
 greek
 hindi
 hungarian
 indonesian
 irish
 italian
 lithuanian
 nepali
 norwegian
 portuguese
 romanian
 russian
 serbian
 spanish
 swedish
 tamil
 turkish
 yiddish
(29 rows)
Further language support is available via extensions such as zhparser.

Development Functions
These functions are for debugging and development use only. Their interface may change in future releases without notice. Functions marked with † require superuser privileges.

Function	Description
bm25_force_merge(index_name) → void	Merge all segments into one (improves query speed)
bm25_spill_index(index_name) → int4	Force memtable spill to disk segment
bm25_dump_index(index_name) † → text	Dump internal index structure (truncated)
bm25_summarize_index(index_name) † → text	Show index statistics without content
Additional file-writing debug functions (bm25_dump_index(text, text) and bm25_debug_pageviz) are available in debug builds only (compile with -DDEBUG_DUMP_INDEX).

-- Merge all segments into one (best after bulk loads)
SELECT bm25_force_merge('docs_idx');

-- Force spill to disk (returns number of entries spilled)
SELECT bm25_spill_index('docs_idx');

-- Quick overview of index statistics
SELECT bm25_summarize_index('docs_idx');

-- Detailed dump for debugging (truncated output)
SELECT bm25_dump_index('docs_idx');
Extension Compatibility
pg_textsearch uses fixed LWLock tranche IDs 1001-1008 to support large numbers of indexes (e.g., partitioned tables with hundreds of partitions). If you use another Postgres extension that also registers fixed tranche IDs in this range, wait event names in pg_stat_activity may be incorrect. Core Postgres tranches use IDs below 100. If you encounter a conflict, please open an issue.

Contributing
See CONTRIBUTING.md for development setup, code style, and how to submit pull requests.

Bug Reports: Create an issue
Feature Requests: Request a feature
General Discussion: Start a discussion
