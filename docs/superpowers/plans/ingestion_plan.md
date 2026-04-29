# Contract Ingestion and Retrieval LangGraph Plan

## Summary
Replace the current raw-text/entity-only ingestion flow with a canonical legal RAG pipeline built around the `clauses` table. The new ingestion graph will parse uploaded documents with Docling, extract full-document contract metadata, create parent document rows plus child clause chunks, contextualize every child chunk with a structured LLM preamble, embed/cache/write 768-dimensional chunks to Postgres, and upsert Graphiti contract events/relationships.

Build a new retrieval graph that uses `MessagesState`, query analysis, Redis exact-match caching, in-database hybrid RRF over `clauses`, optional graph-derived chunk filters, cross-encoder reranking, context grading with bounded retry, and grounded answer generation with citations.

Decisions locked:
- Canonical retrieval storage: `clauses` table.
- Vector dimension: `768`.
- Text search: `pg_textsearch` exclusively; no `tsvector`, no GIN full-text index.
- Tool/external I/O retry UX: 3 immediate consecutive retry attempts, no exponential wait.
- TOON usage: prompt/context serialization only; LangGraph state remains typed objects/messages.
- Backward compatibility: not required.
- No unit-test plan for this pass; use the 18-point checklist below as the implementation correctness guide.

## Key Changes
- Add `parent_documents` as one row per uploaded file and extend `clauses` with `parent_doc_id`, `chunk_index`, `metadata_ JSONB`, `custom_metadata JSONB`, `preamble`, `chunk_text`, `search_text`, and 768-dimensional `embedding`.
- `search_text` should be a generated text column combining `clause_type`, `preamble`, and `chunk_text`, because `pg_textsearch` indexes one column only.
- Create `clauses_bm25_idx` with `USING bm25(search_text) WITH (text_config = 'english', k1 = 1.2, b = 0.75)`. Do not add `text_search_vector` or any `tsvector`-based generated column.
- Create BM25 indexes after bulk load where possible. For large incremental batches, run `SELECT bm25_force_merge('clauses_bm25_idx')` after ingestion to consolidate index segments.
- Keep `entities`, `relationships`, and `events` for legal memory/audit use, but stop treating entity extraction as the source of clause chunks.
- Update ingestion API/service to pass uploaded bytes, filename/source path, user metadata, and document attributes into the graph instead of decoding bytes to raw text in the router.
- Build `src/app/shared/langgraph_layer/ingestion_kb` as:
  `parse_document -> extract_schema -> segment_document -> contextualize_chunks (Send fan-out) -> classify/extract_entities -> embed_store -> graphiti_upsert`.
- Use Docling from `app/shared/rag/document_processing` as the layout-aware parser, exporting structured markdown and preserving table rows; add a table naturalization helper for `search_text` so markdown table pipes do not poison BM25.
- Use Gemini `.with_structured_output(...)` for all structured LLM calls: schema extraction, clause classification, entity extraction, contextual chunk preamble, query analysis, context grading, and grounded generation.
- Use TOON serialization only when rendering structured prompt payloads for LLM nodes; do not store TOON in DB or LangGraph state.
- Add a shared retry helper for external I/O with exactly 3 immediate attempts and no sleep/backoff between attempts; apply it around Gemini, Postgres, Neo4j, Graphiti, Redis, and embedding calls where retry is safe.
- Add Redis embedding cache keyed by `sha256(text_to_embed)` and query-answer cache keyed by `sha256(rewritten_query + doc_ids_filter)`, with 1h TTL by default and 24h TTL for immutable historical document filters.
- Add `sentence-transformers` dependency and a reranker adapter using `BAAI/bge-reranker-v2-m3` by default, falling back to `cross-encoder/ms-marco-MiniLM-L-6-v2` only by config.

## Retrieval DB Metadata
Every child chunk stored in `clauses` for retrieval must include both `metadata_` and `custom_metadata`.

`metadata_` must include:
```yaml
source: filePath
page_no: 0
```

`custom_metadata` must include:
```yaml
source: filePath
page_no: 0
document_summary: <full-document or parent-document summary>
chunk_id: <postgres clause/chunk UUID as string>
chunk_faqs: <list of generated FAQs for this chunk>
chunk_keywords: <list of generated keywords for this chunk>
```

Additional canonical filters from `ContractMetadata` should also be written to `metadata_`, including jurisdiction, contract type, year, party names, effective date, governing law, contract value, termination notice days, and liability cap. Page numbers default to `0` when Docling does not expose page location.

## Retrieval Graph
- Create `src/app/shared/langgraph_layer/retrieval_kb` with `RetrievalState(MessagesState)` plus typed fields for `query_plan`, `retrieved_chunks`, `reranked_chunks`, `context_grade`, `iteration_count`, and `generated_answer`.
- Query Analyzer node:
  rewrites coreferences, decomposes multi-part queries, chooses route `hybrid_postgres | graph_neo4j | both`, classifies query type, sets runtime RRF weights, and checks Redis answer cache before retrieval.
- `QueryPlan` should include `vector_weight` and `keyword_weight`; default legal weighting is `0.4` semantic and `0.6` BM25, with semantic raised for conceptual queries and BM25 raised for exact clause/reference queries.
- Graph route:
  query Graphiti/Neo4j for relevant contract graph facts and `REFERENCES_CLAUSE.postgres_chunk_id`, then pass those IDs as `:chunk_ids` into Postgres retrieval.
- Hybrid Postgres route:
  run one SQL CTE over `clauses` that ranks vector results and `pg_textsearch` BM25 results, fuses with weighted RRF, applies `metadata_` hard filters and optional graph chunk IDs before both search branches, and returns only top K rows to Python.
- Always use explicit `to_bm25query(:query_text, 'clauses_bm25_idx')` in SQLAlchemy raw SQL. Do not rely on implicit BM25 query detection.
- `pg_textsearch` scores are negative; keyword ranking must order BM25 scores ascending so rank 1 is the most negative/best match.
- Exact phrase requirements, such as `"force majeure"`, should use BM25 for candidate generation and an `ILIKE '%force majeure%'` post-filter over the candidate subquery, because `pg_textsearch` does not provide phrase search.
- Add optional minimum sparse relevance gate with `search_text <@> to_bm25query(:query_text, 'clauses_bm25_idx') < :bm25_threshold`, defaulting to no hard threshold unless QueryPlan asks for exact keyword precision.
- Reranker node:
  rerank top 20 chunks locally and keep top 5; mark this node as a Celery task candidate for V2 because it is CPU-bound.
- Context Grader node:
  return `ContextGrade(sufficient, missing_aspects, rewrite_suggestion)`; if insufficient, loop to Query Analyzer with the rewrite suggestion, capped at 2 attempts.
- Generator node:
  output `GeneratedAnswer(answer, citations, confidence)` and require every factual claim to cite exact `chunk_id` and `clause_type`; append fallback disclaimer when confidence is `uncertain`.
- Expose retrieval through a new typed service method and route, preferably `POST /search/ask`, while deprecating the existing Python-side RRF path for this legal KB workflow.

## Corrected RRF SQL Shape
Use this shape as the implementation target for repository retrieval, adapting only table/column names if the migration uses a different exact primary key name.

```sql
WITH
candidate_docs AS (
    SELECT
        chunk_id,
        chunk_text,
        parent_doc_id,
        clause_type,
        preamble,
        metadata_,
        custom_metadata,
        embedding,
        search_text
    FROM clauses
    WHERE
        (:jurisdiction IS NULL OR metadata_->>'jurisdiction' = :jurisdiction)
        AND (:contract_type IS NULL OR metadata_->>'contract_type' = :contract_type)
        AND (:chunk_ids IS NULL OR chunk_id = ANY(:chunk_ids))
),
vector_search AS (
    SELECT
        chunk_id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:query_embedding AS vector)) AS rank
    FROM candidate_docs
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query_embedding AS vector)
    LIMIT 20
),
keyword_search AS (
    SELECT
        chunk_id,
        ROW_NUMBER() OVER (
            ORDER BY search_text <@> to_bm25query(:query_text, 'clauses_bm25_idx')
        ) AS rank
    FROM candidate_docs
    ORDER BY search_text <@> to_bm25query(:query_text, 'clauses_bm25_idx')
    LIMIT 20
),
fused AS (
    SELECT
        COALESCE(v.chunk_id, k.chunk_id) AS chunk_id,
        (:vector_weight * COALESCE(1.0 / (60.0 + v.rank), 0.0)) +
        (:keyword_weight * COALESCE(1.0 / (60.0 + k.rank), 0.0)) AS rrf_score
    FROM vector_search v
    FULL OUTER JOIN keyword_search k ON v.chunk_id = k.chunk_id
)
SELECT
    c.chunk_id,
    c.chunk_text,
    c.preamble,
    c.clause_type,
    c.parent_doc_id,
    c.metadata_,
    c.custom_metadata,
    f.rrf_score
FROM fused f
JOIN clauses c ON c.chunk_id = f.chunk_id
ORDER BY f.rrf_score DESC
LIMIT :limit;
```

## Graphiti Expansion
- Add custom Graphiti entity/edge schemas for `Party`, `Obligation`, `RightOrPermission`, `PenaltyClause`, and the requested edge types.
- Pass Graphiti custom entity/edge Pydantic schemas into `add_episode` calls using `entity_types`, `edge_types`, and `edge_type_map`.
- Write contract event episodes for `contract_signed`, `amendment_effective`, and `expiry_date` using extracted `ContractMetadata`.
- Store `postgres_chunk_id` on `REFERENCES_CLAUSE` edges so graph traversal can return direct Postgres chunk filters for retrieval.
- Use Graphiti episodes for temporal/provenance semantics and direct Neo4j Cypher only where deterministic cross-store links or index/constraint setup require exact properties.

## Correctness Checklist: Original 18-Point Guide
Use this checklist instead of a unit-test plan to verify implementation completeness.

1. **Layout-Aware Parsing Node**
   - `parse_document` is the first ingestion node.
   - Uses Docling.
   - Outputs structured markdown.
   - Tables preserve row/cell structure and are not flattened into unreadable strings.

2. **Parent-Child Chunk Architecture**
   - One parent document row exists per uploaded file.
   - Multiple child `clauses` rows exist per parent.
   - `clauses.parent_doc_id` is a UUID FK.
   - `clauses.chunk_index` is populated.
   - Retrieval fetches child chunks and can return parent/surrounding context.

3. **Contextual Preamble Injection**
   - A separate `contextualize_chunks` node exists.
   - Uses LangGraph `Send` fan-out.
   - Each child chunk receives a preamble like:
     `"This is [clause_type] from [contract_name] between [party_a] and [party_b], effective [date]."`
   - Uses `.with_structured_output(ContextualizedChunk)`.
   - `ContextualizedChunk` includes `preamble`, `text`, and `tokens`.

4. **langextract Structured Extraction Pass**
   - A dedicated `extract_schema` node runs on full document text, not chunks.
   - Extracts `effective_date`, `parties`, `contract_value`, `jurisdiction`, `governing_law`, `termination_notice_days`, and `liability_cap`.
   - Uses `.with_structured_output(ContractMetadata)` against Gemini.
   - Output populates `metadata_ JSONB`.

5. **BM25 / pg_textsearch Dual-Write**
   - `pg_textsearch` is used exclusively.
   - No `tsvector` column exists for this pipeline.
   - No GIN full-text index exists for this pipeline.
   - `clauses_bm25_idx` exists on `search_text`.
   - `search_text` includes clause type, preamble, and chunk text.
   - Retrieval uses `<@> to_bm25query(:query_text, 'clauses_bm25_idx')`.

6. **JSONB Metadata Column for Pre-Filtering**
   - `clauses.metadata_` exists as JSONB.
   - Populated with jurisdiction, contract type, year, party names, and required retrieval metadata.
   - Retrieval can hard-filter with expressions like `metadata_->>'jurisdiction' = 'Delaware'`.

7. **Immediate 3-Attempt Retry Handling**
   - External I/O in ingestion/retrieval nodes has retry handling.
   - Gemini, Postgres writes, Neo4j writes, Graphiti upserts, Redis calls, and embedding calls are covered where retry is safe.
   - Retry policy is exactly 3 immediate attempts with no exponential wait.
   - Transient errors are separated from validation/permanent errors.

8. **MessagesState Standardization**
   - Retrieval graph state inherits from `MessagesState`.
   - Retriever and generator pass `HumanMessage` / `AIMessage` objects through state.
   - Ingestion can remain specialized typed state.
   - Handoff boundary from ingestion to retrieval is typed and message-compatible.

9. **Structured Output Throughout**
   - No raw JSON parsing for structured LLM outputs.
   - Entity extraction uses `EntityExtractionResult`.
   - Clause classification uses `ClauseClassification`.
   - Preamble generation uses `ContextualizedChunk`.
   - Schema extraction uses `ContractMetadata`.

10. **Query Analyzer & Router Node**
   - Retrieval graph starts with Query Analyzer.
   - Performs coreference rewrite.
   - Decomposes multi-part questions.
   - Routes to `hybrid_postgres`, `graph_neo4j`, or `both`.
   - Emits runtime RRF weights.
   - Uses `.with_structured_output(QueryPlan)`.

11. **In-Database RRF via CTE**
   - RRF is computed inside Postgres.
   - Vector and `pg_textsearch` ranks are CTE/subquery branches.
   - BM25 ordering accounts for negative scores.
   - Outer query fuses with weighted `1/(60 + rank)` terms.
   - Python receives only top-K fused rows.

12. **Cross-Encoder Reranker Node**
   - Retrieval candidates top 20 are reranked.
   - Default model is `BAAI/bge-reranker-v2-m3` or configured fallback.
   - Output is cut to top 5.
   - Node is documented as a Celery task candidate for V2.

13. **Context Grader Node**
   - Uses `.with_structured_output(ContextGrade)`.
   - Includes `sufficient`, `missing_aspects`, and `rewrite_suggestion`.
   - If insufficient, loops back to Query Analyzer.
   - Loop is capped at 2 iterations.
   - After cap, returns hardcoded fallback instead of improvising.

14. **Grounded Generator Node**
   - Final generator requires citation for every factual claim.
   - Citations include exact `chunk_id` and `clause_type`.
   - Output schema is `GeneratedAnswer`.
   - If confidence is `uncertain`, fallback disclaimer is appended.

15. **Redis Exact-Match Query Cache**
   - Query Analyzer checks cache before retrieval work.
   - Key is `sha256(rewritten_query + doc_ids_filter)`.
   - Cache hit returns `GeneratedAnswer` directly.
   - TTL is 1 hour normally, 24 hours for immutable historical documents.

16. **Embedding Cache**
   - Embedding API calls are wrapped in Redis cache.
   - Key is `sha256(text_to_embed)`.
   - Repeated boilerplate clauses reuse cached embeddings.

17. **Additional Graphiti Node/Edge Types**
   - Adds nodes: `Party`, `Obligation`, `RightOrPermission`, `PenaltyClause`.
   - Adds edges: `SIGNED_BY`, `SUBSIDIARY_OF`, `OBLIGATED_TO`, `GOVERNED_BY`, `SUPERSEDES`, `REFERENCES_CLAUSE`.
   - `REFERENCES_CLAUSE` carries `postgres_chunk_id`.
   - Graph traversal can return Postgres IDs as hard retrieval filters.

18. **Graphiti Episode Upsert for Contract Events**
   - Ingestion writes event episodes for `contract_signed`, `amendment_effective`, and `expiry_date`.
   - Temporal queries can use Graphiti’s time-aware graph search instead of scanning Postgres.

## Assumptions
- `clauses` becomes canonical for retrieval; existing `search_documents/search_chunks` can remain for older generic search code but will not be used by this legal KB graph.
- Parent table name will be `parent_documents`.
- `clauses.embedding` uses `vector(768)`.
- `metadata_` stores both canonical extracted filters and the required `source/page_no` retrieval metadata.
- `custom_metadata` stores the required chunk-local enrichment fields exactly as listed above.
- `pg_textsearch` is the only sparse retrieval mechanism for this pipeline.
- Fallback response is hardcoded and non-generative when context remains insufficient after 2 retrieval attempts.
