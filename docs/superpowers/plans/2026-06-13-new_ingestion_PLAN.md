# Unified S3-Backed Ingestion, Search, RAG, and Agent Tooling Plan

## Summary
Breaking rebuild the current generic search and legal KB ingestion into one unified document/chunk pipeline. The rejected details from the earlier plan are dropped: no GridFS-first storage, no separate `search_documents/search_chunks` versus `parent_documents/clauses` split, no “generic search unchanged” compatibility promise, and no clauses-only legal store.

The new target is:

`upload file -> store raw object in S3-compatible storage -> parse with Docling/text parser -> classify document -> segment generic chunks or legal clauses -> enrich chunks -> embed/cache -> write unified Postgres tables -> required Graphiti verification for legal chunks -> retrieve through unified BM25/vector/trigram RRF -> expose generic and legal tools to agents`.

Existing old Postgres search data may be dropped.

## Key Decisions
- Use S3-compatible storage first through an object-store port; AWS S3 and MinIO should both work.
- Use one wide Postgres schema for documents and chunks.
- Replace old `search_documents`, `search_chunks`, `parent_documents`, and `clauses` retrieval roles with unified `documents` and `chunks`.
- Use `pg_textsearch` BM25, vector search, and trigram search; drop `tsvector` from the new path.
- Graphiti consistency is required for legal documents before they become retrievable.
- Low-quality ingestion is allowed, but warnings must be stored and surfaced in retrieval/answer responses.
- Verification is checklist/manual smoke-test driven for this pass, not a large automated test suite.

## Phase 1: Storage Contract
Create a storage interface and S3-compatible implementation.

```python
class ObjectStore(Protocol):
    async def put(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> str: ...

    async def get(self, *, uri: str) -> bytes: ...

    async def delete(self, *, uri: str) -> None: ...
```

Add settings:

```python
S3_ENDPOINT_URL: str | None = None
S3_BUCKET_NAME: str
S3_ACCESS_KEY_ID: str
S3_SECRET_ACCESS_KEY: str
S3_REGION: str = "us-east-1"
S3_FORCE_PATH_STYLE: bool = True
```

Use object keys like:

```text
documents/{user_id}/{doc_id}/{sha256}.{extension}
```

## Phase 2: Unified Schema
Replace the old split model with one wide schema.

```sql
CREATE TABLE documents (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id text NOT NULL,
  title text NOT NULL,
  source_uri text,
  object_uri text NOT NULL,
  content_hash text NOT NULL UNIQUE,
  document_kind text NOT NULL DEFAULT 'generic',
  status text NOT NULL DEFAULT 'received',
  jurisdiction text,
  contract_type text,
  parties jsonb NOT NULL DEFAULT '[]',
  metadata_ jsonb NOT NULL DEFAULT '{}',
  quality_warnings jsonb NOT NULL DEFAULT '[]',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id text NOT NULL,
  chunk_index int NOT NULL,
  chunk_kind text NOT NULL DEFAULT 'generic',
  content text NOT NULL,
  preamble text NOT NULL DEFAULT '',
  clause_type text,
  page_no int NOT NULL DEFAULT 0,
  embedding vector(768),
  metadata_ jsonb NOT NULL DEFAULT '{}',
  custom_metadata jsonb NOT NULL DEFAULT '{}',
  quality_warnings jsonb NOT NULL DEFAULT '[]',
  graphiti_episode_id text,
  graphiti_verified boolean NOT NULL DEFAULT false,
  search_text text GENERATED ALWAYS AS (
    COALESCE(clause_type, '') || ' ' ||
    COALESCE(preamble, '') || ' ' ||
    COALESCE(content, '')
  ) STORED,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(document_id, chunk_index)
);
```

Indexes:

```sql
CREATE INDEX chunks_user_doc_idx ON chunks(user_id, document_id);
CREATE INDEX chunks_kind_idx ON chunks(chunk_kind);
CREATE INDEX chunks_metadata_gin_idx ON chunks USING gin(metadata_);
CREATE INDEX chunks_bm25_idx ON chunks
USING bm25(search_text) WITH (text_config='english', k1=1.2, b=0.75);
CREATE INDEX chunks_embedding_idx ON chunks
USING diskann (embedding vector_cosine_ops);
CREATE INDEX chunks_search_text_trgm_idx ON chunks
USING gin(search_text gin_trgm_ops);
```

## Phase 3: Ingestion Graph
Build one ingestion graph with domain-specific branches.

```text
store_raw_file
  -> parse_document
  -> classify_document
  -> segment_chunks
  -> enrich_chunks
  -> embed_chunks
  -> store_postgres
  -> graphiti_commit
  -> verify_graphiti_links
  -> finalize_ingestion
```

State sketch:

```python
class UnifiedIngestionState(BaseModel):
    doc_id: str
    user_id: str
    raw_bytes: bytes
    filename: str
    content_type: str
    object_uri: str | None = None
    document_kind: Literal["generic", "legal_contract", "legal_policy", "other"] = "generic"
    parsed_markdown: str = ""
    chunks: list[PreparedChunk] = []
    quality_warnings: list[QualityWarning] = []
    graphiti_required: bool = False
    status: Literal[
        "received",
        "parsed",
        "stored_postgres",
        "graph_verified",
        "completed",
        "completed_with_warnings",
        "failed",
    ] = "received"
```

Rules:
- Docling parses PDFs/docs and preserves tables.
- Plain text/markdown uses a lightweight parser.
- Generic documents use token chunking.
- Legal contracts use clause-aware segmentation, structured metadata extraction, contextual preambles, `clause_type`, parties, jurisdiction, governing law, effective date, and legal filters.
- Fallback segmentation is allowed only with explicit `quality_warnings`.
- LLM structured work uses `.with_structured_output(...)`.
- TOON is used only for prompt serialization.
- Retry policy is exactly 3 immediate attempts for retry-safe external I/O.
- Validation/quality failures are not retried as transient dependency failures.

## Phase 4: Required Graphiti Protocol
For legal documents, ingestion is not complete until Graphiti writes and verifies chunk links.

```python
class GraphitiChunkLink(BaseModel):
    doc_id: str
    chunk_id: str
    graphiti_episode_id: str
    postgres_chunk_id: str
    verified: bool
```

Protocol:
- Write one Graphiti episode per legal chunk.
- Write contract event episodes for `contract_signed`, `amendment_effective`, and `expiry_date` when extracted.
- Store `postgres_chunk_id` in a recoverable Graphiti link.
- Verify each legal chunk can be found from Graphiti before setting `graphiti_verified = true`.
- If Graphiti fails, mark document `failed`; do not expose legal chunks to legal retrieval.
- Generic documents may skip Graphiti.

## Phase 5: Unified Retrieval Core
Extract current retrieval strengths into a reusable core:
- BM25 branch.
- Vector branch.
- Trigram branch.
- In-database or repository-level RRF.
- Redis query cache.
- Embedding cache.
- Adjacent chunk context assembly.
- Cross-encoder reranking for legal answers.
- Grounded answer generation with citations.

SQL shape:

```sql
WITH candidate_chunks AS (
  SELECT *
  FROM chunks
  WHERE user_id = :user_id
    AND (:document_ids IS NULL OR document_id = ANY(CAST(:document_ids AS uuid[])))
    AND (:document_kind IS NULL OR chunk_kind = :document_kind)
    AND (:jurisdiction IS NULL OR metadata_->>'jurisdiction' = :jurisdiction)
    AND (:contract_type IS NULL OR metadata_->>'contract_type' = :contract_type)
    AND (:require_graphiti_verified IS FALSE OR graphiti_verified IS TRUE)
),
vector_search AS (
  SELECT id, row_number() OVER (
    ORDER BY embedding <=> CAST(:query_embedding AS vector)
  ) AS rank
  FROM candidate_chunks
  WHERE embedding IS NOT NULL
  LIMIT 50
),
bm25_search AS (
  SELECT id, row_number() OVER (
    ORDER BY search_text <@> to_bm25query(:query_text, 'chunks_bm25_idx')
  ) AS rank
  FROM candidate_chunks
  ORDER BY search_text <@> to_bm25query(:query_text, 'chunks_bm25_idx')
  LIMIT 50
),
trigram_search AS (
  SELECT id, row_number() OVER (
    ORDER BY similarity(search_text, :query_text) DESC
  ) AS rank
  FROM candidate_chunks
  WHERE search_text % :query_text
  LIMIT 50
)
-- fuse with weighted 1/(60 + rank), return top K
```

Legal ask flow:

```text
query_analyzer
  -> optional_graphiti_chunk_filter
  -> unified_rrf_search
  -> reranker_top_20_to_5
  -> context_grader
  -> retry_query_if_insufficient_max_2
  -> grounded_generator
```

## Phase 6: Public APIs
Use a clean unified namespace.

- `POST /api/v1/documents/upload`
  - Multipart file upload.
  - Stores raw file in S3-compatible storage.
  - Runs unified ingestion.
  - Returns `doc_id`, status, chunk count, document kind, object URI, and warning summary.

- `GET /api/v1/documents/{doc_id}/status`
  - Returns ingestion status, warnings, Graphiti verification state, and counts.

- `POST /api/v1/search`
  - Generic hybrid search over unified chunks.

- `POST /api/v1/search/rag`
  - Generic retrieved context assembly.

- `POST /api/v1/search/ask`
  - Generic grounded corpus QA.

- `POST /api/v1/legal/ask`
  - Legal KB QA requiring legal filters, citations, and Graphiti-verified chunks.

Response warning shape:

```python
class QualityWarningDTO(BaseModel):
    stage: str
    code: str
    message: str
    severity: Literal["info", "warning", "error"]
```

## Phase 7: Agent Tool Surface
Expose both generic and legal typed tools over the same storage.

Generic tools:
- `search_documents(query, filters)`
- `get_document_context(document_id, chunk_ids)`
- `ask_corpus(query, filters)`

Legal tools:
- `search_contract_clauses(query, jurisdiction, contract_type, parties, clause_type)`
- `ask_contract(query, doc_ids_filter)`
- `get_clause_context(chunk_id)`
- `graph_filtered_contract_search(query)`
- `find_obligations(...)`
- `find_termination_terms(...)`
- `find_risk_clauses(...)`

Tool schema example:

```python
class ContractClauseSearchInput(BaseModel):
    query: str
    doc_ids_filter: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    contract_type: str | None = None
    clause_type: str | None = None
    limit: int = Field(default=10, ge=1, le=50)
```

## Phase 8: Operational Readiness
- Wire object store, ingestion graph, Redis, DB engine, embedding client, LLM, and Graphiti in lifespan.
- Add startup checks for S3 bucket access, Redis, Postgres, `vector`, `pg_textsearch`, `pg_trgm`, `chunks_bm25_idx`, and Graphiti.
- Add structured logs per graph node: `doc_id`, `user_id`, `stage`, `status`, `latency_ms`, `warning_count`, and `failure_class`.
- Add manual smoke checks:
  - Upload generic file.
  - Upload legal contract.
  - Confirm S3 object exists.
  - Confirm chunks are written.
  - Confirm legal chunks are Graphiti verified.
  - Run generic search.
  - Run legal ask with citations.
  - Confirm cross-user retrieval isolation.
- Run `graphify update .` after code implementation changes.

## Acceptance Checklist
- Raw files are stored in S3-compatible storage before parsing.
- Old search data can be dropped.
- Unified `documents` and `chunks` replace old retrieval tables.
- BM25, vector, and trigram retrieval all work from `chunks`.
- All retrieval hard-filters `user_id`.
- Legal chunks include preamble, `clause_type`, metadata, quality warnings, and Graphiti verification.
- Graphiti failure prevents legal document completion.
- Low-quality ingestion completes only with stored warnings.
- Legal answers cite exact `chunk_id` and `clause_type`.
- Generic and legal agent tools both use the unified retrieval core.
