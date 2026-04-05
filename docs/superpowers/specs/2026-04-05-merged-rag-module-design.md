# Merged Reusable RAG Module Design

**Goal:** Replace the educational numbered RAG strategy files in `src/app/shared/rag/` with one reusable production-aligned module that uses the application's async SQLAlchemy engine/session factory, project logger, and `orjson`.

## Scope

This change consolidates the following strategy demos into a single reusable API surface:

- Re-ranking
- Agentic RAG
- Knowledge graph retrieval
- Contextual retrieval
- Query expansion
- Multi-query retrieval
- Context-aware chunking
- Late chunking
- Hierarchical retrieval
- Self-reflective RAG
- Fine-tuned embeddings

It also merges the folder documentation into one reference document and deletes the numbered strategy files.

## Design

### Module shape

Create one module in `src/app/shared/rag/` that exposes:

- focused dataclasses for configuration
- reusable async functions for ingestion, retrieval, and answer synthesis
- one small service class that receives app-scoped dependencies instead of opening its own database connections

The module should be importable from application code without creating global network/database side effects.

### Dependency model

The merged module will:

- rely on `app.state.db_engine` and `app.state.db_session_local`
- use `app.utils.logger`
- use `orjson` for metadata serialization where JSON persistence is needed
- use SQLAlchemy `text(...)` for raw vector and hybrid retrieval queries
- reuse the existing LangChain model factories from `app.shared.langchain_layer.models`

### Storage assumptions

Use the existing `document_vectors` table for persisted chunk content and embeddings. Strategy-specific metadata is stored in `meta_data` so one table can support multiple retrieval patterns without introducing new hardcoded demo tables such as `chunks`, `parent_chunks`, or `child_chunks`.

### Production constraints

- No `psycopg2`
- No module-level singleton DB connections
- No `logging`/`json` standard-library usage inside the new module
- Async-first public API
- Errors logged with project logger and surfaced with clear exceptions or deterministic fallbacks

## Data flow

1. Ingestion helpers chunk and optionally enrich content.
2. Embedding helpers generate vectors for storage or query-time retrieval.
3. Retrieval helpers query `document_vectors` using SQLAlchemy text queries.
4. Strategy helpers compose retrieval outputs into prompts or structured results.
5. Knowledge graph retrieval remains injectable so existing Graphiti/Neo4j wiring can be used without baking credentials into the module.

## Testing

Targeted unit tests cover:

- query deduplication and expansion parsing
- semantic/context-aware chunking behavior
- late chunk embedding blending helper behavior
- hierarchical metadata encoding/decoding helpers
- document row formatting and strategy result normalization

Database integration is not required for this change because the current request is a consolidation/refactor around reusable boundaries.
