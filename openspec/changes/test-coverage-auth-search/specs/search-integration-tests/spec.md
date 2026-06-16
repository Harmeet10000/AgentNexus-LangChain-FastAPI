# Capability: search-integration-tests

## Purpose

Integration tests for the search ingestion pipeline and hybrid search. Catches regressions in chunk dedup, embedding storage, BM25/vector/trigram fusion, and RAG context assembly.

## Requirements

### R1: Test Fixtures
- `tests/conftest.py` with PostgreSQL async session (pgvector enabled)
- Mock `GoogleGenerativeAIEmbeddings` returning fixed 768-dim vectors
- Mock Celery (no real RabbitMQ)
- Content hash dedup via real DB

### R2: Document Ingestion
- Test: `process_ingestion_document` chunks text and stores embeddings
- Test: chunks have correct `chunk_index`, `content`, `embedding` fields
- Test: empty document returns 0 chunks
- Test: large document splits into multiple chunks with overlap
- Test: content hash dedup prevents re-ingestion

### R3: Hybrid Search
- Test: BM25 search finds documents by keyword match
- Test: vector search finds documents by semantic similarity
- Test: trigram search finds documents by fuzzy match
- Test: RRF fusion ranks results from all three sources
- Test: `candidate_limit` parameter limits result count
- Test: metadata filter narrows results

### R4: RAG Context Assembly
- Test: `assemble_rag_context` groups chunks by document
- Test: adjacent chunks from same document are merged
- Test: `max_tokens` parameter limits total context size
- Test: empty search returns empty context

### R5: Cache Behavior
- Test: first search stores result in Redis
- Test: second identical search returns cached result
- Test: `bypass_cache=True` skips Redis cache
- Test: cache TTL expires after configured seconds

### R6: Duplicate Detection
- Test: ingesting same content twice returns `duplicate=True`
- Test: different content with same hash is treated as duplicate

## Acceptance Criteria
- [ ] All R2-R6 tests pass
- [ ] Tests use real PostgreSQL with pgvector
- [ ] Embeddings are mocked (no Gemini API calls)
- [ ] Test execution time < 20s for full search suite

## Non-Goals
- End-to-end RAG with real LLM generation
- Celery worker tests (mock only)
- Load/stress testing
