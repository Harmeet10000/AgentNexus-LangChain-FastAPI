# Capability: embedding-dimension-config

## Purpose
Make embedding dimension configurable instead of hardcoded 768, enabling model-agnostic vector operations.

## Requirements

### R1: Settings Field
- Add `EMBEDDING_DIMENSION: int = Field(default=768, gt=0)` to `Settings`
- Validation: must be positive integer
- Default 768 preserves current behavior (Gemini embedding-2-preview)

### R2: Embedding Normalization Update
- Update `_normalize_embedding(embedding: list[float], expected_dim: int | None = None)`
- If `expected_dim` is None, read from `get_settings().EMBEDDING_DIMENSION`
- If embedding length > expected_dim: truncate
- If embedding length < expected_dim: pad with zeros
- If embedding length == expected_dim: return as-is

### R3: Document Service Integration
- `DocumentQueryService.search()`: pass dimension from settings to `_normalize_embedding()`
- `DocumentQueryService.ask()`: pass dimension from settings to `_normalize_embedding()`
- `process_document_ingestion()`: pass dimension from settings during embedding

### R4: Migration Safety
- Existing vectors in pgvector are 768-dim (Gemini)
- Changing `EMBEDDING_DIMENSION` without re-embedding creates dimension mismatch
- Add startup check: if new dimension differs from existing vectors, log warning
- Document: "Changing EMBEDDING_DIMENSION requires re-indexing all documents"

### R5: Type Safety
- `_normalize_embedding()` signature: `def _normalize_embedding(embedding: list[float], expected_dim: int | None = None) -> list[float]`
- No `Any` types — dimension is always `int`
- `ty` should pass on all call sites

## Acceptance Criteria
- [ ] `EMBEDDING_DIMENSION=1024` in .env produces 1024-dim normalized embeddings
- [ ] Default 768 preserves current behavior (no regression)
- [ ] Changing dimension without re-embedding logs a warning at startup
- [ ] `uv run ty check src/app/features/documents/service.py` passes
- [ ] Unit test: `_normalize_embedding([1.0]*512, expected_dim=768)` returns 512 values + 256 zeros

## Non-Goals
- Auto-detect dimension from embedding client API
- Re-indexing pipeline for dimension changes
- Multi-dimension support (one dimension per deployment)
- Embedding model selection logic
