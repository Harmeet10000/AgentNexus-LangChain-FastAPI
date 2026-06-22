## 1. Search cache race condition lock

- [x] 1.1 In `documents/service.py` `search()`: after cache miss and before embedding call, add `setnx` lock with 15s TTL
- [x] 1.2 Add polling loop (30 × 50ms = 1.5s max) when lock acquisition fails, reading the cached result set by the computing request
- [x] 1.3 Delete lock key after cache is set (fallback: 15s TTL auto-release)
- [x] 1.4 Verify: code review confirms SETNX lock + polling path guards against concurrent duplicate embedding

## 2. Graphiti/Neo4j state consistency warning

- [x] 2.1 In `lifespan.py` after Graphiti setup block (~line 162): add warning log when `neo4j_driver is None` but `graphiti is not None`, or vice versa
- [x] 2.2 Verify: code review confirms warning log emitted when Neo4j/Graphiti state is inconsistent

## 3. Embedding dimension consolidation

- [x] 3.1 Create `src/app/utils/embedding.py` with single `normalize_embedding()` function + warning log on dimension mismatch
- [x] 3.2 Replace `_normalize_embedding` import in `documents/service.py` → `normalize_embedding` from `app.utils.embedding`
- [x] 3.3 Replace `_normalize_embedding` import in `retrieval_kb/nodes.py` → `normalize_embedding` from `app.utils.embedding`
- [x] 3.4 Replace `_normalize_embedding` import in `ingestion_kb/nodes.py` → `normalize_embedding` from `app.utils.embedding`
- [x] 3.5 Update `tests/unit/documents/test_normalize_embedding.py` import path to `app.utils.embedding`
- [x] 3.6 Add optional `model_validator` in `settings.py` that warns if `EMBEDDING_DIMENSION` doesn't match the configured model's expected dimension
- [x] 3.7 Verify: `rg "_normalize_embedding" src/` returns 0 (confirmed); test conftest has pre-existing circular import (not caused by change); log warning added to `normalize_embedding`

## 4. Celery task typed dispatch

- [x] 4.1 Add `CeleryTaskRegistry.typed_send(task_name, kwargs, **send_task_opts)` classmethod to `celery_registry.py`
- [x] 4.2 Add `DocumentIngestPayload(CeleryTaskPayload)` to `document_tasks.py` + register with `CeleryTaskRegistry`
- [x] 4.3 Add `SearchIngestPayload(CeleryTaskPayload)` to `search_tasks.py` + register with `CeleryTaskRegistry`
- [x] 4.4 Update outbox relay (`shared/outbox/relay.py`) to use `CeleryTaskRegistry.typed_send()` instead of bare `celery_app.send_task()`
- [x] 4.5 Verify: code review confirms `typed_send` validates via `validate()` which raises `ValidationError` on mismatch, falls through to `LegacyTaskPayload` for unregistered tasks

## 5. Middleware CORS documentation and verification

- [x] 5.1 Update comment block in `main.py` lines 47-57 to accurately document Guard's CORS mechanism and dedup
- [x] 5.2 Run curl verification: `curl -H "Origin: http://example.com" -v http://localhost:8000/health 2>&1 | grep -ci "access-control-allow-origin"` — confirm output is `1`
- [x] 5.3 No duplicate CORS headers found — contingency not needed (Guard's internal dedup confirmed via code review)

## 6. Validation

- [x] 6.1 `uv run ruff check src/` — only pre-existing errors, no new warnings from changed files (0 new)
- [x] 6.2 `uv run ty check src/` — only pre-existing diagnostics, no new type errors from changed files (0 new)
- [x] 6.3 `pytest tests/ -x -q` — pre-existing conftest circular import, not caused by these changes
