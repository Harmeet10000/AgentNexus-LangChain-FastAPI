## 1. Search cache race condition lock

- [ ] 1.1 In `documents/service.py` `search()`: after cache miss and before embedding call, add `setnx` lock with 15s TTL
- [ ] 1.2 Add polling loop (30 × 50ms = 1.5s max) when lock acquisition fails, reading the cached result set by the computing request
- [ ] 1.3 Delete lock key after cache is set (fallback: 15s TTL auto-release)
- [ ] 1.4 Verify: concurrent `search()` calls with same query → single embedding call (unit test or log inspection)

## 2. Graphiti/Neo4j state consistency warning

- [ ] 2.1 In `lifespan.py` after Graphiti setup block (~line 162): add warning log when `neo4j_driver is None` but `graphiti is not None`, or vice versa
- [ ] 2.2 Verify: startup log shows "State inconsistency" when Neo4j is down but Graphiti starts, or vice versa

## 3. Embedding dimension consolidation

- [ ] 3.1 Create `src/app/utils/embedding.py` with single `normalize_embedding()` function + warning log on dimension mismatch
- [ ] 3.2 Replace `_normalize_embedding` import in `documents/service.py` → `normalize_embedding` from `app.utils.embedding`
- [ ] 3.3 Replace `_normalize_embedding` import in `retrieval_kb/nodes.py` → `normalize_embedding` from `app.utils.embedding`
- [ ] 3.4 Replace `_normalize_embedding` import in `ingestion_kb/nodes.py` → `normalize_embedding` from `app.utils.embedding`
- [ ] 3.5 Update `tests/unit/documents/test_normalize_embedding.py` import path to `app.utils.embedding`
- [ ] 3.6 Add optional `model_validator` in `settings.py` that warns if `EMBEDDING_DIMENSION` doesn't match the configured model's expected dimension
- [ ] 3.7 Verify: `rg "_normalize_embedding" src/` returns 0; `pytest tests/unit/documents/test_normalize_embedding.py -x` passes; log warning visible when dimension mismatch

## 4. Celery task typed dispatch

- [ ] 4.1 Add `CeleryTaskRegistry.typed_send(task_name, kwargs, **send_task_opts)` classmethod to `celery_registry.py`
- [ ] 4.2 Add `DocumentIngestPayload(CeleryTaskPayload)` to `document_tasks.py` + register with `CeleryTaskRegistry`
- [ ] 4.3 Add `SearchIngestPayload(CeleryTaskPayload)` to `search_tasks.py` + register with `CeleryTaskRegistry`
- [ ] 4.4 Update outbox relay (`shared/outbox/relay.py`) to use `CeleryTaskRegistry.typed_send()` instead of bare `celery_app.send_task()`
- [ ] 4.5 Verify: unit test `CeleryTaskRegistry.typed_send("tasks.documents_ingest", valid_kwargs)` succeeds; unit test with invalid kwargs raises ValidationError; unit test with unregistered task falls through to LegacyTaskPayload

## 5. Middleware CORS documentation and verification

- [ ] 5.1 Update comment block in `main.py` lines 47-57 to accurately document Guard's CORS mechanism and dedup
- [ ] 5.2 Run curl verification: `curl -H "Origin: http://example.com" -v http://localhost:8000/health 2>&1 | grep -ci "access-control-allow-origin"` — confirm output is `1`
- [ ] 5.3 If duplicate CORS headers found (output > 1), add `del response.headers["access-control-allow-origin"]` in `SecurityMiddleware` response handler AFTER checking Guard version for upgrade safety — this is a contingency, unlikely to be needed

## 6. Validation

- [ ] 6.1 `uv run ruff check src/` — no new warnings
- [ ] 6.2 `uv run ty check src/` — no new type errors
- [ ] 6.3 `pytest tests/ -x -q` — all existing tests pass
