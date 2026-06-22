## Why

Five independent quality issues surfaced during codebase audit. Each is a latent or active production risk:

1. **Hybrid search cache race condition** — `DocumentQueryService.search()` has a TOCTOU window: two concurrent requests for the same query both miss cache, both embed, both search. The second request pays full cost despite the first already computing. In pathological concurrency (burst traffic to a popular query) this multiplies embedding API costs and PG load by the concurrency factor.

2. **Graphiti initialization after TaskGroup** — Lifespan starts PG/Mongo/Redis/Neo4j in parallel via `TaskGroup`, then initialises Graphiti separately. If Neo4j fails inside the TaskGroup, the driver is set to `None` and Graphiti setup crashes downstream with an unhelpful `AttributeError`. Conversely if Graphiti setup fails after TaskGroup success, the rest of the app starts but graph-dependent features silently return empty results. No health probe catches this gap today.

3. **Hardcoded embedding dimension** — `_normalize_embedding()` (defined in `documents/service.py`, duplicated in `retrieval_kb/nodes.py` and `ingestion_kb/nodes.py`) reads `settings.EMBEDDING_DIMENSION` which defaults to 768 (Gemini). Switching to any other embedding model silently truncates or zero-pads every vector. There is no validation that the runtime dimension matches the actual model output.

4. **Celery tasks scattered across string names** — Tasks are defined with `@celery_app.task(name="tasks.documents_ingest")` in `src/tasks/*.py` but invoked via opaque string keys from outbox events (`event_type="tasks.documents_ingest"`) and potentially from `celery_app.send_task()`. No type safety, no IDE navigation from call-site to implementation, no validation that task names exist at registration time.

5. **Middleware CORS conflict** — `SecurityMiddleware.configure_cors()` (FastAPI Guard) adds CORS middleware internally, then `SecurityMiddleware` itself also manages CORS headers in its response processing. The middleware comment block in `main.py` acknowledges this with the comment "CORS (managed by FastAPI Guard's helper)" but the execution order comment above it lists CORSMiddleware as outermost while the actual call `SecurityMiddleware.configure_cors(app=app, config=guard_config)` happens at middleware-registration time (not execution order). This creates risk of duplicate CORS headers and unexpected interaction between Guard's CORS handler and any future manual CORS configuration.

## What Changes

Five independent change sets, each scoped to a single file or concern:

| # | Scope | Files touched |
|---|-------|-------------|
| 1 | `documents/service.py` search cache | +1 file (< 10 lines) |
| 2 | `lifespan.py` Graphiti guard | +1 file (< 5 lines) |
| 3 | Embedding dimension validation | `documents/service.py`, `retrieval_kb/nodes.py`, `ingestion_kb/nodes.py`, `settings.py` (< 20 lines) |
| 4 | Celery task type safety | `celery_registry.py`, `document_tasks.py`, `search_tasks.py`, `auth_email_tasks.py` — minimal |
| 5 | `main.py` middleware order | +1 file, verify-only |

## Capabilities

### New Capabilities
- **search-cache-dedup** — Deduplication lock for concurrent search cache misses, preventing redundant embedding calls.

### Modified Capabilities
- **graphiti-init** — Health-checked Graphiti initialization that emits a clear degraded-state signal on failure.
- **embedding-dimension** — Runtime dimension validation that fails early on model mismatch.
- **celery-task-registry** — Typed task dispatch that validates kwargs via Pydantic at invocation time.
- **middleware-cors-order** — Verified single-source CORS with no duplicate headers.

## Impact

### Affected Code
- `src/app/features/documents/service.py` — `search()` (issue 1), `_normalize_embedding()` (issue 3)
- `src/app/lifecycle/lifespan.py` — Graphiti setup block (issue 2)
- `src/app/shared/langgraph_layer/retrieval_kb/nodes.py` — duplicate `_normalize_embedding()` (issue 3)
- `src/app/shared/langgraph_layer/ingestion_kb/nodes.py` — duplicate `_normalize_embedding()` (issue 3)
- `src/app/config/settings.py` — `EMBEDDING_DIMENSION` validation (issue 3)
- `src/app/connections/celery_registry.py` — send-task wrapper (issue 4)
- `src/tasks/document_tasks.py` (issue 4)
- `src/tasks/search_tasks.py` (issue 4)
- `src/tasks/auth_email_tasks.py` (issue 4)
- `src/app/main.py` — middleware order comment + verification (issue 5)

### Affected APIs
- No breaking changes to HTTP request/response contracts
- Search cache behavior changes only under concurrent requests (transparent improvement)

### Dependencies Added
- None

### Systems
- CI: `uv run ruff check src/ && uv run ty check src/` must pass
- Manual: `curl -H "Origin: http://example.com" -v http://localhost:8000/health` to verify single CORS header
