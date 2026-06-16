## 1. Quick Wins (Week 1)

- [x] 1.1 Add `EMBEDDING_DIMENSION: int = Field(default=768, gt=0)` to `src/app/config/settings.py`
- [x] 1.2 Update `_normalize_embedding()` in `src/app/features/documents/service.py` to accept optional `expected_dim` parameter, falling back to `settings.EMBEDDING_DIMENSION`
- [x] 1.3 Update all call sites of `_normalize_embedding()` to pass dimension from settings
- [x] 1.4 Add unit test for `_normalize_embedding()` with padding, truncation, and exact-match cases
- [x] 1.5 Run CORS audit: `curl -s -D - -H "Origin: https://example.com" http://localhost:5000/ | grep -i "access-control"` — verify exactly one `Access-Control-Allow-Origin` header
- [x] 1.6 If CORS duplicates found: disable Guard's CORS helper in `main.py`, keep only FastAPI's native CORS
- [x] 1.7 Document middleware execution order in `.opencode/instructions/ARCHITECTURE-RULES.md`

## 2. Deep Health Checks (Week 1-2)

- [x] 2.1 Create `HealthResponse` and `DependencyHealth` Pydantic models in `src/app/utils/response_type.py`
- [x] 2.2 Create `src/app/middleware/health_check.py` with `check_postgres()`, `check_redis()`, `check_neo4j()`, `check_graphiti()` async functions (2s timeout each)
- [x] 2.3 Add `GET /health` endpoint in `src/app/main.py` — parallel `asyncio.gather` with `return_exceptions=True`, return `APIResponse[HealthResponse]`
- [x] 2.4 Return HTTP 200 for healthy/degraded, HTTP 503 for unhealthy (3+ deps failed)
- [x] 2.5 Exempt `/health` from API versioning deprecation headers
- [x] 2.6 Add startup verification in `lifespan.py` after TaskGroup: verify PG, Redis, Neo4j, Graphiti connectivity
- [x] 2.7 Critical deps (PG, Redis): raise `ServiceUnavailableException` on failure. Optional deps (Neo4j, Graphiti): log warning, set `app.state.{dep} = None`
- [x] 2.8 Add K8s probe documentation to README (liveness + readiness config)
- [x] 2.9 Write integration test: health endpoint returns correct status when deps are healthy
- [x] 2.10 Write integration test: health endpoint returns degraded/unhealthy when a dep is down

## 3. API Deprecation Headers (Week 2)

- [x] 3.1 Create `src/app/middleware/api_versioning.py` with middleware that checks `request.url.path.startswith("/api/v1/")`
- [x] 3.2 Inject `Deprecation: true`, `Sunset: {date}`, `Link: </api/v2/>; rel="successor-version"` headers
- [x] 3.3 Add `API_SUNSET_DATE` and `API_V2_BASE_PATH` settings to `Settings`
- [x] 3.4 Register middleware in `main.py` after CORS, before routes
- [x] 3.5 Exempt `/health`, `/metrics`, `/api-docs`, `/api-redoc`, `/swagger.json` from deprecation headers
- [x] 3.6 Add `Sunset` and `Deprecation` to CORS `Access-Control-Expose-Headers` list in settings
- [x] 3.7 Add `deprecated: true` to v1 router OpenAPI config
- [x] 3.8 Write test: v1 routes return deprecation headers, exempt paths do not

## 4. Test Infrastructure Setup (Week 2-3)

- [x] 4.1 Add test dependencies: `testcontainers[postgres,redis,neo4j]`, `factory-boy`, `pact-python` to `pyproject.toml`
- [x] 4.2 Create `tests/conftest.py` with session-scoped testcontainer fixtures for PostgreSQL (with pgvector), Redis, Neo4j
- [x] 4.3 Create `tests/factories.py` with `UserFactory`, `DocumentFactory`, `ChunkFactory`, `TaskResultFactory`
- [x] 4.4 Create `tests/integration/__init__.py` and `tests/contract/__init__.py` directories
- [x] 4.5 Add database truncation fixtures (per-test cleanup for PG, Redis flush, Neo4j session clear)
- [x] 4.6 Write integration test: document ingestion pipeline end-to-end (upload → parse → classify → embed → store)
- [x] 4.7 Write integration test: hybrid search (BM25 + vector + trigram → RRF fusion)
- [x] 4.8 Write integration test: RAG pipeline (search → context assembly → grading → generation)
- [x] 4.9 Write integration test: Redis caching (set → get → TTL expiry)
- [x] 4.10 Write contract test for `GET /health` endpoint (consumer: K8s probes)
- [x] 4.11 Write contract test for `POST /documents/upload` endpoint (consumer: frontend)
- [x] 4.12 Write contract test for `POST /search` endpoint (consumer: mobile app)
- [x] 4.13 Configure coverage gates: 30% on new code, 50% on Tier 1-3 paths in CI

## 5. Typed Celery Registry (Week 3-4)

- [x] 5.1 Create `src/app/tasks/payloads.py` with Pydantic payload models: `DocumentIngestPayload`, `EmbedChunksPayload`, `SearchIndexPayload`, `MemoryDecayPayload`, `PageIndexPayload`, `AuthEmailPayload`, `DocumentExtractionPayload`
- [x] 5.2 Create `src/app/connections/celery_registry.py` with `TaskRegistry` class, `register_task()`, `send_typed_task()`, `LegacyTaskPayload` fallback
- [x] 5.3 Add `CELERY_REGISTRY_ENABLED: bool = Field(default=True)` to settings
- [x] 5.4 Create signal handlers in `src/app/connections/celery_signals.py`: `task_prerun`, `task_postrun`, `task_failure`
- [x] 5.5 Register signals in `src/app/connections/celery.py` on worker init
- [x] 5.6 Migrate `documents_ingest` task as proof of concept: update caller in `service.py` to use `send_typed_task()`
- [x] 5.7 Add deprecation warning when tasks called via string name (legacy path)
- [x] 5.8 Write unit test: valid payload passes validation
- [x] 5.9 Write unit test: invalid payload raises `ValidationError`
- [x] 5.10 Write unit test: legacy `send_task()` still works during migration
- [x] 5.11 Run `uv run ty check src/app/connections/celery_registry.py` — verify passes

## 6. Celery Correlation IDs (Week 4-5)

- [x] 6.1 Create `src/app/connections/celery_signals.py` (if not already created in 5.4) with `task_prerun` handler that extracts `correlation_id` from task headers and sets ContextVar
- [x] 6.2 Add `task_postrun` handler that logs task completion with timing and correlation_id
- [x] 6.3 Add `task_failure` handler that logs failure with traceback and correlation_id
- [x] 6.4 Update `send_typed_task()` to auto-inject current `correlation_id` from ContextVar into task headers
- [x] 6.5 Update `graphiti.add_episode()` calls to include `correlation_id` in metadata/source_description
- [x] 6.6 Write test: HTTP request with `X-Correlation-ID: abc123` → Celery task logs show `correlation_id: abc123`
- [x] 6.7 Write test: Celery-initiated tasks generate their own correlation_id
- [x] 6.8 Write test: task failure logs include correlation_id + traceback

## 7. Cache Deduplication (Week 5-6)

- [x] 7.1 Add `stampede-cache[redis]` to `pyproject.toml` dependencies
- [x] 7.2 Add `CACHE_DEDUP_ENABLED`, `CACHE_DEDUP_TTL_SECONDS`, `CACHE_DEDUP_MAX_ENTRIES`, `CACHE_DEDUP_DISTRIBUTED` settings
- [x] 7.3 Create `src/app/shared/cache/dedup.py` with stampede-cache initialization (Redis backend for distributed dedup)
- [x] 7.4 Apply `@coalesce(ttl=...)` decorator to `DocumentQueryService.search()` (BM25 + vector + trigram fan-out)
- [x] 7.5 Apply `@coalesce(ttl=...)` decorator to `_cached_embedding()` (concurrent identical query embeddings)
- [x] 7.6 Apply coalescing to leaderboard top-N reads (if feature exists)
- [x] 7.7 Add graceful fallback: if `stampede-cache` import fails, bypass dedup (uncached path)
- [x] 7.8 Add debug logging for dedup hits/misses via stampede-cache `onDedupe`/`onHit`/`onMiss` callbacks
- [x] 7.9 Write test: 10 concurrent identical search requests execute search exactly once
- [x] 7.10 Write test: failed search doesn't return stale cache to other waiters
- [x] 7.11 Write test: `CACHE_DEDUP_ENABLED=false` disables dedup
- [x] 7.12 Write test: distributed coalescing works across 2+ async tasks via Redis

## 8. Startup Dependency Verification (Week 5-6)

- [x] 8.1 Add verification functions in `lifespan.py`: `verify_postgres()`, `verify_redis()`, `verify_neo4j()`, `verify_graphiti()`
- [x] 8.2 Each verification: `SELECT 1` (PG), `PING` (Redis), `verify_connectivity()` (Neo4j), search query (Graphiti)
- [x] 8.3 Log per-dependency status: `{"dependency": "postgres", "status": "verified", "latency_ms": N}`
- [x] 8.4 Critical deps: raise `ServiceUnavailableException` on failure. Optional deps: set `app.state.{dep} = None`
- [x] 8.5 Add graceful shutdown verification: log per-dependency close status (best-effort, don't fail)
- [x] 8.6 Write test: app won't start if PostgreSQL is unreachable
- [x] 8.7 Write test: app starts with warning if Neo4j is unreachable
- [x] 8.8 Write test: `app.state.graphiti` is `None` when Graphiti fails (no AttributeError)

## 9. Chaos Testing (Week 6+)

> **Deferred** — requires K8s Litmus installation in `chaos-staging` namespace. Track as separate follow-up.

- [ ] 9.1 Install Litmus in `chaos-staging` Kubernetes namespace
- [ ] 9.2 Create Redis-kill experiment: verify app degrades gracefully (uncached reads succeed)
- [ ] 9.3 Create RabbitMQ-pause experiment: verify Celery tasks queue and resume
- [ ] 9.4 Create DB-slow-query experiment: verify timeout handling and circuit breaker
- [ ] 9.5 Create CPU-stress experiment: verify rate limiter and request queuing
- [ ] 9.6 Schedule weekly Monday 09:00 UTC chaos runs
- [ ] 9.7 Document runbook for chaos test failures (P2 incident process)

## 10. Documentation & Cleanup

- [x] 10.1 Update ARCHITECTURE-RULES.md with middleware order diagram
- [ ] 10.2 Update README with health endpoint docs, K8s probe config, test running instructions
- [ ] 10.3 Add `AGENTS.md` entry for chaos testing procedures
- [ ] 10.4 Create `docs/chaos-testing.md` with experiment descriptions and runbook
- [x] 10.5 Verify all `uv run ruff check src/` and `uv run ty check src/` pass
- [ ] 10.6 Verify all tests pass: `uv run pytest tests/`
