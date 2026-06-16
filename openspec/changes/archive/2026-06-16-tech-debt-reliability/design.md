## Context

The project is a production-grade FastAPI + LangChain/LangGraph modular monolith (Python 3.12, uv, asyncpg, motor, Redis, Celery/RabbitMQ, Neo4j/Graphiti, PostgreSQL, pgvector). Current state:

- **Test coverage ~10%** (9 test files for 27 feature modules) — no contract tests, no chaos tests
- **Celery tasks** use string-based names (`"tasks.documents_ingest"`) with zero compile-time safety
- **Health check** at `GET /` returns static JSON — doesn't verify DB/Redis/Neo4j/Graphiti
- **Embedding normalization** hardcodes 768 dimensions (Gemini-specific)
- **Cache concurrency** — hybrid search has thundering-herd risk on cold cache
- **Middleware order** — Guard's CORS + SecurityMiddleware may duplicate CORS headers
- **Correlation IDs** stop at the HTTP boundary — Celery workers lose tracing context
- **No API deprecation path** — v1 routes exist with no sunset strategy

## Goals / Non-Goals

**Goals:**
- Close test gap to 30% on new code, 50% on Tier 1-3 paths (ongoing, no deadline)
- Typed Celery task registry with incremental migration of 9 existing tasks
- Deep health check endpoint for K8s probes
- Embedding dimension as config (model-agnostic)
- Cache deduplication for hybrid search hot paths
- End-to-end correlation IDs (HTTP → Celery → downstream)
- API v1 deprecation headers
- CORS header audit + fix
- Startup dependency verification (fail-fast)

**Non-Goals:**
- Full test rewrite or migration to a new test framework
- Replace Celery with another task queue
- Add v2 API routes (only add deprecation headers to v1)
- Distributed tracing via OTel SDK (only correlation ID propagation)
- Cache invalidation beyond TTL (no event-driven invalidation)
- Chaos testing in production (only staging)

## Decisions

### D1: Test infrastructure — testcontainers over mock-only

**Decision:** Use `testcontainers` for Postgres, Redis, Neo4j integration tests. Factory-boy for data factories. No full mock rewrites of existing tests.

**Rationale:** Mock-only testing misses real query bugs (pgvector operators, Redis protocol, Neo4j Cypher). Testcontainers spin up real services in Docker — fast enough for CI, catches actual integration failures. Factory-boy avoids test data duplication.

**Alternatives considered:**
- *SQLite for tests*: pgvector not available, Cypher queries impossible — rejected
- *Docker Compose CI*: heavier setup, slower teardown — testcontainers is self-managing
- *Mock everything*: catches logic bugs but misses persistence bugs — insufficient for this codebase

### D2: Typed Celery registry — incremental migration, not big-bang

**Decision:** Add a typed registry module (`src/app/connections/celery_registry.py`) that wraps `celery_app.send_task()` with Pydantic payload validation. Migrate one task per PR over time. Old string-based calls remain supported via a `LegacyTaskPayload` fallback.

**Rationale:** All 9 tasks migrated at once creates a huge diff, blocks other work, and risks breaking running workers during rolling deploys. Incremental migration lets us validate each task's typed payload independently.

**Alternatives considered:**
- *Big-bang migration*: high risk, large diff — rejected for rolling deploy safety
- *New tasks only*: leaves 9 existing tasks untyped — accepted as partial for now
- *Replace Celery with something else*: out of scope — Celery works fine

### D3: Health checks — startup-gated + liveness endpoint

**Decision:** Two-tier health:
1. **Startup gate** (in `lifespan.py`): `asyncio.TaskGroup` already verifies PG/Mongo/Redis/Neo4j. Add Graphiti + HTTPX connectivity. App won't start if critical deps fail.
2. **Liveness endpoint** (`GET /health`): lightweight ping to each dependency with 2s timeout per check. Returns per-dependency status. Used by K8s liveness/readiness probes.

**Rationale:** Startup gate prevents "starts but broken" scenarios. Liveness endpoint catches runtime degradation (connection pool exhaustion, network partition). Two separate concerns.

**Alternatives considered:**
- *Single /health only*: doesn't catch startup failures — rejected
- *Deep health with full query*: too slow for liveness probe — use shallow pings only
- *Use existing Prometheus /metrics*: doesn't check dependency connectivity — insufficient

### D4: Cache deduplication — stampede-cache, not Redis lock

**Decision:** Use `stampede-cache` (v0.1.0, MIT, multi-tier) for hybrid search hot paths. Provides `@coalesce(ttl=60)` decorator for in-flight dedup, `distributed_coalesce()` for cross-instance (K8s), and optional pgvector semantic cache.

**Rationale:** `stampede-cache` is built for LLM-heavy backends — exactly this use case. It provides both in-memory coalescing AND distributed Redis-based deduplication via Lua scripts. Already planned in ScaleForge tier-2 observability. The `@coalesce` decorator is zero-boilerplate.

**Alternatives considered:**
- *cacheflight*: zero deps, production-stable, but single-process only — rejected for multi-worker deployment
- *Redis SETNX lock*: adds network round-trip, fails if Redis is down — rejected
- *No dedup*: thundering herd on cold cache — rejected for production traffic
- *DIY asyncio.Lock*: ~30 lines, but no distributed support — insufficient for K8s

### D5: Correlation IDs — ContextVar + Celery signals

**Decision:** Use `contextvars.ContextVar` for HTTP (already works via `request_state`). For Celery: inject `correlation_id` into task kwargs via `task_prerun` signal. Log it in `task_postrun`. No OTel SDK dependency.

**Rationale:** ContextVar is already the pattern in the HTTP layer. Celery signals are the cleanest injection point — no decorator changes needed on existing tasks. No OTel SDK keeps the dependency footprint small.

**Alternatives considered:**
- *OTel SDK propagation*: heavy dependency for a correlation ID — overkill for now
- *ThreadLocal*: not compatible with async — rejected
- *Pass correlation_id explicitly in every task call*: error-prone, requires touching all callers — rejected

### D6: CORS audit — verify, don't restructure

**Decision:** Add a one-time `curl` verification in CI (or manual test) to confirm exactly one `Access-Control-Allow-Origin` header. If duplication is found, disable Guard's CORS helper and let the explicit CORS middleware handle it.

**Rationale:** The actual bug risk is small (Guard may not add duplicate headers). A simple verification is cheaper than restructuring middleware order.

**Alternatives considered:**
- *Remove Guard CORS entirely*: loses Guard's IP-based CORS restrictions — rejected
- *Use only Guard CORS*: loses FastAPI's native CORS middleware — rejected
- *Add a CORS test to CI*: good long-term — included as a stretch goal

### D7: API deprecation headers — middleware, not per-route

**Decision:** Add a middleware that checks `request.url.path.startswith("/api/v1/")` and injects `Deprecation: true`, `Sunset: Sat, 01 Jan 2027 00:00:00 GMT`, `Link: </api/v2/>; rel="successor-version"` headers on the response. Applied once, covers all v1 routes.

**Rationale:** Middleware is the least invasive approach — no per-route decorators needed. Sunset date aligns with ScaleForge tier-5 API versioning plan.

**Alternatives considered:**
- *Per-route decorator*: error-prone, easy to forget — rejected
- *Response model field*: clutters API contract — rejected
- *No sunset headers*: clients have no migration path — rejected

### D8: Embedding dimension — settings field with validation

**Decision:** Add `EMBEDDING_DIMENSION: int = Field(default=768, gt=0)` to `Settings`. Update `_normalize_embedding()` to accept optional dimension parameter, falling back to settings.

**Rationale:** Single-line config change. Prevents silent truncation/padding if model changes. The validation (`gt=0`) catches misconfiguration early.

**Alternatives considered:**
- *Auto-detect from embedding client*: requires extra API call on startup — latency
- *Hardcode per model*: tight coupling — rejected
- *Remove normalization entirely*: embeddings must be same dimension for pgvector — rejected

## Risks / Trade-offs

- **[Testcontainers CI overhead]** Each integration test suite spins up Docker containers (~5-10s startup). **Mitigation:** Share containers across test modules via session-scoped fixtures; limit to integration tests only.
- **[Celery rolling deploy]** Typed registry + old tasks coexist during migration. If worker restart is missed, old tasks still work via legacy path. **Mitigation:** `LegacyTaskPayload` fallback ensures backward compatibility.
- **[stampede-cache dependency]** New dependency (v0.1.0), beta status. **Mitigation:** 60s TTL is conservative; fallback to uncached path on import error; Redis Lua scripts are battle-tested. Monitor for issues in staging.
- **[Sunset header drift]** Sunset date hardcoded in middleware. If v2 timeline shifts, header becomes misleading. **Mitigation:** Make sunset date a `Settings` field so it's environment-configurable.
- **[CORS false positive]** CORS audit may show no duplication (Guard might not add headers). **Mitigation:** The curl test is fast either way — worth verifying.
- **[Health check latency]** Deep health with 2s timeouts per dependency = max 8s if all fail. **Mitigation:** Parallel checks via `asyncio.gather`, fail-fast on first critical failure.

## Migration Plan

1. **Phase 1 — Quick wins (Week 1):** Embedding dimension config, CORS audit curl test, deep health endpoint
2. **Phase 2 — Test infrastructure (Week 2-3):** Testcontainers setup, fixture modules, factory-boy factories, first integration tests
3. **Phase 3 — Celery typing (Week 3-4):** Registry module, migrate `documents_ingest` as proof of concept
4. **Phase 4 — Observability (Week 4-5):** Celery correlation IDs, API deprecation headers, startup verification
5. **Phase 5 — Cache dedup (Week 5-6):** `async-cache-dedupe` integration on hot paths

**Rollback:** Each phase is independently deployable. No breaking changes to API contracts. Test infrastructure changes have zero production impact.

## Open Questions

- Should Pact contract tests target a specific consumer (e.g., mobile app, frontend) or be generic?
- Is `litmus` available in the staging Kubernetes cluster, or do we need to set it up?
- Should the `/health` endpoint require authentication? (K8s probes need unauthenticated access)
- Should we add a `/ready` endpoint separate from `/health` for Kubernetes readiness vs liveness?
