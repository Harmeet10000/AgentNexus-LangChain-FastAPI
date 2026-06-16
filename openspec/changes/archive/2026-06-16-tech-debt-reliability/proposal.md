## Why

The codebase is production-ready with excellent architecture (modular monolith, async-first, proper layering) but has critical gaps in **test coverage (~10%)**, **observability depth**, **type safety concessions**, and **operational tooling**. These gaps create risk for production incidents, slow down feature velocity, and make refactoring dangerous. This change addresses the highest-impact technical debt across reliability, observability, and developer experience — aligning with the ScaleForge-v2 migration plan's Tier 1-3 priorities.

## What Changes

### Critical (Tier 1 - Reliability)
- **Test infrastructure overhaul**: Pytest fixtures, testcontainers for Postgres/Redis/Neo4j, factory-boy for test data, contract tests (Pact), chaos testing (Litmus) — targeting 30% coverage on new code, 50% on Tier 1-3 paths
- **Typed Celery task registry**: Replace string-based task names (`"tasks.documents_ingest"`) with typed payloads and compile-time validation; incremental migration of 9 existing tasks
- **Deep health checks**: `/health` endpoint verifying Postgres, Redis, Neo4j, Graphiti connectivity — not just static "healthy" response
- **Cache stampede protection**: `stampede-cache` for hybrid search embedding + DB query deduplication with thundering herd prevention

### High (Tier 2 - Observability)
- **Celery correlation IDs**: Propagate `correlation_id` from HTTP request → Celery task → Graphiti/Neo4j for end-to-end tracing
- **Middleware CORS audit**: Verify no duplicate `Access-Control-Allow-Origin` headers from Guard + SecurityMiddleware
- **Embedding dimension as config**: Remove hardcoded 768-dim assumption in `_normalize_embedding()`; make model-agnostic via settings

### Medium (Tier 3 - Security/Operations)
- **API versioning headers**: `Deprecation: true`, `Sunset`, `Link: rel="successor-version"` on `/api/v1/*` responses
- **Structured logging correlation across workers**: ContextVar for HTTP, Celery signals for task lifecycle
- **Dependency health monitoring**: Startup verification that all clients (PG, Redis, Neo4j, Graphiti, HTTPX) are healthy before marking app ready

### Quick Wins (Bundled)
- Embedding dimension configurable via `settings.EMBEDDING_DIMENSION`
- CORS header duplication audit via `curl` test
- Deep health endpoint with per-dependency status
- API versioning deprecation headers on v1 routes

## Capabilities

### New Capabilities
- `test-infrastructure`: Pytest fixtures, testcontainers, factories, contract tests, chaos tests
- `typed-celery-registry`: Typed task payloads, registry, compile-time validation, incremental migration path
- `deep-health-checks`: `/health` endpoint with per-dependency verification (PG, Redis, Neo4j, Graphiti)
- `cache-deduplication`: `stampede-cache` integration for hybrid search and embedding calls with thundering herd prevention
- `celery-correlation-ids`: Correlation ID propagation from HTTP → Celery → downstream services
- `embedding-dimension-config`: Settings-driven embedding dimension, removal of hardcoded 768
- `api-versioning-headers`: Deprecation/Sunset/Link headers on v1 endpoints
- `middleware-cors-audit`: Verification and fix for duplicate CORS headers
- `startup-dependency-verification`: Fail-fast health checks in lifespan before app ready

### Modified Capabilities
- (none — no existing specs in this repo)

## Impact

### Affected Code
- `src/app/lifecycle/lifespan.py` — startup health checks, dependency verification
- `src/app/main.py` — middleware order, health endpoint, versioning headers
- `src/app/config/settings.py` — `EMBEDDING_DIMENSION` setting
- `src/app/features/documents/service.py` — `_normalize_embedding()`, cache dedupe integration
- `src/app/features/documents/dependencies.py` — typed service injection
- `src/tasks/*.py` (9 files) — incremental migration to typed registry
- `src/app/utils/logger.py` — correlation ID ContextVar for Celery
- `src/app/connections/celery.py` — task registry, signal handlers
- `tests/` — new fixture modules, testcontainers config, factory definitions

### Affected APIs
- `GET /health` — new deep health check endpoint (replaces/enhances `GET /`)
- All `/api/v1/*` — add `Deprecation`, `Sunset`, `Link` headers
- No breaking changes to request/response contracts

### Dependencies Added
- `testcontainers` — PostgreSQL, Redis, Neo4j containers for integration tests
- `factory-boy` — test data factories
- `pact-python` — contract testing (consumer-driven)
- `litmus` — chaos testing (dev dependency)
- `stampede-cache[redis]` — cache stampede protection with distributed coalescing
- `opentelemetry-instrumentation-celery` — optional, for correlation propagation

### Systems
- CI/CD: New test stages (contract, chaos), coverage gates (30%/50%)
- K8s: `/health` endpoint for liveness/readiness probes
- Monitoring: Correlation IDs in logs across HTTP + Celery