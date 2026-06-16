# Capability: startup-dependency-verification

## Purpose
Ensure all critical dependencies (PostgreSQL, Redis, Neo4j, Graphiti, HTTPX) are verified at startup before the app accepts traffic.

## Requirements

### R1: Startup Verification in Lifespan
- After `asyncio.TaskGroup` succeeds (all clients created), verify each client is actually connected
- Verification methods:
  - PostgreSQL: `await engine.connect()` + `SELECT 1`
  - Redis: `await redis.ping()`
  - Neo4j: `await driver.verify_connectivity()`
  - Graphiti: `await graphiti.search(query="health", num_results=1)` with timeout
  - HTTPX: no verification needed (created lazily)
- Log per-dependency: `{"dependency": "postgres", "status": "verified", "latency_ms": N}`

### R2: Failure Handling
- **Critical deps** (PostgreSQL, Redis): if verification fails, raise `ServiceUnavailableException` — app won't start
- **Optional deps** (Neo4j, Graphiti): if verification fails, log warning, set `app.state.{dep} = None`, continue
- **Celery**: already handled (optional, non-blocking with 3s timeout)
- **MongoDB**: already verified in TaskGroup

### R3: App State Consistency
- After verification, all `app.state.{dep}` attributes are either:
  - A working client (verification succeeded), or
  - `None` (verification failed, optional dep)
- Feature dependencies check for `None` before using clients
- `getattr(request.app.state, "graphiti", None)` pattern already used — extend to all deps

### R4: Startup Logs
- On success: `{"event": "startup_verified", "postgres": "ok", "redis": "ok", "neo4j": "ok", "graphiti": "ok"}`
- On partial failure: `{"event": "startup_degraded", "postgres": "ok", "redis": "ok", "neo4j": "failed", "graphiti": "skipped"}`
- On critical failure: `{"event": "startup_failed", "postgres": "failed", "error": "connection refused"}`

### R5: Graceful Shutdown Verification
- During shutdown, verify each client was properly closed
- Log per-dependency close status
- Don't fail shutdown if close errors (best-effort)

## Acceptance Criteria
- [ ] App won't start if PostgreSQL is unreachable (raises exception)
- [ ] App won't start if Redis is unreachable (raises exception)
- [ ] App starts with warning if Neo4j is unreachable (optional dep)
- [ ] App starts with warning if Graphiti is unreachable (optional dep)
- [ ] Startup logs show per-dependency verification status
- [ ] `app.state.graphiti` is `None` when Graphiti fails (no AttributeError)

## Non-Goals
- Auto-retry on startup failure (let orchestrator handle restarts)
- Health check endpoint (separate capability: deep-health-checks)
- Connection pool warmup (out of scope)
- Dependency version verification
