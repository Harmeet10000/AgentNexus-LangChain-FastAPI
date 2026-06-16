# Capability: deep-health-checks

## Purpose
Replace static health response with a deep dependency verification endpoint suitable for Kubernetes liveness/readiness probes.

## Requirements

### R1: Health Endpoint
- Route: `GET /health`
- Response model: `APIResponse[HealthResponse]`
- No authentication required (K8s probes need unauthenticated access)
- Parallel dependency checks via `asyncio.gather(return_exceptions=True)`
- Per-dependency timeout: 2s

### R2: HealthResponse Model
```python
class DependencyHealth(BaseModel):
    name: str          # "postgres", "redis", "neo4j", "graphiti", "httpx"
    status: str        # "healthy", "degraded", "unhealthy"
    latency_ms: float  # response time in milliseconds
    error: str | None  # error message if unhealthy

class HealthResponse(BaseModel):
    status: str                        # "healthy", "degraded", "unhealthy"
    version: str                       # app version from settings
    uptime_seconds: float              # time since app started
    dependencies: list[DependencyHealth]
```

### R3: Dependency Checks
- **PostgreSQL**: `SELECT 1` via asyncpg pool
- **Redis**: `PING` via redis.asyncio
- **Neo4j**: `driver.verify_connectivity()`
- **Graphiti**: `graphiti.search(query="health", num_results=1)` with timeout
- **HTTPX**: HEAD request to a configurable health URL (default: internal `/`)

### R4: Status Logic
- `healthy`: all dependencies respond within 2s
- `degraded`: 1-2 dependencies fail or respond slowly (>1s)
- `unhealthy`: 3+ dependencies fail
- Return HTTP 200 if healthy/degraded, HTTP 503 if unhealthy

### R5: Startup Verification
- In `lifespan.py`, after `TaskGroup` succeeds, verify all clients are actually connected
- Log per-dependency status at startup
- If critical deps (PostgreSQL, Redis) fail at startup, raise `ServiceUnavailableException`
- If optional deps (Neo4j, Graphiti) fail, log warning and continue

### R6: K8s Probe Configuration
- Document recommended K8s probe config:
  ```yaml
  livenessProbe:
    httpGet:
      path: /health
      port: 5000
    initialDelaySeconds: 10
    periodSeconds: 15
    timeoutSeconds: 5
    failureThreshold: 3
  readinessProbe:
    httpGet:
      path: /health
      port: 5000
    initialDelaySeconds: 5
    periodSeconds: 10
    timeoutSeconds: 3
  ```

## Acceptance Criteria
- [ ] `GET /health` returns per-dependency status with latency
- [ ] HTTP 200 when healthy/degraded, HTTP 503 when unhealthy
- [ ] Each dependency check times out at 2s (doesn't hang)
- [ ] Startup logs show per-dependency health status
- [ ] K8s probes documented in README or ops runbook

## Non-Goals
- Authentication on health endpoint
- Deep query testing (only connectivity pings)
- Prometheus metrics on health endpoint (separate `/metrics`)
- Auto-remediation of unhealthy dependencies
