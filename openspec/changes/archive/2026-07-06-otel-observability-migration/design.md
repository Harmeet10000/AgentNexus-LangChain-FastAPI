## Context

The project has three signal types (logs, metrics, traces) split across ad-hoc tooling with no correlation.

**Current state:**
- `loguru` for structured logging with `request_state`/`execution_path` ContextVars, console-only output
- `prometheus_client` with `MetricsMiddleware` defined but **commented out** in `main.py:88`
- Separate `MCPObservabilityMiddleware` + `mcp_core/common/metrics.py` for MCP sub-app
- `LangSmith` for LLM-only tracing (kept as-is, not replaced)
- `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-sqlalchemy` already in `pyproject.toml` but **not wired**
- OTel settings defined in `settings.py` but unused
- `trace_layer` decorator uses `time.perf_counter()` + `execution_path` ContextVar (breadcrumbs, not spans)
- Celery workers have signal-based lifecycle logging but no traces or metrics

**Target state:**
- Single OTel pipeline shared across FastAPI, MCP (separate uvicorn), and Celery workers
- Export to self-hosted SigNoz via OTLP (gRPC)
- `/metrics` endpoint serves Prometheus text via OTel Prometheus exporter (same contract, different source)
- Logs correlated to traces via `trace_id`/`span_id` in every log record
- `trace_layer` decorator creates real OTel child spans instead of breadcrumbs
- Full auto-instrumentation: SQLAlchemy, Redis, HTTPX, Celery, ASGI

## Goals / Non-Goals

**Goals:**
- Single OTel bootstrap module shared by all 3 processes (FastAPI, MCP, Celery)
- Every HTTP request and Celery task has a trace root span visible in SigNoz
- Every log line carries its parent `trace_id` and `span_id`
- Metrics exported to both SigNoz (OTLP) and Prometheus format (`/metrics` endpoint)
- `trace_layer` decorator creates real child spans — same API, zero call-site changes
- Auto-instrumentation covers SQLAlchemy, Redis, HTTPX, Celery, ASGI
- LangSmith remains for LLM-specific deep debugging (side-by-side)
- Obsolete code removed: `MetricsMiddleware`, `MCPObservabilityMiddleware`, `mcp_core/common/metrics.py`

**Non-Goals:**
- Replacing LangSmith for LLM trace detail
- Adding file-based JSON log output
- Instrumenting third-party services (Neo4j, MongoDB, Tavily, Crawl4AI) in this change
- Tail-based sampling or OTel collector deployment (head-based sampling at SDK level, push everything)
- Performance optimization beyond OTel defaults (batch sizes, export intervals)
- Migrating existing loguru call sites (they work unchanged — the sink addition is transparent)

## Architecture

### Process Architecture

Three processes share the same OTel bootstrap module `src/app/shared/otel/`:

```
┌──────────────────────────────────────────────────────┐
│  FastAPI (uvicorn :8080)                             │
│  service.name = "langchain-fastapi"                  │
│  ├─ OTel ASGI middleware (outermost)                  │
│  ├─ RequestStateLoggingMiddleware (trace_id inj.)     │
│  ├─ RateLimitMiddleware                              │
│  ├─ SecurityMiddleware                               │
│  └─ AuthMiddleware                                   │
│  Auto-instrumentation: SQLAlchemy, Redis, HTTPX       │
├──────────────────────────────────────────────────────┤
│  MCP (uvicorn :8081)                                 │
│  service.name = "langchain-fastapi-mcp"              │
│  ├─ OTel ASGI middleware (outermost)                  │
│  ├─ MCPAuthMiddleware                                │
│  └─ MCPRateLimitMiddleware                           │
│  Tool calls: manual child spans + OTel metrics       │
├──────────────────────────────────────────────────────┤
│  Celery Worker                                       │
│  service.name = "langchain-fastapi-celery"           │
│  ├─ CeleryInstrumentor (auto task/apply_async spans) │
│  ├─ ResilientTask emits OTel metrics                 │
│  └─ Signal handlers include trace_id in logs         │
└──────────────────────────────────────────────────────┘
```

### Module Structure

```
src/app/shared/otel/
├── __init__.py          # setup_otel(), shutdown_otel() — public API
├── tracer.py            # _setup_tracer_provider(), sampler config
├── metrics.py           # _setup_meter_provider(), PrometheusMetricExporter
├── logs.py              # _setup_logger_provider(), _patch_loguru_sink()
└── instrument.py        # _setup_auto_instrumentation() — all instrumentors

src/app/shared/
└── otel_integrations.py # get_prometheus_metrics() — serves /metrics endpoint
```

### Middleware Ordering (FastAPI)

OTel ASGI middleware must be registered LAST (outermost) to capture all requests:

```
app.add_middleware(RateLimitMiddleware, ...)     # innermost
app.add_middleware(SecurityMiddleware, ...)
app.add_middleware(AuthMiddleware, ...)
app.add_middleware(RequestStateLoggingMiddleware, prefix=settings.API_V1_PREFIX)
app.add_middleware(OpenTelemetryMiddleware,      # outermost — added last
    default_span_details=_otel_span_details,
    excluded_urls="healthz,readyz,metrics")
```

Middleware chain in request order:
1. OTel ASGI middleware — creates root span, captures all traffic
2. RequestStateLoggingMiddleware — injects trace_id into console logs
3. Auth/Security — authentication checks
4. RateLimit — rate limiting
5. FastAPI routing + endpoints:
   - Service/business logic (trace_layer creates child spans)
   - Auto-instrumented DB/cache/HTTP calls (grandchild spans)

### Dual Metric Export

```
PROMETHEUS PATH (sync, on scrape):
  OTel MeterProvider
    → PrometheusMetricExporter (collects on /metrics scrape)
    → FastAPI route: get_prometheus_metrics()
    → Response(content=..., media_type="text/plain; version=0.0.4; charset=utf-8")

OTLP PATH (async, periodic):
  OTel MeterProvider
    → PeriodicExportingMetricReader(OTLPMetricExporter())
    → gRPC → SigNoz
```

## Decisions

1. **Shared bootstrap module over per-process init**
   - Why: all three processes need the same pipeline — same endpoint, same resource attributes, same instrumentors. A single `setup_otel()` keeps configuration in one place.
   - Alternatives considered:
     - Each process imports the same module and calls `setup_otel()` — works but `Resource.service.name` must differ per process. Solved by an optional `service_name` parameter.
     - env-var-driven auto-instrumentation via `opentelemetry-distro` — less explicit, harder to customize sink behavior.

2. **Dual metric export (OTLP + Prometheus)**
   - Why: SigNoz ingests OTLP natively for dashboards. k8s/HashiCorp Nomad scrape `/metrics` for Prometheus format. Both are required.
   - Alternatives considered:
     - OTLP only — breaks existing `/metrics` consumers and k8s scraping
     - Prometheus only — SigNoz can scrape but OTLP is the native path with richer metadata

3. **loguru → OTLP bridge via custom sink**
   - Why: replacing loguru with stdlib logging would touch hundreds of `logger.bind(...)` calls. A loguru sink is invisible to existing code.
   - Alternatives considered:
     - Replace loguru with Python logging + OTel handler — invasive, requires rewriting all structured logging patterns
     - JSON file + Vector/Fluentd sidecar converting to OTel — adds infrastructure complexity for no benefit in single-machine deployments

4. **OTel ASGI middleware over FastAPI middleware**
   - Why: ASGI middleware runs at the protocol level, capturing all traffic before any FastAPI middleware. The OTel package (`opentelemetry-instrumentation-asgi`) handles this correctly.
   - Alternatives considered:
     - FastAPI `@app.middleware("http")` — misses non-HTTP scopes (WebSocket) and runs after some framework internals
     - Manual span creation in `RequestStateLoggingMiddleware` — duplicates what the OTel package already provides

5. **Full auto-instrumentation over manual spans**
   - Why: user chose full auto-instrumentation. `SQLAlchemyInstrumentor`, `RedisInstrumentor`, `HTTPXClientInstrumentor`, and `CeleryInstrumentor` require zero per-call-site changes.
   - Alternatives considered:
     - Manual spans in service/repository layers — more control but 50+ files to touch
     - Skip instrumentation — loses DB/cache/HTTP call visibility

6. **PrometheusMetricExporter over direct prometheus_client**
   - Why: OTel's `PrometheusMetricExporter` renders OTel instruments as Prometheus text. This means code uses `meter.create_counter(...)` instead of `prometheus_client.Counter(...)`, and the `/metrics` endpoint reads from the exporter.
   - Alternatives considered:
     - Keep `prometheus_client` for `/metrics` + OTel for everything else — dual metric APIs, confusing
     - Remove `/metrics` entirely — breaks existing monitoring integrations

7. **CeleryInstrumentor for task spans**
   - Why: auto-instruments task execution, creating `celery.{task_name}` spans with duration, status, and exception recording.
   - Alternatives considered:
     - Manual spans in `ResilientTask.on_success/on_failure` — more work, easier to miss edge cases
     - Skip task tracing — no visibility into worker execution

8. **Instrumentor graceful degradation**
   - Why: if an instrumentor dependency is missing, the app should warn and continue instead of crashing at startup.
   - Implementation: each `instrument()` call is wrapped in try/except with `logger.warning(...)`. Remaining instrumentors still register.

9. **Shutdown force_flush for all providers**
   - Why: OTel buffers spans, metrics, and logs asynchronously. Without `force_flush()` during shutdown, the last ~5s of data is lost on every process restart.
   - Called in both FastAPI and MCP lifespan `shutdown` handlers, and via Celery `worker_shutting_down` signal.

10. **Sampling configuration**
    - Why: user wants to push everything (sample_rate=1.0) but keep the knob for future adjustment.
    - `ParentBased(TraceIdRatioBased(base_rate=settings.OTEL_SAMPLE_RATE))` — preserves traces that already have a parent span.

## Risks / Trade-offs

- **OTLP endpoint dependency**: If SigNoz is down, spans buffer in memory (default `BatchSpanProcessor` queue). Mitigated by bounded queue size and export timeout. If the queue fills, older spans are dropped — no request failure.
- **Log volume to OTel**: Every log line is exported via OTLP. In production, this could be significant volume. Mitigated by log level filtering and `OTEL_ENABLED=False` kill switch.
- **Metric cardinality**: `http.route` with UUID/numeric-ID normalization reduces cardinality for path-based metrics. But `mcp.tool.name` could still grow. Mitigated by keeping tool count bounded (tens, not thousands).
- **Celery instrumentor overhead**: Auto-instrumentation wraps every task execution. For high-throughput task queues, span creation overhead may be measurable. Mitigated by `OTEL_ENABLED` and sampling controls.
- **Trace context propagation**: auto-instrumentors handle `traceparent` header propagation for HTTP. For RabbitMQ/Celery message propagation, context passes through message headers automatically (CeleryInstrumentor handles this).
- **Dual metric export consistency**: OTel instruments feed both readers via the same aggregation. Prometheus exporter collects on scrape (synchronous), OTLP exports on interval (asynchronous). Small aggregation window differences are expected and acceptable.
- **Loguru multiprocessing**: Each Celery worker process calls `setup_otel()` independently, creating its own `LoggerProvider` and OTel sink. No shared state issues.

## Missing Requirements (from gap analysis)

The following requirements were identified during spec review against the final plan:

1. **Idempotency flag on setup_otel()**: Prevents re-initialization if called multiple times (e.g., during tests). A module-level `_otel_initialized: bool = False` flag gates all provider creation.
2. **PrometheusMetricExporter registration**: The OTel meter provider needs a `PrometheusMetricExporter` registered as a reader (synchronous). Not mentioned in original specs.
3. **Instrumentor graceful degradation**: Each `instrument()` call wrapped in try/except with `logger.warning()`. If one fails, the rest still register.
4. **BatchSpanProcessor configuration**: `max_export_batch_size=512`, `schedule_delay_millis=5000`, `max_queue_size=2048`. Specified values for all three.
5. **OTEL_SAMPLE_RATE env var**: Controls `ParentBased(TraceIdRatioBased(base_rate=))`. Defaults to 1.0.
6. **force_flush() on shutdown**: Must be called for TracerProvider, MeterProvider, and LoggerProvider during lifespan shutdown.
7. **Middleware ordering for MCP**: OTel ASGI must be outermost in the MCP middleware stack too.
8. **MCP tool spans as child spans**: `_execute_tool()` needs explicit `start_as_current_span` with attributes `mcp.tool.name`, `mcp.tool.status`.
9. **MCP client spans**: Outbound upstream calls need client spans with server/tool attributes.
10. **Upstream health as OTel gauge**: `set_mcp_upstream_health()` becomes an OTel up-down counter, not a `prometheus_client.Gauge`.
11. **Celery signal handler trace_id**: `log_task_prerun`/`postrun`/`failure` must read OTel span context and include `trace_id` in loguru bindings.
12. **ResilientTask OTel metrics**: `on_success`/`on_failure`/`on_retry` must increment `celery.task.completed_total` and record `celery.task.duration_seconds`.
13. **trace_layer sync function support**: The decorator must handle both async and sync functions via `inspect.iscoroutinefunction()`.
14. **execution_path ContextVar removal**: The `execution_path` ContextVar and `KEEP_EXECUTION_PATH_LENGTH` constant are deleted when `trace_layer` switches to OTel.
15. **SUCCESS level mapping**: Loguru SUCCESS → OTel `SeverityNumber.INFO` (9).
16. **Console trace_id display**: The console format string is updated to include `trace_id` when available.
17. **setup_logging() order preserved**: `setup_logging()` called first (console) → `setup_otel()` called second (OTel sink).
18. **OTEL_LOGS_EXPORTER="none" support**: If set, no LoggerProvider is created.
19. **OTEL_ENABLED flag**: If False, `setup_otel()` returns immediately — no providers, no instrumentors.
20. **Resource with version and environment**: `service.version` from `APP_VERSION` settings, `deployment.environment` from `ENVIRONMENT` settings.

## Migration Plan

1. **Dependencies**: `uv add opentelemetry-instrumentation-asgi opentelemetry-instrumentation-httpx opentelemetry-instrumentation-redis opentelemetry-instrumentation-celery opentelemetry-exporter-prometheus opentelemetry-sdk-logs`
2. **Settings**: Add `OTEL_ENABLED`, `OTEL_LOGS_EXPORTER`, `OTEL_SAMPLE_RATE` to `src/app/config/settings.py`
3. **Bootstrap module**: Create `src/app/shared/otel/{__init__,tracer,metrics,logs,instrument}.py` + `src/app/shared/otel_integrations.py`
4. **FastAPI**: Edit `main.py` (call `setup_otel()`, add `/metrics` route), edit `server_middleware.py` (rewrite `get_metrics`, remove `MetricsMiddleware`), edit middleware `__init__.py`
5. **MCP**: Edit `mcp_core/lifespan_mcp.py` (call `setup_otel()`), edit `mcp_core/server/middleware.py` (replace `MCPObservabilityMiddleware` with OTel ASGI), edit `mcp_core/server/tools.py` (replace `observe_mcp_tool_invocation`), edit `mcp_core/client/manager.py` (replace client observers + health gauge), delete `mcp_core/common/metrics.py`
6. **Celery**: Edit `src/app/connections/celery.py` (call `setup_otel()`, add trace_id to signal handlers, add OTel metrics in ResilientTask)
7. **Logging**: Edit `src/app/utils/logger.py` — rewrite `trace_layer` decorator, add console `trace_id` format, remove JSON file handler
8. **Cleanup**: Remove commented-out `SQLAlchemyInstrumentor` from `postgres.py`, verify no remaining `prometheus_client` imports outside OTel bridge
9. **Verify**: Run SigNoz docker-compose, hit endpoints, confirm traces/metrics/logs appear

## Open Questions

- ~~Should `setup_otel()` accept a `service_name` parameter, or read a per-process env var?~~ Decision: accept optional `service_name: str | None = None` — if None, use `settings.OTEL_SERVICE_NAME`.
- ~~Should the Prometheus exporter use the default registry or a custom one?~~ Decision: use OTel-managed export, not `prometheus_client` registry directly.
- ~~Should the MCP process share the same FastAPI app or be a separate uvicorn?~~ Decision: separate uvicorn process (existing architecture, no change needed).
