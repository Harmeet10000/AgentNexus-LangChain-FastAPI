## 1. Prerequisites — Add Dependencies

- [x] 1.1 Add 7 new OTel instrumentation dependencies to `pyproject.toml`:
  - `opentelemetry-instrumentation-asgi>=0.60b0`
  - `opentelemetry-instrumentation-httpx>=0.60b0`
  - `opentelemetry-instrumentation-redis>=0.60b0`
  - `opentelemetry-instrumentation-celery>=0.60b0`
  - `opentelemetry-exporter-prometheus>=0.60b0`
  - `opentelemetry-sdk-logs>=0.60b0`
- [x] 1.2 Run `uv sync` to install new dependencies
- [x] 1.3 Add 3 new settings fields to `src/app/config/settings.py`:
  - `OTEL_ENABLED: bool = Field(default=True)` — global kill switch
  - `OTEL_LOGS_EXPORTER: str = Field(default="otlp")` — set to `"none"` to disable log export
  - `OTEL_SAMPLE_RATE: float = Field(default=1.0, ge=0.0, le=1.0)` — trace ID ratio for sampling

## 2. Create Shared OTel Bootstrap Module

- [x] 2.1 Create `src/app/shared/otel/` package directory with `__init__.py`
- [x] 2.2 Create `src/app/shared/otel/__init__.py`:
  - Define module-level `_otel_initialized: bool = False` idempotency flag
  - Define `setup_otel(service_name: str | None = None)` — returns immediately if `_otel_initialized` is True
  - Create `Resource` with `service.name`, `service.version` (from `APP_VERSION`), `deployment.environment` (from `ENVIRONMENT`), merged with `OTEL_RESOURCE_ATTRIBUTES` env-var overrides
  - Call `_setup_tracer_provider(resource)`, `_setup_meter_provider(resource)`, `_setup_logger_provider(resource)`, `_setup_auto_instrumentation()`
  - Call `_patch_loguru_sink(logger_provider)`
  - Set `_otel_initialized = True`
  - Define `shutdown_otel()` — calls `force_flush()` on all providers, then `shutdown()`
  - Export `get_tracer(name)`, `get_meter(name)`, `get_logger(name)` convenience functions
  - Handle `OTEL_ENABLED=False` by returning immediately from `setup_otel()`
- [x] 2.3 Create `src/app/shared/otel/tracer.py`:
  - `_setup_tracer_provider(resource: Resource)` function
  - Creates `TracerProvider(resource=resource)` with `BatchSpanProcessor(OTLPSpanExporter(insecure=True))`
  - BatchSpanProcessor config: `max_export_batch_size=512`, `schedule_delay_millis=5000`, `max_queue_size=2048`
  - Sampler: `ParentBased(TraceIdRatioBased(base_rate=settings.OTEL_SAMPLE_RATE))`
  - Set as global: `trace.set_tracer_provider(provider)`
  - Returns the provider for shutdown
- [x] 2.4 Create `src/app/shared/otel/metrics.py`:
  - `_setup_meter_provider(resource: Resource) -> MeterProvider` function
  - Creates `MeterProvider(resource=resource)` with two readers:
    - `PeriodicExportingMetricReader(OTLPMetricExporter(insecure=True))` for SigNoz
    - `PrometheusMetricExporter()` (synchronous, collects on `/metrics` scrape) — exports `PrometheusMetricExporter` instance
  - Set as global: `metrics.set_meter_provider(provider)`
  - Define `get_prometheus_exporter() -> PrometheusMetricExporter` — returns the exporter instance for `/metrics` endpoint
  - Handle `PrometheusMetricExporter` import failing gracefully (log warning, skip registration)
  - Returns `(provider, prometheus_exporter)` tuple
- [x] 2.5 Create `src/app/shared/otel/logs.py`:
  - `_setup_logger_provider(resource: Resource) -> LoggerProvider | None` function
  - If `settings.OTEL_LOGS_EXPORTER == "none"`, return `None`
  - Creates `LoggerProvider(resource=resource)` with:
    - `BatchLogRecordProcessor(OTLPLogExporter(insecure=True), max_export_batch_size=512, schedule_delay_millis=5000, max_queue_size=2048)`
  - Set as global: `logs.set_logger_provider(provider)`
  - Define `_patch_loguru_sink(logger_provider: LoggerProvider)` — adds loguru sink that converts each record to OTel `LogRecord`
  - Sink function:
    - Reads `trace_id`/`span_id` from current OTel span context
    - Maps severity: TRACE→TRACE2, DEBUG→DEBUG, INFO→INFO, SUCCESS→INFO, WARNING→WARN, ERROR→ERROR, CRITICAL→FATAL
    - Sets `severity_text` to original level name (e.g., "SUCCESS")
    - Converts loguru extra dict to OTel attributes
    - Emits via `logger_provider.emit(OTelLogRecord(...))`
  - Sink idempotency: remove old sink if re-adding
  - Returns provider
- [x] 2.6 Create `src/app/shared/otel/instrument.py`:
  - `_setup_auto_instrumentation()` function
  - Each instrumentor wrapped in try/except with `logger.warning("... failed — continuing")`
  - Register in order:
    - `SQLAlchemyInstrumentor().instrument()` (engine=None to auto-discover)
    - `RedisInstrumentor().instrument()`
    - `HTTPXClientInstrumentor().instrument()`
    - `CeleryInstrumentor().instrument()`
    - `ASGIInstrumentor().instrument()`
- [x] 2.7 Create `src/app/shared/otel_integrations.py`:

## 3. Wire into FastAPI

- [x] 3.1 In `src/app/main.py`:
  - Import `setup_otel`, `shutdown_otel` from shared module
  - Import `get_prometheus_metrics` from `src.app.shared.otel_integrations`
  - Call `setup_otel(service_name="langchain-fastapi")` after `configure_langsmith()` and before `create_app()`
  - Guard with `if settings.OTEL_ENABLED`
  - Update `/metrics` endpoint to use `get_prometheus_metrics()`
  - In lifespan shutdown: call `shutdown_otel()` using `try/finally` to ensure flush even on exception
  - Remove commented-out `MetricsMiddleware` import and `app.add_middleware(MetricsMiddleware)` lines if present
- [x] 3.2 Create `_otel_span_details` hook in `src/app/middleware/otel.py`:
  - Defines `default_span_details(scope: dict) -> tuple[str, trace.SpanKind]` for OTel ASGI middleware
  - Normalizes UUID and numeric path segments to `{id}` before setting `http.route`
  - Returns `(normalized_path, SpanKind.SERVER)`
  - See `_normalize_path_otel()` implementation
- [x] 3.3 Update `src/app/middleware/server_middleware.py`:
  - Remove all `prometheus_client` imports and module-level metric definitions
  - Remove `MetricsMiddleware` class entirely
  - Remove `_normalize_path()` function
  - Remove `get_metrics()` function (replaced by `get_prometheus_metrics()` in shared module)
  - Update `RequestStateLoggingMiddleware`:
    - Import `opentelemetry.trace`
    - After creating request state, extract `trace_id` hex from `opentelemetry.trace.get_current_span()`
    - Include `trace_id` in `logger.contextualize()` call when available
- [x] 3.4 Update `src/app/middleware/__init__.py`:
  - Remove `MetricsMiddleware` from imports and `__all__`
  - Remove `get_metrics`, `metrics_registry` exports
  - Keep all other middleware exports
- [x] 3.5 Add OTel ASGI middleware to `main.py` middleware stack:
  - Import `OpenTelemetryMiddleware` from `opentelemetry.instrumentation.asgi`
  - Import `default_span_details` from the OTel hook module
  - Add middleware LAST (outermost) in `create_app()`
  - Add comment explaining: "OTel ASGI must be outermost — captures all traffic before FastAPI middleware"

## 4. Wire into MCP Server

- [x] 4.1 In `src/mcp_core/lifespan_mcp.py`:
  - Import `setup_otel`, `shutdown_otel` from shared module
  - Call `setup_otel(service_name="langchain-fastapi-mcp")` at the start of `serve_mcp()` before `get_mcp_http_app()`
  - Handle `settings.OTEL_ENABLED=False` by skipping initialization
- [x] 4.2 In `src/mcp_core/server/middleware.py`:
  - Remove `from mcp_core.common.metrics import observe_mcp_http_request`
  - Remove `MCPObservabilityMiddleware` class entirely
  - In `build_mcp_http_middleware()`, add `Middleware(OpenTelemetryMiddleware)` as the first entry (outermost)
- [x] 4.3 In `src/mcp_core/server/tools.py`:
  - Remove `from mcp_core.common.metrics import observe_mcp_tool_invocation`
  - Replace `observe_mcp_tool_invocation()` wrapper with explicit OTel span creation in `_execute_tool()`
- [x] 4.4 In `src/mcp_core/client/manager.py`:
  - Remove `from mcp_core.common.metrics import observe_mcp_client_call, set_mcp_upstream_health`
  - Replace `observe_mcp_client_call()` with explicit OTel span
  - Replace `set_mcp_upstream_health()` with OTel gauge
- [x] 4.5 In `src/mcp_core/client/auth.py`:
  - No `mcp_core.common.metrics` imports — no changes needed
- [x] 4.6 Delete `src/mcp_core/common/metrics.py` entirely
- [x] 4.7 Verify no remaining references to `mcp_core.common.metrics` anywhere in the codebase

## 5. Wire into Celery Workers

- [x] 5.1 In `src/app/connections/celery.py`:
  - Import `setup_otel`, `shutdown_otel` from shared module
  - Call `setup_otel(service_name="langchain-fastapi-celery")` at module level before `create_celery_app()`
  - Guard with module-level check: `if settings.OTEL_ENABLED`
  - Register `worker_shutting_down` signal to call `shutdown_otel()` for flushing
- [x] 5.2 Create OTel meter instruments at module level in `celery.py`
- [x] 5.3 In `ResilientTask.on_success()`:
  - Increment `celery.task.completed_total` with `status="success"`
  - Record `celery.task.duration_seconds`
- [x] 5.4 In `ResilientTask.on_failure()`:
  - Increment `celery.task.completed_total` with `status="failure"`
  - Record `celery.task.duration_seconds`
  - Increment `celery.task.retries_total`
- [x] 5.5 In `ResilientTask.on_retry()`:
  - Increment `celery.task.retries_total`
- [x] 5.6 In signal handlers (`log_task_prerun`, `log_task_postrun`, `log_task_retry`, `log_task_failure`):
  - Import `opentelemetry.trace`
  - Read `trace_id` hex string from `trace.get_current_span().get_span_context()`
  - Include `trace_id` in `logger.bind()` calls when valid

## 6. Update Logging Infrastructure

- [x] 6.1 In `src/app/utils/logger.py`:
  - Remove the commented-out JSON file handler block
  - Add `import opentelemetry.trace as otel_trace` at the top
- [x] 6.2 Rewrite `trace_layer` decorator:
  - Import `opentelemetry.trace` with `SpanKind`
  - Replace `time.perf_counter()` / `execution_path` logic with OTel spans
  - Async path: `async with tracer.start_as_current_span(...)` 
  - On exception: `span.record_exception(exc)`
  - Keep `execution_path` — append `func.__name__` for error response `flow` string, reset after execution
- [x] 6.3 Update `console_format()` to include `trace_id`
- [ ] 6.4 Verify `redact_sensitive_data()` patcher execution order (manual verification)

## 7. Lifespan Shutdown — force_flush

- [x] 7.1 Implement `shutdown_otel()` function in `src/app/shared/otel/__init__.py`
- [x] 7.2 In FastAPI lifespan (`src/app/lifecycle/lifespan.py`) — call `shutdown_otel()` as last cleanup step
- [x] 7.3 In MCP lifespan — `setup_otel()` called in `serve_mcp()`, `shutdown_otel()` handled by FastAPI lifespan (shared module state)
- [x] 7.4 In Celery worker (`src/app/connections/celery.py`): register `worker_shutting_down` handler that calls `shutdown_otel()`

## 8. Cleanup Remnants

- [ ] 8.1 Remove commented-out `SQLAlchemyInstrumentor` code from `src/app/connections/postgres.py`
- [x] 8.2 Search for remaining `prometheus_client` imports outside the shared OTel module — only `otel_integrations.py` bridge remains
- [x] 8.3 Remove `prometheus-client` from `pyproject.toml` direct dependencies
- [x] 8.4 Run `uv sync` to update lockfile after removing prometheus-client direct dep
- [x] 8.5 Run `ruff check src/` and fix import-ordering and other issues
- [x] 8.6 Run `ty check src/` and fix type issues (especially from OTel SDK types)

## 9. Manual Verification

- [ ] 9.1 Start SigNoz via docker-compose (or connect to existing instance at configured OTLP endpoint)
- [ ] 9.2 Start FastAPI server: `uv run uvicorn app.main:app`
- [ ] 9.3 Verify traces: hit `GET /health` → open SigNoz → confirm root span `GET /health` with child spans for DB queries
- [ ] 9.4 Verify metrics: `curl localhost:8000/metrics` → confirm Prometheus text output includes `http.server.request_count`, `http.server.duration`
- [ ] 9.5 Verify log correlation: hit any endpoint → open trace in SigNoz → check logs tab for correlated entries with matching `trace_id`
- [ ] 9.6 Start Celery worker: `uv run celery -A app.connections.celery:celery_app worker --loglevel=info`
- [ ] 9.7 Verify Celery traces: trigger task (e.g., document ingest) → SigNoz shows `celery.<task_name>` span
- [ ] 9.8 Start MCP server → hit MCP endpoint → SigNoz shows traces with `mcp.tool.*` attributes
- [ ] 9.9 Verify `trace_layer` spans: hit an endpoint with `@trace_layer("service")` decorated functions → SigNoz waterfall shows `layer.<name>` as child spans under HTTP root span
