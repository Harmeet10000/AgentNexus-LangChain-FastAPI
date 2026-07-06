## Why

The project's observability is fragmented across three incompatible systems with significant blind spots:

- **Logging**: loguru with `request_state`/`execution_path` ContextVars, console-only output, no persistent storage, no trace correlation
- **Metrics**: Prometheus `prometheus_client` with `metrics_registry` defined and `MetricsMiddleware` class written but **disabled** in `main.py:88` — only the MCP sub-application actually collects any metrics
- **Tracing**: LangSmith for LLM-only traces (proprietary, vendor-locked). OpenTelemetry dependencies (`opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-sqlalchemy`) and settings (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`) already exist but are **nowhere wired** into any middleware, instrumentor, or export pipeline
- **Correlation**: `correlation_id` flows through `RequestStateLoggingMiddleware` but is not linked to any distributed trace context
- **Background tasks**: Celery workers have lifecycle logging via signal handlers but no traces, no metrics, and no log correlation
- **MCP (separate uvicorn process)**: Has its own `MCPObservabilityMiddleware` and `mcp_core/common/metrics.py` with manual Prometheus counters, duplicating effort and diverging from the main app

This means every debugging session requires manual cross-referencing: grep logs for a `correlation_id`, guess which request it belonged to, check LangSmith for LLM traces, check prometheus for rate data — none of it connected.

The fix is a single OpenTelemetry pipeline shared across all processes (FastAPI, MCP, Celery) exporting to a self-hosted SigNoz backend, with a `loguru` → OTLP bridge for correlated logs, and a Prometheus-compatible `/metrics` endpoint for k8s scraping.

## What Changes

### New Module: `src/app/shared/otel/`

Five files that bootstrap the shared OTel pipeline:

| File | Purpose |
|---|---|
| `__init__.py` | `setup_otel()` entry point — called once per process at startup |
| `tracer.py` | `TracerProvider` + `BatchSpanProcessor` → `OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)` |
| `metrics.py` | `MeterProvider` with dual export: `PeriodicExportingMetricReader(OTLPMetricExporter())` for SigNoz + `PrometheusMetricExporter()` for `/metrics` |
| `logs.py` | Custom loguru sink → `BatchLogRecordProcessor(OTLPLogExporter())`; reads `trace_id`/`span_id` from `opentelemetry.context.get_current()` before each `LogRecord` |
| `instrument.py` | `setup_auto_instrumentation()` — registers `SQLAlchemyInstrumentor`, `RedisInstrumentor`, `HTTPXClientInstrumentor`, `CeleryInstrumentor`, `ASGIInstrumentor` |

### FastAPI Integration

- `main.py`: Call `setup_otel()` before `create_app()` (alongside `configure_langsmith()`)
- `main.py`: Replace commented `MetricsMiddleware` with OTel ASGI middleware
- `main.py`: `/metrics` endpoint unchanged — reads from `PrometheusMetricExporter` (same Prometheus text format)
- `server_middleware.py`: Remove `MetricsMiddleware` class, `prometheus_client` imports, `metrics_registry` from both file and `__init__.py`
- `RequestStateLoggingMiddleware`: Enriched to inject `trace_id` from OTel context into `logger.bind(trace_id=...)`

### MCP Integration

- `mcp_core/lifespan_mcp.py`: Call `setup_otel()` before starting MCP uvicorn server
- `mcp_core/server/middleware.py`: Remove `MCPObservabilityMiddleware`; replace with OTel ASGI middleware in `build_mcp_http_middleware()`
- `mcp_core/common/metrics.py`: **Delete entire file** — MCP tool metrics use shared OTel meter with `mcp.` prefixed attributes
- Tool-level observers (`observe_mcp_tool_invocation`, `observe_mcp_client_call`): Convert to OTel instruments + span attributes

### Celery Integration

- `celery.py`: Call `setup_otel()` before building Celery app
- Auto-instrument with `CeleryInstrumentor()` for automatic task spans
- `ResilientTask.on_success/on_failure/on_retry`: Add counter + histogram instruments via shared meter (`celery.task.*`)
- Signal handlers enriched with `trace_id` from OTel span context

### Logging

- `logger.py`: Remove commented-out JSON file handler
- `logger.py`: Add loguru OTLP sink that exports every log record as an OTel `LogRecord` with `trace_id`/`span_id` from current context
- `logger.py`: `redact_sensitive_data()` patch kept — runs before OTel sink
- `trace_layer` decorator: Rewrite to create real OTel child spans via `tracer.start_as_current_span()`. Keep `execution_path` ContextVar for error response `flow` string but remove `time.perf_counter()` (duration comes from span end time)

### Dependencies Added

```toml
"opentelemetry-instrumentation-asgi>=0.60b0",
"opentelemetry-instrumentation-httpx>=0.60b0",
"opentelemetry-instrumentation-redis>=0.60b0",
"opentelemetry-instrumentation-celery>=0.60b0",
"opentelemetry-exporter-prometheus>=0.60b0",
"opentelemetry-sdk-logs>=0.60b0",
```

### Settings Changes

Add to existing OTel block in `src/app/config/settings.py`:
```python
OTEL_LOGS_EXPORTER: str = Field(default="otlp")
OTEL_ENABLED: bool = Field(default=True)                 # global kill switch
OTEL_SAMPLE_RATE: float = Field(default=1.0)             # 1.0 = push everything
```

### Removals

| Artifact | Reason |
|---|---|
| `mcp_core/common/metrics.py` | Replaced by shared OTel metrics |
| `MCPObservabilityMiddleware` class | Replaced by OTel ASGI middleware |
| `MetricsMiddleware` class (server_middleware.py) | Replaced by OTel ASGI middleware |
| `prometheus_client` dependency | Replaced by `opentelemetry-exporter-prometheus` (uses `prometheus_client` internally but managed by OTel) |
| Commented-out `SQLAlchemyInstrumentor` in `postgres.py` | Moved to `instrument.py` `setup_auto_instrumentation()` |

## Capabilities

### New Capabilities
- `otel-bootstrap`: Shared OTel tracer, meter, and log provider initialization with automatic export to SigNoz via OTLP
- `otel-fastapi-integration`: OTel ASGI middleware, Prometheus `/metrics` endpoint via OTel exporter, trace-correlated request logging
- `otel-mcp-integration`: Unified OTel pipeline for MCP sub-application, replacing manual observability middleware
- `otel-celery-integration`: OTel auto-instrumentation for Celery workers with task-level spans and task lifecycle metrics
- `otel-trace-layer-upgrade`: Upgrade `@trace_layer` decorator from manual `time.perf_counter()` + ContextVar breadcrumbs to real OTel child spans
- `otel-loguru-otlp-bridge`: Loguru sink that exports structured logs as OTel LogRecords with trace/span ID correlation

### Modified Capabilities
- None.

## Impact

- `src/app/shared/otel/` — 5 new files
- `src/app/main.py` — OTel bootstrap call, middleware change, metrics endpoint update
- `src/app/middleware/server_middleware.py` — Remove MetricsMiddleware, prometheus_client imports
- `src/app/middleware/__init__.py` — Remove re-exports of removed classes
- `src/app/utils/logger.py` — Add OTel loguru sink, rewrite trace_layer, remove JSON file handler
- `src/app/config/settings.py` — Add 3 new OTel fields
- `src/app/connections/celery.py` — OTel bootstrap call, instrument task lifecycle
- `src/app/connections/postgres.py` — Remove commented-out SQLAlchemyInstrumentor
- `mcp_core/server/middleware.py` — Remove MCPObservabilityMiddleware, add OTel ASGI middleware
- `mcp_core/common/metrics.py` — Delete entire file
- `mcp_core/lifespan_mcp.py` — OTel bootstrap call
- `src/mcp_core/server/http.py` — Possibly remove MCPObservabilityMiddleware from build_mcp_http_middleware
- `pyproject.toml` — Add 6 new dependencies, optionally remove `prometheus-client`
