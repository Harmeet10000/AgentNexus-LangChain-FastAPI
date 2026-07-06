## ADDED Requirements

### Requirement: FastAPI server calls setup_otel() on startup
The FastAPI application SHALL call `setup_otel()` from the shared bootstrap module before creating the FastAPI app instance.

#### Scenario: OTel initialized at app module level
- **WHEN** the `src/app/main.py` module loads
- **THEN** `setup_otel()` SHALL be called after `configure_langsmith()` and before `create_app()`
- **AND** the call SHALL use `service_name="langchain-fastapi"`
- **AND** the call SHALL be guarded by `settings.OTEL_ENABLED`

```
# src/app/main.py — changes at module level (line 33 area)
from app.middleware import (
    RequestStateLoggingMiddleware,
    build_fastapi_guard_config,
    get_metrics,
    global_exception_handler,
)
from app.shared.otel import setup_otel        # NEW
from app.shared.langchain_layer import configure_langsmith

configure_langsmith()
load_dotenv(dotenv_path=".env.development")

settings_for_otel = get_settings()            # NEW — load settings before app
if settings_for_otel.OTEL_ENABLED:
    setup_otel(service_name="langchain-fastapi")
```

#### Scenario: setup_otel does not block startup on SigNoz unavailability
- **WHEN** the OTLP endpoint is unreachable at startup
- **THEN** `setup_otel()` SHALL NOT raise or block app startup
- **AND** spans/metrics/logs SHALL be buffered in the SDK and sent when the endpoint becomes available
- **AND** the `BatchSpanProcessor` SHALL handle this transparently with internal retry logic

### Requirement: OTel ASGI middleware replaces MetricsMiddleware
The ASGI middleware stack SHALL include an OpenTelemetry ASGI middleware that creates a root span for every incoming HTTP request and records HTTP server metrics.

#### Scenario: OTel ASGI middleware added as outermost middleware
- **WHEN** the middleware stack is built in `create_app()`
- **THEN** the OTel ASGI middleware SHALL be added as the FIRST middleware (outermost, last `add_middleware` call)
- **AND** it SHALL wrap all inner middleware including `SecurityMiddleware` and `RequestStateLoggingMiddleware`

```
# src/app/main.py — inside create_app(), after all other middleware
# Middleware execution order (outermost first):
#   1. OpenTelemetryMiddleware (NEW) — captures everything, including guard blocks
#   2. RequestStateLoggingMiddleware — correlation ID, trace_id injection
#   3. SecurityMiddleware (Guard) — IP checks, rate limiting
#   4. GZipMiddleware — response compression
#   5. ApiDeprecationMiddleware — Deprecation/Sunset headers
#   6. CORSMiddleware (injected by Guard)
#   7. Route handler

app.add_middleware(GZipMiddleware, minimum_size=15000, compresslevel=6)
app.add_middleware(ApiDeprecationMiddleware, ...)
app.add_middleware(SecurityMiddleware, config=guard_config)
app.add_middleware(RequestStateLoggingMiddleware)

# OTel ASGI middleware — outermost, captures everything including guard blocks
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
app.add_middleware(OpenTelemetryMiddleware)
```

#### Scenario: OTel ASGI middleware creates root span
- **WHEN** an HTTP request arrives
- **THEN** the OTel ASGI middleware SHALL create a root span with attributes for `http.method`, `http.url`, `http.status_code`, `http.route`
- **AND** the root span SHALL be the parent of all downstream spans (SQLAlchemy, Redis, HTTPX calls, trace_layer spans)

#### Scenario: OTel ASGI middleware records server metrics
- **WHEN** an HTTP request completes
- **THEN** the middleware SHALL record:
  - `http.server.request_count` — counter with `http.method`, `http.status_code`, `http.route`
  - `http.server.duration` — histogram with same labels
  - `http.server.active_requests` — up-down counter with `http.method`, `http.route`
- **AND** all metrics SHALL be recorded via the shared meter under the instrumentation scope `opentelemetry.instrumentation.asgi`

#### Scenario: Path normalization prevents high-cardinality metric labels
- **WHEN** recording metrics for parameterized paths like `/users/123`
- **THEN** the `http.route` attribute SHALL use the normalized form `/users/{id}`
- **AND** the OTel ASGI middleware accepts a `default_span_details` hook for custom normalization

```
# Path normalization via OTel ASGI middleware hook
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

def _otel_span_details(scope: dict) -> tuple[str, dict]:
    """Custom span name and attribute extraction with path normalization."""
    path = scope.get("path", "/")
    method = scope.get("method", "GET")

    # Normalize parameterized paths
    normalized = _normalize_path_otel(path)
    span_name = f"{method} {normalized}"
    attributes = {"http.route": normalized}
    return span_name, attributes

app.add_middleware(
    OpenTelemetryMiddleware,
    default_span_details=_otel_span_details,
)
```

Where `_normalize_path_otel()` is a simplified version of the old `_normalize_path()`:

```
def _normalize_path_otel(path: str) -> str:
    if path in {"/", "/health", "/metrics", "/docs", "/redoc", "/openapi.json", "/swagger.json"}:
        return path
    parts = path.split("/")
    return "/".join(
        "{id}" if (part.isdigit() or (len(part) == 36 and part.count("-") == 4)) else part
        for part in parts
    )
```

### Requirement: /metrics endpoint serves Prometheus text format via OTel exporter
The existing `/metrics` endpoint SHALL continue to serve Prometheus text format, reading from the OTel `PrometheusMetricExporter` instead of the old `prometheus_client` registry.

#### Scenario: /metrics response unchanged
- **WHEN** a GET request is made to `/metrics`
- **THEN** the response SHALL have `content-type: text/plain; charset=utf-8; version=0.0.4`
- **AND** the body SHALL contain Prometheus text-format metrics from the OTel exporter
- **AND** the metrics SHALL include both ASGI-instrumented metrics (`http.server.*`) and any registered custom instruments (MCP `mcp.tool.*`, Celery `celery.task.*`)

```
# src/app/middleware/server_middleware.py — get_metrics() rewrite
from app.shared.otel import get_prometheus_metrics

def get_metrics() -> tuple[bytes, str]:
    """Get Prometheus metrics from OTel exporter."""
    return get_prometheus_metrics()
```

#### Scenario: MetricsMiddleware class is removed
- **WHEN** the `server_middleware.py` changes are applied
- **THEN** the `MetricsMiddleware` class SHALL be removed from `server_middleware.py`
- **AND** the `prometheus_client` module-level metrics (`http_requests_total`, `http_request_duration_seconds`, `http_requests_in_progress`, `app_up`) SHALL be removed
- **AND** the `metrics_registry` SHALL be removed from both `server_middleware.py` and the middleware `__init__.py`
- **AND** the `PROJECT` constant SHALL be removed from `server_middleware.py`
- **AND** the `import prometheus_client` block SHALL be removed

```
# server_middleware.py — REMOVED section (lines 13-68 removed entirely)
"""
REMOVED:
- prometheus_client imports
- metrics_registry = CollectorRegistry()
- http_requests_total, http_request_duration_seconds, http_requests_in_progress, app_up
- MetricsMiddleware class (__init__, __call__, send_wrapper)
- _normalize_path() function
- PROJECT = "langchain-fastapi"
"""
```

### Requirement: RequestStateLoggingMiddleware injects trace_id into log context
The existing `RequestStateLoggingMiddleware` SHALL read the current OTel `trace_id` and inject it into the loguru context for request-scoped log lines.

#### Scenario: trace_id attached to request log records
- **WHEN** a request starts processing
- **THEN** the middleware SHALL call `logger.bind(trace_id=<hex trace_id>)` with the trace ID from the OTel span context
- **AND** every log line emitted during the request SHALL include the `trace_id` field

```
# src/app/middleware/server_middleware.py — in RequestStateLoggingMiddleware.__call__

# After creating request state (around line 113 area), BEFORE logger.contextualize():
trace_id = ""
span = trace.get_current_span()
if span is not None:
    span_context = span.get_span_context()
    if span_context.is_valid:
        trace_id = format(span_context.trace_id, "032x")
        state["trace_id"] = trace_id

# Then in the contextualize call:
with logger.contextualize(**state):
    ...
```

#### Scenario: trace_id included in x-correlation-id response header
- **WHEN** the response is sent
- **THEN** the `x-correlation-id` header SHALL contain the `correlation_id` (not trace_id, as this is the request-scoped identifier)
- **AND** the log context SHALL include both `correlation_id` and `trace_id` so they are cross-referenceable

#### Scenario: trace_id appears in console log output
- **WHEN** a log line is emitted during an active trace
- **THEN** the console format SHALL show `trace_id=<hex>` in the meta section
- **AND** the format SHALL omit `trace_id` when the field is empty string

### Requirement: middleware/__init__.py exports are updated
The middleware package `__init__.py` SHALL remove references to deleted classes and update `get_metrics` reference.

#### Scenario: __init__.py cleans up removed exports
- **WHEN** the changes are applied
- **THEN** `MetricsMiddleware` SHALL be removed from the import and `__all__` list
- **AND** `metrics_registry` SHALL be removed from the import and `__all__` list
- **AND** `get_metrics` SHALL remain (its implementation now reads from OTel Prometheus exporter)

```
# src/app/middleware/__init__.py — after change
from .global_exception_handler import global_exception_handler
from .server_middleware import (
    # MetricsMiddleware removed
    # metrics_registry removed
    RequestStateLoggingMiddleware,
    build_fastapi_guard_config,
    get_metrics,
    initialize_fastapi_guard,
)
```

### Requirement: build_fastapi_guard_config SHALL remain unchanged with no prometheus_client dependency
The `build_fastapi_guard_config()` function SHALL have no dependency on Prometheus metrics — it only depends on `settings` and requires no changes.

#### Scenario: guard config unchanged
- **WHEN** `build_fastapi_guard_config()` is called
- **THEN** it SHALL work identically before and after the change
- **AND** no `prometheus_client` references appear in or near this function
