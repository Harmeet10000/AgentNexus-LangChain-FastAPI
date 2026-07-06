## ADDED Requirements

### Requirement: setup_otel() creates a shared TracerProvider with OTLP export
The system SHALL provide a `setup_otel()` function that creates a `TracerProvider` with a `BatchSpanProcessor` pointed at the `OTLPSpanExporter` endpoint configured by `settings.OTEL_EXPORTER_OTLP_ENDPOINT`.

#### Scenario: TracerProvider initialized with correct endpoint
- **WHEN** `setup_otel()` is called
- **THEN** a `TracerProvider` SHALL be created and set as the global tracer provider
- **AND** a `BatchSpanProcessor` SHALL be attached with an `OTLPSpanExporter` using the configured `OTEL_EXPORTER_OTLP_ENDPOINT`
- **AND** the provider SHALL include a `Resource` with `service.name`, `deployment.environment`, and `service.version`
- **AND** the `service.name` SHALL default to `OTEL_SERVICE_NAME` but be overridable by the `service_name` parameter

```
# src/app/shared/otel/tracer.py
import opentelemetry.trace as trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

def _setup_tracer_provider(
    endpoint: str,
    resource: Resource,
    sample_rate: float = 1.0,
) -> TracerProvider:
    provider = TracerProvider(
        resource=resource,
        # ponytail: global sampler — per-service overrides via OTEL_TRACES_SAMPLER_ARG
        sampler=trace.sdk.trace.sampling.ParentBased(
            root=trace.sdk.trace.sampling.TraceIdRatioBased(sample_rate)
        ) if sample_rate < 1.0 else None,
    )
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint),
        max_queue_size=2048,
        max_export_batch_size=512,
        schedule_delay_millis=5000,
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return provider
```

#### Scenario: setup_otel() is idempotent
- **WHEN** `setup_otel()` is called multiple times
- **THEN** it SHALL NOT reinitialize already-configured providers
- **AND** the implementation SHALL use a module-level `_initialized: bool` flag

```
# src/app/shared/otel/__init__.py
_initialized: bool = False
_otel_enabled: bool = True

def setup_otel(
    service_name: str | None = None,
    enabled: bool | None = None,
) -> None:
    global _initialized, _otel_enabled
    if _initialized:
        return
    ...
    _initialized = True
```

#### Scenario: setup_otel accepts optional service_name parameter
- **WHEN** `setup_otel(service_name="langchain-fastapi")` is called
- **THEN** the `Resource` SHALL use `"langchain-fastapi"` as `service.name`
- **WHEN** `setup_otel()` is called without `service_name`
- **THEN** the `Resource` SHALL fall back to `settings.OTEL_SERVICE_NAME`
- **WHEN** both are empty/None
- **THEN** the `Resource` SHALL fall back to `"unknown-service"`

### Requirement: setup_otel() creates a shared MeterProvider with dual export
The system SHALL provide a `MeterProvider` that exports metrics via both OTLP (to SigNoz) and Prometheus text format (for `/metrics` endpoint).

#### Scenario: OTLP metric export configured
- **WHEN** `setup_otel()` is called
- **THEN** a `MeterProvider` SHALL be created with a `PeriodicExportingMetricReader` pointed at an `OTLPMetricExporter` using the configured endpoint
- **AND** the meter provider SHALL use the same `Resource` as the tracer provider
- **AND** the export interval SHALL default to 15000ms, configurable via `OTEL_METRICS_INTERVAL` setting

```
# src/app/shared/otel/metrics.py
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

_otel_prometheus_exporter: PrometheusMetricExporter | None = None

def _setup_meter_provider(
    endpoint: str,
    resource: Resource,
    export_interval_ms: int = 15000,
) -> MeterProvider:
    readers: list[MetricReader] = [
        PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint),
            export_interval_millis=export_interval_ms,
        ),
    ]
    provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)
    return provider
```

#### Scenario: Prometheus-compatible /metrics reading
- **WHEN** `setup_otel()` is called
- **THEN** a `PrometheusMetricExporter` SHALL be created and stored for the `/metrics` endpoint
- **AND** the `/metrics` endpoint SHALL call `generate_latest()` on this exporter to produce Prometheus text format

```
from opentelemetry.exporter.prometheus import PrometheusMetricExporter

def _setup_prometheus_exporter() -> PrometheusMetricExporter:
    global _otel_prometheus_exporter
    exporter = PrometheusMetricExporter()
    _otel_prometheus_exporter = exporter
    return exporter

def get_prometheus_metrics() -> tuple[bytes, str]:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    if _otel_prometheus_exporter is None:
        return b"", CONTENT_TYPE_LATEST
    return generate_latest(_otel_prometheus_exporter.registry), CONTENT_TYPE_LATEST
```

#### Scenario: Dual export does not block on SigNoz unavailability
- **WHEN** the OTLP endpoint is unreachable during metric export
- **THEN** the `PeriodicExportingMetricReader` SHALL log a warning and continue
- **AND** the Prometheus exporter SHALL remain unaffected (it serves data synchronously on scrape)

### Requirement: setup_otel() creates a shared LoggerProvider with OTLP export
The system SHALL provide a `LoggerProvider` connected to a `loguru` sink that exports log records via OTLP.

#### Scenario: LoggerProvider created with BatchLogRecordProcessor
- **WHEN** `setup_otel()` is called
- **THEN** a `LoggerProvider` SHALL be created with a `BatchLogRecordProcessor` pointed at an `OTLPLogExporter` using the configured endpoint
- **AND** the LoggerProvider SHALL be set as the global logger provider
- **AND** the log export SHALL be disabled when `OTEL_ENABLED` is `False`

```
# src/app/shared/otel/logs.py
from opentelemetry.sdk._logs import LoggerProvider, LogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, SimpleLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

def _setup_logger_provider(endpoint: str, resource: Resource) -> LoggerProvider:
    provider = LoggerProvider(resource=resource)
    processor = BatchLogRecordProcessor(
        OTLPLogExporter(endpoint=endpoint),
        max_queue_size=2048,
        max_export_batch_size=512,
        schedule_delay_millis=5000,
    )
    provider.add_log_record_processor(processor)
    # Also set as global for any code using OTel logging API
    from opentelemetry._logs import set_logger_provider as set_global_logger_provider
    set_global_logger_provider(provider)
    return provider
```

#### Scenario: LoggerProvider flush on shutdown
- **WHEN** the application shuts down
- **THEN** `setup_otel()` SHALL provide a `force_flush()` function that calls `force_flush()` on all providers
- **AND** the lifespan shutdown sequence SHALL call `force_flush()` to export remaining spans/metrics/logs

```
def force_flush() -> None:
    """Flush all OTel providers — call during app shutdown."""
    trace.get_tracer_provider().force_flush()
    metrics.get_meter_provider().force_flush()
    from opentelemetry._logs import get_logger_provider
    get_logger_provider().force_flush()
```

### Requirement: setup_otel() registers auto-instrumentors
The system SHALL register auto-instrumentors for SQLAlchemy, Redis, HTTPX, Celery, and ASGI when `setup_otel()` is called.

#### Scenario: SQLAlchemy instrumentor wraps engine creation
- **WHEN** `setup_otel()` is called before engine creation
- **THEN** the `SQLAlchemyInstrumentor` SHALL be configured to instrument any subsequently created engines
- **AND** it SHALL use `engine=None` to instrument all engines globally

```
# src/app/shared/otel/instrument.py
try:
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    SQLAlchemyInstrumentor().instrument(engine=None, enable_commenter=True, commenter_options={})
except Exception:
    logger.warning("SQLAlchemy auto-instrumentation failed — continuing without DB tracing")
```

#### Scenario: Redis instrumentor wraps async Redis client
- **WHEN** `setup_otel()` is called
- **THEN** the `RedisInstrumentor` SHALL be configured to instrument async Redis operations
- **AND** Redis span attributes SHALL include `db.system="redis"`, `db.redis.args_length`

#### Scenario: HTTPX instrumentor wraps outbound requests
- **WHEN** `setup_otel()` is called
- **THEN** the `HTTPXClientInstrumentor` SHALL be configured to instrument all HTTPX client requests
- **AND** HTTPX spans SHALL include `http.url`, `http.method`, `http.status_code`

#### Scenario: Celery instrumentor wraps task execution
- **WHEN** `setup_otel()` is called in the Celery worker process
- **THEN** the `CeleryInstrumentor` SHALL be configured to wrap task execution in spans
- **AND** it SHALL create spans for both task execution (consumer) and `apply_async` (producer)

#### Scenario: ASGI instrumentor wraps the ASGI app
- **WHEN** `setup_otel()` is called
- **THEN** an `ASGIInstrumentor` SHALL be configured but NOT instrumented at bootstrap time (instrumentation happens when the ASGI middleware is added to each app)
- **AND** the instrumentor SHALL be available for the FastAPI and MCP middleware builders

```
# ASGI instrumentor is registered differently — the middleware is added per-app:
# In main.py:  app.add_middleware(OpenTelemetryMiddleware, ...)
# In http.py:  middleware list = [Middleware(OpenTelemetryMiddleware, ...), ...]
```

#### Scenario: Instrumentor registration skips when disabled
- **WHEN** `OTEL_ENABLED` is `False`
- **THEN** no instrumentors SHALL be registered
- **AND** the function SHALL return immediately without setting providers

#### Scenario: Instrumentor failures are non-fatal
- **WHEN** an instrumentor raises on `.instrument()` (e.g., missing library)
- **THEN** the exception SHALL be caught and logged as a warning
- **AND** remaining instrumentors SHALL still be registered

```
def _safe_instrument(name: str, instrument_fn: Callable[[], None]) -> None:
    try:
        instrument_fn()
    except Exception as exc:
        logger.warning(f"OTel instrumentor '{name}' failed: {exc}")
```

### Requirement: Module provides getters for tracer, meter, logger
The module SHALL provide `get_tracer(name)`, `get_meter(name)`, and `get_logger(name)` convenience functions that delegate to the global providers.

#### Scenario: get_tracer returns a tracer from the global provider
- **WHEN** `get_tracer("my.module")` is called after `setup_otel()`
- **THEN** it SHALL return `trace.get_tracer("my.module")` from the configured `TracerProvider`

#### Scenario: get_tracer returns a no-op tracer when uninitialized
- **WHEN** `get_tracer("my.module")` is called before `setup_otel()`
- **THEN** it SHALL return a no-op tracer (not raise)
- **AND** spans created from a no-op tracer SHALL be no-ops (not raise)

```
def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
# No initialization check needed — OTel falls back to no-op automatically
```

### Requirement: Resource is constructed with consistent attributes across processes
The `Resource` SHALL be created once by `setup_otel()` and passed to all three providers to ensure consistent identity.

#### Scenario: Resource includes standard attributes
- **WHEN** the Resource is created
- **THEN** it SHALL include:
  - `service.name` — from `service_name` parameter or `OTEL_SERVICE_NAME` setting
  - `service.version` — from `settings.APP_VERSION`
  - `deployment.environment` — from `settings.ENVIRONMENT`
  - `telemetry.sdk.name` — `"opentelemetry"`
  - `telemetry.sdk.language` — `"python"`
  - `telemetry.sdk.version` — from `opentelemetry.__version__`

```
# src/app/shared/otel/__init__.py
from opentelemetry.sdk.resources import Resource, OTELResourceDetector

def _build_resource(service_name: str | None = None) -> Resource:
    settings = get_settings()
    return Resource.create({
        "service.name": service_name or settings.OTEL_SERVICE_NAME or "unknown-service",
        "service.version": settings.APP_VERSION or "0.0.0",
        "deployment.environment": settings.ENVIRONMENT or "development",
    }).merge(OTELResourceDetector().detect())  # merge env-var overrides
```

### Requirement: setup_otel() uses OTEL_SAMPLE_RATE for head-based sampling
The tracer provider SHALL use the `OTEL_SAMPLE_RATE` setting to decide which traces to sample.

#### Scenario: sample_rate=1.0 samples all traces
- **WHEN** `OTEL_SAMPLE_RATE=1.0`
- **THEN** every trace root span SHALL be sampled (exported) and all three signals logged
- **AND** no sampling decision SHALL be applied at the SDK level (default `AlwaysOn` sampler)

#### Scenario: sample_rate=0.1 samples 10% of traces
- **WHEN** `OTEL_SAMPLE_RATE=0.1`
- **THEN** approximately 10% of root spans SHALL be sampled
- **AND** child spans SHALL inherit the sampling decision from the parent (`ParentBased` sampler)
