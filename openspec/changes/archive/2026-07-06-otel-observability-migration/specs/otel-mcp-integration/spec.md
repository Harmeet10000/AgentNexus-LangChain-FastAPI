## ADDED Requirements

### Requirement: MCP server calls setup_otel() in its lifespan
The MCP uvicorn server SHALL call `setup_otel()` during its startup lifecycle, before accepting requests.

#### Scenario: OTel initialized in MCP lifespan
- **WHEN** the MCP server starts (via `mcp_core/lifespan_mcp.py:serve_mcp()`)
- **THEN** `setup_otel()` SHALL be called before creating the MCP ASGI app
- **AND** the call SHALL use the same shared module as FastAPI, ensuring the same export endpoint and resource attributes

```
# src/mcp_core/lifespan_mcp.py — changes at the top of serve_mcp()
from app.shared.otel import setup_otel
from app.config import get_settings

async def serve_mcp(parent_app: FastAPI) -> MCPServerHandle | None:
    settings = get_settings()
    if not settings.MCP_ENABLE_HTTP:
        return None

    # Initialize OTel with MCP-specific service name
    if settings.OTEL_ENABLED:
        setup_otel(service_name="langchain-fastapi-mcp")

    mcp_app = get_mcp_http_app(parent_app=parent_app)
    ...  # rest unchanged
```

#### Scenario: MCP OTel resources inherit service name from settings
- **WHEN** `setup_otel()` runs in the MCP process with `service_name="langchain-fastapi-mcp"`
- **THEN** the `Resource` SHALL set `service.name` to `langchain-fastapi-mcp` (distinct from FastAPI's `langchain-fastapi` to differentiate processes in SigNoz)
- **AND** `deployment.environment` and `service.version` SHALL match the FastAPI process

### Requirement: MCP ASGI middleware stack uses OTel middleware
The MCP HTTP middleware stack SHALL use the shared OTel ASGI middleware instead of the hand-rolled `MCPObservabilityMiddleware`.

#### Scenario: MCPObservabilityMiddleware class removed
- **WHEN** the `build_mcp_http_middleware()` function runs
- **THEN** `MCPObservabilityMiddleware` SHALL NOT be included in the returned middleware list
- **AND** `MCPObservabilityMiddleware` class SHALL be deleted from `mcp_core/server/middleware.py`
- **AND** the `from mcp_core.common.metrics import observe_mcp_http_request` import SHALL be removed
- **AND** `nanoid` import SHALL be removed if no other code uses it in this file

```
# src/mcp_core/server/middleware.py — BEFORE
from nanoid import generate
from app.config import get_settings
from app.features.auth.security import decode_token
from app.utils import UnauthorizedException, logger
from app.utils.rate_limit.service import RateLimitService
from mcp_core.common.metrics import observe_mcp_http_request    # REMOVE

class MCPObservabilityMiddleware:    # DELETE entire class (lines 135-174)
    ...

def build_mcp_http_middleware(parent_app: Any | None) -> list[Middleware]:
    ...
    return [
        Middleware(cast("Any", MCPObservabilityMiddleware)),     # REMOVE
        Middleware(cast("Any", MCPAuthMiddleware), ...),
        Middleware(cast("Any", MCPRateLimitMiddleware), ...),
    ]
```

#### Scenario: OTel ASGI middleware added to MCP stack
- **WHEN** `build_mcp_http_middleware()` returns its middleware list
- **THEN** the list SHALL include an OTel ASGI middleware instance as the outermost middleware (first element)
- **AND** it SHALL create root spans for all MCP HTTP requests
- **AND** `x-correlation-id` header SHALL still be set (responsibility moves to OTel ASGI middleware's `set_span_headers` or a minimal wrapper)

```
# src/mcp_core/server/middleware.py — AFTER
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

def build_mcp_http_middleware(parent_app: Any | None) -> list[Middleware]:
    settings = get_settings()

    def get_redis() -> Any | None:
        if parent_app is None:
            return None
        return getattr(parent_app.state, "redis", None)

    return [
        # OTel ASGI middleware — outermost, captures all requests
        Middleware(OpenTelemetryMiddleware),
        # Auth and rate limit remain, in this order
        Middleware(cast("Any", MCPAuthMiddleware), enabled=settings.MCP_REQUIRE_AUTH),
        Middleware(
            cast("Any", MCPRateLimitMiddleware),
            redis_getter=get_redis,
            burst=settings.MCP_RATE_LIMIT_BURST,
            rate=settings.MCP_RATE_LIMIT_RATE,
            period_seconds=settings.MCP_RATE_LIMIT_PERIOD_SECONDS,
        ),
    ]
```

#### Scenario: x-correlation-id preserved in MCP responses
- **WHEN** an MCP request is processed
- **THEN** the OTel ASGI middleware SHALL NOT remove the `x-correlation-id` header
- **AND** the middleware SHALL add `server-timing` header with OTel trace ID for debugging:
  `server-timing: traceparent;desc="00-{trace_id}-{span_id}-01"`

### Requirement: MCP tool calls recorded as OTel spans with metrics
Every MCP tool invocation SHALL be recorded as a child span under the MCP HTTP request root span, with tool-level duration and status metrics exported via OTel.

#### Scenario: MCP tool invocation creates child span
- **WHEN** an MCP tool function is called (in `_execute_tool()` in `src/mcp_core/server/tools.py`)
- **THEN** a child span SHALL be created under the current OTel span with attributes `mcp.tool.name`, `mcp.tool.status`
- **AND** the span SHALL record the tool call duration

```
# src/mcp_core/server/tools.py — replacement for observe_mcp_tool_invocation
import opentelemetry.trace as trace
from opentelemetry import metrics

_tracer = trace.get_tracer(__name__)
_meter = metrics.get_meter(__name__)

# Create instruments once at module level
_mcp_tool_calls_total = _meter.create_counter(
    name="mcp.tool.calls_total",
    unit="1",
    description="Total MCP tool calls",
)
_mcp_tool_duration = _meter.create_histogram(
    name="mcp.tool.duration_seconds",
    unit="s",
    description="MCP tool call duration",
)

async def _execute_tool(tool_name: str, fn: Callable, _: Any = None) -> Any:
    start = time.perf_counter()
    status = "success"
    with _tracer.start_as_current_span(f"mcp.tool.{tool_name}") as span:
        span.set_attribute("mcp.tool.name", tool_name)
        try:
            result = fn()
            if hasattr(result, "__await__"):
                result = await result
        except NotFoundException as exc:
            status = "not_found"
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc.detail)))
            logger.bind(tool=tool_name, error=str(exc.detail)).warning("MCP tool failed")
            return _error(str(exc.detail))
        except Exception as exc:
            status = "error"
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            logger.bind(tool=tool_name, error=str(exc)).exception("MCP tool failed")
            return _error("MCP tool execution failed", detail=str(exc))
        else:
            span.set_attribute("mcp.tool.status", "success")
            return result
        finally:
            duration = time.perf_counter() - start
            span.set_attribute("mcp.tool.duration_ms", round(duration * 1000, 2))
            _mcp_tool_calls_total.add(1, {"tool": tool_name, "status": status})
            _mcp_tool_duration.record(duration, {"tool": tool_name, "status": status})
```

#### Scenario: MCP client (upstream) calls recorded as spans
- **WHEN** the MCP server makes an outbound call to an upstream MCP server tool
- **THEN** a client span SHALL be created with attributes `mcp.client.server`, `mcp.client.tool`, `mcp.client.status`
- **AND** `mcp.client.calls_total` and `mcp.client.duration_seconds` metrics SHALL be recorded

```
# src/mcp_core/client/manager.py — replacement for observe_mcp_client_call
_tracer = trace.get_tracer(__name__)
_meter = metrics.get_meter(__name__)

_mcp_client_calls_total = _meter.create_counter(
    name="mcp.client.calls_total",
    unit="1",
    description="Outbound MCP client tool calls",
)
_mcp_client_duration = _meter.create_histogram(
    name="mcp.client.duration_seconds",
    unit="s",
    description="Outbound MCP client tool call duration",
)

# In the method that calls upstream (around line 144):
with _tracer.start_as_current_span(f"mcp.client.{server_name}.{tool_name}") as span:
    span.set_attribute("mcp.client.server", server_name)
    span.set_attribute("mcp.client.tool", tool_name)
    start = time.perf_counter()
    try:
        result = await session.call_tool(tool_name, arguments=arguments)
        status = "success"
    except Exception as exc:
        status = "error"
        span.record_exception(exc)
        span.set_status(trace.Status(trace.StatusCode.ERROR))
        raise
    finally:
        duration = time.perf_counter() - start
        span.set_attribute("mcp.client.status", status)
        span.set_attribute("mcp.client.duration_ms", round(duration * 1000, 2))
        _mcp_client_calls_total.add(1, {"server": server_name, "tool": tool_name, "status": status})
        _mcp_client_duration.record(duration, {"server": server_name, "tool": tool_name, "status": status})
```

### Requirement: mcp_core/common/metrics.py is deleted
The standalone MCP metrics module SHALL be removed entirely. All observability goes through the shared OTel pipeline.

#### Scenario: mcp_core/common/metrics.py file removed
- **WHEN** the change is applied
- **THEN** `src/mcp_core/common/metrics.py` SHALL be deleted
- **AND** all imports of this module SHALL be removed:
  - `mcp_core/server/middleware.py` — remove `from mcp_core.common.metrics import observe_mcp_http_request`
  - `mcp_core/server/tools.py` — remove `from mcp_core.common.metrics import observe_mcp_tool_invocation`
  - `mcp_core/client/manager.py` — remove `from mcp_core.common.metrics import observe_mcp_client_call, set_mcp_upstream_health`

#### Scenario: Upstream health gauge preserved via OTel
- **WHEN** the MCP server checks upstream server health (`set_mcp_upstream_health()` is called)
- **THEN** the health state SHALL be recorded as an OTel observable gauge via the shared meter with `server` and `project` attributes

```
# src/mcp_core/client/manager.py — replacement for set_mcp_upstream_health
from opentelemetry import metrics

_meter = metrics.get_meter(__name__)
_mcp_upstream_health = _meter.create_up_down_counter(
    name="mcp.upstream.server_health",
    unit="1",
    description="Health gauge for configured upstream MCP servers (1=healthy, 0=unhealthy)",
)

def set_mcp_upstream_health(server_name: str, healthy: bool) -> None:
    _mcp_upstream_health.add(
        1 if healthy else -1,
        {"server": server_name},
    )
```

### Requirement: MCP middleware.py cleans up unused imports
After removing `MCPObservabilityMiddleware` and the metrics import, the file SHALL clean up all unused imports.

#### Scenario: Clean up nanoid, time, json, cast imports if unused
- **WHEN** `MCPObservabilityMiddleware` is removed
- **THEN** `from nanoid import generate` SHALL be removed (only used by observability middleware)
- **AND** `import time` SHALL be removed if not used elsewhere in the file
- **AND** `json` and `cast` imports SHALL be checked — they may still be needed by `_send_json_response` and auth/rate-limit middleware
