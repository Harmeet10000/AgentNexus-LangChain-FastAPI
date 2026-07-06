## ADDED Requirements

### Requirement: trace_layer decorator creates OTel child spans instead of manual timing
The `trace_layer` decorator SHALL stop using `time.perf_counter()` and ContextVar breadcrumbs, and instead create proper OTel child spans.

#### Scenario: trace_layer uses OTel tracer
- **WHEN** a function decorated with `@trace_layer("some.layer")` is called
- **THEN** a child span SHALL be created under the current OTel trace context
- **AND** the span name SHALL be the layer name passed as argument
- **AND** the span SHALL include attributes for `layer`, `function`, `module`

```
# src/app/utils/logger.py — before (current implementation)
import time
from contextvars import ContextVar

execution_path: ContextVar[list[str]] = ContextVar("execution_path", default=[])

KEEP_EXECUTION_PATH_LENGTH = 10
_request_state: ContextVar[dict[str, Any]] = ContextVar("request_state", default={})

def trace_layer(layer_name: str) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            path = execution_path.get()
            path.append(layer_name)
            if len(path) > KEEP_EXECUTION_PATH_LENGTH:
                path.pop(0)
            execution_path.set(path)
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                logger.bind(
                    layer=layer_name,
                    function=func.__name__,
                    module=func.__module__,
                    duration_ms=round(elapsed * 1000, 2),
                    execution_path=list(path),
                ).debug("Layer trace")
        return async_wrapper
    return decorator

# src/app/utils/logger.py — after (OTel-native, execution_path kept for flow)
import opentelemetry.trace as trace_lib
from opentelemetry.trace import SpanKind

_otel_tracer = trace_lib.get_tracer(__name__)

def trace_layer(layer_name: str) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Keep execution_path for error response flow string
            current_flow = execution_path.get().copy()
            current_flow.append(func.__name__)
            if len(current_flow) > KEEP_EXECUTION_PATH_LENGTH:
                current_flow.pop(0)
            token = execution_path.set(current_flow)
            span_name = f"layer.{layer_name}"
            with _otel_tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL,
                attributes={
                    "layer": layer_name,
                    "function": func.__name__,
                    "module": func.__module__,
                },
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(trace_lib.Status(trace_lib.StatusCode.ERROR))
                    raise
                return result
            finally:
                execution_path.reset(token)
        return async_wrapper
    return decorator
```

#### Scenario: sync functions also supported
- **WHEN** a synchronous function is decorated with `@trace_layer("some.layer")`
- **THEN** the decorator SHALL detect the sync function and create an OTel span without `await`
- **AND** SHALL wrap the execution in a sync context manager

```
# Inside trace_layer decorator — detect sync vs async
def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            ...
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            current_flow = execution_path.get().copy()
            current_flow.append(func.__name__)
            if len(current_flow) > KEEP_EXECUTION_PATH_LENGTH:
                current_flow.pop(0)
            token = execution_path.set(current_flow)
            span_name = f"layer.{layer_name}"
            with _otel_tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL, attributes={"layer": layer_name, "function": func.__name__, "module": func.__module__}):
                try:
                    return func(*args, **kwargs)
                finally:
                    execution_path.reset(token)
        return sync_wrapper
```

### Requirement: execution_path ContextVar preserved for error response flow string
The `execution_path` ContextVar SHALL be kept for building the `flow` string in error responses, but its population SHALL be simplified to track only function names (not layer names) since OTel spans now provide the full execution trace.

#### Scenario: execution_path populated by trace_layer
- **WHEN** a `@trace_layer("service")` decorated function executes
- **THEN** the decorator SHALL append `func.__name__` to `execution_path` for the error response `flow` string
- **AND** `KEEP_EXECUTION_PATH_LENGTH = 10` SHALL remain as a cap
- **AND** the path SHALL NOT be duplicated in the debug log (already removed)

```
# Inside trace_layer wrapper — execution_path kept for error responses
current_flow = execution_path.get().copy()
current_flow.append(func.__name__)
if len(current_flow) > KEEP_EXECUTION_PATH_LENGTH:
    current_flow.pop(0)
token = execution_path.set(current_flow)
try:
    return await func(*args, **kwargs)
finally:
    execution_path.reset(token)
```

### Requirement: trace_layer debug log simplified
The existing debug log in `trace_layer` SHALL be removed. OTel spans replace the logging-based layer tracing.

#### Scenario: Debug log line removed from trace_layer
- **WHEN** the decorator no longer uses manual timing
- **THEN** the `logger.bind(...).debug("Layer trace")` call SHALL be removed from the wrapper function
- **AND** the layer information SHALL be available in OTel spans via span attributes

### Requirement: existing callers of trace_layer are unchanged
The `@trace_layer(...)` decorator is used across multiple service files. The interface SHALL NOT change — only the implementation.

#### Scenario: trace_layer interface stable
- **WHEN** a module imports and uses `@trace_layer("some.layer")`
- **THEN** the import `from app.utils.logger import trace_layer` SHALL continue to work
- **AND** the function signature SHALL remain `trace_layer(layer_name: str) -> Callable[..., Any]`
- **AND** all existing call sites SHALL NOT require changes

### Requirement: _request_state ContextVar preserved
The `_request_state` ContextVar SHALL remain unchanged. It is used outside `trace_layer` for request-scoped state.

#### Scenario: _request_state not affected
- **WHEN** `trace_layer` is rewritten
- **THEN** `_request_state` SHALL remain in `logger.py` with its current behavior
