# Middleware and Prompt Lifecycle

Source lines: 16617-18197 from the original FastMCP documentation dump.

Client logging, middleware, pagination, progress reporting, prompts, prompt registration behavior, and notifications.

---

# Client Logging
Source: https://gofastmcp.com/servers/logging

Send log messages back to MCP clients through the context.

<Tip>
  This documentation covers **MCP client logging**—sending messages from your server to MCP clients. For standard server-side logging (e.g., writing to files, console), use `fastmcp.utilities.logging.get_logger()` or Python's built-in `logging` module.
</Tip>

Server logging allows MCP tools to send debug, info, warning, and error messages back to the client. Unlike standard Python logging, MCP server logging sends messages directly to the client, making them visible in the client's interface or logs.

## Basic Usage

Use the context logging methods within any tool function:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context

mcp = FastMCP("LoggingDemo")

@mcp.tool
async def analyze_data(data: list[float], ctx: Context) -> dict:
    """Analyze numerical data with comprehensive logging."""
    await ctx.debug("Starting analysis of numerical data")
    await ctx.info(f"Analyzing {len(data)} data points")

    try:
        if not data:
            await ctx.warning("Empty data list provided")
            return {"error": "Empty data list"}

        result = sum(data) / len(data)
        await ctx.info(f"Analysis complete, average: {result}")
        return {"average": result, "count": len(data)}

    except Exception as e:
        await ctx.error(f"Analysis failed: {str(e)}")
        raise
```

## Log Levels

| Level           | Use Case                                                        |
| --------------- | --------------------------------------------------------------- |
| `ctx.debug()`   | Detailed execution information for diagnosing problems          |
| `ctx.info()`    | General information about normal program execution              |
| `ctx.warning()` | Potentially harmful situations that don't prevent execution     |
| `ctx.error()`   | Error events that might still allow the application to continue |

## Structured Logging

All logging methods accept an `extra` parameter for sending structured data to the client. This is useful for creating rich, queryable logs.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
async def process_transaction(transaction_id: str, amount: float, ctx: Context):
    await ctx.info(
        f"Processing transaction {transaction_id}",
        extra={
            "transaction_id": transaction_id,
            "amount": amount,
            "currency": "USD"
        }
    )
```

## Server-Side Logs

Messages sent to clients via `ctx.log()` and its convenience methods are also logged to the server's log at `DEBUG` level. Enable debug logging on the `fastmcp.server.context.to_client` logger to see these messages:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import logging
from fastmcp.utilities.logging import get_logger

to_client_logger = get_logger(name="fastmcp.server.context.to_client")
to_client_logger.setLevel(level=logging.DEBUG)
```

## Client Handling

Log messages are sent to the client through the MCP protocol. How clients handle these messages depends on their implementation—development clients may display logs in real-time, production clients may store them for analysis, and integration clients may forward them to external logging systems.

See [Client Logging](/clients/logging) for details on how clients handle server log messages.


# Middleware
Source: https://gofastmcp.com/servers/middleware

Add cross-cutting functionality to your MCP server with middleware that intercepts and modifies requests and responses.

<VersionBadge />

Middleware adds behavior that applies across multiple operations—authentication, logging, rate limiting, or request transformation—without modifying individual tools or resources.

<Tip>
  MCP middleware is a FastMCP-specific concept and is not part of the official MCP protocol specification.
</Tip>

## Overview

MCP middleware forms a pipeline around your server's operations. When a request arrives, it flows through each middleware in order—each can inspect, modify, or reject the request before passing it along. After the operation completes, the response flows back through the same middleware in reverse order.

```
Request → Middleware A → Middleware B → Handler → Middleware B → Middleware A → Response
```

This bidirectional flow means middleware can:

* **Pre-process**: Validate authentication, log incoming requests, check rate limits
* **Post-process**: Transform responses, record timing metrics, handle errors consistently

The key decision point is `call_next(context)`. Calling it continues the chain; not calling it stops processing entirely.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext

class LoggingMiddleware(Middleware):
    async def on_message(self, context: MiddlewareContext, call_next):
        print(f"→ {context.method}")
        result = await call_next(context)
        print(f"← {context.method}")
        return result

mcp = FastMCP("MyServer")
mcp.add_middleware(LoggingMiddleware())
```

### Execution Order

Middleware executes in the order added to the server. The first middleware runs first on the way in and last on the way out:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware

mcp = FastMCP("MyServer")
mcp.add_middleware(ErrorHandlingMiddleware())   # 1st in, last out
mcp.add_middleware(RateLimitingMiddleware())    # 2nd in, 2nd out
mcp.add_middleware(LoggingMiddleware())         # 3rd in, first out
```

This ordering matters. Place error handling early so it catches exceptions from all subsequent middleware. Place logging late so it records the actual execution after other middleware has processed the request.

### Server Composition

When using [mounted servers](/servers/composition), middleware behavior follows a clear hierarchy:

* **Parent middleware** runs for all requests, including those routed to mounted servers
* **Mounted server middleware** only runs for requests handled by that specific server

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware.logging import LoggingMiddleware

parent = FastMCP("Parent")
parent.add_middleware(AuthMiddleware())  # Runs for ALL requests

child = FastMCP("Child")
child.add_middleware(LoggingMiddleware())  # Only runs for child's tools

parent.mount(child, namespace="child")
```

Requests to `child_tool` flow through the parent's `AuthMiddleware` first, then through the child's `LoggingMiddleware`.

## Hooks

Rather than processing every message identically, FastMCP provides specialized hooks at different levels of specificity. Multiple hooks fire for a single request, going from general to specific:

| Level     | Hooks                                                     | Purpose                                         |
| --------- | --------------------------------------------------------- | ----------------------------------------------- |
| Message   | `on_message`                                              | All MCP traffic (requests and notifications)    |
| Type      | `on_request`, `on_notification`                           | Requests expecting responses vs fire-and-forget |
| Operation | `on_call_tool`, `on_read_resource`, `on_get_prompt`, etc. | Specific MCP operations                         |

When a client calls a tool, the middleware chain processes `on_message` first, then `on_request`, then `on_call_tool`. This hierarchy lets you target exactly the right scope—use `on_message` for logging everything, `on_request` for authentication, and `on_call_tool` for tool-specific behavior.

### Hook Signature

Every hook follows the same pattern:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def hook_name(self, context: MiddlewareContext, call_next) -> result_type:
    # Pre-processing
    result = await call_next(context)
    # Post-processing
    return result
```

**Parameters:**

* `context` — `MiddlewareContext` containing request information
* `call_next` — Async function to continue the middleware chain

**Returns:** The appropriate result type for the hook (varies by operation).

### MiddlewareContext

The `context` parameter provides access to request details:

| Attribute         | Type       | Description                                   |
| ----------------- | ---------- | --------------------------------------------- |
| `method`          | `str`      | MCP method name (e.g., `"tools/call"`)        |
| `source`          | `str`      | Origin: `"client"` or `"server"`              |
| `type`            | `str`      | Message type: `"request"` or `"notification"` |
| `message`         | `object`   | The MCP message data                          |
| `timestamp`       | `datetime` | When the request was received                 |
| `fastmcp_context` | `Context`  | FastMCP context object (if available)         |

### Message Hooks

#### on\_message

Called for every MCP message—both requests and notifications.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_message(self, context: MiddlewareContext, call_next):
    result = await call_next(context)
    return result
```

Use for: Logging, metrics, or any cross-cutting concern that applies to all traffic.

#### on\_request

Called for MCP requests that expect a response.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_request(self, context: MiddlewareContext, call_next):
    result = await call_next(context)
    return result
```

Use for: Authentication, authorization, request validation.

#### on\_notification

Called for fire-and-forget MCP notifications.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_notification(self, context: MiddlewareContext, call_next):
    await call_next(context)
    # Notifications don't return values
```

Use for: Event logging, async side effects.

### Operation Hooks

#### on\_call\_tool

Called when a tool is executed. The `context.message` contains `name` (tool name) and `arguments` (dict).

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_call_tool(self, context: MiddlewareContext, call_next):
    tool_name = context.message.name
    args = context.message.arguments
    result = await call_next(context)
    return result
```

**Returns:** Tool execution result or raises `ToolError`.

#### on\_read\_resource

Called when a resource is read. The `context.message` contains `uri` (resource URI).

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_read_resource(self, context: MiddlewareContext, call_next):
    uri = context.message.uri
    result = await call_next(context)
    return result
```

**Returns:** Resource content.

#### on\_get\_prompt

Called when a prompt is retrieved. The `context.message` contains `name` (prompt name) and `arguments` (dict).

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_get_prompt(self, context: MiddlewareContext, call_next):
    prompt_name = context.message.name
    result = await call_next(context)
    return result
```

**Returns:** Prompt messages.

#### on\_list\_tools

Called when listing available tools. Returns a list of FastMCP `Tool` objects before MCP conversion.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_list_tools(self, context: MiddlewareContext, call_next):
    tools = await call_next(context)
    # Filter or modify the tool list
    return tools
```

**Returns:** `list[Tool]` — Can be filtered before returning to client.

#### on\_list\_resources

Called when listing available resources. Returns FastMCP `Resource` objects.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_list_resources(self, context: MiddlewareContext, call_next):
    resources = await call_next(context)
    return resources
```

**Returns:** `list[Resource]`

#### on\_list\_resource\_templates

Called when listing resource templates.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_list_resource_templates(self, context: MiddlewareContext, call_next):
    templates = await call_next(context)
    return templates
```

**Returns:** `list[ResourceTemplate]`

#### on\_list\_prompts

Called when listing available prompts.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_list_prompts(self, context: MiddlewareContext, call_next):
    prompts = await call_next(context)
    return prompts
```

**Returns:** `list[Prompt]`

#### on\_initialize

<VersionBadge />

Called when a client connects and initializes the session. This hook cannot modify the initialization response.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from mcp import McpError
from mcp.types import ErrorData

async def on_initialize(self, context: MiddlewareContext, call_next):
    client_info = context.message.params.get("clientInfo", {})
    client_name = client_info.get("name", "unknown")

    # Reject before call_next to send error to client
    if client_name == "blocked-client":
        raise McpError(ErrorData(code=-32000, message="Client not supported"))

    await call_next(context)
    print(f"Client {client_name} initialized")
```

**Returns:** `None` — The initialization response is handled internally by the MCP protocol.

<Warning>
  Raising `McpError` after `call_next()` will only log the error, not send it to the client. The response has already been sent. Always reject **before** `call_next()`.
</Warning>

### Raw Handler

For complete control over all messages, override `__call__` instead of individual hooks:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import Middleware, MiddlewareContext

class RawMiddleware(Middleware):
    async def __call__(self, context: MiddlewareContext, call_next):
        print(f"Processing: {context.method}")
        result = await call_next(context)
        print(f"Completed: {context.method}")
        return result
```

This bypasses the hook dispatch system entirely. Use when you need uniform handling regardless of message type.

### Session Availability

<VersionBadge />

The MCP session may not be available during certain phases like initialization. Check before accessing session-specific attributes:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_request(self, context: MiddlewareContext, call_next):
    ctx = context.fastmcp_context

    if ctx.request_context:
        # MCP session available
        session_id = ctx.session_id
        request_id = ctx.request_id
    else:
        # Session not yet established (e.g., during initialization)
        # Use HTTP helpers if needed
        from fastmcp.server.dependencies import get_http_headers
        headers = get_http_headers()

    return await call_next(context)
```

For HTTP-specific data (headers, client IP) when using HTTP transports, see [HTTP Requests](/servers/context#http-requests).

## Built-in Middleware

FastMCP includes production-ready middleware for common server concerns.

### Logging

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.logging import LoggingMiddleware, StructuredLoggingMiddleware
```

`LoggingMiddleware` provides human-readable request and response logging. `StructuredLoggingMiddleware` outputs JSON-formatted logs for aggregation tools like Datadog or Splunk.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware.logging import LoggingMiddleware

mcp = FastMCP("MyServer")
mcp.add_middleware(LoggingMiddleware(
    include_payloads=True,
    max_payload_length=1000
))
```

| Parameter            | Type     | Default       | Description                          |
| -------------------- | -------- | ------------- | ------------------------------------ |
| `include_payloads`   | `bool`   | `False`       | Log request/response content         |
| `max_payload_length` | `int`    | `500`         | Truncate payloads beyond this length |
| `logger`             | `Logger` | module logger | Custom logger instance               |

### Timing

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.timing import TimingMiddleware, DetailedTimingMiddleware
```

`TimingMiddleware` logs execution duration for all requests. `DetailedTimingMiddleware` provides per-operation timing with separate tracking for tools, resources, and prompts.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware.timing import TimingMiddleware

mcp = FastMCP("MyServer")
mcp.add_middleware(TimingMiddleware())
```

### Caching

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.caching import ResponseCachingMiddleware
```

Caches tool calls, resource reads, and list operations with TTL-based expiration.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware.caching import ResponseCachingMiddleware

mcp = FastMCP("MyServer")
mcp.add_middleware(ResponseCachingMiddleware())
```

Each operation type can be configured independently using settings classes:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.caching import (
    ResponseCachingMiddleware,
    CallToolSettings,
    ListToolsSettings,
    ReadResourceSettings
)

mcp.add_middleware(ResponseCachingMiddleware(
    list_tools_settings=ListToolsSettings(ttl=30),
    call_tool_settings=CallToolSettings(included_tools=["expensive_tool"]),
    read_resource_settings=ReadResourceSettings(enabled=False)
))
```

| Settings Class          | Configures                  |
| ----------------------- | --------------------------- |
| `ListToolsSettings`     | `on_list_tools` caching     |
| `CallToolSettings`      | `on_call_tool` caching      |
| `ListResourcesSettings` | `on_list_resources` caching |
| `ReadResourceSettings`  | `on_read_resource` caching  |
| `ListPromptsSettings`   | `on_list_prompts` caching   |
| `GetPromptSettings`     | `on_get_prompt` caching     |

Each settings class accepts:

* `enabled` — Enable/disable caching for this operation
* `ttl` — Time-to-live in seconds
* `included_*` / `excluded_*` — Whitelist or blacklist specific items

For persistence or distributed deployments, configure a different storage backend:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.caching import ResponseCachingMiddleware
from key_value.aio.stores.disk import DiskStore

mcp.add_middleware(ResponseCachingMiddleware(
    cache_storage=DiskStore(directory="cache")
))
```

See [Storage Backends](/servers/storage-backends) for complete options.

<Note>
  Cache keys are based on the operation name and arguments only — they do not include user or session identity. If your tools return user-specific data derived from auth context (e.g., headers or session state) rather than from the request arguments, you should either disable caching for those tools or ensure user identity is part of the tool arguments.
</Note>

### Rate Limiting

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.rate_limiting import (
    RateLimitingMiddleware,
    SlidingWindowRateLimitingMiddleware
)
```

`RateLimitingMiddleware` uses a token bucket algorithm allowing controlled bursts. `SlidingWindowRateLimitingMiddleware` provides precise time-window rate limiting without burst allowance.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

mcp = FastMCP("MyServer")
mcp.add_middleware(RateLimitingMiddleware(
    max_requests_per_second=10.0,
    burst_capacity=20
))
```

| Parameter                 | Type       | Default | Description                  |
| ------------------------- | ---------- | ------- | ---------------------------- |
| `max_requests_per_second` | `float`    | `10.0`  | Sustained request rate       |
| `burst_capacity`          | `int`      | `20`    | Maximum burst size           |
| `client_id_func`          | `Callable` | `None`  | Custom client identification |

For sliding window rate limiting:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.rate_limiting import SlidingWindowRateLimitingMiddleware

mcp.add_middleware(SlidingWindowRateLimitingMiddleware(
    max_requests=100,
    window_minutes=1
))
```

### Error Handling

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware, RetryMiddleware
```

`ErrorHandlingMiddleware` provides centralized error logging and transformation. `RetryMiddleware` automatically retries with exponential backoff for transient failures.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware

mcp = FastMCP("MyServer")
mcp.add_middleware(ErrorHandlingMiddleware(
    include_traceback=True,
    transform_errors=True,
    error_callback=my_error_callback
))
```

| Parameter           | Type       | Default | Description                      |
| ------------------- | ---------- | ------- | -------------------------------- |
| `include_traceback` | `bool`     | `False` | Include stack traces in logs     |
| `transform_errors`  | `bool`     | `False` | Convert exceptions to MCP errors |
| `error_callback`    | `Callable` | `None`  | Custom callback on errors        |

For automatic retries:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.error_handling import RetryMiddleware

mcp.add_middleware(RetryMiddleware(
    max_retries=3,
    retry_exceptions=(ConnectionError, TimeoutError)
))
```

### Ping

<VersionBadge />

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import PingMiddleware
```

Keeps long-lived connections alive by sending periodic pings.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware import PingMiddleware

mcp = FastMCP("MyServer")
mcp.add_middleware(PingMiddleware(interval_ms=5000))
```

| Parameter     | Type  | Default | Description                   |
| ------------- | ----- | ------- | ----------------------------- |
| `interval_ms` | `int` | `30000` | Ping interval in milliseconds |

The ping task starts on the first message and stops automatically when the session ends. Most useful for stateful HTTP connections; has no effect on stateless connections.

### Tool Injection

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.tool_injection import (
    ToolInjectionMiddleware,
    PromptToolMiddleware,
    ResourceToolMiddleware
)
```

`ToolInjectionMiddleware` dynamically injects tools during request processing. `PromptToolMiddleware` and `ResourceToolMiddleware` provide compatibility layers for clients that cannot list or access prompts and resources directly—they expose those capabilities as tools.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.server.middleware.tool_injection import ToolInjectionMiddleware

def my_tool_fn(a: int, b: int) -> int:
    return a + b

my_tool = Tool.from_function(fn=my_tool_fn, name="my_tool")
mcp.add_middleware(ToolInjectionMiddleware(tools=[my_tool]))
```

### Response Limiting

<VersionBadge />

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.response_limiting import ResponseLimitingMiddleware
```

Large tool responses can overwhelm LLM context windows or cause memory issues. You can add response-limiting middleware to enforce size constraints on tool outputs.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware.response_limiting import ResponseLimitingMiddleware

mcp = FastMCP("MyServer")

# Limit all tool responses to 500KB
mcp.add_middleware(ResponseLimitingMiddleware(max_size=500_000))

@mcp.tool
def search(query: str) -> str:
    # This could return a very large result
    return "x" * 1_000_000  # 1MB response

# When called, the response will be truncated to ~500KB with:
# "...\n\n[Response truncated due to size limit]"
```

When a response exceeds the limit, the middleware extracts all text content, joins it together, truncates to fit within the limit, and returns a single `TextContent` block. For non-text responses, the serialized JSON is used as the text source.

<Note>
  If a tool defines an `output_schema`, truncated responses will no longer conform to that schema — the client will receive a plain `TextContent` block instead of the expected structured output. Keep this in mind when setting size limits for tools with structured responses.
</Note>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Limit only specific tools
mcp.add_middleware(ResponseLimitingMiddleware(
    max_size=100_000,
    tools=["search", "fetch_data"],
))
```

| Parameter           | Type                | Default                                        | Description                                  |
| ------------------- | ------------------- | ---------------------------------------------- | -------------------------------------------- |
| `max_size`          | `int`               | `1_000_000`                                    | Maximum response size in bytes (1MB default) |
| `truncation_suffix` | `str`               | `"\n\n[Response truncated due to size limit]"` | Suffix appended to truncated responses       |
| `tools`             | `list[str] \| None` | `None`                                         | Limit only these tools (None = all tools)    |

### Combining Middleware

Order matters. Place middleware that should run first (on the way in) earliest:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware

mcp = FastMCP("Production Server")

mcp.add_middleware(ErrorHandlingMiddleware())   # Catch all errors
mcp.add_middleware(RateLimitingMiddleware(max_requests_per_second=50))
mcp.add_middleware(TimingMiddleware())
mcp.add_middleware(LoggingMiddleware())

@mcp.tool
def my_tool(data: str) -> str:
    return f"Processed: {data}"
```

## Custom Middleware

When the built-in middleware doesn't fit your needs—custom authentication schemes, domain-specific logging, or request transformation—subclass `Middleware` and override the hooks you need.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext

class CustomMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        # Pre-processing
        print(f"→ {context.method}")

        result = await call_next(context)

        # Post-processing
        print(f"← {context.method}")
        return result

mcp = FastMCP("MyServer")
mcp.add_middleware(CustomMiddleware())
```

Override only the hooks relevant to your use case. Unoverridden hooks pass through automatically.

### Denying Requests

Raise the appropriate error type to stop processing and return an error to the client.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import ToolError

class AuthMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        tool_name = context.message.name

        if tool_name in ["delete_all", "admin_config"]:
            raise ToolError("Access denied: requires admin privileges")

        return await call_next(context)
```

| Operation        | Error Type      |
| ---------------- | --------------- |
| Tool calls       | `ToolError`     |
| Resource reads   | `ResourceError` |
| Prompt retrieval | `PromptError`   |
| General requests | `McpError`      |

Do not return error values or skip `call_next()` to indicate errors—raise exceptions for proper error propagation.

### Modifying Requests

Change the message before passing it down the chain.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import Middleware, MiddlewareContext

class InputSanitizer(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        if context.message.name == "search":
            # Normalize search query
            query = context.message.arguments.get("query", "")
            context.message.arguments["query"] = query.strip().lower()

        return await call_next(context)
```

### Modifying Responses

Transform results after the handler executes.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import Middleware, MiddlewareContext

class ResponseEnricher(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        result = await call_next(context)

        if context.message.name == "get_data" and result.structured_content:
            result.structured_content["processed_by"] = "enricher"

        return result
```

For more complex tool transformations, consider [Transforms](/servers/transforms/transforms) instead.

### Filtering Lists

List operations return FastMCP objects that you can filter before they reach the client. When filtering list results, also block execution in the corresponding operation hook to maintain consistency:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import ToolError

class PrivateToolFilter(Middleware):
    async def on_list_tools(self, context: MiddlewareContext, call_next):
        tools = await call_next(context)
        return [tool for tool in tools if "private" not in tool.tags]

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        if context.fastmcp_context:
            tool = await context.fastmcp_context.fastmcp.get_tool(context.message.name)
            if "private" in tool.tags:
                raise ToolError("Tool not found")

        return await call_next(context)
```

### Accessing Component Metadata

During execution hooks, component metadata (like tags) isn't directly available. Look up the component through the server:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import ToolError

class TagBasedAuth(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        if context.fastmcp_context:
            try:
                tool = await context.fastmcp_context.fastmcp.get_tool(context.message.name)

                if "requires-auth" in tool.tags:
                    # Check authentication here
                    pass

            except Exception:
                pass  # Let execution handle missing tools

        return await call_next(context)
```

The same pattern works for resources and prompts:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resource = await context.fastmcp_context.fastmcp.get_resource(context.message.uri)
prompt = await context.fastmcp_context.fastmcp.get_prompt(context.message.name)
```

### Storing State

<VersionBadge />

Middleware can store state that tools access later through the FastMCP context.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import Middleware, MiddlewareContext

class UserMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        # Extract user from headers (HTTP transport)
        from fastmcp.server.dependencies import get_http_headers
        headers = get_http_headers() or {}
        user_id = headers.get("x-user-id", "anonymous")

        # Store for tools to access
        if context.fastmcp_context:
            context.fastmcp_context.set_state("user_id", user_id)

        return await call_next(context)
```

Tools retrieve the state:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context

mcp = FastMCP("MyServer")

@mcp.tool
def get_user_data(ctx: Context) -> str:
    user_id = ctx.get_state("user_id")
    return f"Data for user: {user_id}"
```

See [Context State Management](/servers/context#state-management) for details.

### Constructor Parameters

Initialize middleware with configuration:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import Middleware, MiddlewareContext

class ConfigurableMiddleware(Middleware):
    def __init__(self, api_key: str, rate_limit: int = 100):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.request_counts = {}

    async def on_request(self, context: MiddlewareContext, call_next):
        # Use self.api_key, self.rate_limit, etc.
        return await call_next(context)

mcp.add_middleware(ConfigurableMiddleware(
    api_key="secret",
    rate_limit=50
))
```

### Error Handling in Custom Middleware

Wrap `call_next()` to handle errors from downstream middleware and handlers.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware import Middleware, MiddlewareContext

class ErrorLogger(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        try:
            return await call_next(context)
        except Exception as e:
            print(f"Error in {context.method}: {type(e).__name__}: {e}")
            raise  # Re-raise to let error propagate
```

Catching and not re-raising suppresses the error entirely. Usually you want to log and re-raise.

### Complete Example

Authentication middleware checking API keys for specific tools:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from fastmcp.exceptions import ToolError

class ApiKeyAuth(Middleware):
    def __init__(self, valid_keys: set[str], protected_tools: set[str]):
        self.valid_keys = valid_keys
        self.protected_tools = protected_tools

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        tool_name = context.message.name

        if tool_name not in self.protected_tools:
            return await call_next(context)

        headers = get_http_headers() or {}
        api_key = headers.get("x-api-key")

        if api_key not in self.valid_keys:
            raise ToolError(f"Invalid API key for protected tool: {tool_name}")

        return await call_next(context)

mcp = FastMCP("Secure Server")
mcp.add_middleware(ApiKeyAuth(
    valid_keys={"key-1", "key-2"},
    protected_tools={"delete_user", "admin_panel"}
))

@mcp.tool
def delete_user(user_id: str) -> str:
    return f"Deleted user {user_id}"

@mcp.tool
def get_user(user_id: str) -> str:
    return f"User {user_id}"  # Not protected
```


# Pagination
Source: https://gofastmcp.com/servers/pagination

Control how servers return large lists of components to clients.

<VersionBadge />

When a server exposes many tools, resources, or prompts, returning them all in a single response can be impractical. MCP supports pagination for list operations, allowing servers to return results in manageable chunks that clients can fetch incrementally.

## Server Configuration

By default, FastMCP servers return all components in a single response for backward compatibility. To enable pagination, set the `list_page_size` parameter when creating your server. This value determines the maximum number of items returned per page across all list operations.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

# Enable pagination with 50 items per page
server = FastMCP("ComponentRegistry", list_page_size=50)

# Register tools (in practice, these might come from a database or config)
@server.tool
def search(query: str) -> str:
    return f"Results for: {query}"

@server.tool
def analyze(data: str) -> dict:
    return {"status": "analyzed", "data": data}

# ... many more tools, resources, prompts
```

When `list_page_size` is configured, the `tools/list`, `resources/list`, `resources/templates/list`, and `prompts/list` endpoints all paginate their responses. Each response includes a `nextCursor` field when more results exist, which clients use to fetch subsequent pages.

### Cursor Format

Cursors are opaque base64-encoded strings per the MCP specification. Clients should treat them as black boxes, passing them unchanged between requests. The cursor encodes the offset into the result set, but this is an implementation detail that may change.

## Client Behavior

The FastMCP Client handles pagination transparently. Convenience methods like `list_tools()`, `list_resources()`, `list_resource_templates()`, and `list_prompts()` automatically fetch all pages and return the complete list. Existing code continues to work without modification.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

async with Client(server) as client:
    # Returns all 200 tools, fetching pages automatically
    tools = await client.list_tools()
    print(f"Total tools: {len(tools)}")  # 200
```

### Manual Pagination

For scenarios where you want to process results incrementally (memory-constrained environments, progress reporting, or early termination), use the `_mcp` variants with explicit cursor handling.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

async with Client(server) as client:
    # Fetch first page
    result = await client.list_tools_mcp()
    print(f"Page 1: {len(result.tools)} tools")

    # Continue fetching while more pages exist
    while result.nextCursor:
        result = await client.list_tools_mcp(cursor=result.nextCursor)
        print(f"Next page: {len(result.tools)} tools")
```

The `_mcp` methods return the raw MCP protocol objects, which include both the items and the `nextCursor` for the next page. When `nextCursor` is `None`, you've reached the end of the result set.

All four list operations support manual pagination:

| Operation          | Convenience Method          | Manual Method                             |
| ------------------ | --------------------------- | ----------------------------------------- |
| Tools              | `list_tools()`              | `list_tools_mcp(cursor=...)`              |
| Resources          | `list_resources()`          | `list_resources_mcp(cursor=...)`          |
| Resource Templates | `list_resource_templates()` | `list_resource_templates_mcp(cursor=...)` |
| Prompts            | `list_prompts()`            | `list_prompts_mcp(cursor=...)`            |

## When to Use Pagination

Pagination becomes valuable when your server exposes a large number of components. Consider enabling it when:

* Your server dynamically generates many components (e.g., from a database or file system)
* Memory usage is a concern for clients
* You want to reduce initial response latency

For servers with a fixed, modest number of components (fewer than 100), pagination adds complexity without meaningful benefit. The default behavior of returning everything in one response is simpler and efficient for typical use cases.


# Progress Reporting
Source: https://gofastmcp.com/servers/progress

Update clients on the progress of long-running operations through the MCP context.

Progress reporting allows MCP tools to notify clients about the progress of long-running operations. Clients can display progress indicators and provide better user experience during time-consuming tasks.

## Basic Usage

Use `ctx.report_progress()` to send progress updates to the client. The method accepts a `progress` value representing how much work is complete, and an optional `total` representing the full scope of work.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context
import asyncio

mcp = FastMCP("ProgressDemo")

@mcp.tool
async def process_items(items: list[str], ctx: Context) -> dict:
    """Process a list of items with progress updates."""
    total = len(items)
    results = []

    for i, item in enumerate(items):
        await ctx.report_progress(progress=i, total=total)
        await asyncio.sleep(0.1)
        results.append(item.upper())

    await ctx.report_progress(progress=total, total=total)
    return {"processed": len(results), "results": results}
```

## Progress Patterns

| Pattern       | Description                      | Example                           |
| ------------- | -------------------------------- | --------------------------------- |
| Percentage    | Progress as 0-100 percentage     | `progress=75, total=100`          |
| Absolute      | Completed items of a known count | `progress=3, total=10`            |
| Indeterminate | Progress without known endpoint  | `progress=files_found` (no total) |

For multi-stage operations, map each stage to a portion of the total progress range. A four-stage operation might allocate 0-25% to validation, 25-60% to export, 60-80% to transform, and 80-100% to import.

## Client Requirements

Progress reporting requires clients to support progress handling. Clients must send a `progressToken` in the initial request to receive progress updates. If no progress token is provided, progress calls have no effect (they don't error).

See [Client Progress](/clients/progress) for details on implementing client-side progress handling.


# Prompts
Source: https://gofastmcp.com/servers/prompts

Create reusable, parameterized prompt templates for MCP clients.

Prompts are reusable message templates that help LLMs generate structured, purposeful responses. FastMCP simplifies defining these templates, primarily using the `@mcp.prompt` decorator.

## What Are Prompts?

Prompts provide parameterized message templates for LLMs. When a client requests a prompt:

1. FastMCP finds the corresponding prompt definition.
2. If it has parameters, they are validated against your function signature.
3. Your function executes with the validated inputs.
4. The generated message(s) are returned to the LLM to guide its response.

This allows you to define consistent, reusable templates that LLMs can use across different clients and contexts.

## Prompts

### The `@prompt` Decorator

The most common way to define a prompt is by decorating a Python function. The decorator uses the function name as the prompt's identifier.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.prompts import Message

mcp = FastMCP(name="PromptServer")

# Basic prompt returning a string (converted to user message automatically)
@mcp.prompt
def ask_about_topic(topic: str) -> str:
    """Generates a user message asking for an explanation of a topic."""
    return f"Can you please explain the concept of '{topic}'?"

# Prompt returning multiple messages
@mcp.prompt
def generate_code_request(language: str, task_description: str) -> list[Message]:
    """Generates a conversation for code generation."""
    return [
        Message(f"Write a {language} function that performs the following task: {task_description}"),
        Message("I'll help you write that function.", role="assistant"),
    ]
```

**Key Concepts:**

* **Name:** By default, the prompt name is taken from the function name.
* **Parameters:** The function parameters define the inputs needed to generate the prompt.
* **Inferred Metadata:** By default:
  * Prompt Name: Taken from the function name (`ask_about_topic`).
  * Prompt Description: Taken from the function's docstring.

<Tip>
  Functions with `*args` or `**kwargs` are not supported as prompts. This restriction exists because FastMCP needs to generate a complete parameter schema for the MCP protocol, which isn't possible with variable argument lists.
</Tip>

#### Decorator Arguments

While FastMCP infers the name and description from your function, you can override these and add additional metadata using arguments to the `@mcp.prompt` decorator:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.prompt(
    name="analyze_data_request",          # Custom prompt name
    description="Creates a request to analyze data with specific parameters",  # Custom description
    tags={"analysis", "data"},            # Optional categorization tags
    meta={"version": "1.1", "author": "data-team"}  # Custom metadata
)
def data_analysis_prompt(
    data_uri: str = Field(description="The URI of the resource containing the data."),
    analysis_type: str = Field(default="summary", description="Type of analysis.")
) -> str:
    """This docstring is ignored when description is provided."""
    return f"Please perform a '{analysis_type}' analysis on the data found at {data_uri}."
```

<Card icon="code" title="@prompt Decorator Arguments">
  <ParamField type="str | None">
    Sets the explicit prompt name exposed via MCP. If not provided, uses the function name
  </ParamField>

  <ParamField type="str | None">
    A human-readable title for the prompt
  </ParamField>

  <ParamField type="str | None">
    Provides the description exposed via MCP. If set, the function's docstring is ignored for this purpose
  </ParamField>

  <ParamField type="set[str] | None">
    A set of strings used to categorize the prompt. These can be used by the server and, in some cases, by clients to filter or group available prompts.
  </ParamField>

  <ParamField type="bool">
    <Warning>Deprecated in v3.0.0. Use `mcp.enable()` / `mcp.disable()` at the server level instead.</Warning>
    A boolean to enable or disable the prompt. See [Component Visibility](#component-visibility) for the recommended approach.
  </ParamField>

  <ParamField type="list[Icon] | None">
    <VersionBadge />

    Optional list of icon representations for this prompt. See [Icons](/servers/icons) for detailed examples
  </ParamField>

  <ParamField type="dict[str, Any] | None">
    <VersionBadge />

    Optional meta information about the prompt. This data is passed through to the MCP client as the `meta` field of the client-side prompt object and can be used for custom metadata, versioning, or other application-specific purposes.
  </ParamField>

  <ParamField type="str | int | None">
    <VersionBadge />

    Optional version identifier for this prompt. See [Versioning](/servers/versioning) for details.
  </ParamField>
</Card>

#### Using with Methods

For decorating instance or class methods, use the standalone `@prompt` decorator and register the bound method. See [Tools: Using with Methods](/servers/tools#using-with-methods) for the pattern.

### Argument Types

<VersionBadge />

The MCP specification requires that all prompt arguments be passed as strings, but FastMCP allows you to use typed annotations for better developer experience. When you use complex types like `list[int]` or `dict[str, str]`, FastMCP:

1. **Automatically converts** string arguments from MCP clients to the expected types
2. **Generates helpful descriptions** showing the exact JSON string format needed
3. **Preserves direct usage** - you can still call prompts with properly typed arguments

Since the MCP specification only allows string arguments, clients need to know what string format to use for complex types. FastMCP solves this by automatically enhancing the argument descriptions with JSON schema information, making it clear to both humans and LLMs how to format their arguments.

<CodeGroup>
  ```python Python Code theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  @mcp.prompt
  def analyze_data(
      numbers: list[int],
      metadata: dict[str, str], 
      threshold: float
  ) -> str:
      """Analyze numerical data."""
      avg = sum(numbers) / len(numbers)
      return f"Average: {avg}, above threshold: {avg > threshold}"
  ```

  ```json Resulting MCP Prompt theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  {
    "name": "analyze_data",
    "description": "Analyze numerical data.",
    "arguments": [
      {
        "name": "numbers",
        "description": "Provide as a JSON string matching the following schema: {\"items\":{\"type\":\"integer\"},\"type\":\"array\"}",
        "required": true
      },
      {
        "name": "metadata", 
        "description": "Provide as a JSON string matching the following schema: {\"additionalProperties\":{\"type\":\"string\"},\"type\":\"object\"}",
        "required": true
      },
      {
        "name": "threshold",
        "description": "Provide as a JSON string matching the following schema: {\"type\":\"number\"}",
        "required": true
      }
    ]
  }
  ```
</CodeGroup>

**MCP clients will call this prompt with string arguments:**

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
{
  "numbers": "[1, 2, 3, 4, 5]",
  "metadata": "{\"source\": \"api\", \"version\": \"1.0\"}",
  "threshold": "2.5"
}
```

**But you can still call it directly with proper types:**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# This also works for direct calls
result = await prompt.render({
    "numbers": [1, 2, 3, 4, 5],
    "metadata": {"source": "api", "version": "1.0"}, 
    "threshold": 2.5
})
```

<Warning>
  Keep your type annotations simple when using this feature. Complex nested types or custom classes may not convert reliably from JSON strings. The automatically generated schema descriptions are the only guidance users receive about the expected format.

  Good choices: `list[int]`, `dict[str, str]`, `float`, `bool`
  Avoid: Complex Pydantic models, deeply nested structures, custom classes
</Warning>

### Return Values

Prompt functions must return one of these types:

* **`str`**: Sent as a single user message.
* **`list[Message | str]`**: A sequence of messages (a conversation). Strings are auto-converted to user Messages.
* **`PromptResult`**: Full control over messages, description, and metadata. See [PromptResult](#promptresult) below.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.prompts import Message

@mcp.prompt
def roleplay_scenario(character: str, situation: str) -> list[Message]:
    """Sets up a roleplaying scenario with initial messages."""
    return [
        Message(f"Let's roleplay. You are {character}. The situation is: {situation}"),
        Message("Okay, I understand. I am ready. What happens next?", role="assistant")
    ]
```

#### Message

<VersionBadge />

`Message` provides a user-friendly wrapper for prompt messages with automatic serialization.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.prompts import Message

# String content (user role by default)
Message("Hello, world!")

# Explicit role
Message("I can help with that.", role="assistant")

# Auto-serialized to JSON text
Message({"key": "value"})
Message(["item1", "item2"])
```

`Message` accepts two fields:

**`content`** - The message content. Strings pass through directly. Other types (dict, list, BaseModel) are automatically JSON-serialized to text.

**`role`** - The message role, either `"user"` (default) or `"assistant"`.

<Card title="Message">
  <ParamField type="Any">
    The content data. Strings pass through directly. Other types (dict, list, BaseModel) are automatically JSON-serialized.
  </ParamField>

  <ParamField type="Literal['user', 'assistant']">
    The message role.
  </ParamField>
</Card>

#### PromptResult

<VersionBadge />

`PromptResult` gives you explicit control over prompt responses: multiple messages, roles, and metadata at both the message and result level.

````python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.prompts import PromptResult, Message

mcp = FastMCP(name="PromptServer")

@mcp.prompt
def code_review(code: str) -> PromptResult:
    """Returns a code review prompt with metadata."""
    return PromptResult(
        messages=[
            Message(f"Please review this code:\n\n```\n{code}\n```"),
            Message("I'll analyze this code for issues.", role="assistant"),
        ],
        description="Code review prompt",
        meta={"review_type": "security", "priority": "high"}
    )
````

For simple cases, you can pass a string directly to `PromptResult`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
return PromptResult("Please help me with this task")  # auto-converts to single Message
```

<Card title="PromptResult">
  <ParamField type="str | list[Message]">
    Messages to return. Strings are wrapped as a single user Message.
  </ParamField>

  <ParamField type="str | None">
    Optional description of the prompt result. If not provided, defaults to the prompt's docstring.
  </ParamField>

  <ParamField type="dict[str, Any] | None">
    Result-level metadata, included in the MCP response's `_meta` field. Use this for runtime metadata like categorization, priority, or other client-specific data.
  </ParamField>
</Card>

<Note>
  The `meta` field in `PromptResult` is for runtime metadata specific to this render response. This is separate from the `meta` parameter in `@mcp.prompt(meta={...})`, which provides static metadata about the prompt definition itself (returned when listing prompts).
</Note>

You can still return plain `str` or `list[Message | str]` from your prompt functions—`PromptResult` is opt-in for when you need to include metadata.

### Required vs. Optional Parameters

Parameters in your function signature are considered **required** unless they have a default value.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.prompt
def data_analysis_prompt(
    data_uri: str,                        # Required - no default value
    analysis_type: str = "summary",       # Optional - has default value
    include_charts: bool = False          # Optional - has default value
) -> str:
    """Creates a request to analyze data with specific parameters."""
    prompt = f"Please perform a '{analysis_type}' analysis on the data found at {data_uri}."
    if include_charts:
        prompt += " Include relevant charts and visualizations."
    return prompt
```

In this example, the client *must* provide `data_uri`. If `analysis_type` or `include_charts` are omitted, their default values will be used.

### Component Visibility

<VersionBadge />

You can control which prompts are enabled for clients using server-level enabled control. Disabled prompts don't appear in `list_prompts` and can't be called.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.prompt(tags={"public"})
def public_prompt(topic: str) -> str:
    return f"Discuss: {topic}"

@mcp.prompt(tags={"internal"})
def internal_prompt() -> str:
    return "Internal system prompt"

# Disable specific prompts by key
mcp.disable(keys={"prompt:internal_prompt"})

# Disable prompts by tag
mcp.disable(tags={"internal"})

# Or use allowlist mode - only enable prompts with specific tags
mcp.enable(tags={"public"}, only=True)
```

See [Visibility](/servers/visibility) for the complete visibility control API including key formats, tag-based filtering, and provider-level control.

### Async Prompts

FastMCP supports both standard (`def`) and asynchronous (`async def`) functions as prompts. Synchronous functions automatically run in a threadpool to avoid blocking the event loop.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Synchronous prompt (runs in threadpool)
@mcp.prompt
def simple_question(question: str) -> str:
    """Generates a simple question to ask the LLM."""
    return f"Question: {question}"

# Asynchronous prompt
@mcp.prompt
async def data_based_prompt(data_id: str) -> str:
    """Generates a prompt based on data that needs to be fetched."""
    # In a real scenario, you might fetch data from a database or API
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/data/{data_id}") as response:
            data = await response.json()
            return f"Analyze this data: {data['content']}"
```

Use `async def` when your prompt function performs I/O operations like network requests or database queries, since async is more efficient than threadpool dispatch.

### Accessing MCP Context

<VersionBadge />

Prompts can access additional MCP information and features through the `Context` object. To access it, add a parameter to your prompt function with a type annotation of `Context`:

```python {6} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context

mcp = FastMCP(name="PromptServer")

@mcp.prompt
async def generate_report_request(report_type: str, ctx: Context) -> str:
    """Generates a request for a report."""
    return f"Please create a {report_type} report. Request ID: {ctx.request_id}"
```

For full documentation on the Context object and all its capabilities, see the [Context documentation](/servers/context).

### Notifications

<VersionBadge />

FastMCP automatically sends `notifications/prompts/list_changed` notifications to connected clients when prompts are added, enabled, or disabled. This allows clients to stay up-to-date with the current prompt set without manually polling for changes.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.prompt
def example_prompt() -> str:
    return "Hello!"

# These operations trigger notifications:
mcp.add_prompt(example_prompt)               # Sends prompts/list_changed notification
mcp.disable(keys={"prompt:example_prompt"})  # Sends prompts/list_changed notification
mcp.enable(keys={"prompt:example_prompt"})   # Sends prompts/list_changed notification
```

Notifications are only sent when these operations occur within an active MCP request context (e.g., when called from within a tool or other MCP operation). Operations performed during server initialization do not trigger notifications.

Clients can handle these notifications using a [message handler](/clients/notifications) to automatically refresh their prompt lists or update their interfaces.

## Server Behavior

### Duplicate Prompts

<VersionBadge />

You can configure how the FastMCP server handles attempts to register multiple prompts with the same name. Use the `on_duplicate_prompts` setting during `FastMCP` initialization.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP(
    name="PromptServer",
    on_duplicate_prompts="error"  # Raise an error if a prompt name is duplicated
)

@mcp.prompt
def greeting(): return "Hello, how can I help you today?"

# This registration attempt will raise a ValueError because
# "greeting" is already registered and the behavior is "error".
# @mcp.prompt
# def greeting(): return "Hi there! What can I do for you?"
```

The duplicate behavior options are:

* `"warn"` (default): Logs a warning, and the new prompt replaces the old one.
* `"error"`: Raises a `ValueError`, preventing the duplicate registration.
* `"replace"`: Silently replaces the existing prompt with the new one.
* `"ignore"`: Keeps the original prompt and ignores the new registration attempt.

## Versioning

<VersionBadge />

Prompts support versioning, allowing you to maintain multiple implementations under the same name while clients automatically receive the highest version. See [Versioning](/servers/versioning) for complete documentation on version comparison, retrieval, and migration patterns.
