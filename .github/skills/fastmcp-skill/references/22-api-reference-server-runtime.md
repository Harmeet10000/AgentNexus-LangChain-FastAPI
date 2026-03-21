# API Reference: Server Runtime

Source lines: 38289-43132 from the original FastMCP documentation dump.

Package-level API reference for context, dependencies, elicitation, event store, HTTP, lifespan, middleware, OpenAPI, providers, and proxy support.

---

# context
Source: https://gofastmcp.com/python-sdk/fastmcp-server-context



# `fastmcp.server.context`

## Functions

### `set_transport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L88"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_transport(transport: TransportType) -> Token[TransportType | None]
```

Set the current transport type. Returns token for reset.

### `reset_transport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L95"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
reset_transport(token: Token[TransportType | None]) -> None
```

Reset transport to previous value.

### `set_context` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L125"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_context(context: Context) -> Generator[Context, None, None]
```

## Classes

### `LogData` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L101"><Icon icon="github" /></a></sup>

Data object for passing log arguments to client-side handlers.

This provides an interface to match the Python standard library logging,
for compatibility with structured logging.

### `Context` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L134"><Icon icon="github" /></a></sup>

Context object providing access to MCP capabilities.

This provides a cleaner interface to MCP's RequestContext functionality.
It gets injected into tool and resource functions that request it via type hints.

To use context in a tool function, add a parameter with the Context type annotation:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@server.tool
async def my_tool(x: int, ctx: Context) -> str:
    # Log messages to the client
    await ctx.info(f"Processing {x}")
    await ctx.debug("Debug info")
    await ctx.warning("Warning message")
    await ctx.error("Error message")

    # Report progress
    await ctx.report_progress(50, 100, "Processing")

    # Access resources
    data = await ctx.read_resource("resource://data")

    # Get request info
    request_id = ctx.request_id
    client_id = ctx.client_id

    # Manage state across the session (persists across requests)
    await ctx.set_state("key", "value")
    value = await ctx.get_state("key")

    # Store non-serializable values for the current request only
    await ctx.set_state("client", http_client, serializable=False)

    return str(x)
```

State Management:
Context provides session-scoped state that persists across requests within
the same MCP session. State is automatically keyed by session, ensuring
isolation between different clients.

State set during `on_initialize` middleware will persist to subsequent tool
calls when using the same session object (STDIO, SSE, single-server HTTP).
For distributed/serverless HTTP deployments where different machines handle
the init and tool calls, state is isolated by the mcp-session-id header.

The context parameter name can be anything as long as it's annotated with Context.
The context is optional - tools that don't need it can omit the parameter.

**Methods:**

#### `is_background_task` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L207"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_background_task(self) -> bool
```

True when this context is running in a background task (Docket worker).

When True, certain operations like elicit() and sample() will use
task-aware implementations that can pause the task and wait for
client input.

#### `task_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L226"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
task_id(self) -> str | None
```

Get the background task ID if running in a background task.

Returns None if not running in a background task context.

#### `origin_request_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L234"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
origin_request_id(self) -> str | None
```

Get the request ID that originated this execution, if available.

In foreground request mode, this is the current request\_id.
In background task mode, this is the request\_id captured when the task
was submitted, if one was available.

#### `fastmcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L246"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp(self) -> FastMCP
```

Get the FastMCP instance.

#### `request_context` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L321"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
request_context(self) -> RequestContext[ServerSession, Any, Request] | None
```

Access to the underlying request context.

Returns None when the MCP session has not been established yet.
Returns the full RequestContext once the MCP session is available.

For HTTP request access in middleware, use `get_http_request()` from fastmcp.server.dependencies,
which works whether or not the MCP session is available.

Example in middleware:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def on_request(self, context, call_next):
    ctx = context.fastmcp_context
    if ctx.request_context:
        # MCP session available - can access session_id, request_id, etc.
        session_id = ctx.session_id
    else:
        # MCP session not available yet - use HTTP helpers
        from fastmcp.server.dependencies import get_http_request
        request = get_http_request()
    return await call_next(context)
```

#### `lifespan_context` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L350"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
lifespan_context(self) -> dict[str, Any]
```

Access the server's lifespan context.

Returns the context dict yielded by the server's lifespan function.
Returns an empty dict if no lifespan was configured or if the MCP
session is not yet established.

In background tasks (Docket workers), where request\_context is not
available, falls back to reading from the FastMCP server's lifespan
result directly.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@server.tool
def my_tool(ctx: Context) -> str:
    db = ctx.lifespan_context.get("db")
    if db:
        return db.query("SELECT 1")
    return "No database connection"
```

#### `report_progress` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L381"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
report_progress(self, progress: float, total: float | None = None, message: str | None = None) -> None
```

Report progress for the current operation.

Works in both foreground (MCP progress notifications) and background
(Docket task execution) contexts.

**Args:**

* `progress`: Current progress value e.g. 24
* `total`: Optional total value e.g. 100
* `message`: Optional status message describing current progress

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L474"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self) -> list[SDKResource]
```

List all available resources from the server.

**Returns:**

* List of Resource objects available on the server

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L490"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self) -> list[SDKPrompt]
```

List all available prompts from the server.

**Returns:**

* List of Prompt objects available on the server

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L506"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> GetPromptResult
```

Get a prompt by name with optional arguments.

**Args:**

* `name`: The name of the prompt to get
* `arguments`: Optional arguments to pass to the prompt

**Returns:**

* The prompt result

#### `read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L525"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource(self, uri: str | AnyUrl) -> ResourceResult
```

Read a resource by URI.

**Args:**

* `uri`: Resource URI to read

**Returns:**

* ResourceResult with contents

#### `log` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L541"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
log(self, message: str, level: LoggingLevel | None = None, logger_name: str | None = None, extra: Mapping[str, Any] | None = None) -> None
```

Send a log message to the client.

Messages sent to Clients are also logged to the `fastmcp.server.context.to_client` logger with a level of `DEBUG`.

**Args:**

* `message`: Log message
* `level`: Optional log level. One of "debug", "info", "notice", "warning", "error", "critical",
  "alert", or "emergency". Default is "info".
* `logger_name`: Optional logger name
* `extra`: Optional mapping for additional arguments

#### `transport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L571"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transport(self) -> TransportType | None
```

Get the current transport type.

Returns the transport type used to run this server: "stdio", "sse",
or "streamable-http". Returns None if called outside of a server context.

#### `client_supports_extension` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L579"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
client_supports_extension(self, extension_id: str) -> bool
```

Check whether the connected client supports a given MCP extension.

Inspects the `extensions` extra field on `ClientCapabilities`
sent by the client during initialization.

Returns `False` when no session is available (e.g., outside a
request context) or when the client did not advertise the extension.

Example::

from fastmcp.server.apps import UI\_EXTENSION\_ID

@mcp.tool
async def my\_tool(ctx: Context) -> str:
if ctx.client\_supports\_extension(UI\_EXTENSION\_ID):
return "UI-capable client"
return "text-only client"

#### `client_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L607"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
client_id(self) -> str | None
```

Get the client ID if available.

#### `request_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L616"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
request_id(self) -> str
```

Get the unique ID for this request.

Raises RuntimeError if MCP request context is not available.

#### `session_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L629"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
session_id(self) -> str
```

Get the MCP session ID for ALL transports.

Returns the session ID that can be used as a key for session-based
data storage (e.g., Redis) to share data between tool calls within
the same client session.

**Returns:**

* The session ID for StreamableHTTP transports, or a generated ID
* for other transports.

#### `session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L686"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
session(self) -> ServerSession
```

Access to the underlying session for advanced usage.

In request mode: Returns the session from the active request context.
In background task mode: Returns the session stored at Context creation.

Raises RuntimeError if no session is available.

#### `debug` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L712"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
debug(self, message: str, logger_name: str | None = None, extra: Mapping[str, Any] | None = None) -> None
```

Send a `DEBUG`-level message to the connected MCP Client.

Messages sent to Clients are also logged to the `fastmcp.server.context.to_client` logger with a level of `DEBUG`.

#### `info` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L728"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
info(self, message: str, logger_name: str | None = None, extra: Mapping[str, Any] | None = None) -> None
```

Send a `INFO`-level message to the connected MCP Client.

Messages sent to Clients are also logged to the `fastmcp.server.context.to_client` logger with a level of `DEBUG`.

#### `warning` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L744"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
warning(self, message: str, logger_name: str | None = None, extra: Mapping[str, Any] | None = None) -> None
```

Send a `WARNING`-level message to the connected MCP Client.

Messages sent to Clients are also logged to the `fastmcp.server.context.to_client` logger with a level of `DEBUG`.

#### `error` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L760"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
error(self, message: str, logger_name: str | None = None, extra: Mapping[str, Any] | None = None) -> None
```

Send a `ERROR`-level message to the connected MCP Client.

Messages sent to Clients are also logged to the `fastmcp.server.context.to_client` logger with a level of `DEBUG`.

#### `list_roots` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L776"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_roots(self) -> list[Root]
```

List the roots available to the server, as indicated by the client.

#### `send_notification` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L781"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
send_notification(self, notification: mcp.types.ServerNotificationType) -> None
```

Send a notification to the client immediately.

**Args:**

* `notification`: An MCP notification instance (e.g., ToolListChangedNotification())

#### `close_sse_stream` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L791"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
close_sse_stream(self) -> None
```

Close the current response stream to trigger client reconnection.

When using StreamableHTTP transport with an EventStore configured, this
method gracefully closes the HTTP connection for the current request.
The client will automatically reconnect (after `retry_interval` milliseconds)
and resume receiving events from where it left off via the EventStore.

This is useful for long-running operations to avoid load balancer timeouts.
Instead of holding a connection open for minutes, you can periodically close
and let the client reconnect.

#### `sample_step` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L830"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
sample_step(self, messages: str | Sequence[str | SamplingMessage]) -> SampleStep
```

Make a single LLM sampling call.

This is a stateless function that makes exactly one LLM call and optionally
executes any requested tools. Use this for fine-grained control over the
sampling loop.

**Args:**

* `messages`: The message(s) to send. Can be a string, list of strings,
  or list of SamplingMessage objects.
* `system_prompt`: Optional system prompt for the LLM.
* `temperature`: Optional sampling temperature.
* `max_tokens`: Maximum tokens to generate. Defaults to 512.
* `model_preferences`: Optional model preferences.
* `tools`: Optional list of tools the LLM can use.
* `tool_choice`: Tool choice mode ("auto", "required", or "none").
* `execute_tools`: If True (default), execute tool calls and append results
  to history. If False, return immediately with tool\_calls available
  in the step for manual execution.
* `mask_error_details`: If True, mask detailed error messages from tool
  execution. When None (default), uses the global settings value.
  Tools can raise ToolError to bypass masking.
* `tool_concurrency`: Controls parallel execution of tools:
* None (default): Sequential execution (one at a time)
* 0: Unlimited parallel execution
* N > 0: Execute at most N tools concurrently
  If any tool has sequential=True, all tools execute sequentially
  regardless of this setting.

**Returns:**

* SampleStep containing:
* * .response: The raw LLM response
* * .history: Messages including input, assistant response, and tool results
* * .is\_tool\_use: True if the LLM requested tool execution
* * .tool\_calls: List of tool calls (if any)
* * .text: The text content (if any)

#### `sample` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L909"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
sample(self, messages: str | Sequence[str | SamplingMessage]) -> SamplingResult[ResultT]
```

Overload: With result\_type, returns SamplingResult\[ResultT].

#### `sample` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L925"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
sample(self, messages: str | Sequence[str | SamplingMessage]) -> SamplingResult[str]
```

Overload: Without result\_type, returns SamplingResult\[str].

#### `sample` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L940"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
sample(self, messages: str | Sequence[str | SamplingMessage]) -> SamplingResult[ResultT] | SamplingResult[str]
```

Send a sampling request to the client and await the response.

This method runs to completion automatically. When tools are provided,
it executes a tool loop: if the LLM returns a tool use request, the tools
are executed and the results are sent back to the LLM. This continues
until the LLM provides a final text response.

When result\_type is specified, a synthetic `final_response` tool is
created. The LLM calls this tool to provide the structured response,
which is validated against the result\_type and returned as `.result`.

For fine-grained control over the sampling loop, use sample\_step() instead.

**Args:**

* `messages`: The message(s) to send. Can be a string, list of strings,
  or list of SamplingMessage objects.
* `system_prompt`: Optional system prompt for the LLM.
* `temperature`: Optional sampling temperature.
* `max_tokens`: Maximum tokens to generate. Defaults to 512.
* `model_preferences`: Optional model preferences.
* `tools`: Optional list of tools the LLM can use. Accepts plain
  functions or SamplingTools.
* `result_type`: Optional type for structured output. When specified,
  a synthetic `final_response` tool is created and the LLM's
  response is validated against this type.
* `mask_error_details`: If True, mask detailed error messages from tool
  execution. When None (default), uses the global settings value.
  Tools can raise ToolError to bypass masking.
* `tool_concurrency`: Controls parallel execution of tools:
* None (default): Sequential execution (one at a time)
* 0: Unlimited parallel execution
* N > 0: Execute at most N tools concurrently
  If any tool has sequential=True, all tools execute sequentially
  regardless of this setting.

**Returns:**

* SamplingResult\[T] containing:
* * .text: The text representation (raw text or JSON for structured)
* * .result: The typed result (str for text, parsed object for structured)
* * .history: All messages exchanged during sampling

#### `elicit` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1015"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
elicit(self, message: str, response_type: None) -> AcceptedElicitation[dict[str, Any]] | DeclinedElicitation | CancelledElicitation
```

#### `elicit` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1027"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
elicit(self, message: str, response_type: type[T]) -> AcceptedElicitation[T] | DeclinedElicitation | CancelledElicitation
```

#### `elicit` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1037"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
elicit(self, message: str, response_type: list[str]) -> AcceptedElicitation[str] | DeclinedElicitation | CancelledElicitation
```

#### `elicit` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1047"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
elicit(self, message: str, response_type: dict[str, dict[str, str]]) -> AcceptedElicitation[str] | DeclinedElicitation | CancelledElicitation
```

#### `elicit` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1057"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
elicit(self, message: str, response_type: list[list[str]]) -> AcceptedElicitation[list[str]] | DeclinedElicitation | CancelledElicitation
```

#### `elicit` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1069"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
elicit(self, message: str, response_type: list[dict[str, dict[str, str]]]) -> AcceptedElicitation[list[str]] | DeclinedElicitation | CancelledElicitation
```

#### `elicit` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1081"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
elicit(self, message: str, response_type: type[T] | list[str] | dict[str, dict[str, str]] | list[list[str]] | list[dict[str, dict[str, str]]] | None = None) -> AcceptedElicitation[T] | AcceptedElicitation[dict[str, Any]] | AcceptedElicitation[str] | AcceptedElicitation[list[str]] | DeclinedElicitation | CancelledElicitation
```

Send an elicitation request to the client and await the response.

Call this method at any time to request additional information from
the user through the client. The client must support elicitation,
or the request will error.

Note that the MCP protocol only supports simple object schemas with
primitive types. You can provide a dataclass, TypedDict, or BaseModel to
comply. If you provide a primitive type, an object schema with a single
"value" field will be generated for the MCP interaction and
automatically deconstructed into the primitive type upon response.

If the response\_type is None, the generated schema will be that of an
empty object in order to comply with the MCP protocol requirements.
Clients must send an empty object ("")in response.

**Args:**

* `message`: A human-readable message explaining what information is needed
* `response_type`: The type of the response, which should be a primitive
  type or dataclass or BaseModel. If it is a primitive type, an
  object schema with a single "value" field will be generated.

#### `set_state` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1195"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_state(self, key: str, value: Any) -> None
```

Set a value in the state store.

By default, values are stored in the session-scoped state store and
persist across requests within the same MCP session. Values must be
JSON-serializable (dicts, lists, strings, numbers, etc.).

For non-serializable values (e.g., HTTP clients, database connections),
pass `serializable=False`. These values are stored in a request-scoped
dict and only live for the current MCP request (tool call, resource
read, or prompt render). They will not be available in subsequent
requests.

The key is automatically prefixed with the session identifier.

#### `get_state` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1237"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_state(self, key: str) -> Any
```

Get a value from the state store.

Checks request-scoped state first (set with `serializable=False`),
then falls back to the session-scoped state store.

Returns None if the key is not found.

#### `delete_state` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1251"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
delete_state(self, key: str) -> None
```

Delete a value from the state store.

Removes from both request-scoped and session-scoped stores.

#### `enable_components` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1272"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
enable_components(self) -> None
```

Enable components matching criteria for this session only.

Session rules override global transforms. Rules accumulate - each call
adds a new rule to the session. Later marks override earlier ones
(Visibility transform semantics).

Sends notifications to this session only: ToolListChangedNotification,
ResourceListChangedNotification, and PromptListChangedNotification.

**Args:**

* `names`: Component names or URIs to match.
* `keys`: Component keys to match (e.g., ).
* `version`: Component version spec to match.
* `tags`: Tags to match (component must have at least one).
* `components`: Component types to match (e.g., ).
* `match_all`: If True, matches all components regardless of other criteria.

#### `disable_components` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1310"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
disable_components(self) -> None
```

Disable components matching criteria for this session only.

Session rules override global transforms. Rules accumulate - each call
adds a new rule to the session. Later marks override earlier ones
(Visibility transform semantics).

Sends notifications to this session only: ToolListChangedNotification,
ResourceListChangedNotification, and PromptListChangedNotification.

**Args:**

* `names`: Component names or URIs to match.
* `keys`: Component keys to match (e.g., ).
* `version`: Component version spec to match.
* `tags`: Tags to match (component must have at least one).
* `components`: Component types to match (e.g., ).
* `match_all`: If True, matches all components regardless of other criteria.

#### `reset_visibility` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/context.py#L1348"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
reset_visibility(self) -> None
```

Clear all session visibility rules.

Use this to reset session visibility back to global defaults.

Sends notifications to this session only: ToolListChangedNotification,
ResourceListChangedNotification, and PromptListChangedNotification.


# dependencies
Source: https://gofastmcp.com/python-sdk/fastmcp-server-dependencies



# `fastmcp.server.dependencies`

Dependency injection for FastMCP.

DI features (Depends, CurrentContext, CurrentFastMCP) work without pydocket
using the uncalled-for DI engine. Only task-related dependencies (CurrentDocket,
CurrentWorker) and background task execution require fastmcp\[tasks].

## Functions

### `get_task_context` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L97"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_task_context() -> TaskContextInfo | None
```

Get the current task context if running inside a background task worker.

This function extracts task information from the Docket execution context.
Returns None if not running in a task context (e.g., foreground execution).

**Returns:**

* TaskContextInfo with task\_id and session\_id, or None if not in a task.

### `register_task_session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L135"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_task_session(session_id: str, session: ServerSession) -> None
```

Register a session for Context access in background tasks.

Called automatically when a task is submitted to Docket. The session is
stored as a weakref so it doesn't prevent garbage collection when the
client disconnects.

**Args:**

* `session_id`: The session identifier
* `session`: The ServerSession instance

### `get_task_session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L149"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_task_session(session_id: str) -> ServerSession | None
```

Get a registered session by ID if still alive.

**Args:**

* `session_id`: The session identifier

**Returns:**

* The ServerSession if found and alive, None otherwise

### `is_docket_available` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L185"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_docket_available() -> bool
```

Check if pydocket is installed.

### `require_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L198"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
require_docket(feature: str) -> None
```

Raise ImportError with install instructions if docket not available.

**Args:**

* `feature`: Description of what requires docket (e.g., "`task=True`",
  "CurrentDocket()"). Will be included in the error message.

### `transform_context_annotations` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L223"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_context_annotations(fn: Callable[..., Any]) -> Callable[..., Any]
```

Transform ctx: Context into ctx: Context = CurrentContext().

Transforms ALL params typed as Context to use Docket's DI system,
unless they already have a Dependency-based default (like CurrentContext()).

This unifies the legacy type annotation DI with Docket's Depends() system,
allowing both patterns to work through a single resolution path.

Note: Only POSITIONAL\_OR\_KEYWORD parameters are reordered (params with defaults
after those without). KEYWORD\_ONLY parameters keep their position since Python
allows them to have defaults in any order.

**Args:**

* `fn`: Function to transform

**Returns:**

* Function with modified signature (same function object, updated **signature**)

### `get_context` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L364"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_context() -> Context
```

Get the current FastMCP Context instance directly.

### `get_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L374"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_server() -> FastMCP
```

Get the current FastMCP server instance directly.

**Returns:**

* The active FastMCP server

**Raises:**

* `RuntimeError`: If no server in context

### `get_http_request` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L392"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_http_request() -> Request
```

Get the current HTTP request.

Tries MCP SDK's request\_ctx first, then falls back to FastMCP's HTTP context.

### `get_http_headers` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L412"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_http_headers(include_all: bool = False, include: set[str] | None = None) -> dict[str, str]
```

Extract headers from the current HTTP request if available.

Never raises an exception, even if there is no active HTTP request (in which case
an empty dict is returned).

By default, strips problematic headers like `content-length` and `authorization`
that cause issues if forwarded to downstream services. If `include_all` is True,
all headers are returned.

The `include` parameter allows specific headers to be included even if they would
normally be excluded. This is useful for proxy transports that need to forward
authorization headers to upstream MCP servers.

### `get_access_token` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L469"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_access_token() -> AccessToken | None
```

Get the FastMCP access token from the current context.

This function first tries to get the token from the current HTTP request's scope,
which is more reliable for long-lived connections where the SDK's auth\_context\_var
may become stale after token refresh. Falls back to the SDK's context var if no
request is available. In background tasks (Docket workers), falls back to the
token snapshot stored in Redis at task submission time.

**Returns:**

* The access token if an authenticated user is available, None otherwise.

### `without_injected_parameters` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L541"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
without_injected_parameters(fn: Callable[..., Any]) -> Callable[..., Any]
```

Create a wrapper function without injected parameters.

Returns a wrapper that excludes Context and Docket dependency parameters,
making it safe to use with Pydantic TypeAdapter for schema generation and
validation. The wrapper internally handles all dependency resolution and
Context injection when called.

Handles:

* Legacy Context injection (always works)
* Depends() injection (always works - uses docket or vendored DI engine)

**Args:**

* `fn`: Original function with Context and/or dependencies

**Returns:**

* Async wrapper function without injected parameters

### `resolve_dependencies` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L690"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resolve_dependencies(fn: Callable[..., Any], arguments: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]
```

Resolve dependencies for a FastMCP function.

This function:

1. Filters out any dependency parameter names from user arguments (security)
2. Resolves Depends() parameters via the DI system

The filtering prevents external callers from overriding injected parameters by
providing values for dependency parameter names. This is a security feature.

Note: Context injection is handled via transform\_context\_annotations() which
converts `ctx: Context` to `ctx: Context = Depends(get_context)` at registration
time, so all injection goes through the unified DI system.

**Args:**

* `fn`: The function to resolve dependencies for
* `arguments`: User arguments (may contain keys that match dependency names,
  which will be filtered out)

### `CurrentContext` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L893"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
CurrentContext() -> Context
```

Get the current FastMCP Context instance.

This dependency provides access to the active FastMCP Context for the
current MCP operation (tool/resource/prompt call).

**Returns:**

* A dependency that resolves to the active Context instance

**Raises:**

* `RuntimeError`: If no active context found (during resolution)

### `OptionalCurrentContext` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L918"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
OptionalCurrentContext() -> Context | None
```

Get the current FastMCP Context, or None when no context is active.

### `CurrentDocket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L941"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
CurrentDocket() -> Docket
```

Get the current Docket instance managed by FastMCP.

This dependency provides access to the Docket instance that FastMCP
automatically creates for background task scheduling.

**Returns:**

* A dependency that resolves to the active Docket instance

**Raises:**

* `RuntimeError`: If not within a FastMCP server context
* `ImportError`: If fastmcp\[tasks] not installed

### `CurrentWorker` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L986"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
CurrentWorker() -> Worker
```

Get the current Docket Worker instance managed by FastMCP.

This dependency provides access to the Worker instance that FastMCP
automatically creates for background task processing.

**Returns:**

* A dependency that resolves to the active Worker instance

**Raises:**

* `RuntimeError`: If not within a FastMCP server context
* `ImportError`: If fastmcp\[tasks] not installed

### `CurrentFastMCP` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1028"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
CurrentFastMCP() -> FastMCP
```

Get the current FastMCP server instance.

This dependency provides access to the active FastMCP server.

**Returns:**

* A dependency that resolves to the active FastMCP server

**Raises:**

* `RuntimeError`: If no server in context (during resolution)

### `CurrentRequest` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1063"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
CurrentRequest() -> Request
```

Get the current HTTP request.

This dependency provides access to the Starlette Request object for the
current HTTP request. Only available when running over HTTP transports
(SSE or Streamable HTTP).

**Returns:**

* A dependency that resolves to the active Starlette Request

**Raises:**

* `RuntimeError`: If no HTTP request in context (e.g., STDIO transport)

### `CurrentHeaders` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1099"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
CurrentHeaders() -> dict[str, str]
```

Get the current HTTP request headers.

This dependency provides access to the HTTP headers for the current request,
including the authorization header. Returns an empty dictionary when no HTTP
request is available, making it safe to use in code that might run over any
transport.

**Returns:**

* A dependency that resolves to a dictionary of header name -> value

### `CurrentAccessToken` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1318"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
CurrentAccessToken() -> AccessToken
```

Get the current access token for the authenticated user.

This dependency provides access to the AccessToken for the current
authenticated request. Raises an error if no authentication is present.

**Returns:**

* A dependency that resolves to the active AccessToken

**Raises:**

* `RuntimeError`: If no authenticated user (use get\_access\_token() for optional)

### `TokenClaim` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1370"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
TokenClaim(name: str) -> str
```

Get a specific claim from the access token.

This dependency extracts a single claim value from the current access token.
It's useful for getting user identifiers, roles, or other token claims
without needing the full token object.

**Args:**

* `name`: The name of the claim to extract (e.g., "oid", "sub", "email")

**Returns:**

* A dependency that resolves to the claim value as a string

**Raises:**

* `RuntimeError`: If no access token is available or claim is missing

## Classes

### `TaskContextInfo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L83"><Icon icon="github" /></a></sup>

Information about the current background task context.

Returned by `get_task_context()` when running inside a Docket worker.
Contains identifiers needed to communicate with the MCP session.

### `ProgressLike` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1127"><Icon icon="github" /></a></sup>

Protocol for progress tracking interface.

Defines the common interface between InMemoryProgress (server context)
and Docket's Progress (worker context).

**Methods:**

#### `current` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1135"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
current(self) -> int | None
```

Current progress value.

#### `total` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1140"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
total(self) -> int
```

Total/target progress value.

#### `message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1145"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
message(self) -> str | None
```

Current progress message.

#### `set_total` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1149"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_total(self, total: int) -> None
```

Set the total/target value for progress tracking.

#### `increment` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1153"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
increment(self, amount: int = 1) -> None
```

Atomically increment the current progress value.

#### `set_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1157"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_message(self, message: str | None) -> None
```

Update the progress status message.

### `InMemoryProgress` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1162"><Icon icon="github" /></a></sup>

In-memory progress tracker for immediate tool execution.

Provides the same interface as Docket's Progress but stores state in memory
instead of Redis. Useful for testing and immediate execution where
progress doesn't need to be observable across processes.

**Methods:**

#### `current` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1182"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
current(self) -> int | None
```

#### `total` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1186"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
total(self) -> int
```

#### `message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1190"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
message(self) -> str | None
```

#### `set_total` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1193"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_total(self, total: int) -> None
```

Set the total/target value for progress tracking.

#### `increment` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1199"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
increment(self, amount: int = 1) -> None
```

Atomically increment the current progress value.

#### `set_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1208"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_message(self, message: str | None) -> None
```

Update the progress status message.

### `Progress` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1213"><Icon icon="github" /></a></sup>

FastMCP Progress dependency that works in both server and worker contexts.

Handles three execution modes:

* In Docket worker: Uses the execution's progress (observable via Redis)
* In FastMCP server with Docket: Falls back to in-memory progress
* In FastMCP server without Docket: Uses in-memory progress

This allows tools to use Progress() regardless of whether they're called
immediately or as background tasks, and regardless of whether pydocket
is installed.

**Methods:**

#### `current` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1250"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
current(self) -> int | None
```

Current progress value.

#### `total` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1256"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
total(self) -> int
```

Total/target progress value.

#### `message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1262"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
message(self) -> str | None
```

Current progress message.

#### `set_total` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1267"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_total(self, total: int) -> None
```

Set the total/target value for progress tracking.

#### `increment` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1272"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
increment(self, amount: int = 1) -> None
```

Atomically increment the current progress value.

#### `set_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/dependencies.py#L1277"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_message(self, message: str | None) -> None
```

Update the progress status message.


# elicitation
Source: https://gofastmcp.com/python-sdk/fastmcp-server-elicitation



# `fastmcp.server.elicitation`

## Functions

### `parse_elicit_response_type` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L132"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse_elicit_response_type(response_type: Any) -> ElicitConfig
```

Parse response\_type into schema and handling configuration.

Supports multiple syntaxes:

* None: Empty object schema, expect empty response
* dict: `{"low": {"title": "..."}}` -> single-select titled enum
* list patterns:
  * `[["a", "b"]]` -> multi-select untitled
  * `[{"low": {...}}]` -> multi-select titled
  * `["a", "b"]` -> single-select untitled
* `list[X]` type annotation: multi-select with type
* Scalar types (bool, int, float, str, Literal, Enum): single value
* Other types (dataclass, BaseModel): use directly

### `handle_elicit_accept` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L265"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
handle_elicit_accept(config: ElicitConfig, content: Any) -> AcceptedElicitation[Any]
```

Handle an accepted elicitation response.

**Args:**

* `config`: The elicitation configuration from parse\_elicit\_response\_type
* `content`: The response content from the client

**Returns:**

* AcceptedElicitation with the extracted/validated data

### `get_elicitation_schema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L324"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_elicitation_schema(response_type: type[T]) -> dict[str, Any]
```

Get the schema for an elicitation response.

**Args:**

* `response_type`: The type of the response

### `validate_elicitation_json_schema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L343"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_elicitation_json_schema(schema: dict[str, Any]) -> None
```

Validate that a JSON schema follows MCP elicitation requirements.

This ensures the schema is compatible with MCP elicitation requirements:

* Must be an object schema
* Must only contain primitive field types (string, number, integer, boolean)
* Must be flat (no nested objects or arrays of objects)
* Allows const fields (for Literal types) and enum fields (for Enum types)
* Only primitive types and their nullable variants are allowed

**Args:**

* `schema`: The JSON schema to validate

**Raises:**

* `TypeError`: If the schema doesn't meet MCP elicitation requirements

## Classes

### `ElicitationJsonSchema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L36"><Icon icon="github" /></a></sup>

Custom JSON schema generator for MCP elicitation that always inlines enums.

MCP elicitation requires inline enum schemas without $ref/$defs references.
This generator ensures enums are always generated inline for compatibility.
Optionally adds enumNames for better UI display when available.

**Methods:**

#### `generate_inner` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L44"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_inner(self, schema: core_schema.CoreSchema) -> JsonSchemaValue
```

Override to prevent ref generation for enums and handle list schemas.

#### `list_schema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L57"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_schema(self, schema: core_schema.ListSchema) -> JsonSchemaValue
```

Generate schema for list types, detecting enum items for multi-select.

#### `enum_schema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L94"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
enum_schema(self, schema: core_schema.EnumSchema) -> JsonSchemaValue
```

Generate inline enum schema.

Always generates enum pattern: `{"enum": [value, ...]}`
Titled enums are handled separately via dict-based syntax in ctx.elicit().

### `AcceptedElicitation` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L105"><Icon icon="github" /></a></sup>

Result when user accepts the elicitation.

### `ScalarElicitationType` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L113"><Icon icon="github" /></a></sup>

### `ElicitConfig` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/elicitation.py#L118"><Icon icon="github" /></a></sup>

Configuration for an elicitation request.

**Attributes:**

* `schema`: The JSON schema to send to the client
* `response_type`: The type to validate responses with (None for raw schemas)
* `is_raw`: True if schema was built directly (extract "value" from response)


# event_store
Source: https://gofastmcp.com/python-sdk/fastmcp-server-event_store



# `fastmcp.server.event_store`

EventStore implementation backed by AsyncKeyValue.

This module provides an EventStore implementation that enables SSE polling/resumability
for Streamable HTTP transports. Events are stored using the key\_value package's
AsyncKeyValue protocol, allowing users to configure any compatible backend
(in-memory, Redis, etc.) following the same pattern as ResponseCachingMiddleware.

## Classes

### `EventEntry` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/event_store.py#L26"><Icon icon="github" /></a></sup>

Stored event entry.

### `StreamEventList` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/event_store.py#L34"><Icon icon="github" /></a></sup>

List of event IDs for a stream.

### `EventStore` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/event_store.py#L40"><Icon icon="github" /></a></sup>

EventStore implementation backed by AsyncKeyValue.

Enables SSE polling/resumability by storing events that can be replayed
when clients reconnect. Works with any AsyncKeyValue backend (memory, Redis, etc.)
following the same pattern as ResponseCachingMiddleware and OAuthProxy.

**Args:**

* `storage`: AsyncKeyValue backend. Defaults to MemoryStore.
* `max_events_per_stream`: Maximum events to retain per stream. Default 100.
* `ttl`: Event TTL in seconds. Default 3600 (1 hour). Set to None for no expiration.

**Methods:**

#### `store_event` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/event_store.py#L94"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
store_event(self, stream_id: StreamId, message: JSONRPCMessage | None) -> EventId
```

Store an event and return its ID.

**Args:**

* `stream_id`: ID of the stream the event belongs to
* `message`: The JSON-RPC message to store, or None for priming events

**Returns:**

* The generated event ID for the stored event

#### `replay_events_after` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/event_store.py#L135"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
replay_events_after(self, last_event_id: EventId, send_callback: EventCallback) -> StreamId | None
```

Replay events that occurred after the specified event ID.

**Args:**

* `last_event_id`: The ID of the last event the client received
* `send_callback`: A callback function to send events to the client

**Returns:**

* The stream ID of the replayed events, or None if the event ID was not found


# http
Source: https://gofastmcp.com/python-sdk/fastmcp-server-http



# `fastmcp.server.http`

## Functions

### `set_http_request` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/http.py#L77"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_http_request(request: Request) -> Generator[Request, None, None]
```

### `create_base_app` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/http.py#L110"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_base_app(routes: list[BaseRoute], middleware: list[Middleware], debug: bool = False, lifespan: Callable | None = None) -> StarletteWithLifespan
```

Create a base Starlette app with common middleware and routes.

**Args:**

* `routes`: List of routes to include in the app
* `middleware`: List of middleware to include in the app
* `debug`: Whether to enable debug mode
* `lifespan`: Optional lifespan manager for the app

**Returns:**

* A Starlette application

### `create_sse_app` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/http.py#L139"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_sse_app(server: FastMCP[LifespanResultT], message_path: str, sse_path: str, auth: AuthProvider | None = None, debug: bool = False, routes: list[BaseRoute] | None = None, middleware: list[Middleware] | None = None) -> StarletteWithLifespan
```

Return an instance of the SSE server app.

**Args:**

* `server`: The FastMCP server instance
* `message_path`: Path for SSE messages
* `sse_path`: Path for SSE connections
* `auth`: Optional authentication provider (AuthProvider)
* `debug`: Whether to enable debug mode
* `routes`: Optional list of custom routes
* `middleware`: Optional list of middleware

Returns:
A Starlette application with RequestContextMiddleware

### `create_streamable_http_app` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/http.py#L266"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_streamable_http_app(server: FastMCP[LifespanResultT], streamable_http_path: str, event_store: EventStore | None = None, retry_interval: int | None = None, auth: AuthProvider | None = None, json_response: bool = False, stateless_http: bool = False, debug: bool = False, routes: list[BaseRoute] | None = None, middleware: list[Middleware] | None = None) -> StarletteWithLifespan
```

Return an instance of the StreamableHTTP server app.

**Args:**

* `server`: The FastMCP server instance
* `streamable_http_path`: Path for StreamableHTTP connections
* `event_store`: Optional event store for SSE polling/resumability
* `retry_interval`: Optional retry interval in milliseconds for SSE polling.
  Controls how quickly clients should reconnect after server-initiated
  disconnections. Requires event\_store to be set. Defaults to SDK default.
* `auth`: Optional authentication provider (AuthProvider)
* `json_response`: Whether to use JSON response format
* `stateless_http`: Whether to use stateless mode (new transport per request)
* `debug`: Whether to enable debug mode
* `routes`: Optional list of custom routes
* `middleware`: Optional list of middleware

**Returns:**

* A Starlette application with StreamableHTTP support

## Classes

### `StreamableHTTPASGIApp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/http.py#L32"><Icon icon="github" /></a></sup>

ASGI application wrapper for Streamable HTTP server transport.

### `StarletteWithLifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/http.py#L70"><Icon icon="github" /></a></sup>

**Methods:**

#### `lifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/http.py#L72"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
lifespan(self) -> Lifespan[Starlette]
```

### `RequestContextMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/http.py#L85"><Icon icon="github" /></a></sup>

Middleware that stores each request in a ContextVar and sets transport type.


# lifespan
Source: https://gofastmcp.com/python-sdk/fastmcp-server-lifespan



# `fastmcp.server.lifespan`

Composable lifespans for FastMCP servers.

This module provides a `@lifespan` decorator for creating composable server lifespans
that can be combined using the `|` operator.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

@lifespan
async def db_lifespan(server):
    conn = await connect_db()
    yield {"db": conn}
    await conn.close()

@lifespan
async def cache_lifespan(server):
    cache = await connect_cache()
    yield {"cache": cache}
    await cache.close()

mcp = FastMCP("server", lifespan=db_lifespan | cache_lifespan)
```

To compose with existing `@asynccontextmanager` lifespans, wrap them explicitly:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from contextlib import asynccontextmanager
from fastmcp.server.lifespan import lifespan, ContextManagerLifespan

@asynccontextmanager
async def legacy_lifespan(server):
    yield {"legacy": True}

@lifespan
async def new_lifespan(server):
    yield {"new": True}

# Wrap the legacy lifespan explicitly
combined = ContextManagerLifespan(legacy_lifespan) | new_lifespan
```

## Functions

### `lifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/lifespan.py#L172"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
lifespan(fn: LifespanFn) -> Lifespan
```

Decorator to create a composable lifespan.

Use this decorator on an async generator function to make it composable
with other lifespans using the `|` operator.

**Args:**

* `fn`: An async generator function that takes a FastMCP server and yields
  a dict for the lifespan context.

**Returns:**

* A composable Lifespan wrapper.

## Classes

### `Lifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/lifespan.py#L61"><Icon icon="github" /></a></sup>

Composable lifespan wrapper.

Wraps an async generator function and enables composition via the `|` operator.
The wrapped function should yield a dict that becomes part of the lifespan context.

### `ContextManagerLifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/lifespan.py#L110"><Icon icon="github" /></a></sup>

Lifespan wrapper for already-wrapped context manager functions.

Use this for functions already decorated with @asynccontextmanager.

### `ComposedLifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/lifespan.py#L137"><Icon icon="github" /></a></sup>

Two lifespans composed together.

Enters the left lifespan first, then the right. Exits in reverse order.
Results are shallow-merged into a single dict.


# low_level
Source: https://gofastmcp.com/python-sdk/fastmcp-server-low_level



# `fastmcp.server.low_level`

## Classes

### `MiddlewareServerSession` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L36"><Icon icon="github" /></a></sup>

ServerSession that routes initialization requests through FastMCP middleware.

**Methods:**

#### `fastmcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L46"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp(self) -> FastMCP
```

Get the FastMCP instance.

#### `client_supports_extension` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L53"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
client_supports_extension(self, extension_id: str) -> bool
```

Check if the connected client supports a given MCP extension.

Inspects the `extensions` extra field on `ClientCapabilities`
sent by the client during initialization.

### `LowLevelServer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L152"><Icon icon="github" /></a></sup>

**Methods:**

#### `fastmcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L166"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp(self) -> FastMCP
```

Get the FastMCP instance.

#### `create_initialization_options` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L173"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_initialization_options(self, notification_options: NotificationOptions | None = None, experimental_capabilities: dict[str, dict[str, Any]] | None = None, **kwargs: Any) -> InitializationOptions
```

#### `get_capabilities` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L188"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_capabilities(self, notification_options: NotificationOptions, experimental_capabilities: dict[str, dict[str, Any]]) -> mcp.types.ServerCapabilities
```

Override to set capabilities.tasks as a first-class field per SEP-1686.

This ensures task capabilities appear in capabilities.tasks instead of
capabilities.experimental.tasks, which is required by the MCP spec and
enables proper task detection by clients like VS Code Copilot 1.107+.

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L222"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, read_stream: MemoryObjectReceiveStream[SessionMessage | Exception], write_stream: MemoryObjectSendStream[SessionMessage], initialization_options: InitializationOptions, raise_exceptions: bool = False, stateless: bool = False)
```

Overrides the run method to use the MiddlewareServerSession.

#### `read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L258"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource(self) -> Callable[[Callable[[AnyUrl], Awaitable[mcp.types.ReadResourceResult | mcp.types.CreateTaskResult]]], Callable[[AnyUrl], Awaitable[mcp.types.ReadResourceResult | mcp.types.CreateTaskResult]]]
```

Decorator for registering a read\_resource handler with CreateTaskResult support.

The MCP SDK's read\_resource decorator does not support returning CreateTaskResult
for background task execution. This decorator wraps the result in ServerResult.

This decorator can be removed once the MCP SDK adds native CreateTaskResult support
for resources.

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/low_level.py#L302"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self) -> Callable[[Callable[[str, dict[str, Any] | None], Awaitable[mcp.types.GetPromptResult | mcp.types.CreateTaskResult]]], Callable[[str, dict[str, Any] | None], Awaitable[mcp.types.GetPromptResult | mcp.types.CreateTaskResult]]]
```

Decorator for registering a get\_prompt handler with CreateTaskResult support.

The MCP SDK's get\_prompt decorator does not support returning CreateTaskResult
for background task execution. This decorator wraps the result in ServerResult.

This decorator can be removed once the MCP SDK adds native CreateTaskResult support
for prompts.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-__init__



# `fastmcp.server.middleware`

*This module is empty or contains only private/internal implementations.*


# authorization
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-authorization



# `fastmcp.server.middleware.authorization`

Authorization middleware for FastMCP.

This module provides middleware-based authorization using callable auth checks.
AuthMiddleware applies auth checks globally to all components on the server.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth import require_scopes, restrict_tag
from fastmcp.server.middleware import AuthMiddleware

# Require specific scope for all components
mcp = FastMCP(middleware=[
    AuthMiddleware(auth=require_scopes("api"))
])

# Tag-based: components tagged "admin" require "admin" scope
mcp = FastMCP(middleware=[
    AuthMiddleware(auth=restrict_tag("admin", scopes=["admin"]))
])
```

## Classes

### `AuthMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/authorization.py#L51"><Icon icon="github" /></a></sup>

Global authorization middleware using callable checks.

This middleware applies auth checks to all components (tools, resources,
prompts) on the server. It uses the same callable API as component-level
auth checks.

The middleware:

* Filters tools/resources/prompts from list responses based on auth checks
* Checks auth before tool execution, resource read, and prompt render
* Skips all auth checks for STDIO transport (no OAuth concept)

**Args:**

* `auth`: A single auth check function or list of check functions.
  All checks must pass for authorization to succeed (AND logic).

**Methods:**

#### `on_list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/authorization.py#L85"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_tools(self, context: MiddlewareContext[mt.ListToolsRequest], call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]]) -> Sequence[Tool]
```

Filter tools/list response based on auth checks.

#### `on_call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/authorization.py#L113"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_call_tool(self, context: MiddlewareContext[mt.CallToolRequestParams], call_next: CallNext[mt.CallToolRequestParams, ToolResult]) -> ToolResult
```

Check auth before tool execution.

#### `on_list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/authorization.py#L156"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_resources(self, context: MiddlewareContext[mt.ListResourcesRequest], call_next: CallNext[mt.ListResourcesRequest, Sequence[Resource]]) -> Sequence[Resource]
```

Filter resources/list response based on auth checks.

#### `on_read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/authorization.py#L183"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_read_resource(self, context: MiddlewareContext[mt.ReadResourceRequestParams], call_next: CallNext[mt.ReadResourceRequestParams, ResourceResult]) -> ResourceResult
```

Check auth before resource read.

#### `on_list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/authorization.py#L226"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_resource_templates(self, context: MiddlewareContext[mt.ListResourceTemplatesRequest], call_next: CallNext[mt.ListResourceTemplatesRequest, Sequence[ResourceTemplate]]) -> Sequence[ResourceTemplate]
```

Filter resource templates/list response based on auth checks.

#### `on_list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/authorization.py#L255"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_prompts(self, context: MiddlewareContext[mt.ListPromptsRequest], call_next: CallNext[mt.ListPromptsRequest, Sequence[Prompt]]) -> Sequence[Prompt]
```

Filter prompts/list response based on auth checks.

#### `on_get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/authorization.py#L282"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_get_prompt(self, context: MiddlewareContext[mt.GetPromptRequestParams], call_next: CallNext[mt.GetPromptRequestParams, PromptResult]) -> PromptResult
```

Check auth before prompt render.


# caching
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-caching



# `fastmcp.server.middleware.caching`

A middleware for response caching.

## Classes

### `CachableResourceContent` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L38"><Icon icon="github" /></a></sup>

A wrapper for ResourceContent that can be cached.

### `CachableResourceResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L46"><Icon icon="github" /></a></sup>

A wrapper for ResourceResult that can be cached.

**Methods:**

#### `get_size` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L52"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_size(self) -> int
```

#### `wrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L56"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wrap(cls, value: ResourceResult) -> Self
```

#### `unwrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L67"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
unwrap(self) -> ResourceResult
```

### `CachableToolResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L79"><Icon icon="github" /></a></sup>

**Methods:**

#### `wrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L85"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wrap(cls, value: ToolResult) -> Self
```

#### `unwrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L92"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
unwrap(self) -> ToolResult
```

### `CachableMessage` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L100"><Icon icon="github" /></a></sup>

A wrapper for Message that can be cached.

### `CachablePromptResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L107"><Icon icon="github" /></a></sup>

A wrapper for PromptResult that can be cached.

**Methods:**

#### `get_size` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L114"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_size(self) -> int
```

#### `wrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L118"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wrap(cls, value: PromptResult) -> Self
```

#### `unwrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L127"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
unwrap(self) -> PromptResult
```

### `SharedMethodSettings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L138"><Icon icon="github" /></a></sup>

Shared config for a cache method.

### `ListToolsSettings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L145"><Icon icon="github" /></a></sup>

Configuration options for Tool-related caching.

### `ListResourcesSettings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L149"><Icon icon="github" /></a></sup>

Configuration options for Resource-related caching.

### `ListPromptsSettings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L153"><Icon icon="github" /></a></sup>

Configuration options for Prompt-related caching.

### `CallToolSettings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L157"><Icon icon="github" /></a></sup>

Configuration options for Tool-related caching.

### `ReadResourceSettings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L164"><Icon icon="github" /></a></sup>

Configuration options for Resource-related caching.

### `GetPromptSettings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L168"><Icon icon="github" /></a></sup>

Configuration options for Prompt-related caching.

### `ResponseCachingStatistics` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L172"><Icon icon="github" /></a></sup>

### `ResponseCachingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L181"><Icon icon="github" /></a></sup>

The response caching middleware offers a simple way to cache responses to mcp methods. The Middleware
supports cache invalidation via notifications from the server. The Middleware implements TTL-based caching
but cache implementations may offer additional features like LRU eviction, size limits, and more.

When items are retrieved from the cache they will no longer be the original objects, but rather no-op objects
this means that response caching may not be compatible with other middleware that expects original subclasses.

Notes:

* Caches `tools/call`, `resources/read`, `prompts/get`, `tools/list`, `resources/list`, and `prompts/list` requests.
* Cache keys are derived from method name and arguments.

**Methods:**

#### `on_list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L285"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_tools(self, context: MiddlewareContext[mcp.types.ListToolsRequest], call_next: CallNext[mcp.types.ListToolsRequest, Sequence[Tool]]) -> Sequence[Tool]
```

List tools from the cache, if caching is enabled, and the result is in the cache. Otherwise,
otherwise call the next middleware and store the result in the cache if caching is enabled.

#### `on_list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L324"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_resources(self, context: MiddlewareContext[mcp.types.ListResourcesRequest], call_next: CallNext[mcp.types.ListResourcesRequest, Sequence[Resource]]) -> Sequence[Resource]
```

List resources from the cache, if caching is enabled, and the result is in the cache. Otherwise,
otherwise call the next middleware and store the result in the cache if caching is enabled.

#### `on_list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L363"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_prompts(self, context: MiddlewareContext[mcp.types.ListPromptsRequest], call_next: CallNext[mcp.types.ListPromptsRequest, Sequence[Prompt]]) -> Sequence[Prompt]
```

List prompts from the cache, if caching is enabled, and the result is in the cache. Otherwise,
otherwise call the next middleware and store the result in the cache if caching is enabled.

#### `on_call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L400"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_call_tool(self, context: MiddlewareContext[mcp.types.CallToolRequestParams], call_next: CallNext[mcp.types.CallToolRequestParams, ToolResult]) -> ToolResult
```

Call a tool from the cache, if caching is enabled, and the result is in the cache. Otherwise,
otherwise call the next middleware and store the result in the cache if caching is enabled.

#### `on_read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L433"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_read_resource(self, context: MiddlewareContext[mcp.types.ReadResourceRequestParams], call_next: CallNext[mcp.types.ReadResourceRequestParams, ResourceResult]) -> ResourceResult
```

Read a resource from the cache, if caching is enabled, and the result is in the cache. Otherwise,
otherwise call the next middleware and store the result in the cache if caching is enabled.

#### `on_get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L461"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_get_prompt(self, context: MiddlewareContext[mcp.types.GetPromptRequestParams], call_next: CallNext[mcp.types.GetPromptRequestParams, PromptResult]) -> PromptResult
```

Get a prompt from the cache, if caching is enabled, and the result is in the cache. Otherwise,
otherwise call the next middleware and store the result in the cache if caching is enabled.

#### `statistics` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py#L499"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
statistics(self) -> ResponseCachingStatistics
```

Get the statistics for the cache.


# dereference
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-dereference



# `fastmcp.server.middleware.dereference`

Middleware that dereferences \$ref in JSON schemas before sending to clients.

## Classes

### `DereferenceRefsMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/dereference.py#L15"><Icon icon="github" /></a></sup>

Dereferences \$ref in component schemas before sending to clients.

Some MCP clients (e.g., VS Code Copilot) don't handle JSON Schema $ref
properly. This middleware inlines all $ref definitions so schemas are
self-contained. Enabled by default via `FastMCP(dereference_schemas=True)`.

**Methods:**

#### `on_list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/dereference.py#L24"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_tools(self, context: MiddlewareContext[mt.ListToolsRequest], call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]]) -> Sequence[Tool]
```

#### `on_list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/dereference.py#L33"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_resource_templates(self, context: MiddlewareContext[mt.ListResourceTemplatesRequest], call_next: CallNext[mt.ListResourceTemplatesRequest, Sequence[ResourceTemplate]]) -> Sequence[ResourceTemplate]
```


# error_handling
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-error_handling



# `fastmcp.server.middleware.error_handling`

Error handling middleware for consistent error responses and tracking.

## Classes

### `ErrorHandlingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/error_handling.py#L18"><Icon icon="github" /></a></sup>

Middleware that provides consistent error handling and logging.

Catches exceptions, logs them appropriately, and converts them to
proper MCP error responses. Also tracks error patterns for monitoring.

**Methods:**

#### `on_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/error_handling.py#L120"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_message(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Handle errors for all messages.

#### `get_error_stats` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/error_handling.py#L131"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_error_stats(self) -> dict[str, int]
```

Get error statistics for monitoring.

### `RetryMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/error_handling.py#L136"><Icon icon="github" /></a></sup>

Middleware that implements automatic retry logic for failed requests.

Retries requests that fail with transient errors, using exponential
backoff to avoid overwhelming the server or external dependencies.

**Methods:**

#### `on_request` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/error_handling.py#L192"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_request(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Implement retry logic for requests.


# logging
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-logging



# `fastmcp.server.middleware.logging`

Comprehensive logging middleware for FastMCP servers.

## Functions

### `default_serializer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/logging.py#L15"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_serializer(data: Any) -> str
```

The default serializer for Payloads in the logging middleware.

## Classes

### `BaseLoggingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/logging.py#L20"><Icon icon="github" /></a></sup>

Base class for logging middleware.

**Methods:**

#### `on_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/logging.py#L124"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_message(self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any]) -> Any
```

Log messages for configured methods.

### `LoggingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/logging.py#L148"><Icon icon="github" /></a></sup>

Middleware that provides comprehensive request and response logging.

Logs all MCP messages with configurable detail levels. Useful for debugging,
monitoring, and understanding server usage patterns.

### `StructuredLoggingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/logging.py#L203"><Icon icon="github" /></a></sup>

Middleware that provides structured JSON logging for better log analysis.

Outputs structured logs that are easier to parse and analyze with log
aggregation tools like ELK stack, Splunk, or cloud logging services.


# middleware
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-middleware



# `fastmcp.server.middleware.middleware`

## Functions

### `make_middleware_wrapper` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L66"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
make_middleware_wrapper(middleware: Middleware, call_next: CallNext[T, R]) -> CallNext[T, R]
```

Create a wrapper that applies a single middleware to a context. The
closure bakes in the middleware and call\_next function, so it can be
passed to other functions that expect a call\_next function.

## Classes

### `CallNext` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L42"><Icon icon="github" /></a></sup>

### `MiddlewareContext` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L47"><Icon icon="github" /></a></sup>

Unified context for all middleware operations.

**Methods:**

#### `copy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L62"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
copy(self, **kwargs: Any) -> MiddlewareContext[T]
```

### `Middleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L79"><Icon icon="github" /></a></sup>

Base class for FastMCP middleware with dispatching hooks.

**Methods:**

#### `on_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L128"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_message(self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any]) -> Any
```

#### `on_request` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L135"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_request(self, context: MiddlewareContext[mt.Request[Any, Any]], call_next: CallNext[mt.Request[Any, Any], Any]) -> Any
```

#### `on_notification` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L142"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_notification(self, context: MiddlewareContext[mt.Notification[Any, Any]], call_next: CallNext[mt.Notification[Any, Any], Any]) -> Any
```

#### `on_initialize` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L149"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_initialize(self, context: MiddlewareContext[mt.InitializeRequest], call_next: CallNext[mt.InitializeRequest, mt.InitializeResult | None]) -> mt.InitializeResult | None
```

#### `on_call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L156"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_call_tool(self, context: MiddlewareContext[mt.CallToolRequestParams], call_next: CallNext[mt.CallToolRequestParams, ToolResult]) -> ToolResult
```

#### `on_read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L163"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_read_resource(self, context: MiddlewareContext[mt.ReadResourceRequestParams], call_next: CallNext[mt.ReadResourceRequestParams, ResourceResult]) -> ResourceResult
```

#### `on_get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L170"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_get_prompt(self, context: MiddlewareContext[mt.GetPromptRequestParams], call_next: CallNext[mt.GetPromptRequestParams, PromptResult]) -> PromptResult
```

#### `on_list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L177"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_tools(self, context: MiddlewareContext[mt.ListToolsRequest], call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]]) -> Sequence[Tool]
```

#### `on_list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L184"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_resources(self, context: MiddlewareContext[mt.ListResourcesRequest], call_next: CallNext[mt.ListResourcesRequest, Sequence[Resource]]) -> Sequence[Resource]
```

#### `on_list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L191"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_resource_templates(self, context: MiddlewareContext[mt.ListResourceTemplatesRequest], call_next: CallNext[mt.ListResourceTemplatesRequest, Sequence[ResourceTemplate]]) -> Sequence[ResourceTemplate]
```

#### `on_list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/middleware.py#L200"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_prompts(self, context: MiddlewareContext[mt.ListPromptsRequest], call_next: CallNext[mt.ListPromptsRequest, Sequence[Prompt]]) -> Sequence[Prompt]
```


# ping
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-ping



# `fastmcp.server.middleware.ping`

Ping middleware for keeping client connections alive.

## Classes

### `PingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/ping.py#L10"><Icon icon="github" /></a></sup>

Middleware that sends periodic pings to keep client connections alive.

Starts a background ping task on first message from each session. The task
sends server-to-client pings at the configured interval until the session
ends.

**Methods:**

#### `on_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/ping.py#L42"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_message(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Start ping task on first message from a session.


# rate_limiting
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-rate_limiting



# `fastmcp.server.middleware.rate_limiting`

Rate limiting middleware for protecting FastMCP servers from abuse.

## Classes

### `RateLimitError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/rate_limiting.py#L15"><Icon icon="github" /></a></sup>

Error raised when rate limit is exceeded.

### `TokenBucketRateLimiter` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/rate_limiting.py#L22"><Icon icon="github" /></a></sup>

Token bucket implementation for rate limiting.

**Methods:**

#### `consume` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/rate_limiting.py#L38"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
consume(self, tokens: int = 1) -> bool
```

Try to consume tokens from the bucket.

**Args:**

* `tokens`: Number of tokens to consume

**Returns:**

* True if tokens were available and consumed, False otherwise

### `SlidingWindowRateLimiter` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/rate_limiting.py#L61"><Icon icon="github" /></a></sup>

Sliding window rate limiter implementation.

**Methods:**

#### `is_allowed` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/rate_limiting.py#L76"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_allowed(self) -> bool
```

Check if a request is allowed.

### `RateLimitingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/rate_limiting.py#L92"><Icon icon="github" /></a></sup>

Middleware that implements rate limiting to prevent server abuse.

Uses a token bucket algorithm by default, allowing for burst traffic
while maintaining a sustainable long-term rate.

**Methods:**

#### `on_request` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/rate_limiting.py#L152"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_request(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Apply rate limiting to requests.

### `SlidingWindowRateLimitingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/rate_limiting.py#L170"><Icon icon="github" /></a></sup>

Middleware that implements sliding window rate limiting.

Uses a sliding window approach which provides more precise rate limiting
but uses more memory to track individual request timestamps.

**Methods:**

#### `on_request` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/rate_limiting.py#L219"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_request(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Apply sliding window rate limiting to requests.


# response_limiting
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-response_limiting



# `fastmcp.server.middleware.response_limiting`

Response limiting middleware for controlling tool response sizes.

## Classes

### `ResponseLimitingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/response_limiting.py#L20"><Icon icon="github" /></a></sup>

Middleware that limits the response size of tool calls.

Intercepts tool call responses and enforces size limits. If a response
exceeds the limit, it extracts text content, truncates it, and returns
a single TextContent block.

**Methods:**

#### `on_call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/response_limiting.py#L93"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_call_tool(self, context: MiddlewareContext[mt.CallToolRequestParams], call_next: CallNext[mt.CallToolRequestParams, ToolResult]) -> ToolResult
```

Intercept tool calls and limit response size.


# timing
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-timing



# `fastmcp.server.middleware.timing`

Timing middleware for measuring and logging request performance.

## Classes

### `TimingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L10"><Icon icon="github" /></a></sup>

Middleware that logs the execution time of requests.

Only measures and logs timing for request messages (not notifications).
Provides insights into performance characteristics of your MCP server.

**Methods:**

#### `on_request` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L39"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_request(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Time request execution and log the results.

### `DetailedTimingMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L60"><Icon icon="github" /></a></sup>

Enhanced timing middleware with per-operation breakdowns.

Provides detailed timing information for different types of MCP operations,
allowing you to identify performance bottlenecks in specific operations.

**Methods:**

#### `on_call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L111"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Time tool execution.

#### `on_read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L118"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_read_resource(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Time resource reading.

#### `on_get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L127"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_get_prompt(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Time prompt retrieval.

#### `on_list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L134"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_tools(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Time tool listing.

#### `on_list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L140"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_resources(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Time resource listing.

#### `on_list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L146"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_resource_templates(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Time resource template listing.

#### `on_list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/timing.py#L152"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_prompts(self, context: MiddlewareContext, call_next: CallNext) -> Any
```

Time prompt listing.


# tool_injection
Source: https://gofastmcp.com/python-sdk/fastmcp-server-middleware-tool_injection



# `fastmcp.server.middleware.tool_injection`

A middleware for injecting tools into the MCP server context.

## Functions

### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/tool_injection.py#L54"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(context: Context) -> list[Prompt]
```

List prompts available on the server.

### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/tool_injection.py#L64"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(context: Context, name: Annotated[str, 'The name of the prompt to render.'], arguments: Annotated[dict[str, Any] | None, 'The arguments to pass to the prompt.'] = None) -> mcp.types.GetPromptResult
```

Render a prompt available on the server.

### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/tool_injection.py#L88"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(context: Context) -> list[mcp.types.Resource]
```

List resources available on the server.

### `read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/tool_injection.py#L98"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource(context: Context, uri: Annotated[AnyUrl | str, 'The URI of the resource to read.']) -> ResourceResult
```

Read a resource available on the server.

## Classes

### `ToolInjectionMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/tool_injection.py#L21"><Icon icon="github" /></a></sup>

A middleware for injecting tools into the context.

**Methods:**

#### `on_list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/tool_injection.py#L32"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_tools(self, context: MiddlewareContext[mcp.types.ListToolsRequest], call_next: CallNext[mcp.types.ListToolsRequest, Sequence[Tool]]) -> Sequence[Tool]
```

Inject tools into the response.

#### `on_call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/tool_injection.py#L41"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_call_tool(self, context: MiddlewareContext[mcp.types.CallToolRequestParams], call_next: CallNext[mcp.types.CallToolRequestParams, ToolResult]) -> ToolResult
```

Intercept tool calls to injected tools.

### `PromptToolMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/tool_injection.py#L80"><Icon icon="github" /></a></sup>

A middleware for injecting prompts as tools into the context.

### `ResourceToolMiddleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/middleware/tool_injection.py#L111"><Icon icon="github" /></a></sup>

A middleware for injecting resources as tools into the context.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-mixins-__init__



# `fastmcp.server.mixins`

Server mixins for FastMCP.


# lifespan
Source: https://gofastmcp.com/python-sdk/fastmcp-server-mixins-lifespan



# `fastmcp.server.mixins.lifespan`

Lifespan and Docket task infrastructure for FastMCP Server.

## Classes

### `LifespanMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/lifespan.py#L24"><Icon icon="github" /></a></sup>

Mixin providing lifespan and Docket task infrastructure for FastMCP.

**Methods:**

#### `docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/lifespan.py#L28"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
docket(self: FastMCP) -> Docket | None
```

Get the Docket instance if Docket support is enabled.

Returns None if Docket is not enabled or server hasn't been started yet.


# mcp_operations
Source: https://gofastmcp.com/python-sdk/fastmcp-server-mixins-mcp_operations



# `fastmcp.server.mixins.mcp_operations`

MCP protocol handler setup and wire-format handlers for FastMCP Server.

## Classes

### `MCPOperationsMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/mcp_operations.py#L90"><Icon icon="github" /></a></sup>

Mixin providing MCP protocol handler setup and wire-format handlers.

Note: Methods registered with SDK decorators (e.g., \_list\_tools\_mcp, \_call\_tool\_mcp)
cannot use `self: FastMCP` type hints because the SDK's `get_type_hints()` fails
to resolve FastMCP at runtime (it's only available under TYPE\_CHECKING). When
type hints fail to resolve, the SDK falls back to calling handlers with no arguments.
These methods use untyped `self` to avoid this issue.


# transport
Source: https://gofastmcp.com/python-sdk/fastmcp-server-mixins-transport



# `fastmcp.server.mixins.transport`

Transport-related methods for FastMCP Server.

## Classes

### `TransportMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/transport.py#L34"><Icon icon="github" /></a></sup>

Mixin providing transport-related methods for FastMCP.

Includes HTTP/stdio/SSE transport handling and custom HTTP routes.

**Methods:**

#### `run_async` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/transport.py#L40"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_async(self: FastMCP, transport: Transport | None = None, show_banner: bool | None = None, **transport_kwargs: Any) -> None
```

Run the FastMCP server asynchronously.

**Args:**

* `transport`: Transport protocol to use ("stdio", "http", "sse", or "streamable-http")
* `show_banner`: Whether to display the server banner. If None, uses the
  FASTMCP\_SHOW\_SERVER\_BANNER setting (default: True).

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/transport.py#L74"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self: FastMCP, transport: Transport | None = None, show_banner: bool | None = None, **transport_kwargs: Any) -> None
```

Run the FastMCP server. Note this is a synchronous function.

**Args:**

* `transport`: Transport protocol to use ("http", "stdio", "sse", or "streamable-http")
* `show_banner`: Whether to display the server banner. If None, uses the
  FASTMCP\_SHOW\_SERVER\_BANNER setting (default: True).

#### `custom_route` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/transport.py#L97"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
custom_route(self: FastMCP, path: str, methods: list[str], name: str | None = None, include_in_schema: bool = True) -> Callable[[Callable[[Request], Awaitable[Response]]], Callable[[Request], Awaitable[Response]]]
```

Decorator to register a custom HTTP route on the FastMCP server.

Allows adding arbitrary HTTP endpoints outside the standard MCP protocol,
which can be useful for OAuth callbacks, health checks, or admin APIs.
The handler function must be an async function that accepts a Starlette
Request and returns a Response.

**Args:**

* `path`: URL path for the route (e.g., "/auth/callback")
* `methods`: List of HTTP methods to support (e.g., \["GET", "POST"])
* `name`: Optional name for the route (to reference this route with
  Starlette's reverse URL lookup feature)
* `include_in_schema`: Whether to include in OpenAPI schema, defaults to True

#### `run_stdio_async` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/transport.py#L158"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_stdio_async(self: FastMCP, show_banner: bool = True, log_level: str | None = None, stateless: bool = False) -> None
```

Run the server using stdio transport.

**Args:**

* `show_banner`: Whether to display the server banner
* `log_level`: Log level for the server
* `stateless`: Whether to run in stateless mode (no session initialization)

#### `run_http_async` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/transport.py#L200"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_http_async(self: FastMCP, show_banner: bool = True, transport: Literal['http', 'streamable-http', 'sse'] = 'http', host: str | None = None, port: int | None = None, log_level: str | None = None, path: str | None = None, uvicorn_config: dict[str, Any] | None = None, middleware: list[ASGIMiddleware] | None = None, json_response: bool | None = None, stateless_http: bool | None = None, stateless: bool | None = None) -> None
```

Run the server using HTTP transport.

**Args:**

* `transport`: Transport protocol to use - "http" (default), "streamable-http", or "sse"
* `host`: Host address to bind to (defaults to settings.host)
* `port`: Port to bind to (defaults to settings.port)
* `log_level`: Log level for the server (defaults to settings.log\_level)
* `path`: Path for the endpoint (defaults to settings.streamable\_http\_path or settings.sse\_path)
* `uvicorn_config`: Additional configuration for the Uvicorn server
* `middleware`: A list of middleware to apply to the app
* `json_response`: Whether to use JSON response format (defaults to settings.json\_response)
* `stateless_http`: Whether to use stateless HTTP (defaults to settings.stateless\_http)
* `stateless`: Alias for stateless\_http for CLI consistency

#### `http_app` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/transport.py#L279"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
http_app(self: FastMCP, path: str | None = None, middleware: list[ASGIMiddleware] | None = None, json_response: bool | None = None, stateless_http: bool | None = None, transport: Literal['http', 'streamable-http', 'sse'] = 'http', event_store: EventStore | None = None, retry_interval: int | None = None) -> StarletteWithLifespan
```

Create a Starlette app using the specified HTTP transport.

**Args:**

* `path`: The path for the HTTP endpoint
* `middleware`: A list of middleware to apply to the app
* `json_response`: Whether to use JSON response format
* `stateless_http`: Whether to use stateless mode (new transport per request)
* `transport`: Transport protocol to use - "http", "streamable-http", or "sse"
* `event_store`: Optional event store for SSE polling/resumability. When set,
  enables clients to reconnect and resume receiving events after
  server-initiated disconnections. Only used with streamable-http transport.
* `retry_interval`: Optional retry interval in milliseconds for SSE polling.
  Controls how quickly clients should reconnect after server-initiated
  disconnections. Requires event\_store to be set. Only used with
  streamable-http transport.

**Returns:**

* A Starlette application configured with the specified transport


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-openapi-__init__



# `fastmcp.server.openapi`

OpenAPI server implementation for FastMCP.

.. deprecated::
This module is deprecated. Import from fastmcp.server.providers.openapi instead.

The recommended approach is to use OpenAPIProvider with FastMCP:

from fastmcp import FastMCP
from fastmcp.server.providers.openapi import OpenAPIProvider
import httpx

client = httpx.AsyncClient(base\_url="[https://api.example.com](https://api.example.com)")
provider = OpenAPIProvider(openapi\_spec=spec, client=client)

mcp = FastMCP("My API Server")
mcp.add\_provider(provider)

FastMCPOpenAPI is still available but deprecated.


# components
Source: https://gofastmcp.com/python-sdk/fastmcp-server-openapi-components



# `fastmcp.server.openapi.components`

OpenAPI component implementations - backwards compatibility stub.

This module is deprecated. Import from fastmcp.server.providers.openapi instead.


# routing
Source: https://gofastmcp.com/python-sdk/fastmcp-server-openapi-routing



# `fastmcp.server.openapi.routing`

Route mapping logic for OpenAPI operations.

.. deprecated::
This module is deprecated. Import from fastmcp.server.providers.openapi instead.


# server
Source: https://gofastmcp.com/python-sdk/fastmcp-server-openapi-server



# `fastmcp.server.openapi.server`

FastMCPOpenAPI - backwards compatibility wrapper.

This class is deprecated. Use FastMCP with OpenAPIProvider instead:

from fastmcp import FastMCP
from fastmcp.server.providers.openapi import OpenAPIProvider
import httpx

client = httpx.AsyncClient(base\_url="[https://api.example.com](https://api.example.com)")
provider = OpenAPIProvider(openapi\_spec=spec, client=client)
mcp = FastMCP("My API Server", providers=\[provider])

## Classes

### `FastMCPOpenAPI` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/openapi/server.py#L30"><Icon icon="github" /></a></sup>

FastMCP server implementation that creates components from an OpenAPI schema.

.. deprecated::
Use FastMCP with OpenAPIProvider instead. This class will be
removed in a future version.

Example (deprecated):

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.openapi import FastMCPOpenAPI
import httpx

server = FastMCPOpenAPI(
    openapi_spec=spec,
    client=httpx.AsyncClient(),
)
```


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-__init__



# `fastmcp.server.providers`

Providers for dynamic MCP components.

This module provides the `Provider` abstraction for providing tools,
resources, and prompts dynamically at runtime.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool

class DatabaseProvider(Provider):
    def __init__(self, db_url: str):
        self.db = Database(db_url)

    async def _list_tools(self) -> list[Tool]:
        rows = await self.db.fetch("SELECT * FROM tools")
        return [self._make_tool(row) for row in rows]

    async def _get_tool(self, name: str) -> Tool | None:
        row = await self.db.fetchone("SELECT * FROM tools WHERE name = ?", name)
        return self._make_tool(row) if row else None

mcp = FastMCP("Server", providers=[DatabaseProvider(db_url)])
```


# aggregate
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-aggregate



# `fastmcp.server.providers.aggregate`

AggregateProvider for combining multiple providers into one.

This module provides `AggregateProvider`, a utility class that presents
multiple providers as a single unified provider. Useful when you want to
combine custom providers without creating a full FastMCP server.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.providers import AggregateProvider

# Combine multiple providers into one
combined = AggregateProvider()
combined.add_provider(provider1)
combined.add_provider(provider2, namespace="api")  # Tools become "api_foo"

# Use like any other provider
tools = await combined.list_tools()
```

## Classes

### `AggregateProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/aggregate.py#L46"><Icon icon="github" /></a></sup>

Utility provider that combines multiple providers into one.

Components are aggregated from all providers. For get\_\* operations,
providers are queried in parallel and the highest version is returned.

When adding providers with a namespace, wrap\_transform() is used to apply
the Namespace transform. This means namespace transformation is handled
by the wrapped provider, not by AggregateProvider.

Errors from individual providers are logged and skipped (graceful degradation).

**Methods:**

#### `add_provider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/aggregate.py#L78"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_provider(self, provider: Provider) -> None
```

Add a provider with optional namespace.

If the provider is a FastMCP server, it's automatically wrapped in
FastMCPProvider to ensure middleware is invoked correctly.

**Args:**

* `provider`: The provider to add.
* `namespace`: Optional namespace prefix. When set:
* Tools become "namespace\_toolname"
* Resources become "protocol://namespace/path"
* Prompts become "namespace\_promptname"

#### `get_tasks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/aggregate.py#L243"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tasks(self) -> Sequence[FastMCPComponent]
```

Get all task-eligible components from all providers.

#### `lifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/aggregate.py#L256"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
lifespan(self) -> AsyncIterator[None]
```

Combine lifespans of all providers.


# base
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-base



# `fastmcp.server.providers.base`

Base Provider class for dynamic MCP components.

This module provides the `Provider` abstraction for providing tools,
resources, and prompts dynamically at runtime.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool

class DatabaseProvider(Provider):
    def __init__(self, db_url: str):
        super().__init__()
        self.db = Database(db_url)

    async def _list_tools(self) -> list[Tool]:
        rows = await self.db.fetch("SELECT * FROM tools")
        return [self._make_tool(row) for row in rows]

    async def _get_tool(self, name: str) -> Tool | None:
        row = await self.db.fetchone("SELECT * FROM tools WHERE name = ?", name)
        return self._make_tool(row) if row else None

mcp = FastMCP("Server", providers=[DatabaseProvider(db_url)])
```

## Classes

### `Provider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L51"><Icon icon="github" /></a></sup>

Base class for dynamic component providers.

Subclass and override whichever methods you need. Default implementations
return empty lists / None, so you only need to implement what your provider
supports.

**Methods:**

#### `transforms` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L76"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transforms(self) -> list[Transform]
```

All transforms applied to components from this provider.

#### `add_transform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L80"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_transform(self, transform: Transform) -> None
```

Add a transform to this provider.

Transforms modify components (tools, resources, prompts) as they flow
through the provider. They're applied in order - first added is innermost.

**Args:**

* `transform`: The transform to add.

#### `wrap_transform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L100"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wrap_transform(self, transform: Transform) -> Provider
```

Return a new provider with this transform applied (immutable).

Unlike add\_transform() which mutates this provider, wrap\_transform()
returns a new provider that wraps this one. The original provider
is unchanged.

This is useful when you want to apply transforms without side effects,
such as adding the same provider to multiple aggregators with different
namespaces.

**Args:**

* `transform`: The transform to apply.

**Returns:**

* A new provider that wraps this one with the transform applied.

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L136"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self) -> Sequence[Tool]
```

List tools with all transforms applied.

Applies transforms sequentially: base → transforms (in order).
Each transform receives the result from the previous transform.
Components may be marked as disabled but are NOT filtered here -
filtering happens at the server level to allow session transforms to override.

**Returns:**

* Transformed sequence of tools (including disabled ones).

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L152"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, version: VersionSpec | None = None) -> Tool | None
```

Get tool by transformed name with all transforms applied.

Note: This method does NOT filter disabled components. The Server
(FastMCP) performs enabled filtering after all transforms complete,
allowing session-level transforms to override provider-level disables.

**Args:**

* `name`: The transformed tool name to look up.
* `version`: Optional version filter. If None, returns highest version.

**Returns:**

* The tool if found (may be marked disabled), None if not found.

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L178"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self) -> Sequence[Resource]
```

List resources with all transforms applied.

Components may be marked as disabled but are NOT filtered here.

#### `get_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L188"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource(self, uri: str, version: VersionSpec | None = None) -> Resource | None
```

Get resource by transformed URI with all transforms applied.

Note: This method does NOT filter disabled components. The Server
(FastMCP) performs enabled filtering after all transforms complete.

**Args:**

* `uri`: The transformed resource URI to look up.
* `version`: Optional version filter. If None, returns highest version.

**Returns:**

* The resource if found (may be marked disabled), None if not found.

#### `list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L213"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates(self) -> Sequence[ResourceTemplate]
```

List resource templates with all transforms applied.

Components may be marked as disabled but are NOT filtered here.

#### `get_resource_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L223"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_template(self, uri: str, version: VersionSpec | None = None) -> ResourceTemplate | None
```

Get resource template by transformed URI with all transforms applied.

Note: This method does NOT filter disabled components. The Server
(FastMCP) performs enabled filtering after all transforms complete.

**Args:**

* `uri`: The transformed template URI to look up.
* `version`: Optional version filter. If None, returns highest version.

**Returns:**

* The template if found (may be marked disabled), None if not found.

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L250"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self) -> Sequence[Prompt]
```

List prompts with all transforms applied.

Components may be marked as disabled but are NOT filtered here.

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L260"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self, name: str, version: VersionSpec | None = None) -> Prompt | None
```

Get prompt by transformed name with all transforms applied.

Note: This method does NOT filter disabled components. The Server
(FastMCP) performs enabled filtering after all transforms complete.

**Args:**

* `name`: The transformed prompt name to look up.
* `version`: Optional version filter. If None, returns highest version.

**Returns:**

* The prompt if found (may be marked disabled), None if not found.

#### `get_tasks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L418"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tasks(self) -> Sequence[FastMCPComponent]
```

Return components that should be registered as background tasks.

Override to customize which components are task-eligible.
Default calls list\_\* methods, applies provider transforms, and filters
for components with task\_config.mode != 'forbidden'.

Used by the server during startup to register functions with Docket.

#### `lifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L463"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
lifespan(self) -> AsyncIterator[None]
```

User-overridable lifespan for custom setup and teardown.

Override this method to perform provider-specific initialization
like opening database connections, setting up external resources,
or other state management needed for the provider's lifetime.

The lifespan scope matches the server's lifespan - code before yield
runs at startup, code after yield runs at shutdown.

#### `enable` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L492"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
enable(self) -> Self
```

Enable components matching all specified criteria.

Adds a visibility transform that marks matching components as enabled.
Later transforms override earlier ones, so enable after disable makes
the component enabled.

With only=True, switches to allowlist mode - first disables everything,
then enables matching components.

**Args:**

* `names`: Component names or URIs to enable.
* `keys`: Component keys to enable (e.g., ).
* `version`: Component version spec to enable (e.g., VersionSpec(eq="v1") or
  VersionSpec(gte="v2")). Unversioned components will not match.
* `tags`: Enable components with these tags.
* `components`: Component types to include (e.g., ).
* `only`: If True, ONLY enable matching components (allowlist mode).

**Returns:**

* Self for method chaining.

#### `disable` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/base.py#L541"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
disable(self) -> Self
```

Disable components matching all specified criteria.

Adds a visibility transform that marks matching components as disabled.
Components can be re-enabled by calling enable() with matching criteria
(the later transform wins).

**Args:**

* `names`: Component names or URIs to disable.
* `keys`: Component keys to disable (e.g., ).
* `version`: Component version spec to disable (e.g., VersionSpec(eq="v1") or
  VersionSpec(gte="v2")). Unversioned components will not match.
* `tags`: Disable components with these tags.
* `components`: Component types to include (e.g., ).

**Returns:**

* Self for method chaining.


# fastmcp_provider
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-fastmcp_provider



# `fastmcp.server.providers.fastmcp_provider`

FastMCPProvider for wrapping FastMCP servers as providers.

This module provides the `FastMCPProvider` class that wraps a FastMCP server
and exposes its components through the Provider interface.

It also provides FastMCPProvider\* component classes that delegate execution to
the wrapped server's middleware, ensuring middleware runs when components are
executed.

## Classes

### `FastMCPProviderTool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L72"><Icon icon="github" /></a></sup>

Tool that delegates execution to a wrapped server's middleware.

When `run()` is called, this tool invokes the wrapped server's
`_call_tool_middleware()` method, ensuring the server's middleware
chain is executed.

**Methods:**

#### `wrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L94"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wrap(cls, server: Any, tool: Tool) -> FastMCPProviderTool
```

Wrap a Tool to delegate execution to the server's middleware.

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L147"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, arguments: dict[str, Any]) -> ToolResult
```

Delegate to child server's call\_tool() without task\_meta.

This is called when the tool is used within a TransformedTool
forwarding function or other contexts where task\_meta is not available.

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L166"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

### `FastMCPProviderResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L173"><Icon icon="github" /></a></sup>

Resource that delegates reading to a wrapped server's read\_resource().

When `read()` is called, this resource invokes the wrapped server's
`read_resource()` method, ensuring the server's middleware chain is executed.

**Methods:**

#### `wrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L194"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wrap(cls, server: Any, resource: Resource) -> FastMCPProviderResource
```

Wrap a Resource to delegate reading to the server's middleware.

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L237"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

### `FastMCPProviderPrompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L244"><Icon icon="github" /></a></sup>

Prompt that delegates rendering to a wrapped server's render\_prompt().

When `render()` is called, this prompt invokes the wrapped server's
`render_prompt()` method, ensuring the server's middleware chain is executed.

**Methods:**

#### `wrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L265"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wrap(cls, server: Any, prompt: Prompt) -> FastMCPProviderPrompt
```

Wrap a Prompt to delegate rendering to the server's middleware.

#### `render` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L316"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
render(self, arguments: dict[str, Any] | None = None) -> PromptResult
```

Delegate to child server's render\_prompt() without task\_meta.

This is called when the prompt is used within a transformed context
or other contexts where task\_meta is not available.

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L335"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

### `FastMCPProviderResourceTemplate` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L342"><Icon icon="github" /></a></sup>

Resource template that creates FastMCPProviderResources.

When `create_resource()` is called, this template creates a
FastMCPProviderResource that will invoke the wrapped server's middleware
when read.

**Methods:**

#### `wrap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L364"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wrap(cls, server: Any, template: ResourceTemplate) -> FastMCPProviderResourceTemplate
```

Wrap a ResourceTemplate to create FastMCPProviderResources.

#### `create_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L385"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_resource(self, uri: str, params: dict[str, Any]) -> Resource
```

Create a FastMCPProviderResource for the given URI.

The `uri` is the external/transformed URI (e.g., with namespace prefix).
We use `_original_uri_template` with `params` to construct the internal
URI that the nested server understands.

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L435"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self, arguments: dict[str, Any]) -> str | bytes | ResourceResult
```

Read the resource content for background task execution.

Reads the resource via the wrapped server and returns the ResourceResult.
This method is called by Docket during background task execution.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L456"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

No-op: the child's actual template is registered via get\_tasks().

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L459"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, params: dict[str, Any], **kwargs: Any) -> Execution
```

Schedule this template for background execution via docket.

The child's FunctionResourceTemplate.fn is registered (via get\_tasks),
and it expects splatted \*\*kwargs, so we splat params here.

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L478"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

### `FastMCPProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L490"><Icon icon="github" /></a></sup>

Provider that wraps a FastMCP server.

This provider enables mounting one FastMCP server onto another, exposing
the mounted server's tools, resources, and prompts through the parent
server.

Components returned by this provider are wrapped in FastMCPProvider\*
classes that delegate execution to the wrapped server's middleware chain.
This ensures middleware runs when components are executed.

**Methods:**

#### `get_tasks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L655"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tasks(self) -> Sequence[FastMCPComponent]
```

Return task-eligible components from the mounted server.

Returns the child's ACTUAL components (not wrapped) so their actual
functions get registered with Docket. Gets components with child
server's transforms applied, then applies this provider's transforms
for correct registration keys.

#### `lifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/fastmcp_provider.py#L696"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
lifespan(self) -> AsyncIterator[None]
```

Start the mounted server's user lifespan.

This starts only the wrapped server's user-defined lifespan, NOT its
full \_lifespan\_manager() (which includes Docket). The parent server's
Docket handles all background tasks.


# filesystem
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-filesystem



# `fastmcp.server.providers.filesystem`

FileSystemProvider for filesystem-based component discovery.

FileSystemProvider scans a directory for Python files, imports them, and
registers any Tool, Resource, ResourceTemplate, or Prompt objects found.

Components are created using the standalone decorators from fastmcp.tools,
fastmcp.resources, and fastmcp.prompts:

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# In mcp/tools.py
from fastmcp.tools import tool

@tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

# In main.py
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.providers import FileSystemProvider

mcp = FastMCP("MyServer", providers=[FileSystemProvider(Path(__file__).parent / "mcp")])
```

## Classes

### `FileSystemProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/filesystem.py#L47"><Icon icon="github" /></a></sup>

Provider that discovers components from the filesystem.

Scans a directory for Python files and registers any Tool, Resource,
ResourceTemplate, or Prompt objects found. Components are created using
the standalone decorators:

* @tool from fastmcp.tools
* @resource from fastmcp.resources
* @prompt from fastmcp.prompts

**Args:**

* `root`: Root directory to scan. Defaults to current directory.
* `reload`: If True, re-scan files on every request (dev mode).
  Defaults to False (scan once at init, cache results).


# filesystem_discovery
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-filesystem_discovery



# `fastmcp.server.providers.filesystem_discovery`

File discovery and module import utilities for filesystem-based routing.

This module provides functions to:

1. Discover Python files in a directory tree
2. Import modules (as packages if **init**.py exists, else directly)
3. Extract decorated components (Tool, Resource, Prompt objects) from imported modules

## Functions

### `discover_files` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/filesystem_discovery.py#L32"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
discover_files(root: Path) -> list[Path]
```

Recursively discover all Python files under a directory.

Excludes **init**.py files (they're for package structure, not components).

**Args:**

* `root`: Root directory to scan.

**Returns:**

* List of .py file paths, sorted for deterministic order.

### `import_module_from_file` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/filesystem_discovery.py#L109"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import_module_from_file(file_path: Path) -> ModuleType
```

Import a Python file as a module.

If the file is part of a package (directory has **init**.py), imports
it as a proper package member (relative imports work). Otherwise,
imports directly using spec\_from\_file\_location.

**Args:**

* `file_path`: Path to the Python file.

**Returns:**

* The imported module.

**Raises:**

* `ImportError`: If the module cannot be imported.

### `extract_components` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/filesystem_discovery.py#L175"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
extract_components(module: ModuleType) -> list[FastMCPComponent]
```

Extract all MCP components from a module.

Scans all module attributes for instances of Tool, Resource,
ResourceTemplate, or Prompt objects created by standalone decorators,
or functions decorated with @tool/@resource/@prompt that have **fastmcp** metadata.

**Args:**

* `module`: The imported module to scan.

**Returns:**

* List of component objects (Tool, Resource, ResourceTemplate, Prompt).

### `discover_and_import` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/filesystem_discovery.py#L295"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
discover_and_import(root: Path) -> DiscoveryResult
```

Discover files, import modules, and extract components.

This is the main entry point for filesystem-based discovery.

**Args:**

* `root`: Root directory to scan.

**Returns:**

* DiscoveryResult with components and any failed files.

## Classes

### `DiscoveryResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/filesystem_discovery.py#L24"><Icon icon="github" /></a></sup>

Result of filesystem discovery.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-local_provider-__init__



# `fastmcp.server.providers.local_provider`

LocalProvider for locally-defined MCP components.

This module provides the `LocalProvider` class that manages tools, resources,
templates, and prompts registered via decorators or direct methods.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-local_provider-decorators-__init__



# `fastmcp.server.providers.local_provider.decorators`

Decorator mixins for LocalProvider.

This module provides mixin classes that add decorator functionality
to LocalProvider for tools, resources, templates, and prompts.


# prompts
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-local_provider-decorators-prompts



# `fastmcp.server.providers.local_provider.decorators.prompts`

Prompt decorator mixin for LocalProvider.

This module provides the PromptDecoratorMixin class that adds prompt
registration functionality to LocalProvider.

## Classes

### `PromptDecoratorMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/prompts.py#L29"><Icon icon="github" /></a></sup>

Mixin class providing prompt decorator functionality for LocalProvider.

This mixin contains all methods related to:

* Prompt registration via add\_prompt()
* Prompt decorator (@provider.prompt)

**Methods:**

#### `add_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/prompts.py#L37"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_prompt(self: LocalProvider, prompt: Prompt | Callable[..., Any]) -> Prompt
```

Add a prompt to this provider's storage.

Accepts either a Prompt object or a decorated function with **fastmcp** metadata.

#### `prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/prompts.py#L74"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prompt(self: LocalProvider, name_or_fn: F) -> F
```

#### `prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/prompts.py#L91"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prompt(self: LocalProvider, name_or_fn: str | None = None) -> Callable[[F], F]
```

#### `prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/prompts.py#L107"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prompt(self: LocalProvider, name_or_fn: str | AnyFunction | None = None) -> Callable[[AnyFunction], FunctionPrompt] | FunctionPrompt | partial[Callable[[AnyFunction], FunctionPrompt] | FunctionPrompt]
```

Decorator to register a prompt.

This decorator supports multiple calling patterns:

* @provider.prompt (without parentheses)
* @provider.prompt() (with empty parentheses)
* @provider.prompt("custom\_name") (with name as first argument)
* @provider.prompt(name="custom\_name") (with name as keyword argument)
* provider.prompt(function, name="custom\_name") (direct function call)

**Args:**

* `name_or_fn`: Either a function (when used as @prompt), a string name, or None
* `name`: Optional name for the prompt (keyword-only, alternative to name\_or\_fn)
* `title`: Optional title for the prompt
* `description`: Optional description of what the prompt does
* `icons`: Optional icons for the prompt
* `tags`: Optional set of tags for categorizing the prompt
* `enabled`: Whether the prompt is enabled (default True). If False, adds to blocklist.
* `meta`: Optional meta information about the prompt
* `task`: Optional task configuration for background execution
* `auth`: Optional authorization checks for the prompt

**Returns:**

* The registered FunctionPrompt or a decorator function.


# resources
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-local_provider-decorators-resources



# `fastmcp.server.providers.local_provider.decorators.resources`

Resource decorator mixin for LocalProvider.

This module provides the ResourceDecoratorMixin class that adds resource
and template registration functionality to LocalProvider.

## Classes

### `ResourceDecoratorMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/resources.py#L29"><Icon icon="github" /></a></sup>

Mixin class providing resource decorator functionality for LocalProvider.

This mixin contains all methods related to:

* Resource registration via add\_resource()
* Resource template registration via add\_template()
* Resource decorator (@provider.resource)

**Methods:**

#### `add_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/resources.py#L38"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_resource(self: LocalProvider, resource: Resource | ResourceTemplate | Callable[..., Any]) -> Resource | ResourceTemplate
```

Add a resource to this provider's storage.

Accepts either a Resource/ResourceTemplate object or a decorated function with **fastmcp** metadata.

#### `add_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/resources.py#L101"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_template(self: LocalProvider, template: ResourceTemplate) -> ResourceTemplate
```

Add a resource template to this provider's storage.

#### `resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/resources.py#L107"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resource(self: LocalProvider, uri: str) -> Callable[[F], F]
```

Decorator to register a function as a resource.

If the URI contains parameters (e.g. "resource://") or the function
has parameters, it will be registered as a template resource.

**Args:**

* `uri`: URI for the resource (e.g. "resource://my-resource" or "resource://")
* `name`: Optional name for the resource
* `title`: Optional title for the resource
* `description`: Optional description of the resource
* `icons`: Optional icons for the resource
* `mime_type`: Optional MIME type for the resource
* `tags`: Optional set of tags for categorizing the resource
* `enabled`: Whether the resource is enabled (default True). If False, adds to blocklist.
* `annotations`: Optional annotations about the resource's behavior
* `meta`: Optional meta information about the resource
* `task`: Optional task configuration for background execution
* `auth`: Optional authorization checks for the resource

**Returns:**

* A decorator function.


# tools
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-local_provider-decorators-tools



# `fastmcp.server.providers.local_provider.decorators.tools`

Tool decorator mixin for LocalProvider.

This module provides the ToolDecoratorMixin class that adds tool
registration functionality to LocalProvider.

## Classes

### `ToolDecoratorMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/tools.py#L146"><Icon icon="github" /></a></sup>

Mixin class providing tool decorator functionality for LocalProvider.

This mixin contains all methods related to:

* Tool registration via add\_tool()
* Tool decorator (@provider.tool)

**Methods:**

#### `add_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/tools.py#L154"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_tool(self: LocalProvider, tool: Tool | Callable[..., Any]) -> Tool
```

Add a tool to this provider's storage.

Accepts either a Tool object or a decorated function with **fastmcp** metadata.

#### `tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/tools.py#L206"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool(self: LocalProvider, name_or_fn: F) -> F
```

#### `tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/tools.py#L228"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool(self: LocalProvider, name_or_fn: str | None = None) -> Callable[[F], F]
```

#### `tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/decorators/tools.py#L253"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool(self: LocalProvider, name_or_fn: str | AnyFunction | None = None) -> Callable[[AnyFunction], FunctionTool] | FunctionTool | partial[Callable[[AnyFunction], FunctionTool] | FunctionTool]
```

Decorator to register a tool.

This decorator supports multiple calling patterns:

* @provider.tool (without parentheses)
* @provider.tool() (with empty parentheses)
* @provider.tool("custom\_name") (with name as first argument)
* @provider.tool(name="custom\_name") (with name as keyword argument)
* provider.tool(function, name="custom\_name") (direct function call)

**Args:**

* `name_or_fn`: Either a function (when used as @tool), a string name, or None
* `name`: Optional name for the tool (keyword-only, alternative to name\_or\_fn)
* `title`: Optional title for the tool
* `description`: Optional description of what the tool does
* `icons`: Optional icons for the tool
* `tags`: Optional set of tags for categorizing the tool
* `output_schema`: Optional JSON schema for the tool's output
* `annotations`: Optional annotations about the tool's behavior
* `exclude_args`: Optional list of argument names to exclude from the tool schema
* `meta`: Optional meta information about the tool
* `enabled`: Whether the tool is enabled (default True). If False, adds to blocklist.
* `task`: Optional task configuration for background execution
* `serializer`: Deprecated. Return ToolResult from your tools for full control over serialization.

**Returns:**

* The registered FunctionTool or a decorator function.


# local_provider
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-local_provider-local_provider



# `fastmcp.server.providers.local_provider.local_provider`

LocalProvider for locally-defined MCP components.

This module provides the `LocalProvider` class that manages tools, resources,
templates, and prompts registered via decorators or direct methods.

LocalProvider can be used standalone and attached to multiple servers:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.providers import LocalProvider

# Create a reusable provider with tools
provider = LocalProvider()

@provider.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Attach to any server
from fastmcp import FastMCP
server1 = FastMCP("Server1", providers=[provider])
server2 = FastMCP("Server2", providers=[provider])
```

## Classes

### `LocalProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/local_provider.py#L51"><Icon icon="github" /></a></sup>

Provider for locally-defined components.

Supports decorator-based registration (`@provider.tool`, `@provider.resource`,
`@provider.prompt`) and direct object registration methods.

When used standalone, LocalProvider uses default settings. When attached
to a FastMCP server via the server's decorators, server-level settings
like `_tool_serializer` and `_support_tasks_by_default` are injected.

**Methods:**

#### `remove_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/local_provider.py#L229"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
remove_tool(self, name: str, version: str | None = None) -> None
```

Remove tool(s) from this provider's storage.

**Args:**

* `name`: The tool name.
* `version`: If None, removes ALL versions. If specified, removes only that version.

**Raises:**

* `KeyError`: If no matching tool is found.

#### `remove_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/local_provider.py#L257"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
remove_resource(self, uri: str, version: str | None = None) -> None
```

Remove resource(s) from this provider's storage.

**Args:**

* `uri`: The resource URI.
* `version`: If None, removes ALL versions. If specified, removes only that version.

**Raises:**

* `KeyError`: If no matching resource is found.

#### `remove_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/local_provider.py#L285"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
remove_template(self, uri_template: str, version: str | None = None) -> None
```

Remove resource template(s) from this provider's storage.

**Args:**

* `uri_template`: The template URI pattern.
* `version`: If None, removes ALL versions. If specified, removes only that version.

**Raises:**

* `KeyError`: If no matching template is found.

#### `remove_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/local_provider.py#L315"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
remove_prompt(self, name: str, version: str | None = None) -> None
```

Remove prompt(s) from this provider's storage.

**Args:**

* `name`: The prompt name.
* `version`: If None, removes ALL versions. If specified, removes only that version.

**Raises:**

* `KeyError`: If no matching prompt is found.

#### `get_tasks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/local_provider/local_provider.py#L449"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tasks(self) -> Sequence[FastMCPComponent]
```

Return components eligible for background task execution.

Returns components that have task\_config.mode != 'forbidden'.
This includes both FunctionTool/Resource/Prompt instances created via
decorators and custom Tool/Resource/Prompt subclasses.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-openapi-__init__



# `fastmcp.server.providers.openapi`

OpenAPI provider for FastMCP.

This module provides OpenAPI integration for FastMCP through the Provider pattern.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.providers.openapi import OpenAPIProvider
import httpx

client = httpx.AsyncClient(base_url="https://api.example.com")
provider = OpenAPIProvider(openapi_spec=spec, client=client)
mcp = FastMCP("API Server", providers=[provider])
```


# components
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-openapi-components



# `fastmcp.server.providers.openapi.components`

OpenAPI component classes: Tool, Resource, and ResourceTemplate.

## Classes

### `OpenAPITool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/components.py#L118"><Icon icon="github" /></a></sup>

Tool implementation for OpenAPI endpoints.

**Methods:**

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/components.py#L160"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, arguments: dict[str, Any]) -> ToolResult
```

Execute the HTTP request using RequestDirector.

### `OpenAPIResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/components.py#L235"><Icon icon="github" /></a></sup>

Resource implementation for OpenAPI endpoints.

**Methods:**

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/components.py#L265"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> ResourceResult
```

Fetch the resource data by making an HTTP request.

### `OpenAPIResourceTemplate` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/components.py#L349"><Icon icon="github" /></a></sup>

Resource template implementation for OpenAPI endpoints.

**Methods:**

#### `create_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/components.py#L381"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_resource(self, uri: str, params: dict[str, Any], context: Context | None = None) -> Resource
```

Create a resource with the given parameters.


# provider
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-openapi-provider



# `fastmcp.server.providers.openapi.provider`

OpenAPIProvider for creating MCP components from OpenAPI specifications.

## Classes

### `OpenAPIProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/provider.py#L51"><Icon icon="github" /></a></sup>

Provider that creates MCP components from an OpenAPI specification.

Components are created eagerly during initialization by parsing the OpenAPI
spec. Each component makes HTTP calls to the described API endpoints.

**Methods:**

#### `lifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/provider.py#L181"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
lifespan(self) -> AsyncIterator[None]
```

Manage the lifecycle of the auto-created httpx client.

#### `get_tasks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/provider.py#L431"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tasks(self) -> Sequence[FastMCPComponent]
```

Return empty list - OpenAPI components don't support tasks.


# routing
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-openapi-routing



# `fastmcp.server.providers.openapi.routing`

Route mapping logic for OpenAPI operations.

## Classes

### `MCPType` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/routing.py#L42"><Icon icon="github" /></a></sup>

Type of FastMCP component to create from a route.

### `RouteMap` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/openapi/routing.py#L59"><Icon icon="github" /></a></sup>

Mapping configuration for HTTP routes to FastMCP component types.


# proxy
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-proxy



# `fastmcp.server.providers.proxy`

ProxyProvider for proxying to remote MCP servers.

This module provides the `ProxyProvider` class that proxies components from
a remote MCP server via a client factory. It also provides proxy component
classes that forward execution to remote servers.

## Functions

### `default_proxy_roots_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L723"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_proxy_roots_handler(context: RequestContext[ClientSession, LifespanContextT]) -> RootsList
```

Forward list roots request from remote server to proxy's connected clients.

### `default_proxy_sampling_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L731"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_proxy_sampling_handler(messages: list[mcp.types.SamplingMessage], params: mcp.types.CreateMessageRequestParams, context: RequestContext[ClientSession, LifespanContextT]) -> mcp.types.CreateMessageResult
```

Forward sampling request from remote server to proxy's connected clients.

### `default_proxy_elicitation_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L754"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_proxy_elicitation_handler(message: str, response_type: type, params: mcp.types.ElicitRequestParams, context: RequestContext[ClientSession, LifespanContextT]) -> ElicitResult
```

Forward elicitation request from remote server to proxy's connected clients.

### `default_proxy_log_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L776"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_proxy_log_handler(message: LogMessage) -> None
```

Forward log notification from remote server to proxy's connected clients.

### `default_proxy_progress_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L784"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_proxy_progress_handler(progress: float, total: float | None, message: str | None) -> None
```

Forward progress notification from remote server to proxy's connected clients.

## Classes

### `ProxyTool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L67"><Icon icon="github" /></a></sup>

A Tool that represents and executes a tool on a remote server.

**Methods:**

#### `model_copy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L84"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
model_copy(self, **kwargs: Any) -> ProxyTool
```

Override to preserve \_backend\_name when name changes.

#### `from_mcp_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L94"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_mcp_tool(cls, client_factory: ClientFactoryT, mcp_tool: mcp.types.Tool) -> ProxyTool
```

Factory method to create a ProxyTool from a raw MCP tool schema.

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L111"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, arguments: dict[str, Any], context: Context | None = None) -> ToolResult
```

Executes the tool by making a call through the client.

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L166"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

### `ProxyResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L173"><Icon icon="github" /></a></sup>

A Resource that represents and reads a resource from a remote server.

**Methods:**

#### `model_copy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L198"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
model_copy(self, **kwargs: Any) -> ProxyResource
```

Override to preserve \_backend\_uri when uri changes.

#### `from_mcp_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L208"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_mcp_resource(cls, client_factory: ClientFactoryT, mcp_resource: mcp.types.Resource) -> ProxyResource
```

Factory method to create a ProxyResource from a raw MCP resource schema.

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L228"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> ResourceResult
```

Read the resource content from the remote server.

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L273"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

### `ProxyTemplate` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L280"><Icon icon="github" /></a></sup>

A ResourceTemplate that represents and creates resources from a remote server template.

**Methods:**

#### `model_copy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L297"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
model_copy(self, **kwargs: Any) -> ProxyTemplate
```

Override to preserve \_backend\_uri\_template when uri\_template changes.

#### `from_mcp_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L307"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_mcp_template(cls, client_factory: ClientFactoryT, mcp_template: mcp.types.ResourceTemplate) -> ProxyTemplate
```

Factory method to create a ProxyTemplate from a raw MCP template schema.

#### `create_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L326"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_resource(self, uri: str, params: dict[str, Any], context: Context | None = None) -> ProxyResource
```

Create a resource from the template by calling the remote server.

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L388"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

### `ProxyPrompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L395"><Icon icon="github" /></a></sup>

A Prompt that represents and renders a prompt from a remote server.

**Methods:**

#### `model_copy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L412"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
model_copy(self, **kwargs: Any) -> ProxyPrompt
```

Override to preserve \_backend\_name when name changes.

#### `from_mcp_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L422"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_mcp_prompt(cls, client_factory: ClientFactoryT, mcp_prompt: mcp.types.Prompt) -> ProxyPrompt
```

Factory method to create a ProxyPrompt from a raw MCP prompt schema.

#### `render` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L446"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
render(self, arguments: dict[str, Any]) -> PromptResult
```

Render the prompt by making a call through the client.

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L468"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

### `ProxyProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L480"><Icon icon="github" /></a></sup>

Provider that proxies to a remote MCP server via a client factory.

This provider fetches components from a remote server and returns Proxy\*
component instances that forward execution to the remote server.

All components returned by this provider have task\_config.mode="forbidden"
because tasks cannot be executed through a proxy.

**Methods:**

#### `get_tasks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L605"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tasks(self) -> Sequence[FastMCPComponent]
```

Return empty list since proxy components don't support tasks.

Override the base implementation to avoid calling list\_tools() during
server lifespan initialization, which would open the client before any
context is set. All Proxy\* components have task\_config.mode="forbidden".

### `FastMCPProxy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L676"><Icon icon="github" /></a></sup>

A FastMCP server that acts as a proxy to a remote MCP-compliant server.

This is a convenience wrapper that creates a FastMCP server with a
ProxyProvider. For more control, use FastMCP with add\_provider(ProxyProvider(...)).

### `ProxyClient` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L848"><Icon icon="github" /></a></sup>

A proxy client that forwards advanced interactions between a remote MCP server and the proxy's connected clients.

Supports forwarding roots, sampling, elicitation, logging, and progress.

### `StatefulProxyClient` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L881"><Icon icon="github" /></a></sup>

A proxy client that provides a stateful client factory for the proxy server.

The stateful proxy client bound its copy to the server session.
And it will be disconnected when the session is exited.

This is useful to proxy a stateful mcp server such as the Playwright MCP server.
Note that it is essential to ensure that the proxy server itself is also stateful.

Because session reuse means the receive-loop task inherits a stale
`request_ctx` ContextVar snapshot, the default proxy handlers are
replaced with versions that restore the ContextVar before forwarding.
`ProxyTool.run` stashes the current `RequestContext` in
`_proxy_rc_ref` before each backend call, and the handlers consult
it to detect (and correct) staleness.

**Methods:**

#### `clear` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L932"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
clear(self)
```

Clear all cached clients and force disconnect them.

#### `new_stateful` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/proxy.py#L938"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
new_stateful(self) -> Client[ClientTransportT]
```

Create a new stateful proxy client instance with the same configuration.

Use this method as the client factory for stateful proxy server.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-skills-__init__



# `fastmcp.server.providers.skills`

Skills providers for exposing agent skills as MCP resources.

This module provides a two-layer architecture for skill discovery:

* **SkillProvider**: Handles a single skill folder, exposing its files as resources.
* **SkillsDirectoryProvider**: Scans a directory, creates a SkillProvider per folder.
* **Vendor providers**: Platform-specific providers for Claude, Cursor, VS Code, Codex,
  Gemini, Goose, Copilot, and OpenCode.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from pathlib import Path
from fastmcp import FastMCP
from fastmcp.server.providers.skills import ClaudeSkillsProvider, SkillProvider

mcp = FastMCP("Skills Server")

# Load a single skill
mcp.add_provider(SkillProvider(Path.home() / ".claude/skills/pdf-processing"))

# Or load all skills in a directory
mcp.add_provider(ClaudeSkillsProvider())  # Uses ~/.claude/skills/
```


# claude_provider
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-skills-claude_provider



# `fastmcp.server.providers.skills.claude_provider`

Claude-specific skills provider for Claude Code skills.

## Classes

### `ClaudeSkillsProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/claude_provider.py#L11"><Icon icon="github" /></a></sup>

Provider for Claude Code skills from \~/.claude/skills/.

A convenience subclass that sets the default root to Claude's skills location.

**Args:**

* `reload`: If True, re-scan on every request. Defaults to False.
* `supporting_files`: How supporting files are exposed:
* "template": Accessed via ResourceTemplate, hidden from list\_resources().
* "resources": Each file exposed as individual Resource in list\_resources().


# directory_provider
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-skills-directory_provider



# `fastmcp.server.providers.skills.directory_provider`

Directory scanning provider for discovering multiple skills.

## Classes

### `SkillsDirectoryProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/directory_provider.py#L19"><Icon icon="github" /></a></sup>

Provider that scans directories and creates a SkillProvider per skill folder.

This extends AggregateProvider to combine multiple SkillProviders into one.
Each subdirectory containing a main file (default: SKILL.md) becomes a skill.
Can scan multiple root directories - if a skill name appears in multiple roots,
the first one found wins.

**Args:**

* `roots`: Root directory(ies) containing skill folders. Can be a single path
  or a sequence of paths.
* `reload`: If True, re-discover skills on each request. Defaults to False.
* `main_file_name`: Name of the main skill file. Defaults to "SKILL.md".
* `supporting_files`: How supporting files are exposed in child SkillProviders:
* "template": Accessed via ResourceTemplate, hidden from list\_resources().
* "resources": Each file exposed as individual Resource in list\_resources().


# skill_provider
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-skills-skill_provider



# `fastmcp.server.providers.skills.skill_provider`

Basic skill provider for handling a single skill folder.

## Classes

### `SkillResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L35"><Icon icon="github" /></a></sup>

A resource representing a skill's main file or manifest.

**Methods:**

#### `get_meta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L41"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_meta(self) -> dict[str, Any]
```

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L50"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> str | bytes | ResourceResult
```

Read the resource content.

### `SkillFileTemplate` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L70"><Icon icon="github" /></a></sup>

A template for accessing files within a skill.

**Methods:**

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L75"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self, arguments: dict[str, Any]) -> str | bytes | ResourceResult
```

Read a file from the skill directory.

#### `create_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L115"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_resource(self, uri: str, params: dict[str, Any]) -> Resource
```

Create a resource for the given URI and parameters.

Note: This is not typically used since \_read() handles file reading directly.
Provided for compatibility with the ResourceTemplate interface.

### `SkillFileResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L141"><Icon icon="github" /></a></sup>

A resource representing a specific file within a skill.

**Methods:**

#### `get_meta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L147"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_meta(self) -> dict[str, Any]
```

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L155"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> str | bytes | ResourceResult
```

Read the file content.

### `SkillProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L179"><Icon icon="github" /></a></sup>

Provider that exposes a single skill folder as MCP resources.

Each skill folder must contain a main file (default: SKILL.md) and may
contain additional supporting files.

Exposes:

* A Resource for the main file (skill:///SKILL.md)
* A Resource for the synthetic manifest (skill:///\_manifest)
* Supporting files via ResourceTemplate or Resources (configurable)

**Args:**

* `skill_path`: Path to the skill directory.
* `main_file_name`: Name of the main skill file. Defaults to "SKILL.md".
* `supporting_files`: How supporting files (everything except main file and
  manifest) are exposed to clients:
* "template": Accessed via ResourceTemplate, hidden from list\_resources().
  Clients discover files by reading the manifest first.
* "resources": Each file exposed as individual Resource in list\_resources().
  Full enumeration upfront.

**Methods:**

#### `skill_info` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/skill_provider.py#L271"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
skill_info(self) -> SkillInfo
```

Get the loaded skill info.


# vendor_providers
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-skills-vendor_providers



# `fastmcp.server.providers.skills.vendor_providers`

Vendor-specific skills providers for various AI coding platforms.

## Classes

### `CursorSkillsProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/vendor_providers.py#L11"><Icon icon="github" /></a></sup>

Cursor skills from \~/.cursor/skills/.

### `VSCodeSkillsProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/vendor_providers.py#L29"><Icon icon="github" /></a></sup>

VS Code skills from \~/.copilot/skills/.

### `CodexSkillsProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/vendor_providers.py#L47"><Icon icon="github" /></a></sup>

Codex skills from /etc/codex/skills/ and \~/.codex/skills/.

Scans both system-level and user-level directories. System skills take
precedence if duplicates exist.

### `GeminiSkillsProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/vendor_providers.py#L73"><Icon icon="github" /></a></sup>

Gemini skills from \~/.gemini/skills/.

### `GooseSkillsProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/vendor_providers.py#L91"><Icon icon="github" /></a></sup>

Goose skills from \~/.config/agents/skills/.

### `CopilotSkillsProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/vendor_providers.py#L109"><Icon icon="github" /></a></sup>

GitHub Copilot skills from \~/.copilot/skills/.

### `OpenCodeSkillsProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/providers/skills/vendor_providers.py#L127"><Icon icon="github" /></a></sup>

OpenCode skills from \~/.config/opencode/skills/.


# wrapped_provider
Source: https://gofastmcp.com/python-sdk/fastmcp-server-providers-wrapped_provider



# `fastmcp.server.providers.wrapped_provider`

WrappedProvider for immutable transform composition.

This module provides `_WrappedProvider`, an internal class that wraps a provider
with an additional transform. Created by `Provider.wrap_transform()`.


# proxy
Source: https://gofastmcp.com/python-sdk/fastmcp-server-proxy



# `fastmcp.server.proxy`

Backwards compatibility - import from fastmcp.server.providers.proxy instead.

This module re-exports all proxy-related classes from their new location
at fastmcp.server.providers.proxy. Direct imports from this module are
deprecated and will be removed in a future version.
