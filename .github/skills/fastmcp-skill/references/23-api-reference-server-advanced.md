# API Reference: Server Advanced

Source lines: 43133-46745 from the original FastMCP documentation dump.

Package-level API reference for sampling, server tasks, telemetry, transforms, and core tool classes.

---

# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-sampling-__init__



# `fastmcp.server.sampling`

Sampling module for FastMCP servers.


# run
Source: https://gofastmcp.com/python-sdk/fastmcp-server-sampling-run



# `fastmcp.server.sampling.run`

Sampling types and helper functions for FastMCP servers.

## Functions

### `determine_handler_mode` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L132"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
determine_handler_mode(context: Context, needs_tools: bool) -> bool
```

Determine whether to use fallback handler or client for sampling.

**Args:**

* `context`: The MCP context.
* `needs_tools`: Whether the sampling request requires tool support.

**Returns:**

* True if fallback handler should be used, False to use client.

**Raises:**

* `ValueError`: If client lacks required capability and no fallback configured.

### `call_sampling_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L191"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_sampling_handler(context: Context, messages: list[SamplingMessage]) -> CreateMessageResult | CreateMessageResultWithTools
```

Make LLM call using the fallback handler.

Note: This function expects the caller (sample\_step) to have validated that
sampling\_handler is set via determine\_handler\_mode(). The checks below are
safeguards against internal misuse.

### `execute_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L242"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
execute_tools(tool_calls: list[ToolUseContent], tool_map: dict[str, SamplingTool], mask_error_details: bool = False, tool_concurrency: int | None = None) -> list[ToolResultContent]
```

Execute tool calls and return results.

**Args:**

* `tool_calls`: List of tool use requests from the LLM.
* `tool_map`: Mapping from tool name to SamplingTool.
* `mask_error_details`: If True, mask detailed error messages from tool execution.
  When masked, only generic error messages are returned to the LLM.
  Tools can explicitly raise ToolError to bypass masking when they want
  to provide specific error messages to the LLM.
* `tool_concurrency`: Controls parallel execution of tools:
* None (default): Sequential execution (one at a time)
* 0: Unlimited parallel execution
* N > 0: Execute at most N tools concurrently
  If any tool has sequential=True, all tools execute sequentially
  regardless of this setting.

**Returns:**

* List of tool result content blocks in the same order as tool\_calls.

### `prepare_messages` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L352"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prepare_messages(messages: str | Sequence[str | SamplingMessage]) -> list[SamplingMessage]
```

Convert various message formats to a list of SamplingMessage objects.

### `prepare_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L371"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prepare_tools(tools: Sequence[SamplingTool | FunctionTool | TransformedTool | Callable[..., Any]] | None) -> list[SamplingTool] | None
```

Convert tools to SamplingTool objects.

Accepts SamplingTool instances, FunctionTool instances, TransformedTool instances,
or plain callable functions. FunctionTool and TransformedTool are converted using
from\_callable\_tool(), while plain functions use from\_function().

**Args:**

* `tools`: Sequence of tools to prepare. Can be SamplingTool, FunctionTool,
  TransformedTool, or plain callable functions.

**Returns:**

* List of SamplingTool instances, or None if tools is None.

### `extract_tool_calls` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L407"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
extract_tool_calls(response: CreateMessageResult | CreateMessageResultWithTools) -> list[ToolUseContent]
```

Extract tool calls from a response.

### `create_final_response_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L419"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_final_response_tool(result_type: type) -> SamplingTool
```

Create a synthetic 'final\_response' tool for structured output.

This tool is used to capture structured responses from the LLM.
The tool's schema is derived from the result\_type.

### `sample_step_impl` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L455"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
sample_step_impl(context: Context, messages: str | Sequence[str | SamplingMessage]) -> SampleStep
```

Implementation of Context.sample\_step().

Make a single LLM sampling call. This is a stateless function that makes
exactly one LLM call and optionally executes any requested tools.

### `sample_impl` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L572"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
sample_impl(context: Context, messages: str | Sequence[str | SamplingMessage]) -> SamplingResult[ResultT]
```

Implementation of Context.sample().

Send a sampling request to the client and await the response. This method
runs to completion automatically, executing a tool loop until the LLM
provides a final text response.

## Classes

### `SamplingResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L54"><Icon icon="github" /></a></sup>

Result of a sampling operation.

**Attributes:**

* `text`: The text representation of the result (raw text or JSON for structured).
* `result`: The typed result (str for text, parsed object for structured output).
* `history`: All messages exchanged during sampling.

### `SampleStep` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L69"><Icon icon="github" /></a></sup>

Result of a single sampling call.

Represents what the LLM returned in this step plus the message history.

**Methods:**

#### `is_tool_use` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L79"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_tool_use(self) -> bool
```

True if the LLM is requesting tool execution.

#### `text` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L86"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
text(self) -> str | None
```

Extract text from the response, if available.

#### `tool_calls` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/run.py#L99"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool_calls(self) -> list[ToolUseContent]
```

Get the list of tool calls from the response.


# sampling_tool
Source: https://gofastmcp.com/python-sdk/fastmcp-server-sampling-sampling_tool



# `fastmcp.server.sampling.sampling_tool`

SamplingTool for use during LLM sampling requests.

## Classes

### `SamplingTool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/sampling_tool.py#L20"><Icon icon="github" /></a></sup>

A tool that can be used during LLM sampling.

SamplingTools bundle a tool's schema (name, description, parameters) with
an executor function, enabling servers to execute agentic workflows where
the LLM can request tool calls during sampling.

In most cases, pass functions directly to ctx.sample():

def search(query: str) -> str:
'''Search the web.'''
return web\_search(query)

result = await context.sample(
messages="Find info about Python",
tools=\[search],  # Plain functions work directly
)

Create a SamplingTool explicitly when you need custom name/description:

tool = SamplingTool.from\_function(search, name="web\_search")

**Methods:**

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/sampling_tool.py#L51"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, arguments: dict[str, Any] | None = None) -> Any
```

Execute the tool with the given arguments.

**Args:**

* `arguments`: Dictionary of arguments to pass to the tool function.

**Returns:**

* The result of executing the tool function.

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/sampling_tool.py#L81"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any]) -> SamplingTool
```

Create a SamplingTool from a function.

The function's signature is analyzed to generate a JSON schema for
the tool's parameters. Type hints are used to determine parameter types.

**Args:**

* `fn`: The function to create a tool from.
* `name`: Optional name override. Defaults to the function's name.
* `description`: Optional description override. Defaults to the function's docstring.
* `sequential`: If True, this tool requires sequential execution and prevents
  parallel execution of all tools in the batch. Set to True for tools
  with shared state, file writes, or other operations that cannot run
  concurrently. Defaults to False.

**Returns:**

* A SamplingTool wrapping the function.

**Raises:**

* `ValueError`: If the function is a lambda without a name override.

#### `from_callable_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/sampling/sampling_tool.py#L123"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_callable_tool(cls, tool: FunctionTool | TransformedTool) -> SamplingTool
```

Create a SamplingTool from a FunctionTool or TransformedTool.

Reuses existing server tools in sampling contexts. For TransformedTool,
the tool's .run() method is used to ensure proper argument transformation,
and the ToolResult is automatically unwrapped.

**Args:**

* `tool`: A FunctionTool or TransformedTool to convert.
* `name`: Optional name override. Defaults to tool.name.
* `description`: Optional description override. Defaults to tool.description.

**Raises:**

* `TypeError`: If the tool is not a FunctionTool or TransformedTool.


# server
Source: https://gofastmcp.com/python-sdk/fastmcp-server-server



# `fastmcp.server.server`

FastMCP - A more ergonomic interface for MCP servers.

## Functions

### `default_lifespan` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L170"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_lifespan(server: FastMCP[LifespanResultT]) -> AsyncIterator[Any]
```

Default lifespan context manager that does nothing.

**Args:**

* `server`: The server instance this lifespan is managing

**Returns:**

* An empty dictionary as the lifespan result.

### `create_proxy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L2084"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_proxy(target: Client[ClientTransportT] | ClientTransport | FastMCP[Any] | FastMCP1Server | AnyUrl | Path | MCPConfig | dict[str, Any] | str, **settings: Any) -> FastMCPProxy
```

Create a FastMCP proxy server for the given target.

This is the recommended way to create a proxy server. For lower-level control,
use `FastMCPProxy` or `ProxyProvider` directly from `fastmcp.server.providers.proxy`.

**Args:**

* `target`: The backend to proxy to. Can be:
* A Client instance (connected or disconnected)
* A ClientTransport
* A FastMCP server instance
* A URL string or AnyUrl
* A Path to a server script
* An MCPConfig or dict
* `**settings`: Additional settings passed to FastMCPProxy (name, etc.)

**Returns:**

* A FastMCPProxy server that proxies to the target.

## Classes

### `StateValue` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L206"><Icon icon="github" /></a></sup>

Wrapper for stored context state values.

### `FastMCP` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L212"><Icon icon="github" /></a></sup>

**Methods:**

#### `name` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L353"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
name(self) -> str
```

#### `instructions` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L357"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
instructions(self) -> str | None
```

#### `instructions` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L361"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
instructions(self, value: str | None) -> None
```

#### `version` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L365"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
version(self) -> str | None
```

#### `website_url` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L369"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
website_url(self) -> str | None
```

#### `icons` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L373"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
icons(self) -> list[mcp.types.Icon]
```

#### `local_provider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L380"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
local_provider(self) -> LocalProvider
```

The server's local provider, which stores directly-registered components.

Use this to remove components:

mcp.local\_provider.remove\_tool("my\_tool")
mcp.local\_provider.remove\_resource("data://info")
mcp.local\_provider.remove\_prompt("my\_prompt")

#### `add_middleware` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L402"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_middleware(self, middleware: Middleware) -> None
```

#### `add_provider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L405"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_provider(self, provider: Provider) -> None
```

Add a provider for dynamic tools, resources, and prompts.

Providers are queried in registration order. The first provider to return
a non-None result wins. Static components (registered via decorators)
always take precedence over providers.

**Args:**

* `provider`: A Provider instance that will provide components dynamically.
* `namespace`: Optional namespace prefix. When set:
* Tools become "namespace\_toolname"
* Resources become "protocol://namespace/path"
* Prompts become "namespace\_promptname"

#### `get_tasks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L427"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tasks(self) -> Sequence[FastMCPComponent]
```

Get task-eligible components with all transforms applied.

Overrides AggregateProvider.get\_tasks() to apply server-level transforms
after aggregation. AggregateProvider handles provider-level namespacing.

#### `add_transform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L456"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_transform(self, transform: Transform) -> None
```

Add a server-level transform.

Server-level transforms are applied after all providers are aggregated.
They transform tools, resources, and prompts from ALL providers.

**Args:**

* `transform`: The transform to add.

#### `add_tool_transformation` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L476"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_tool_transformation(self, tool_name: str, transformation: ToolTransformConfig) -> None
```

Add a tool transformation.

.. deprecated::
Use `add_transform(ToolTransform({...}))` instead.

#### `remove_tool_transformation` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L493"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
remove_tool_transformation(self, _tool_name: str) -> None
```

Remove a tool transformation.

.. deprecated::
Tool transformations are now immutable. Use enable/disable controls instead.

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L508"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self) -> Sequence[Tool]
```

List all enabled tools from providers.

Overrides Provider.list\_tools() to add visibility filtering, auth filtering,
and middleware execution. Returns all versions (no deduplication).
Protocol handlers deduplicate for MCP wire format.

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L578"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, version: VersionSpec | None = None) -> Tool | None
```

Get a tool by name, filtering disabled tools.

Overrides Provider.get\_tool() to add visibility filtering after all
transforms (including session-level) have been applied. This ensures
session transforms can override provider-level disables.

**Args:**

* `name`: The tool name.
* `version`: Version filter (None returns highest version).

**Returns:**

* The tool if found and enabled, None otherwise.

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L604"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self) -> Sequence[Resource]
```

List all enabled resources from providers.

Overrides Provider.list\_resources() to add visibility filtering, auth filtering,
and middleware execution. Returns all versions (no deduplication).
Protocol handlers deduplicate for MCP wire format.

#### `get_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L676"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource(self, uri: str, version: VersionSpec | None = None) -> Resource | None
```

Get a resource by URI, filtering disabled resources.

Overrides Provider.get\_resource() to add visibility filtering after all
transforms (including session-level) have been applied.

**Args:**

* `uri`: The resource URI.
* `version`: Version filter (None returns highest version).

**Returns:**

* The resource if found and enabled, None otherwise.

#### `list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L701"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates(self) -> Sequence[ResourceTemplate]
```

List all enabled resource templates from providers.

Overrides Provider.list\_resource\_templates() to add visibility filtering,
auth filtering, and middleware execution. Returns all versions (no deduplication).
Protocol handlers deduplicate for MCP wire format.

#### `get_resource_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L775"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_template(self, uri: str, version: VersionSpec | None = None) -> ResourceTemplate | None
```

Get a resource template by URI, filtering disabled templates.

Overrides Provider.get\_resource\_template() to add visibility filtering after
all transforms (including session-level) have been applied.

**Args:**

* `uri`: The template URI.
* `version`: Version filter (None returns highest version).

**Returns:**

* The template if found and enabled, None otherwise.

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L800"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self) -> Sequence[Prompt]
```

List all enabled prompts from providers.

Overrides Provider.list\_prompts() to add visibility filtering, auth filtering,
and middleware execution. Returns all versions (no deduplication).
Protocol handlers deduplicate for MCP wire format.

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L870"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self, name: str, version: VersionSpec | None = None) -> Prompt | None
```

Get a prompt by name, filtering disabled prompts.

Overrides Provider.get\_prompt() to add visibility filtering after all
transforms (including session-level) have been applied.

**Args:**

* `name`: The prompt name.
* `version`: Version filter (None returns highest version).

**Returns:**

* The prompt if found and enabled, None otherwise.

#### `call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L896"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult
```

#### `call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L907"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> mcp.types.CreateTaskResult
```

#### `call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L917"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult | mcp.types.CreateTaskResult
```

Call a tool by name.

This is the public API for executing tools. By default, middleware is applied.

**Args:**

* `name`: The tool name
* `arguments`: Tool arguments (optional)
* `version`: Specific version to call. If None, calls highest version.
* `run_middleware`: If True (default), apply the middleware chain.
  Set to False when called from middleware to avoid re-applying.
* `task_meta`: If provided, execute as a background task and return
  CreateTaskResult. If None (default), execute synchronously and
  return ToolResult.

**Returns:**

* ToolResult when task\_meta is None.
* CreateTaskResult when task\_meta is provided.

**Raises:**

* `NotFoundError`: If tool not found or disabled
* `ToolError`: If tool execution fails
* `ValidationError`: If arguments fail validation

#### `read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1013"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource(self, uri: str) -> ResourceResult
```

#### `read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1023"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource(self, uri: str) -> mcp.types.CreateTaskResult
```

#### `read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1032"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource(self, uri: str) -> ResourceResult | mcp.types.CreateTaskResult
```

Read a resource by URI.

This is the public API for reading resources. By default, middleware is applied.
Checks concrete resources first, then templates.

**Args:**

* `uri`: The resource URI
* `version`: Specific version to read. If None, reads highest version.
* `run_middleware`: If True (default), apply the middleware chain.
  Set to False when called from middleware to avoid re-applying.
* `task_meta`: If provided, execute as a background task and return
  CreateTaskResult. If None (default), execute synchronously and
  return ResourceResult.

**Returns:**

* ResourceResult when task\_meta is None.
* CreateTaskResult when task\_meta is provided.

**Raises:**

* `NotFoundError`: If resource not found or disabled
* `ResourceError`: If resource read fails

#### `render_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1166"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
render_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> PromptResult
```

#### `render_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1177"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
render_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> mcp.types.CreateTaskResult
```

#### `render_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1187"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
render_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> PromptResult | mcp.types.CreateTaskResult
```

Render a prompt by name.

This is the public API for rendering prompts. By default, middleware is applied.
Use get\_prompt() to retrieve the prompt definition without rendering.

**Args:**

* `name`: The prompt name
* `arguments`: Prompt arguments (optional)
* `version`: Specific version to render. If None, renders highest version.
* `run_middleware`: If True (default), apply the middleware chain.
  Set to False when called from middleware to avoid re-applying.
* `task_meta`: If provided, execute as a background task and return
  CreateTaskResult. If None (default), execute synchronously and
  return PromptResult.

**Returns:**

* PromptResult when task\_meta is None.
* CreateTaskResult when task\_meta is provided.

**Raises:**

* `NotFoundError`: If prompt not found or disabled
* `PromptError`: If prompt rendering fails

#### `add_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1263"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_tool(self, tool: Tool | Callable[..., Any]) -> Tool
```

Add a tool to the server.

The tool function can optionally request a Context object by adding a parameter
with the Context type annotation. See the @tool decorator for examples.

**Args:**

* `tool`: The Tool instance or @tool-decorated function to register

**Returns:**

* The tool instance that was added to the server.

#### `remove_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1277"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
remove_tool(self, name: str, version: str | None = None) -> None
```

Remove tool(s) from the server.

.. deprecated::
Use `mcp.local_provider.remove_tool(name)` instead.

**Args:**

* `name`: The name of the tool to remove.
* `version`: If None, removes ALL versions. If specified, removes only that version.

**Raises:**

* `NotFoundError`: If no matching tool is found.

#### `tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1307"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool(self, name_or_fn: F) -> F
```

#### `tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1328"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool(self, name_or_fn: str | None = None) -> Callable[[F], F]
```

#### `tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1348"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool(self, name_or_fn: str | AnyFunction | None = None) -> Callable[[AnyFunction], FunctionTool] | FunctionTool | partial[Callable[[AnyFunction], FunctionTool] | FunctionTool]
```

Decorator to register a tool.

Tools can optionally request a Context object by adding a parameter with the
Context type annotation. The context provides access to MCP capabilities like
logging, progress reporting, and resource access.

This decorator supports multiple calling patterns:

* @server.tool (without parentheses)
* @server.tool (with empty parentheses)
* @server.tool("custom\_name") (with name as first argument)
* @server.tool(name="custom\_name") (with name as keyword argument)
* server.tool(function, name="custom\_name") (direct function call)

**Args:**

* `name_or_fn`: Either a function (when used as @tool), a string name, or None
* `name`: Optional name for the tool (keyword-only, alternative to name\_or\_fn)
* `description`: Optional description of what the tool does
* `tags`: Optional set of tags for categorizing the tool
* `output_schema`: Optional JSON schema for the tool's output
* `annotations`: Optional annotations about the tool's behavior
* `exclude_args`: Optional list of argument names to exclude from the tool schema.
  Deprecated: Use `Depends()` for dependency injection instead.
* `meta`: Optional meta information about the tool

**Examples:**

Register a tool with a custom name:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@server.tool
def my_tool(x: int) -> str:
    return str(x)

# Register a tool with a custom name
@server.tool
def my_tool(x: int) -> str:
    return str(x)

@server.tool("custom_name")
def my_tool(x: int) -> str:
    return str(x)

@server.tool(name="custom_name")
def my_tool(x: int) -> str:
    return str(x)

# Direct function call
server.tool(my_function, name="custom_name")
```

#### `add_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1447"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_resource(self, resource: Resource | Callable[..., Any]) -> Resource | ResourceTemplate
```

Add a resource to the server.

**Args:**

* `resource`: A Resource instance or @resource-decorated function to add

**Returns:**

* The resource instance that was added to the server.

#### `add_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1460"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_template(self, template: ResourceTemplate) -> ResourceTemplate
```

Add a resource template to the server.

**Args:**

* `template`: A ResourceTemplate instance to add

**Returns:**

* The template instance that was added to the server.

#### `resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1471"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resource(self, uri: str) -> Callable[[F], F]
```

Decorator to register a function as a resource.

The function will be called when the resource is read to generate its content.
The function can return:

* str for text content
* bytes for binary content
* other types will be converted to JSON

Resources can optionally request a Context object by adding a parameter with the
Context type annotation. The context provides access to MCP capabilities like
logging, progress reporting, and session information.

If the URI contains parameters (e.g. "resource://") or the function
has parameters, it will be registered as a template resource.

**Args:**

* `uri`: URI for the resource (e.g. "resource://my-resource" or "resource://")
* `name`: Optional name for the resource
* `description`: Optional description of the resource
* `mime_type`: Optional MIME type for the resource
* `tags`: Optional set of tags for categorizing the resource
* `annotations`: Optional annotations about the resource's behavior
* `meta`: Optional meta information about the resource

**Examples:**

Register a resource with a custom name:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@server.resource("resource://my-resource")
def get_data() -> str:
    return "Hello, world!"

@server.resource("resource://my-resource")
async get_data() -> str:
    data = await fetch_data()
    return f"Hello, world! {data}"

@server.resource("resource://{city}/weather")
def get_weather(city: str) -> str:
    return f"Weather for {city}"

@server.resource("resource://{city}/weather")
async def get_weather_with_context(city: str, ctx: Context) -> str:
    await ctx.info(f"Fetching weather for {city}")
    return f"Weather for {city}"

@server.resource("resource://{city}/weather")
async def get_weather(city: str) -> str:
    data = await fetch_weather(city)
    return f"Weather for {city}: {data}"
```

#### `add_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1590"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_prompt(self, prompt: Prompt | Callable[..., Any]) -> Prompt
```

Add a prompt to the server.

**Args:**

* `prompt`: A Prompt instance or @prompt-decorated function to add

**Returns:**

* The prompt instance that was added to the server.

#### `prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1602"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prompt(self, name_or_fn: F) -> F
```

#### `prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1618"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prompt(self, name_or_fn: str | None = None) -> Callable[[F], F]
```

#### `prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1633"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prompt(self, name_or_fn: str | AnyFunction | None = None) -> Callable[[AnyFunction], FunctionPrompt] | FunctionPrompt | partial[Callable[[AnyFunction], FunctionPrompt] | FunctionPrompt]
```

Decorator to register a prompt.

Prompts can optionally request a Context object by adding a parameter with the
Context type annotation. The context provides access to MCP capabilities like
logging, progress reporting, and session information.

This decorator supports multiple calling patterns:

* @server.prompt (without parentheses)
* @server.prompt() (with empty parentheses)
* @server.prompt("custom\_name") (with name as first argument)
* @server.prompt(name="custom\_name") (with name as keyword argument)
* server.prompt(function, name="custom\_name") (direct function call)

Args:
name\_or\_fn: Either a function (when used as @prompt), a string name, or None
name: Optional name for the prompt (keyword-only, alternative to name\_or\_fn)
description: Optional description of what the prompt does
tags: Optional set of tags for categorizing the prompt
meta: Optional meta information about the prompt

Examples:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@server.prompt
def analyze_table(table_name: str) -> list[Message]:
    schema = read_table_schema(table_name)
    return [
        {
            "role": "user",
            "content": f"Analyze this schema:
{schema}"
        }
    ]

@server.prompt()
async def analyze_with_context(table_name: str, ctx: Context) -> list[Message]:
    await ctx.info(f"Analyzing table {table_name}")
    schema = read_table_schema(table_name)
    return [
        {
            "role": "user",
            "content": f"Analyze this schema:
{schema}"
        }
    ]

@server.prompt("custom_name")
async def analyze_file(path: str) -> list[Message]:
    content = await read_file(path)
    return [
        {
            "role": "user",
            "content": {
                "type": "resource",
                "resource": {
                    "uri": f"file://{path}",
                    "text": content
                }
            }
        }
    ]

@server.prompt(name="custom_name")
def another_prompt(data: str) -> list[Message]:
    return [{"role": "user", "content": data}]

# Direct function call
server.prompt(my_function, name="custom_name")
```

#### `mount` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1733"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mount(self, server: FastMCP[LifespanResultT], namespace: str | None = None, as_proxy: bool | None = None, tool_names: dict[str, str] | None = None, prefix: str | None = None) -> None
```

Mount another FastMCP server on this server with an optional namespace.

Unlike importing (with import\_server), mounting establishes a dynamic connection
between servers. When a client interacts with a mounted server's objects through
the parent server, requests are forwarded to the mounted server in real-time.
This means changes to the mounted server are immediately reflected when accessed
through the parent.

When a server is mounted with a namespace:

* Tools from the mounted server are accessible with namespaced names.
  Example: If server has a tool named "get\_weather", it will be available as "namespace\_get\_weather".
* Resources are accessible with namespaced URIs.
  Example: If server has a resource with URI "weather://forecast", it will be available as
  "weather://namespace/forecast".
* Templates are accessible with namespaced URI templates.
  Example: If server has a template with URI "weather://location/", it will be available
  as "weather://namespace/location/".
* Prompts are accessible with namespaced names.
  Example: If server has a prompt named "weather\_prompt", it will be available as
  "namespace\_weather\_prompt".

When a server is mounted without a namespace (namespace=None), its tools, resources, templates,
and prompts are accessible with their original names. Multiple servers can be mounted
without namespaces, and they will be tried in order until a match is found.

The mounted server's lifespan is executed when the parent server starts, and its
middleware chain is invoked for all operations (tool calls, resource reads, prompts).

**Args:**

* `server`: The FastMCP server to mount.
* `namespace`: Optional namespace to use for the mounted server's objects. If None,
  the server's objects are accessible with their original names.
* `as_proxy`: Deprecated. Mounted servers now always have their lifespan and
  middleware invoked. To create a proxy server, use create\_proxy()
  explicitly before mounting.
* `tool_names`: Optional mapping of original tool names to custom names. Use this
  to override namespaced names. Keys are the original tool names from the
  mounted server.
* `prefix`: Deprecated. Use namespace instead.

#### `import_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1827"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import_server(self, server: FastMCP[LifespanResultT], prefix: str | None = None) -> None
```

Import the MCP objects from another FastMCP server into this one,
optionally with a given prefix.

.. deprecated::
Use :meth:`mount` instead. `import_server` will be removed in a
future version.

Note that when a server is *imported*, its objects are immediately
registered to the importing server. This is a one-time operation and
future changes to the imported server will not be reflected in the
importing server. Server-level configurations and lifespans are not imported.

When a server is imported with a prefix:

* The tools are imported with prefixed names
  Example: If server has a tool named "get\_weather", it will be
  available as "prefix\_get\_weather"
* The resources are imported with prefixed URIs using the new format
  Example: If server has a resource with URI "weather://forecast", it will
  be available as "weather://prefix/forecast"
* The templates are imported with prefixed URI templates using the new format
  Example: If server has a template with URI "weather://location/", it will
  be available as "weather://prefix/location/"
* The prompts are imported with prefixed names
  Example: If server has a prompt named "weather\_prompt", it will be available as
  "prefix\_weather\_prompt"

When a server is imported without a prefix (prefix=None), its tools, resources,
templates, and prompts are imported with their original names.

**Args:**

* `server`: The FastMCP server to import
* `prefix`: Optional prefix to use for the imported server's objects. If None,
  objects are imported with their original names.

#### `from_openapi` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1927"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_openapi(cls, openapi_spec: dict[str, Any], client: httpx.AsyncClient | None = None, name: str = 'OpenAPI Server', route_maps: list[RouteMap] | None = None, route_map_fn: OpenAPIRouteMapFn | None = None, mcp_component_fn: OpenAPIComponentFn | None = None, mcp_names: dict[str, str] | None = None, tags: set[str] | None = None, validate_output: bool = True, **settings: Any) -> Self
```

Create a FastMCP server from an OpenAPI specification.

**Args:**

* `openapi_spec`: OpenAPI schema as a dictionary
* `client`: Optional httpx AsyncClient for making HTTP requests.
  If not provided, a default client is created using the first
  server URL from the OpenAPI spec with a 30-second timeout.
* `name`: Name for the MCP server
* `route_maps`: Optional list of RouteMap objects defining route mappings
* `route_map_fn`: Optional callable for advanced route type mapping
* `mcp_component_fn`: Optional callable for component customization
* `mcp_names`: Optional dictionary mapping operationId to component names
* `tags`: Optional set of tags to add to all components
* `validate_output`: If True (default), tools use the output schema
  extracted from the OpenAPI spec for response validation. If
  False, a permissive schema is used instead, allowing any
  response structure while still returning structured JSON.
* `**settings`: Additional settings passed to FastMCP

**Returns:**

* A FastMCP server with an OpenAPIProvider attached.

#### `from_fastapi` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L1978"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_fastapi(cls, app: Any, name: str | None = None, route_maps: list[RouteMap] | None = None, route_map_fn: OpenAPIRouteMapFn | None = None, mcp_component_fn: OpenAPIComponentFn | None = None, mcp_names: dict[str, str] | None = None, httpx_client_kwargs: dict[str, Any] | None = None, tags: set[str] | None = None, **settings: Any) -> Self
```

Create a FastMCP server from a FastAPI application.

**Args:**

* `app`: FastAPI application instance
* `name`: Name for the MCP server (defaults to app.title)
* `route_maps`: Optional list of RouteMap objects defining route mappings
* `route_map_fn`: Optional callable for advanced route type mapping
* `mcp_component_fn`: Optional callable for component customization
* `mcp_names`: Optional dictionary mapping operationId to component names
* `httpx_client_kwargs`: Optional kwargs passed to httpx.AsyncClient.
  Use this to configure timeout and other client settings.
* `tags`: Optional set of tags to add to all components
* `**settings`: Additional settings passed to FastMCP

**Returns:**

* A FastMCP server with an OpenAPIProvider attached.

#### `as_proxy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L2033"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
as_proxy(cls, backend: Client[ClientTransportT] | ClientTransport | FastMCP[Any] | FastMCP1Server | AnyUrl | Path | MCPConfig | dict[str, Any] | str, **settings: Any) -> FastMCPProxy
```

Create a FastMCP proxy server for the given backend.

.. deprecated::
Use :func:`fastmcp.server.create_proxy` instead.
This method will be removed in a future version.

The `backend` argument can be either an existing `fastmcp.client.Client`
instance or any value accepted as the `transport` argument of
`fastmcp.client.Client`. This mirrors the convenience of the
`fastmcp.client.Client` constructor.

#### `generate_name` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/server.py#L2070"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_name(cls, name: str | None = None) -> str
```


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-__init__



# `fastmcp.server.tasks`

MCP SEP-1686 background tasks support.

This module implements protocol-level background task execution for MCP servers.


# capabilities
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-capabilities



# `fastmcp.server.tasks.capabilities`

SEP-1686 task capabilities declaration.

## Functions

### `get_task_capabilities` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/capabilities.py#L20"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_task_capabilities() -> ServerTasksCapability | None
```

Return the SEP-1686 task capabilities.

Returns task capabilities as a first-class ServerCapabilities field,
declaring support for list, cancel, and request operations per SEP-1686.

Returns None if pydocket is not installed (no task support).

Note: prompts/resources are passed via extra\_data since the SDK types
don't include them yet (FastMCP supports them ahead of the spec).


# config
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-config



# `fastmcp.server.tasks.config`

TaskConfig for MCP SEP-1686 background task execution modes.

This module defines the configuration for how tools, resources, and prompts
handle task-augmented execution as specified in SEP-1686.

## Classes

### `TaskMeta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/config.py#L25"><Icon icon="github" /></a></sup>

Metadata for task-augmented execution requests.

When passed to call\_tool/read\_resource/get\_prompt, signals that
the operation should be submitted as a background task.

**Attributes:**

* `ttl`: Client-requested TTL in milliseconds. If None, uses server default.
* `fn_key`: Docket routing key. Auto-derived from component name if None.

### `TaskConfig` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/config.py#L41"><Icon icon="github" /></a></sup>

Configuration for MCP background task execution (SEP-1686).

Controls how a component handles task-augmented requests:

* "forbidden": Component does not support task execution. Clients must not
  request task augmentation; server returns -32601 if they do.
* "optional": Component supports both synchronous and task execution.
  Client may request task augmentation or call normally.
* "required": Component requires task execution. Clients must request task
  augmentation; server returns -32601 if they don't.

**Methods:**

#### `from_bool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/config.py#L79"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_bool(cls, value: bool) -> TaskConfig
```

Convert boolean task flag to TaskConfig.

**Args:**

* `value`: True for "optional" mode, False for "forbidden" mode.

**Returns:**

* TaskConfig with appropriate mode.

#### `supports_tasks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/config.py#L90"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
supports_tasks(self) -> bool
```

Check if this component supports task execution.

**Returns:**

* True if mode is "optional" or "required", False if "forbidden".

#### `validate_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/config.py#L98"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_function(self, fn: Callable[..., Any], name: str) -> None
```

Validate that function is compatible with this task config.

Task execution requires:

1. fastmcp\[tasks] to be installed (pydocket)
2. Async functions

Raises ImportError if mode is "optional" or "required" but pydocket
is not installed. Raises ValueError if function is synchronous.

**Args:**

* `fn`: The function to validate (handles callable classes and staticmethods).
* `name`: Name for error messages.

**Raises:**

* `ImportError`: If task execution is enabled but pydocket not installed.
* `ValueError`: If task execution is enabled but function is sync.


# elicitation
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-elicitation



# `fastmcp.server.tasks.elicitation`

Background task elicitation support (SEP-1686).

This module provides elicitation capabilities for background tasks running
in Docket workers. Unlike regular MCP requests, background tasks don't have
an active request context, so elicitation requires special handling:

1. Set task status to "input\_required" via Redis
2. Send notifications/tasks/status with elicitation metadata
3. Wait for client to send input via tasks/sendInput
4. Resume task execution with the provided input

This uses the public MCP SDK APIs where possible, with minimal use of
internal APIs for background task coordination.

## Functions

### `elicit_for_task` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/elicitation.py#L42"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
elicit_for_task(task_id: str, session: ServerSession | None, message: str, schema: dict[str, Any], fastmcp: FastMCP) -> mcp.types.ElicitResult
```

Send an elicitation request from a background task.

This function handles the complexity of eliciting user input when running
in a Docket worker context where there's no active MCP request.

**Args:**

* `task_id`: The background task ID
* `session`: The MCP ServerSession for this task
* `message`: The message to display to the user
* `schema`: The JSON schema for the expected response
* `fastmcp`: The FastMCP server instance

**Returns:**

* ElicitResult containing the user's response

**Raises:**

* `RuntimeError`: If Docket is not available
* `McpError`: If the elicitation request fails

### `relay_elicitation` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/elicitation.py#L234"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
relay_elicitation(session: ServerSession, session_id: str, task_id: str, elicitation: dict[str, Any], fastmcp: FastMCP) -> None
```

Relay elicitation from a background task worker to the client.

Called by the notification subscriber when it detects an input\_required
notification with elicitation metadata. Sends a standard elicitation/create
request to the client session, then uses handle\_task\_input() to push the
response to Redis so the blocked worker can resume.

**Args:**

* `session`: MCP ServerSession
* `session_id`: Session identifier
* `task_id`: Background task ID
* `elicitation`: Elicitation metadata (message, requestedSchema)
* `fastmcp`: FastMCP server instance

### `handle_task_input` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/elicitation.py#L290"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
handle_task_input(task_id: str, session_id: str, action: str, content: dict[str, Any] | None, fastmcp: FastMCP) -> bool
```

Handle input sent to a background task via tasks/sendInput.

This is called when a client sends input in response to an elicitation
request from a background task.

**Args:**

* `task_id`: The background task ID
* `session_id`: The MCP session ID
* `action`: The elicitation action ("accept", "decline", "cancel")
* `content`: The response content (for "accept" action)
* `fastmcp`: The FastMCP server instance

**Returns:**

* True if the input was successfully stored, False otherwise


# handlers
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-handlers



# `fastmcp.server.tasks.handlers`

SEP-1686 task execution handlers.

Handles queuing tool/prompt/resource executions to Docket as background tasks.

## Functions

### `submit_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/handlers.py#L34"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
submit_to_docket(task_type: Literal['tool', 'resource', 'template', 'prompt'], key: str, component: Tool | Resource | ResourceTemplate | Prompt, arguments: dict[str, Any] | None = None, task_meta: TaskMeta | None = None) -> mcp.types.CreateTaskResult
```

Submit any component to Docket for background execution (SEP-1686).

Unified handler for all component types. Called by component's internal
methods (\_run, \_read, \_render) when task metadata is present and mode allows.

Queues the component's method to Docket, stores raw return values,
and converts to MCP types on retrieval.

**Args:**

* `task_type`: Component type for task key construction
* `key`: The component key as seen by MCP layer (with namespace prefix)
* `component`: The component instance (Tool, Resource, ResourceTemplate, Prompt)
* `arguments`: Arguments/params (None for Resource which has no args)
* `task_meta`: Task execution metadata. If task\_meta.ttl is provided, it
  overrides the server default (docket.execution\_ttl).

**Returns:**

* Task stub with proper Task object


# keys
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-keys



# `fastmcp.server.tasks.keys`

Task key management for SEP-1686 background tasks.

Task keys encode security scoping and metadata in the Docket key format:
`{session_id}:{client_task_id}:{task_type}:{component_identifier}`

This format provides:

* Session-based security scoping (prevents cross-session access)
* Task type identification (tool/prompt/resource)
* Component identification (name or URI for result conversion)

## Functions

### `build_task_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/keys.py#L15"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
build_task_key(session_id: str, client_task_id: str, task_type: str, component_identifier: str) -> str
```

Build Docket task key with embedded metadata.

Format: `{session_id}:{client_task_id}:{task_type}:{component_identifier}`

The component\_identifier is URI-encoded to handle special characters (colons, slashes, etc.).

**Args:**

* `session_id`: Session ID for security scoping
* `client_task_id`: Client-provided task ID
* `task_type`: Type of task ("tool", "prompt", "resource")
* `component_identifier`: Tool name, prompt name, or resource URI

**Returns:**

* Encoded task key for Docket

**Examples:**

> > > build\_task\_key("session123", "task456", "tool", "my\_tool")
> > > 'session123:task456:tool:my\_tool'
> > > build\_task\_key("session123", "task456", "resource", "file://data.txt")
> > > 'session123:task456:resource:file%3A%2F%2Fdata.txt'

### `parse_task_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/keys.py#L47"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse_task_key(task_key: str) -> dict[str, str]
```

Parse Docket task key to extract metadata.

**Args:**

* `task_key`: Encoded task key from Docket

**Returns:**

* Dict with keys: session\_id, client\_task\_id, task\_type, component\_identifier

**Examples:**

> > > parse\_task\_key("session123:task456:tool:my\_tool")
> > > `{'session_id': 'session123', 'client_task_id': 'task456', 'task_type': 'tool', 'component_identifier': 'my_tool'}`
> > > parse\_task\_key("session123:task456:resource:file%3A%2F%2Fdata.txt")
> > > `{'session_id': 'session123', 'client_task_id': 'task456', 'task_type': 'resource', 'component_identifier': 'file://data.txt'}`

### `get_client_task_id_from_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/keys.py#L78"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_client_task_id_from_key(task_key: str) -> str
```

Extract just the client task ID from a task key.

**Args:**

* `task_key`: Full encoded task key

**Returns:**

* Client-provided task ID (second segment)


# notifications
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-notifications



# `fastmcp.server.tasks.notifications`

Distributed notification queue for background task events (SEP-1686).

Enables distributed Docket workers to send MCP notifications to clients
without holding session references. Workers push to a Redis queue,
the MCP server process subscribes and forwards to the client's session.

Pattern: Fire-and-forward with retry

* One queue per session\_id
* LPUSH/BRPOP for reliable ordered delivery
* Retry up to 3 times on delivery failure, then discard
* TTL-based expiration for stale messages

Note: Docket's execution.subscribe() handles task state/progress events via
Redis Pub/Sub. This module handles elicitation-specific notifications that
require reliable delivery (input\_required prompts, cancel signals).

## Functions

### `push_notification` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/notifications.py#L48"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
push_notification(session_id: str, notification: dict[str, Any], docket: Docket) -> None
```

Push notification to session's queue (called from Docket worker).

Used for elicitation-specific notifications (input\_required, cancel)
that need reliable delivery across distributed processes.

**Args:**

* `session_id`: Target session's identifier
* `notification`: MCP notification dict (method, params, \_meta)
* `docket`: Docket instance for Redis access

### `notification_subscriber_loop` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/notifications.py#L76"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
notification_subscriber_loop(session_id: str, session: ServerSession, docket: Docket, fastmcp: FastMCP) -> None
```

Subscribe to notification queue and forward to session.

Runs in the MCP server process. Bridges distributed workers to clients.

This loop:

1. Maintains a heartbeat (active subscriber marker for debugging)
2. Blocks on BRPOP waiting for notifications
3. Forwards notifications to the client's session
4. Retries failed deliveries, then discards (no dead-letter queue)

**Args:**

* `session_id`: Session identifier to subscribe to
* `session`: MCP ServerSession for sending notifications
* `docket`: Docket instance for Redis access
* `fastmcp`: FastMCP server instance (for elicitation relay)

### `ensure_subscriber_running` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/notifications.py#L238"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ensure_subscriber_running(session_id: str, session: ServerSession, docket: Docket, fastmcp: FastMCP) -> None
```

Start notification subscriber if not already running (idempotent).

Subscriber is created on first task submission and cleaned up on disconnect.
Safe to call multiple times for the same session.

**Args:**

* `session_id`: Session identifier
* `session`: MCP ServerSession
* `docket`: Docket instance
* `fastmcp`: FastMCP server instance (for elicitation relay)

### `stop_subscriber` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/notifications.py#L278"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
stop_subscriber(session_id: str) -> None
```

Stop notification subscriber for a session.

Called when session disconnects. Pending messages remain in queue
for delivery if client reconnects (with TTL expiration).

**Args:**

* `session_id`: Session identifier

### `get_subscriber_count` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/notifications.py#L298"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_subscriber_count() -> int
```

Get number of active subscribers (for monitoring).


# requests
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-requests



# `fastmcp.server.tasks.requests`

SEP-1686 task request handlers.

Handles MCP task protocol requests: tasks/get, tasks/result, tasks/list, tasks/cancel.
These handlers query and manage existing tasks (contrast with handlers.py which creates tasks).

This module requires fastmcp\[tasks] (pydocket). It is only imported when docket is available.

## Functions

### `tasks_get_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/requests.py#L137"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tasks_get_handler(server: FastMCP, params: dict[str, Any]) -> GetTaskResult
```

Handle MCP 'tasks/get' request (SEP-1686).

**Args:**

* `server`: FastMCP server instance
* `params`: Request params containing taskId

**Returns:**

* Task status response with spec-compliant fields

### `tasks_result_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/requests.py#L222"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tasks_result_handler(server: FastMCP, params: dict[str, Any]) -> Any
```

Handle MCP 'tasks/result' request (SEP-1686).

Converts raw task return values to MCP types based on task type.

**Args:**

* `server`: FastMCP server instance
* `params`: Request params containing taskId

**Returns:**

* MCP result (CallToolResult, GetPromptResult, or ReadResourceResult)

### `tasks_list_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/requests.py#L398"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tasks_list_handler(server: FastMCP, params: dict[str, Any]) -> ListTasksResult
```

Handle MCP 'tasks/list' request (SEP-1686).

Note: With client-side tracking, this returns minimal info.

**Args:**

* `server`: FastMCP server instance
* `params`: Request params (cursor, limit)

**Returns:**

* Response with tasks list and pagination

### `tasks_cancel_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/requests.py#L416"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tasks_cancel_handler(server: FastMCP, params: dict[str, Any]) -> CancelTaskResult
```

Handle MCP 'tasks/cancel' request (SEP-1686).

Cancels a running task, transitioning it to cancelled state.

**Args:**

* `server`: FastMCP server instance
* `params`: Request params containing taskId

**Returns:**

* Task status response showing cancelled state


# routing
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-routing



# `fastmcp.server.tasks.routing`

Task routing helper for MCP components.

Provides unified task mode enforcement and docket routing logic.

## Functions

### `check_background_task` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/routing.py#L26"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
check_background_task(component: Tool | Resource | ResourceTemplate | Prompt, task_type: TaskType, arguments: dict[str, Any] | None = None, task_meta: TaskMeta | None = None) -> mcp.types.CreateTaskResult | None
```

Check task mode and submit to background if requested.

**Args:**

* `component`: The MCP component
* `task_type`: Type of task ("tool", "resource", "template", "prompt")
* `arguments`: Arguments for tool/prompt/template execution
* `task_meta`: Task execution metadata. If provided, execute as background task.

**Returns:**

* CreateTaskResult if submitted to docket, None for sync execution

**Raises:**

* `McpError`: If mode="required" but no task metadata, or mode="forbidden"
  but task metadata is present


# subscriptions
Source: https://gofastmcp.com/python-sdk/fastmcp-server-tasks-subscriptions



# `fastmcp.server.tasks.subscriptions`

Task subscription helpers for sending MCP notifications (SEP-1686).

Subscribes to Docket execution state changes and sends notifications/tasks/status
to clients when their tasks change state.

This module requires fastmcp\[tasks] (pydocket). It is only imported when docket is available.

## Functions

### `subscribe_to_task_updates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/subscriptions.py#L31"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
subscribe_to_task_updates(task_id: str, task_key: str, session: ServerSession, docket: Docket, poll_interval_ms: int = 5000) -> None
```

Subscribe to Docket execution events and send MCP notifications.

Per SEP-1686 lines 436-444, servers MAY send notifications/tasks/status
when task state changes. This is an optional optimization that reduces
client polling frequency.

**Args:**

* `task_id`: Client-visible task ID (server-generated UUID)
* `task_key`: Internal Docket execution key (includes session, type, component)
* `session`: MCP ServerSession for sending notifications
* `docket`: Docket instance for subscribing to execution events
* `poll_interval_ms`: Poll interval in milliseconds to include in notifications


# telemetry
Source: https://gofastmcp.com/python-sdk/fastmcp-server-telemetry



# `fastmcp.server.telemetry`

Server-side telemetry helpers.

## Functions

### `get_auth_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/telemetry.py#L13"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_auth_span_attributes() -> dict[str, str]
```

Get auth attributes for the current request, if authenticated.

### `get_session_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/telemetry.py#L30"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_session_span_attributes() -> dict[str, str]
```

Get session attributes for the current request.

### `server_span` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/telemetry.py#L56"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
server_span(name: str, method: str, server_name: str, component_type: str, component_key: str, resource_uri: str | None = None) -> Generator[Span, None, None]
```

Create a SERVER span with standard MCP attributes and auth context.

Automatically records any exception on the span and sets error status.

### `delegate_span` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/telemetry.py#L100"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
delegate_span(name: str, provider_type: str, component_key: str) -> Generator[Span, None, None]
```

Create an INTERNAL span for provider delegation.

Used by FastMCPProvider when delegating to mounted servers.
Automatically records any exception on the span and sets error status.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-__init__



# `fastmcp.server.transforms`

Transform system for component transformations.

Transforms modify components (tools, resources, prompts). List operations use a pure
function pattern where transforms receive sequences and return transformed sequences.
Get operations use a middleware pattern with `call_next` to chain lookups.

Unlike middleware (which operates on requests), transforms are observable by the
system for task registration, tag filtering, and component introspection.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import Namespace

server = FastMCP("Server")
mount = server.mount(other_server)
mount.add_transform(Namespace("api"))  # Tools become api_toolname
```

## Classes

### `GetToolNext` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L36"><Icon icon="github" /></a></sup>

Protocol for get\_tool call\_next functions.

### `GetResourceNext` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L44"><Icon icon="github" /></a></sup>

Protocol for get\_resource call\_next functions.

### `GetResourceTemplateNext` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L52"><Icon icon="github" /></a></sup>

Protocol for get\_resource\_template call\_next functions.

### `GetPromptNext` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L60"><Icon icon="github" /></a></sup>

Protocol for get\_prompt call\_next functions.

### `Transform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L68"><Icon icon="github" /></a></sup>

Base class for component transformations.

List operations use a pure function pattern: transforms receive sequences
and return transformed sequences. Get operations use a middleware pattern
with `call_next` to chain lookups.

**Methods:**

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L95"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

List tools with transformation applied.

**Args:**

* `tools`: Sequence of tools to transform.

**Returns:**

* Transformed sequence of tools.

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L106"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, call_next: GetToolNext) -> Tool | None
```

Get a tool by name.

**Args:**

* `name`: The requested tool name (may be transformed).
* `call_next`: Callable to get tool from downstream.
* `version`: Optional version filter to apply.

**Returns:**

* The tool if found, None otherwise.

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L125"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self, resources: Sequence[Resource]) -> Sequence[Resource]
```

List resources with transformation applied.

**Args:**

* `resources`: Sequence of resources to transform.

**Returns:**

* Transformed sequence of resources.

#### `get_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L136"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource(self, uri: str, call_next: GetResourceNext) -> Resource | None
```

Get a resource by URI.

**Args:**

* `uri`: The requested resource URI (may be transformed).
* `call_next`: Callable to get resource from downstream.
* `version`: Optional version filter to apply.

**Returns:**

* The resource if found, None otherwise.

#### `list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L159"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates(self, templates: Sequence[ResourceTemplate]) -> Sequence[ResourceTemplate]
```

List resource templates with transformation applied.

**Args:**

* `templates`: Sequence of resource templates to transform.

**Returns:**

* Transformed sequence of resource templates.

#### `get_resource_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L172"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_template(self, uri: str, call_next: GetResourceTemplateNext) -> ResourceTemplate | None
```

Get a resource template by URI.

**Args:**

* `uri`: The requested template URI (may be transformed).
* `call_next`: Callable to get template from downstream.
* `version`: Optional version filter to apply.

**Returns:**

* The resource template if found, None otherwise.

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L195"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self, prompts: Sequence[Prompt]) -> Sequence[Prompt]
```

List prompts with transformation applied.

**Args:**

* `prompts`: Sequence of prompts to transform.

**Returns:**

* Transformed sequence of prompts.

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/__init__.py#L206"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self, name: str, call_next: GetPromptNext) -> Prompt | None
```

Get a prompt by name.

**Args:**

* `name`: The requested prompt name (may be transformed).
* `call_next`: Callable to get prompt from downstream.
* `version`: Optional version filter to apply.

**Returns:**

* The prompt if found, None otherwise.


# catalog
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-catalog



# `fastmcp.server.transforms.catalog`

Base class for transforms that need to read the real component catalog.

Some transforms replace `list_tools()` output with synthetic components
(e.g. a search interface) while still needing access to the *real*
(auth-filtered) catalog at call time.  `CatalogTransform` provides the
bypass machinery so subclasses can call `get_tool_catalog()` without
triggering their own replacement logic.

## Re-entrancy problem

When a synthetic tool handler calls `get_tool_catalog()`, that calls
`ctx.fastmcp.list_tools()` which re-enters the transform pipeline —
including *this* transform's `list_tools()`.  If the subclass overrides
`list_tools()` directly, the re-entrant call would hit the subclass's
replacement logic again (returning synthetic tools instead of the real
catalog).  A `super()` call can't prevent this because Python can't
short-circuit a method after `super()` returns.

Solution: `CatalogTransform` owns `list_tools()` and uses a
per-instance `ContextVar` to detect re-entrant calls.  During bypass,
it passes through to the base `Transform.list_tools()` (a no-op).
Otherwise, it delegates to `transform_tools()` — the subclass hook
where replacement logic lives.  Same pattern for resources, prompts,
and resource templates.

This is *not* the same as the `Provider._list_tools()` convention
(which produces raw components with no arguments).  `transform_tools()`
receives the current catalog and returns a transformed version.  The
distinct name avoids confusion between the two patterns.

Usage::

class MyTransform(CatalogTransform):
async def transform\_tools(self, tools):
return \[self.\_make\_search\_tool()]

def \_make\_search\_tool(self):
async def search(ctx: Context = None):
real\_tools = await self.get\_tool\_catalog(ctx)
...
return Tool.from\_function(fn=search, name="search")

## Classes

### `CatalogTransform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L64"><Icon icon="github" /></a></sup>

Transform that needs access to the real component catalog.

Subclasses override `transform_tools()` / `transform_resources()`
/ `transform_prompts()` / `transform_resource_templates()`
instead of the `list_*()` methods.  The base class owns
`list_*()` and handles re-entrant bypass automatically — subclasses
never see re-entrant calls from `get_*_catalog()`.

The `get_*_catalog()` methods fetch the real (auth-filtered) catalog
by temporarily setting a bypass flag so that this transform's
`list_*()` passes through without calling the subclass hook.

**Methods:**

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L88"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L93"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self, resources: Sequence[Resource]) -> Sequence[Resource]
```

#### `list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L98"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates(self, templates: Sequence[ResourceTemplate]) -> Sequence[ResourceTemplate]
```

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L105"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self, prompts: Sequence[Prompt]) -> Sequence[Prompt]
```

#### `transform_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L114"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

Transform the tool catalog.

Override this method to replace, filter, or augment the tool listing.
The default implementation passes through unchanged.

Do NOT override `list_tools()` directly — the base class uses it
to handle re-entrant bypass when `get_tool_catalog()` reads the
real catalog.

#### `transform_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L126"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_resources(self, resources: Sequence[Resource]) -> Sequence[Resource]
```

Transform the resource catalog.

Override this method to replace, filter, or augment the resource listing.
The default implementation passes through unchanged.

Do NOT override `list_resources()` directly — the base class uses it
to handle re-entrant bypass when `get_resource_catalog()` reads the
real catalog.

#### `transform_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L140"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_resource_templates(self, templates: Sequence[ResourceTemplate]) -> Sequence[ResourceTemplate]
```

Transform the resource template catalog.

Override this method to replace, filter, or augment the template listing.
The default implementation passes through unchanged.

Do NOT override `list_resource_templates()` directly — the base class
uses it to handle re-entrant bypass when
`get_resource_template_catalog()` reads the real catalog.

#### `transform_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L154"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_prompts(self, prompts: Sequence[Prompt]) -> Sequence[Prompt]
```

Transform the prompt catalog.

Override this method to replace, filter, or augment the prompt listing.
The default implementation passes through unchanged.

Do NOT override `list_prompts()` directly — the base class uses it
to handle re-entrant bypass when `get_prompt_catalog()` reads the
real catalog.

#### `get_tool_catalog` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L170"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool_catalog(self, ctx: Context) -> Sequence[Tool]
```

Fetch the real tool catalog, bypassing this transform.

**Args:**

* `ctx`: The current request context.
* `run_middleware`: Whether to run middleware on the inner call.
  Defaults to True because this is typically called from a
  tool handler where list\_tools middleware has not yet run.

#### `get_resource_catalog` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L187"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_catalog(self, ctx: Context) -> Sequence[Resource]
```

Fetch the real resource catalog, bypassing this transform.

**Args:**

* `ctx`: The current request context.
* `run_middleware`: Whether to run middleware on the inner call.
  Defaults to True because this is typically called from a
  tool handler where list\_resources middleware has not yet run.

#### `get_prompt_catalog` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L204"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt_catalog(self, ctx: Context) -> Sequence[Prompt]
```

Fetch the real prompt catalog, bypassing this transform.

**Args:**

* `ctx`: The current request context.
* `run_middleware`: Whether to run middleware on the inner call.
  Defaults to True because this is typically called from a
  tool handler where list\_prompts middleware has not yet run.

#### `get_resource_template_catalog` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L221"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_template_catalog(self, ctx: Context) -> Sequence[ResourceTemplate]
```

Fetch the real resource template catalog, bypassing this transform.

**Args:**

* `ctx`: The current request context.
* `run_middleware`: Whether to run middleware on the inner call.
  Defaults to True because this is typically called from a
  tool handler where list\_resource\_templates middleware has
  not yet run.


# namespace
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-namespace



# `fastmcp.server.transforms.namespace`

Namespace transform for prefixing component names.

## Classes

### `Namespace` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/namespace.py#L28"><Icon icon="github" /></a></sup>

Prefixes component names with a namespace.

* Tools: name → namespace\_name
* Prompts: name → namespace\_name
* Resources: protocol://path → protocol://namespace/path
* Resource Templates: same as resources

**Methods:**

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/namespace.py#L97"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

Prefix tool names with namespace.

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/namespace.py#L103"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, call_next: GetToolNext) -> Tool | None
```

Get tool by namespaced name.

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/namespace.py#L119"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self, resources: Sequence[Resource]) -> Sequence[Resource]
```

Add namespace path segment to resource URIs.

#### `get_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/namespace.py#L126"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource(self, uri: str, call_next: GetResourceNext) -> Resource | None
```

Get resource by namespaced URI.

#### `list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/namespace.py#L146"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates(self, templates: Sequence[ResourceTemplate]) -> Sequence[ResourceTemplate]
```

Add namespace path segment to template URIs.

#### `get_resource_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/namespace.py#L155"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_template(self, uri: str, call_next: GetResourceTemplateNext) -> ResourceTemplate | None
```

Get resource template by namespaced URI.

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/namespace.py#L177"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self, prompts: Sequence[Prompt]) -> Sequence[Prompt]
```

Prefix prompt names with namespace.

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/namespace.py#L183"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self, name: str, call_next: GetPromptNext) -> Prompt | None
```

Get prompt by namespaced name.


# prompts_as_tools
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-prompts_as_tools



# `fastmcp.server.transforms.prompts_as_tools`

Transform that exposes prompts as tools.

This transform generates tools for listing and getting prompts, enabling
clients that only support tools to access prompt functionality.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import PromptsAsTools

mcp = FastMCP("Server")
mcp.add_transform(PromptsAsTools(mcp))
# Now has list_prompts and get_prompt tools
```

## Classes

### `PromptsAsTools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/prompts_as_tools.py#L35"><Icon icon="github" /></a></sup>

Transform that adds tools for listing and getting prompts.

Generates two tools:

* `list_prompts`: Lists all prompts from the provider
* `get_prompt`: Gets a specific prompt with optional arguments

The transform captures a provider reference at construction and queries it
for prompts when the generated tools are called. When used with FastMCP,
the provider's auth and visibility filtering is automatically applied.

**Methods:**

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/prompts_as_tools.py#L66"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

Add prompt tools to the tool list.

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/prompts_as_tools.py#L74"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, call_next: GetToolNext) -> Tool | None
```

Get a tool by name, including generated prompt tools.


# resources_as_tools
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-resources_as_tools



# `fastmcp.server.transforms.resources_as_tools`

Transform that exposes resources as tools.

This transform generates tools for listing and reading resources, enabling
clients that only support tools to access resource functionality.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import ResourcesAsTools

mcp = FastMCP("Server")
mcp.add_transform(ResourcesAsTools(mcp))
# Now has list_resources and read_resource tools
```

## Classes

### `ResourcesAsTools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/resources_as_tools.py#L32"><Icon icon="github" /></a></sup>

Transform that adds tools for listing and reading resources.

Generates two tools:

* `list_resources`: Lists all resources and templates from the provider
* `read_resource`: Reads a resource by URI

The transform captures a provider reference at construction and queries it
for resources when the generated tools are called. When used with FastMCP,
the provider's auth and visibility filtering is automatically applied.

**Methods:**

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/resources_as_tools.py#L63"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

Add resource tools to the tool list.

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/resources_as_tools.py#L71"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, call_next: GetToolNext) -> Tool | None
```

Get a tool by name, including generated resource tools.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-search-__init__



# `fastmcp.server.transforms.search`

Search transforms for tool discovery.

Search transforms collapse a large tool catalog into a search interface,
letting LLMs discover tools on demand instead of seeing the full list.

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms.search import RegexSearchTransform

mcp = FastMCP("Server")
mcp.add_transform(RegexSearchTransform())
# list_tools now returns only search_tools + call_tool
```


# base
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-search-base



# `fastmcp.server.transforms.search.base`

Base class for search transforms.

Search transforms replace `list_tools()` output with a small set of
synthetic tools — a search tool and a call-tool proxy — so LLMs can
discover tools on demand instead of receiving the full catalog.

All concrete search transforms (`RegexSearchTransform`,
`BM25SearchTransform`, etc.) inherit from `BaseSearchTransform` and
implement `_make_search_tool()` and `_search()` to provide their
specific search strategy.

Example::

from fastmcp import FastMCP
from fastmcp.server.transforms.search import RegexSearchTransform

mcp = FastMCP("Server")

@mcp.tool
def add(a: int, b: int) -> int: ...

@mcp.tool
def multiply(x: float, y: float) -> float: ...

# Clients now see only `search_tools` and `call_tool`.

# The original tools are discoverable via search.

mcp.add\_transform(RegexSearchTransform())

## Functions

### `serialize_tools_for_output_json` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/search/base.py#L60"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
serialize_tools_for_output_json(tools: Sequence[Tool]) -> list[dict[str, Any]]
```

Serialize tools to the same dict format as `list_tools` output.

### `serialize_tools_for_output_markdown` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/search/base.py#L138"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
serialize_tools_for_output_markdown(tools: Sequence[Tool]) -> str
```

Serialize tools to compact markdown, using \~65-70% fewer tokens than JSON.

## Classes

### `BaseSearchTransform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/search/base.py#L154"><Icon icon="github" /></a></sup>

Replace the tool listing with a search interface.

When this transform is active, `list_tools()` returns only:

* Any tools listed in `always_visible` (pinned).
* A **search tool** that finds tools matching a query.
* A **call\_tool** proxy that executes tools discovered via search.

Hidden tools remain callable — `get_tool()` delegates unknown
names downstream, so direct calls and the call-tool proxy both work.

Search results respect the full auth pipeline: middleware, visibility
transforms, and component-level auth checks all apply.

**Args:**

* `max_results`: Maximum number of tools returned per search.
* `always_visible`: Tool names that stay in the `list_tools`
  output alongside the synthetic search/call tools.
* `search_tool_name`: Name of the generated search tool.
* `call_tool_name`: Name of the generated call-tool proxy.

**Methods:**

#### `transform_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/search/base.py#L199"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

Replace the catalog with pinned + synthetic search/call tools.

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/search/base.py#L204"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, call_next: GetToolNext) -> Tool | None
```

Intercept synthetic tool names; delegate everything else.


# bm25
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-search-bm25



# `fastmcp.server.transforms.search.bm25`

BM25-based search transform.

## Classes

### `BM25SearchTransform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/search/bm25.py#L86"><Icon icon="github" /></a></sup>

Search transform using BM25 Okapi relevance ranking.

Maintains an in-memory index that is lazily rebuilt when the tool
catalog changes (detected via a hash of tool names).


# regex
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-search-regex



# `fastmcp.server.transforms.search.regex`

Regex-based search transform.

## Classes

### `RegexSearchTransform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/search/regex.py#L15"><Icon icon="github" /></a></sup>

Search transform using regex pattern matching.

Tools are matched against their name, description, and parameter
information using `re.search` with `re.IGNORECASE`.


# tool_transform
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-tool_transform



# `fastmcp.server.transforms.tool_transform`

Transform for applying tool transformations.

## Classes

### `ToolTransform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/tool_transform.py#L16"><Icon icon="github" /></a></sup>

Applies tool transformations to modify tool schemas.

Wraps ToolTransformConfig to apply argument renames, schema changes,
hidden arguments, and other transformations at the transform level.

**Methods:**

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/tool_transform.py#L64"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

Apply transforms to matching tools.

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/tool_transform.py#L75"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, call_next: GetToolNext) -> Tool | None
```

Get tool by transformed name.


# version_filter
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-version_filter



# `fastmcp.server.transforms.version_filter`

Version filter transform for filtering components by version range.

## Classes

### `VersionFilter` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/version_filter.py#L24"><Icon icon="github" /></a></sup>

Filters components by version range.

When applied to a provider or server, components within the version range
are visible, and unversioned components are included by default. Within
that filtered set, the highest version of each component is exposed to
clients (standard deduplication behavior). Set
`include_unversioned=False` to exclude unversioned components.

Parameters mirror comparison operators for clarity:

# Versions \< 3.0 (v1 and v2)

server.add\_transform(VersionFilter(version\_lt="3.0"))

# Versions >= 2.0 and \< 3.0 (only v2.x)

server.add\_transform(VersionFilter(version\_gte="2.0", version\_lt="3.0"))

Works with any version string - PEP 440 (1.0, 2.0) or dates (2025-01-01).

**Args:**

* `version_gte`: Versions >= this value pass through.
* `version_lt`: Versions \< this value pass through.
* `include_unversioned`: Whether unversioned components (`version=None`)
  should pass through the filter. Defaults to True.

**Methods:**

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/version_filter.py#L80"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/version_filter.py#L87"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, call_next: GetToolNext) -> Tool | None
```

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/version_filter.py#L96"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self, resources: Sequence[Resource]) -> Sequence[Resource]
```

#### `get_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/version_filter.py#L103"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource(self, uri: str, call_next: GetResourceNext) -> Resource | None
```

#### `list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/version_filter.py#L116"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates(self, templates: Sequence[ResourceTemplate]) -> Sequence[ResourceTemplate]
```

#### `get_resource_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/version_filter.py#L125"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_template(self, uri: str, call_next: GetResourceTemplateNext) -> ResourceTemplate | None
```

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/version_filter.py#L138"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self, prompts: Sequence[Prompt]) -> Sequence[Prompt]
```

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/version_filter.py#L145"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self, name: str, call_next: GetPromptNext) -> Prompt | None
```


# visibility
Source: https://gofastmcp.com/python-sdk/fastmcp-server-transforms-visibility



# `fastmcp.server.transforms.visibility`

Visibility transform for marking component visibility state.

Each Visibility instance marks components via internal metadata. Multiple
visibility transforms can be stacked - later transforms override earlier ones.
Final filtering happens at the Provider level.

## Functions

### `is_enabled` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L271"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_enabled(component: FastMCPComponent) -> bool
```

Check if component is enabled.

Returns True if:

* No visibility mark exists (default is enabled)
* Visibility mark is True

Returns False if visibility mark is False.

**Args:**

* `component`: Component to check.

**Returns:**

* True if component should be enabled/visible to clients.

### `get_visibility_rules` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L300"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_visibility_rules(context: Context) -> list[dict[str, Any]]
```

Load visibility rule dicts from session state.

### `save_visibility_rules` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L305"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
save_visibility_rules(context: Context, rules: list[dict[str, Any]]) -> None
```

Save visibility rule dicts to session state and send notifications.

**Args:**

* `context`: The context to save rules for.
* `rules`: The visibility rules to save.
* `components`: Optional hint about which component types are affected.
  If None, sends notifications for all types (safe default).
  If provided, only sends notifications for specified types.

### `create_visibility_transforms` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L332"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_visibility_transforms(rules: list[dict[str, Any]]) -> list[Visibility]
```

Convert rule dicts to Visibility transforms.

### `get_session_transforms` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L360"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_session_transforms(context: Context) -> list[Visibility]
```

Get session-specific Visibility transforms from state store.

### `enable_components` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L372"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
enable_components(context: Context) -> None
```

Enable components matching criteria for this session only.

Session rules override global transforms. Rules accumulate - each call
adds a new rule to the session. Later marks override earlier ones
(Visibility transform semantics).

Sends notifications to this session only: ToolListChangedNotification,
ResourceListChangedNotification, and PromptListChangedNotification.

**Args:**

* `context`: The context for this session.
* `names`: Component names or URIs to match.
* `keys`: Component keys to match (e.g., ).
* `version`: Component version spec to match.
* `tags`: Tags to match (component must have at least one).
* `components`: Component types to match (e.g., ).
* `match_all`: If True, matches all components regardless of other criteria.

### `disable_components` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L426"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
disable_components(context: Context) -> None
```

Disable components matching criteria for this session only.

Session rules override global transforms. Rules accumulate - each call
adds a new rule to the session. Later marks override earlier ones
(Visibility transform semantics).

Sends notifications to this session only: ToolListChangedNotification,
ResourceListChangedNotification, and PromptListChangedNotification.

**Args:**

* `context`: The context for this session.
* `names`: Component names or URIs to match.
* `keys`: Component keys to match (e.g., ).
* `version`: Component version spec to match.
* `tags`: Tags to match (component must have at least one).
* `components`: Component types to match (e.g., ).
* `match_all`: If True, matches all components regardless of other criteria.

### `reset_visibility` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L480"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
reset_visibility(context: Context) -> None
```

Clear all session visibility rules.

Use this to reset session visibility back to global defaults.

Sends notifications to this session only: ToolListChangedNotification,
ResourceListChangedNotification, and PromptListChangedNotification.

**Args:**

* `context`: The context for this session.

### `apply_session_transforms` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L497"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
apply_session_transforms(components: Sequence[ComponentT]) -> Sequence[ComponentT]
```

Apply session-specific visibility transforms to components.

This helper applies session-level enable/disable rules by marking
components with their visibility state. Session transforms override
global transforms due to mark-based semantics (later marks win).

**Args:**

* `components`: The components to apply session transforms to.

**Returns:**

* The components with session transforms applied.

## Classes

### `Visibility` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L39"><Icon icon="github" /></a></sup>

Sets visibility state on matching components.

Does NOT filter inline - just marks components with visibility state.
Later transforms in the chain can override earlier marks.
Final filtering happens at the Provider level after all transforms run.

**Methods:**

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L196"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

Mark tools by visibility state.

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L200"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, call_next: GetToolNext) -> Tool | None
```

Mark tool if found.

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L213"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self, resources: Sequence[Resource]) -> Sequence[Resource]
```

Mark resources by visibility state.

#### `get_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L217"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource(self, uri: str, call_next: GetResourceNext) -> Resource | None
```

Mark resource if found.

#### `list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L234"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates(self, templates: Sequence[ResourceTemplate]) -> Sequence[ResourceTemplate]
```

Mark resource templates by visibility state.

#### `get_resource_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L240"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_template(self, uri: str, call_next: GetResourceTemplateNext) -> ResourceTemplate | None
```

Mark resource template if found.

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L257"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self, prompts: Sequence[Prompt]) -> Sequence[Prompt]
```

Mark prompts by visibility state.

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/visibility.py#L261"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self, name: str, call_next: GetPromptNext) -> Prompt | None
```

Mark prompt if found.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-tools-__init__



# `fastmcp.tools`

*This module is empty or contains only private/internal implementations.*


# function_parsing
Source: https://gofastmcp.com/python-sdk/fastmcp-tools-function_parsing



# `fastmcp.tools.function_parsing`

Function introspection and schema generation for FastMCP tools.

## Classes

### `ParsedFunction` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_parsing.py#L82"><Icon icon="github" /></a></sup>

**Methods:**

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_parsing.py#L91"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any], exclude_args: list[str] | None = None, validate: bool = True, wrap_non_object_output_schema: bool = True) -> ParsedFunction
```


# function_tool
Source: https://gofastmcp.com/python-sdk/fastmcp-tools-function_tool



# `fastmcp.tools.function_tool`

Standalone @tool decorator for FastMCP.

## Functions

### `tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_tool.py#L375"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool(name_or_fn: str | Callable[..., Any] | None = None) -> Any
```

Standalone decorator to mark a function as an MCP tool.

Returns the original function with metadata attached. Register with a server
using mcp.add\_tool().

## Classes

### `DecoratedTool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_tool.py#L56"><Icon icon="github" /></a></sup>

Protocol for functions decorated with @tool.

### `ToolMeta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_tool.py#L65"><Icon icon="github" /></a></sup>

Metadata attached to functions by the @tool decorator.

### `FunctionTool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_tool.py#L87"><Icon icon="github" /></a></sup>

**Methods:**

#### `to_mcp_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_tool.py#L91"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_tool(self, **overrides: Any) -> mcp.types.Tool
```

Convert the FastMCP tool to an MCP tool.

Extends the base implementation to add task execution mode if enabled.

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_tool.py#L110"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any]) -> FunctionTool
```

Create a FunctionTool from a function.

**Args:**

* `fn`: The function to wrap
* `metadata`: ToolMeta object with all configuration. If provided,
  individual parameters must not be passed.
* `name, title, etc.`: Individual parameters for backwards compatibility.
  Cannot be used together with metadata parameter.

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_tool.py#L253"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, arguments: dict[str, Any]) -> ToolResult
```

Run the tool with arguments.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_tool.py#L298"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this tool with docket for background execution.

FunctionTool registers the underlying function, which has the user's
Depends parameters for docket to resolve.

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/function_tool.py#L308"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, arguments: dict[str, Any], **kwargs: Any) -> Execution
```

Schedule this tool for background execution via docket.

FunctionTool splats the arguments dict since .fn expects \*\*kwargs.

**Args:**

* `docket`: The Docket instance
* `arguments`: Tool arguments
* `fn_key`: Function lookup key in Docket registry (defaults to self.key)
* `task_key`: Redis storage key for the result
* `**kwargs`: Additional kwargs passed to docket.add()


# tool
Source: https://gofastmcp.com/python-sdk/fastmcp-tools-tool



# `fastmcp.tools.tool`

## Functions

### `default_serializer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L64"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_serializer(data: Any) -> str
```

## Classes

### `ToolResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L68"><Icon icon="github" /></a></sup>

**Methods:**

#### `to_mcp_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L121"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_result(self) -> list[ContentBlock] | tuple[list[ContentBlock], dict[str, Any]] | CallToolResult
```

### `Tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L137"><Icon icon="github" /></a></sup>

Internal tool registration info.

**Methods:**

#### `to_mcp_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L179"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_tool(self, **overrides: Any) -> MCPTool
```

Convert the FastMCP tool to an MCP tool.

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L206"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any]) -> FunctionTool
```

Create a Tool from a function.

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L246"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, arguments: dict[str, Any]) -> ToolResult
```

Run the tool with arguments.

This method is not implemented in the base Tool class and must be
implemented by subclasses.

`run()` can EITHER return a list of ContentBlocks, or a tuple of
(list of ContentBlocks, dict of structured output).

#### `convert_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L258"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
convert_result(self, raw_value: Any) -> ToolResult
```

Convert a raw result to ToolResult.

Handles ToolResult passthrough and converts raw values using the tool's
attributes (serializer, output\_schema) for proper conversion.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L356"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this tool with docket for background execution.

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L362"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, arguments: dict[str, Any], **kwargs: Any) -> Execution
```

Schedule this tool for background execution via docket.

**Args:**

* `docket`: The Docket instance
* `arguments`: Tool arguments
* `fn_key`: Function lookup key in Docket registry (defaults to self.key)
* `task_key`: Redis storage key for the result
* `**kwargs`: Additional kwargs passed to docket.add()

#### `from_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L386"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_tool(cls, tool: Tool | Callable[..., Any]) -> TransformedTool
```

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L434"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```


# tool_transform
Source: https://gofastmcp.com/python-sdk/fastmcp-tools-tool_transform



# `fastmcp.tools.tool_transform`

## Functions

### `forward` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L40"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
forward(**kwargs: Any) -> ToolResult
```

Forward to parent tool with argument transformation applied.

This function can only be called from within a transformed tool's custom
function. It applies argument transformation (renaming, validation) before
calling the parent tool.

For example, if the parent tool has args `x` and `y`, but the transformed
tool has args `a` and `b`, and an `transform_args` was provided that maps `x` to
`a` and `y` to `b`, then `forward(a=1, b=2)` will call the parent tool with
`x=1` and `y=2`.

**Args:**

* `**kwargs`: Arguments to forward to the parent tool (using transformed names).

**Returns:**

* The ToolResult from the parent tool execution.

**Raises:**

* `RuntimeError`: If called outside a transformed tool context.
* `TypeError`: If provided arguments don't match the transformed schema.

### `forward_raw` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L70"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
forward_raw(**kwargs: Any) -> ToolResult
```

Forward directly to parent tool without transformation.

This function bypasses all argument transformation and validation, calling the parent
tool directly with the provided arguments. Use this when you need to call the parent
with its original parameter names and structure.

For example, if the parent tool has args `x` and `y`, then `forward_raw(x=1,
y=2)` will call the parent tool with `x=1` and `y=2`.

**Args:**

* `**kwargs`: Arguments to pass directly to the parent tool (using original names).

**Returns:**

* The ToolResult from the parent tool execution.

**Raises:**

* `RuntimeError`: If called outside a transformed tool context.

### `apply_transformations_to_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L979"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
apply_transformations_to_tools(tools: dict[str, Tool], transformations: dict[str, ToolTransformConfig]) -> dict[str, Tool]
```

Apply a list of transformations to a list of tools. Tools that do not have any transformations
are left unchanged.

Note: tools dict is keyed by prefixed key (e.g., "tool:my\_tool"),
but transformations are keyed by tool name (e.g., "my\_tool").

## Classes

### `ArgTransform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L97"><Icon icon="github" /></a></sup>

Configuration for transforming a parent tool's argument.

This class allows fine-grained control over how individual arguments are transformed
when creating a new tool from an existing one. You can rename arguments, change their
descriptions, add default values, or hide them from clients while passing constants.

**Attributes:**

* `name`: New name for the argument. Use None to keep original name, or ... for no change.
* `description`: New description for the argument. Use None to remove description, or ... for no change.
* `default`: New default value for the argument. Use ... for no change.
* `default_factory`: Callable that returns a default value. Cannot be used with default.
* `type`: New type for the argument. Use ... for no change.
* `hide`: If True, hide this argument from clients but pass a constant value to parent.
* `required`: If True, make argument required (remove default). Use ... for no change.
* `examples`: Examples for the argument. Use ... for no change.

**Examples:**

Rename argument 'old\_name' to 'new\_name'

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(name="new_name")
```

Change description only

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(description="Updated description")
```

Add a default value (makes argument optional)

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(default=42)
```

Add a default factory (makes argument optional)

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(default_factory=lambda: time.time())
```

Change the type

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(type=str)
```

Hide the argument entirely from clients

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(hide=True)
```

Hide argument but pass a constant value to parent

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(hide=True, default="constant_value")
```

Hide argument but pass a factory-generated value to parent

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(hide=True, default_factory=lambda: uuid.uuid4().hex)
```

Make an optional parameter required (removes any default)

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(required=True)
```

Combine multiple transformations

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ArgTransform(name="new_name", description="New desc", default=None, type=int)
```

### `ArgTransformConfig` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L211"><Icon icon="github" /></a></sup>

A model for requesting a single argument transform.

**Methods:**

#### `to_arg_transform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L229"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_arg_transform(self) -> ArgTransform
```

Convert the argument transform to a FastMCP argument transform.

### `TransformedTool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L235"><Icon icon="github" /></a></sup>

A tool that is transformed from another tool.

This class represents a tool that has been created by transforming another tool.
It supports argument renaming, schema modification, custom function injection,
structured output control, and provides context for the forward() and forward\_raw() functions.

The transformation can be purely schema-based (argument renaming, dropping, etc.)
or can include a custom function that uses forward() to call the parent tool
with transformed arguments. Output schemas and structured outputs are automatically
inherited from the parent tool but can be overridden or disabled.

**Attributes:**

* `parent_tool`: The original tool that this tool was transformed from.
* `fn`: The function to execute when this tool is called (either the forwarding
  function for pure transformations or a custom user function).
* `forwarding_fn`: Internal function that handles argument transformation and
  validation when forward() is called from custom functions.

**Methods:**

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L264"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, arguments: dict[str, Any]) -> ToolResult
```

Run the tool with context set for forward() functions.

This method executes the tool's function while setting up the context
that allows forward() and forward\_raw() to work correctly within custom
functions.

**Args:**

* `arguments`: Dictionary of arguments to pass to the tool's function.

**Returns:**

* ToolResult object containing content and optional structured output.

#### `from_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L369"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_tool(cls, tool: Tool | Callable[..., Any], name: str | None = None, version: str | NotSetT | None = NotSet, title: str | NotSetT | None = NotSet, description: str | NotSetT | None = NotSet, tags: set[str] | None = None, transform_fn: Callable[..., Any] | None = None, transform_args: dict[str, ArgTransform] | None = None, annotations: ToolAnnotations | NotSetT | None = NotSet, output_schema: dict[str, Any] | NotSetT | None = NotSet, serializer: Callable[[Any], str] | NotSetT | None = NotSet, meta: dict[str, Any] | NotSetT | None = NotSet) -> TransformedTool
```

Create a transformed tool from a parent tool.

**Args:**

* `tool`: The parent tool to transform.
* `transform_fn`: Optional custom function. Can use forward() and forward\_raw()
  to call the parent tool. Functions with \*\*kwargs receive transformed
  argument names.
* `name`: New name for the tool. Defaults to parent tool's name.
* `version`: New version for the tool. Defaults to parent tool's version.
* `title`: New title for the tool. Defaults to parent tool's title.
* `transform_args`: Optional transformations for parent tool arguments.
  Only specified arguments are transformed, others pass through unchanged:
* Simple rename (str)
* Complex transformation (rename/description/default/drop) (ArgTransform)
* Drop the argument (None)
* `description`: New description. Defaults to parent's description.
* `tags`: New tags. Defaults to parent's tags.
* `annotations`: New annotations. Defaults to parent's annotations.
* `output_schema`: Control output schema for structured outputs:
* None (default): Inherit from transform\_fn if available, then parent tool
* dict: Use custom output schema
* False: Disable output schema and structured outputs
* `serializer`: Deprecated. Return ToolResult from your tools for full control over serialization.
* `meta`: Control meta information:
* NotSet (default): Inherit from parent tool
* dict: Use custom meta information
* None: Remove meta information

**Returns:**

* TransformedTool with the specified transformations.

**Examples:**

# Transform specific arguments only

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
Tool.from_tool(parent, transform_args={"old": "new"})  # Others unchanged
```

# Custom function with partial transforms

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def custom(x: int, y: int) -> str:
    result = await forward(x=x, y=y)
    return f"Custom: {result}"

Tool.from_tool(parent, transform_fn=custom, transform_args={"a": "x", "b": "y"})
```

# Using \*\*kwargs (gets all args, transformed and untransformed)

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def flexible(**kwargs) -> str:
    result = await forward(**kwargs)
    return f"Got: {kwargs}"

Tool.from_tool(parent, transform_fn=flexible, transform_args={"a": "x"})
```

# Control structured outputs and schemas

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Custom output schema
Tool.from_tool(parent, output_schema={
    "type": "object",
    "properties": {"status": {"type": "string"}}
})

# Disable structured outputs
Tool.from_tool(parent, output_schema=None)

# Return ToolResult for full control
async def custom_output(**kwargs) -> ToolResult:
    result = await forward(**kwargs)
    return ToolResult(
        content=[TextContent(text="Summary")],
        structured_content={"processed": True}
    )
```

### `ToolTransformConfig` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L925"><Icon icon="github" /></a></sup>

Provides a way to transform a tool.

**Methods:**

#### `apply` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool_transform.py#L958"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
apply(self, tool: Tool) -> TransformedTool
```

Create a TransformedTool from a provided tool and this transformation configuration.
