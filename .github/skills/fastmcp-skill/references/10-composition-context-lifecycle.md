# Composition, Context, and Lifecycle

Source lines: 14859-16616 from the original FastMCP documentation dump.

Composing servers, context access, dependency injection, icons, lifespans, and related runtime composition patterns.

---

# Composing Servers
Source: https://gofastmcp.com/servers/composition

Combine multiple servers into one

<VersionBadge />

As your application grows, you'll want to split it into focused servers — one for weather, one for calendar, one for admin — and combine them into a single server that clients connect to. That's what `mount()` does.

When you mount a server, all its tools, resources, and prompts become available through the parent. The connection is live: add a tool to the child after mounting, and it's immediately visible through the parent.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

weather = FastMCP("Weather")

@weather.tool
def get_forecast(city: str) -> str:
    """Get weather forecast for a city."""
    return f"Sunny in {city}"

@weather.resource("data://cities")
def list_cities() -> list[str]:
    """List supported cities."""
    return ["London", "Paris", "Tokyo"]

main = FastMCP("MainApp")
main.mount(weather)

# main now serves get_forecast and data://cities
```

## Mounting External Servers

Mount remote HTTP servers or subprocess-based MCP servers using `create_proxy()`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server import create_proxy

mcp = FastMCP("Orchestrator")

# Mount a remote HTTP server (URLs work directly)
mcp.mount(create_proxy("http://api.example.com/mcp"), namespace="api")

# Mount local Python scripts (file paths work directly)
mcp.mount(create_proxy("./my_server.py"), namespace="local")
```

### Mounting npm/uvx Packages

For npm packages or Python tools, use the config dict format:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server import create_proxy

mcp = FastMCP("Orchestrator")

# Mount npm package via config
github_config = {
    "mcpServers": {
        "default": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"]
        }
    }
}
mcp.mount(create_proxy(github_config), namespace="github")

# Mount Python tool via config
sqlite_config = {
    "mcpServers": {
        "default": {
            "command": "uvx",
            "args": ["mcp-server-sqlite", "--db", "data.db"]
        }
    }
}
mcp.mount(create_proxy(sqlite_config), namespace="db")
```

Or use explicit transport classes:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server import create_proxy
from fastmcp.client.transports import NpxStdioTransport, UvxStdioTransport

mcp = FastMCP("Orchestrator")

mcp.mount(
    create_proxy(NpxStdioTransport(package="@modelcontextprotocol/server-github")),
    namespace="github"
)
mcp.mount(
    create_proxy(UvxStdioTransport(tool_name="mcp-server-sqlite", tool_args=["--db", "data.db"])),
    namespace="db"
)
```

For advanced configuration, see [Proxying](/servers/providers/proxy).

## Namespacing

<VersionBadge />

When mounting multiple servers, use namespaces to avoid naming conflicts:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
weather = FastMCP("Weather")
calendar = FastMCP("Calendar")

@weather.tool
def get_data() -> str:
    return "Weather data"

@calendar.tool
def get_data() -> str:
    return "Calendar data"

main = FastMCP("Main")
main.mount(weather, namespace="weather")
main.mount(calendar, namespace="calendar")

# Tools are now:
# - weather_get_data
# - calendar_get_data
```

### How Namespacing Works

| Component Type | Without Namespace | With `namespace="api"` |
| -------------- | ----------------- | ---------------------- |
| Tool           | `my_tool`         | `api_my_tool`          |
| Prompt         | `my_prompt`       | `api_my_prompt`        |
| Resource       | `data://info`     | `data://api/info`      |
| Template       | `data://{id}`     | `data://api/{id}`      |

Namespacing uses [transforms](/servers/transforms/transforms) under the hood.

## Dynamic Composition

Because `mount()` creates a live link, you can add components to a child server after mounting and they'll be immediately available through the parent:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
main = FastMCP("Main")
main.mount(dynamic_server, namespace="dynamic")

# Add a tool AFTER mounting - it's accessible through main
@dynamic_server.tool
def added_later() -> str:
    return "Added after mounting!"
```

## Tag Filtering

<VersionBadge />

Parent server tag filters apply recursively to mounted servers:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
api_server = FastMCP("API")

@api_server.tool(tags={"production"})
def prod_endpoint() -> str:
    return "Production data"

@api_server.tool(tags={"development"})
def dev_endpoint() -> str:
    return "Debug data"

# Mount with production filter
prod_app = FastMCP("Production")
prod_app.mount(api_server, namespace="api")
prod_app.enable(tags={"production"}, only=True)

# Only prod_endpoint (namespaced as api_prod_endpoint) is visible
```

## Performance Considerations

Operations like `list_tools()` on the parent are affected by the performance of all mounted servers. This is particularly noticeable with:

* HTTP-based mounted servers (300-400ms vs 1-2ms for local tools)
* Mounted servers with slow initialization
* Deep mounting hierarchies

If low latency is critical, consider implementing caching strategies or limiting mounting depth.

## Custom Routes

<VersionBadge />

Custom HTTP routes defined with `@server.custom_route()` are also forwarded when mounting:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
subserver = FastMCP("Sub")

@subserver.custom_route("/health", methods=["GET"])
async def health_check():
    return {"status": "ok"}

main = FastMCP("Main")
main.mount(subserver, namespace="sub")

# /health is now accessible through main's HTTP app
```

## Conflict Resolution

<VersionBadge />

When mounting multiple servers with the same namespace (or no namespace), the **most recently mounted** server takes precedence for conflicting component names:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
server_a = FastMCP("A")
server_b = FastMCP("B")

@server_a.tool
def shared_tool() -> str:
    return "From A"

@server_b.tool
def shared_tool() -> str:
    return "From B"

main = FastMCP("Main")
main.mount(server_a)
main.mount(server_b)

# shared_tool returns "From B" (most recently mounted)
```


# MCP Context
Source: https://gofastmcp.com/servers/context

Access MCP capabilities like logging, progress, and resources within your MCP objects.

When defining FastMCP [tools](/servers/tools), [resources](/servers/resources), resource templates, or [prompts](/servers/prompts), your functions might need to interact with the underlying MCP session or access advanced server capabilities. FastMCP provides the `Context` object for this purpose.

<Note>
  You access Context through FastMCP's dependency injection system. For other injectable values like HTTP requests, access tokens, and custom dependencies, see [Dependency Injection](/servers/dependency-injection).
</Note>

## What Is Context?

The `Context` object provides a clean interface to access MCP features within your functions, including:

* **Logging**: Send debug, info, warning, and error messages back to the client
* **Progress Reporting**: Update the client on the progress of long-running operations
* **Resource Access**: List and read data from resources registered with the server
* **Prompt Access**: List and retrieve prompts registered with the server
* **LLM Sampling**: Request the client's LLM to generate text based on provided messages
* **User Elicitation**: Request structured input from users during tool execution
* **Session State**: Store data that persists across requests within an MCP session
* **Session Visibility**: [Control which components are visible](/servers/visibility#per-session-visibility) to the current session
* **Request Information**: Access metadata about the current request
* **Server Access**: When needed, access the underlying FastMCP server instance

## Accessing the Context

<VersionBadge />

The preferred way to access context is using the `CurrentContext()` dependency:

```python {1, 6} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context

mcp = FastMCP(name="Context Demo")

@mcp.tool
async def process_file(file_uri: str, ctx: Context = CurrentContext()) -> str:
    """Processes a file, using context for logging and resource access."""
    await ctx.info(f"Processing {file_uri}")
    return "Processed file"
```

This works with tools, resources, and prompts:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context

mcp = FastMCP(name="Context Demo")

@mcp.resource("resource://user-data")
async def get_user_data(ctx: Context = CurrentContext()) -> dict:
    await ctx.debug("Fetching user data")
    return {"user_id": "example"}

@mcp.prompt
async def data_analysis_request(dataset: str, ctx: Context = CurrentContext()) -> str:
    return f"Please analyze the following dataset: {dataset}"
```

**Key Points:**

* Dependency parameters are automatically excluded from the MCP schema—clients never see them.
* Context methods are async, so your function usually needs to be async as well.
* **Each MCP request receives a new context object.** Context is scoped to a single request; state or data set in one request will not be available in subsequent requests.
* Context is only available during a request; attempting to use context methods outside a request will raise errors.

### Legacy Type-Hint Injection

For backwards compatibility, you can still access context by simply adding a parameter with the `Context` type hint. FastMCP will automatically inject the context instance:

```python {1, 6} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context

mcp = FastMCP(name="Context Demo")

@mcp.tool
async def process_file(file_uri: str, ctx: Context) -> str:
    """Processes a file, using context for logging and resource access."""
    # Context is injected automatically based on the type hint
    return "Processed file"
```

This approach still works for tools, resources, and prompts. The parameter name doesn't matter—only the `Context` type hint is important. The type hint can also be a union (`Context | None`) or use `Annotated[]`.

### Via `get_context()` Function

<VersionBadge />

For code nested deeper within your function calls where passing context through parameters is inconvenient, use `get_context()` to retrieve the active context from anywhere within a request's execution flow:

```python {2,9} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context

mcp = FastMCP(name="Dependency Demo")

# Utility function that needs context but doesn't receive it as a parameter
async def process_data(data: list[float]) -> dict:
    # Get the active context - only works when called within a request
    ctx = get_context()
    await ctx.info(f"Processing {len(data)} data points")

@mcp.tool
async def analyze_dataset(dataset_name: str) -> dict:
    # Call utility function that uses context internally
    data = load_data(dataset_name)
    await process_data(data)
```

**Important Notes:**

* The `get_context()` function should only be used within the context of a server request. Calling it outside of a request will raise a `RuntimeError`.
* The `get_context()` function is server-only and should not be used in client code.

## Context Capabilities

FastMCP provides several advanced capabilities through the context object. Each capability has dedicated documentation with comprehensive examples and best practices:

### Logging

Send debug, info, warning, and error messages back to the MCP client for visibility into function execution.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
await ctx.debug("Starting analysis")
await ctx.info(f"Processing {len(data)} items") 
await ctx.warning("Deprecated parameter used")
await ctx.error("Processing failed")
```

See [Server Logging](/servers/logging) for complete documentation and examples.

### Client Elicitation

<VersionBadge />

Request structured input from clients during tool execution, enabling interactive workflows and progressive disclosure. This is a new feature in the 6/18/2025 MCP spec.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
result = await ctx.elicit("Enter your name:", response_type=str)
if result.action == "accept":
    name = result.data
```

See [User Elicitation](/servers/elicitation) for detailed examples and supported response types.

### LLM Sampling

<VersionBadge />

Request the client's LLM to generate text based on provided messages, useful for leveraging AI capabilities within your tools.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
response = await ctx.sample("Analyze this data", temperature=0.7)
```

See [LLM Sampling](/servers/sampling) for comprehensive usage and advanced techniques.

### Progress Reporting

Update clients on the progress of long-running operations, enabling progress indicators and better user experience.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
await ctx.report_progress(progress=50, total=100)  # 50% complete
```

See [Progress Reporting](/servers/progress) for detailed patterns and examples.

### Resource Access

List and read data from resources registered with your FastMCP server, allowing access to files, configuration, or dynamic content.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# List available resources
resources = await ctx.list_resources()

# Read a specific resource
content_list = await ctx.read_resource("resource://config")
content = content_list[0].content
```

**Method signatures:**

* **`ctx.list_resources() -> list[MCPResource]`**: <VersionBadge /> Returns list of all available resources
* **`ctx.read_resource(uri: str | AnyUrl) -> list[ReadResourceContents]`**: Returns a list of resource content parts

### Prompt Access

<VersionBadge />

List and retrieve prompts registered with your FastMCP server, allowing tools and middleware to discover and use available prompts programmatically.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# List available prompts
prompts = await ctx.list_prompts()

# Get a specific prompt with arguments
result = await ctx.get_prompt("analyze_data", {"dataset": "users"})
messages = result.messages
```

**Method signatures:**

* **`ctx.list_prompts() -> list[MCPPrompt]`**: Returns list of all available prompts
* **`ctx.get_prompt(name: str, arguments: dict[str, Any] | None = None) -> GetPromptResult`**: Get a specific prompt with optional arguments

### Session State

<VersionBadge />

Store data that persists across multiple requests within the same MCP session. Session state is automatically keyed by the client's session, ensuring isolation between different clients.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context

mcp = FastMCP("stateful-app")

@mcp.tool
async def increment_counter(ctx: Context) -> int:
    """Increment a counter that persists across tool calls."""
    count = await ctx.get_state("counter") or 0
    await ctx.set_state("counter", count + 1)
    return count + 1

@mcp.tool
async def get_counter(ctx: Context) -> int:
    """Get the current counter value."""
    return await ctx.get_state("counter") or 0
```

Each client session has its own isolated state—two different clients calling `increment_counter` will each have their own counter.

**Method signatures:**

* **`await ctx.set_state(key, value, *, serializable=True)`**: Store a value in session state
* **`await ctx.get_state(key)`**: Retrieve a value (returns None if not found)
* **`await ctx.delete_state(key)`**: Remove a value from session state

<Note>
  State methods are async and require `await`. State expires after 1 day to prevent unbounded memory growth.
</Note>

#### Non-Serializable Values

By default, state values must be JSON-serializable (dicts, lists, strings, numbers, etc.) so they can be persisted across requests. For non-serializable values like HTTP clients or database connections, pass `serializable=False`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
async def my_tool(ctx: Context) -> str:
    # This object can't be JSON-serialized
    client = SomeHTTPClient(base_url="https://api.example.com")
    await ctx.set_state("client", client, serializable=False)

    # Retrieve it later in the same request
    client = await ctx.get_state("client")
    return await client.fetch("/data")
```

Values stored with `serializable=False` only live for the current MCP request (a single tool call, resource read, or prompt render). They will not be available in subsequent requests within the session.

#### Custom Storage Backends

By default, session state uses an in-memory store suitable for single-server deployments. For distributed or serverless deployments, provide a custom storage backend:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from key_value.aio.stores.redis import RedisStore

# Use Redis for distributed state
mcp = FastMCP("distributed-app", session_state_store=RedisStore(...))
```

Any backend compatible with the [py-key-value-aio](https://github.com/strawgate/py-key-value) `AsyncKeyValue` protocol works. See [Storage Backends](/servers/storage-backends) for more options including Redis, DynamoDB, and MongoDB.

#### State During Initialization

State set during `on_initialize` middleware persists to subsequent tool calls when using the same session object (STDIO, SSE, single-server HTTP). For distributed/serverless HTTP deployments where different machines handle init and tool calls, state is isolated by the `mcp-session-id` header.

### Session Visibility

<VersionBadge />

Tools can customize which components are visible to their current session using `ctx.enable_components()`, `ctx.disable_components()`, and `ctx.reset_visibility()`. These methods apply visibility rules that affect only the calling session, leaving other sessions unchanged. See [Per-Session Visibility](/servers/visibility#per-session-visibility) for complete documentation, filter criteria, and patterns like namespace activation.

### Change Notifications

<VersionBadge />

FastMCP automatically sends list change notifications when components (such as tools, resources, or prompts) are added, removed, enabled, or disabled. In rare cases where you need to manually trigger these notifications, you can use the context's notification methods:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import mcp.types

@mcp.tool
async def custom_tool_management(ctx: Context) -> str:
    """Example of manual notification after custom tool changes."""
    await ctx.send_notification(mcp.types.ToolListChangedNotification())
    await ctx.send_notification(mcp.types.ResourceListChangedNotification())
    await ctx.send_notification(mcp.types.PromptListChangedNotification())
    return "Notifications sent"
```

These methods are primarily used internally by FastMCP's automatic notification system and most users will not need to invoke them directly.

### FastMCP Server

To access the underlying FastMCP server instance, you can use the `ctx.fastmcp` property:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
async def my_tool(ctx: Context) -> None:
    # Access the FastMCP server instance
    server_name = ctx.fastmcp.name
    ...
```

### Transport

<VersionBadge />

The `ctx.transport` property indicates which transport is being used to run the server. This is useful when your tool needs to behave differently depending on whether the server is running over STDIO, SSE, or Streamable HTTP. For example, you might want to return shorter responses over STDIO or adjust timeout behavior based on transport characteristics.

The transport type is set once when the server starts and remains constant for the server's lifetime. It returns `None` when called outside of a server context (for example, in unit tests or when running code outside of an MCP request).

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context

mcp = FastMCP("example")

@mcp.tool
def connection_info(ctx: Context) -> str:
    if ctx.transport == "stdio":
        return "Connected via STDIO"
    elif ctx.transport == "sse":
        return "Connected via SSE"
    elif ctx.transport == "streamable-http":
        return "Connected via Streamable HTTP"
    else:
        return "Transport unknown"
```

**Property signature:** `ctx.transport -> Literal["stdio", "sse", "streamable-http"] | None`

### MCP Request

Access metadata about the current request and client.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
async def request_info(ctx: Context) -> dict:
    """Return information about the current request."""
    return {
        "request_id": ctx.request_id,
        "client_id": ctx.client_id or "Unknown client"
    }
```

**Available Properties:**

* **`ctx.request_id -> str`**: Get the unique ID for the current MCP request
* **`ctx.client_id -> str | None`**: Get the ID of the client making the request, if provided during initialization
* **`ctx.session_id -> str`**: Get the MCP session ID for session-based data sharing. Raises `RuntimeError` if the MCP session is not yet established.

#### Request Context Availability

<VersionBadge />

The `ctx.request_context` property provides access to the underlying MCP request context, but returns `None` when the MCP session has not been established yet. This typically occurs:

* During middleware execution in the `on_request` hook before the MCP handshake completes
* During the initialization phase of client connections

The MCP request context is distinct from the HTTP request. For HTTP transports, HTTP request data may be available even when the MCP session is not yet established.

To safely access the request context in situations where it may not be available:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import get_http_request

mcp = FastMCP(name="Session Aware Demo")

@mcp.tool
async def session_info(ctx: Context) -> dict:
    """Return session information when available."""

    # Check if MCP session is available
    if ctx.request_context:
        # MCP session available - can access MCP-specific attributes
        return {
            "session_id": ctx.session_id,
            "request_id": ctx.request_id,
            "has_meta": ctx.request_context.meta is not None
        }
    else:
        # MCP session not available - use HTTP helpers for request data (if using HTTP transport)
        request = get_http_request()
        return {
            "message": "MCP session not available",
            "user_agent": request.headers.get("user-agent", "Unknown")
        }
```

For HTTP request access that works regardless of MCP session availability (when using HTTP transports), use the [HTTP request helpers](/servers/dependency-injection#http-request) like `get_http_request()` and `get_http_headers()`.

#### Client Metadata

<VersionBadge />

Clients can send contextual information with their requests using the `meta` parameter. This metadata is accessible through `ctx.request_context.meta` and is available for all MCP operations (tools, resources, prompts).

The `meta` field is `None` when clients don't provide metadata. When provided, metadata is accessible via attribute access (e.g., `meta.user_id`) rather than dictionary access. The structure of metadata is determined by the client making the request.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
def send_email(to: str, subject: str, body: str, ctx: Context) -> str:
    """Send an email, logging metadata about the request."""

    # Access client-provided metadata
    meta = ctx.request_context.meta

    if meta:
        # Meta is accessed as an object with attribute access
        user_id = meta.user_id if hasattr(meta, 'user_id') else None
        trace_id = meta.trace_id if hasattr(meta, 'trace_id') else None

        # Use metadata for logging, observability, etc.
        if trace_id:
            log_with_trace(f"Sending email for user {user_id}", trace_id)

    # Send the email...
    return f"Email sent to {to}"
```

<Warning>
  The MCP request is part of the low-level MCP SDK and intended for advanced use cases. Most users will not need to use it directly.
</Warning>


# Dependency Injection
Source: https://gofastmcp.com/servers/dependency-injection

Inject runtime values like HTTP requests, access tokens, and custom dependencies into your MCP components.

FastMCP uses dependency injection to provide runtime values to your tools, resources, and prompts. Instead of passing context through every layer of your code, you declare what you need as parameter defaults—FastMCP resolves them automatically when your function runs.

The dependency injection system is powered by [Docket](https://github.com/chrisguidry/docket) and its dependency system [uncalled-for](https://github.com/chrisguidry/uncalled-for). Core DI features like `Depends()` and `CurrentContext()` work without installing Docket. For background tasks and advanced task-related dependencies, install `fastmcp[tasks]`. For comprehensive coverage of dependency patterns, see the [Docket dependency documentation](https://docket.lol/en/latest/dependency-injection/).

<Note>
  Dependency parameters are automatically excluded from the MCP schema—clients never see them as callable parameters. This separation keeps your function signatures clean while giving you access to the runtime context you need.
</Note>

## How Dependency Injection Works

Dependency injection in FastMCP follows a simple pattern: declare a parameter with a recognized type annotation or a dependency default value, and FastMCP injects the resolved value at runtime.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.context import Context

mcp = FastMCP("Demo")


@mcp.tool
async def my_tool(query: str, ctx: Context) -> str:
    await ctx.info(f"Processing: {query}")
    return f"Results for: {query}"
```

When a client calls `my_tool`, they only see `query` as a parameter. The `ctx` parameter is injected automatically because it has a `Context` type annotation—FastMCP recognizes this and provides the active context for the request.

This works identically for tools, resources, resource templates, and prompts.

### Explicit Dependencies with CurrentContext

For more explicit code, you can use `CurrentContext()` as a default value instead of relying on the type annotation:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context

mcp = FastMCP("Demo")


@mcp.tool
async def my_tool(query: str, ctx: Context = CurrentContext()) -> str:
    await ctx.info(f"Processing: {query}")
    return f"Results for: {query}"
```

Both approaches work identically. The type-annotation approach is more concise; the explicit `CurrentContext()` approach makes the dependency injection visible in the signature.

## Built-in Dependencies

### MCP Context

The MCP Context provides logging, progress reporting, resource access, and other request-scoped operations. See [MCP Context](/servers/context) for the full API.

**Dependency injection:** Use a `Context` type annotation (FastMCP injects automatically) or `CurrentContext()`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.context import Context

mcp = FastMCP("Demo")


@mcp.tool
async def process_data(data: str, ctx: Context) -> str:
    await ctx.info(f"Processing: {data}")
    return "Done"


# Or explicitly with CurrentContext()
from fastmcp.dependencies import CurrentContext

@mcp.tool
async def process_data(data: str, ctx: Context = CurrentContext()) -> str:
    ...
```

**Function:** Use `get_context()` in helper functions or middleware:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.dependencies import get_context

async def log_something(message: str):
    ctx = get_context()
    await ctx.info(message)
```

### Server Instance

<VersionBadge />

Access the FastMCP server instance for introspection or server-level configuration.

**Dependency injection:** Use `CurrentFastMCP()`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import CurrentFastMCP

mcp = FastMCP("Demo")


@mcp.tool
async def server_info(server: FastMCP = CurrentFastMCP()) -> str:
    return f"Server: {server.name}"
```

**Function:** Use `get_server()`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.dependencies import get_server

def get_server_name() -> str:
    return get_server().name
```

### HTTP Request

<VersionBadge />

Access the Starlette Request when running over HTTP transports (SSE or Streamable HTTP).

**Dependency injection:** Use `CurrentRequest()`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import CurrentRequest
from starlette.requests import Request

mcp = FastMCP("Demo")


@mcp.tool
async def client_info(request: Request = CurrentRequest()) -> dict:
    return {
        "user_agent": request.headers.get("user-agent", "Unknown"),
        "client_ip": request.client.host if request.client else "Unknown",
    }
```

**Function:** Use `get_http_request()`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.dependencies import get_http_request

def get_client_ip() -> str:
    request = get_http_request()
    return request.client.host if request.client else "Unknown"
```

<Note>
  Both raise `RuntimeError` when called outside an HTTP context (e.g., STDIO transport). Use HTTP Headers if you need graceful fallback.
</Note>

### HTTP Headers

<VersionBadge />

Access HTTP headers with graceful fallback—returns an empty dictionary when no HTTP request is available, making it safe for code that might run over any transport.

**Dependency injection:** Use `CurrentHeaders()`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import CurrentHeaders

mcp = FastMCP("Demo")


@mcp.tool
async def get_auth_type(headers: dict = CurrentHeaders()) -> str:
    auth = headers.get("authorization", "")
    return "Bearer" if auth.startswith("Bearer ") else "None"
```

**Function:** Use `get_http_headers()`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.dependencies import get_http_headers

def get_user_agent() -> str:
    headers = get_http_headers()
    return headers.get("user-agent", "Unknown")
```

By default, problematic headers like `host` and `content-length` are excluded. Use `get_http_headers(include_all=True)` to include all headers.

### Access Token

<VersionBadge />

Access the authenticated user's token when your server uses authentication.

**Dependency injection:** Use `CurrentAccessToken()` (raises if not authenticated):

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import CurrentAccessToken
from fastmcp.server.auth import AccessToken

mcp = FastMCP("Demo")


@mcp.tool
async def get_user_id(token: AccessToken = CurrentAccessToken()) -> str:
    return token.claims.get("sub", "unknown")
```

**Function:** Use `get_access_token()` (returns `None` if not authenticated):

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.dependencies import get_access_token

@mcp.tool
async def get_user_info() -> dict:
    token = get_access_token()
    if token is None:
        return {"authenticated": False}
    return {"authenticated": True, "user": token.claims.get("sub")}
```

The `AccessToken` object provides:

* **`client_id`**: The OAuth client identifier
* **`scopes`**: List of granted permission scopes
* **`expires_at`**: Token expiration timestamp (if available)
* **`claims`**: Dictionary of all token claims (JWT claims or provider-specific data)

### Token Claims

When you need just one specific value from the token—like a user ID or tenant identifier—`TokenClaim()` extracts it directly without needing the full token object.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.dependencies import TokenClaim

mcp = FastMCP("Demo")


@mcp.tool
async def add_expense(
    amount: float,
    user_id: str = TokenClaim("oid"),  # Azure object ID
) -> dict:
    await db.insert({"user_id": user_id, "amount": amount})
    return {"status": "created", "user_id": user_id}
```

`TokenClaim()` raises a `RuntimeError` if the claim doesn't exist, listing available claims to help with debugging.

Common claims vary by identity provider:

| Provider    | User ID Claim | Email Claim | Name Claim |
| ----------- | ------------- | ----------- | ---------- |
| Azure/Entra | `oid`         | `email`     | `name`     |
| GitHub      | `sub`         | `email`     | `name`     |
| Google      | `sub`         | `email`     | `name`     |
| Auth0       | `sub`         | `email`     | `name`     |

### Background Task Dependencies

<VersionBadge />

For background task execution, FastMCP provides dependencies that integrate with [Docket](https://github.com/chrisguidry/docket). These require installing `fastmcp[tasks]`.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import CurrentDocket, CurrentWorker, Progress

mcp = FastMCP("Task Demo")


@mcp.tool(task=True)
async def long_running_task(
    data: str,
    docket=CurrentDocket(),
    worker=CurrentWorker(),
    progress=Progress(),
) -> str:
    await progress.set_total(100)

    for i in range(100):
        # Process chunk...
        await progress.increment()
        await progress.set_message(f"Processing chunk {i + 1}")

    return "Complete"
```

* **`CurrentDocket()`**: Access the Docket instance for scheduling additional background work
* **`CurrentWorker()`**: Access the worker processing tasks (name, concurrency settings)
* **`Progress()`**: Track task progress with atomic updates

<Note>
  Task dependencies require `pip install 'fastmcp[tasks]'`. They're only available within task-enabled components (`task=True`). For comprehensive task patterns, see the [Docket documentation](https://chrisguidry.github.io/docket/dependencies/).
</Note>

## Custom Dependencies

Beyond the built-in dependencies, you can create your own to inject configuration, database connections, API clients, or any other values your functions need.

### Using Depends()

The `Depends()` function wraps any callable and injects its return value. This works with synchronous functions, async functions, and async context managers.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import Depends

mcp = FastMCP("Custom Deps Demo")


def get_config() -> dict:
    return {"api_url": "https://api.example.com", "timeout": 30}


async def get_user_id() -> int:
    # Could fetch from database, external service, etc.
    return 42


@mcp.tool
async def fetch_data(
    query: str,
    config: dict = Depends(get_config),
    user_id: int = Depends(get_user_id),
) -> str:
    return f"User {user_id} fetching '{query}' from {config['api_url']}"
```

### Caching

Dependencies are cached per-request. If multiple parameters use the same dependency, or if nested dependencies share a common dependency, it's resolved once and the same instance is reused.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import Depends

mcp = FastMCP("Caching Demo")


def get_db_connection():
    print("Connecting to database...")  # Only printed once per request
    return {"connection": "active"}


def get_user_repo(db=Depends(get_db_connection)):
    return {"db": db, "type": "user"}


def get_order_repo(db=Depends(get_db_connection)):
    return {"db": db, "type": "order"}


@mcp.tool
async def process_order(
    order_id: str,
    users=Depends(get_user_repo),
    orders=Depends(get_order_repo),
) -> str:
    # Both repos share the same db connection
    return f"Processed order {order_id}"
```

### Resource Management

For dependencies that need cleanup—database connections, file handles, HTTP clients—use an async context manager. The cleanup code runs after your function completes, even if an error occurs.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.dependencies import Depends

mcp = FastMCP("Resource Demo")


@asynccontextmanager
async def get_database():
    db = await connect_to_database()
    try:
        yield db
    finally:
        await db.close()


@mcp.tool
async def query_users(sql: str, db=Depends(get_database)) -> list:
    return await db.execute(sql)
```

### Nested Dependencies

Dependencies can depend on other dependencies. FastMCP resolves them in the correct order and applies caching across the dependency tree.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import Depends

mcp = FastMCP("Nested Demo")


def get_base_url() -> str:
    return "https://api.example.com"


def get_api_client(base_url: str = Depends(get_base_url)) -> dict:
    return {"base_url": base_url, "version": "v1"}


@mcp.tool
async def call_api(endpoint: str, client: dict = Depends(get_api_client)) -> str:
    return f"Calling {client['base_url']}/{client['version']}/{endpoint}"
```

For advanced dependency patterns—like `TaskArgument()` for accessing task parameters, or custom `Dependency` subclasses—see the [Docket dependency documentation](https://chrisguidry.github.io/docket/dependencies/).


# User Elicitation
Source: https://gofastmcp.com/servers/elicitation

Request structured input from users during tool execution through the MCP context.

<VersionBadge />

User elicitation allows MCP servers to request structured input from users during tool execution. Instead of requiring all inputs upfront, tools can interactively ask for missing parameters, clarification, or additional context as needed.

Elicitation enables tools to pause execution and request specific information from users:

* **Missing parameters**: Ask for required information not provided initially
* **Clarification requests**: Get user confirmation or choices for ambiguous scenarios
* **Progressive disclosure**: Collect complex information step-by-step
* **Dynamic workflows**: Adapt tool behavior based on user responses

For example, a file management tool might ask "Which directory should I create?" or a data analysis tool might request "What date range should I analyze?"

## Overview

Use the `ctx.elicit()` method within any tool function to request user input. Specify the message to display and the type of response you expect.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context
from dataclasses import dataclass

mcp = FastMCP("Elicitation Server")

@dataclass
class UserInfo:
    name: str
    age: int

@mcp.tool
async def collect_user_info(ctx: Context) -> str:
    """Collect user information through interactive prompts."""
    result = await ctx.elicit(
        message="Please provide your information",
        response_type=UserInfo
    )

    if result.action == "accept":
        user = result.data
        return f"Hello {user.name}, you are {user.age} years old"
    elif result.action == "decline":
        return "Information not provided"
    else:  # cancel
        return "Operation cancelled"
```

The elicitation result contains an `action` field indicating how the user responded:

| Action    | Description                                                     |
| --------- | --------------------------------------------------------------- |
| `accept`  | User provided valid input—data is available in the `data` field |
| `decline` | User chose not to provide the requested information             |
| `cancel`  | User cancelled the entire operation                             |

FastMCP also provides typed result classes for pattern matching:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    DeclinedElicitation,
    CancelledElicitation,
)

@mcp.tool
async def pattern_example(ctx: Context) -> str:
    result = await ctx.elicit("Enter your name:", response_type=str)

    match result:
        case AcceptedElicitation(data=name):
            return f"Hello {name}!"
        case DeclinedElicitation():
            return "No name provided"
        case CancelledElicitation():
            return "Operation cancelled"
```

### Multi-Turn Elicitation

Tools can make multiple elicitation calls to gather information progressively:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
async def plan_meeting(ctx: Context) -> str:
    """Plan a meeting by gathering details step by step."""

    title_result = await ctx.elicit("What's the meeting title?", response_type=str)
    if title_result.action != "accept":
        return "Meeting planning cancelled"

    duration_result = await ctx.elicit("Duration in minutes?", response_type=int)
    if duration_result.action != "accept":
        return "Meeting planning cancelled"

    priority_result = await ctx.elicit(
        "Is this urgent?",
        response_type=["yes", "no"]
    )
    if priority_result.action != "accept":
        return "Meeting planning cancelled"

    urgent = priority_result.data == "yes"
    return f"Meeting '{title_result.data}' for {duration_result.data} minutes (Urgent: {urgent})"
```

### Client Requirements

Elicitation requires the client to implement an elicitation handler. If a client doesn't support elicitation, calls to `ctx.elicit()` will raise an error indicating that elicitation is not supported.

See [Client Elicitation](/clients/elicitation) for details on how clients handle these requests.

## Schema and Response Types

The server must send a schema to the client indicating the type of data it expects in response to the elicitation request. The MCP spec only supports a limited subset of JSON Schema types for elicitation responses—specifically JSON **objects** with **primitive** properties including `string`, `number` (or `integer`), `boolean`, and `enum` fields.

FastMCP makes it easy to request a broader range of types, including scalars (e.g. `str`) or no response at all, by automatically wrapping them in MCP-compatible object schemas.

### Scalar Types

You can request simple scalar data types for basic input, such as a string, integer, or boolean. When you request a scalar type, FastMCP automatically wraps it in an object schema for MCP spec compatibility. Clients will see a schema requesting a single "value" field of the requested type. Once clients respond, the provided object is "unwrapped" and the scalar value is returned directly in the `data` field.

<CodeGroup>
  ```python title="String" theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  @mcp.tool
  async def get_user_name(ctx: Context) -> str:
      result = await ctx.elicit("What's your name?", response_type=str)

      if result.action == "accept":
          return f"Hello, {result.data}!"
      return "No name provided"
  ```

  ```python title="Integer" theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  @mcp.tool
  async def pick_a_number(ctx: Context) -> str:
      result = await ctx.elicit("Pick a number!", response_type=int)

      if result.action == "accept":
          return f"You picked {result.data}"
      return "No number provided"
  ```

  ```python title="Boolean" theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  @mcp.tool
  async def pick_a_boolean(ctx: Context) -> str:
      result = await ctx.elicit("True or false?", response_type=bool)

      if result.action == "accept":
          return f"You picked {result.data}"
      return "No boolean provided"
  ```
</CodeGroup>

### No Response

Sometimes, the goal of an elicitation is to simply get a user to approve or reject an action. Pass `None` as the response type to indicate that no data is expected. The `data` field will be `None` when the user accepts.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
async def approve_action(ctx: Context) -> str:
    result = await ctx.elicit("Approve this action?", response_type=None)

    if result.action == "accept":
        return do_action()
    else:
        raise ValueError("Action rejected")
```

### Constrained Options

Constrain the user's response to a specific set of values using a `Literal` type, Python enum, or a list of strings as a convenient shortcut.

<CodeGroup>
  ```python title="List of strings" theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  @mcp.tool
  async def set_priority(ctx: Context) -> str:
      result = await ctx.elicit(
          "What priority level?",
          response_type=["low", "medium", "high"],
      )

      if result.action == "accept":
          return f"Priority set to: {result.data}"
  ```

  ```python title="Literal type" theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  from typing import Literal

  @mcp.tool
  async def set_priority(ctx: Context) -> str:
      result = await ctx.elicit(
          "What priority level?",
          response_type=Literal["low", "medium", "high"]
      )

      if result.action == "accept":
          return f"Priority set to: {result.data}"
      return "No priority set"
  ```

  ```python title="Python enum" theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  from enum import Enum

  class Priority(Enum):
      LOW = "low"
      MEDIUM = "medium"
      HIGH = "high"

  @mcp.tool
  async def set_priority(ctx: Context) -> str:
      result = await ctx.elicit("What priority level?", response_type=Priority)

      if result.action == "accept":
          return f"Priority set to: {result.data.value}"
      return "No priority set"
  ```
</CodeGroup>

### Multi-Select

<VersionBadge />

Enable multi-select by wrapping your choices in an additional list level. This allows users to select multiple values from the available options.

<CodeGroup>
  ```python title="List of strings" theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  @mcp.tool
  async def select_tags(ctx: Context) -> str:
      result = await ctx.elicit(
          "Choose tags",
          response_type=[["bug", "feature", "documentation"]]  # Note: list of a list
      )

      if result.action == "accept":
          tags = result.data
          return f"Selected tags: {', '.join(tags)}"
  ```

  ```python title="list[Enum] type" theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  from enum import Enum

  class Tag(Enum):
      BUG = "bug"
      FEATURE = "feature"
      DOCS = "documentation"

  @mcp.tool
  async def select_tags(ctx: Context) -> str:
      result = await ctx.elicit(
          "Choose tags",
          response_type=list[Tag]
      )
      if result.action == "accept":
          tags = [tag.value for tag in result.data]
          return f"Selected: {', '.join(tags)}"
  ```
</CodeGroup>

### Titled Options

<VersionBadge />

For better UI display, provide human-readable titles for enum options. FastMCP generates SEP-1330 compliant schemas using the `oneOf` pattern with `const` and `title` fields.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
async def set_priority(ctx: Context) -> str:
    result = await ctx.elicit(
        "What priority level?",
        response_type={
            "low": {"title": "Low Priority"},
            "medium": {"title": "Medium Priority"},
            "high": {"title": "High Priority"}
        }
    )

    if result.action == "accept":
        return f"Priority set to: {result.data}"
```

For multi-select with titles, wrap the dict in a list:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
async def select_priorities(ctx: Context) -> str:
    result = await ctx.elicit(
        "Choose priorities",
        response_type=[{
            "low": {"title": "Low Priority"},
            "medium": {"title": "Medium Priority"},
            "high": {"title": "High Priority"}
        }]
    )

    if result.action == "accept":
        return f"Selected: {', '.join(result.data)}"
```

### Structured Responses

Request structured data with multiple fields by using a dataclass, typed dict, or Pydantic model as the response type. Note that the MCP spec only supports shallow objects with scalar (string, number, boolean) or enum properties.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from dataclasses import dataclass
from typing import Literal

@dataclass
class TaskDetails:
    title: str
    description: str
    priority: Literal["low", "medium", "high"]
    due_date: str

@mcp.tool
async def create_task(ctx: Context) -> str:
    result = await ctx.elicit(
        "Please provide task details",
        response_type=TaskDetails
    )

    if result.action == "accept":
        task = result.data
        return f"Created task: {task.title} (Priority: {task.priority})"
    return "Task creation cancelled"
```

### Default Values

<VersionBadge />

Provide default values for elicitation fields using Pydantic's `Field(default=...)`. Clients will pre-populate form fields with these defaults. Fields with default values are automatically marked as optional.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from pydantic import BaseModel, Field
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskDetails(BaseModel):
    title: str = Field(description="Task title")
    description: str = Field(default="", description="Task description")
    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority")

@mcp.tool
async def create_task(ctx: Context) -> str:
    result = await ctx.elicit("Please provide task details", response_type=TaskDetails)
    if result.action == "accept":
        return f"Created: {result.data.title}"
    return "Task creation cancelled"
```

Default values are supported for strings, integers, numbers, booleans, and enums.


# Icons
Source: https://gofastmcp.com/servers/icons

Add visual icons to your servers, tools, resources, and prompts

<VersionBadge />

Icons provide visual representations for your MCP servers and components, helping client applications present better user interfaces. When displayed in MCP clients, icons help users quickly identify and navigate your server's capabilities.

## Icon Format

Icons use the standard MCP Icon type from the MCP protocol specification. Each icon specifies a source URL or data URI, and optionally includes MIME type and size information.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from mcp.types import Icon

icon = Icon(
    src="https://example.com/icon.png",
    mimeType="image/png",
    sizes=["48x48"]
)
```

The fields serve different purposes:

* **src**: URL or data URI pointing to the icon image
* **mimeType** (optional): MIME type of the image (e.g., "image/png", "image/svg+xml")
* **sizes** (optional): Array of size descriptors (e.g., \["48x48"], \["any"])

## Server Icons

Add icons and a website URL to your server for display in client applications. Multiple icons at different sizes help clients choose the best resolution for their display context.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from mcp.types import Icon

mcp = FastMCP(
    name="WeatherService",
    website_url="https://weather.example.com",
    icons=[
        Icon(
            src="https://weather.example.com/icon-48.png",
            mimeType="image/png",
            sizes=["48x48"]
        ),
        Icon(
            src="https://weather.example.com/icon-96.png",
            mimeType="image/png",
            sizes=["96x96"]
        ),
    ]
)
```

Server icons appear in MCP client interfaces to help users identify your server among others they may have installed.

## Component Icons

Icons can be added to individual tools, resources, resource templates, and prompts. This helps users visually distinguish between different component types and purposes.

### Tool Icons

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from mcp.types import Icon

@mcp.tool(
    icons=[Icon(src="https://example.com/calculator-icon.png")]
)
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

### Resource Icons

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.resource(
    "config://settings",
    icons=[Icon(src="https://example.com/config-icon.png")]
)
def get_settings() -> dict:
    """Retrieve application settings."""
    return {"theme": "dark", "language": "en"}
```

### Resource Template Icons

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.resource(
    "user://{user_id}/profile",
    icons=[Icon(src="https://example.com/user-icon.png")]
)
def get_user_profile(user_id: str) -> dict:
    """Get a user's profile."""
    return {"id": user_id, "name": f"User {user_id}"}
```

### Prompt Icons

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.prompt(
    icons=[Icon(src="https://example.com/prompt-icon.png")]
)
def analyze_code(code: str):
    """Create a prompt for code analysis."""
    return f"Please analyze this code:\n\n{code}"
```

## Using Data URIs

For small icons or when you want to embed the icon directly without external dependencies, use data URIs. This approach eliminates the need for hosting and ensures the icon is always available.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from mcp.types import Icon
from fastmcp.utilities.types import Image

# SVG icon as data URI
svg_icon = Icon(
    src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6Ii8+PC9zdmc+",
    mimeType="image/svg+xml"
)

@mcp.tool(icons=[svg_icon])
def my_tool() -> str:
    """A tool with an embedded SVG icon."""
    return "result"
```

### Generating Data URIs from Files

FastMCP provides the `Image` utility class to convert local image files into data URIs.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from mcp.types import Icon
from fastmcp.utilities.types import Image

# Generate a data URI from a local image file
img = Image(path="./assets/brand/favicon.png")
icon = Icon(src=img.to_data_uri())

@mcp.tool(icons=[icon])
def file_icon_tool() -> str:
    """A tool with an icon generated from a local file."""
    return "result"
```

This approach is useful when you have local image assets and want to embed them directly in your server definition.


# Lifespans
Source: https://gofastmcp.com/servers/lifespan

Server-level setup and teardown with composable lifespans

<VersionBadge />

Lifespans let you run code once when the server starts and clean up when it stops. Unlike per-session handlers, lifespans run exactly once regardless of how many clients connect.

## Basic Usage

Use the `@lifespan` decorator to define a lifespan:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

@lifespan
async def app_lifespan(server):
    # Setup: runs once when server starts
    print("Starting up...")
    try:
        yield {"started_at": "2024-01-01"}
    finally:
        # Teardown: runs when server stops
        print("Shutting down...")

mcp = FastMCP("MyServer", lifespan=app_lifespan)
```

The dict you yield becomes the **lifespan context**, accessible from tools.

<Note>
  Always use `try/finally` for cleanup code to ensure it runs even if the server is cancelled.
</Note>

## Accessing Lifespan Context

Access the lifespan context in tools via `ctx.lifespan_context`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context
from fastmcp.server.lifespan import lifespan

@lifespan
async def app_lifespan(server):
    # Initialize shared state
    data = {"users": ["alice", "bob"]}
    yield {"data": data}

mcp = FastMCP("MyServer", lifespan=app_lifespan)

@mcp.tool
def list_users(ctx: Context) -> list[str]:
    data = ctx.lifespan_context["data"]
    return data["users"]
```

## Composing Lifespans

Compose multiple lifespans with the `|` operator:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

@lifespan
async def config_lifespan(server):
    config = {"debug": True, "version": "1.0"}
    yield {"config": config}

@lifespan
async def data_lifespan(server):
    data = {"items": []}
    yield {"data": data}

# Compose with |
mcp = FastMCP("MyServer", lifespan=config_lifespan | data_lifespan)
```

Composed lifespans:

* Enter in order (left to right)
* Exit in reverse order (right to left)
* Merge their context dicts (later values overwrite earlier on conflict)

## Backwards Compatibility

Existing `@asynccontextmanager` lifespans still work when passed directly to FastMCP:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from contextlib import asynccontextmanager
from fastmcp import FastMCP

@asynccontextmanager
async def legacy_lifespan(server):
    yield {"key": "value"}

mcp = FastMCP("MyServer", lifespan=legacy_lifespan)
```

To compose an `@asynccontextmanager` function with `@lifespan` functions, wrap it with `ContextManagerLifespan`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from contextlib import asynccontextmanager
from fastmcp.server.lifespan import lifespan, ContextManagerLifespan

@asynccontextmanager
async def legacy_lifespan(server):
    yield {"legacy": True}

@lifespan
async def new_lifespan(server):
    yield {"new": True}

# Wrap the legacy lifespan explicitly for composition
combined = ContextManagerLifespan(legacy_lifespan) | new_lifespan
```

## With FastAPI

When mounting FastMCP into FastAPI, use `combine_lifespans` to run both your app's lifespan and the MCP server's lifespan:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.utilities.lifespan import combine_lifespans

@asynccontextmanager
async def app_lifespan(app):
    print("FastAPI starting...")
    yield
    print("FastAPI shutting down...")

mcp = FastMCP("Tools")
mcp_app = mcp.http_app()

app = FastAPI(lifespan=combine_lifespans(app_lifespan, mcp_app.lifespan))
app.mount("/mcp", mcp_app)
```

See the [FastAPI integration guide](/integrations/fastapi#combining-lifespans) for full details.
