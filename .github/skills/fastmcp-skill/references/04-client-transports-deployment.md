# Client Transports and Deployment

Source lines: 3389-5086 from the original FastMCP documentation dump.

Tool calling, transports, HTTP deployment, ASGI mounting, production deployment, Horizon hosting, and run() transport choices.

---

# Calling Tools
Source: https://gofastmcp.com/clients/tools

Execute server-side tools and handle structured results.

<VersionBadge />

Use this when you need to execute server-side functions and process their results.

Tools are executable functions exposed by MCP servers. The client's `call_tool()` method executes a tool by name with arguments and returns structured results.

## Basic Execution

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.call_tool("add", {"a": 5, "b": 3})
    # result -> CallToolResult with structured and unstructured data

    # Access structured data (automatically deserialized)
    print(result.data)  # 8

    # Access traditional content blocks
    print(result.content[0].text)  # "8"
```

Arguments are passed as a dictionary. For multi-server clients, tool names are automatically prefixed with the server name (e.g., `weather_get_forecast` for a tool named `get_forecast` on the `weather` server).

## Execution Options

The `call_tool()` method supports timeout control and progress monitoring:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    # With timeout (aborts if execution takes longer than 2 seconds)
    result = await client.call_tool(
        "long_running_task",
        {"param": "value"},
        timeout=2.0
    )

    # With progress handler
    result = await client.call_tool(
        "long_running_task",
        {"param": "value"},
        progress_handler=my_progress_handler
    )
```

## Structured Results

<VersionBadge />

Tool execution returns a `CallToolResult` object. The `.data` property provides fully hydrated Python objects including complex types like datetimes and UUIDs, reconstructed from the server's output schema.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from datetime import datetime
from uuid import UUID

async with client:
    result = await client.call_tool("get_weather", {"city": "London"})

    # FastMCP reconstructs complete Python objects
    weather = result.data
    print(f"Temperature: {weather.temperature}C at {weather.timestamp}")

    # Complex types are properly deserialized
    assert isinstance(weather.timestamp, datetime)
    assert isinstance(weather.station_id, UUID)

    # Raw structured JSON is also available
    print(f"Raw JSON: {result.structured_content}")
```

<Card icon="code" title="CallToolResult Properties">
  <ResponseField name=".data" type="Any">
    Fully hydrated Python objects with complex type support (datetimes, UUIDs, custom classes). FastMCP exclusive.
  </ResponseField>

  <ResponseField name=".content" type="list[mcp.types.ContentBlock]">
    Standard MCP content blocks (`TextContent`, `ImageContent`, `AudioContent`, etc.).
  </ResponseField>

  <ResponseField name=".structured_content" type="dict[str, Any] | None">
    Standard MCP structured JSON data as sent by the server.
  </ResponseField>

  <ResponseField name=".is_error" type="bool">
    Boolean indicating if the tool execution failed.
  </ResponseField>
</Card>

For tools without output schemas or when deserialization fails, `.data` will be `None`. Fall back to content blocks in that case:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.call_tool("legacy_tool", {"param": "value"})

    if result.data is not None:
        print(f"Structured: {result.data}")
    else:
        for content in result.content:
            if hasattr(content, 'text'):
                print(f"Text result: {content.text}")
```

<Tip>
  FastMCP servers automatically wrap primitive results (like `int`, `str`, `bool`) in a `{"result": value}` structure. FastMCP clients automatically unwrap this, so you get the original value in `.data`.
</Tip>

## Error Handling

By default, `call_tool()` raises a `ToolError` if the tool execution fails:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.exceptions import ToolError

async with client:
    try:
        result = await client.call_tool("potentially_failing_tool", {"param": "value"})
        print("Tool succeeded:", result.data)
    except ToolError as e:
        print(f"Tool failed: {e}")
```

To handle errors manually instead of catching exceptions, disable automatic error raising:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.call_tool(
        "potentially_failing_tool",
        {"param": "value"},
        raise_on_error=False
    )

    if result.is_error:
        print(f"Tool failed: {result.content[0].text}")
    else:
        print(f"Tool succeeded: {result.data}")
```

## Sending Metadata

<VersionBadge />

The `meta` parameter sends ancillary information alongside tool calls for observability, debugging, or client identification:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.call_tool(
        name="send_email",
        arguments={
            "to": "user@example.com",
            "subject": "Hello",
            "body": "Welcome!"
        },
        meta={
            "trace_id": "abc-123",
            "request_source": "mobile_app"
        }
    )
```

See [Client Metadata](/servers/context#client-metadata) to learn how servers access this data.

## Raw Protocol Access

For complete control, use `call_tool_mcp()` which returns the raw MCP protocol object:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.call_tool_mcp("my_tool", {"param": "value"})
    # result -> mcp.types.CallToolResult

    if result.isError:
        print(f"Tool failed: {result.content}")
    else:
        print(f"Tool succeeded: {result.content}")
        # Note: No automatic deserialization with call_tool_mcp()
```


# Client Transports
Source: https://gofastmcp.com/clients/transports

Configure how clients connect to and communicate with MCP servers.

<VersionBadge />

Transports handle the underlying connection between your client and MCP servers. While the client can automatically select a transport based on what you pass to it, instantiating transports explicitly gives you full control over configuration.

## STDIO Transport

STDIO transport communicates with MCP servers through subprocess pipes. When using STDIO, your client launches and manages the server process, controlling its lifecycle and environment.

<Warning>
  STDIO servers run in isolated environments by default. They do not inherit your shell's environment variables. You must explicitly pass any configuration the server needs.
</Warning>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

transport = StdioTransport(
    command="python",
    args=["my_server.py", "--verbose"],
    env={"API_KEY": "secret", "LOG_LEVEL": "DEBUG"},
    cwd="/path/to/server"
)
client = Client(transport)
```

For convenience, the client can infer STDIO transport from file paths, though this limits configuration options:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

client = Client("my_server.py")  # Limited - no configuration options
```

### Environment Variables

Since STDIO servers do not inherit your environment, you need strategies for passing configuration.

**Selective forwarding** passes only the variables your server needs:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import os
from fastmcp.client.transports import StdioTransport

required_vars = ["API_KEY", "DATABASE_URL", "REDIS_HOST"]
env = {var: os.environ[var] for var in required_vars if var in os.environ}

transport = StdioTransport(command="python", args=["server.py"], env=env)
client = Client(transport)
```

**Loading from .env files** keeps configuration separate from code:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from dotenv import dotenv_values
from fastmcp.client.transports import StdioTransport

env = dotenv_values(".env")
transport = StdioTransport(command="python", args=["server.py"], env=env)
client = Client(transport)
```

### Session Persistence

STDIO transports maintain sessions across multiple client contexts by default (`keep_alive=True`). This reuses the same subprocess for multiple connections, improving performance.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.client.transports import StdioTransport

transport = StdioTransport(command="python", args=["server.py"])
client = Client(transport)

async def efficient_multiple_operations():
    async with client:
        await client.ping()

    async with client:  # Reuses the same subprocess
        await client.call_tool("process_data", {"file": "data.csv"})
```

For complete isolation between connections, disable session persistence:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transport = StdioTransport(command="python", args=["server.py"], keep_alive=False)
```

## HTTP Transport

<VersionBadge />

HTTP transport connects to MCP servers running as web services. This is the recommended transport for production deployments.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="https://api.example.com/mcp",
    headers={
        "Authorization": "Bearer your-token-here",
        "X-Custom-Header": "value"
    }
)
client = Client(transport)
```

FastMCP also provides authentication helpers:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

client = Client(
    "https://api.example.com/mcp",
    auth=BearerAuth("your-token-here")
)
```

### SSE Transport

Server-Sent Events transport is maintained for backward compatibility. Use Streamable HTTP for new deployments unless you have specific infrastructure requirements.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.client.transports import SSETransport

transport = SSETransport(
    url="https://api.example.com/sse",
    headers={"Authorization": "Bearer token"}
)
client = Client(transport)
```

## In-Memory Transport

In-memory transport connects directly to a FastMCP server instance within the same Python process. This eliminates both subprocess management and network overhead, making it ideal for testing.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Client
import os

mcp = FastMCP("TestServer")

@mcp.tool
def greet(name: str) -> str:
    prefix = os.environ.get("GREETING_PREFIX", "Hello")
    return f"{prefix}, {name}!"

client = Client(mcp)

async with client:
    result = await client.call_tool("greet", {"name": "World"})
```

<Note>
  Unlike STDIO transports, in-memory servers share the same memory space and environment variables as your client code.
</Note>

## Multi-Server Configuration

<VersionBadge />

Connect to multiple servers defined in a configuration dictionary:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather.example.com/mcp",
            "transport": "http"
        },
        "assistant": {
            "command": "python",
            "args": ["./assistant.py"],
            "env": {"LOG_LEVEL": "INFO"}
        }
    }
}

client = Client(config)

async with client:
    # Tools are namespaced by server
    weather = await client.call_tool("weather_get_forecast", {"city": "NYC"})
    answer = await client.call_tool("assistant_ask", {"question": "What?"})
```

### Tool Transformations

FastMCP supports tool transformations within the configuration. You can change names, descriptions, tags, and arguments for tools from a server.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather.example.com/mcp",
            "transport": "http",
            "tools": {
                "weather_get_forecast": {
                    "name": "miami_weather",
                    "description": "Get the weather for Miami",
                    "arguments": {
                        "city": {
                            "default": "Miami",
                            "hide": True,
                        }
                    }
                }
            }
        }
    }
}
```

To filter tools by tag, use `include_tags` or `exclude_tags` at the server level:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather.example.com/mcp",
            "include_tags": ["forecast"]  # Only tools with this tag
        }
    }
}
```


# HTTP Deployment
Source: https://gofastmcp.com/deployment/http

Deploy your FastMCP server over HTTP for remote access

<Tip>
  STDIO transport is perfect for local development and desktop applications. But to unlock the full potential of MCP—centralized services, multi-client access, and network availability—you need remote HTTP deployment.
</Tip>

This guide walks you through deploying your FastMCP server as a remote MCP service that's accessible via a URL. Once deployed, your MCP server will be available over the network, allowing multiple clients to connect simultaneously and enabling integration with cloud-based LLM applications. This guide focuses specifically on remote MCP deployment, not local STDIO servers.

## Choosing Your Approach

FastMCP provides two ways to deploy your server as an HTTP service. Understanding the trade-offs helps you choose the right approach for your needs.

The **direct HTTP server** approach is simpler and perfect for getting started quickly. You modify your server's `run()` method to use HTTP transport, and FastMCP handles all the web server configuration. This approach works well for standalone deployments where you want your MCP server to be the only service running on a port.

The **ASGI application** approach gives you more control and flexibility. Instead of running the server directly, you create an ASGI application that can be served by Uvicorn. This approach is better when you need advanced server features like multiple workers, custom middleware, or when you're integrating with existing web applications.

### Direct HTTP Server

The simplest way to get your MCP server online is to use the built-in `run()` method with HTTP transport. This approach handles all the server configuration for you and is ideal when you want a standalone MCP server without additional complexity.

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool
def process_data(input: str) -> str:
    """Process data on the server"""
    return f"Processed: {input}"

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
```

Run your server with a simple Python command:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
python server.py
```

Your server is now accessible at `http://localhost:8000/mcp` (or use your server's actual IP address for remote access).

This approach is ideal when you want to get online quickly with minimal configuration. It's perfect for internal tools, development environments, or simple deployments where you don't need advanced server features. The built-in server handles all the HTTP details, letting you focus on your MCP implementation.

### ASGI Application

For production deployments, you'll often want more control over how your server runs. FastMCP can create a standard ASGI application that works with any ASGI server like Uvicorn, Gunicorn, or Hypercorn. This approach is particularly useful when you need to configure advanced server options, run multiple workers, or integrate with existing infrastructure.

```python app.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool
def process_data(input: str) -> str:
    """Process data on the server"""
    return f"Processed: {input}"

# Create ASGI application
app = mcp.http_app()
```

Run with any ASGI server - here's an example with Uvicorn:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
uvicorn app:app --host 0.0.0.0 --port 8000
```

Your server is accessible at the same URL: `http://localhost:8000/mcp` (or use your server's actual IP address for remote access).

The ASGI approach shines in production environments where you need reliability and performance. You can run multiple worker processes to handle concurrent requests, add custom middleware for logging or monitoring, integrate with existing deployment pipelines, or mount your MCP server as part of a larger application.

## Configuring Your Server

### Custom Path

By default, your MCP server is accessible at `/mcp/` on your domain. You can customize this path to fit your URL structure or avoid conflicts with existing endpoints. This is particularly useful when integrating MCP into an existing application or following specific API conventions.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Option 1: With mcp.run()
mcp.run(transport="http", host="0.0.0.0", port=8000, path="/api/mcp/")

# Option 2: With ASGI app
app = mcp.http_app(path="/api/mcp/")
```

Now your server is accessible at `http://localhost:8000/api/mcp/`.

### Authentication

<Warning>
  Authentication is **highly recommended** for remote MCP servers. Some LLM clients require authentication for remote servers and will refuse to connect without it.
</Warning>

FastMCP supports multiple authentication methods to secure your remote server. See the [Authentication Overview](/servers/auth/authentication) for complete configuration options including Bearer tokens, JWT, and OAuth.

If you're mounting an authenticated server under a path prefix, see [Mounting Authenticated Servers](#mounting-authenticated-servers) below for important routing considerations.

### Health Checks

Health check endpoints are essential for monitoring your deployed server and ensuring it's responding correctly. FastMCP allows you to add custom routes alongside your MCP endpoints, making it easy to implement health checks that work with both deployment approaches.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from starlette.responses import JSONResponse

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "healthy", "service": "mcp-server"})
```

This health endpoint will be available at `http://localhost:8000/health` and can be used by load balancers, monitoring systems, or deployment platforms to verify your server is running.

### Custom Middleware

<VersionBadge />

Add custom Starlette middleware to your FastMCP ASGI apps:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Create your FastMCP server
mcp = FastMCP("MyServer")

# Define middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

# Create ASGI app with middleware
http_app = mcp.http_app(middleware=middleware)
```

### CORS for Browser-Based Clients

<Tip>
  Most MCP clients, including those that you access through a browser like ChatGPT or Claude, don't need CORS configuration. Only enable CORS if you're working with an MCP client that connects directly from a browser, such as debugging tools or inspectors.
</Tip>

CORS (Cross-Origin Resource Sharing) is needed when JavaScript running in a web browser connects directly to your MCP server. This is different from using an LLM through a browser—in that case, the browser connects to the LLM service, and the LLM service connects to your MCP server (no CORS needed).

Browser-based MCP clients that need CORS include:

* **MCP Inspector** - Browser-based debugging tool for testing MCP servers
* **Custom browser-based MCP clients** - If you're building a web app that directly connects to MCP servers

For these scenarios, add CORS middleware with the specific headers required for MCP protocol:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

mcp = FastMCP("MyServer")

# Configure CORS for browser-based clients
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins; use specific origins for security
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "mcp-protocol-version",
            "mcp-session-id",
            "Authorization",
            "Content-Type",
        ],
        expose_headers=["mcp-session-id"],
    )
]

app = mcp.http_app(middleware=middleware)
```

**Key configuration details:**

* **`allow_origins`**: Specify exact origins (e.g., `["http://localhost:3000"]`) rather than `["*"]` for production deployments
* **`allow_headers`**: Must include `mcp-protocol-version`, `mcp-session-id`, and `Authorization` (for authenticated servers)
* **`expose_headers`**: Must include `mcp-session-id` so JavaScript can read the session ID from responses and send it in subsequent requests

Without `expose_headers=["mcp-session-id"]`, browsers will receive the session ID but JavaScript won't be able to access it, causing session management to fail.

<Warning>
  **Production Security**: Never use `allow_origins=["*"]` in production. Specify the exact origins of your browser-based clients. Using wildcards exposes your server to unauthorized access from any website.
</Warning>

### SSE Polling for Long-Running Operations

<VersionBadge />

<Note>
  This feature only applies to the **StreamableHTTP transport** (the default for `http_app()`). It does not apply to the legacy SSE transport (`transport="sse"`).
</Note>

When running tools that take a long time to complete, you may encounter issues with load balancers or proxies terminating connections that stay idle too long. [SEP-1699](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1699) introduces SSE polling to solve this by allowing the server to gracefully close connections and have clients automatically reconnect.

To enable SSE polling, configure an `EventStore` when creating your HTTP application:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Context
from fastmcp.server.event_store import EventStore

mcp = FastMCP("My Server")

@mcp.tool
async def long_running_task(ctx: Context) -> str:
    """A task that takes several minutes to complete."""
    for i in range(100):
        await ctx.report_progress(i, 100)

        # Periodically close the connection to avoid load balancer timeouts
        # Client will automatically reconnect and resume receiving progress
        if i % 30 == 0 and i > 0:
            await ctx.close_sse_stream()

        await do_expensive_work()

    return "Done!"

# Configure with EventStore for resumability
event_store = EventStore()
app = mcp.http_app(
    event_store=event_store,
    retry_interval=2000,  # Client reconnects after 2 seconds
)
```

**How it works:**

1. When `event_store` is configured, the server stores all events (progress updates, results) with unique IDs
2. Calling `ctx.close_sse_stream()` gracefully closes the HTTP connection
3. The client automatically reconnects with a `Last-Event-ID` header
4. The server replays any events the client missed during the disconnection

The `retry_interval` parameter (in milliseconds) controls how long clients wait before reconnecting. Choose a value that balances responsiveness with server load.

<Note>
  `close_sse_stream()` is a no-op if called without an `EventStore` configured, so you can safely include it in tools that may run in different deployment configurations.
</Note>

#### Custom Storage Backends

By default, `EventStore` uses in-memory storage. For production deployments with multiple server instances, you can provide a custom storage backend using the `key_value` package:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.event_store import EventStore
from key_value.aio.stores.redis import RedisStore

# Use Redis for distributed deployments
redis_store = RedisStore(url="redis://localhost:6379")
event_store = EventStore(
    storage=redis_store,
    max_events_per_stream=100,  # Keep last 100 events per stream
    ttl=3600,  # Events expire after 1 hour
)

app = mcp.http_app(event_store=event_store)
```

## Integration with Web Frameworks

If you already have a web application running, you can add MCP capabilities by mounting a FastMCP server as a sub-application. This allows you to expose MCP tools alongside your existing API endpoints, sharing the same domain and infrastructure. The MCP server becomes just another route in your application, making it easy to manage and deploy.

### Mounting in Starlette

Mount your FastMCP server in a Starlette application:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

# Create your FastMCP server
mcp = FastMCP("MyServer")

@mcp.tool
def analyze(data: str) -> dict:
    return {"result": f"Analyzed: {data}"}

# Create the ASGI app
mcp_app = mcp.http_app(path='/mcp')

# Create a Starlette app and mount the MCP server
app = Starlette(
    routes=[
        Mount("/mcp-server", app=mcp_app),
        # Add other routes as needed
    ],
    lifespan=mcp_app.lifespan,
)
```

The MCP endpoint will be available at `/mcp-server/mcp/` of the resulting Starlette app.

<Warning>
  For Streamable HTTP transport, you **must** pass the lifespan context from the FastMCP app to the resulting Starlette app, as nested lifespans are not recognized. Otherwise, the FastMCP server's session manager will not be properly initialized.
</Warning>

#### Nested Mounts

You can create complex routing structures by nesting mounts:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

# Create your FastMCP server
mcp = FastMCP("MyServer")

# Create the ASGI app
mcp_app = mcp.http_app(path='/mcp')

# Create nested application structure
inner_app = Starlette(routes=[Mount("/inner", app=mcp_app)])
app = Starlette(
    routes=[Mount("/outer", app=inner_app)],
    lifespan=mcp_app.lifespan,
)
```

In this setup, the MCP server is accessible at the `/outer/inner/mcp/` path.

### FastAPI Integration

For FastAPI-specific integration patterns including both mounting MCP servers into FastAPI apps and generating MCP servers from FastAPI apps, see the [FastAPI Integration guide](/integrations/fastapi).

Here's a quick example showing how to add MCP to an existing FastAPI application:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastapi import FastAPI
from fastmcp import FastMCP

# Create your MCP server
mcp = FastMCP("API Tools")

@mcp.tool
def query_database(query: str) -> dict:
    """Run a database query"""
    return {"result": "data"}

# Create the MCP ASGI app with path="/" since we'll mount at /mcp
mcp_app = mcp.http_app(path="/")

# Create FastAPI app with MCP lifespan (required for session management)
api = FastAPI(lifespan=mcp_app.lifespan)

@api.get("/api/status")
def status():
    return {"status": "ok"}

# Mount MCP at /mcp
api.mount("/mcp", mcp_app)

# Run with: uvicorn app:api --host 0.0.0.0 --port 8000
```

Your existing API remains at `http://localhost:8000/api` while MCP is available at `http://localhost:8000/mcp`.

<Warning>
  Just like with Starlette, you **must** pass the lifespan from the MCP app to FastAPI. Without this, the session manager won't initialize properly and requests will fail.
</Warning>

## Mounting Authenticated Servers

<VersionBadge />

<Tip>
  This section only applies if you're **mounting an OAuth-protected FastMCP server under a path prefix** (like `/api`) inside another application using `Mount()`.

  If you're deploying your FastMCP server at root level without any `Mount()` prefix, the well-known routes are automatically included in `mcp.http_app()` and you don't need to do anything special.
</Tip>

OAuth specifications (RFC 8414 and RFC 9728) require discovery metadata to be accessible at well-known paths under the root level of your domain. When you mount an OAuth-protected FastMCP server under a path prefix like `/api`, this creates a routing challenge: your operational OAuth endpoints move under the prefix, but discovery endpoints must remain at the root.

<Warning>
  **Common Mistakes to Avoid:**

  1. **Forgetting to mount `.well-known` routes at root** - FastMCP cannot do this automatically when your server is mounted under a path prefix. You must explicitly mount well-known routes at the root level.

  2. **Including mount prefix in both base\_url AND mcp\_path** - The mount prefix (like `/api`) should only be in `base_url`, not in `mcp_path`. Otherwise you'll get double paths.

     ✅ **Correct:**

     ```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
     base_url = "http://localhost:8000/api"
     mcp_path = "/mcp"
     # Result: /api/mcp
     ```

     ❌ **Wrong:**

     ```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
     base_url = "http://localhost:8000/api"
     mcp_path = "/api/mcp"
     # Result: /api/api/mcp (double prefix!)
     ```

  Follow the configuration instructions below to set up mounting correctly.
</Warning>

<Warning>
  **CORS Middleware Conflicts:**

  If you're integrating FastMCP into an existing application with its own CORS middleware, be aware that layering CORS middleware can cause conflicts (such as 404 errors on `.well-known` routes or OPTIONS requests).

  FastMCP and the MCP SDK already handle CORS for OAuth routes. If you need CORS on your own application routes, consider using the sub-app pattern: mount FastMCP and your routes as separate apps, each with their own middleware, rather than adding application-wide CORS middleware.
</Warning>

### Route Types

OAuth-protected MCP servers expose two categories of routes:

**Operational routes** handle the OAuth flow and MCP protocol:

* `/authorize` - OAuth authorization endpoint
* `/token` - Token exchange endpoint
* `/auth/callback` - OAuth callback handler
* `/mcp` - MCP protocol endpoint

**Discovery routes** provide metadata for OAuth clients:

* `/.well-known/oauth-authorization-server` - Authorization server metadata
* `/.well-known/oauth-protected-resource/*` - Protected resource metadata

When you mount your MCP app under a prefix, operational routes move with it, but discovery routes must stay at root level for RFC compliance.

### Configuration Parameters

Three parameters control where routes are located and how they combine:

**`base_url`** tells clients where to find operational endpoints. This includes any Starlette `Mount()` path prefix (e.g., `/api`):

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
base_url="http://localhost:8000/api"  # Includes mount prefix
```

**`mcp_path`** is the internal FastMCP endpoint path, which gets appended to `base_url`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp_path="/mcp"  # Internal MCP path, NOT the mount prefix
```

**`issuer_url`** (optional) controls the authorization server identity for OAuth discovery. Defaults to `base_url`.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Usually not needed - just set base_url and it works
issuer_url="http://localhost:8000"  # Only if you want root-level discovery
```

When `issuer_url` has a path (either explicitly or by defaulting from `base_url`), FastMCP creates path-aware discovery routes per RFC 8414. For example, if `base_url` is `http://localhost:8000/api`, the authorization server metadata will be at `/.well-known/oauth-authorization-server/api`.

**Key Invariant:** `base_url + mcp_path = actual externally-accessible MCP URL`

Example:

* `base_url`: `http://localhost:8000/api` (mount prefix `/api`)
* `mcp_path`: `/mcp` (internal path)
* Result: `http://localhost:8000/api/mcp` (final MCP endpoint)

Note that the mount prefix (`/api` from `Mount("/api", ...)`) goes in `base_url`, while `mcp_path` is just the internal MCP route. Don't include the mount prefix in both places or you'll get `/api/api/mcp`.

### Mounting Strategy

When mounting an OAuth-protected server under a path prefix, declare your URLs upfront to make the relationships clear:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from starlette.applications import Starlette
from starlette.routing import Mount

# Define the routing structure
ROOT_URL = "http://localhost:8000"
MOUNT_PREFIX = "/api"
MCP_PATH = "/mcp"
```

Create the auth provider with `base_url`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
auth = GitHubProvider(
    client_id="your-client-id",
    client_secret="your-client-secret",
    base_url=f"{ROOT_URL}{MOUNT_PREFIX}",  # Operational endpoints under prefix
    # issuer_url defaults to base_url - path-aware discovery works automatically
)
```

Create the MCP app, which generates operational routes at the specified path:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp = FastMCP("Protected Server", auth=auth)
mcp_app = mcp.http_app(path=MCP_PATH)
```

Retrieve the discovery routes from the auth provider. The `mcp_path` argument should match the path used when creating the MCP app:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
well_known_routes = auth.get_well_known_routes(mcp_path=MCP_PATH)
```

Finally, mount everything in the Starlette app with discovery routes at root and the MCP app under the prefix:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
app = Starlette(
    routes=[
        *well_known_routes,  # Discovery routes at root level
        Mount(MOUNT_PREFIX, app=mcp_app),  # Operational routes under prefix
    ],
    lifespan=mcp_app.lifespan,
)
```

This configuration produces the following URL structure:

* MCP endpoint: `http://localhost:8000/api/mcp`
* OAuth authorization: `http://localhost:8000/api/authorize`
* OAuth callback: `http://localhost:8000/api/auth/callback`
* Authorization server metadata: `http://localhost:8000/.well-known/oauth-authorization-server/api`
* Protected resource metadata: `http://localhost:8000/.well-known/oauth-protected-resource/api/mcp`

Both discovery endpoints use path-aware URLs per RFC 8414 and RFC 9728, matching the `base_url` path.

### Complete Example

Here's a complete working example showing all the pieces together:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from starlette.applications import Starlette
from starlette.routing import Mount
import uvicorn

# Define routing structure
ROOT_URL = "http://localhost:8000"
MOUNT_PREFIX = "/api"
MCP_PATH = "/mcp"

# Create OAuth provider
auth = GitHubProvider(
    client_id="your-client-id",
    client_secret="your-client-secret",
    base_url=f"{ROOT_URL}{MOUNT_PREFIX}",
    # issuer_url defaults to base_url - path-aware discovery works automatically
)

# Create MCP server
mcp = FastMCP("Protected Server", auth=auth)

@mcp.tool
def analyze(data: str) -> dict:
    return {"result": f"Analyzed: {data}"}

# Create MCP app
mcp_app = mcp.http_app(path=MCP_PATH)

# Get discovery routes for root level
well_known_routes = auth.get_well_known_routes(mcp_path=MCP_PATH)

# Assemble the application
app = Starlette(
    routes=[
        *well_known_routes,
        Mount(MOUNT_PREFIX, app=mcp_app),
    ],
    lifespan=mcp_app.lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

For more details on OAuth authentication, see the [Authentication guide](/servers/auth).

## Production Deployment

### Running with Uvicorn

When deploying to production, you'll want to optimize your server for performance and reliability. Uvicorn provides several options to improve your server's capabilities:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Run with basic configuration
uvicorn app:app --host 0.0.0.0 --port 8000

# Run with multiple workers for production (requires stateless mode - see below)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Horizontal Scaling

<VersionBadge />

When deploying FastMCP behind a load balancer or running multiple server instances, you need to understand how the HTTP transport handles sessions and configure your server appropriately.

#### Understanding Sessions

By default, FastMCP's Streamable HTTP transport maintains server-side sessions. Sessions enable stateful MCP features like [elicitation](/servers/elicitation) and [sampling](/servers/sampling), where the server needs to maintain context across multiple requests from the same client.

This works perfectly for single-instance deployments. However, sessions are stored in memory on each server instance, which creates challenges when scaling horizontally.

#### Without Stateless Mode

When running multiple server instances behind a load balancer (Traefik, nginx, HAProxy, Kubernetes, etc.), requests from the same client may be routed to different instances:

1. Client connects to Instance A → session created on Instance A
2. Next request routes to Instance B → session doesn't exist → **request fails**

You might expect sticky sessions (session affinity) to solve this, but they don't work reliably with MCP clients.

<Warning>
  **Why sticky sessions don't work:** Most MCP clients—including Cursor and Claude Code—use `fetch()` internally and don't properly forward `Set-Cookie` headers. Without cookies, load balancers can't identify which instance should handle subsequent requests. This is a limitation in how these clients implement HTTP, not something you can fix with load balancer configuration.
</Warning>

#### Enabling Stateless Mode

For horizontally scaled deployments, enable stateless HTTP mode. In stateless mode, each request creates a fresh transport context, eliminating the need for session affinity entirely.

**Option 1: Via constructor**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("My Server", stateless_http=True)

@mcp.tool
def process(data: str) -> str:
    return f"Processed: {data}"

app = mcp.http_app()
```

**Option 2: Via `run()`**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
if __name__ == "__main__":
    mcp.run(transport="http", stateless_http=True)
```

**Option 3: Via environment variable**

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
FASTMCP_STATELESS_HTTP=true uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables

Production deployments should never hardcode sensitive information like API keys or authentication tokens. Instead, use environment variables to configure your server at runtime. This keeps your code secure and makes it easy to deploy the same code to different environments with different configurations.

Here's an example using bearer token authentication (though OAuth is recommended for production):

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import os
from fastmcp import FastMCP
from fastmcp.server.auth import BearerTokenAuth

# Read configuration from environment
auth_token = os.environ.get("MCP_AUTH_TOKEN")
if auth_token:
    auth = BearerTokenAuth(token=auth_token)
    mcp = FastMCP("Production Server", auth=auth)
else:
    mcp = FastMCP("Production Server")

app = mcp.http_app()
```

Deploy with your secrets safely stored in environment variables:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
MCP_AUTH_TOKEN=secret uvicorn app:app --host 0.0.0.0 --port 8000
```

### OAuth Token Security

<VersionBadge />

If you're using the [OAuth Proxy](/servers/auth/oauth-proxy), FastMCP issues its own JWT tokens to clients instead of forwarding upstream provider tokens. This maintains proper OAuth 2.0 token boundaries.

**Default Behavior (Development Only):**

By default, FastMCP automatically manages cryptographic keys:

* **Mac/Windows**: Keys are generated and stored in your system keyring, surviving server restarts. Suitable **only** for development and local testing.
* **Linux**: Keys are ephemeral (random salt at startup), so tokens are invalidated on restart.

This automatic approach is convenient for development but not suitable for production deployments.

**For Production:**

Production requires explicit key management to ensure tokens survive restarts and can be shared across multiple server instances. This requires the following two things working together:

1. **Explicit JWT signing key** for signing tokens issued to clients
2. **Persistent network-accessible storage** for upstream tokens (wrapped in `FernetEncryptionWrapper` to encrypt sensitive data at rest)

**Configuration:**

Add two parameters to your auth provider:

```python {8-12} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from key_value.aio.stores.redis import RedisStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from cryptography.fernet import Fernet

auth = GitHubProvider(
    client_id=os.environ["GITHUB_CLIENT_ID"],
    client_secret=os.environ["GITHUB_CLIENT_SECRET"],
    jwt_signing_key=os.environ["JWT_SIGNING_KEY"],
    client_storage=FernetEncryptionWrapper(
        key_value=RedisStore(host="redis.example.com", port=6379),
        fernet=Fernet(os.environ["STORAGE_ENCRYPTION_KEY"])
    ),
    base_url="https://your-server.com"  # use HTTPS
)
```

Both parameters are required for production. Without an explicit signing key, keys are signed using a key derived from the client\_secret, which will cause invalidation upon rotation of the client secret. Without persistent storage, tokens are local to the server and won't be trusted across hosts. **Wrap your storage backend in `FernetEncryptionWrapper` to encrypt sensitive OAuth tokens at rest** - without encryption, tokens are stored in plaintext.

For more details on the token architecture and key management, see [OAuth Proxy Key and Storage Management](/servers/auth/oauth-proxy#key-and-storage-management).

## Reverse Proxy (nginx)

In production, you'll typically run your FastMCP server behind a reverse proxy like nginx. A reverse proxy provides TLS termination, domain-based routing, static file serving, and an additional layer of security between the internet and your application.

### Running FastMCP as a Linux Service

Before configuring nginx, you need your FastMCP server running as a background service. A systemd unit file ensures your server starts automatically and restarts on failure.

Create a file at `/etc/systemd/system/fastmcp.service`:

```ini theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
[Unit]
Description=FastMCP Server
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/fastmcp
ExecStart=/opt/fastmcp/.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
Environment="PATH=/opt/fastmcp/.venv/bin"

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
sudo systemctl daemon-reload
sudo systemctl enable fastmcp
sudo systemctl start fastmcp
```

This assumes your ASGI application is in `/opt/fastmcp/app.py` with a virtual environment at `/opt/fastmcp/.venv`. Adjust paths to match your deployment layout.

### nginx Configuration

FastMCP's Streamable HTTP transport uses Server-Sent Events (SSE) for streaming responses. This requires specific nginx settings to prevent buffering from breaking the event stream.

Create a site configuration at `/etc/nginx/sites-available/fastmcp`:

```nginx theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
server {
    listen 80;
    server_name mcp.example.com;

    # Redirect HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name mcp.example.com;

    ssl_certificate /etc/letsencrypt/live/mcp.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mcp.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Required for SSE (Server-Sent Events) streaming
        proxy_buffering off;
        proxy_cache off;

        # Allow long-lived connections for streaming responses
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

Enable the site and reload nginx:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
sudo ln -s /etc/nginx/sites-available/fastmcp /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

Your FastMCP server is now accessible at `https://mcp.example.com/mcp`.

<Warning>
  **SSE buffering is the most common issue.** If clients connect but never receive streaming responses (progress updates, tool results), verify that `proxy_buffering off` is set. Without it, nginx buffers the entire SSE stream and delivers it only when the connection closes, which breaks real-time communication.
</Warning>

### Key Considerations

When deploying FastMCP behind a reverse proxy, keep these points in mind:

* **Disable buffering**: SSE requires `proxy_buffering off` so events reach clients immediately. This is the single most important setting.
* **Increase timeouts**: The default nginx `proxy_read_timeout` is 60 seconds. Long-running MCP tools will cause the connection to drop. Set timeouts to at least 300 seconds, or higher if your tools run longer. For tools that may exceed any timeout, use [SSE Polling](#sse-polling-for-long-running-operations) to gracefully handle proxy disconnections.
* **Use HTTP/1.1**: Set `proxy_http_version 1.1` and `proxy_set_header Connection ''` to enable keep-alive connections between nginx and your server. Clearing the `Connection` header prevents clients from sending `Connection: close` to your upstream, which would break SSE streams. Both settings are required for proper SSE support.
* **Forward headers**: Pass `X-Forwarded-For` and `X-Forwarded-Proto` so your FastMCP server can determine the real client IP and protocol. This is important for logging and for OAuth redirect URLs.
* **TLS termination**: Let nginx handle TLS certificates (e.g., via Let's Encrypt with Certbot). Your FastMCP server can then run on plain HTTP internally.

### Mounting Under a Path Prefix

If you want your MCP server available at a subpath like `https://example.com/api/mcp` instead of at the root domain, adjust the nginx `location` block:

```nginx theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
location /api/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_http_version 1.1;
    proxy_set_header Connection '';
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Required for SSE streaming
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
}
```

Note the trailing `/` on both `location /api/` and `proxy_pass http://127.0.0.1:8000/` — this ensures nginx strips the `/api` prefix before forwarding to your server. If you're using OAuth authentication with a mount prefix, see [Mounting Authenticated Servers](#mounting-authenticated-servers) for additional configuration.

## Testing Your Deployment

Once your server is deployed, you'll need to verify it's accessible and functioning correctly. For comprehensive testing strategies including connectivity tests, client testing, and authentication testing, see the [Testing Your Server](/development/tests) guide.

## Hosting Your Server

This guide has shown you how to create an HTTP-accessible MCP server, but you'll still need a hosting provider to make it available on the internet. Your FastMCP server can run anywhere that supports Python web applications:

* **Cloud VMs** (AWS EC2, Google Compute Engine, Azure VMs)
* **Container platforms** (Cloud Run, Container Instances, ECS)
* **Platform-as-a-Service** (Railway, Render, Vercel)
* **Edge platforms** (Cloudflare Workers)
* **Kubernetes clusters** (self-managed or managed)

The key requirements are Python 3.10+ support and the ability to expose an HTTP port. Most providers will require you to package your server (requirements.txt, Dockerfile, etc.) according to their deployment format. For managed, zero-configuration deployment, see [Prefect Horizon](/deployment/prefect-horizon).


# Prefect Horizon
Source: https://gofastmcp.com/deployment/prefect-horizon

The MCP platform from the FastMCP team

[Prefect Horizon](https://www.prefect.io/horizon) is a platform for deploying and managing MCP servers. Built by the FastMCP team at [Prefect](https://www.prefect.io), Horizon provides managed hosting, authentication, access control, and a registry of MCP capabilities.

Horizon includes a **free personal tier for FastMCP users**, making it the fastest way to get a secure, production-ready server URL with built-in OAuth authentication.

<Info>
  Horizon is free for personal projects. Enterprise governance features are available for teams deploying to thousands of users.
</Info>

## The Platform

Horizon is organized into four integrated pillars:

* **Deploy**: Managed hosting with CI/CD, scaling, monitoring, and rollbacks. Push code and get a live, governed endpoint in 60 seconds.
* **Registry**: A central catalog of MCP servers across your organization—first-party, third-party, and curated remix servers composed from multiple sources.
* **Gateway**: Role-based access control, authentication, and audit logs. Define what agents can see and do at the tool level.
* **Agents**: A permissioned chat interface for interacting with any MCP server or curated combination of servers.

This guide focuses on **Horizon Deploy**, the managed hosting layer that gives you the fastest path from a FastMCP server to a production URL.

## Prerequisites

To use Horizon, you'll need a [GitHub](https://github.com) account and a GitHub repo containing a FastMCP server. If you don't have one yet, Horizon can create a starter repo for you during onboarding.

Your repo can be public or private, but must include at least a Python file containing a FastMCP server instance.

<Tip>
  To verify your file is compatible with Horizon, run `fastmcp inspect <file.py:server_object>` to see what Horizon will see when it runs your server.
</Tip>

If you have a `requirements.txt` or `pyproject.toml` in the repo, Horizon will automatically detect your server's dependencies and install them. Your file *can* have an `if __name__ == "__main__"` block, but it will be ignored by Horizon.

For example, a minimal server file might look like:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"
```

## Getting Started

There are just three steps to deploying a server to Horizon:

### Step 1: Select a Repository

Visit [horizon.prefect.io](https://horizon.prefect.io) and sign in with your GitHub account. Connect your GitHub account to grant Horizon access to your repositories, then select the repo you want to deploy.

<img alt="Horizon repository selection" />

### Step 2: Configure Your Server

Next, you'll configure how Horizon should build and deploy your server.

<img alt="Horizon server configuration" />

The configuration screen lets you specify:

* **Server name**: A unique name for your server. This determines your server's URL.
* **Description**: A brief description of what your server does.
* **Entrypoint**: The Python file containing your FastMCP server (e.g., `main.py`). This field has the same syntax as the `fastmcp run` command—use `main.py:mcp` to specify a specific object in the file.
* **Authentication**: When enabled, only authenticated users in your organization can connect. Horizon handles all the OAuth complexity for you.

Horizon will automatically detect your server's Python dependencies from either a `requirements.txt` or `pyproject.toml` file.

### Step 3: Deploy and Connect

Click **Deploy Server** and Horizon will clone your repository, build your server, and deploy it to a unique URL—typically in under 60 seconds.

<img alt="Horizon deployment view showing live server" />

Once deployed, your server is accessible at a URL like:

```
https://your-server-name.fastmcp.app/mcp
```

Horizon monitors your repo and redeploys automatically whenever you push to `main`. It also builds preview deployments for every PR, so you can test changes before they go live.

## Testing Your Server

Horizon provides two ways to verify your server is working before connecting external clients.

### Inspector

The Inspector gives you a structured view of everything your server exposes—tools, resources, and prompts. You can click any tool, fill in the inputs, execute it, and see the output. This is useful for systematically validating each capability and debugging specific behaviors.

### ChatMCP

For quick end-to-end testing, ChatMCP lets you interact with your server conversationally. It uses a fast model optimized for rapid iteration—you can verify the server works, test tool calls in context, and confirm the overall behavior before sharing it with others.

<img alt="Horizon ChatMCP interface" />

ChatMCP is designed for testing, not as a daily work environment. Once you've confirmed your server works, you can copy connection snippets for Claude Desktop, Cursor, Claude Code, and other MCP clients—or use the FastMCP client library to connect programmatically.

## Horizon Agents

Beyond testing individual servers, Horizon lets you create **Agents**—chat interfaces backed by one or more MCP servers. While ChatMCP tests a single server, Agents let you compose capabilities from multiple servers into a unified experience.

<img alt="Horizon Agent configuration" />

To create an agent:

1. Navigate to **Agents** in the sidebar
2. Click **Create Agent** and give it a name and description
3. Add MCP servers to the agent—these can be servers you've deployed to Horizon or external servers in the registry

Once configured, you can chat with your agent directly in Horizon:

<img alt="Chatting with a Horizon Agent" />

Agents are useful for creating purpose-built interfaces that combine tools from different servers. For example, you might create an agent that has access to both your company's internal data server and a general-purpose utilities server.


# Running Your Server
Source: https://gofastmcp.com/deployment/running-server

Learn how to run your FastMCP server locally for development and testing

FastMCP servers can be run in different ways depending on your needs. This guide focuses on running servers locally for development and testing. For production deployment to a URL, see the [HTTP Deployment](/deployment/http) guide.

## The `run()` Method

Every FastMCP server needs to be started to accept connections. The simplest way to run a server is by calling the `run()` method on your FastMCP instance. This method starts the server and blocks until it's stopped, handling all the connection management for you.

<Tip>
  For maximum compatibility, it's best practice to place the `run()` call within an `if __name__ == "__main__":` block. This ensures the server starts only when the script is executed directly, not when imported as a module.
</Tip>

```python {9-10} my_server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP(name="MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

You can now run this MCP server by executing `python my_server.py`.

## Transport Protocols

MCP servers communicate with clients through different transport protocols. Think of transports as the "language" your server speaks to communicate with clients. FastMCP supports three main transport protocols, each designed for specific use cases and deployment scenarios.

The choice of transport determines how clients connect to your server, what network capabilities are available, and how many clients can connect simultaneously. Understanding these transports helps you choose the right approach for your application.

### STDIO Transport (Default)

STDIO (Standard Input/Output) is the default transport for FastMCP servers. When you call `run()` without arguments, your server uses STDIO transport. This transport communicates through standard input and output streams, making it perfect for command-line tools and desktop applications like Claude Desktop.

With STDIO transport, the client spawns a new server process for each session and manages its lifecycle. The server reads MCP messages from stdin and writes responses to stdout. This is why STDIO servers don't stay running - they're started on-demand by the client.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()  # Uses STDIO transport by default
```

STDIO is ideal for:

* Local development and testing
* Claude Desktop integration
* Command-line tools
* Single-user applications

### HTTP Transport (Streamable)

HTTP transport turns your MCP server into a web service accessible via a URL. This transport uses the Streamable HTTP protocol, which allows clients to connect over the network. Unlike STDIO where each client gets its own process, an HTTP server can handle multiple clients simultaneously.

The Streamable HTTP protocol provides full bidirectional communication between client and server, supporting all MCP operations including streaming responses. This makes it the recommended choice for network-based deployments.

To use HTTP transport, specify it in the `run()` method along with networking options:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    # Start an HTTP server on port 8000
    mcp.run(transport="http", host="127.0.0.1", port=8000)
```

Your server is now accessible at `http://localhost:8000/mcp`. This URL is the MCP endpoint that clients will connect to. HTTP transport enables:

* Network accessibility
* Multiple concurrent clients
* Integration with web infrastructure
* Remote deployment capabilities

For production HTTP deployment with authentication and advanced configuration, see the [HTTP Deployment](/deployment/http) guide.

### SSE Transport (Legacy)

Server-Sent Events (SSE) transport was the original HTTP-based transport for MCP. While still supported for backward compatibility, it has limitations compared to the newer Streamable HTTP transport. SSE only supports server-to-client streaming, making it less efficient for bidirectional communication.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
if __name__ == "__main__":
    # SSE transport - use HTTP instead for new projects
    mcp.run(transport="sse", host="127.0.0.1", port=8000)
```

We recommend using HTTP transport instead of SSE for all new projects. SSE remains available only for compatibility with older clients that haven't upgraded to Streamable HTTP.

### Choosing the Right Transport

Each transport serves different needs. STDIO is perfect when you need simple, local execution - it's what Claude Desktop and most command-line tools expect. HTTP transport is essential when you need network access, want to serve multiple clients, or plan to deploy your server remotely. SSE exists only for backward compatibility and shouldn't be used in new projects.

Consider your deployment scenario: Are you building a tool for local use? STDIO is your best choice. Need a centralized service that multiple clients can access? HTTP transport is the way to go.

## The FastMCP CLI

FastMCP provides a powerful command-line interface for running servers without modifying the source code. The CLI can automatically find and run your server with different transports, manage dependencies, and handle development workflows:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py
```

The CLI automatically finds a FastMCP instance in your file (named `mcp`, `server`, or `app`) and runs it with the specified options. This is particularly useful for testing different transports or configurations without changing your code.

### Dependency Management

The CLI integrates with `uv` to manage Python environments and dependencies:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Run with a specific Python version
fastmcp run server.py --python 3.11

# Run with additional packages
fastmcp run server.py --with pandas --with numpy

# Run with dependencies from a requirements file
fastmcp run server.py --with-requirements requirements.txt

# Combine multiple options
fastmcp run server.py --python 3.10 --with httpx --transport http

# Run within a specific project directory
fastmcp run server.py --project /path/to/project
```

<Note>
  When using `--python`, `--with`, `--project`, or `--with-requirements`, the server runs via `uv run` subprocess instead of using your local environment.
</Note>

### Passing Arguments to Servers

When servers accept command line arguments (using argparse, click, or other libraries), you can pass them after `--`:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run config_server.py -- --config config.json
fastmcp run database_server.py -- --database-path /tmp/db.sqlite --debug
```

This is useful for servers that need configuration files, database paths, API keys, or other runtime options.

For more CLI features including development mode with the MCP Inspector, see the [CLI documentation](/cli/running).

### Auto-Reload for Development

<VersionBadge />

During development, you can use the `--reload` flag to automatically restart your server when source files change:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py --reload
```

The server watches for changes to Python files in the current directory and restarts automatically when you save changes. This provides a fast feedback loop during development without manually stopping and starting the server.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Watch specific directories for changes
fastmcp run server.py --reload --reload-dir ./src --reload-dir ./lib

# Combine with other options
fastmcp run server.py --reload --transport http --port 8080
```

<Note>
  Auto-reload uses stateless mode to enable seamless restarts. For stdio transport, this is fully featured. For HTTP transport, some bidirectional features like elicitation are not available during reload mode.
</Note>

SSE transport does not support auto-reload due to session limitations. Use HTTP transport instead if you need both network access and auto-reload.

### Async Usage

FastMCP servers are built on async Python, but the framework provides both synchronous and asynchronous APIs to fit your application's needs. The `run()` method we've been using is actually a synchronous wrapper around the async server implementation.

For applications that are already running in an async context, FastMCP provides the `run_async()` method:

```python {10-12} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
import asyncio

mcp = FastMCP(name="MyServer")

@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

async def main():
    # Use run_async() in async contexts
    await mcp.run_async(transport="http", port=8000)

if __name__ == "__main__":
    asyncio.run(main())
```

<Warning>
  The `run()` method cannot be called from inside an async function because it creates its own async event loop internally. If you attempt to call `run()` from inside an async function, you'll get an error about the event loop already running.

  Always use `run_async()` inside async functions and `run()` in synchronous contexts.
</Warning>

Both `run()` and `run_async()` accept the same transport arguments, so all the examples above apply to both methods.

## Custom Routes

When using HTTP transport, you might want to add custom web endpoints alongside your MCP server. This is useful for health checks, status pages, or simple APIs. FastMCP lets you add custom routes using the `@custom_route` decorator:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

mcp = FastMCP("MyServer")

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

@mcp.tool
def process(data: str) -> str:
    return f"Processed: {data}"

if __name__ == "__main__":
    mcp.run(transport="http")  # Health check at http://localhost:8000/health
```

Custom routes are served by the same web server as your MCP endpoint. They're available at the root of your domain while the MCP endpoint is at `/mcp/`. For more complex web applications, consider [mounting your MCP server into a FastAPI or Starlette app](/deployment/http#integration-with-web-frameworks).

## Alternative Initialization Patterns

The `if __name__ == "__main__"` pattern works well for standalone scripts, but some deployment scenarios require different approaches. FastMCP handles these cases automatically.

### CLI-Only Servers

When using the FastMCP CLI, you don't need the `if __name__` block at all. The CLI will find your FastMCP instance and run it:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# server.py
from fastmcp import FastMCP

mcp = FastMCP("MyServer")  # CLI looks for 'mcp', 'server', or 'app'

@mcp.tool
def process(data: str) -> str:
    return f"Processed: {data}"

# No if __name__ block needed - CLI will find and run 'mcp'
```

### ASGI Applications

For ASGI deployment (running with Uvicorn or similar), you'll want to create an ASGI application object. This approach is common in production deployments where you need more control over the server configuration:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# app.py
from fastmcp import FastMCP

def create_app():
    mcp = FastMCP("MyServer")
    
    @mcp.tool
    def process(data: str) -> str:
        return f"Processed: {data}"
    
    return mcp.http_app()

app = create_app()  # Uvicorn will use this
```

See the [HTTP Deployment](/deployment/http) guide for more ASGI deployment patterns.
