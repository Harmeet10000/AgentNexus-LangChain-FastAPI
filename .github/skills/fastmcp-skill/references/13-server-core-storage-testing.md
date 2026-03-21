# Server Core, Storage, and Testing

Source lines: 20825-22016 from the original FastMCP documentation dump.

Server instructions, storage backends, background tasks, telemetry, and testing guidance.

---

# The FastMCP Server
Source: https://gofastmcp.com/servers/server

The core FastMCP server class for building MCP applications

The `FastMCP` class is the central piece of every FastMCP application. It acts as the container for your tools, resources, and prompts, managing communication with MCP clients and orchestrating the entire server lifecycle.

## Creating a Server

Instantiate a server by providing a name that identifies it in client applications and logs. You can also provide instructions that help clients understand the server's purpose.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP(name="MyAssistantServer")

# Instructions help clients understand how to interact with the server
mcp_with_instructions = FastMCP(
    name="HelpfulAssistant",
    instructions="""
        This server provides data analysis tools.
        Call get_average() to analyze numerical data.
    """,
)
```

The `FastMCP` constructor accepts several configuration options. The most commonly used parameters control server identity, authentication, and component behavior.

<Card icon="code" title="FastMCP Constructor Parameters">
  <ParamField type="str">
    A human-readable name for your server
  </ParamField>

  <ParamField type="str | None">
    Description of how to interact with this server. These instructions help clients understand the server's purpose and available functionality
  </ParamField>

  <ParamField type="str | None">
    Version string for your server. If not provided, defaults to the FastMCP library version
  </ParamField>

  <ParamField type="str | None">
    <VersionBadge />

    URL to a website with more information about your server. Displayed in client applications
  </ParamField>

  <ParamField type="list[Icon] | None">
    <VersionBadge />

    List of icon representations for your server. Icons help users visually identify your server in client applications. See [Icons](/servers/icons) for detailed examples
  </ParamField>

  <ParamField type="OAuthProvider | TokenVerifier | None">
    Authentication provider for securing HTTP-based transports. See [Authentication](/servers/auth/authentication) for configuration options
  </ParamField>

  <ParamField type="Lifespan | AsyncContextManager | None">
    Server-level setup and teardown logic. See [Lifespans](/servers/lifespan) for composable lifespans
  </ParamField>

  <ParamField type="list[Tool | Callable] | None">
    A list of tools (or functions to convert to tools) to add to the server. In some cases, providing tools programmatically may be more convenient than using the `@mcp.tool` decorator
  </ParamField>

  <ParamField type="list[Transform] | None">
    <VersionBadge />

    Server-level [transforms](/servers/transforms/transforms) to apply to all components. Transforms modify how tools, resources, and prompts are presented to clients — for example, [search transforms](/servers/transforms/tool-search) replace large catalogs with on-demand discovery, and [CodeMode](/servers/transforms/code-mode) lets LLMs write scripts that chain tool calls in a sandbox
  </ParamField>

  <ParamField type="Literal[&#x22;warn&#x22;, &#x22;error&#x22;, &#x22;replace&#x22;, &#x22;ignore&#x22;]">
    How to handle duplicate component registrations
  </ParamField>

  <ParamField type="bool">
    <VersionBadge />

    Controls how tool input parameters are validated. When `False` (default), FastMCP uses Pydantic's flexible validation that coerces compatible inputs (e.g., `"10"` to `10` for int parameters). When `True`, uses the MCP SDK's JSON Schema validation to validate inputs against the exact schema before passing them to your function, rejecting any type mismatches. The default mode improves compatibility with LLM clients while maintaining type safety. See [Input Validation Modes](/servers/tools#input-validation-modes) for details
  </ParamField>

  <ParamField type="int | None">
    <VersionBadge />

    Maximum number of items per page for list operations (`tools/list`, `resources/list`, etc.). When `None` (default), all results are returned in a single response. When set, responses are paginated and include a `nextCursor` for fetching additional pages. See [Pagination](/servers/pagination) for details
  </ParamField>
</Card>

## Components

FastMCP servers expose three types of components to clients. Each type serves a distinct purpose in the MCP protocol.

### Tools

Tools are functions that clients can invoke to perform actions or access external systems. They're the primary way clients interact with your server's capabilities.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b
```

See [Tools](/servers/tools) for detailed documentation.

### Resources

Resources expose data that clients can read. Unlike tools, resources are passive data sources that clients pull from rather than invoke.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.resource("data://config")
def get_config() -> dict:
    """Provides the application configuration."""
    return {"theme": "dark", "version": "1.0"}
```

See [Resources](/servers/resources) for detailed documentation.

### Resource Templates

Resource templates are parameterized resources. The client provides values for template parameters in the URI, and the server returns data specific to those parameters.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Retrieves a user's profile by ID."""
    return {"id": user_id, "name": f"User {user_id}", "status": "active"}
```

See [Resource Templates](/servers/resources#resource-templates) for detailed documentation.

### Prompts

Prompts are reusable message templates that guide LLM interactions. They help establish consistent patterns for how clients should frame requests.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.prompt
def analyze_data(data_points: list[float]) -> str:
    """Creates a prompt asking for analysis of numerical data."""
    formatted_data = ", ".join(str(point) for point in data_points)
    return f"Please analyze these data points: {formatted_data}"
```

See [Prompts](/servers/prompts) for detailed documentation.

## Tag-Based Filtering

<VersionBadge />

Tags let you categorize components and selectively expose them based on configurable include/exclude sets. This is useful for creating different views of your server for different environments or user types.

Components can be tagged when defined using the `tags` parameter. A component can have multiple tags, and filtering operates on tag membership.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool(tags={"public", "utility"})
def public_tool() -> str:
    return "This tool is public"

@mcp.tool(tags={"internal", "admin"})
def admin_tool() -> str:
    return "This tool is for admins only"
```

The filtering logic works as follows:

* **Enable with `only=True`**: Switches to allowlist mode — only components with at least one matching tag are exposed
* **Disable**: Components with any matching tag are hidden
* **Precedence**: Later calls override earlier ones, so call `disable` after `enable` to exclude from an allowlist

<Tip>
  To ensure a component is never exposed, you can set `enabled=False` on the component itself. See the component-specific documentation for details.
</Tip>

Configure tag-based filtering after creating your server.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Only expose components tagged with "public"
mcp = FastMCP()
mcp.enable(tags={"public"}, only=True)

# Hide components tagged as "internal" or "deprecated"
mcp = FastMCP()
mcp.disable(tags={"internal", "deprecated"})

# Combine both: show admin tools but hide deprecated ones
mcp = FastMCP()
mcp.enable(tags={"admin"}, only=True).disable(tags={"deprecated"})
```

This filtering applies to all component types (tools, resources, resource templates, and prompts) and affects both listing and access.

## Running the Server

FastMCP servers communicate with clients through transport mechanisms. Start your server by calling `mcp.run()`, typically within an `if __name__ == "__main__":` block. This pattern ensures compatibility with various MCP clients.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP(name="MyServer")

@mcp.tool
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    # Defaults to STDIO transport
    mcp.run()

    # Or use HTTP transport
    # mcp.run(transport="http", host="127.0.0.1", port=9000)
```

FastMCP supports several transports:

* **STDIO** (default): For local integrations and CLI tools
* **HTTP**: For web services using the Streamable HTTP protocol
* **SSE**: Legacy web transport (deprecated)

The server can also be run using the FastMCP CLI. For detailed information on transports and configuration, see the [Running Your Server](/deployment/running-server) guide.

## Custom Routes

When running with HTTP transport, you can add custom web routes alongside your MCP endpoint using the `@custom_route` decorator. This is useful for auxiliary endpoints like health checks.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

mcp = FastMCP("MyServer")

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

if __name__ == "__main__":
    mcp.run(transport="http")  # Health check at http://localhost:8000/health
```

Custom routes are served alongside your MCP endpoint and are useful for:

* Health check endpoints for monitoring
* Simple status or info endpoints
* Basic webhooks or callbacks

For more complex web applications, consider [mounting your MCP server into a FastAPI or Starlette app](/deployment/http#integration-with-web-frameworks).


# Storage Backends
Source: https://gofastmcp.com/servers/storage-backends

Configure persistent and distributed storage for caching and OAuth state management

<VersionBadge />

FastMCP uses pluggable storage backends for caching responses and managing OAuth state. By default, all storage is in-memory, which is perfect for development but doesn't persist across restarts. FastMCP includes support for multiple storage backends, and you can easily extend it with custom implementations.

<Tip>
  The storage layer is powered by **[py-key-value-aio](https://github.com/strawgate/py-key-value)**, an async key-value library maintained by a core FastMCP maintainer. This library provides a unified interface for multiple backends, making it easy to swap implementations based on your deployment needs.
</Tip>

## Available Backends

### In-Memory Storage

**Best for:** Development, testing, single-process deployments

In-memory storage is the default for all FastMCP storage needs. It's fast, requires no setup, and is perfect for getting started.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from key_value.aio.stores.memory import MemoryStore

# Used by default - no configuration needed
# But you can also be explicit:
cache_store = MemoryStore()
```

**Characteristics:**

* ✅ No setup required
* ✅ Very fast
* ❌ Data lost on restart
* ❌ Not suitable for multi-process deployments

### File Storage

**Best for:** Single-server production deployments, persistent caching

File storage persists data to the filesystem as one JSON file per key, allowing it to survive server restarts. This is the default backend for OAuth storage on Mac and Windows.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from pathlib import Path
from key_value.aio.stores.filetree import (
    FileTreeStore,
    FileTreeV1KeySanitizationStrategy,
    FileTreeV1CollectionSanitizationStrategy,
)
from fastmcp.server.middleware.caching import ResponseCachingMiddleware

storage_dir = Path("/var/cache/fastmcp")
store = FileTreeStore(
    data_directory=storage_dir,
    key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(storage_dir),
    collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(storage_dir),
)

# Persistent response cache
middleware = ResponseCachingMiddleware(cache_storage=store)
```

The sanitization strategies ensure keys and collection names are safe for the filesystem — alphanumeric names pass through as-is for readability, while special characters are hashed to prevent path traversal.

**Characteristics:**

* ✅ Data persists across restarts
* ✅ No external dependencies
* ✅ Human-readable files on disk
* ❌ Not suitable for distributed deployments
* ❌ Filesystem access required

### Redis

**Best for:** Distributed production deployments, shared caching across multiple servers

<Note>
  Redis support requires an optional dependency: `pip install 'py-key-value-aio[redis]'`
</Note>

Redis provides distributed caching and state management, ideal for production deployments with multiple server instances.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from key_value.aio.stores.redis import RedisStore
from fastmcp.server.middleware.caching import ResponseCachingMiddleware

# Distributed response cache
middleware = ResponseCachingMiddleware(
    cache_storage=RedisStore(host="redis.example.com", port=6379)
)
```

With authentication:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from key_value.aio.stores.redis import RedisStore

cache_store = RedisStore(
    host="redis.example.com",
    port=6379,
    password="your-redis-password"
)
```

For OAuth token storage:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import os
from fastmcp.server.auth.providers.github import GitHubProvider
from key_value.aio.stores.redis import RedisStore

auth = GitHubProvider(
    client_id=os.environ["GITHUB_CLIENT_ID"],
    client_secret=os.environ["GITHUB_CLIENT_SECRET"],
    base_url="https://your-server.com",
    jwt_signing_key=os.environ["JWT_SIGNING_KEY"],
    client_storage=RedisStore(host="redis.example.com", port=6379)
)
```

**Characteristics:**

* ✅ Distributed and highly available
* ✅ Fast in-memory performance
* ✅ Works across multiple server instances
* ✅ Built-in TTL support
* ❌ Requires Redis infrastructure
* ❌ Network latency vs local storage

### Other Backends from py-key-value-aio

The py-key-value-aio library includes additional implementations for various storage systems:

* **DynamoDB** - AWS distributed database
* **MongoDB** - NoSQL document store
* **Elasticsearch** - Distributed search and analytics
* **Memcached** - Distributed memory caching
* **RocksDB** - Embedded high-performance key-value store
* **Valkey** - Redis-compatible server

For configuration details on these backends, consult the [py-key-value-aio documentation](https://github.com/strawgate/py-key-value).

<Warning>
  Before using these backends in production, review the [py-key-value documentation](https://github.com/strawgate/py-key-value) to understand the maturity level and limitations of your chosen backend. Some backends may be in preview or have specific constraints that make them unsuitable for production use.
</Warning>

## Use Cases in FastMCP

### Server-Side OAuth Token Storage

The [OAuth Proxy](/servers/auth/oauth-proxy) and OAuth auth providers use storage for persisting OAuth client registrations and upstream tokens. **By default, storage is automatically encrypted using `FernetEncryptionWrapper`.** When providing custom storage, wrap it in `FernetEncryptionWrapper` to encrypt sensitive OAuth tokens at rest.

**Development (default behavior):**

By default, FastMCP automatically manages keys and storage based on your platform:

* **Mac/Windows**: Keys are auto-managed via system keyring, storage defaults to disk. Suitable **only** for development and local testing.
* **Linux**: Keys are ephemeral, storage defaults to memory.

No configuration needed:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.auth.providers.github import GitHubProvider

auth = GitHubProvider(
    client_id="your-id",
    client_secret="your-secret",
    base_url="https://your-server.com"
)
```

**Production:**

For production deployments, configure explicit keys and persistent network-accessible storage with encryption:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import os
from fastmcp.server.auth.providers.github import GitHubProvider
from key_value.aio.stores.redis import RedisStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from cryptography.fernet import Fernet

auth = GitHubProvider(
    client_id=os.environ["GITHUB_CLIENT_ID"],
    client_secret=os.environ["GITHUB_CLIENT_SECRET"],
    base_url="https://your-server.com",
    # Explicit JWT signing key (required for production)
    jwt_signing_key=os.environ["JWT_SIGNING_KEY"],
    # Encrypted persistent storage (required for production)
    client_storage=FernetEncryptionWrapper(
        key_value=RedisStore(host="redis.example.com", port=6379),
        fernet=Fernet(os.environ["STORAGE_ENCRYPTION_KEY"])
    )
)
```

Both parameters are required for production. **Wrap your storage in `FernetEncryptionWrapper` to encrypt sensitive OAuth tokens at rest** - without it, tokens are stored in plaintext. See [OAuth Token Security](/deployment/http#oauth-token-security) and [Key and Storage Management](/servers/auth/oauth-proxy#key-and-storage-management) for complete setup details.

### Response Caching Middleware

The [Response Caching Middleware](/servers/middleware#caching-middleware) caches tool calls, resource reads, and prompt requests. Storage configuration is passed via the `cache_storage` parameter:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from pathlib import Path
from fastmcp import FastMCP
from fastmcp.server.middleware.caching import ResponseCachingMiddleware
from key_value.aio.stores.filetree import (
    FileTreeStore,
    FileTreeV1KeySanitizationStrategy,
    FileTreeV1CollectionSanitizationStrategy,
)

mcp = FastMCP("My Server")

cache_dir = Path("cache")
cache_store = FileTreeStore(
    data_directory=cache_dir,
    key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(cache_dir),
    collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(cache_dir),
)

# Cache to disk instead of memory
mcp.add_middleware(ResponseCachingMiddleware(cache_storage=cache_store))
```

For multi-server deployments sharing a Redis instance:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.middleware.caching import ResponseCachingMiddleware
from key_value.aio.stores.redis import RedisStore
from key_value.aio.wrappers.prefix_collections import PrefixCollectionsWrapper

base_store = RedisStore(host="redis.example.com")
namespaced_store = PrefixCollectionsWrapper(
    key_value=base_store,
    prefix="my-server"
)

middleware = ResponseCachingMiddleware(cache_storage=namespaced_store)
```

### Client-Side OAuth Token Storage

The [FastMCP Client](/clients/client) uses storage for persisting OAuth tokens locally. By default, tokens are stored in memory:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from pathlib import Path
from fastmcp.client.auth import OAuthClientProvider
from key_value.aio.stores.filetree import (
    FileTreeStore,
    FileTreeV1KeySanitizationStrategy,
    FileTreeV1CollectionSanitizationStrategy,
)

# Store tokens on disk for persistence across restarts
token_dir = Path("~/.local/share/fastmcp/tokens").expanduser()
token_storage = FileTreeStore(
    data_directory=token_dir,
    key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(token_dir),
    collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(token_dir),
)

oauth_provider = OAuthClientProvider(
    mcp_url="https://your-mcp-server.com/mcp/sse",
    token_storage=token_storage
)
```

This allows clients to reconnect without re-authenticating after restarts.

## Choosing a Backend

| Backend  | Development | Single Server | Multi-Server | Cloud Native |
| -------- | ----------- | ------------- | ------------ | ------------ |
| Memory   | ✅ Best      | ⚠️ Limited    | ❌            | ❌            |
| File     | ✅ Good      | ✅ Recommended | ❌            | ⚠️           |
| Redis    | ⚠️ Overkill | ✅ Good        | ✅ Best       | ✅ Best       |
| DynamoDB | ❌           | ⚠️            | ✅            | ✅ Best (AWS) |
| MongoDB  | ❌           | ⚠️            | ✅            | ✅ Good       |

**Decision tree:**

1. **Just starting?** Use **Memory** (default) - no configuration needed
2. **Single server, needs persistence?** Use **File**
3. **Multiple servers or cloud deployment?** Use **Redis** or **DynamoDB**
4. **Existing infrastructure?** Look for a matching py-key-value-aio backend

## More Resources

* [py-key-value-aio GitHub](https://github.com/strawgate/py-key-value) - Full library documentation
* [Response Caching Middleware](/servers/middleware#caching-middleware) - Using storage for caching
* [OAuth Token Security](/deployment/http#oauth-token-security) - Production OAuth configuration
* [HTTP Deployment](/deployment/http) - Complete deployment guide


# Background Tasks
Source: https://gofastmcp.com/servers/tasks

Run long-running operations asynchronously with progress tracking

<VersionBadge />

<Tip>
  Background tasks require the `tasks` optional extra. See [installation instructions](#enabling-background-tasks) below.
</Tip>

FastMCP implements the MCP background task protocol ([SEP-1686](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks)), giving your servers a production-ready distributed task scheduler with a single decorator change.

<Tip>
  **What is Docket?** FastMCP's task system is powered by [Docket](https://github.com/chrisguidry/docket), originally built by [Prefect](https://prefect.io) to power [Prefect Cloud](https://www.prefect.io/prefect/cloud)'s managed task scheduling and execution service, where it processes millions of concurrent tasks every day. Docket is now open-sourced for the community.
</Tip>

## What Are MCP Background Tasks?

In MCP, all component interactions are blocking by default. When a client calls a tool, reads a resource, or fetches a prompt, it sends a request and waits for the response. For operations that take seconds or minutes, this creates a poor user experience.

The MCP background task protocol solves this by letting clients:

1. **Start** an operation and receive a task ID immediately
2. **Track** progress as the operation runs
3. **Retrieve** the result when ready

FastMCP handles all of this for you. Add `task=True` to your decorator, and your function gains full background execution with progress reporting, distributed processing, and horizontal scaling.

### MCP Background Tasks vs Python Concurrency

You can always use Python's concurrency primitives (asyncio, threads, multiprocessing) or external task queues in your FastMCP servers. FastMCP is just Python—run code however you like.

MCP background tasks are different: they're **protocol-native**. This means MCP clients that support the task protocol can start operations, receive progress updates, and retrieve results through the standard MCP interface. The coordination happens at the protocol level, not inside your application code.

## Enabling Background Tasks

<VersionBadge /> Background tasks require the `tasks` extra:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install "fastmcp[tasks]"
```

Add `task=True` to any tool, resource, resource template, or prompt decorator. This marks the component as capable of background execution.

```python {6} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import asyncio
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool(task=True)
async def slow_computation(duration: int) -> str:
    """A long-running operation."""
    for i in range(duration):
        await asyncio.sleep(1)
    return f"Completed in {duration} seconds"
```

When a client requests background execution, the call returns immediately with a task ID. The work executes in a background worker, and the client can poll for status or wait for the result.

<Warning>
  Background tasks require async functions. Attempting to use `task=True` with a sync function raises a `ValueError` at registration time.
</Warning>

## Execution Modes

For fine-grained control over task execution behavior, use `TaskConfig` instead of the boolean shorthand. The MCP task protocol defines three execution modes:

| Mode          | Client calls without task | Client calls with task      |
| ------------- | ------------------------- | --------------------------- |
| `"forbidden"` | Executes synchronously    | Error: task not supported   |
| `"optional"`  | Executes synchronously    | Executes as background task |
| `"required"`  | Error: task required      | Executes as background task |

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.tasks import TaskConfig

mcp = FastMCP("MyServer")

# Supports both sync and background execution (default when task=True)
@mcp.tool(task=TaskConfig(mode="optional"))
async def flexible_task() -> str:
    return "Works either way"

# Requires background execution - errors if client doesn't request task
@mcp.tool(task=TaskConfig(mode="required"))
async def must_be_background() -> str:
    return "Only runs as a background task"

# No task support (default when task=False or omitted)
@mcp.tool(task=TaskConfig(mode="forbidden"))
async def sync_only() -> str:
    return "Never runs as background task"
```

The boolean shortcuts map to these modes:

* `task=True` → `TaskConfig(mode="optional")`
* `task=False` → `TaskConfig(mode="forbidden")`

### Poll Interval

<VersionBadge />

When clients poll for task status, the server tells them how frequently to check back. By default, FastMCP suggests a 5-second interval, but you can customize this per component:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from datetime import timedelta
from fastmcp import FastMCP
from fastmcp.server.tasks import TaskConfig

mcp = FastMCP("MyServer")

# Poll every 2 seconds for a fast-completing task
@mcp.tool(task=TaskConfig(mode="optional", poll_interval=timedelta(seconds=2)))
async def quick_task() -> str:
    return "Done quickly"

# Poll every 30 seconds for a long-running task
@mcp.tool(task=TaskConfig(mode="optional", poll_interval=timedelta(seconds=30)))
async def slow_task() -> str:
    return "Eventually done"
```

Shorter intervals give clients faster feedback but increase server load. Longer intervals reduce load but delay status updates.

### Server-Wide Default

To enable background task support for all components by default, pass `tasks=True` to the constructor. Individual decorators can still override this with `task=False`.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp = FastMCP("MyServer", tasks=True)
```

<Warning>
  If your server defines any synchronous tools, resources, or prompts, you will need to explicitly set `task=False` on their decorators to avoid an error.
</Warning>

### Graceful Degradation

When a client requests background execution but the component has `mode="forbidden"`, FastMCP executes synchronously and returns the result inline. This follows the SEP-1686 specification for graceful degradation—clients can always request background execution without worrying about server capabilities.

Conversely, when a component has `mode="required"` but the client doesn't request background execution, FastMCP returns an error indicating that task execution is required.

### Configuration

| Environment Variable | Default     | Description                                         |
| -------------------- | ----------- | --------------------------------------------------- |
| `FASTMCP_DOCKET_URL` | `memory://` | Backend URL (`memory://` or `redis://host:port/db`) |

## Backends

FastMCP supports two backends for task execution, each with different tradeoffs.

### In-Memory Backend (Default)

The in-memory backend (`memory://`) requires zero configuration and works out of the box.

**Advantages:**

* No external dependencies
* Simple single-process deployment

**Disadvantages:**

* **Ephemeral**: If the server restarts, all pending tasks are lost
* **Higher latency**: \~250ms task pickup time vs single-digit milliseconds with Redis
* **No horizontal scaling**: Single process only—you cannot add additional workers

### Redis Backend

For production deployments, use Redis (or Valkey) as your backend by setting `FASTMCP_DOCKET_URL=redis://localhost:6379`.

**Advantages:**

* **Persistent**: Tasks survive server restarts
* **Fast**: Single-digit millisecond task pickup latency
* **Scalable**: Add workers to distribute load across processes or machines

## Workers

Every FastMCP server with task-enabled components automatically starts an **embedded worker**. You do not need to start a separate worker process for tasks to execute.

To scale horizontally, add more workers using the CLI:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp tasks worker server.py
```

Each additional worker pulls tasks from the same queue, distributing load across processes. Configure worker concurrency via environment:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
export FASTMCP_DOCKET_CONCURRENCY=20
fastmcp tasks worker server.py
```

<Note>
  Additional workers only work with Redis/Valkey backends. The in-memory backend is single-process only.
</Note>

<Warning>
  Task-enabled components must be defined at server startup to be registered with all workers. Components added dynamically after the server starts will not be available for background execution.
</Warning>

## Progress Reporting

The `Progress` dependency lets you report progress back to clients. Inject it as a parameter with a default value, and FastMCP will provide the active progress reporter.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.dependencies import Progress

mcp = FastMCP("MyServer")

@mcp.tool(task=True)
async def process_files(files: list[str], progress: Progress = Progress()) -> str:
    await progress.set_total(len(files))

    for file in files:
        await progress.set_message(f"Processing {file}")
        # ... do work ...
        await progress.increment()

    return f"Processed {len(files)} files"
```

The progress API:

* `await progress.set_total(n)` — Set the total number of steps
* `await progress.increment(amount=1)` — Increment progress
* `await progress.set_message(text)` — Update the status message

Progress works in both immediate and background execution modes—you can use the same code regardless of how the client invokes your function.

## Docket Dependencies

FastMCP exposes Docket's full dependency injection system within your task-enabled functions. Beyond `Progress`, you can access the Docket instance, worker information, and use advanced features like retries and timeouts.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from docket import Docket, Worker
from fastmcp import FastMCP
from fastmcp.dependencies import Progress, CurrentDocket, CurrentWorker

mcp = FastMCP("MyServer")

@mcp.tool(task=True)
async def my_task(
    progress: Progress = Progress(),
    docket: Docket = CurrentDocket(),
    worker: Worker = CurrentWorker(),
) -> str:
    # Schedule additional background work
    await docket.add(another_task, arg1, arg2)

    # Access worker metadata
    worker_name = worker.name

    return "Done"
```

With `CurrentDocket()`, you can schedule additional background tasks, chain work together, and coordinate complex workflows. See the [Docket documentation](https://chrisguidry.github.io/docket/) for the complete API, including retry policies, timeouts, and custom dependencies.


# OpenTelemetry
Source: https://gofastmcp.com/servers/telemetry

Native OpenTelemetry instrumentation for distributed tracing.

FastMCP includes native OpenTelemetry instrumentation for observability. Traces are automatically generated for tool, prompt, resource, and resource template operations, providing visibility into server behavior, request handling, and provider delegation chains.

## How It Works

FastMCP uses the OpenTelemetry API for instrumentation. This means:

* **Zero configuration required** - Instrumentation is always active
* **No overhead when unused** - Without an SDK, all operations are no-ops
* **Bring your own SDK** - You control collection, export, and sampling
* **Works with any OTEL backend** - Jaeger, Zipkin, Datadog, New Relic, etc.

## Enabling Telemetry

The easiest way to export traces is using `opentelemetry-instrument`, which configures the SDK automatically:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install
```

Then run your server with tracing enabled:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
opentelemetry-instrument \
  --service_name my-fastmcp-server \
  --exporter_otlp_endpoint http://localhost:4317 \
  fastmcp run server.py
```

Or configure via environment variables:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
export OTEL_SERVICE_NAME=my-fastmcp-server
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

opentelemetry-instrument fastmcp run server.py
```

This works with any OTLP-compatible backend (Jaeger, Zipkin, Grafana Tempo, Datadog, etc.) and requires no changes to your FastMCP code.

<Card title="OpenTelemetry Python Documentation" icon="book" href="https://opentelemetry.io/docs/languages/python/">
  Learn more about the OpenTelemetry Python SDK, auto-instrumentation, and available exporters.
</Card>

## Tracing

FastMCP creates spans for all MCP operations, providing end-to-end visibility into request handling.

### Server Spans

The server creates spans for each operation using [MCP semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/):

| Span Name              | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `tools/call {name}`    | Tool execution (e.g., `tools/call get_weather`)          |
| `resources/read {uri}` | Resource read (e.g., `resources/read config://database`) |
| `prompts/get {name}`   | Prompt render (e.g., `prompts/get greeting`)             |

For mounted servers, an additional `delegate {name}` span shows the delegation to the child server.

### Client Spans

The FastMCP client creates spans for outgoing requests with the same naming pattern (`tools/call {name}`, `resources/read {uri}`, `prompts/get {name}`).

### Span Hierarchy

Spans form a hierarchy showing the request flow. For mounted servers:

```
tools/call weather_forecast (CLIENT)
  └── tools/call weather_forecast (SERVER, provider=FastMCPProvider)
        └── delegate get_weather (INTERNAL)
              └── tools/call get_weather (SERVER, provider=LocalProvider)
```

For proxy providers connecting to remote servers:

```
tools/call remote_search (CLIENT)
  └── tools/call remote_search (SERVER, provider=ProxyProvider)
        └── [remote server spans via trace context propagation]
```

## Programmatic Configuration

For more control, configure the SDK in your Python code before importing FastMCP:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure the SDK with OTLP exporter
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Now import and use FastMCP - traces will be exported automatically
from fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

<Tip>
  The SDK must be configured **before** importing FastMCP to ensure the tracer provider is set when FastMCP initializes.
</Tip>

### Local Development

For quick local trace visualization, [otel-desktop-viewer](https://github.com/CtrlSpice/otel-desktop-viewer) is a lightweight single-binary tool:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# macOS
brew install nico-barbas/brew/otel-desktop-viewer

# Or download from GitHub releases
```

Run it alongside your server:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Terminal 1: Start the viewer (UI at http://localhost:8000, OTLP on :4317)
otel-desktop-viewer

# Terminal 2: Run your server with tracing
opentelemetry-instrument fastmcp run server.py
```

For more features, use [Jaeger](https://www.jaegertracing.io/):

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

Then view traces at [http://localhost:16686](http://localhost:16686)

## Custom Spans

You can add your own spans using the FastMCP tracer:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.telemetry import get_tracer

mcp = FastMCP("custom-spans")

@mcp.tool()
async def complex_operation(input: str) -> str:
    tracer = get_tracer()

    with tracer.start_as_current_span("parse_input") as span:
        span.set_attribute("input.length", len(input))
        parsed = parse(input)

    with tracer.start_as_current_span("process_data") as span:
        span.set_attribute("data.count", len(parsed))
        result = process(parsed)

    return result
```

## Error Handling

When errors occur, spans are automatically marked with error status and the exception is recorded:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool()
def risky_operation() -> str:
    raise ValueError("Something went wrong")

# The span will have:
# - status = ERROR
# - exception event with stack trace
```

## Attributes Reference

### RPC Semantic Conventions

Standard [RPC semantic conventions](https://opentelemetry.io/docs/specs/semconv/rpc/rpc-spans/):

| Attribute     | Value               |
| ------------- | ------------------- |
| `rpc.system`  | `"mcp"`             |
| `rpc.service` | Server name         |
| `rpc.method`  | MCP protocol method |

### MCP Semantic Conventions

FastMCP implements the [OpenTelemetry MCP semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/):

| Attribute          | Description                                                                 |
| ------------------ | --------------------------------------------------------------------------- |
| `mcp.method.name`  | The MCP method being called (`tools/call`, `resources/read`, `prompts/get`) |
| `mcp.session.id`   | Session identifier for the MCP connection                                   |
| `mcp.resource.uri` | The resource URI (for resource operations)                                  |

### Auth Attributes

Standard [identity attributes](https://opentelemetry.io/docs/specs/semconv/attributes-registry/enduser/):

| Attribute       | Description                                       |
| --------------- | ------------------------------------------------- |
| `enduser.id`    | Client ID from access token (when authenticated)  |
| `enduser.scope` | Space-separated OAuth scopes (when authenticated) |

### FastMCP Custom Attributes

All custom attributes use the `fastmcp.` prefix for features unique to FastMCP:

| Attribute                | Description                                                          |
| ------------------------ | -------------------------------------------------------------------- |
| `fastmcp.server.name`    | Server name                                                          |
| `fastmcp.component.type` | `tool`, `resource`, `prompt`, or `resource_template`                 |
| `fastmcp.component.key`  | Full component identifier (e.g., `tool:greet`)                       |
| `fastmcp.provider.type`  | Provider class (`LocalProvider`, `FastMCPProvider`, `ProxyProvider`) |

Provider-specific attributes for delegation context:

| Attribute                        | Description                                  |
| -------------------------------- | -------------------------------------------- |
| `fastmcp.delegate.original_name` | Original tool/prompt name before namespacing |
| `fastmcp.delegate.original_uri`  | Original resource URI before namespacing     |
| `fastmcp.proxy.backend_name`     | Remote server tool/prompt name               |
| `fastmcp.proxy.backend_uri`      | Remote server resource URI                   |

## Testing with Telemetry

For testing, use the in-memory exporter:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import pytest
from collections.abc import Generator
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from fastmcp import FastMCP

@pytest.fixture
def trace_exporter() -> Generator[InMemorySpanExporter, None, None]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    original_provider = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    trace.set_tracer_provider(original_provider)

async def test_tool_creates_span(trace_exporter: InMemorySpanExporter) -> None:
    mcp = FastMCP("test")

    @mcp.tool()
    def hello() -> str:
        return "world"

    await mcp.call_tool("hello", {})

    spans = trace_exporter.get_finished_spans()
    assert any(s.name == "tools/call hello" for s in spans)
```


# Testing your FastMCP Server
Source: https://gofastmcp.com/servers/testing

How to test your FastMCP server.

The best way to ensure a reliable and maintainable FastMCP Server is to test it! The FastMCP Client combined with Pytest provides a simple and powerful way to test your FastMCP servers.

## Prerequisites

Testing FastMCP servers requires `pytest-asyncio` to handle async test functions and fixtures. Install it as a development dependency:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install pytest-asyncio
```

We recommend configuring pytest to automatically handle async tests by setting the asyncio mode to `auto` in your `pyproject.toml`:

```toml theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

This eliminates the need to decorate every async test with `@pytest.mark.asyncio`.

## Testing with Pytest Fixtures

Using Pytest Fixtures, you can wrap your FastMCP Server in a Client instance that makes interacting with your server fast and easy. This is especially useful when building your own MCP Servers and enables a tight development loop by allowing you to avoid using a separate tool like MCP Inspector during development:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from my_project.main import mcp

@pytest.fixture
async def main_mcp_client():
    async with Client(transport=mcp) as mcp_client:
        yield mcp_client

async def test_list_tools(main_mcp_client: Client[FastMCPTransport]):
    list_tools = await main_mcp_client.list_tools()

    assert len(list_tools) == 5
```

We recommend the [inline-snapshot library](https://github.com/15r10nk/inline-snapshot) for asserting complex data structures coming from your MCP Server. This library allows you to write tests that are easy to read and understand, and are also easy to update when the data structure changes.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from inline_snapshot import snapshot

async def test_list_tools(main_mcp_client: Client[FastMCPTransport]):
    list_tools = await main_mcp_client.list_tools()

    assert list_tools == snapshot()
```

Simply run `pytest --inline-snapshot=fix,create` to fill in the `snapshot()` with actual data.

<Tip>
  For values that change you can leverage the [dirty-equals](https://github.com/samuelcolvin/dirty-equals) library to perform flexible equality assertions on dynamic or non-deterministic values.
</Tip>

Using the pytest `parametrize` decorator, you can easily test your tools with a wide variety of inputs.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import pytest
from my_project.main import mcp

from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
@pytest.fixture
async def main_mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.parametrize(
    "first_number, second_number, expected",
    [
        (1, 2, 3),
        (2, 3, 5),
        (3, 4, 7),
    ],
)
async def test_add(
    first_number: int,
    second_number: int,
    expected: int,
    main_mcp_client: Client[FastMCPTransport],
):
    result = await main_mcp_client.call_tool(
        name="add", arguments={"x": first_number, "y": second_number}
    )
    assert result.data is not None
    assert isinstance(result.data, int)
    assert result.data == expected
```

<Tip>
  The [FastMCP Repository contains thousands of tests](https://github.com/PrefectHQ/fastmcp/tree/main/tests) for the FastMCP Client and Server. Everything from connecting to remote MCP servers, to testing tools, resources, and prompts is covered, take a look for inspiration!
</Tip>
