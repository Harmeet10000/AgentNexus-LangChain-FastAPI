# Upgrade Guides

Source lines: 29118-30304 from the original FastMCP documentation dump.

Migration guidance from FastMCP 2, the low-level SDK, and the MCP SDK.

---

# Upgrading from FastMCP 2
Source: https://gofastmcp.com/getting-started/upgrading/from-fastmcp-2

Migration instructions for upgrading between FastMCP versions

This guide covers breaking changes and migration steps when upgrading FastMCP.

## v3.0.0

For most servers, upgrading to v3 is straightforward. The breaking changes below affect deprecated constructor kwargs, sync-to-async shifts, a few renamed methods, and some less commonly used features.

### Install

Since you already have `fastmcp` installed, you need to explicitly request the new version — `pip install fastmcp` won't upgrade an existing installation:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install --upgrade fastmcp
# or
uv add --upgrade fastmcp
```

If you pin versions in a requirements file or `pyproject.toml`, update your pin to `fastmcp>=3.0.0,<4`.

<Info>
  **New repository home.** As part of the v3 release, FastMCP's GitHub repository has moved from `jlowin/fastmcp` to [`PrefectHQ/fastmcp`](https://github.com/PrefectHQ/fastmcp) under [Prefect](https://prefect.io)'s stewardship. GitHub automatically redirects existing clones and bookmarks, so nothing breaks — but you can update your local remote whenever convenient:

  ```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  git remote set-url origin https://github.com/PrefectHQ/fastmcp.git
  ```

  If you reference the repository URL in dependency specifications (e.g., `git+https://github.com/jlowin/fastmcp.git`), update those to the new location.
</Info>

<Prompt description="Copy this prompt into any LLM along with your server code to get automated upgrade guidance.">
  You are upgrading a FastMCP v2 server to FastMCP v3.0. Analyze the provided code and identify every change needed. The full upgrade guide is at [https://gofastmcp.com/getting-started/upgrading/from-fastmcp-2](https://gofastmcp.com/getting-started/upgrading/from-fastmcp-2) and the complete FastMCP documentation is at [https://gofastmcp.com](https://gofastmcp.com) — fetch these for complete context.

  BREAKING CHANGES (will crash at import or runtime):

  1. CONSTRUCTOR KWARGS REMOVED: FastMCP() no longer accepts these kwargs (raises TypeError):
     * Transport settings: host, port, log\_level, debug, sse\_path, streamable\_http\_path, json\_response, stateless\_http
       Fix: pass to run() or run\_http\_async() instead, e.g. mcp.run(transport="http", host="0.0.0.0", port=8080)
     * message\_path: set via environment variable FASTMCP\_MESSAGE\_PATH only (not a run() kwarg)
     * Duplicate handling: on\_duplicate\_tools, on\_duplicate\_resources, on\_duplicate\_prompts
       Fix: use unified on\_duplicate= parameter
     * Tool settings: tool\_serializer, include\_tags, exclude\_tags, tool\_transformations
       Fix: use ToolResult returns, server.enable()/disable(), server.add\_transform()

  2. COMPONENT METHODS REMOVED:
     * tool.enable()/disable() raises NotImplementedError
       Fix: server.disable(names=, components=) or server.disable(tags=)
     * get\_tools()/get\_resources()/get\_prompts()/get\_resource\_templates() removed
       Fix: use list\_tools()/list\_resources()/list\_prompts()/list\_resource\_templates() — these return lists, not dicts

  3. ASYNC STATE: ctx.set\_state() and ctx.get\_state() are now async (must be awaited).
     State values must be JSON-serializable unless serializable=False is passed.
     Each FastMCP instance has its own state store, so serializable state set by parent middleware isn't visible to mounted tools by default.
     Fix: pass the same session\_state\_store to both servers, or use serializable=False (request-scoped state is always shared).

  4. PROMPTS: mcp.types.PromptMessage replaced by fastmcp.prompts.Message.
     Before: PromptMessage(role="user", content=TextContent(type="text", text="Hello"))
     After:  Message("Hello")  # role defaults to "user", accepts plain strings
     Also: if prompts return raw dicts like `{"role": "user", "content": "..."}`, these must become Message objects.
     v2 silently coerced dicts; v3 requires typed Message objects or plain strings.

  5. AUTH PROVIDERS: No longer auto-load from env vars. Pass client\_id, client\_secret explicitly via os.environ.

  6. WSTRANSPORT: Removed. Use StreamableHttpTransport.

  7. OPENAPI: timeout parameter removed from OpenAPIProvider. Set timeout on the httpx.AsyncClient instead.

  8. METADATA: Namespace changed from "\_fastmcp" to "fastmcp" in tool.meta. The include\_fastmcp\_meta parameter is removed (always included).

  9. ENV VAR: FASTMCP\_SHOW\_CLI\_BANNER renamed to FASTMCP\_SHOW\_SERVER\_BANNER.

  10. DECORATORS: @mcp.tool, @mcp.resource, @mcp.prompt now return the original function, not a component object. Code that accesses .name, .description, or other component attributes on the decorated result will crash with AttributeError.
      Fix: set FASTMCP\_DECORATOR\_MODE=object for v2 compat (itself deprecated).

  11. OAUTH STORAGE: Default OAuth client storage changed from DiskStore to FileTreeStore due to pickle deserialization vulnerability in diskcache (CVE-2025-69872). Clients using default storage will re-register automatically on first connection. If using DiskStore explicitly, switch to FileTreeStore or add pip install 'py-key-value-aio\[disk]'.

  12. REPO MOVE: GitHub repository moved from jlowin/fastmcp to PrefectHQ/fastmcp. Update git remotes and dependency URLs that reference the old location.

  13. BACKGROUND TASKS: FastMCP's background task system (SEP-1686) is now an optional dependency. If the code uses task=True or TaskConfig, add pip install "fastmcp\[tasks]".

  DEPRECATIONS (still work but emit warnings):

  * mount(prefix="x") -> mount(namespace="x")
  * import\_server(sub) -> mount(sub)
  * FastMCP.as\_proxy(url) -> from fastmcp.server import create\_proxy; create\_proxy(url)
  * from fastmcp.server.proxy -> from fastmcp.server.providers.proxy
  * from fastmcp.server.openapi import FastMCPOpenAPI -> from fastmcp.server.providers.openapi import OpenAPIProvider; use FastMCP("name", providers=\[OpenAPIProvider(...)])
  * mcp.add\_tool\_transformation(name, cfg) -> from fastmcp.server.transforms import ToolTransform; mcp.add\_transform(ToolTransform(...))

  For each issue found, show the original line, explain why it breaks, and provide the corrected code.
</Prompt>

### Breaking Changes

**Transport and server settings removed from constructor**

In v2, you could configure transport settings directly in the `FastMCP()` constructor. In v3, `FastMCP()` is purely about your server's identity and behavior — transport configuration happens when you actually start serving. Passing any of the old kwargs now raises `TypeError` with a migration hint.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
mcp = FastMCP("server", host="0.0.0.0", port=8080)
mcp.run()

# After
mcp = FastMCP("server")
mcp.run(transport="http", host="0.0.0.0", port=8080)
```

The full list of removed kwargs and their replacements:

* `host`, `port`, `log_level`, `debug`, `sse_path`, `streamable_http_path`, `json_response`, `stateless_http` — pass to `run()`, `run_http_async()`, or `http_app()`, or set via environment variables (e.g. `FASTMCP_HOST`)
* `message_path` — set via environment variable `FASTMCP_MESSAGE_PATH` only (not a `run()` kwarg)
* `on_duplicate_tools`, `on_duplicate_resources`, `on_duplicate_prompts` — consolidated into a single `on_duplicate=` parameter
* `tool_serializer` — return [`ToolResult`](/servers/tools#custom-serialization) from your tools instead
* `include_tags` / `exclude_tags` — use `server.enable(tags=..., only=True)` / `server.disable(tags=...)` after construction
* `tool_transformations` — use `server.add_transform(ToolTransform(...))` after construction

**OAuth storage backend changed (diskcache CVE)**

The default OAuth client storage has moved from `DiskStore` to `FileTreeStore` to address a pickle deserialization vulnerability in diskcache ([CVE-2025-69872](https://github.com/PrefectHQ/fastmcp/issues/3166)).

If you were using the default storage (i.e., not passing an explicit `client_storage`), clients will need to re-register on their first connection after upgrading. This happens automatically — no user action required, and it's the same flow that already occurs whenever a server restarts with in-memory storage.

If you were passing a `DiskStore` explicitly, you can either [switch to `FileTreeStore`](/servers/storage-backends) (recommended) or keep using `DiskStore` by adding the dependency yourself:

<Warning>
  Keeping `DiskStore` requires `pip install 'py-key-value-aio[disk]'`, which re-introduces the vulnerable `diskcache` package into your dependency tree.
</Warning>

**Component enable()/disable() moved to server**

In v2, you could enable or disable individual components by calling methods on the component object itself. In v3, visibility is controlled through the server (or provider), which lets you target components by name, tag, or type without needing a reference to the object:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
tool = await server.get_tool("my_tool")
tool.disable()

# After
server.disable(names={"my_tool"}, components={"tool"})
```

Calling `.enable()` or `.disable()` on a component object now raises `NotImplementedError`. See [Visibility](/servers/visibility) for the full API, including tag-based filtering and per-session visibility.

**Listing methods renamed and return lists**

The `get_tools()`, `get_resources()`, `get_prompts()`, and `get_resource_templates()` methods have been renamed to `list_tools()`, `list_resources()`, `list_prompts()`, and `list_resource_templates()`. More importantly, they now return lists instead of dicts — so code that indexes by name needs to change:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
tools = await server.get_tools()
tool = tools["my_tool"]

# After
tools = await server.list_tools()
tool = next((t for t in tools if t.name == "my_tool"), None)
```

**Prompts use Message class**

Prompt functions now use FastMCP's `Message` class instead of `mcp.types.PromptMessage`. The new class is simpler — it accepts a plain string and defaults to `role="user"`, so most prompts become one-liners:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
from mcp.types import PromptMessage, TextContent

@mcp.prompt
def my_prompt() -> PromptMessage:
    return PromptMessage(role="user", content=TextContent(type="text", text="Hello"))

# After
from fastmcp.prompts import Message

@mcp.prompt
def my_prompt() -> Message:
    return Message("Hello")
```

If your prompt functions return raw dicts with `role` and `content` keys, those also need to change. v2 silently coerced dicts into prompt messages, but v3 requires typed `Message` objects (or plain strings for single user messages):

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before (v2 accepted this)
@mcp.prompt
def my_prompt():
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "How can I help?"},
    ]

# After
from fastmcp.prompts import Message

@mcp.prompt
def my_prompt() -> list[Message]:
    return [
        Message("Hello"),
        Message("How can I help?", role="assistant"),
    ]
```

**Context state methods are async**

`ctx.set_state()` and `ctx.get_state()` are now async because state in v3 is session-scoped and backed by a pluggable storage backend (rather than a simple dict). This means state persists across multiple tool calls within the same session:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
ctx.set_state("key", "value")
value = ctx.get_state("key")

# After
await ctx.set_state("key", "value")
value = await ctx.get_state("key")
```

State values must also be JSON-serializable by default (dicts, lists, strings, numbers, etc.). If you need to store non-serializable values like an HTTP client, pass `serializable=False` — these values are request-scoped and only available during the current tool call:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
await ctx.set_state("client", my_http_client, serializable=False)
```

**Mounted servers have isolated state stores**

Each `FastMCP` instance has its own state store. In v2 this wasn't noticeable because mounted tools ran in the parent's context, but in v3's provider architecture each server is isolated. Non-serializable state (`serializable=False`) is request-scoped and automatically shared across mount boundaries. For serializable state, pass the same `session_state_store` to both servers:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from key_value.aio.stores.memory import MemoryStore

store = MemoryStore()
parent = FastMCP("Parent", session_state_store=store)
child = FastMCP("Child", session_state_store=store)
parent.mount(child, namespace="child")
```

**Auth provider environment variables removed**

In v2, auth providers like `GitHubProvider` could auto-load configuration from environment variables with a `FASTMCP_SERVER_AUTH_*` prefix. This magic has been removed — pass values explicitly:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before (v2) — client_id and client_secret loaded automatically
# from FASTMCP_SERVER_AUTH_GITHUB_CLIENT_ID, etc.
auth = GitHubProvider()

# After (v3) — pass values explicitly
import os
from fastmcp.server.auth.providers.github import GitHubProvider

auth = GitHubProvider(
    client_id=os.environ["GITHUB_CLIENT_ID"],
    client_secret=os.environ["GITHUB_CLIENT_SECRET"],
)
```

**WSTransport removed**

The deprecated WebSocket client transport has been removed. Use `StreamableHttpTransport` instead:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
from fastmcp.client.transports import WSTransport
transport = WSTransport("ws://localhost:8000/ws")

# After
from fastmcp.client.transports import StreamableHttpTransport
transport = StreamableHttpTransport("http://localhost:8000/mcp")
```

**OpenAPI `timeout` parameter removed**

`OpenAPIProvider` no longer accepts a `timeout` parameter. Configure timeout on the httpx client directly. The `client` parameter is also now optional — when omitted, a default client is created from the spec's `servers` URL with a 30-second timeout:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
provider = OpenAPIProvider(spec, client, timeout=60)

# After
client = httpx.AsyncClient(base_url="https://api.example.com", timeout=60)
provider = OpenAPIProvider(spec, client)
```

**Metadata namespace renamed**

The FastMCP metadata key in component `meta` dicts changed from `_fastmcp` to `fastmcp`. If you read metadata from tool or resource objects, update the key:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
tags = tool.meta.get("_fastmcp", {}).get("tags", [])

# After
tags = tool.meta.get("fastmcp", {}).get("tags", [])
```

Metadata is now always included — the `include_fastmcp_meta` parameter has been removed from `FastMCP()` and `to_mcp_tool()`, so there is no way to suppress it.

**Server banner environment variable renamed**

`FASTMCP_SHOW_CLI_BANNER` is now `FASTMCP_SHOW_SERVER_BANNER`.

**Decorators return functions**

In v2, `@mcp.tool` transformed your function into a `FunctionTool` object. In v3, decorators return your original function unchanged — which means decorated functions stay callable for testing, reuse, and composition:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

greet("World")  # Works! Returns "Hello, World!"
```

If you have code that treats the decorated result as a `FunctionTool` (e.g., accessing `.name` or `.description`), set `FASTMCP_DECORATOR_MODE=object` for v2 compatibility. This escape hatch is itself deprecated and will be removed in a future release.

**Background tasks require optional dependency**

FastMCP's background task system (SEP-1686) is now behind an optional extra. If your server uses background tasks, install with:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install "fastmcp[tasks]"
```

Without the extra, configuring a tool with `task=True` or `TaskConfig` will raise an import error at runtime. See [Background Tasks](/servers/tasks) for details.

### Deprecated Features

These still work but emit warnings. Update when convenient.

**mount() prefix → namespace**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Deprecated
main.mount(subserver, prefix="api")

# New
main.mount(subserver, namespace="api")
```

**import\_server() → mount()**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Deprecated
main.import_server(subserver)

# New
main.mount(subserver)
```

**Module import paths for proxy and OpenAPI**

The proxy and OpenAPI modules have moved under `providers` to reflect v3's provider-based architecture:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Deprecated
from fastmcp.server.proxy import FastMCPProxy
from fastmcp.server.openapi import FastMCPOpenAPI

# New
from fastmcp.server.providers.proxy import FastMCPProxy
from fastmcp.server.providers.openapi import OpenAPIProvider
```

`FastMCPOpenAPI` itself is deprecated — use `FastMCP` with an `OpenAPIProvider` instead:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Deprecated
from fastmcp.server.openapi import FastMCPOpenAPI
server = FastMCPOpenAPI(spec, client)

# New
from fastmcp import FastMCP
from fastmcp.server.providers.openapi import OpenAPIProvider
server = FastMCP("my_api", providers=[OpenAPIProvider(spec, client)])
```

**add\_tool\_transformation() → add\_transform()**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Deprecated
mcp.add_tool_transformation("name", config)

# New
from fastmcp.server.transforms import ToolTransform
mcp.add_transform(ToolTransform({"name": config}))
```

**FastMCP.as\_proxy() → create\_proxy()**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Deprecated
proxy = FastMCP.as_proxy("http://example.com/mcp")

# New
from fastmcp.server import create_proxy
proxy = create_proxy("http://example.com/mcp")
```

## v2.14.0

### OpenAPI Parser Promotion

The experimental OpenAPI parser is now standard. Update imports:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
from fastmcp.experimental.server.openapi import FastMCPOpenAPI

# After
from fastmcp.server.openapi import FastMCPOpenAPI
```

### Removed Deprecated Features

* `BearerAuthProvider` → use `JWTVerifier`
* `Context.get_http_request()` → use `get_http_request()` from dependencies
* `from fastmcp import Image` → use `from fastmcp.utilities.types import Image`
* `FastMCP(dependencies=[...])` → use `fastmcp.json` configuration
* `FastMCPProxy(client=...)` → use `client_factory=lambda: ...`
* `output_schema=False` → use `output_schema=None`

## v2.13.0

### OAuth Token Key Management

The OAuth proxy now issues its own JWT tokens. For production, provide explicit keys:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
auth = GitHubProvider(
    client_id=os.environ["GITHUB_CLIENT_ID"],
    client_secret=os.environ["GITHUB_CLIENT_SECRET"],
    base_url="https://your-server.com",
    jwt_signing_key=os.environ["JWT_SIGNING_KEY"],
    client_storage=RedisStore(host="redis.example.com"),
)
```

See [OAuth Token Security](/deployment/http#oauth-token-security) for details.


# Upgrading from the MCP Low-Level SDK
Source: https://gofastmcp.com/getting-started/upgrading/from-low-level-sdk

Upgrade your MCP server from the low-level Python SDK's Server class to FastMCP

If you've been building MCP servers directly on the `mcp` package's `Server` class — writing `list_tools()` and `call_tool()` handlers, hand-crafting JSON Schema dicts, and wiring up transport boilerplate — this guide is for you. FastMCP replaces all of that machinery with a declarative, Pythonic API where your functions *are* the protocol surface.

The core idea: instead of telling the SDK what your tools look like and then separately implementing them, you write ordinary Python functions and let FastMCP derive the protocol layer from your code. Type hints become JSON Schema. Docstrings become descriptions. Return values are serialized automatically. The plumbing you wrote to satisfy the protocol just disappears.

<Note>
  This guide covers upgrading from **v1** of the `mcp` package. We'll provide a separate guide when v2 ships.
</Note>

<Note>
  Already using FastMCP 1.0 via `from mcp.server.fastmcp import FastMCP`? Your upgrade is simpler — see the [FastMCP 1.0 upgrade guide](/getting-started/upgrading/from-mcp-sdk) instead.
</Note>

<Prompt description="Copy this prompt into any LLM along with your server code to get automated upgrade guidance.">
  You are upgrading an MCP server from the `mcp` package's low-level Server class (v1) to FastMCP 3.0. The server currently uses `mcp.server.Server` (or `mcp.server.lowlevel.server.Server`) with manual handler registration. Analyze the provided code and rewrite it using FastMCP's high-level API. The full guide is at [https://gofastmcp.com/getting-started/upgrading/from-low-level-sdk](https://gofastmcp.com/getting-started/upgrading/from-low-level-sdk) and the complete FastMCP documentation is at [https://gofastmcp.com](https://gofastmcp.com) — fetch these for complete context.

  UPGRADE RULES:

  1. IMPORTS: Replace all `mcp.*` imports with FastMCP equivalents.
     * `from mcp.server import Server` or `from mcp.server.lowlevel.server import Server` → `from fastmcp import FastMCP`
     * `import mcp.types as types` → remove (not needed for most code)
     * `from mcp.server.stdio import stdio_server` → remove (handled by mcp.run())
     * `from mcp.server.sse import SseServerTransport` → remove (handled by mcp.run())

  2. SERVER: Replace `Server("name")` with `FastMCP("name")`.

  3. TOOLS: Replace the list\_tools + call\_tool handler pair with individual @mcp.tool decorators.
     * Delete the `@server.list_tools()` handler entirely
     * Delete the `@server.call_tool()` handler entirely
     * For each tool that was listed in list\_tools and dispatched in call\_tool, create a new function:
       * Decorate it with `@mcp.tool`
       * Use the tool name as the function name (or pass name= to the decorator)
       * Use the docstring for the description (or pass description= to the decorator)
       * Convert the inputSchema JSON Schema into typed Python parameters (e.g., `{"type": "integer"}` → `int`, `{"type": "string"}` → `str`, `{"type": "array", "items": {"type": "string"}}` → `list[str]`)
       * Return plain Python values (`str`, `int`, `dict`, etc.) instead of `list[types.TextContent(...)]`
       * If the tool returned `types.ImageContent` or `types.EmbeddedResource`, use `from fastmcp.utilities.types import Image` or return the appropriate type

  4. RESOURCES: Replace the list\_resources + list\_resource\_templates + read\_resource handler trio with individual @mcp.resource decorators.
     * Delete all three handlers
     * For each static resource, create a function decorated with `@mcp.resource("uri://...")`
     * For each resource template, use `@mcp.resource("uri://{param}/path")` with `{param}` in the URI and a matching function parameter
     * Return str for text content, bytes for binary content
     * Set `mime_type=` in the decorator if needed

  5. PROMPTS: Replace the list\_prompts + get\_prompt handler pair with individual @mcp.prompt decorators.
     * Delete both handlers
     * For each prompt, create a function decorated with `@mcp.prompt`
     * Convert PromptArgument definitions into typed function parameters
     * Return str for simple single-message prompts (auto-wrapped as user message)
     * Return `list[Message]` for multi-message prompts: `from fastmcp.prompts import Message`
     * `Message("text")` defaults to `role="user"`; use `Message("text", role="assistant")` for assistant messages

  6. TRANSPORT: Replace all transport boilerplate with mcp.run().
     * `async with stdio_server() as (r, w): await server.run(r, w, ...)` → `mcp.run()` (`stdio` is the default)
     * SSE/Starlette setup → `mcp.run(transport="sse", host="...", port=...)`
     * Streamable HTTP setup → `mcp.run(transport="http", host="...", port=...)`
     * Delete asyncio.run(main()) boilerplate — use `if __name__ == "__main__": mcp.run()`

  7. CONTEXT: Replace `server.request_context` with FastMCP's Context parameter.
     * Add `from fastmcp import Context` and add a `ctx: Context` parameter to any tool that needs it
     * `server.request_context.session.send_log_message(...)` → `await ctx.info("message")` or `await ctx.warning("message")`
     * Progress reporting → `await ctx.report_progress(current, total)`

  For each change, show the original code, explain what it did, and provide the FastMCP equivalent.
</Prompt>

## Install

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install --upgrade fastmcp
# or
uv add fastmcp
```

FastMCP includes the `mcp` package as a transitive dependency, so you don't lose access to anything.

## Server and Transport

The `Server` class requires you to choose a transport, connect streams, build initialization options, and run an event loop. FastMCP collapses all of that into a constructor and a `run()` call.

<CodeGroup>
  ```python Before theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  import asyncio
  from mcp.server import Server
  from mcp.server.stdio import stdio_server

  server = Server("my-server")

  # ... register handlers ...

  async def main():
      async with stdio_server() as (read_stream, write_stream):
          await server.run(
              read_stream,
              write_stream,
              server.create_initialization_options(),
          )

  asyncio.run(main())
  ```

  ```python After theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  from fastmcp import FastMCP

  mcp = FastMCP("my-server")

  # ... register tools, resources, prompts ...

  if __name__ == "__main__":
      mcp.run()
  ```
</CodeGroup>

Need HTTP instead of stdio? With the `Server` class, you'd wire up Starlette routes and `SseServerTransport` or `StreamableHTTPSessionManager`. With FastMCP:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp.run(transport="http", host="0.0.0.0", port=8000)
```

## Tools

This is where the difference is most dramatic. The `Server` class requires two handlers — one to describe your tools (with hand-written JSON Schema) and another to dispatch calls by name. FastMCP eliminates both by deriving everything from your function signature.

<CodeGroup>
  ```python Before theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  import mcp.types as types
  from mcp.server import Server

  server = Server("math")

  @server.list_tools()
  async def list_tools() -> list[types.Tool]:
      return [
          types.Tool(
              name="add",
              description="Add two numbers",
              inputSchema={
                  "type": "object",
                  "properties": {
                      "a": {"type": "number"},
                      "b": {"type": "number"},
                  },
                  "required": ["a", "b"],
              },
          ),
          types.Tool(
              name="multiply",
              description="Multiply two numbers",
              inputSchema={
                  "type": "object",
                  "properties": {
                      "a": {"type": "number"},
                      "b": {"type": "number"},
                  },
                  "required": ["a", "b"],
              },
          ),
      ]

  @server.call_tool()
  async def call_tool(
      name: str, arguments: dict
  ) -> list[types.TextContent]:
      if name == "add":
          result = arguments["a"] + arguments["b"]
          return [types.TextContent(type="text", text=str(result))]
      elif name == "multiply":
          result = arguments["a"] * arguments["b"]
          return [types.TextContent(type="text", text=str(result))]
      raise ValueError(f"Unknown tool: {name}")
  ```

  ```python After theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  from fastmcp import FastMCP

  mcp = FastMCP("math")

  @mcp.tool
  def add(a: float, b: float) -> float:
      """Add two numbers"""
      return a + b

  @mcp.tool
  def multiply(a: float, b: float) -> float:
      """Multiply two numbers"""
      return a * b
  ```
</CodeGroup>

Each `@mcp.tool` function is self-contained: its name becomes the tool name, its docstring becomes the description, its type annotations become the JSON Schema, and its return value is serialized automatically. No routing. No schema dictionaries. No content-type wrappers.

### Type Mapping

When converting your `inputSchema` to Python type hints:

| JSON Schema                                      | Python Type                 |
| ------------------------------------------------ | --------------------------- |
| `{"type": "string"}`                             | `str`                       |
| `{"type": "number"}`                             | `float`                     |
| `{"type": "integer"}`                            | `int`                       |
| `{"type": "boolean"}`                            | `bool`                      |
| `{"type": "array", "items": {"type": "string"}}` | `list[str]`                 |
| `{"type": "object"}`                             | `dict`                      |
| Optional property (not in `required`)            | `param: str \| None = None` |

### Return Values

With the `Server` class, tools return `list[types.TextContent | types.ImageContent | ...]`. In FastMCP, return plain Python values — strings, numbers, dicts, lists, dataclasses, Pydantic models — and serialization is handled for you.

For images or other non-text content, FastMCP provides helpers:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

mcp = FastMCP("media")

@mcp.tool
def create_chart(data: list[float]) -> Image:
    """Generate a chart from data."""
    png_bytes = generate_chart(data)  # your logic
    return Image(data=png_bytes, format="png")
```

## Resources

The `Server` class uses three handlers for resources: `list_resources()` to enumerate them, `list_resource_templates()` for URI templates, and `read_resource()` to serve content — all with manual routing by URI. FastMCP replaces all three with per-resource decorators.

<CodeGroup>
  ```python Before theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  import json
  import mcp.types as types
  from mcp.server import Server
  from pydantic import AnyUrl

  server = Server("data")

  @server.list_resources()
  async def list_resources() -> list[types.Resource]:
      return [
          types.Resource(
              uri=AnyUrl("config://app"),
              name="app_config",
              description="Application configuration",
              mimeType="application/json",
          ),
          types.Resource(
              uri=AnyUrl("config://features"),
              name="feature_flags",
              description="Active feature flags",
              mimeType="application/json",
          ),
      ]

  @server.list_resource_templates()
  async def list_resource_templates() -> list[types.ResourceTemplate]:
      return [
          types.ResourceTemplate(
              uriTemplate="users://{user_id}/profile",
              name="user_profile",
              description="User profile by ID",
          ),
          types.ResourceTemplate(
              uriTemplate="projects://{project_id}/status",
              name="project_status",
              description="Project status by ID",
          ),
      ]

  @server.read_resource()
  async def read_resource(uri: AnyUrl) -> str:
      uri_str = str(uri)
      if uri_str == "config://app":
          return json.dumps({"debug": False, "version": "1.0"})
      if uri_str == "config://features":
          return json.dumps({"dark_mode": True, "beta": False})
      if uri_str.startswith("users://"):
          user_id = uri_str.split("/")[2]
          return json.dumps({"id": user_id, "name": f"User {user_id}"})
      if uri_str.startswith("projects://"):
          project_id = uri_str.split("/")[2]
          return json.dumps({"id": project_id, "status": "active"})
      raise ValueError(f"Unknown resource: {uri}")
  ```

  ```python After theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  import json
  from fastmcp import FastMCP

  mcp = FastMCP("data")

  @mcp.resource("config://app", mime_type="application/json")
  def app_config() -> str:
      """Application configuration"""
      return json.dumps({"debug": False, "version": "1.0"})

  @mcp.resource("config://features", mime_type="application/json")
  def feature_flags() -> str:
      """Active feature flags"""
      return json.dumps({"dark_mode": True, "beta": False})

  @mcp.resource("users://{user_id}/profile")
  def user_profile(user_id: str) -> str:
      """User profile by ID"""
      return json.dumps({"id": user_id, "name": f"User {user_id}"})

  @mcp.resource("projects://{project_id}/status")
  def project_status(project_id: str) -> str:
      """Project status by ID"""
      return json.dumps({"id": project_id, "status": "active"})
  ```
</CodeGroup>

Static resources and URI templates use the same `@mcp.resource` decorator — FastMCP detects `{placeholders}` in the URI and automatically registers a template. The function parameter `user_id` maps directly to the `{user_id}` placeholder.

## Prompts

Same pattern: the `Server` class uses `list_prompts()` and `get_prompt()` with manual routing. FastMCP uses one decorator per prompt.

<CodeGroup>
  ```python Before theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  import mcp.types as types
  from mcp.server import Server

  server = Server("prompts")

  @server.list_prompts()
  async def list_prompts() -> list[types.Prompt]:
      return [
          types.Prompt(
              name="review_code",
              description="Review code for issues",
              arguments=[
                  types.PromptArgument(
                      name="code",
                      description="The code to review",
                      required=True,
                  ),
                  types.PromptArgument(
                      name="language",
                      description="Programming language",
                      required=False,
                  ),
              ],
          )
      ]

  @server.get_prompt()
  async def get_prompt(
      name: str, arguments: dict[str, str] | None
  ) -> types.GetPromptResult:
      if name == "review_code":
          code = (arguments or {}).get("code", "")
          language = (arguments or {}).get("language", "")
          lang_note = f" (written in {language})" if language else ""
          return types.GetPromptResult(
              description="Code review prompt",
              messages=[
                  types.PromptMessage(
                      role="user",
                      content=types.TextContent(
                          type="text",
                          text=f"Please review this code{lang_note}:\n\n{code}",
                      ),
                  )
              ],
          )
      raise ValueError(f"Unknown prompt: {name}")
  ```

  ```python After theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  from fastmcp import FastMCP

  mcp = FastMCP("prompts")

  @mcp.prompt
  def review_code(code: str, language: str | None = None) -> str:
      """Review code for issues"""
      lang_note = f" (written in {language})" if language else ""
      return f"Please review this code{lang_note}:\n\n{code}"
  ```
</CodeGroup>

Returning a `str` from a prompt function automatically wraps it as a user message. For multi-turn prompts, return a `list[Message]`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.prompts import Message

mcp = FastMCP("prompts")

@mcp.prompt
def debug_session(error: str) -> list[Message]:
    """Start a debugging conversation"""
    return [
        Message(f"I'm seeing this error:\n\n{error}"),
        Message("I'll help you debug that. Can you share the relevant code?", role="assistant"),
    ]
```

## Request Context

The `Server` class exposes request context through `server.request_context`, which gives you the raw `ServerSession` for sending notifications. FastMCP replaces this with a typed `Context` object injected into any function that declares it.

<CodeGroup>
  ```python Before theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  import mcp.types as types
  from mcp.server import Server

  server = Server("worker")

  @server.call_tool()
  async def call_tool(name: str, arguments: dict):
      if name == "process_data":
          ctx = server.request_context
          await ctx.session.send_log_message(
              level="info", data="Starting processing..."
          )
          # ... do work ...
          await ctx.session.send_log_message(
              level="info", data="Done!"
          )
          return [types.TextContent(type="text", text="Processed")]
  ```

  ```python After theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  from fastmcp import FastMCP, Context

  mcp = FastMCP("worker")

  @mcp.tool
  async def process_data(ctx: Context) -> str:
      """Process data with progress logging"""
      await ctx.info("Starting processing...")
      # ... do work ...
      await ctx.info("Done!")
      return "Processed"
  ```
</CodeGroup>

The `Context` object provides logging (`ctx.debug()`, `ctx.info()`, `ctx.warning()`, `ctx.error()`), progress reporting (`ctx.report_progress()`), resource subscriptions, session state, and more. See [Context](/servers/context) for the full API.

## Complete Example

A full server upgrade, showing how all the pieces fit together:

<CodeGroup>
  ```python Before expandable theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  import asyncio
  import json
  import mcp.types as types
  from mcp.server import Server
  from mcp.server.stdio import stdio_server
  from pydantic import AnyUrl

  server = Server("demo")

  @server.list_tools()
  async def list_tools() -> list[types.Tool]:
      return [
          types.Tool(
              name="greet",
              description="Greet someone by name",
              inputSchema={
                  "type": "object",
                  "properties": {
                      "name": {"type": "string"},
                  },
                  "required": ["name"],
              },
          )
      ]

  @server.call_tool()
  async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
      if name == "greet":
          return [types.TextContent(type="text", text=f"Hello, {arguments['name']}!")]
      raise ValueError(f"Unknown tool: {name}")

  @server.list_resources()
  async def list_resources() -> list[types.Resource]:
      return [
          types.Resource(
              uri=AnyUrl("info://version"),
              name="version",
              description="Server version",
          )
      ]

  @server.read_resource()
  async def read_resource(uri: AnyUrl) -> str:
      if str(uri) == "info://version":
          return json.dumps({"version": "1.0.0"})
      raise ValueError(f"Unknown resource: {uri}")

  @server.list_prompts()
  async def list_prompts() -> list[types.Prompt]:
      return [
          types.Prompt(
              name="summarize",
              description="Summarize text",
              arguments=[
                  types.PromptArgument(name="text", required=True)
              ],
          )
      ]

  @server.get_prompt()
  async def get_prompt(
      name: str, arguments: dict[str, str] | None
  ) -> types.GetPromptResult:
      if name == "summarize":
          return types.GetPromptResult(
              description="Summarize text",
              messages=[
                  types.PromptMessage(
                      role="user",
                      content=types.TextContent(
                          type="text",
                          text=f"Summarize:\n\n{(arguments or {}).get('text', '')}",
                      ),
                  )
              ],
          )
      raise ValueError(f"Unknown prompt: {name}")

  async def main():
      async with stdio_server() as (read_stream, write_stream):
          await server.run(
              read_stream, write_stream,
              server.create_initialization_options(),
          )

  asyncio.run(main())
  ```

  ```python After theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  import json
  from fastmcp import FastMCP

  mcp = FastMCP("demo")

  @mcp.tool
  def greet(name: str) -> str:
      """Greet someone by name"""
      return f"Hello, {name}!"

  @mcp.resource("info://version")
  def version() -> str:
      """Server version"""
      return json.dumps({"version": "1.0.0"})

  @mcp.prompt
  def summarize(text: str) -> str:
      """Summarize text"""
      return f"Summarize:\n\n{text}"

  if __name__ == "__main__":
      mcp.run()
  ```
</CodeGroup>

## What's Next

Once you've upgraded, you have access to everything FastMCP provides beyond the basics:

* **[Server composition](/servers/composition)** — Mount sub-servers to build modular applications
* **[Middleware](/servers/middleware)** — Add logging, rate limiting, error handling, and caching
* **[Proxy servers](/servers/providers/proxy)** — Create a proxy to any existing MCP server
* **[OpenAPI integration](/integrations/openapi)** — Generate an MCP server from an OpenAPI spec
* **[Authentication](/servers/auth/authentication)** — Built-in OAuth and token verification
* **[Testing](/servers/testing)** — Test your server directly in Python without running a subprocess

Explore the full documentation at [gofastmcp.com](https://gofastmcp.com).


# Upgrading from the MCP SDK
Source: https://gofastmcp.com/getting-started/upgrading/from-mcp-sdk

Upgrade from FastMCP in the MCP Python SDK to the standalone FastMCP framework

If your server starts with `from mcp.server.fastmcp import FastMCP`, you're using FastMCP 1.0 — the version bundled with v1 of the `mcp` package. Upgrading to the standalone FastMCP framework is easy. **For most servers, it's a single import change.**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
from mcp.server.fastmcp import FastMCP

# After
from fastmcp import FastMCP
```

That's it. Your `@mcp.tool`, `@mcp.resource`, and `@mcp.prompt` decorators, your `mcp.run()` call, and the rest of your server code all work as-is.

<Tip>
  **Why upgrade?** FastMCP 1.0 pioneered the Pythonic MCP server experience, and we're proud it was bundled into the `mcp` package. The standalone FastMCP project has since grown into a full framework for taking MCP servers from prototype to production — with composition, middleware, proxy servers, authentication, and much more. Upgrading gives you access to all of that, plus ongoing updates and fixes.
</Tip>

## Install

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install --upgrade fastmcp
# or
uv add fastmcp
```

FastMCP includes the `mcp` package as a dependency, so you don't lose access to anything. Update your import, run your server, and if your tools work, you're done.

<Prompt description="Copy this prompt into any LLM along with your server code to get automated upgrade guidance.">
  You are upgrading an MCP server from FastMCP 1.0 (bundled in the `mcp` package v1) to standalone FastMCP 3.0. Analyze the provided code and identify every change needed. The full upgrade guide is at [https://gofastmcp.com/getting-started/upgrading/from-mcp-sdk](https://gofastmcp.com/getting-started/upgrading/from-mcp-sdk) and the complete FastMCP documentation is at [https://gofastmcp.com](https://gofastmcp.com) — fetch these for complete context.

  STEP 1 — IMPORT (required for all servers):
  Change "from mcp.server.fastmcp import FastMCP" to "from fastmcp import FastMCP".

  STEP 2 — CONSTRUCTOR KWARGS (only if FastMCP() receives transport settings):
  FastMCP() no longer accepts: host, port, log\_level, debug, sse\_path, streamable\_http\_path, json\_response, stateless\_http.
  Fix: pass these to run() instead.
  Before: `mcp = FastMCP("server", host="0.0.0.0", port=8080); mcp.run()`
  After:  `mcp = FastMCP("server"); mcp.run(transport="http", host="0.0.0.0", port=8080)`

  STEP 3 — PROMPTS (only if using PromptMessage directly or returning dicts):
  mcp.types.PromptMessage is replaced by fastmcp.prompts.Message.
  Before: `PromptMessage(role="user", content=TextContent(type="text", text="Hello"))`
  After:  `Message("Hello")`  — role defaults to "user", accepts plain strings.
  Also: if prompts return raw dicts like `{"role": "user", "content": "..."}`, these must become Message objects or plain strings.
  The MCP SDK's FastMCP 1.0 silently coerced dicts; standalone FastMCP requires typed returns.

  STEP 4 — OTHER MCP IMPORTS (only if importing from mcp.\* directly):
  Direct imports from the `mcp` package (e.g., `import mcp.types`, `from mcp.server.stdio import stdio_server`) still work because FastMCP includes `mcp` as a dependency. However, prefer FastMCP's own APIs where equivalents exist:

  * mcp.types.TextContent for tool returns → just return plain Python values (str, int, dict, etc.)
  * mcp.types.ImageContent → fastmcp.utilities.types.Image
  * from mcp.server.stdio import stdio\_server → not needed, mcp.run() handles transport

  STEP 5 — DECORATORS (only if treating decorated functions as objects):
  @mcp.tool, @mcp.resource, @mcp.prompt now return the original function, not a component object. Code that accesses .name or .description on the decorated result needs updating. Set FASTMCP\_DECORATOR\_MODE=object temporarily to restore v1 behavior (this compat setting is itself deprecated).

  For each issue found, show the original line, explain what changed, and provide the corrected code.
</Prompt>

## What Might Need Updating

Most servers need nothing beyond the import change. Skim the sections below to see if any apply.

### Constructor Settings

If you passed transport settings like `host` or `port` directly to `FastMCP()`, those now belong on `run()`. This keeps your server definition independent of how it's deployed:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Before
mcp = FastMCP("my-server", host="0.0.0.0", port=8080)
mcp.run()

# After
mcp = FastMCP("my-server")
mcp.run(transport="http", host="0.0.0.0", port=8080)
```

If you pass the old kwargs, you'll get a clear `TypeError` with a migration hint.

### Prompts

If your prompt functions return `mcp.types.PromptMessage` objects or raw dicts with `role`/`content` keys, you'll need to upgrade to FastMCP's `Message` class. Or just return a plain string — it's automatically wrapped as a user message. The MCP SDK's bundled FastMCP 1.0 silently coerced dicts into messages; standalone FastMCP requires typed `Message` objects or strings.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("prompts")

@mcp.prompt
def review(code: str) -> str:
    """Review code for issues"""
    return f"Please review this code:\n\n{code}"
```

For multi-turn prompts:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.prompts import Message

@mcp.prompt
def debug(error: str) -> list[Message]:
    """Start a debugging session"""
    return [
        Message(f"I'm seeing this error:\n\n{error}"),
        Message("I'll help debug that. Can you share the relevant code?", role="assistant"),
    ]
```

### Other `mcp.*` Imports

If your server imports directly from the `mcp` package — like `import mcp.types` or `from mcp.server.stdio import stdio_server` — those still work. FastMCP includes `mcp` as a dependency, so nothing breaks.

Where FastMCP provides its own API for the same thing, it's worth switching over:

| mcp Package                                       | FastMCP Equivalent                          |
| ------------------------------------------------- | ------------------------------------------- |
| `mcp.types.TextContent(type="text", text=str(x))` | Just return `x` from your tool              |
| `mcp.types.ImageContent(...)`                     | `from fastmcp.utilities.types import Image` |
| `mcp.types.PromptMessage(...)`                    | `from fastmcp.prompts import Message`       |
| `from mcp.server.stdio import stdio_server`       | Not needed — `mcp.run()` handles transport  |

For anything without a FastMCP equivalent (e.g., specific protocol types you use directly), the `mcp.*` import is fine to keep.

### Decorated Functions

In FastMCP 1.0, `@mcp.tool` returned a `FunctionTool` object. Now decorators return your original function unchanged — so decorated functions stay callable for testing, reuse, and composition:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
def greet(name: str) -> str:
    """Greet someone"""
    return f"Hello, {name}!"

# This works now — the function is still a regular function
assert greet("World") == "Hello, World!"
```

If you have code that accesses `.name`, `.description`, or other attributes on the decorated result, that will need updating. This is uncommon — most servers don't interact with the tool object directly. If you need the old behavior temporarily, set `FASTMCP_DECORATOR_MODE=object` to restore it (this compatibility setting is itself deprecated and will be removed in a future release).

## Verify the Upgrade

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Install
pip install --upgrade fastmcp

# Check version
fastmcp version

# Run your server
python my_server.py
```

You can also inspect your server's registered components with the FastMCP CLI:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp inspect my_server.py
```

## Looking Ahead

The MCP ecosystem is evolving fast. Part of FastMCP's job is to absorb that complexity on your behalf — as the protocol and its tooling grow, we do the work so your server code doesn't have to change.
