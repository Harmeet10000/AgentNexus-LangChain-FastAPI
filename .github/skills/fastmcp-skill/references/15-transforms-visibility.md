# Transforms and Visibility

Source lines: 23453-25090 from the original FastMCP documentation dump.

Namespace transforms, prompts/resources as tools, search, tool transforms, versioning, and component visibility controls.

---

# Namespace Transform
Source: https://gofastmcp.com/servers/transforms/namespace

Prefix component names to prevent conflicts

<VersionBadge />

The `Namespace` transform prefixes all component names, preventing conflicts when composing multiple servers.

Tools and prompts receive an underscore-separated prefix. Resources and templates receive a path-segment prefix in their URIs.

| Component | Original      | With `Namespace("api")` |
| --------- | ------------- | ----------------------- |
| Tool      | `my_tool`     | `api_my_tool`           |
| Prompt    | `my_prompt`   | `api_my_prompt`         |
| Resource  | `data://info` | `data://api/info`       |
| Template  | `data://{id}` | `data://api/{id}`       |

The most common use is through the `mount()` method's `namespace` parameter.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

weather = FastMCP("Weather")
calendar = FastMCP("Calendar")

@weather.tool
def get_data() -> str:
    return "Weather data"

@calendar.tool
def get_data() -> str:
    return "Calendar data"

# Without namespacing, these would conflict
main = FastMCP("Main")
main.mount(weather, namespace="weather")
main.mount(calendar, namespace="calendar")

# Clients see: weather_get_data, calendar_get_data
```

You can also apply namespacing directly using the `Namespace` transform.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import Namespace

mcp = FastMCP("Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Namespace all components
mcp.add_transform(Namespace("api"))

# Tool is now: api_greet
```


# Prompts as Tools
Source: https://gofastmcp.com/servers/transforms/prompts-as-tools

Expose prompts to tool-only clients

<VersionBadge />

Some MCP clients only support tools. They cannot list or get prompts directly because they lack prompt protocol support. The `PromptsAsTools` transform bridges this gap by generating tools that provide access to your server's prompts.

When you add `PromptsAsTools` to a server, it creates two tools that clients can call instead of using the prompt protocol:

* **`list_prompts`** returns JSON describing all available prompts and their arguments
* **`get_prompt`** renders a specific prompt with provided arguments

This means any client that can call tools can now access prompts, even if the client has no native prompt support.

## Basic Usage

Pass your server to `PromptsAsTools` when adding the transform. The transform queries that server for prompts whenever the generated tools are called.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import PromptsAsTools

mcp = FastMCP("My Server")

@mcp.prompt
def analyze_code(code: str, language: str = "python") -> str:
    """Analyze code for potential issues."""
    return f"Analyze this {language} code:\n{code}"

@mcp.prompt
def explain_concept(concept: str) -> str:
    """Explain a programming concept."""
    return f"Explain: {concept}"

# Add the transform - creates list_prompts and get_prompt tools
mcp.add_transform(PromptsAsTools(mcp))
```

Clients now see three items: whatever tools you defined directly, plus `list_prompts` and `get_prompt`.

## Listing Prompts

The `list_prompts` tool returns JSON with metadata for each prompt, including its arguments.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
result = await client.call_tool("list_prompts", {})
prompts = json.loads(result.data)
# [
#   {
#     "name": "analyze_code",
#     "description": "Analyze code for potential issues.",
#     "arguments": [
#       {"name": "code", "description": null, "required": true},
#       {"name": "language", "description": null, "required": false}
#     ]
#   },
#   {
#     "name": "explain_concept",
#     "description": "Explain a programming concept.",
#     "arguments": [
#       {"name": "concept", "description": null, "required": true}
#     ]
#   }
#]
```

Each argument includes:

* `name`: The argument name
* `description`: Optional description from type hints or docstrings
* `required`: Whether the argument must be provided

## Getting Prompts

The `get_prompt` tool accepts a prompt name and optional arguments dict. It returns the rendered prompt as JSON with a messages array.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Prompt with required and optional arguments
result = await client.call_tool(
    "get_prompt",
    {
        "name": "analyze_code",
        "arguments": {
            "code": "x = 1\nprint(x)",
            "language": "python"
        }
    }
)

response = json.loads(result.data)
# {
#   "messages": [
#     {
#       "role": "user",
#       "content": "Analyze this python code:\nx = 1\nprint(x)"
#     }
#   ]
# }
```

If a prompt has no arguments, you can omit the `arguments` field or pass an empty dict:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
result = await client.call_tool(
    "get_prompt",
    {"name": "simple_prompt"}
)
```

## Message Format

Rendered prompts return a messages array following the standard MCP format. Each message includes:

* `role`: The message role ("user" or "assistant")
* `content`: The message text content

Multi-message prompts are supported - the array will contain all messages in order.

## Binary Content

Unlike resources, prompts always return text content. There is no binary encoding needed.


# Resources as Tools
Source: https://gofastmcp.com/servers/transforms/resources-as-tools

Expose resources to tool-only clients

<VersionBadge />

Some MCP clients only support tools. They cannot list or read resources directly because they lack resource protocol support. The `ResourcesAsTools` transform bridges this gap by generating tools that provide access to your server's resources.

When you add `ResourcesAsTools` to a server, it creates two tools that clients can call instead of using the resource protocol:

* **`list_resources`** returns JSON describing all available resources and templates
* **`read_resource`** reads a specific resource by URI

This means any client that can call tools can now access resources, even if the client has no native resource support.

## Basic Usage

Pass your server to `ResourcesAsTools` when adding the transform. The transform queries that server for resources whenever the generated tools are called.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import ResourcesAsTools

mcp = FastMCP("My Server")

@mcp.resource("config://app")
def app_config() -> str:
    """Application configuration."""
    return '{"app_name": "My App", "version": "1.0.0"}'

@mcp.resource("user://{user_id}/profile")
def user_profile(user_id: str) -> str:
    """Get a user's profile by ID."""
    return f'{{"user_id": "{user_id}", "name": "User {user_id}"}}'

# Add the transform - creates list_resources and read_resource tools
mcp.add_transform(ResourcesAsTools(mcp))
```

Clients now see three tools: whatever tools you defined directly, plus `list_resources` and `read_resource`.

## Static Resources vs Templates

Resources come in two forms, and the `list_resources` tool distinguishes between them in its JSON output.

Static resources have fixed URIs. They represent concrete data that exists at a known location. In the listing output, static resources include a `uri` field containing the exact URI to request.

Resource templates have parameterized URIs with placeholders like `{user_id}`. They represent patterns for accessing dynamic data. In the listing output, templates include a `uri_template` field showing the pattern with its placeholders.

When a client calls `list_resources`, it receives JSON like this:

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
[
  {
    "uri": "config://app",
    "name": "app_config",
    "description": "Application configuration.",
    "mime_type": "text/plain"
  },
  {
    "uri_template": "user://{user_id}/profile",
    "name": "user_profile",
    "description": "Get a user's profile by ID."
  }
]
```

The client can distinguish resource types by checking which field is present: `uri` for static resources, `uri_template` for templates.

## Reading Resources

The `read_resource` tool accepts a single `uri` argument. For static resources, pass the exact URI. For templates, fill in the placeholders with actual values.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Reading a static resource
result = await client.call_tool("read_resource", {"uri": "config://app"})
print(result.data)  # '{"app_name": "My App", "version": "1.0.0"}'

# Reading a templated resource - fill in {user_id} with an actual ID
result = await client.call_tool("read_resource", {"uri": "user://42/profile"})
print(result.data)  # '{"user_id": "42", "name": "User 42"}'
```

The transform handles template matching automatically. When you request `user://42/profile`, it matches against the `user://{user_id}/profile` template, extracts `user_id=42`, and calls your resource function with that parameter.

## Binary Content

Resources that return binary data (like images or files) are automatically base64-encoded when read through the `read_resource` tool. This ensures binary content can be transmitted as a string in the tool response.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.resource("data://binary", mime_type="application/octet-stream")
def binary_data() -> bytes:
    return b"\x00\x01\x02\x03"

# Client receives base64-encoded string
result = await client.call_tool("read_resource", {"uri": "data://binary"})
decoded = base64.b64decode(result.data)  # b'\x00\x01\x02\x03'
```


# Tool Search
Source: https://gofastmcp.com/servers/transforms/tool-search

Replace large tool catalogs with on-demand search

<VersionBadge />

When a server exposes hundreds or thousands of tools, sending the full catalog to an LLM wastes tokens and degrades tool selection accuracy. Search transforms solve this by replacing the tool listing with a search interface — the LLM discovers tools on demand instead of receiving everything upfront.

## How It Works

When you add a search transform, `list_tools()` returns just two synthetic tools instead of the full catalog:

* **`search_tools`** finds tools matching a query and returns their full definitions
* **`call_tool`** executes a discovered tool by name

The original tools are still callable. They're hidden from the listing but remain fully functional — the search transform controls *discovery*, not *access*.

Both synthetic tools search across tool names, descriptions, parameter names, and parameter descriptions. A search for `"email"` would match a tool named `send_email`, a tool with "email" in its description, or a tool with an `email_address` parameter.

Search results are returned in the same JSON format as `list_tools`, including the full input schema, so the LLM can construct valid calls immediately without a second round-trip.

## Search Strategies

FastMCP provides two search transforms. They share the same interface — two synthetic tools, same configuration options — but differ in how they match queries to tools.

### Regex Search

`RegexSearchTransform` matches tools against a regex pattern using case-insensitive `re.search`. It has zero overhead and no index to build, making it a good default when the LLM knows roughly what it's looking for.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms.search import RegexSearchTransform

mcp = FastMCP("My Server", transforms=[RegexSearchTransform()])

@mcp.tool
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the database for records matching the query."""
    ...

@mcp.tool
def delete_record(record_id: str) -> bool:
    """Delete a record from the database by its ID."""
    ...

@mcp.tool
def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email to the given recipient."""
    ...
```

The LLM's `search_tools` call takes a `pattern` parameter — a regex string:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Exact substring match
result = await client.call_tool("search_tools", {"pattern": "database"})
# Returns: search_database, delete_record

# Regex pattern
result = await client.call_tool("search_tools", {"pattern": "send.*email|notify"})
# Returns: send_email
```

Results are returned in catalog order. If the pattern is invalid regex, the search returns an empty list rather than raising an error.

### BM25 Search

`BM25SearchTransform` ranks tools by relevance using the [BM25 Okapi](https://en.wikipedia.org/wiki/Okapi_BM25) algorithm. It's better for natural language queries because it scores each tool based on term frequency and document rarity, returning results ranked by relevance rather than filtering by match/no-match.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms.search import BM25SearchTransform

mcp = FastMCP("My Server", transforms=[BM25SearchTransform()])

# ... define tools ...
```

The LLM's `search_tools` call takes a `query` parameter — natural language:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
result = await client.call_tool("search_tools", {
    "query": "tools for deleting things from the database"
})
# Returns: delete_record ranked first, search_database second
```

BM25 builds an in-memory index from the searchable text of all tools. The index is created lazily on the first search and automatically rebuilt whenever the tool catalog changes — for example, when tools are added, removed, or have their descriptions updated. The staleness check is based on a hash of all searchable text, so description changes are detected even when tool names stay the same.

### Which to Choose

Use **regex** when your LLM is good at constructing targeted patterns and you want deterministic, predictable results. Regex is also simpler to debug — you can see exactly what pattern was sent.

Use **BM25** when your LLM tends to describe what it needs in natural language, or when your tool catalog has nuanced descriptions where relevance ranking adds value. BM25 handles partial matches and synonyms better because it scores on individual terms rather than requiring a single pattern to match.

## Configuration

Both search transforms accept the same configuration options.

### Limiting Results

By default, search returns at most 5 tools. Adjust `max_results` based on your catalog size and how much context you want the LLM to receive per search:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp.add_transform(RegexSearchTransform(max_results=10))
mcp.add_transform(BM25SearchTransform(max_results=3))
```

With regex, results stop as soon as the limit is reached (first N matches in catalog order). With BM25, all tools are scored and the top N by relevance are returned.

### Pinning Tools

Some tools should always be visible regardless of search. Use `always_visible` to pin them in the listing alongside the synthetic tools:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp.add_transform(RegexSearchTransform(
    always_visible=["help", "status"],
))

# list_tools returns: help, status, search_tools, call_tool
```

Pinned tools appear directly in `list_tools` so the LLM can call them without searching. They're excluded from search results to avoid duplication.

### Custom Tool Names

The default names `search_tools` and `call_tool` can be changed to avoid conflicts with real tools:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp.add_transform(RegexSearchTransform(
    search_tool_name="find_tools",
    call_tool_name="run_tool",
))
```

## The `call_tool` Proxy

The `call_tool` proxy forwards calls to the real tool. When a client calls `call_tool(name="search_database", arguments={...})`, the proxy resolves `search_database` through the server's normal tool pipeline — including transforms and middleware — and executes it.

The proxy rejects attempts to call the synthetic tools themselves. `call_tool(name="call_tool")` raises an error rather than recursing.

<Note>
  Tools discovered through search can also be called directly via `client.call_tool("search_database", {...})` without going through the proxy. The proxy exists for LLMs that only know about the tools returned by `list_tools` and need a way to invoke discovered tools through a tool they can see.
</Note>

## Auth and Visibility

Search results respect the full authorization pipeline. Tools filtered by middleware, visibility transforms, or component-level auth checks won't appear in search results.

The search tool queries `list_tools()` through the complete pipeline at search time, so the same filtering that controls what a client sees in the listing also controls what they can discover through search.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.transforms import Visibility
from fastmcp.server.transforms.search import RegexSearchTransform

mcp = FastMCP("My Server")

# ... define tools ...

# Disable admin tools globally
mcp.add_transform(Visibility(False, tags={"admin"}))

# Add search — admin tools won't appear in results
mcp.add_transform(RegexSearchTransform())
```

Session-level visibility changes (via `ctx.disable_components()`) are also reflected immediately in search results.


# Tool Transformation
Source: https://gofastmcp.com/servers/transforms/tool-transformation

Modify tool schemas - rename, reshape arguments, and customize behavior

<VersionBadge />

Tool transformation lets you modify tool schemas - renaming tools, changing descriptions, adjusting tags, and reshaping argument schemas. FastMCP provides two mechanisms that share the same configuration options but differ in timing.

**Deferred transformation** with `ToolTransform` applies modifications when tools flow through a transform chain. Use this for tools from mounted servers, proxies, or other providers where you don't control the source directly.

**Immediate transformation** with `Tool.from_tool()` creates a modified tool object right away. Use this when you have direct access to a tool and want to transform it before registration.

## ToolTransform

The `ToolTransform` class is a transform that modifies tools as they flow through a provider. Provide a dictionary mapping original tool names to their transformation configuration.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import ToolTransform
from fastmcp.tools.tool_transform import ToolTransformConfig

mcp = FastMCP("Server")

@mcp.tool
def verbose_internal_data_fetcher(query: str) -> str:
    """Fetches data from the internal database."""
    return f"Results for: {query}"

# Rename the tool to something simpler
mcp.add_transform(ToolTransform({
    "verbose_internal_data_fetcher": ToolTransformConfig(
        name="search",
        description="Search the database.",
    )
}))

# Clients see "search" with the cleaner description
```

`ToolTransform` is useful when you want to modify tools from mounted or proxied servers without changing the original source.

## Tool.from\_tool()

Use `Tool.from_tool()` when you have the tool object and want to create a transformed version for registration.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.tools import Tool, tool
from fastmcp.tools.tool_transform import ArgTransform

# Create a tool without registering it
@tool
def search(q: str, limit: int = 10) -> list[str]:
    """Search for items."""
    return [f"Result {i} for {q}" for i in range(limit)]

# Transform it before registration
better_search = Tool.from_tool(
    search,
    name="find_items",
    description="Find items matching your search query.",
    transform_args={
        "q": ArgTransform(
            name="query",
            description="The search terms to look for.",
        ),
    },
)

mcp = FastMCP("Server")
mcp.add_tool(better_search)
```

The standalone `@tool` decorator (from `fastmcp.tools`) creates a Tool object without registering it to any server. This separates creation from registration, letting you transform tools before deciding where they go.

## Modification Options

Both mechanisms support the same modifications.

**Tool-level options:**

| Option        | Description                                             |
| ------------- | ------------------------------------------------------- |
| `name`        | New name for the tool                                   |
| `description` | New description                                         |
| `title`       | Human-readable title                                    |
| `tags`        | Set of tags for categorization                          |
| `annotations` | MCP ToolAnnotations                                     |
| `meta`        | Custom metadata dictionary                              |
| `enabled`     | Whether the tool is visible to clients (default `True`) |

**Argument-level options** (via `ArgTransform` or `ArgTransformConfig`):

| Option            | Description                                              |
| ----------------- | -------------------------------------------------------- |
| `name`            | Rename the argument                                      |
| `description`     | New description for the argument                         |
| `default`         | New default value                                        |
| `default_factory` | Callable that generates a default (requires `hide=True`) |
| `hide`            | Remove from client-visible schema                        |
| `required`        | Make an optional argument required                       |
| `type`            | Change the argument's type                               |
| `examples`        | Example values for the argument                          |

## Hiding Arguments

Hide arguments to simplify the interface or inject values the client shouldn't control.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.tools.tool_transform import ArgTransform

# Hide with a constant value
transform_args = {
    "api_key": ArgTransform(hide=True, default="secret-key"),
}

# Hide with a dynamic value
import uuid
transform_args = {
    "request_id": ArgTransform(hide=True, default_factory=lambda: str(uuid.uuid4())),
}
```

Hidden arguments disappear from the tool's schema. The client never sees them, but the underlying function receives the configured value.

<Warning>
  `default_factory` requires `hide=True`. Visible arguments need static defaults that can be represented in JSON Schema.
</Warning>

## Renaming Arguments

Rename arguments to make them more intuitive for LLMs or match your API conventions.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.tools import Tool, tool
from fastmcp.tools.tool_transform import ArgTransform

@tool
def search(q: str, n: int = 10) -> list[str]:
    """Search for items."""
    return []

better_search = Tool.from_tool(
    search,
    transform_args={
        "q": ArgTransform(name="query", description="Search terms"),
        "n": ArgTransform(name="max_results", description="Maximum results to return"),
    },
)
```

## Custom Transform Functions

For advanced scenarios, provide a `transform_fn` that intercepts tool execution. The function can validate inputs, modify outputs, or add custom logic while still calling the original tool via `forward()`.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.tools import Tool, tool
from fastmcp.tools.tool_transform import forward, ArgTransform

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b

async def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise ValueError("Cannot divide by zero")
    return await forward(numerator=numerator, denominator=denominator)

safe_division = Tool.from_tool(
    divide,
    name="safe_divide",
    transform_fn=safe_divide,
    transform_args={
        "a": ArgTransform(name="numerator"),
        "b": ArgTransform(name="denominator"),
    },
)

mcp = FastMCP("Server")
mcp.add_tool(safe_division)
```

The `forward()` function handles argument mapping automatically. Call it with the transformed argument names, and it maps them back to the original function's parameters.

For direct access to the original function without mapping, use `forward_raw()` with the original parameter names.

## Context-Aware Tool Factories

You can write functions that act as "factories," generating specialized versions of a tool for different contexts. For example, create a `get_my_data` tool for the current user by hiding the `user_id` parameter and providing it automatically.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.tools import Tool, tool
from fastmcp.tools.tool_transform import ArgTransform

# A generic tool that requires a user_id
@tool
def get_user_data(user_id: str, query: str) -> str:
    """Fetch data for a specific user."""
    return f"Data for user {user_id}: {query}"


def create_user_tool(user_id: str) -> Tool:
    """Factory that creates a user-specific version of get_user_data."""
    return Tool.from_tool(
        get_user_data,
        name="get_my_data",
        description="Fetch your data. No need to specify a user ID.",
        transform_args={
            "user_id": ArgTransform(hide=True, default=user_id),
        },
    )


# Create a server with a tool customized for the current user
mcp = FastMCP("User Server")
current_user_id = "user-123"  # e.g., from auth context
mcp.add_tool(create_user_tool(current_user_id))

# Clients see "get_my_data(query: str)" — user_id is injected automatically
```

This pattern is useful for multi-tenant servers where each connection gets tools pre-configured with their identity, or for wrapping generic tools with environment-specific defaults.


# Transforms Overview
Source: https://gofastmcp.com/servers/transforms/transforms

Modify components as they flow through your server

<VersionBadge />

Transforms modify components as they flow from providers to clients. When a client asks "what tools do you have?", the request passes through each transform in the chain. Each transform can modify the components before passing them along.

## Mental Model

Think of transforms as filters in a pipeline. Components flow from providers through transforms to reach clients:

```
Provider → [Transform A] → [Transform B] → Client
```

When listing components, transforms receive sequences and return transformed sequences—a pure function pattern. When getting a specific component by name, transforms use a middleware pattern with `call_next`, working in reverse: mapping the client's requested name back to the original, then transforming the result.

## Built-in Transforms

FastMCP provides several transforms for common use cases:

* **[Namespace](/servers/transforms/namespace)** - Prefix component names to prevent conflicts when composing servers
* **[Tool Transformation](/servers/transforms/tool-transformation)** - Rename tools, modify descriptions, reshape arguments
* **[Enabled](/servers/visibility)** - Control which components are visible at runtime
* **[Tool Search](/servers/transforms/tool-search)** - Replace large tool catalogs with on-demand search
* **[Resources as Tools](/servers/transforms/resources-as-tools)** - Expose resources to tool-only clients
* **[Prompts as Tools](/servers/transforms/prompts-as-tools)** - Expose prompts to tool-only clients
* **[Code Mode (Experimental)](/servers/transforms/code-mode)** - Replace many tools with programmable `search` + `execute`

## Server vs Provider Transforms

Transforms can be added at two levels, each serving different purposes.

### Provider-Level Transforms

Provider transforms apply to components from a specific provider. They run first, modifying components before they reach the server level.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.providers import FastMCPProvider
from fastmcp.server.transforms import Namespace, ToolTransform
from fastmcp.tools.tool_transform import ToolTransformConfig

sub_server = FastMCP("Sub")

@sub_server.tool
def process(data: str) -> str:
    return f"Processed: {data}"

# Create provider and add transforms
provider = FastMCPProvider(sub_server)
provider.add_transform(Namespace("api"))
provider.add_transform(ToolTransform({
    "api_process": ToolTransformConfig(description="Process data through the API"),
}))

main = FastMCP("Main", providers=[provider])
# Tool is now: api_process with updated description
```

When using `mount()`, the returned provider reference lets you add transforms directly.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
main = FastMCP("Main")
mount = main.mount(sub_server, namespace="api")
mount.add_transform(ToolTransform({...}))
```

### Server-Level Transforms

Server transforms apply to all components from all providers. They run after provider transforms, seeing the already-transformed names.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import Namespace

mcp = FastMCP("Server", transforms=[Namespace("v1")])

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

# All tools become v1_toolname
```

Server-level transforms are useful for API versioning or applying consistent naming across your entire server.

### Transform Order

Transforms stack in the order they're added. The first transform added is innermost (closest to the provider), and subsequent transforms wrap it.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.providers import FastMCPProvider
from fastmcp.server.transforms import Namespace, ToolTransform
from fastmcp.tools.tool_transform import ToolTransformConfig

provider = FastMCPProvider(server)
provider.add_transform(Namespace("api"))           # Applied first
provider.add_transform(ToolTransform({             # Sees namespaced names
    "api_verbose_name": ToolTransformConfig(name="short"),
}))

# Flow: "verbose_name" -> "api_verbose_name" -> "short"
```

When a client requests "short", the transforms reverse the mapping: ToolTransform maps "short" to "api\_verbose\_name", then Namespace strips the prefix to find "verbose\_name" in the provider.

## Custom Transforms

Create custom transforms by subclassing `Transform` and overriding the methods you need.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from collections.abc import Sequence
from fastmcp.server.transforms import Transform, GetToolNext
from fastmcp.tools.tool import Tool

class TagFilter(Transform):
    """Filter tools to only those with specific tags."""

    def __init__(self, required_tags: set[str]):
        self.required_tags = required_tags

    async def list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        return [t for t in tools if t.tags & self.required_tags]

    async def get_tool(self, name: str, call_next: GetToolNext) -> Tool | None:
        tool = await call_next(name)
        if tool and tool.tags & self.required_tags:
            return tool
        return None
```

The `Transform` base class provides default implementations that pass through unchanged. Override only the methods relevant to your transform.

Each component type has two methods with different patterns:

| Method                                  | Pattern       | Purpose                             |
| --------------------------------------- | ------------- | ----------------------------------- |
| `list_tools(tools)`                     | Pure function | Transform the sequence of tools     |
| `get_tool(name, call_next)`             | Middleware    | Transform lookup by name            |
| `list_resources(resources)`             | Pure function | Transform the sequence of resources |
| `get_resource(uri, call_next)`          | Middleware    | Transform lookup by URI             |
| `list_resource_templates(templates)`    | Pure function | Transform the sequence of templates |
| `get_resource_template(uri, call_next)` | Middleware    | Transform template lookup by URI    |
| `list_prompts(prompts)`                 | Pure function | Transform the sequence of prompts   |
| `get_prompt(name, call_next)`           | Middleware    | Transform lookup by name            |

List methods receive sequences directly and return transformed sequences. Get methods use `call_next` for routing flexibility—when a client requests "new\_name", your transform maps it back to "original\_name" before calling `call_next()`.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
class PrefixTransform(Transform):
    def __init__(self, prefix: str):
        self.prefix = prefix

    async def list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        return [t.model_copy(update={"name": f"{self.prefix}_{t.name}"}) for t in tools]

    async def get_tool(self, name: str, call_next: GetToolNext) -> Tool | None:
        # Reverse the prefix to find the original
        if not name.startswith(f"{self.prefix}_"):
            return None
        original = name[len(self.prefix) + 1:]
        tool = await call_next(original)
        if tool:
            return tool.model_copy(update={"name": name})
        return None
```


# Versioning
Source: https://gofastmcp.com/servers/versioning

Serve multiple API versions from a single codebase

<VersionBadge />

Component versioning lets you maintain multiple implementations of the same tool, resource, or prompt under a single identifier. You register each version, and FastMCP handles the rest: clients see the highest version by default, but you can filter to expose exactly the versions you want.

The primary use case is serving different API versions from one codebase. Instead of maintaining separate deployments for v1 and v2 clients, you version your components and use `VersionFilter` to create distinct API surfaces.

## Versioned API Surfaces

Consider a server that needs to support both v1 and v2 clients. The v2 API adds new parameters to existing tools, and you want both versions to coexist cleanly. Define your components on a shared provider, then create separate servers with different version filters.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.providers import LocalProvider
from fastmcp.server.transforms import VersionFilter

# Define versioned components on a shared provider
components = LocalProvider()

@components.tool(version="1.0")
def calculate(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

@components.tool(version="2.0")
def calculate(x: int, y: int, z: int = 0) -> int:
    """Add two or three numbers."""
    return x + y + z

# Create servers that share the provider with different filters
api_v1 = FastMCP("API v1", providers=[components])
api_v1.add_transform(VersionFilter(version_lt="2.0"))

api_v2 = FastMCP("API v2", providers=[components])
api_v2.add_transform(VersionFilter(version_gte="2.0"))
```

Clients connecting to `api_v1` see the two-argument `calculate`. Clients connecting to `api_v2` see the three-argument version. Both servers share the same component definitions.

`VersionFilter` accepts two keyword-only parameters that mirror comparison operators: `version_gte` (greater than or equal) and `version_lt` (less than). You can use either or both to define your version range.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Versions < 3.0 (v1.x and v2.x)
VersionFilter(version_lt="3.0")

# Versions >= 2.0 (v2.x and later)
VersionFilter(version_gte="2.0")

# Versions in range [2.0, 3.0) (only v2.x)
VersionFilter(version_gte="2.0", version_lt="3.0")
```

<Note>
  **Unversioned components are exempt from version filtering by default.** Set `include_unversioned=False` to exclude them. Including them by default ensures that adding version filtering to a server with mixed versioned and unversioned components doesn't accidentally hide the unversioned ones. To prevent confusion, FastMCP forbids mixing versioned and unversioned components with the same name.
</Note>

### Filtering Mounted Servers

When you mount child servers and apply a `VersionFilter` to the parent, the filter applies to components from mounted servers as well. Range filtering (`version_gte` and `version_lt`) is handled at the provider level, meaning mounted servers don't need to know about the parent's version constraints.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import VersionFilter

# Child server with versioned components
child = FastMCP("Child")

@child.tool(version="1.0")
def process(data: str) -> str:
    return data.upper()

@child.tool(version="2.0")
def process(data: str, mode: str = "default") -> str:
    return data.upper() if mode == "default" else data.lower()

# Parent server mounts child and applies version filter
parent = FastMCP("Parent")
parent.mount(child, namespace="child")
parent.add_transform(VersionFilter(version_lt="2.0"))

# Clients see only child_process v1.0
```

The parent's `VersionFilter` sees components after they've been namespaced, but filters based on version regardless of namespace. This lets you apply version policies consistently across your entire server hierarchy.

## Declaring Versions

Add a `version` parameter to any component decorator. FastMCP stores versions as strings and groups components by their identifier (name for tools and prompts, URI for resources).

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool(version="1.0")
def process(data: str) -> str:
    """Original processing."""
    return data.upper()

@mcp.tool(version="2.0")
def process(data: str, mode: str = "default") -> str:
    """Enhanced processing with mode selection."""
    if mode == "reverse":
        return data[::-1].upper()
    return data.upper()
```

Both versions are registered. When a client lists tools, they see only `process` with version 2.0 (the highest). When they invoke `process`, version 2.0 executes. The same pattern applies to resources and prompts.

### Versioned vs Unversioned Components

For any given component name, you must choose one approach: either version all implementations or version none of them. Mixing versioned and unversioned components with the same name raises an error at registration time.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool
def calculate(x: int, y: int) -> int:
    """Unversioned tool."""
    return x + y

@mcp.tool(version="2.0")  # Raises ValueError
def calculate(x: int, y: int, z: int = 0) -> int:
    """Cannot mix versioned with unversioned."""
    return x + y + z
```

The error message explains the conflict: "Cannot add versioned tool 'calculate' (version='2.0'): an unversioned tool with this name already exists. Either version all components or none."

This restriction helps keep version filtering behavior predictable.

Resources and prompts follow the same pattern.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.resource("config://app", version="1.0")
def config_v1() -> str:
    return '{"format": "legacy"}'

@mcp.resource("config://app", version="2.0")
def config_v2() -> str:
    return '{"format": "modern", "schema": "v2"}'

@mcp.prompt(version="1.0")
def summarize(text: str) -> str:
    return f"Summarize: {text}"

@mcp.prompt(version="2.0")
def summarize(text: str, style: str = "concise") -> str:
    return f"Summarize in a {style} style: {text}"
```

### Version Discovery

When clients list components, each versioned component includes metadata about all available versions. This lets clients discover what versions exist before deciding which to use. The `meta.fastmcp.versions` field contains all registered versions sorted from highest to lowest.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

async with Client(server) as client:
    tools = await client.list_tools()

    for tool in tools:
        if tool.meta:
            fastmcp_meta = tool.meta.get("fastmcp", {})
            # Current version being returned (highest by default)
            print(f"Version: {fastmcp_meta.get('version')}")
            # All available versions for this component
            print(f"Available: {fastmcp_meta.get('versions')}")
```

For a tool with versions `"1.0"` and `"2.0"`, listing returns the `2.0` implementation with `meta.fastmcp.version` set to `"2.0"` and `meta.fastmcp.versions` set to `["2.0", "1.0"]`. Unversioned components omit these fields entirely.

This discovery mechanism enables clients to make informed decisions about which version to request, support graceful degradation when newer versions introduce breaking changes, or display version information in developer tools.

## Requesting Specific Versions

By default, clients receive and invoke the highest version of each component. When you need a specific version, FastMCP provides two approaches: the FastMCP client API for Python applications, and the MCP protocol mechanism for any MCP-compatible client.

### FastMCP Client

The FastMCP client's `call_tool` and `get_prompt` methods accept an optional `version` parameter. When specified, the server executes that exact version instead of the highest.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

async with Client(server) as client:
    # Call the highest version (default behavior)
    result = await client.call_tool("calculate", {"x": 1, "y": 2})

    # Call a specific version
    result_v1 = await client.call_tool("calculate", {"x": 1, "y": 2}, version="1.0")

    # Get a specific prompt version
    prompt = await client.get_prompt("summarize", {"text": "..."}, version="1.0")
```

If the requested version doesn't exist, the server raises a `NotFoundError`. This ensures you get exactly what you asked for rather than silently falling back to a different version.

### MCP Protocol

For generic MCP clients that don't have built-in version support, pass the version through the `_meta` field in arguments. FastMCP servers extract the version from `_meta.fastmcp.version` before processing.

<CodeGroup>
  ```json Tool Call Arguments theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  {
    "x": 1,
    "y": 2,
    "_meta": {
      "fastmcp": {
        "version": "1.0"
      }
    }
  }
  ```

  ```json Prompt Arguments theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  {
    "text": "Summarize this document...",
    "_meta": {
      "fastmcp": {
        "version": "1.0"
      }
    }
  }
  ```
</CodeGroup>

The `_meta` field is part of the MCP request params, not arguments, so your component implementation never sees it. This convention allows version selection to work across any MCP client without requiring protocol changes. The FastMCP client handles this automatically when you pass the `version` parameter.

## Version Comparison

FastMCP compares versions to determine which is "highest" when multiple versions share an identifier. The comparison behavior depends on the version format.

For [PEP 440](https://peps.python.org/pep-0440/) versions (like `"1.0"`, `"2.1.3"`, `"1.0a1"`), FastMCP uses semantic comparison where numeric segments are compared as numbers.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# PEP 440 versions compare semantically
"1" < "2" < "10"           # Numeric order (not "1" < "10" < "2")
"1.9" < "1.10"             # Numeric order (not "1.10" < "1.9")
"1.0a1" < "1.0b1" < "1.0"  # Pre-releases sort before releases
```

For other formats (dates, custom schemes), FastMCP falls back to lexicographic string comparison. This works well for ISO dates and other naturally sortable formats.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Non-PEP 440 versions compare as strings
"2025-01-15" < "2025-02-01"  # ISO dates sort correctly
"alpha" < "beta"             # Alphabetical order
```

The `v` prefix is stripped before comparison, so `"v1.0"` and `"1.0"` are treated as equal for sorting purposes.

## Retrieving Specific Versions

Server-side code can retrieve specific versions rather than just the highest. This is useful during migrations when you need to compare behavior between versions or access legacy implementations.

The `get_tool`, `get_resource`, and `get_prompt` methods accept an optional `version` parameter. Without it, they return the highest version. With it, they return exactly that version.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool(version="1.0")
def add(x: int, y: int) -> int:
    return x + y

@mcp.tool(version="2.0")
def add(x: int, y: int) -> int:
    return x + y + 100  # Different behavior

# Get highest version (default)
tool = await mcp.get_tool("add")
print(tool.version)  # "2.0"

# Get specific version
tool_v1 = await mcp.get_tool("add", version="1.0")
print(tool_v1.version)  # "1.0"
```

If the requested version doesn't exist, a `NotFoundError` is raised.

## Removing Versions

The `remove_tool`, `remove_resource`, and `remove_prompt` methods on the server's [local provider](/servers/providers/local) accept an optional `version` parameter that controls what gets removed.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Remove ALL versions of a component
mcp.local_provider.remove_tool("calculate")

# Remove only a specific version
mcp.local_provider.remove_tool("calculate", version="1.0")
```

When you remove a specific version, other versions remain registered. When you remove without specifying a version, all versions are removed.

## Migration Workflow

Versioning supports gradual migration when updating component behavior. You can deploy new versions alongside old ones, verify the new behavior works correctly, then clean up.

When migrating an existing unversioned component to use versioning, start by assigning an initial version to your existing implementation. Then add the new version alongside it.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool(version="1.0")
def process_data(input: str) -> str:
    """Original implementation, now versioned."""
    return legacy_process(input)

@mcp.tool(version="2.0")
def process_data(input: str, options: dict | None = None) -> str:
    """Updated implementation with new options parameter."""
    return modern_process(input, options or {})
```

Clients automatically see version 2.0 (the highest). During the transition, your server code can still access the original implementation via `get_tool("process_data", version="1.0")`.

Once the migration is complete, remove the old version.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp.local_provider.remove_tool("process_data", version="1.0")
```


# Component Visibility
Source: https://gofastmcp.com/servers/visibility

Control which components are available to clients

<VersionBadge />

Components can be dynamically enabled or disabled at runtime. A disabled tool disappears from listings and cannot be called. This enables runtime access control, feature flags, and context-aware component exposure.

## Component Visibility

Every FastMCP server provides `enable()` and `disable()` methods for controlling component availability.

### Disabling Components

The `disable()` method marks components as disabled. Disabled components are filtered out from all client queries.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("Server")

@mcp.tool(tags={"admin"})
def delete_everything() -> str:
    """Delete all data."""
    return "Deleted"

@mcp.tool(tags={"admin"})
def reset_system() -> str:
    """Reset the system."""
    return "Reset"

@mcp.tool
def get_status() -> str:
    """Get system status."""
    return "OK"

# Disable admin tools
mcp.disable(tags={"admin"})

# Clients only see: get_status
```

### Enabling Components

The `enable()` method re-enables previously disabled components.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Re-enable admin tools
mcp.enable(tags={"admin"})

# Clients now see all three tools
```

## Keys and Tags

Visibility filtering works with two identifiers: keys (for specific components) and tags (for groups).

### Component Keys

Every component has a unique key in the format `{type}:{identifier}`.

| Component | Key Format       | Example                  |
| --------- | ---------------- | ------------------------ |
| Tool      | `tool:{name}`    | `tool:delete_everything` |
| Resource  | `resource:{uri}` | `resource:data://config` |
| Template  | `template:{uri}` | `template:file://{path}` |
| Prompt    | `prompt:{name}`  | `prompt:analyze`         |

Use keys to target specific components.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Disable a specific tool
mcp.disable(keys={"tool:delete_everything"})

# Disable multiple specific components
mcp.disable(keys={"tool:reset_system", "resource:data://secrets"})
```

### Tags

Tags group components for bulk operations. Define tags when creating components, then filter by them.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("Server")

@mcp.tool(tags={"public", "read"})
def get_data() -> str:
    return "data"

@mcp.tool(tags={"admin", "write"})
def set_data(value: str) -> str:
    return f"Set: {value}"

@mcp.tool(tags={"admin", "dangerous"})
def delete_data() -> str:
    return "Deleted"

# Disable all admin tools
mcp.disable(tags={"admin"})

# Disable all dangerous tools (some overlap with admin)
mcp.disable(tags={"dangerous"})
```

A component is disabled if it has **any** of the disabled tags. The component doesn't need all the tags; one match is enough.

### Combining Keys and Tags

You can specify both keys and tags in a single call. The filters combine additively.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Disable specific tools AND all dangerous-tagged components
mcp.disable(keys={"tool:debug_info"}, tags={"dangerous"})
```

## Allowlist Mode

By default, visibility filtering uses blocklist mode: everything is enabled unless explicitly disabled. The `only=True` parameter switches to allowlist mode, where **only** specified components are enabled.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("Server")

@mcp.tool(tags={"safe"})
def read_only_operation() -> str:
    return "Read"

@mcp.tool(tags={"safe"})
def list_items() -> list[str]:
    return ["a", "b", "c"]

@mcp.tool(tags={"dangerous"})
def delete_all() -> str:
    return "Deleted"

@mcp.tool
def untagged_tool() -> str:
    return "Untagged"

# Only enable safe tools - everything else is disabled
mcp.enable(tags={"safe"}, only=True)

# Clients see: read_only_operation, list_items
# Disabled: delete_all, untagged_tool
```

Allowlist mode is useful for restrictive environments where you want to explicitly opt-in components rather than opt-out.

### Allowlist Behavior

When you call `enable(only=True)`:

1. Default visibility state switches to "disabled"
2. Previous allowlists are cleared
3. Only specified keys/tags become enabled

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Start fresh - only enable these specific tools
mcp.enable(keys={"tool:safe_read", "tool:safe_write"}, only=True)

# Later, switch to a different allowlist
mcp.enable(tags={"production"}, only=True)
```

### Ordering and Overrides

Later `enable()` and `disable()` calls override earlier ones. This lets you create broad rules with specific exceptions.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp.enable(tags={"api"}, only=True)  # Allow all api-tagged
mcp.disable(keys={"tool:api_admin"})  # Later disable overrides for this tool

# api_admin is disabled because the later disable() overrides the allowlist
```

You can always re-enable something that was disabled by adding another `enable()` call after it.

## Server vs Provider

Visibility state operates at two levels: the server and individual providers.

### Server-Level

Server-level visibility state applies to all components from all providers. When you call `mcp.enable()` or `mcp.disable()`, you're filtering the final view that clients see.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

main = FastMCP("Main")
main.mount(sub_server, namespace="api")

@main.tool(tags={"internal"})
def local_debug() -> str:
    return "Debug"

# Disable internal tools from ALL sources
main.disable(tags={"internal"})
```

### Provider-Level

Each provider can add its own visibility transforms. These run before server-level transforms, so the server can override provider-level disables.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.providers import LocalProvider

# Create provider with visibility control
admin_tools = LocalProvider()

@admin_tools.tool(tags={"admin"})
def admin_action() -> str:
    return "Admin"

@admin_tools.tool
def regular_action() -> str:
    return "Regular"

# Disable at provider level
admin_tools.disable(tags={"admin"})

# Server can override if needed
mcp = FastMCP("Server", providers=[admin_tools])
mcp.enable(names={"admin_action"})  # Re-enables despite provider disable
```

Provider-level transforms are useful for setting default visibility that servers can selectively override.

### Layered Transforms

Provider transforms run first, then server transforms. Later transforms override earlier ones, so the server has final say.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.providers import LocalProvider

provider = LocalProvider()

@provider.tool(tags={"feature", "beta"})
def new_feature() -> str:
    return "New"

# Provider enables feature-tagged
provider.enable(tags={"feature"}, only=True)

# Server disables beta-tagged (runs after provider)
mcp = FastMCP("Server", providers=[provider])
mcp.disable(tags={"beta"})

# new_feature is disabled (server's later disable overrides provider's enable)
```

## Per-Session Visibility

Server-level visibility changes affect all connected clients simultaneously. When you need different clients to see different components, use per-session visibility instead.

Session visibility lets individual sessions customize their view of available components. When a tool calls `ctx.enable_components()` or `ctx.disable_components()`, those rules apply only to the current session. Other sessions continue to see the global defaults. This enables patterns like progressive disclosure, role-based access, and on-demand feature activation.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.context import Context

mcp = FastMCP("Session-Aware Server")

@mcp.tool(tags={"premium"})
def premium_analysis(data: str) -> str:
    """Advanced analysis available to premium users."""
    return f"Premium analysis of: {data}"

@mcp.tool
async def unlock_premium(ctx: Context) -> str:
    """Unlock premium features for this session."""
    await ctx.enable_components(tags={"premium"})
    return "Premium features unlocked"

@mcp.tool
async def reset_features(ctx: Context) -> str:
    """Reset to default feature set."""
    await ctx.reset_visibility()
    return "Features reset to defaults"

# Premium tools are disabled globally by default
mcp.disable(tags={"premium"})
```

All sessions start with `premium_analysis` hidden. When a session calls `unlock_premium`, that session gains access to premium tools while other sessions remain unaffected. Calling `reset_features` returns the session to the global defaults.

### How Session Rules Work

Session rules override global transforms. When listing components, FastMCP first applies global enable/disable rules, then applies session-specific rules on top. Rules within a session accumulate, and later rules override earlier ones for the same component.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
async def customize_session(ctx: Context) -> str:
    # Enable finance tools for this session
    await ctx.enable_components(tags={"finance"})

    # Also enable admin tools
    await ctx.enable_components(tags={"admin"})

    # Later: disable a specific admin tool
    await ctx.disable_components(names={"dangerous_admin_tool"})

    return "Session customized"
```

Each call adds a rule to the session. The `dangerous_admin_tool` ends up disabled because its disable rule was added after the admin enable rule.

### Filter Criteria

The session visibility methods accept the same filter criteria as `server.enable()` and `server.disable()`:

| Parameter    | Description                                                                |
| ------------ | -------------------------------------------------------------------------- |
| `names`      | Component names or URIs to match                                           |
| `keys`       | Component keys (e.g., `{"tool:my_tool"}`)                                  |
| `tags`       | Tags to match (component must have at least one)                           |
| `version`    | Version specification to match                                             |
| `components` | Component types (`{"tool"}`, `{"resource"}`, `{"prompt"}`, `{"template"}`) |
| `match_all`  | If `True`, matches all components regardless of other criteria             |

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.utilities.versions import VersionSpec

@mcp.tool
async def enable_recent_tools(ctx: Context) -> str:
    """Enable only tools from version 2.0.0 or later."""
    await ctx.enable_components(
        version=VersionSpec(gte="2.0.0"),
        components={"tool"}
    )
    return "Recent tools enabled"
```

### Automatic Notifications

When session visibility changes, FastMCP automatically sends notifications to that session. Clients receive `ToolListChangedNotification`, `ResourceListChangedNotification`, and `PromptListChangedNotification` so they can refresh their component lists. These notifications go only to the affected session.

When you specify the `components` parameter, FastMCP optimizes by sending only the relevant notifications:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Only sends ToolListChangedNotification
await ctx.enable_components(tags={"finance"}, components={"tool"})

# Sends all three notifications (no components filter)
await ctx.enable_components(tags={"finance"})
```

### Namespace Activation Pattern

A common pattern organizes tools into namespaces using tag prefixes, disables them globally, then provides activation tools that unlock namespaces on demand:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.context import Context

server = FastMCP("Multi-Domain Assistant")

# Finance namespace
@server.tool(tags={"namespace:finance"})
def analyze_portfolio(symbols: list[str]) -> str:
    return f"Analysis for: {', '.join(symbols)}"

@server.tool(tags={"namespace:finance"})
def get_market_data(symbol: str) -> dict:
    return {"symbol": symbol, "price": 150.25}

# Admin namespace
@server.tool(tags={"namespace:admin"})
def list_users() -> list[str]:
    return ["alice", "bob", "charlie"]

# Activation tools - always visible
@server.tool
async def activate_finance(ctx: Context) -> str:
    await ctx.enable_components(tags={"namespace:finance"})
    return "Finance tools activated"

@server.tool
async def activate_admin(ctx: Context) -> str:
    await ctx.enable_components(tags={"namespace:admin"})
    return "Admin tools activated"

@server.tool
async def deactivate_all(ctx: Context) -> str:
    await ctx.reset_visibility()
    return "All namespaces deactivated"

# Disable namespace tools globally
server.disable(tags={"namespace:finance", "namespace:admin"})
```

Sessions start seeing only the activation tools. Calling `activate_finance` reveals finance tools for that session only. Multiple namespaces can be activated independently, and `deactivate_all` returns to the initial state.

### Method Reference

* **`await ctx.enable_components(...) -> None`**: Enable matching components for this session
* **`await ctx.disable_components(...) -> None`**: Disable matching components for this session
* **`await ctx.reset_visibility() -> None`**: Clear all session rules, returning to global defaults

## Client Notifications

When visibility state changes, FastMCP automatically notifies connected clients. Clients supporting the MCP notification protocol receive `list_changed` events and can refresh their component lists.

This happens automatically. You don't need to trigger notifications manually.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# This automatically notifies clients
mcp.disable(tags={"maintenance"})

# Clients receive: tools/list_changed, resources/list_changed, etc.
```

## Filtering Logic

Understanding the filtering logic helps when debugging visibility state issues.

The `is_enabled()` function checks a component's internal metadata:

1. If the component has `meta.fastmcp._internal.visibility = False`, it's disabled
2. If the component has `meta.fastmcp._internal.visibility = True`, it's enabled
3. If no visibility state is set, the component is enabled by default

When multiple `enable()` and `disable()` calls are made, transforms are applied in order. **Later transforms override earlier ones**, so the last matching transform wins.

## The Visibility Transform

Under the hood, `enable()` and `disable()` add `Visibility` transforms to the server or provider. The `Visibility` transform marks components with visibility metadata, and the server applies the final filter after all provider and server transforms complete.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.transforms import Visibility

mcp = FastMCP("Server")

# Using the convenience method (recommended)
mcp.disable(names={"secret_tool"})

# Equivalent to:
mcp.add_transform(Visibility(False, names={"secret_tool"}))
```

Server-level transforms override provider-level transforms. If a component is disabled at the provider level but enabled at the server level, the server-level `enable()` can re-enable it.
