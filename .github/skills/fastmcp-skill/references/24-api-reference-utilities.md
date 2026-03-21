# API Reference: Utilities

Source lines: 46746-49219 from the original FastMCP documentation dump.

Utility package reference for auth, CLI helpers, JSON schema tools, lifecycle helpers, OpenAPI helpers, pagination, skills, testing, timeout, and version helpers.

---

# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-__init__



# `fastmcp.utilities`

FastMCP utility modules.


# async_utils
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-async_utils



# `fastmcp.utilities.async_utils`

Async utilities for FastMCP.

## Functions

### `call_sync_fn_in_threadpool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/async_utils.py#L13"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_sync_fn_in_threadpool(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any
```

Call a sync function in a threadpool to avoid blocking the event loop.

Uses anyio.to\_thread.run\_sync which properly propagates contextvars,
making this safe for functions that depend on context (like dependency injection).

### `gather` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/async_utils.py#L38"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
gather(*awaitables: Awaitable[T]) -> list[T] | list[T | BaseException]
```

Run awaitables concurrently and return results in order.

Uses anyio TaskGroup for structured concurrency.

**Args:**

* `*awaitables`: Awaitables to run concurrently
* `return_exceptions`: If True, exceptions are returned in results.
  If False, first exception cancels all and raises.

**Returns:**

* List of results in the same order as input awaitables.


# auth
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-auth



# `fastmcp.utilities.auth`

Authentication utility helpers.

## Functions

### `decode_jwt_header` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/auth.py#L32"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
decode_jwt_header(token: str) -> dict[str, Any]
```

Decode JWT header without signature verification.

Useful for extracting the key ID (kid) for JWKS lookup.

**Args:**

* `token`: JWT token string (header.payload.signature)

**Returns:**

* Decoded header as a dictionary

**Raises:**

* `ValueError`: If token is not a valid JWT format

### `decode_jwt_payload` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/auth.py#L49"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
decode_jwt_payload(token: str) -> dict[str, Any]
```

Decode JWT payload without signature verification.

Use only for tokens received directly from trusted sources (e.g., IdP token endpoints).

**Args:**

* `token`: JWT token string (header.payload.signature)

**Returns:**

* Decoded payload as a dictionary

**Raises:**

* `ValueError`: If token is not a valid JWT format

### `parse_scopes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/auth.py#L66"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse_scopes(value: Any) -> list[str] | None
```

Parse scopes from environment variables or settings values.

Accepts either a JSON array string, a comma- or space-separated string,
a list of strings, or `None`. Returns a list of scopes or `None` if
no value is provided.


# cli
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-cli



# `fastmcp.utilities.cli`

## Functions

### `is_already_in_uv_subprocess` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/cli.py#L28"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_already_in_uv_subprocess() -> bool
```

Check if we're already running in a FastMCP uv subprocess.

### `load_and_merge_config` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/cli.py#L33"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_and_merge_config(server_spec: str | None, **cli_overrides) -> tuple[MCPServerConfig, str]
```

Load config from server\_spec and apply CLI overrides.

This consolidates the config parsing logic that was duplicated across
run, inspect, and dev commands.

**Args:**

* `server_spec`: Python file, config file, URL, or None to auto-detect
* `cli_overrides`: CLI arguments that override config values

**Returns:**

* Tuple of (MCPServerConfig, resolved\_server\_spec)

### `log_server_banner` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/cli.py#L201"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
log_server_banner(server: FastMCP[Any]) -> None
```

Creates and logs a formatted banner with server information and logo.


# components
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-components



# `fastmcp.utilities.components`

## Functions

### `get_fastmcp_metadata` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L26"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_fastmcp_metadata(meta: dict[str, Any] | None) -> FastMCPMeta
```

Extract FastMCP metadata from a component's meta dict.

Handles both the current `fastmcp` namespace and the legacy `_fastmcp`
namespace for compatibility with older FastMCP servers.

## Classes

### `FastMCPMeta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L20"><Icon icon="github" /></a></sup>

### `FastMCPComponent` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L61"><Icon icon="github" /></a></sup>

Base class for FastMCP tools, prompts, resources, and resource templates.

**Methods:**

#### `make_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L112"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
make_key(cls, identifier: str) -> str
```

Construct the lookup key for this component type.

**Args:**

* `identifier`: The raw identifier (name for tools/prompts, uri for resources)

**Returns:**

* A prefixed key like "tool:name" or "resource:uri"

#### `key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L126"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
key(self) -> str
```

The globally unique lookup key for this component.

Format: ":@" or ":@"
e.g. "tool:my\_tool\@v2", "tool:my\_tool@", "resource:file://x.txt@"

The @ suffix is ALWAYS present to enable unambiguous parsing of keys
(URIs may contain @ characters, so we always include the delimiter).

Subclasses should override this to use their specific identifier.
Base implementation uses name.

#### `get_meta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L141"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_meta(self) -> dict[str, Any]
```

Get the meta information about the component.

Returns a dict that always includes a `fastmcp` key containing:

* `tags`: sorted list of component tags
* `version`: component version (only if set)

Internal keys (prefixed with `_`) are stripped from the fastmcp namespace.

#### `enable` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L189"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
enable(self) -> None
```

Removed in 3.0. Use server.enable(keys=\[...]) instead.

#### `disable` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L196"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
disable(self) -> None
```

Removed in 3.0. Use server.disable(keys=\[...]) instead.

#### `copy` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L203"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
copy(self) -> Self
```

Create a copy of the component.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L207"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this component with docket for background execution.

No-ops if task\_config.mode is "forbidden". Subclasses override to
register their callable (self.run, self.read, self.render, or self.fn).

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L215"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, *args: Any, **kwargs: Any) -> Execution
```

Schedule this component for background execution via docket.

Subclasses override this to handle their specific calling conventions:

* Tool: add\_to\_docket(docket, arguments: dict, \*\*kwargs)
* Resource: add\_to\_docket(docket, \*\*kwargs)
* ResourceTemplate: add\_to\_docket(docket, params: dict, \*\*kwargs)
* Prompt: add\_to\_docket(docket, arguments: dict | None, \*\*kwargs)

The \*\*kwargs are passed through to docket.add() (e.g., key=task\_key).

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/components.py#L237"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

Return span attributes for telemetry.

Subclasses should call super() and merge their specific attributes.


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


# exceptions
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-exceptions



# `fastmcp.utilities.exceptions`

## Functions

### `iter_exc` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/exceptions.py#L12"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
iter_exc(group: BaseExceptionGroup)
```

### `get_catch_handlers` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/exceptions.py#L42"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_catch_handlers() -> Mapping[type[BaseException] | Iterable[type[BaseException]], Callable[[BaseExceptionGroup[Any]], Any]]
```


# http
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-http



# `fastmcp.utilities.http`

## Functions

### `find_available_port` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/http.py#L4"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
find_available_port() -> int
```

Find an available port by letting the OS assign one.


# inspect
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-inspect



# `fastmcp.utilities.inspect`

Utilities for inspecting FastMCP instances.

## Functions

### `inspect_fastmcp_v2` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L100"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
inspect_fastmcp_v2(mcp: FastMCP[Any]) -> FastMCPInfo
```

Extract information from a FastMCP v2.x instance.

**Args:**

* `mcp`: The FastMCP v2.x instance to inspect

**Returns:**

* FastMCPInfo dataclass containing the extracted information

### `inspect_fastmcp_v1` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L236"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
inspect_fastmcp_v1(mcp: FastMCP1x) -> FastMCPInfo
```

Extract information from a FastMCP v1.x instance using a Client.

**Args:**

* `mcp`: The FastMCP v1.x instance to inspect

**Returns:**

* FastMCPInfo dataclass containing the extracted information

### `inspect_fastmcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L378"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
inspect_fastmcp(mcp: FastMCP[Any] | FastMCP1x) -> FastMCPInfo
```

Extract information from a FastMCP instance into a dataclass.

This function automatically detects whether the instance is FastMCP v1.x or v2.x
and uses the appropriate extraction method.

**Args:**

* `mcp`: The FastMCP instance to inspect (v1.x or v2.x)

**Returns:**

* FastMCPInfo dataclass containing the extracted information

### `format_fastmcp_info` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L403"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
format_fastmcp_info(info: FastMCPInfo) -> bytes
```

Format FastMCPInfo as FastMCP-specific JSON.

This includes FastMCP-specific fields like tags, enabled, annotations, etc.

### `format_mcp_info` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L432"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
format_mcp_info(mcp: FastMCP[Any] | FastMCP1x) -> bytes
```

Format server info as standard MCP protocol JSON.

Uses Client to get the standard MCP protocol format with camelCase fields.
Includes version metadata at the top level.

### `format_info` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L465"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
format_info(mcp: FastMCP[Any] | FastMCP1x, format: InspectFormat | Literal['fastmcp', 'mcp'], info: FastMCPInfo | None = None) -> bytes
```

Format server information according to the specified format.

**Args:**

* `mcp`: The FastMCP instance
* `format`: Output format ("fastmcp" or "mcp")
* `info`: Pre-extracted FastMCPInfo (optional, will be extracted if not provided)

**Returns:**

* JSON bytes in the requested format

## Classes

### `ToolInfo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L19"><Icon icon="github" /></a></sup>

Information about a tool.

### `PromptInfo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L35"><Icon icon="github" /></a></sup>

Information about a prompt.

### `ResourceInfo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L49"><Icon icon="github" /></a></sup>

Information about a resource.

### `TemplateInfo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L65"><Icon icon="github" /></a></sup>

Information about a resource template.

### `FastMCPInfo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L82"><Icon icon="github" /></a></sup>

Information extracted from a FastMCP instance.

### `InspectFormat` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/inspect.py#L396"><Icon icon="github" /></a></sup>

Output format for inspect command.


# json_schema
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-json_schema



# `fastmcp.utilities.json_schema`

## Functions

### `dereference_refs` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/json_schema.py#L56"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
dereference_refs(schema: dict[str, Any]) -> dict[str, Any]
```

Resolve all \$ref references in a JSON schema by inlining definitions.

This function resolves $ref references that point to $defs, replacing them
with the actual definition content while preserving sibling keywords (like
description, default, examples) that Pydantic places alongside \$ref.

This is necessary because some MCP clients (e.g., VS Code Copilot) don't
properly handle \$ref in tool input schemas.

For self-referencing/circular schemas where full dereferencing is not possible,
this function falls back to resolving only the root-level $ref while preserving
$defs for nested references.

**Args:**

* `schema`: JSON schema dict that may contain \$ref references

**Returns:**

* A new schema dict with $ref resolved where possible and $defs removed
* when no longer needed

### `resolve_root_ref` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/json_schema.py#L185"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resolve_root_ref(schema: dict[str, Any]) -> dict[str, Any]
```

Resolve \$ref at root level to meet MCP spec requirements.

MCP specification requires outputSchema to have "type": "object" at the root level.
When Pydantic generates schemas for self-referential models, it uses $ref at the
root level pointing to $defs. This function resolves such references by inlining
the referenced definition while preserving \$defs for nested references.

**Args:**

* `schema`: JSON schema dict that may have \$ref at root level

**Returns:**

* A new schema dict with root-level \$ref resolved, or the original schema
* if no resolution is needed

### `compress_schema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/json_schema.py#L418"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
compress_schema(schema: dict[str, Any], prune_params: list[str] | None = None, prune_additional_properties: bool = False, prune_titles: bool = False, dereference: bool = False) -> dict[str, Any]
```

Compress and optimize a JSON schema for MCP compatibility.

**Args:**

* `schema`: The schema to compress
* `prune_params`: List of parameter names to remove from properties
* `prune_additional_properties`: Whether to remove additionalProperties: false.
  Defaults to False to maintain MCP client compatibility, as some clients
  (e.g., Claude) require additionalProperties: false for strict validation.
* `prune_titles`: Whether to remove title fields from the schema
* `dereference`: Whether to dereference \$ref by inlining definitions.
  Defaults to False; dereferencing is typically handled by
  middleware at serve-time instead.


# json_schema_type
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-json_schema_type



# `fastmcp.utilities.json_schema_type`

Convert JSON Schema to Python types with validation.

The json\_schema\_to\_type function converts a JSON Schema into a Python type that can be used
for validation with Pydantic. It supports:

* Basic types (string, number, integer, boolean, null)
* Complex types (arrays, objects)
* Format constraints (date-time, email, uri)
* Numeric constraints (minimum, maximum, multipleOf)
* String constraints (minLength, maxLength, pattern)
* Array constraints (minItems, maxItems, uniqueItems)
* Object properties with defaults
* References and recursive schemas
* Enums and constants
* Union types

Example:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

# Name is optional and will be inferred from schema's "title" property if not provided
Person = json_schema_to_type(schema)
# Creates a validated dataclass with name, age, and optional email fields
```

## Functions

### `json_schema_to_type` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/json_schema_type.py#L111"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
json_schema_to_type(schema: Mapping[str, Any], name: str | None = None) -> type
```

Convert JSON schema to appropriate Python type with validation.

**Args:**

* `schema`: A JSON Schema dictionary defining the type structure and validation rules
* `name`: Optional name for object schemas. Only allowed when schema type is "object".
  If not provided for objects, name will be inferred from schema's "title"
  property or default to "Root".

**Returns:**

* A Python type (typically a dataclass for objects) with Pydantic validation

**Raises:**

* `ValueError`: If a name is provided for a non-object schema

**Examples:**

Create a dataclass from an object schema:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
schema = {
    "type": "object",
    "title": "Person",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

Person = json_schema_to_type(schema)
# Creates a dataclass with name, age, and optional email fields:
# @dataclass
# class Person:
#     name: str
#     age: int
#     email: str | None = None
```

Person(name="John", age=30)

Create a scalar type with constraints:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
schema = {
    "type": "string",
    "minLength": 3,
    "pattern": "^[A-Z][a-z]+$"
}

NameType = json_schema_to_type(schema)
# Creates Annotated[str, StringConstraints(min_length=3, pattern="^[A-Z][a-z]+$")]

@dataclass
class Name:
    name: NameType
```

## Classes

### `JSONSchema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/json_schema_type.py#L78"><Icon icon="github" /></a></sup>


# lifespan
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-lifespan



# `fastmcp.utilities.lifespan`

Lifespan utilities for combining async context manager lifespans.

## Functions

### `combine_lifespans` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/lifespan.py#L12"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
combine_lifespans(*lifespans: Callable[[AppT], AbstractAsyncContextManager[Mapping[str, Any] | None]]) -> Callable[[AppT], AbstractAsyncContextManager[dict[str, Any]]]
```

Combine multiple lifespans into a single lifespan.

Useful when mounting FastMCP into FastAPI and you need to run
both your app's lifespan and the MCP server's lifespan.

Works with both FastAPI-style lifespans (yield None) and FastMCP-style
lifespans (yield dict). Results are merged; later lifespans override
earlier ones on key conflicts.

Lifespans are entered in order and exited in reverse order (LIFO).

**Args:**

* `*lifespans`: Lifespan context manager factories to combine.

**Returns:**

* A combined lifespan context manager factory.


# logging
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-logging



# `fastmcp.utilities.logging`

Logging utilities for FastMCP.

## Functions

### `get_logger` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/logging.py#L14"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_logger(name: str) -> logging.Logger
```

Get a logger nested under FastMCP namespace.

**Args:**

* `name`: the name of the logger, which will be prefixed with 'FastMCP.'

**Returns:**

* a configured logger instance

### `configure_logging` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/logging.py#L29"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
configure_logging(level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] | int = 'INFO', logger: logging.Logger | None = None, enable_rich_tracebacks: bool | None = None, **rich_kwargs: Any) -> None
```

Configure logging for FastMCP.

**Args:**

* `logger`: the logger to configure
* `level`: the log level to use
* `rich_kwargs`: the parameters to use for creating RichHandler

### `temporary_log_level` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/logging.py#L111"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
temporary_log_level(level: str | None, logger: logging.Logger | None = None, enable_rich_tracebacks: bool | None = None, **rich_kwargs: Any)
```

Context manager to temporarily set log level and restore it afterwards.

**Args:**

* `level`: The temporary log level to set (e.g., "DEBUG", "INFO")
* `logger`: Optional logger to configure (defaults to FastMCP logger)
* `enable_rich_tracebacks`: Whether to enable rich tracebacks
* `**rich_kwargs`: Additional parameters for RichHandler


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-mcp_server_config-__init__



# `fastmcp.utilities.mcp_server_config`

FastMCP Configuration module.

This module provides versioned configuration support for FastMCP servers.
The current version is v1, which is re-exported here for convenience.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-mcp_server_config-v1-__init__



# `fastmcp.utilities.mcp_server_config.v1`

*This module is empty or contains only private/internal implementations.*


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-mcp_server_config-v1-environments-__init__



# `fastmcp.utilities.mcp_server_config.v1.environments`

Environment configuration for MCP servers.


# base
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-mcp_server_config-v1-environments-base



# `fastmcp.utilities.mcp_server_config.v1.environments.base`

## Classes

### `Environment` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/environments/base.py#L7"><Icon icon="github" /></a></sup>

Base class for environment configuration.

**Methods:**

#### `build_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/environments/base.py#L13"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
build_command(self, command: list[str]) -> list[str]
```

Build the full command with environment setup.

**Args:**

* `command`: Base command to wrap with environment setup

**Returns:**

* Full command ready for subprocess execution

#### `prepare` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/environments/base.py#L23"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prepare(self, output_dir: Path | None = None) -> None
```

Prepare the environment (optional, can be no-op).

**Args:**

* `output_dir`: Directory for persistent environment setup


# uv
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-mcp_server_config-v1-environments-uv



# `fastmcp.utilities.mcp_server_config.v1.environments.uv`

## Classes

### `UVEnvironment` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/environments/uv.py#L14"><Icon icon="github" /></a></sup>

Configuration for Python environment setup.

**Methods:**

#### `build_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/environments/uv.py#L49"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
build_command(self, command: list[str]) -> list[str]
```

Build complete uv run command with environment args and command to execute.

**Args:**

* `command`: Command to execute (e.g., \["fastmcp", "run", "server.py"])

**Returns:**

* Complete command ready for subprocess.run, including "uv" prefix if needed.
* If no environment configuration is set, returns the command unchanged.

#### `prepare` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/environments/uv.py#L109"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prepare(self, output_dir: Path | None = None) -> None
```

Prepare the Python environment using uv.

**Args:**

* `output_dir`: Directory where the persistent uv project will be created.
  If None, creates a temporary directory for ephemeral use.


# mcp_server_config
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-mcp_server_config-v1-mcp_server_config



# `fastmcp.utilities.mcp_server_config.v1.mcp_server_config`

FastMCP Configuration File Support.

This module provides support for fastmcp.json configuration files that allow
users to specify server settings in a declarative format instead of using
command-line arguments.

## Functions

### `generate_schema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L416"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_schema(output_path: Path | str | None = None) -> dict[str, Any] | None
```

Generate JSON schema for fastmcp.json files.

This is used to create the schema file that IDEs can use for
validation and auto-completion.

**Args:**

* `output_path`: Optional path to write the schema to. If provided,
  writes the schema and returns None. If not provided,
  returns the schema as a dictionary.

**Returns:**

* JSON schema as a dictionary if output\_path is None, otherwise None

## Classes

### `Deployment` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L36"><Icon icon="github" /></a></sup>

Configuration for server deployment and runtime settings.

**Methods:**

#### `apply_runtime_settings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L85"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
apply_runtime_settings(self, config_path: Path | None = None) -> None
```

Apply runtime settings like environment variables and working directory.

**Args:**

* `config_path`: Path to config file for resolving relative paths

Environment variables support interpolation with ${VAR_NAME} syntax.
For example: "API_URL": "https://api.$.example.com"
will substitute the value of the ENVIRONMENT variable at runtime.

### `MCPServerConfig` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L134"><Icon icon="github" /></a></sup>

Configuration for a FastMCP server.

This configuration file allows you to specify all settings needed to run
a FastMCP server in a declarative format.

**Methods:**

#### `validate_source` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L183"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_source(cls, v: dict | Source) -> SourceType
```

Validate and convert source to proper format.

Supports:

* Dict format: `{"path": "server.py", "entrypoint": "app"}`
* FileSystemSource instance (passed through)

No string parsing happens here - that's only at CLI boundaries.
MCPServerConfig works only with properly typed objects.

#### `validate_environment` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L199"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_environment(cls, v: dict | Any) -> EnvironmentType
```

Ensure environment has a type field for discrimination.

For backward compatibility, if no type is specified, default to "uv".

#### `validate_deployment` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L210"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_deployment(cls, v: dict | Deployment) -> Deployment
```

Validate and convert deployment to Deployment.

Accepts:

* Deployment instance
* dict that can be converted to Deployment

#### `from_file` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L223"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_file(cls, file_path: Path) -> MCPServerConfig
```

Load configuration from a JSON file.

**Args:**

* `file_path`: Path to the configuration file

**Returns:**

* MCPServerConfig instance

**Raises:**

* `FileNotFoundError`: If the file doesn't exist
* `json.JSONDecodeError`: If the file is not valid JSON
* `pydantic.ValidationError`: If the configuration is invalid

#### `from_cli_args` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L246"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_cli_args(cls, source: FileSystemSource, transport: Literal['stdio', 'http', 'sse', 'streamable-http'] | None = None, host: str | None = None, port: int | None = None, path: str | None = None, log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] | None = None, python: str | None = None, dependencies: list[str] | None = None, requirements: str | None = None, project: str | None = None, editable: str | None = None, env: dict[str, str] | None = None, cwd: str | None = None, args: list[str] | None = None) -> MCPServerConfig
```

Create a config from CLI arguments.

This allows us to have a single code path where everything
goes through a config object.

**Args:**

* `source`: Server source (FileSystemSource instance)
* `transport`: Transport protocol
* `host`: Host for HTTP transport
* `port`: Port for HTTP transport
* `path`: URL path for server
* `log_level`: Logging level
* `python`: Python version
* `dependencies`: Python packages to install
* `requirements`: Path to requirements file
* `project`: Path to project directory
* `editable`: Path to install in editable mode
* `env`: Environment variables
* `cwd`: Working directory
* `args`: Server arguments

**Returns:**

* MCPServerConfig instance

#### `find_config` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L323"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
find_config(cls, start_path: Path | None = None) -> Path | None
```

Find a fastmcp.json file in the specified directory.

**Args:**

* `start_path`: Directory to look in (defaults to current directory)

**Returns:**

* Path to the configuration file, or None if not found

#### `prepare` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L342"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prepare(self, skip_source: bool = False, output_dir: Path | None = None) -> None
```

Prepare environment and source for execution.

When output\_dir is provided, creates a persistent uv project.
When output\_dir is None, does ephemeral caching (for backwards compatibility).

**Args:**

* `skip_source`: Skip source preparation if True
* `output_dir`: Directory to create the persistent uv project in (optional)

#### `prepare_environment` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L363"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prepare_environment(self, output_dir: Path | None = None) -> None
```

Prepare the Python environment.

**Args:**

* `output_dir`: If provided, creates a persistent uv project in this directory.
  If None, just populates uv's cache for ephemeral use.

Delegates to the environment's prepare() method

#### `prepare_source` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L374"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prepare_source(self) -> None
```

Prepare the source for loading.

Delegates to the source's prepare() method.

#### `run_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/mcp_server_config.py#L381"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_server(self, **kwargs: Any) -> None
```

Load and run the server with this configuration.

**Args:**

* `**kwargs`: Additional arguments to pass to server.run\_async()
  These override config settings


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-mcp_server_config-v1-sources-__init__



# `fastmcp.utilities.mcp_server_config.v1.sources`

*This module is empty or contains only private/internal implementations.*


# base
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-mcp_server_config-v1-sources-base



# `fastmcp.utilities.mcp_server_config.v1.sources.base`

## Classes

### `Source` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/sources/base.py#L7"><Icon icon="github" /></a></sup>

Abstract base class for all source types.

**Methods:**

#### `prepare` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/sources/base.py#L12"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prepare(self) -> None
```

Prepare the source (download, clone, install, etc).

For sources that need preparation (e.g., git clone, download),
this method performs that preparation. For sources that don't
need preparation (e.g., local files), this is a no-op.

#### `load_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/sources/base.py#L22"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_server(self) -> Any
```

Load and return the FastMCP server instance.

Must be called after prepare() if the source requires preparation.
All information needed to load the server should be available
as attributes on the source instance.


# filesystem
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-mcp_server_config-v1-sources-filesystem



# `fastmcp.utilities.mcp_server_config.v1.sources.filesystem`

## Classes

### `FileSystemSource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/sources/filesystem.py#L15"><Icon icon="github" /></a></sup>

Source for local Python files.

**Methods:**

#### `parse_path_with_object` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/sources/filesystem.py#L28"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse_path_with_object(cls, v: str) -> str
```

Parse path:object syntax and extract the object name.

This validator runs before the model is created, allowing us to
handle the "file.py:object" syntax at the model boundary.

#### `load_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/mcp_server_config/v1/sources/filesystem.py#L63"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_server(self) -> Any
```

Load server from filesystem.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-openapi-__init__



# `fastmcp.utilities.openapi`

OpenAPI utilities for FastMCP - refactored for better maintainability.


# director
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-openapi-director



# `fastmcp.utilities.openapi.director`

Request director using openapi-core for stateless HTTP request building.

## Classes

### `RequestDirector` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/director.py#L16"><Icon icon="github" /></a></sup>

Builds httpx.Request objects from HTTPRoute and arguments using openapi-core.

**Methods:**

#### `build` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/director.py#L23"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
build(self, route: HTTPRoute, flat_args: dict[str, Any], base_url: str = 'http://localhost') -> httpx.Request
```

Constructs a final httpx.Request object, handling all OpenAPI serialization.

**Args:**

* `route`: HTTPRoute containing OpenAPI operation details
* `flat_args`: Flattened arguments from LLM (may include suffixed parameters)
* `base_url`: Base URL for the request

**Returns:**

* httpx.Request: Properly formatted HTTP request


# formatters
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-openapi-formatters



# `fastmcp.utilities.openapi.formatters`

Parameter formatting functions for OpenAPI operations.

## Functions

### `format_array_parameter` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/formatters.py#L12"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
format_array_parameter(values: list, parameter_name: str, is_query_parameter: bool = False) -> str | list
```

Format an array parameter according to OpenAPI specifications.

**Args:**

* `values`: List of values to format
* `parameter_name`: Name of the parameter (for error messages)
* `is_query_parameter`: If True, can return list for explode=True behavior

**Returns:**

* String (comma-separated) or list (for query params with explode=True)

### `format_deep_object_parameter` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/formatters.py#L66"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
format_deep_object_parameter(param_value: dict, parameter_name: str) -> dict[str, str]
```

Format a dictionary parameter for deep-object style serialization.

According to OpenAPI 3.0 spec, deepObject style with explode=true serializes
object properties as separate query parameters with bracket notation.

For example, `{"id": "123", "type": "user"}` becomes
`param[id]=123&param[type]=user`.

**Args:**

* `param_value`: Dictionary value to format
* `parameter_name`: Name of the parameter

**Returns:**

* Dictionary with bracketed parameter names as keys

### `generate_example_from_schema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/formatters.py#L100"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_example_from_schema(schema: JsonSchema | None) -> Any
```

Generate a simple example value from a JSON schema dictionary.
Very basic implementation focusing on types.

### `format_json_for_description` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/formatters.py#L183"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
format_json_for_description(data: Any, indent: int = 2) -> str
```

Formats Python data as a JSON string block for Markdown.

### `format_description_with_responses` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/formatters.py#L192"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
format_description_with_responses(base_description: str, responses: dict[str, Any], parameters: list[ParameterInfo] | None = None, request_body: RequestBodyInfo | None = None) -> str
```

Formats the base description string with response, parameter, and request body information.

**Args:**

* `base_description`: The initial description to be formatted.
* `responses`: A dictionary of response information, keyed by status code.
* `parameters`: A list of parameter information,
  including path and query parameters. Each parameter includes details such as name,
  location, whether it is required, and a description.
* `request_body`: Information about the request body,
  including its description, whether it is required, and its content schema.

**Returns:**

* The formatted description string with additional details about responses, parameters,
* and the request body.


# json_schema_converter
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-openapi-json_schema_converter



# `fastmcp.utilities.openapi.json_schema_converter`

Clean OpenAPI 3.0 to JSON Schema converter for the experimental parser.

This module provides a systematic approach to converting OpenAPI 3.0 schemas
to JSON Schema, inspired by py-openapi-schema-to-json-schema but optimized
for our specific use case.

## Functions

### `convert_openapi_schema_to_json_schema` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/json_schema_converter.py#L38"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
convert_openapi_schema_to_json_schema(schema: dict[str, Any], openapi_version: str | None = None, remove_read_only: bool = False, remove_write_only: bool = False, convert_one_of_to_any_of: bool = True) -> dict[str, Any]
```

Convert an OpenAPI schema to JSON Schema format.

This is a clean, systematic approach that:

1. Removes OpenAPI-specific fields
2. Converts nullable fields to type arrays (for OpenAPI 3.0 only)
3. Converts oneOf to anyOf for overlapping union handling
4. Recursively processes nested schemas
5. Optionally removes readOnly/writeOnly properties

**Args:**

* `schema`: OpenAPI schema dictionary
* `openapi_version`: OpenAPI version for optimization
* `remove_read_only`: Whether to remove readOnly properties
* `remove_write_only`: Whether to remove writeOnly properties
* `convert_one_of_to_any_of`: Whether to convert oneOf to anyOf

**Returns:**

* JSON Schema-compatible dictionary

### `convert_schema_definitions` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/json_schema_converter.py#L322"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
convert_schema_definitions(schema_definitions: dict[str, Any] | None, openapi_version: str | None = None, **kwargs) -> dict[str, Any]
```

Convert a dictionary of OpenAPI schema definitions to JSON Schema.

**Args:**

* `schema_definitions`: Dictionary of schema definitions
* `openapi_version`: OpenAPI version for optimization
* `**kwargs`: Additional arguments passed to convert\_openapi\_schema\_to\_json\_schema

**Returns:**

* Dictionary of converted schema definitions


# models
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-openapi-models



# `fastmcp.utilities.openapi.models`

Intermediate Representation (IR) models for OpenAPI operations.

## Classes

### `ParameterInfo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/models.py#L17"><Icon icon="github" /></a></sup>

Represents a single parameter for an HTTP operation in our IR.

### `RequestBodyInfo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/models.py#L29"><Icon icon="github" /></a></sup>

Represents the request body for an HTTP operation in our IR.

### `ResponseInfo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/models.py#L39"><Icon icon="github" /></a></sup>

Represents response information in our IR.

### `HTTPRoute` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/models.py#L47"><Icon icon="github" /></a></sup>

Intermediate Representation for a single OpenAPI operation.


# parser
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-openapi-parser



# `fastmcp.utilities.openapi.parser`

OpenAPI parsing logic for converting OpenAPI specs to HTTPRoute objects.

## Functions

### `parse_openapi_to_http_routes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/parser.py#L55"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse_openapi_to_http_routes(openapi_dict: dict[str, Any]) -> list[HTTPRoute]
```

Parses an OpenAPI schema dictionary into a list of HTTPRoute objects
using the openapi-pydantic library.

Supports both OpenAPI 3.0.x and 3.1.x versions.

## Classes

### `OpenAPIParser` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/parser.py#L109"><Icon icon="github" /></a></sup>

Unified parser for OpenAPI schemas with generic type parameters to handle both 3.0 and 3.1.

**Methods:**

#### `parse` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/parser.py#L663"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse(self) -> list[HTTPRoute]
```

Parse the OpenAPI schema into HTTP routes.


# schemas
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-openapi-schemas



# `fastmcp.utilities.openapi.schemas`

Schema manipulation utilities for OpenAPI operations.

## Functions

### `clean_schema_for_display` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/schemas.py#L12"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
clean_schema_for_display(schema: JsonSchema | None) -> JsonSchema | None
```

Clean up a schema dictionary for display by removing internal/complex fields.

### `extract_output_schema_from_responses` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/openapi/schemas.py#L474"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
extract_output_schema_from_responses(responses: dict[str, ResponseInfo], schema_definitions: dict[str, Any] | None = None, openapi_version: str | None = None) -> dict[str, Any] | None
```

Extract output schema from OpenAPI responses for use as MCP tool output schema.

This function finds the first successful response (200, 201, 202, 204) with a
JSON-compatible content type and extracts its schema. If the schema is not an
object type, it wraps it to comply with MCP requirements.

**Args:**

* `responses`: Dictionary of ResponseInfo objects keyed by status code
* `schema_definitions`: Optional schema definitions to include in the output schema
* `openapi_version`: OpenAPI version string, used to optimize nullable field handling

**Returns:**

* MCP-compliant output schema with potential wrapping, or None if no suitable schema found


# pagination
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-pagination



# `fastmcp.utilities.pagination`

Pagination utilities for MCP list operations.

## Functions

### `paginate_sequence` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/pagination.py#L50"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
paginate_sequence(items: Sequence[T], cursor: str | None, page_size: int) -> tuple[list[T], str | None]
```

Paginate a sequence of items.

**Args:**

* `items`: The full sequence to paginate.
* `cursor`: Optional cursor from a previous request. None for first page.
* `page_size`: Maximum number of items per page.

**Returns:**

* Tuple of (page\_items, next\_cursor). next\_cursor is None if no more pages.

**Raises:**

* `ValueError`: If the cursor is invalid.

## Classes

### `CursorState` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/pagination.py#L16"><Icon icon="github" /></a></sup>

Internal representation of pagination cursor state.

The cursor encodes the offset into the result set. This is opaque to clients
per the MCP spec - they should not parse or modify cursors.

**Methods:**

#### `encode` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/pagination.py#L25"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
encode(self) -> str
```

Encode cursor state to an opaque string.

#### `decode` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/pagination.py#L31"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
decode(cls, cursor: str) -> CursorState
```

Decode cursor from an opaque string.

**Raises:**

* `ValueError`: If the cursor is invalid or malformed.


# skills
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-skills



# `fastmcp.utilities.skills`

Client utilities for discovering and downloading skills from MCP servers.

## Functions

### `list_skills` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/skills.py#L43"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_skills(client: Client) -> list[SkillSummary]
```

List all available skills from an MCP server.

Discovers skills by finding resources with URIs matching the
`skill://{name}/SKILL.md` pattern.

**Args:**

* `client`: Connected FastMCP client

**Returns:**

* List of SkillSummary objects with name, description, and URI

### `get_skill_manifest` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/skills.py#L87"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_skill_manifest(client: Client, skill_name: str) -> SkillManifest
```

Get the manifest for a specific skill.

**Args:**

* `client`: Connected FastMCP client
* `skill_name`: Name of the skill

**Returns:**

* SkillManifest with file listing

**Raises:**

* `ValueError`: If manifest cannot be read or parsed

### `download_skill` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/skills.py#L127"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
download_skill(client: Client, skill_name: str, target_dir: str | Path) -> Path
```

Download a skill and all its files to a local directory.

Creates a subdirectory named after the skill containing all files.

**Args:**

* `client`: Connected FastMCP client
* `skill_name`: Name of the skill to download
* `target_dir`: Directory where skill folder will be created
* `overwrite`: If True, overwrite existing skill directory. If False
  (default), raise FileExistsError if directory exists.

**Returns:**

* Path to the downloaded skill directory

**Raises:**

* `ValueError`: If skill cannot be found or downloaded
* `FileExistsError`: If skill directory exists and overwrite=False

### `sync_skills` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/skills.py#L214"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
sync_skills(client: Client, target_dir: str | Path) -> list[Path]
```

Download all available skills from a server.

**Args:**

* `client`: Connected FastMCP client
* `target_dir`: Directory where skill folders will be created
* `overwrite`: If True, overwrite existing files

**Returns:**

* List of paths to downloaded skill directories

## Classes

### `SkillSummary` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/skills.py#L18"><Icon icon="github" /></a></sup>

Summary information about a skill available on a server.

### `SkillFile` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/skills.py#L27"><Icon icon="github" /></a></sup>

Information about a file within a skill.

### `SkillManifest` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/skills.py#L36"><Icon icon="github" /></a></sup>

Full manifest of a skill including all files.


# tests
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-tests



# `fastmcp.utilities.tests`

## Functions

### `temporary_settings` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/tests.py#L24"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
temporary_settings(**kwargs: Any)
```

Temporarily override FastMCP setting values.

**Args:**

* `**kwargs`: The settings to override, including nested settings.

### `run_server_in_process` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/tests.py#L75"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_server_in_process(server_fn: Callable[..., None], *args: Any, **kwargs: Any) -> Generator[str, None, None]
```

Context manager that runs a FastMCP server in a separate process and
returns the server URL. When the context manager is exited, the server process is killed.

**Args:**

* `server_fn`: The function that runs a FastMCP server. FastMCP servers are
  not pickleable, so we need a function that creates and runs one.
* `*args`: Arguments to pass to the server function.
* `provide_host_and_port`: Whether to provide the host and port to the server function as kwargs.
* `host`: Host to bind the server to (default: "127.0.0.1").
* `port`: Port to bind the server to (default: find available port).
* `**kwargs`: Keyword arguments to pass to the server function.

**Returns:**

* The server URL.

### `run_server_async` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/tests.py#L143"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_server_async(server: FastMCP, port: int | None = None, transport: Literal['http', 'streamable-http', 'sse'] = 'http', path: str = '/mcp', host: str = '127.0.0.1') -> AsyncGenerator[str, None]
```

Start a FastMCP server as an asyncio task for in-process async testing.

This is the recommended way to test FastMCP servers. It runs the server
as an async task in the same process, eliminating subprocess coordination,
sleeps, and cleanup issues.

**Args:**

* `server`: FastMCP server instance
* `port`: Port to bind to (default: find available port)
* `transport`: Transport type ("http", "streamable-http", or "sse")
* `path`: URL path for the server (default: "/mcp")
* `host`: Host to bind to (default: "127.0.0.1")

## Classes

### `HeadlessOAuth` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/tests.py#L225"><Icon icon="github" /></a></sup>

OAuth provider that bypasses browser interaction for testing.

This simulates the complete OAuth flow programmatically by making HTTP requests
instead of opening a browser and running a callback server. Useful for automated testing.

**Methods:**

#### `redirect_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/tests.py#L238"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
redirect_handler(self, authorization_url: str) -> None
```

Make HTTP request to authorization URL and store response for callback handler.

#### `callback_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/tests.py#L244"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
callback_handler(self) -> tuple[str, str | None]
```

Parse stored response and return (auth\_code, state).


# timeout
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-timeout



# `fastmcp.utilities.timeout`

Timeout normalization utilities.

## Functions

### `normalize_timeout_to_timedelta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/timeout.py#L8"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
normalize_timeout_to_timedelta(value: int | float | datetime.timedelta | None) -> datetime.timedelta | None
```

Normalize a timeout value to a timedelta.

**Args:**

* `value`: Timeout value as int/float (seconds), timedelta, or None

**Returns:**

* timedelta if value provided, None otherwise

### `normalize_timeout_to_seconds` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/timeout.py#L28"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
normalize_timeout_to_seconds(value: int | float | datetime.timedelta | None) -> float | None
```

Normalize a timeout value to seconds (float).

**Args:**

* `value`: Timeout value as int/float (seconds), timedelta, or None.
  Zero values are treated as "disabled" and return None.

**Returns:**

* float seconds if value provided and non-zero, None otherwise


# types
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-types



# `fastmcp.utilities.types`

Common types used across FastMCP.

## Functions

### `get_fn_name` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L34"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_fn_name(fn: Callable[..., Any]) -> str
```

### `get_cached_typeadapter` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L45"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_cached_typeadapter(cls: T) -> TypeAdapter[T]
```

TypeAdapters are heavy objects, and in an application context we'd typically
create them once in a global scope and reuse them as often as possible.
However, this isn't feasible for user-generated functions. Instead, we use a
cache to minimize the cost of creating them as much as possible.

### `issubclass_safe` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L120"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
issubclass_safe(cls: type, base: type) -> bool
```

Check if cls is a subclass of base, even if cls is a type variable.

### `is_class_member_of_type` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L130"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_class_member_of_type(cls: Any, base: type) -> bool
```

Check if cls is a member of base, even if cls is a type variable.

Base can be a type, a UnionType, or an Annotated type. Generic types are not
considered members (e.g. T is not a member of list\[T]).

### `find_kwarg_by_type` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L152"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
find_kwarg_by_type(fn: Callable, kwarg_type: type) -> str | None
```

Find the name of the kwarg that is of type kwarg\_type.

Includes union types that contain the kwarg\_type, as well as Annotated types.

### `create_function_without_params` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L178"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_function_without_params(fn: Callable[..., Any], exclude_params: list[str]) -> Callable[..., Any]
```

Create a new function with the same code but without the specified parameters in annotations.

This is used to exclude parameters from type adapter processing when they can't be serialized.
The excluded parameters are removed from the function's **annotations** dictionary.

### `replace_type` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L451"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
replace_type(type_, type_map: dict[type, type])
```

Given a (possibly generic, nested, or otherwise complex) type, replaces all
instances of old\_type with new\_type.

This is useful for transforming types when creating tools.

**Args:**

* `type_`: The type to replace instances of old\_type with new\_type.
* `old_type`: The type to replace.
* `new_type`: The type to replace old\_type with.

Examples:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
>>> replace_type(list[int | bool], {int: str})
list[str | bool]

>>> replace_type(list[list[int]], {int: str})
list[list[str]]
```

## Classes

### `FastMCPBaseModel` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L38"><Icon icon="github" /></a></sup>

Base model for FastMCP models.

### `Image` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L235"><Icon icon="github" /></a></sup>

Helper class for returning images from tools.

**Methods:**

#### `to_image_content` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L286"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_image_content(self, mime_type: str | None = None, annotations: Annotations | None = None) -> mcp.types.ImageContent
```

Convert to MCP ImageContent.

#### `to_data_uri` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L301"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_data_uri(self, mime_type: str | None = None) -> str
```

Get image as a data URI.

### `Audio` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L307"><Icon icon="github" /></a></sup>

Helper class for returning audio from tools.

**Methods:**

#### `to_audio_content` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L344"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_audio_content(self, mime_type: str | None = None, annotations: Annotations | None = None) -> mcp.types.AudioContent
```

### `File` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L365"><Icon icon="github" /></a></sup>

Helper class for returning file data from tools.

**Methods:**

#### `to_resource_content` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L404"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_resource_content(self, mime_type: str | None = None, annotations: Annotations | None = None) -> mcp.types.EmbeddedResource
```

### `ContextSamplingFallbackProtocol` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/types.py#L488"><Icon icon="github" /></a></sup>


# ui
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-ui



# `fastmcp.utilities.ui`

Shared UI utilities for FastMCP HTML pages.

This module provides reusable HTML/CSS components for OAuth callbacks,
consent pages, and other user-facing interfaces.

## Functions

### `create_page` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/ui.py#L453"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_page(content: str, title: str = 'FastMCP', additional_styles: str = '', csp_policy: str = "default-src 'none'; style-src 'unsafe-inline'; img-src https: data:; base-uri 'none'") -> str
```

Create a complete HTML page with FastMCP styling.

**Args:**

* `content`: HTML content to place inside the page
* `title`: Page title
* `additional_styles`: Extra CSS to include
* `csp_policy`: Content Security Policy header value.
  If empty string "", the CSP meta tag is omitted entirely.

**Returns:**

* Complete HTML page as string

### `create_logo` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/ui.py#L501"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_logo(icon_url: str | None = None, alt_text: str = 'FastMCP') -> str
```

Create logo HTML.

**Args:**

* `icon_url`: Optional custom icon URL. If not provided, uses the FastMCP logo.
* `alt_text`: Alt text for the logo image.

**Returns:**

* HTML for logo image tag.

### `create_status_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/ui.py#L516"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_status_message(message: str, is_success: bool = True) -> str
```

Create a status message with icon.

**Args:**

* `message`: Status message text
* `is_success`: True for success (✓), False for error (✕)

**Returns:**

* HTML for status message

### `create_info_box` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/ui.py#L539"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_info_box(content: str, is_error: bool = False, centered: bool = False, monospace: bool = False) -> str
```

Create an info box.

**Args:**

* `content`: HTML content for the info box
* `is_error`: True for error styling, False for normal
* `centered`: True to center the text, False for left-aligned
* `monospace`: True to use gray monospace font styling instead of blue

**Returns:**

* HTML for info box

### `create_detail_box` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/ui.py#L568"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_detail_box(rows: list[tuple[str, str]]) -> str
```

Create a detail box with key-value pairs.

**Args:**

* `rows`: List of (label, value) tuples

**Returns:**

* HTML for detail box

### `create_button_group` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/ui.py#L591"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_button_group(buttons: list[tuple[str, str, str]]) -> str
```

Create a group of buttons.

**Args:**

* `buttons`: List of (text, value, css\_class) tuples

**Returns:**

* HTML for button group

### `create_secure_html_response` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/ui.py#L609"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_secure_html_response(html: str, status_code: int = 200) -> HTMLResponse
```

Create an HTMLResponse with security headers.

Adds X-Frame-Options: DENY to prevent clickjacking attacks per MCP security best practices.

**Args:**

* `html`: HTML content to return
* `status_code`: HTTP status code

**Returns:**

* HTMLResponse with security headers


# version_check
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-version_check



# `fastmcp.utilities.version_check`

Version checking utilities for FastMCP.

## Functions

### `get_latest_version` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/version_check.py#L98"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_latest_version(include_prereleases: bool = False) -> str | None
```

Get the latest version of FastMCP from PyPI, using cache when available.

**Args:**

* `include_prereleases`: If True, include pre-release versions.

**Returns:**

* The latest version string, or None if unavailable.

### `check_for_newer_version` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/version_check.py#L124"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
check_for_newer_version() -> str | None
```

Check if a newer version of FastMCP is available.

**Returns:**

* The latest version string if newer than current, None otherwise.


# versions
Source: https://gofastmcp.com/python-sdk/fastmcp-utilities-versions



# `fastmcp.utilities.versions`

Version comparison utilities for component versioning.

This module provides utilities for comparing component versions. Versions are
strings that are first attempted to be parsed as PEP 440 versions (using the
`packaging` library), falling back to lexicographic string comparison.

Examples:

* "1", "2", "10" → parsed as PEP 440, compared semantically (1 \< 2 \< 10)
* "1.0", "2.0" → parsed as PEP 440
* "v1.0" → 'v' prefix stripped, parsed as "1.0"
* "2025-01-15" → not valid PEP 440, compared as strings
* None → sorts lowest (unversioned components)

## Functions

### `parse_version_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L187"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse_version_key(version: str | None) -> VersionKey
```

Parse a version string into a sortable key.

**Args:**

* `version`: The version string, or None for unversioned.

**Returns:**

* A VersionKey suitable for sorting.

### `version_sort_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L199"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
version_sort_key(component: FastMCPComponent) -> VersionKey
```

Get a sort key for a component based on its version.

Use with sorted() or max() to order components by version.

**Args:**

* `component`: The component to get a sort key for.

**Returns:**

* A sortable VersionKey.

### `compare_versions` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L219"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
compare_versions(a: str | None, b: str | None) -> int
```

Compare two version strings.

**Args:**

* `a`: First version string (or None).
* `b`: Second version string (or None).

**Returns:**

* -1 if a \< b, 0 if a == b, 1 if a > b.

### `is_version_greater` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L241"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_version_greater(a: str | None, b: str | None) -> bool
```

Check if version a is greater than version b.

**Args:**

* `a`: First version string (or None).
* `b`: Second version string (or None).

**Returns:**

* True if a > b, False otherwise.

### `max_version` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L254"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
max_version(a: str | None, b: str | None) -> str | None
```

Return the greater of two versions.

**Args:**

* `a`: First version string (or None).
* `b`: Second version string (or None).

**Returns:**

* The greater version, or None if both are None.

### `min_version` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L271"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
min_version(a: str | None, b: str | None) -> str | None
```

Return the lesser of two versions.

**Args:**

* `a`: First version string (or None).
* `b`: Second version string (or None).

**Returns:**

* The lesser version, or None if both are None.

## Classes

### `VersionSpec` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L28"><Icon icon="github" /></a></sup>

Specification for filtering components by version.

Used by transforms and providers to filter components to a specific
version or version range. Unversioned components (version=None) always
match any spec.

**Args:**

* `gte`: If set, only versions >= this value match.
* `lt`: If set, only versions \< this value match.
* `eq`: If set, only this exact version matches (gte/lt ignored).

**Methods:**

#### `matches` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L45"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
matches(self, version: str | None) -> bool
```

Check if a version matches this spec.

**Args:**

* `version`: The version to check, or None for unversioned.
* `match_none`: Whether unversioned (None) components match. Defaults to True
  for backward compatibility with retrieval operations. Set to False
  when filtering (e.g., enable/disable) to exclude unversioned components
  from version-specific rules.

**Returns:**

* True if the version matches the spec.

#### `intersect` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L78"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
intersect(self, other: VersionSpec | None) -> VersionSpec
```

Return a spec that satisfies both this spec and other.

Used by transforms to combine caller constraints with filter constraints.
For example, if a VersionFilter has lt="3.0" and caller requests eq="1.0",
the intersection validates "1.0" is in range and returns the exact spec.

**Args:**

* `other`: Another spec to intersect with, or None.

**Returns:**

* A VersionSpec that matches only versions satisfying both specs.

### `VersionKey` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/versions.py#L114"><Icon icon="github" /></a></sup>

A comparable version key that handles None, PEP 440 versions, and strings.

Comparison order:

1. None (unversioned) sorts lowest
2. PEP 440 versions sort by semantic version order
3. Invalid versions (strings) sort lexicographically
4. When comparing PEP 440 vs string, PEP 440 comes first
