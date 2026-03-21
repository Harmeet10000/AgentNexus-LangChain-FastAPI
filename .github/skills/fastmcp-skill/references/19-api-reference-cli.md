# API Reference: CLI

Source lines: 30305-31469 from the original FastMCP documentation dump.

Package-level API reference for fastmcp.cli and install/run/task helpers.

---

# Contrib Modules
Source: https://gofastmcp.com/patterns/contrib

Community-contributed modules extending FastMCP

<VersionBadge />

FastMCP includes a `contrib` package that holds community-contributed modules. These modules extend FastMCP's functionality but aren't officially maintained by the core team.

Contrib modules provide additional features, integrations, or patterns that complement the core FastMCP library. They offer a way for the community to share useful extensions while keeping the core library focused and maintainable.

The available modules can be viewed in the [contrib directory](https://github.com/PrefectHQ/fastmcp/tree/main/src/fastmcp/contrib).

## Usage

To use a contrib module, import it from the `fastmcp.contrib` package:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.contrib import my_module
```

## Important Considerations

* **Stability**: Modules in `contrib` may have different testing requirements or stability guarantees compared to the core library.
* **Compatibility**: Changes to core FastMCP might break modules in `contrib` without explicit warnings in the main changelog.
* **Dependencies**: Contrib modules may have additional dependencies not required by the core library. These dependencies are typically documented in the module's README or separate requirements files.

## Contributing

We welcome contributions to the `contrib` package! If you have a module that extends FastMCP in a useful way, consider contributing it:

1. Create a new directory in `src/fastmcp/contrib/` for your module
2. Add proper tests for your module in `tests/contrib/`
3. Include comprehensive documentation in a README.md file, including usage and examples, as well as any additional dependencies or installation instructions
4. Submit a pull request

The ideal contrib module:

* Solves a specific use case or integration need
* Follows FastMCP coding standards
* Includes thorough documentation and examples
* Has comprehensive tests
* Specifies any additional dependencies


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-__init__



# `fastmcp.cli`

FastMCP CLI package.


# auth
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-auth



# `fastmcp.cli.auth`

Authentication-related CLI commands.


# cimd
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-cimd



# `fastmcp.cli.cimd`

CIMD (Client ID Metadata Document) CLI commands.

## Functions

### `create_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/cimd.py#L32"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_command() -> None
```

Generate a CIMD document for hosting.

Create a Client ID Metadata Document that you can host at an HTTPS URL.
The URL where you host this document becomes your client\_id.

After creating the document, host it at an HTTPS URL with a non-root path,
for example: [https://myapp.example.com/oauth/client.json](https://myapp.example.com/oauth/client.json)

### `validate_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/cimd.py#L144"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_command(url: Annotated[str, cyclopts.Parameter(help='URL of the CIMD document to validate')]) -> None
```

Validate a hosted CIMD document.

Fetches the document from the given URL and validates:

* URL is valid CIMD URL (HTTPS, non-root path)
* Document is valid JSON
* Document conforms to CIMD schema
* client\_id in document matches the URL


# cli
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-cli



# `fastmcp.cli.cli`

FastMCP CLI tools using Cyclopts.

## Functions

### `with_argv` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/cli.py#L73"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
with_argv(args: list[str] | None)
```

Temporarily replace sys.argv if args provided.

This context manager is used at the CLI boundary to inject
server arguments when needed, without mutating sys.argv deep
in the source loading logic.

Args are provided without the script name, so we preserve sys.argv\[0]
and replace the rest.

### `version` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/cli.py#L96"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
version()
```

Display version information and platform details.

### `inspector` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/cli.py#L142"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
inspector(server_spec: str | None = None) -> None
```

Run an MCP server with the MCP Inspector for development.

**Args:**

* `server_spec`: Python file to run, optionally with :object suffix, or None to auto-detect fastmcp.json

### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/cli.py#L337"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(server_spec: str | None = None, *server_args: str) -> None
```

Run an MCP server or connect to a remote one.

The server can be specified in several ways:

1. Module approach: "server.py" - runs the module directly, looking for an object named 'mcp', 'server', or 'app'
2. Import approach: "server.py:app" - imports and runs the specified server object
3. URL approach: "[http://server-url](http://server-url)" - connects to a remote server and creates a proxy
4. MCPConfig file: "mcp.json" - runs as a proxy server for the MCP Servers in the MCPConfig file
5. FastMCP config: "fastmcp.json" - runs server using FastMCP configuration
6. No argument: looks for fastmcp.json in current directory
7. Module mode: "-m my\_module" - runs the module directly via python -m

Server arguments can be passed after -- :
fastmcp run server.py -- --config config.json --debug

**Args:**

* `server_spec`: Python file, object specification (file:obj), config file, URL, or None to auto-detect

### `inspect` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/cli.py#L712"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
inspect(server_spec: str | None = None) -> None
```

Inspect an MCP server and display information or generate a JSON report.

This command analyzes an MCP server. Without flags, it displays a text summary.
Use --format to output complete JSON data.

**Examples:**

# Show text summary

fastmcp inspect server.py

# Output FastMCP format JSON to stdout

fastmcp inspect server.py --format fastmcp

# Save MCP protocol format to file (format required with -o)

fastmcp inspect server.py --format mcp -o manifest.json

# Inspect from fastmcp.json configuration

fastmcp inspect fastmcp.json
fastmcp inspect  # auto-detect fastmcp.json

**Args:**

* `server_spec`: Python file to inspect, optionally with :object suffix, or fastmcp.json

### `prepare` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/cli.py#L954"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prepare(config_path: Annotated[str | None, cyclopts.Parameter(help='Path to fastmcp.json configuration file')] = None, output_dir: Annotated[str | None, cyclopts.Parameter(help='Directory to create the persistent environment in')] = None, skip_source: Annotated[bool, cyclopts.Parameter(help='Skip source preparation (e.g., git clone)')] = False) -> None
```

Prepare a FastMCP project by creating a persistent uv environment.

This command creates a persistent uv project with all dependencies installed:

* Creates a pyproject.toml with dependencies from the config
* Installs all Python packages into a .venv
* Prepares the source (git clone, download, etc.) unless --skip-source

After running this command, you can use:
fastmcp run \<config> --project \<output-dir>

This is useful for:

* CI/CD pipelines with separate build and run stages
* Docker images where you prepare during build
* Production deployments where you want fast startup times


# client
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-client



# `fastmcp.cli.client`

Client-side CLI commands for querying and invoking MCP servers.

## Functions

### `resolve_server_spec` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/client.py#L42"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resolve_server_spec(server_spec: str | None) -> str | dict[str, Any] | ClientTransport
```

Turn CLI inputs into something `Client()` accepts.

Exactly one of `server_spec` or `command` should be provided.

Resolution order for `server_spec`:

1. URLs (`http://`, `https://`) — passed through as-is.
   If `--transport` is `sse`, the URL is rewritten to end with `/sse`
   so `infer_transport` picks the right transport.
2. Existing file paths, or strings ending in `.py`/`.js`/`.json`.
3. Anything else — name-based resolution via `resolve_name`.

When `command` is provided, the string is shell-split into a
`StdioTransport(command, args)`.

### `coerce_value` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/client.py#L263"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
coerce_value(raw: str, schema: dict[str, Any]) -> Any
```

Coerce a string CLI value according to a JSON-Schema type hint.

### `parse_tool_arguments` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/client.py#L297"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse_tool_arguments(raw_args: tuple[str, ...], input_json: str | None, input_schema: dict[str, Any]) -> dict[str, Any]
```

Build a tool-call argument dict from CLI inputs.

A single JSON object argument is treated as the full argument dict.
`--input-json` provides the base dict; `key=value` pairs override.
Values are coerced using the tool's `inputSchema`.

### `format_tool_signature` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/client.py#L369"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
format_tool_signature(tool: mcp.types.Tool) -> str
```

Build `name(param: type, ...) -> return_type` from a tool's JSON schemas.

### `list_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/client.py#L625"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_command(server_spec: Annotated[str | None, cyclopts.Parameter(help='Server URL, Python file, MCPConfig JSON, or .js file')] = None) -> None
```

List tools available on an MCP server.

**Examples:**

fastmcp list [http://localhost:8000/mcp](http://localhost:8000/mcp)
fastmcp list server.py
fastmcp list mcp.json --json
fastmcp list --command 'npx -y @mcp/server' --resources
fastmcp list [http://server/mcp](http://server/mcp) --transport sse

### `call_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/client.py#L774"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_command(server_spec: Annotated[str | None, cyclopts.Parameter(help='Server URL, Python file, MCPConfig JSON, or .js file')] = None, target: Annotated[str, cyclopts.Parameter(help='Tool name, resource URI, or prompt name (with --prompt)')] = '', *arguments: str) -> None
```

Call a tool, read a resource, or get a prompt on an MCP server.

By default the target is treated as a tool name. If the target
contains `://` it is treated as a resource URI. Pass `--prompt`
to treat it as a prompt name.

Arguments are passed as key=value pairs. Use --input-json for complex
or nested arguments.

**Examples:**

```
fastmcp call server.py greet name=World
fastmcp call server.py resource://docs/readme
fastmcp call server.py analyze --prompt data='[1,2,3]'
fastmcp call http://server/mcp create --input-json '{"tags": ["a","b"]}'
```

### `discover_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/client.py#L875"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
discover_command() -> None
```

Discover MCP servers configured in editor and project configs.

Scans Claude Desktop, Claude Code, Cursor, Gemini CLI, Goose, and
project-level mcp.json files for MCP server definitions.

Discovered server names can be used directly with `fastmcp list`
and `fastmcp call` instead of specifying a URL or file path.

**Examples:**

fastmcp discover
fastmcp discover --source claude-code
fastmcp discover --source cursor --source gemini --json
fastmcp list weather
fastmcp call cursor:weather get\_forecast city=London


# discovery
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-discovery



# `fastmcp.cli.discovery`

Discover MCP servers configured in editor config files.

Scans filesystem-readable config files from editors like Claude Desktop,
Claude Code, Cursor, Gemini CLI, and Goose, as well as project-level
`mcp.json` files. Each discovered server can be resolved by name
(or `source:name`) so the CLI can connect without requiring a URL
or file path.

## Functions

### `discover_servers` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/discovery.py#L314"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
discover_servers(start_dir: Path | None = None) -> list[DiscoveredServer]
```

Run all scanners and return the combined results.

Duplicate names across sources are preserved — callers can
use :pyattr:`DiscoveredServer.qualified_name` to disambiguate.

### `resolve_name` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/discovery.py#L331"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resolve_name(name: str, start_dir: Path | None = None) -> ClientTransport
```

Resolve a server name (or `source:name`) to a transport.

Raises :class:`ValueError` when the name is not found or is ambiguous.

## Classes

### `DiscoveredServer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/discovery.py#L37"><Icon icon="github" /></a></sup>

A single MCP server found in an editor or project config.

**Methods:**

#### `qualified_name` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/discovery.py#L46"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
qualified_name(self) -> str
```

Fully qualified `source:name` identifier.

#### `transport_summary` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/discovery.py#L51"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transport_summary(self) -> str
```

Human-readable one-liner describing the transport.


# generate
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-generate



# `fastmcp.cli.generate`

Generate a standalone CLI script and agent skill from an MCP server.

## Functions

### `serialize_transport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/generate.py#L122"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
serialize_transport(resolved: str | dict[str, Any] | ClientTransport) -> tuple[str, set[str]]
```

Serialize a resolved transport to a Python expression string.

Returns `(expression, extra_imports)` where *extra\_imports* is a set of
import lines needed by the expression.

### `generate_cli_script` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/generate.py#L283"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_cli_script(server_name: str, server_spec: str, transport_code: str, extra_imports: set[str], tools: list[mcp.types.Tool]) -> str
```

Generate the full CLI script source code.

### `generate_skill_content` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/generate.py#L621"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_skill_content(server_name: str, cli_filename: str, tools: list[mcp.types.Tool]) -> str
```

Generate a SKILL.md file for a generated CLI script.

### `generate_cli_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/generate.py#L672"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_cli_command(server_spec: Annotated[str, cyclopts.Parameter(help='Server URL, Python file, MCPConfig JSON, discovered name, or .js file')], output: Annotated[str, cyclopts.Parameter(help='Output file path (default: cli.py)')] = 'cli.py') -> None
```

Generate a standalone CLI script from an MCP server.

Connects to the server, reads its tools/resources/prompts, and writes
a Python script that can invoke them directly. Also generates a SKILL.md
agent skill file unless --no-skill is passed.

**Examples:**

fastmcp generate-cli weather
fastmcp generate-cli weather my\_cli.py
fastmcp generate-cli [http://localhost:8000/mcp](http://localhost:8000/mcp)
fastmcp generate-cli server.py output.py -f
fastmcp generate-cli weather --no-skill


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-install-__init__



# `fastmcp.cli.install`

Install subcommands for FastMCP CLI using Cyclopts.


# claude_code
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-install-claude_code



# `fastmcp.cli.install.claude_code`

Claude Code integration for FastMCP install using Cyclopts.

## Functions

### `find_claude_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/claude_code.py#L20"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
find_claude_command() -> str | None
```

Find the Claude Code CLI command.

Checks common installation locations since 'claude' is often a shell alias
that doesn't work with subprocess calls.

### `check_claude_code_available` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/claude_code.py#L68"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
check_claude_code_available() -> bool
```

Check if Claude Code CLI is available.

### `install_claude_code` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/claude_code.py#L73"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
install_claude_code(file: Path, server_object: str | None, name: str) -> bool
```

Install FastMCP server in Claude Code.

**Args:**

* `file`: Path to the server file
* `server_object`: Optional server object name (for :object suffix)
* `name`: Name for the server in Claude Code
* `with_editable`: Optional list of directories to install in editable mode
* `with_packages`: Optional list of additional packages to install
* `env_vars`: Optional dictionary of environment variables
* `python_version`: Optional Python version to use
* `with_requirements`: Optional requirements file to install from
* `project`: Optional project directory to run within

**Returns:**

* True if installation was successful, False otherwise

### `claude_code_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/claude_code.py#L153"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
claude_code_command(server_spec: str) -> None
```

Install an MCP server in Claude Code.

**Args:**

* `server_spec`: Python file to install, optionally with :object suffix


# claude_desktop
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-install-claude_desktop



# `fastmcp.cli.install.claude_desktop`

Claude Desktop integration for FastMCP install using Cyclopts.

## Functions

### `get_claude_config_path` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/claude_desktop.py#L20"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_claude_config_path() -> Path | None
```

Get the Claude config directory based on platform.

### `install_claude_desktop` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/claude_desktop.py#L38"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
install_claude_desktop(file: Path, server_object: str | None, name: str) -> bool
```

Install FastMCP server in Claude Desktop.

**Args:**

* `file`: Path to the server file
* `server_object`: Optional server object name (for :object suffix)
* `name`: Name for the server in Claude's config
* `with_editable`: Optional list of directories to install in editable mode
* `with_packages`: Optional list of additional packages to install
* `env_vars`: Optional dictionary of environment variables
* `python_version`: Optional Python version to use
* `with_requirements`: Optional requirements file to install from
* `project`: Optional project directory to run within

**Returns:**

* True if installation was successful, False otherwise

### `claude_desktop_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/claude_desktop.py#L125"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
claude_desktop_command(server_spec: str) -> None
```

Install an MCP server in Claude Desktop.

**Args:**

* `server_spec`: Python file to install, optionally with :object suffix


# cursor
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-install-cursor



# `fastmcp.cli.install.cursor`

Cursor integration for FastMCP install using Cyclopts.

## Functions

### `generate_cursor_deeplink` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/cursor.py#L22"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_cursor_deeplink(server_name: str, server_config: StdioMCPServer) -> str
```

Generate a Cursor deeplink for installing the MCP server.

**Args:**

* `server_name`: Name of the server
* `server_config`: Server configuration

**Returns:**

* Deeplink URL that can be clicked to install the server

### `open_deeplink` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/cursor.py#L47"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
open_deeplink(deeplink: str) -> bool
```

Attempt to open a Cursor deeplink URL using the system's default handler.

**Args:**

* `deeplink`: The deeplink URL to open

**Returns:**

* True if the command succeeded, False otherwise

### `install_cursor_workspace` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/cursor.py#L59"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
install_cursor_workspace(file: Path, server_object: str | None, name: str, workspace_path: Path) -> bool
```

Install FastMCP server to workspace-specific Cursor configuration.

**Args:**

* `file`: Path to the server file
* `server_object`: Optional server object name (for :object suffix)
* `name`: Name for the server in Cursor
* `workspace_path`: Path to the workspace directory
* `with_editable`: Optional list of directories to install in editable mode
* `with_packages`: Optional list of additional packages to install
* `env_vars`: Optional dictionary of environment variables
* `python_version`: Optional Python version to use
* `with_requirements`: Optional requirements file to install from
* `project`: Optional project directory to run within

**Returns:**

* True if installation was successful, False otherwise

### `install_cursor` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/cursor.py#L140"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
install_cursor(file: Path, server_object: str | None, name: str) -> bool
```

Install FastMCP server in Cursor.

**Args:**

* `file`: Path to the server file
* `server_object`: Optional server object name (for :object suffix)
* `name`: Name for the server in Cursor
* `with_editable`: Optional list of directories to install in editable mode
* `with_packages`: Optional list of additional packages to install
* `env_vars`: Optional dictionary of environment variables
* `python_version`: Optional Python version to use
* `with_requirements`: Optional requirements file to install from
* `project`: Optional project directory to run within
* `workspace`: Optional workspace directory for project-specific installation

**Returns:**

* True if installation was successful, False otherwise

### `cursor_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/cursor.py#L225"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
cursor_command(server_spec: str) -> None
```

Install an MCP server in Cursor.

**Args:**

* `server_spec`: Python file to install, optionally with :object suffix


# gemini_cli
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-install-gemini_cli



# `fastmcp.cli.install.gemini_cli`

Gemini CLI integration for FastMCP install using Cyclopts.

## Functions

### `find_gemini_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/gemini_cli.py#L20"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
find_gemini_command() -> str | None
```

Find the Gemini CLI command.

### `check_gemini_cli_available` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/gemini_cli.py#L64"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
check_gemini_cli_available() -> bool
```

Check if Gemini CLI is available.

### `install_gemini_cli` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/gemini_cli.py#L69"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
install_gemini_cli(file: Path, server_object: str | None, name: str) -> bool
```

Install FastMCP server in Gemini CLI.

**Args:**

* `file`: Path to the server file
* `server_object`: Optional server object name (for :object suffix)
* `name`: Name for the server in Gemini CLI
* `with_editable`: Optional list of directories to install in editable mode
* `with_packages`: Optional list of additional packages to install
* `env_vars`: Optional dictionary of environment variables
* `python_version`: Optional Python version to use
* `with_requirements`: Optional requirements file to install from
* `project`: Optional project directory to run within

**Returns:**

* True if installation was successful, False otherwise

### `gemini_cli_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/gemini_cli.py#L150"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
gemini_cli_command(server_spec: str) -> None
```

Install an MCP server in Gemini CLI.

**Args:**

* `server_spec`: Python file to install, optionally with :object suffix


# goose
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-install-goose



# `fastmcp.cli.install.goose`

Goose integration for FastMCP install using Cyclopts.

## Functions

### `generate_goose_deeplink` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/goose.py#L29"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_goose_deeplink(name: str, command: str, args: list[str]) -> str
```

Generate a Goose deeplink for installing an MCP extension.

**Args:**

* `name`: Human-readable display name for the extension.
* `command`: The executable command (e.g. "uv").
* `args`: Arguments to the command.
* `description`: Short description shown in Goose.

**Returns:**

* A goose://extension?... deeplink URL.

### `install_goose` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/goose.py#L86"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
install_goose(file: Path, server_object: str | None, name: str) -> bool
```

Install FastMCP server in Goose via deeplink.

**Args:**

* `file`: Path to the server file.
* `server_object`: Optional server object name (for :object suffix).
* `name`: Name for the extension in Goose.
* `with_packages`: Optional list of additional packages to install.
* `python_version`: Optional Python version to use.

**Returns:**

* True if installation was successful, False otherwise.

### `goose_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/goose.py#L136"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
goose_command(server_spec: str) -> None
```

Install an MCP server in Goose.

Uses uvx to run the server. Environment variables are not included
in the deeplink; use `fastmcp install mcp-json` to generate a full
config for manual installation.

**Args:**

* `server_spec`: Python file to install, optionally with :object suffix


# mcp_json
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-install-mcp_json



# `fastmcp.cli.install.mcp_json`

MCP configuration JSON generation for FastMCP install using Cyclopts.

## Functions

### `install_mcp_json` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/mcp_json.py#L20"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
install_mcp_json(file: Path, server_object: str | None, name: str) -> bool
```

Generate MCP configuration JSON for manual installation.

**Args:**

* `file`: Path to the server file
* `server_object`: Optional server object name (for :object suffix)
* `name`: Name for the server in MCP config
* `with_editable`: Optional list of directories to install in editable mode
* `with_packages`: Optional list of additional packages to install
* `env_vars`: Optional dictionary of environment variables
* `copy`: If True, copy to clipboard instead of printing to stdout
* `python_version`: Optional Python version to use
* `with_requirements`: Optional requirements file to install from
* `project`: Optional project directory to run within

**Returns:**

* True if generation was successful, False otherwise

### `mcp_json_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/mcp_json.py#L98"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
mcp_json_command(server_spec: str) -> None
```

Generate MCP configuration JSON for manual installation.

**Args:**

* `server_spec`: Python file to install, optionally with :object suffix


# shared
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-install-shared



# `fastmcp.cli.install.shared`

Shared utilities for install commands.

## Functions

### `parse_env_var` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/shared.py#L21"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse_env_var(env_var: str) -> tuple[str, str]
```

Parse environment variable string in format KEY=VALUE.

### `process_common_args` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/shared.py#L32"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
process_common_args(server_spec: str, server_name: str | None, with_packages: list[str] | None, env_vars: list[str] | None, env_file: Path | None) -> tuple[Path, str | None, str, list[str], dict[str, str] | None]
```

Process common arguments shared by all install commands.

Handles both fastmcp.json config files and traditional file.py:object syntax.

### `open_deeplink` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/shared.py#L148"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
open_deeplink(url: str) -> bool
```

Attempt to open a deeplink URL using the system's default handler.

**Args:**

* `url`: The deeplink URL to open.
* `expected_scheme`: The URL scheme to validate (e.g. "cursor", "goose").

**Returns:**

* True if the command succeeded, False otherwise.


# stdio
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-install-stdio



# `fastmcp.cli.install.stdio`

Stdio command generation for FastMCP install using Cyclopts.

## Functions

### `install_stdio` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/stdio.py#L21"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
install_stdio(file: Path, server_object: str | None) -> bool
```

Generate the stdio command for running a FastMCP server.

**Args:**

* `file`: Path to the server file
* `server_object`: Optional server object name (for :object suffix)
* `with_editable`: Optional list of directories to install in editable mode
* `with_packages`: Optional list of additional packages to install
* `copy`: If True, copy to clipboard instead of printing to stdout
* `python_version`: Optional Python version to use
* `with_requirements`: Optional requirements file to install from
* `project`: Optional project directory to run within

**Returns:**

* True if generation was successful, False otherwise

### `stdio_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/install/stdio.py#L78"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
stdio_command(server_spec: str) -> None
```

Generate the stdio command for running a FastMCP server.

Outputs the shell command that an MCP host would use to start this server
over stdio transport. Useful for manual configuration or debugging.

**Args:**

* `server_spec`: Python file to run, optionally with :object suffix


# run
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-run



# `fastmcp.cli.run`

FastMCP run command implementation with enhanced type hints.

## Functions

### `is_url` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/run.py#L83"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_url(path: str) -> bool
```

Check if a string is a URL.

### `create_client_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/run.py#L89"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_client_server(url: str) -> Any
```

Create a FastMCP server from a client URL.

**Args:**

* `url`: The URL to connect to

**Returns:**

* A FastMCP server instance

### `create_mcp_config_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/run.py#L109"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_mcp_config_server(mcp_config_path: Path) -> FastMCP[None]
```

Create a FastMCP server from a MCPConfig.

### `load_mcp_server_config` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/run.py#L118"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
load_mcp_server_config(config_path: Path) -> MCPServerConfig
```

Load a FastMCP configuration from a fastmcp.json file.

**Args:**

* `config_path`: Path to fastmcp.json file

**Returns:**

* MCPServerConfig object

### `run_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/run.py#L135"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_command(server_spec: str, transport: TransportType | None = None, host: str | None = None, port: int | None = None, path: str | None = None, log_level: LogLevelType | None = None, server_args: list[str] | None = None, show_banner: bool = True, use_direct_import: bool = False, skip_source: bool = False, stateless: bool = False) -> None
```

Run a MCP server or connect to a remote one.

**Args:**

* `server_spec`: Python file, object specification (file:obj), config file, or URL
* `transport`: Transport protocol to use
* `host`: Host to bind to when using http transport
* `port`: Port to bind to when using http transport
* `path`: Path to bind to when using http transport
* `log_level`: Log level
* `server_args`: Additional arguments to pass to the server
* `show_banner`: Whether to show the server banner
* `use_direct_import`: Whether to use direct import instead of subprocess
* `skip_source`: Whether to skip source preparation step
* `stateless`: Whether to run in stateless mode (no session)

### `run_module_command` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/run.py#L260"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_module_command(module_name: str) -> None
```

Run a Python module directly using `python -m <module>`.

When `-m` is used, the module manages its own server startup.
No server-object discovery or transport overrides are applied.

**Args:**

* `module_name`: Dotted module name (e.g. `my_package`).
* `env_command_builder`: An optional callable that wraps a command list
  with environment setup (e.g. `UVEnvironment.build_command`).
* `extra_args`: Extra arguments forwarded after the module name.

### `run_v1_server_async` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/run.py#L299"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_v1_server_async(server: FastMCP1x, host: str | None = None, port: int | None = None, transport: TransportType | None = None) -> None
```

Run a FastMCP 1.x server using async methods.

**Args:**

* `server`: FastMCP 1.x server instance
* `host`: Host to bind to
* `port`: Port to bind to
* `transport`: Transport protocol to use

### `run_with_reload` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/run.py#L364"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run_with_reload(cmd: list[str], reload_dirs: list[Path] | None = None, is_stdio: bool = False) -> None
```

Run a command with file watching and auto-reload.

**Args:**

* `cmd`: Command to run as subprocess (should include --no-reload)
* `reload_dirs`: Directories to watch for changes (default: cwd)
* `is_stdio`: Whether this is stdio transport


# tasks
Source: https://gofastmcp.com/python-sdk/fastmcp-cli-tasks



# `fastmcp.cli.tasks`

FastMCP tasks CLI for Docket task management.

## Functions

### `check_distributed_backend` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/tasks.py#L22"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
check_distributed_backend() -> None
```

Check if Docket is configured with a distributed backend.

The CLI worker runs as a separate process, so it needs Redis/Valkey
to coordinate with the main server process.

**Raises:**

* `SystemExit`: If using memory:// URL

### `worker` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/cli/tasks.py#L61"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
worker(server_spec: Annotated[str | None, cyclopts.Parameter(help='Python file to run, optionally with :object suffix, or None to auto-detect fastmcp.json')] = None) -> None
```

Start an additional worker to process background tasks.

Connects to your Docket backend and processes tasks in parallel with
any other running workers. Configure via environment variables
(FASTMCP\_DOCKET\_\*).
