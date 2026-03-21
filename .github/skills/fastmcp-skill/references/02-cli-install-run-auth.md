# CLI, Installation, and Client Auth

Source lines: 1046-2045 from the original FastMCP documentation dump.

Client CLI commands, inspecting servers, installing servers, running servers, and client-side bearer/CIMD/OAuth authentication helpers.

---

# Client Commands
Source: https://gofastmcp.com/cli/client

List tools, call them, and discover configured servers

<VersionBadge />

The CLI can act as an MCP client — connecting to any server (local or remote) to list what it exposes and call its tools directly. This is useful for development, debugging, scripting, and giving shell-capable LLM agents access to MCP servers.

## Listing Tools

`fastmcp list` connects to a server and prints its tools as function signatures, showing parameter names, types, and descriptions at a glance:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list http://localhost:8000/mcp
fastmcp list server.py
fastmcp list weather  # name-based resolution
```

When you need the full JSON Schema for a tool's inputs or outputs — for understanding nested objects, enum constraints, or complex types — opt in with `--input-schema` or `--output-schema`:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list server.py --input-schema
```

### Resources and Prompts

By default, only tools are shown. Add `--resources` or `--prompts` to include those:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list server.py --resources --prompts
```

### Machine-Readable Output

The `--json` flag switches to structured JSON with full schemas included. This is the format to use when feeding tool definitions to an LLM or building automation:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list server.py --json
```

### Options

| Option        | Flag                | Description                                           |
| ------------- | ------------------- | ----------------------------------------------------- |
| Command       | `--command`         | Connect via stdio (e.g., `'npx -y @mcp/server'`)      |
| Transport     | `--transport`, `-t` | Force `http` or `sse` for URL targets                 |
| Resources     | `--resources`       | Include resources in output                           |
| Prompts       | `--prompts`         | Include prompts in output                             |
| Input Schema  | `--input-schema`    | Show full input schemas                               |
| Output Schema | `--output-schema`   | Show full output schemas                              |
| JSON          | `--json`            | Structured JSON output                                |
| Timeout       | `--timeout`         | Connection timeout in seconds                         |
| Auth          | `--auth`            | `oauth` (default for HTTP), a bearer token, or `none` |

## Calling Tools

`fastmcp call` invokes a single tool on a server. Pass arguments as `key=value` pairs — the CLI fetches the tool's schema and coerces your string values to the right types automatically:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp call server.py greet name=World
fastmcp call http://localhost:8000/mcp search query=hello limit=5
```

Type coercion is schema-driven: `"5"` becomes the integer `5` when the schema expects an integer. Booleans accept `true`/`false`, `yes`/`no`, and `1`/`0`. Arrays and objects are parsed as JSON.

### Complex Arguments

For tools with nested or structured parameters, `key=value` syntax gets awkward. Pass a single JSON object instead:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp call server.py create_item '{"name": "Widget", "tags": ["sale"], "metadata": {"color": "blue"}}'
```

Or use `--input-json` to provide a base dictionary, then override individual keys with `key=value` pairs:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp call server.py search --input-json '{"query": "hello", "limit": 5}' limit=10
```

### Error Handling

If you misspell a tool name, the CLI suggests corrections via fuzzy matching. Missing required arguments produce a clear message with the tool's signature as a reminder. Tool execution errors are printed with a non-zero exit code, making the CLI straightforward to use in scripts.

### Structured Output

`--json` emits the raw result including content blocks, error status, and structured content:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp call server.py get_weather city=London --json
```

### Interactive Elicitation

Some tools request additional input during execution through MCP's elicitation mechanism. When this happens, the CLI prompts you in the terminal — showing each field's name, type, and whether it's required. You can type `decline` to skip a question or `cancel` to abort the call entirely.

### Options

| Option     | Flag                | Description                                      |
| ---------- | ------------------- | ------------------------------------------------ |
| Command    | `--command`         | Connect via stdio                                |
| Transport  | `--transport`, `-t` | Force `http` or `sse`                            |
| Input JSON | `--input-json`      | Base arguments as JSON (merged with `key=value`) |
| JSON       | `--json`            | Raw JSON output                                  |
| Timeout    | `--timeout`         | Connection timeout in seconds                    |
| Auth       | `--auth`            | `oauth`, a bearer token, or `none`               |

## Discovering Configured Servers

`fastmcp discover` scans your machine for MCP servers configured in editors and tools. It checks:

* **Claude Desktop** — `claude_desktop_config.json`
* **Claude Code** — `~/.claude.json`
* **Cursor** — `.cursor/mcp.json` (walks up from current directory)
* **Gemini CLI** — `~/.gemini/settings.json`
* **Goose** — `~/.config/goose/config.yaml`
* **Project** — `./mcp.json` in the current directory

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp discover
```

The output groups servers by source, showing each server's name and transport. Filter by source or get machine-readable output:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp discover --source claude-code
fastmcp discover --source cursor --source gemini --json
```

Any server that appears here can be used by name with `list`, `call`, and other commands — so you can go from "I have a server in Claude Code" to querying it without copying URLs or paths.

## LLM Agent Integration

For LLM agents that can execute shell commands but don't have native MCP support, the CLI provides a clean bridge. The agent calls `fastmcp list --json` to discover available tools with full schemas, then `fastmcp call --json` to invoke them with structured results.

Because the CLI handles connection management, transport selection, and type coercion internally, the agent doesn't need to understand MCP protocol details — it just reads JSON and constructs shell commands.


# Inspecting Servers
Source: https://gofastmcp.com/cli/inspecting

View a server's components and metadata

<VersionBadge />

`fastmcp inspect` loads a server and reports what it contains — its tools, resources, prompts, version, and metadata. The default output is a human-readable summary:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp inspect server.py
```

```
Server: MyServer
Instructions: A helpful MCP server
Version: 1.0.0

Components:
  Tools: 5
  Prompts: 2
  Resources: 3
  Templates: 1

Environment:
  FastMCP: 2.0.0
  MCP: 1.0.0

Use --format [fastmcp|mcp] for complete JSON output
```

## JSON Output

For programmatic use, two JSON formats are available:

**FastMCP format** (`--format fastmcp`) includes everything FastMCP knows about the server — tool tags, enabled status, output schemas, annotations, and custom metadata. Field names use `snake_case`. This is the format for debugging and introspecting FastMCP servers.

**MCP protocol format** (`--format mcp`) shows exactly what MCP clients see through the protocol — only standard MCP fields, `camelCase` names, no FastMCP-specific extensions. This is the format for verifying client compatibility and debugging what clients actually receive.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Full FastMCP metadata to stdout
fastmcp inspect server.py --format fastmcp

# MCP protocol view saved to file
fastmcp inspect server.py --format mcp -o manifest.json
```

## Options

| Option      | Flag             | Description                                   |
| ----------- | ---------------- | --------------------------------------------- |
| Format      | `--format`, `-f` | `fastmcp` or `mcp` (required when using `-o`) |
| Output File | `--output`, `-o` | Save to file instead of stdout                |

## Entrypoints

The `inspect` command supports the same local entrypoints as [`fastmcp run`](/cli/running): inferred instances, explicit entrypoints, factory functions, and `fastmcp.json` configs.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp inspect server.py                  # inferred instance
fastmcp inspect server.py:my_server        # explicit entrypoint
fastmcp inspect server.py:create_server    # factory function
fastmcp inspect fastmcp.json               # config file
```

<Warning>
  `inspect` only works with local files and `fastmcp.json` — it doesn't connect to remote URLs or standard MCP config files.
</Warning>


# Install MCP Servers
Source: https://gofastmcp.com/cli/install-mcp

Install MCP servers into Claude, Cursor, Gemini, and other clients

<VersionBadge />

`fastmcp install` registers a server with an MCP client application so the client can launch it automatically. Each MCP client runs servers in its own isolated environment, which means dependencies need to be explicitly declared — you can't rely on whatever happens to be installed locally.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp install claude-desktop server.py
fastmcp install claude-code server.py --with pandas --with matplotlib
fastmcp install cursor server.py -e .
```

<Warning>
  `uv` must be installed and available in your system PATH. Both Claude Desktop and Cursor run servers in isolated environments managed by `uv`. On macOS, install it globally with Homebrew for Claude Desktop compatibility: `brew install uv`.
</Warning>

## Supported Clients

| Client           | Install method                                          |
| ---------------- | ------------------------------------------------------- |
| `claude-code`    | Claude Code's built-in MCP management                   |
| `claude-desktop` | Direct config file modification                         |
| `cursor`         | Deeplink that opens Cursor for confirmation             |
| `gemini-cli`     | Gemini CLI's built-in MCP management                    |
| `goose`          | Deeplink that opens Goose for confirmation (uses `uvx`) |
| `mcp-json`       | Generates standard MCP JSON config for manual use       |
| `stdio`          | Outputs the shell command to run via stdio              |

## Declaring Dependencies

Because MCP clients run servers in isolation, you need to tell the install command what your server needs. There are two approaches:

**Command-line flags** let you specify dependencies directly:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp install claude-desktop server.py --with pandas --with "sqlalchemy>=2.0"
fastmcp install cursor server.py -e . --with-requirements requirements.txt
```

**`fastmcp.json`** configuration files declare dependencies alongside the server definition. When you install from a config file, dependencies are picked up automatically:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp install claude-desktop fastmcp.json
fastmcp install claude-desktop  # auto-detects fastmcp.json in current directory
```

See [Server Configuration](/deployment/server-configuration) for the full config format.

## Options

| Option                | Flag                    | Description                          |
| --------------------- | ----------------------- | ------------------------------------ |
| Server Name           | `--server-name`, `-n`   | Custom name for the server           |
| Editable Package      | `--with-editable`, `-e` | Install a directory in editable mode |
| Extra Packages        | `--with`                | Additional packages (repeatable)     |
| Environment Variables | `--env`                 | `KEY=VALUE` pairs (repeatable)       |
| Environment File      | `--env-file`, `-f`      | Load env vars from a `.env` file     |
| Python                | `--python`              | Python version (e.g., `3.11`)        |
| Project               | `--project`             | Run within a uv project directory    |
| Requirements          | `--with-requirements`   | Install from a requirements file     |

## Examples

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Basic install with auto-detected server instance
fastmcp install claude-desktop server.py

# Install from fastmcp.json with auto-detection
fastmcp install claude-desktop

# Explicit entrypoint with dependencies
fastmcp install claude-desktop server.py:my_server \
  --server-name "My Analysis Server" \
  --with pandas

# With environment variables
fastmcp install claude-code server.py \
  --env API_KEY=secret \
  --env DEBUG=true

# With env file
fastmcp install cursor server.py --env-file .env

# Specific Python version and requirements file
fastmcp install claude-desktop server.py \
  --python 3.11 \
  --with-requirements requirements.txt
```

## Generating MCP JSON

The `mcp-json` target generates standard MCP configuration JSON instead of installing into a specific client. This is useful for clients that FastMCP doesn't directly support, for CI/CD environments, or for sharing server configs:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp install mcp-json server.py
```

The output follows the standard format used by Claude Desktop, Cursor, and other MCP clients:

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
{
  "server-name": {
    "command": "uv",
    "args": ["run", "--with", "fastmcp", "fastmcp", "run", "/path/to/server.py"],
    "env": {
      "API_KEY": "value"
    }
  }
}
```

Use `--copy` to send it to your clipboard instead of stdout.

## Generating Stdio Commands

The `stdio` target outputs the shell command an MCP host would use to start your server over stdio:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp install stdio server.py
# Output: uv run --with fastmcp fastmcp run /absolute/path/to/server.py
```

When installing from a `fastmcp.json`, dependencies from the config are included automatically:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp install stdio fastmcp.json
# Output: uv run --with fastmcp --with pillow --with 'qrcode[pil]>=8.0' fastmcp run /path/to/server.py
```

Use `--copy` to copy to clipboard.

<Tip>
  `fastmcp install` is designed for local server files with stdio transport. For remote servers running over HTTP, use your client's native configuration — FastMCP's value here is simplifying the complex local setup with `uv`, dependencies, and environment variables.
</Tip>


# CLI
Source: https://gofastmcp.com/cli/overview

The fastmcp command-line interface

The `fastmcp` CLI is installed automatically with FastMCP. It's the primary way to run, test, install, and interact with MCP servers from your terminal.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp --help
```

## Commands at a Glance

| Command                                                        | What it does                                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [`run`](/cli/running)                                          | Run a server (local file, factory function, remote URL, or config file)         |
| [`dev inspector`](/cli/running#development-with-the-inspector) | Launch a server inside the MCP Inspector for interactive testing                |
| [`install`](/cli/install-mcp)                                  | Install a server into Claude Code, Claude Desktop, Cursor, Gemini CLI, or Goose |
| [`inspect`](/cli/inspecting)                                   | Print a server's tools, resources, and prompts as a summary or JSON report      |
| [`list`](/cli/client)                                          | List a server's tools (and optionally resources and prompts)                    |
| [`call`](/cli/client#calling-tools)                            | Call a single tool with arguments                                               |
| [`discover`](/cli/client#discovering-configured-servers)       | Find MCP servers configured in your editors and tools                           |
| [`generate-cli`](/cli/generate-cli)                            | Scaffold a standalone typed CLI from a server's tool schemas                    |
| [`project prepare`](/cli/running#pre-building-environments)    | Pre-install dependencies into a reusable uv project                             |
| [`auth cimd`](/cli/auth)                                       | Create and validate CIMD documents for OAuth                                    |
| `version`                                                      | Print version info (`--copy` to copy to clipboard)                              |

## Server Targets

Most commands need to know *which server* to talk to. You pass a "server spec" as the first argument, and FastMCP resolves the right transport automatically.

**URLs** connect to a running HTTP server:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list http://localhost:8000/mcp
fastmcp call http://localhost:8000/mcp get_forecast city=London
```

**Python files** are loaded directly — no `mcp.run()` boilerplate needed. FastMCP finds a server instance named `mcp`, `server`, or `app` in the file, or you can specify one explicitly:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list server.py
fastmcp run server.py:my_custom_server
```

**Config files** work too — both FastMCP's own `fastmcp.json` format and standard MCP config files with an `mcpServers` key:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run fastmcp.json
fastmcp list mcp-config.json
```

**Stdio commands** connect to any MCP server that speaks over standard I/O. Use `--command` instead of a positional argument:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list --command 'npx -y @modelcontextprotocol/server-github'
```

### Name-Based Resolution

If your servers are already configured in an editor or tool, you can refer to them by name. FastMCP scans configs from Claude Desktop, Claude Code, Cursor, Gemini CLI, and Goose:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list weather
fastmcp call weather get_forecast city=London
```

When the same name appears in multiple configs, use the `source:name` form to be specific:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list claude-code:my-server
fastmcp call cursor:weather get_forecast city=London
```

Run [`fastmcp discover`](/cli/client#discovering-configured-servers) to see what's available on your machine.

## Authentication

When targeting an HTTP URL, the CLI enables OAuth authentication by default. If the server requires it, you'll be guided through the flow (typically opening a browser). If it doesn't, the setup is a silent no-op.

To skip authentication entirely — useful for local development servers — pass `--auth none`:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp call http://localhost:8000/mcp my_tool --auth none
```

You can also pass a bearer token directly:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list http://localhost:8000/mcp --auth "Bearer sk-..."
```

## Transport Override

FastMCP defaults to Streamable HTTP for URL targets. If the server only supports Server-Sent Events (SSE), force the older transport:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp list http://localhost:8000 --transport sse
```


# Running Servers
Source: https://gofastmcp.com/cli/running

Start, develop, and configure servers from the command line

## Starting a Server

`fastmcp run` starts a server. Point it at a Python file, a factory function, a remote URL, or a config file:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py
fastmcp run server.py:create_server
fastmcp run https://example.com/mcp
fastmcp run fastmcp.json
```

By default, the server runs over **stdio** — the transport that MCP clients like Claude Desktop expect. To serve over HTTP instead, specify the transport:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py --transport http
fastmcp run server.py --transport http --host 0.0.0.0 --port 9000
```

### Entrypoints

FastMCP supports several ways to locate and start your server:

**Inferred instance** — FastMCP imports the file and looks for a variable named `mcp`, `server`, or `app`:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py
```

**Explicit instance** — point at a specific variable:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py:my_server
```

**Factory function** — FastMCP calls the function and uses the returned server. Useful when your server needs async setup or configuration that runs before startup:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run server.py:create_server
```

**Remote URL** — starts a local proxy that bridges to a remote server. Handy for local development against a deployed server, or for bridging a remote HTTP server to stdio:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run https://example.com/mcp
```

**FastMCP config** — uses a `fastmcp.json` file that declaratively specifies the server, its dependencies, and deployment settings. When you run `fastmcp run` with no arguments, it auto-detects `fastmcp.json` in the current directory:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run
fastmcp run my-config.fastmcp.json
```

See [Server Configuration](/deployment/server-configuration) for the full `fastmcp.json` format.

**MCP config** — runs servers defined in a standard MCP configuration file (any `.json` with an `mcpServers` key):

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run mcp.json
```

<Warning>
  `fastmcp run` completely ignores the `if __name__ == "__main__"` block. Any setup code in that block won't execute. If you need initialization logic to run, use a [factory function](/cli/overview#factory-functions).
</Warning>

### Options

| Option         | Flag                       | Description                                             |
| -------------- | -------------------------- | ------------------------------------------------------- |
| Transport      | `--transport`, `-t`        | `stdio` (default), `http`, or `sse`                     |
| Host           | `--host`                   | Bind address for HTTP (default: `127.0.0.1`)            |
| Port           | `--port`, `-p`             | Bind port for HTTP (default: `8000`)                    |
| Path           | `--path`                   | URL path for HTTP (default: `/mcp/`)                    |
| Log Level      | `--log-level`, `-l`        | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`         |
| No Banner      | `--no-banner`              | Suppress the startup banner                             |
| Auto-Reload    | `--reload` / `--no-reload` | Watch for file changes and restart automatically        |
| Reload Dirs    | `--reload-dir`             | Directories to watch (repeatable)                       |
| Skip Env       | `--skip-env`               | Don't set up a uv environment (use when already in one) |
| Python         | `--python`                 | Python version to use (e.g., `3.11`)                    |
| Extra Packages | `--with`                   | Additional packages to install (repeatable)             |
| Project        | `--project`                | Run within a specific uv project directory              |
| Requirements   | `--with-requirements`      | Install from a requirements file                        |

### Dependency Management

By default, `fastmcp run` uses your current Python environment directly. When you pass `--python`, `--with`, `--project`, or `--with-requirements`, it switches to running via `uv run` in a subprocess, which handles dependency isolation automatically.

The `--skip-env` flag is useful when you're already inside an activated venv, a Docker container with pre-installed dependencies, or a uv-managed project — it prevents uv from trying to set up another environment layer.

## Development with the Inspector

`fastmcp dev inspector` launches your server inside the [MCP Inspector](https://github.com/modelcontextprotocol/inspector), a browser-based tool for interactively testing MCP servers. Auto-reload is on by default, so your server restarts when you save changes.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp dev inspector server.py
fastmcp dev inspector server.py -e . --with pandas
```

<Tip>
  The Inspector always runs your server via `uv run` in a subprocess — it never uses your local environment directly. Specify dependencies with `--with`, `--with-editable`, `--with-requirements`, or through a `fastmcp.json` file.
</Tip>

<Warning>
  The Inspector connects over **stdio only**. When it launches, you may need to select "STDIO" from the transport dropdown and click connect. To test a server over HTTP, start it separately with `fastmcp run server.py --transport http` and point the Inspector at the URL.
</Warning>

| Option            | Flag                       | Description                          |
| ----------------- | -------------------------- | ------------------------------------ |
| Editable Package  | `--with-editable`, `-e`    | Install a directory in editable mode |
| Extra Packages    | `--with`                   | Additional packages (repeatable)     |
| Inspector Version | `--inspector-version`      | MCP Inspector version to use         |
| UI Port           | `--ui-port`                | Port for the Inspector UI            |
| Server Port       | `--server-port`            | Port for the Inspector proxy         |
| Auto-Reload       | `--reload` / `--no-reload` | File watching (default: on)          |
| Reload Dirs       | `--reload-dir`             | Directories to watch (repeatable)    |
| Python            | `--python`                 | Python version                       |
| Project           | `--project`                | Run within a uv project directory    |
| Requirements      | `--with-requirements`      | Install from a requirements file     |

## Pre-Building Environments

`fastmcp project prepare` creates a persistent uv project from a `fastmcp.json` file, pre-installing all dependencies. This separates environment setup from server execution — install once, run many times.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Step 1: Build the environment (slow, does dependency resolution)
fastmcp project prepare fastmcp.json --output-dir ./env

# Step 2: Run using the prepared environment (fast, no install step)
fastmcp run fastmcp.json --project ./env
```

The prepared directory contains a `pyproject.toml`, a `.venv` with all packages installed, and a `uv.lock` for reproducibility. This is particularly useful in deployment scenarios where you want deterministic, pre-built environments.


# Bearer Token Authentication
Source: https://gofastmcp.com/clients/auth/bearer

Authenticate your FastMCP client with a Bearer token.

<VersionBadge />

<Tip>
  Bearer Token authentication is only relevant for HTTP-based transports.
</Tip>

You can configure your FastMCP client to use **bearer authentication** by supplying a valid access token. This is most appropriate for service accounts, long-lived API keys, CI/CD, applications where authentication is managed separately, or other non-interactive authentication methods.

A Bearer token is a JSON Web Token (JWT) that is used to authenticate a request. It is most commonly used in the `Authorization` header of an HTTP request, using the `Bearer` scheme:

```http theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
Authorization: Bearer <token>
```

## Client Usage

The most straightforward way to use a pre-existing Bearer token is to provide it as a string to the `auth` parameter of the `fastmcp.Client` or transport instance. FastMCP will automatically format it correctly for the `Authorization` header and bearer scheme.

<Tip>
  If you're using a string token, do not include the `Bearer` prefix. FastMCP will add it for you.
</Tip>

```python {5} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

async with Client(
    "https://your-server.fastmcp.app/mcp", 
    auth="<your-token>",
) as client:
    await client.ping()
```

You can also supply a Bearer token to a transport instance, such as `StreamableHttpTransport` or `SSETransport`:

```python {6} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    "http://your-server.fastmcp.app/mcp", 
    auth="<your-token>",
)

async with Client(transport) as client:
    await client.ping()
```

## `BearerAuth` Helper

If you prefer to be more explicit and not rely on FastMCP to transform your string token, you can use the `BearerAuth` class yourself, which implements the `httpx.Auth` interface.

```python {6} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

async with Client(
    "https://your-server.fastmcp.app/mcp", 
    auth=BearerAuth(token="<your-token>"),
) as client:
    await client.ping()
```

## Custom Headers

If the MCP server expects a custom header or token scheme, you can manually set the client's `headers` instead of using the `auth` parameter by setting them on your transport:

```python {5} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async with Client(
    transport=StreamableHttpTransport(
        "https://your-server.fastmcp.app/mcp", 
        headers={"X-API-Key": "<your-token>"},
    ),
) as client:
    await client.ping()
```


# CIMD Authentication
Source: https://gofastmcp.com/clients/auth/cimd

Use Client ID Metadata Documents for verifiable, domain-based client identity.

<VersionBadge />

<Tip>
  CIMD authentication is only relevant for HTTP-based transports and requires a server that advertises CIMD support.
</Tip>

With standard OAuth, your client registers dynamically with every server it connects to, receiving a fresh `client_id` each time. This works, but the server has no way to verify *who* your client actually is — any client can claim any name during registration.

CIMD (Client ID Metadata Documents) flips this around. You host a small JSON document at an HTTPS URL you control, and that URL becomes your `client_id`. When your client connects to a server, the server fetches your metadata document and can verify your identity through your domain ownership. Users see a verified domain badge in the consent screen instead of an unverified client name.

## Client Usage

Pass your CIMD document URL to the `client_metadata_url` parameter of `OAuth`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.auth import OAuth

async with Client(
    "https://mcp-server.example.com/mcp",
    auth=OAuth(
        client_metadata_url="https://myapp.example.com/oauth/client.json",
    ),
) as client:
    await client.ping()
```

When the server supports CIMD, the client uses your metadata URL as its `client_id` instead of performing Dynamic Client Registration. The server fetches your document, validates it, and proceeds with the standard OAuth authorization flow.

<Note>
  You don't need to pass `mcp_url` when using `OAuth` with `Client(auth=...)` — the transport provides the server URL automatically.
</Note>

## Creating a CIMD Document

A CIMD document is a JSON file that describes your client. The most important field is `client_id`, which must exactly match the URL where you host the document.

Use the FastMCP CLI to generate one:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp auth cimd create \
    --name "My Application" \
    --redirect-uri "http://localhost:*/callback" \
    --client-id "https://myapp.example.com/oauth/client.json"
```

This produces:

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
{
  "client_id": "https://myapp.example.com/oauth/client.json",
  "client_name": "My Application",
  "redirect_uris": ["http://localhost:*/callback"],
  "token_endpoint_auth_method": "none",
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

If you omit `--client-id`, the CLI generates a placeholder value and reminds you to update it before hosting.

### CLI Options

The `create` command accepts these flags:

| Flag                   | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| `--name`               | Human-readable client name (required)                               |
| `--redirect-uri`, `-r` | Allowed redirect URIs — can be specified multiple times (required)  |
| `--client-id`          | The URL where you'll host this document (sets `client_id` directly) |
| `--output`, `-o`       | Write to a file instead of stdout                                   |
| `--scope`              | Space-separated list of scopes the client may request               |
| `--client-uri`         | URL of the client's home page                                       |
| `--logo-uri`           | URL of the client's logo image                                      |
| `--no-pretty`          | Output compact JSON                                                 |

### Redirect URIs

The `redirect_uris` field supports wildcard port matching for localhost. The pattern `http://localhost:*/callback` matches any port, which is useful for development clients that bind to random available ports (which is what FastMCP's `OAuth` helper does by default).

## Hosting Requirements

CIMD documents must be hosted at a publicly accessible HTTPS URL with a non-root path:

* **HTTPS required** — HTTP URLs are rejected for security
* **Non-root path** — The URL must have a path component (e.g., `/oauth/client.json`, not just `/`)
* **Public accessibility** — The server must be able to fetch the document over the internet
* **Matching `client_id`** — The `client_id` field in the document must exactly match the hosting URL

Common hosting options include static file hosting services like GitHub Pages, Cloudflare Pages, Vercel, or S3 — anywhere you can serve a JSON file over HTTPS.

## Validating Your Document

Before deploying, verify your hosted document passes validation:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp auth cimd validate https://myapp.example.com/oauth/client.json
```

The validator fetches the document and checks that:

* The URL is valid (HTTPS, non-root path)
* The document is well-formed JSON conforming to the CIMD schema
* The `client_id` in the document matches the URL it was fetched from

## How It Works

When your client connects to a CIMD-enabled server, the flow works like this:

<Steps>
  <Step title="Client Presents Metadata URL">
    Your client sends its `client_metadata_url` as the `client_id` in the OAuth authorization request.
  </Step>

  <Step title="Server Recognizes CIMD URL">
    The server sees that the `client_id` is an HTTPS URL with a path — the signature of a CIMD client — and skips Dynamic Client Registration.
  </Step>

  <Step title="Server Fetches and Validates">
    The server fetches your JSON document from the URL, validates that `client_id` matches the URL, and extracts your client metadata (name, redirect URIs, scopes).
  </Step>

  <Step title="Authorization Proceeds">
    The standard OAuth flow continues: browser opens for user consent, authorization code exchange, token issuance. The consent screen shows your verified domain.
  </Step>
</Steps>

The server caches your CIMD document according to HTTP cache headers, so subsequent requests don't require re-fetching.

## Server Configuration

CIMD is a server-side feature that your MCP server must support. FastMCP's OAuth proxy providers (GitHub, Google, Auth0, etc.) support CIMD by default. See the [OAuth Proxy CIMD documentation](/servers/auth/oauth-proxy#cimd-support) for server-side configuration, including private key JWT authentication and security details.


# OAuth Authentication
Source: https://gofastmcp.com/clients/auth/oauth

Authenticate your FastMCP client via OAuth 2.1.

<VersionBadge />

<Tip>
  OAuth authentication is only relevant for HTTP-based transports and requires user interaction via a web browser.
</Tip>

When your FastMCP client needs to access an MCP server protected by OAuth 2.1, and the process requires user interaction (like logging in and granting consent), you should use the Authorization Code Flow. FastMCP provides the `fastmcp.client.auth.OAuth` helper to simplify this entire process.

This flow is common for user-facing applications where the application acts on behalf of the user.

## Client Usage

### Default Configuration

The simplest way to use OAuth is to pass the string `"oauth"` to the `auth` parameter of the `Client` or transport instance. FastMCP will automatically configure the client to use OAuth with default settings:

```python {4} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

# Uses default OAuth settings
async with Client("https://your-server.fastmcp.app/mcp", auth="oauth") as client:
    await client.ping()
```

### `OAuth` Helper

To fully configure the OAuth flow, use the `OAuth` helper and pass it to the `auth` parameter of the `Client` or transport instance. `OAuth` manages the complexities of the OAuth 2.1 Authorization Code Grant with PKCE (Proof Key for Code Exchange) for enhanced security, and implements the full `httpx.Auth` interface.

```python {2, 4, 6} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.auth import OAuth

oauth = OAuth(scopes=["user"])

async with Client("https://your-server.fastmcp.app/mcp", auth=oauth) as client:
    await client.ping()
```

<Note>
  You don't need to pass `mcp_url` when using `OAuth` with `Client(auth=...)` — the transport provides the server URL automatically.
</Note>

#### `OAuth` Parameters

* **`scopes`** (`str | list[str]`, optional): OAuth scopes to request. Can be space-separated string or list of strings
* **`client_name`** (`str`, optional): Client name for dynamic registration. Defaults to `"FastMCP Client"`
* **`client_id`** (`str`, optional): Pre-registered OAuth client ID. When provided, skips Dynamic Client Registration entirely. See [Pre-Registered Clients](#pre-registered-clients)
* **`client_secret`** (`str`, optional): OAuth client secret for pre-registered clients. Optional — public clients that rely on PKCE can omit this
* **`client_metadata_url`** (`str`, optional): URL-based client identity (CIMD). See [CIMD Authentication](/clients/auth/cimd) for details
* **`token_storage`** (`AsyncKeyValue`, optional): Storage backend for persisting OAuth tokens. Defaults to in-memory storage (tokens lost on restart). See [Token Storage](#token-storage) for encrypted storage options
* **`additional_client_metadata`** (`dict[str, Any]`, optional): Extra metadata for client registration
* **`callback_port`** (`int`, optional): Fixed port for OAuth callback server. If not specified, uses a random available port
* **`httpx_client_factory`** (`McpHttpClientFactory`, optional): Factory for creating httpx clients

## OAuth Flow

The OAuth flow is triggered when you use a FastMCP `Client` configured to use OAuth.

<Steps>
  <Step title="Token Check">
    The client first checks the configured `token_storage` backend for existing, valid tokens for the target server. If one is found, it will be used to authenticate the client.
  </Step>

  <Step title="OAuth Server Discovery">
    If no valid tokens exist, the client attempts to discover the OAuth server's endpoints using a well-known URI (e.g., `/.well-known/oauth-authorization-server`) based on the `mcp_url`.
  </Step>

  <Step title="Client Registration">
    If a `client_id` is provided, the client uses those pre-registered credentials directly and skips this step entirely. Otherwise, if a `client_metadata_url` is configured and the server supports CIMD, the client uses its metadata URL as its identity. As a fallback, the client performs Dynamic Client Registration (RFC 7591) if the server supports it.
  </Step>

  <Step title="Local Callback Server">
    A temporary local HTTP server is started on an available port (or the port specified via `callback_port`). This server's address (e.g., `http://127.0.0.1:<port>/callback`) acts as the `redirect_uri` for the OAuth flow.
  </Step>

  <Step title="Browser Interaction">
    The user's default web browser is automatically opened, directing them to the OAuth server's authorization endpoint. The user logs in and grants (or denies) the requested `scopes`.
  </Step>

  <Step title="Authorization Code & Token Exchange">
    Upon approval, the OAuth server redirects the user's browser to the local callback server with an `authorization_code`. The client captures this code and exchanges it with the OAuth server's token endpoint for an `access_token` (and often a `refresh_token`) using PKCE for security.
  </Step>

  <Step title="Token Caching">
    The obtained tokens are saved to the configured `token_storage` backend for future use, eliminating the need for repeated browser interactions.
  </Step>

  <Step title="Authenticated Requests">
    The access token is automatically included in the `Authorization` header for requests to the MCP server.
  </Step>

  <Step title="Refresh Token">
    If the access token expires, the client will automatically use the refresh token to get a new access token.
  </Step>
</Steps>

## Token Storage

<VersionBadge />

By default, tokens are stored in memory and lost when your application restarts. For persistent storage, pass an `AsyncKeyValue`-compatible storage backend to the `token_storage` parameter.

<Warning>
  **Security Consideration**: Use encrypted storage for production. MCP clients can accumulate OAuth credentials for many servers over time, and a compromised token store could expose access to multiple services.
</Warning>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.auth import OAuth
from key_value.aio.stores.disk import DiskStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from cryptography.fernet import Fernet
import os

# Create encrypted disk storage
encrypted_storage = FernetEncryptionWrapper(
    key_value=DiskStore(directory="~/.fastmcp/oauth-tokens"),
    fernet=Fernet(os.environ["OAUTH_STORAGE_ENCRYPTION_KEY"])
)

oauth = OAuth(token_storage=encrypted_storage)

async with Client("https://your-server.fastmcp.app/mcp", auth=oauth) as client:
    await client.ping()
```

You can use any `AsyncKeyValue`-compatible backend from the [key-value library](https://github.com/strawgate/py-key-value) including Redis, DynamoDB, and more. Wrap your storage in `FernetEncryptionWrapper` for encryption.

<Note>
  When selecting a storage backend, review the [py-key-value documentation](https://github.com/strawgate/py-key-value) to understand the maturity level and limitations of your chosen backend. Some backends may be in preview or have constraints that affect production suitability.
</Note>

## CIMD Authentication

<VersionBadge />

Client ID Metadata Documents (CIMD) provide an alternative to Dynamic Client Registration. Instead of registering with each server, your client hosts a static JSON document at an HTTPS URL. That URL becomes your client's identity, and servers can verify who you are through your domain ownership.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.auth import OAuth

async with Client(
    "https://mcp-server.example.com/mcp",
    auth=OAuth(
        client_metadata_url="https://myapp.example.com/oauth/client.json",
    ),
) as client:
    await client.ping()
```

See the [CIMD Authentication](/clients/auth/cimd) page for complete documentation on creating, hosting, and validating CIMD documents.

## Pre-Registered Clients

<VersionBadge />

Some OAuth servers don't support Dynamic Client Registration — the MCP spec explicitly makes DCR optional. If your client has been pre-registered with the server (you already have a `client_id` and optionally a `client_secret`), you can provide them directly to skip DCR entirely.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.auth import OAuth

async with Client(
    "https://mcp-server.example.com/mcp",
    auth=OAuth(
        client_id="my-registered-client-id",
        client_secret="my-client-secret",
    ),
) as client:
    await client.ping()
```

Public clients that rely on PKCE for security can omit `client_secret`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
oauth = OAuth(client_id="my-public-client-id")
```

<Note>
  When using pre-registered credentials, the client will not attempt Dynamic Client Registration. If the server rejects the credentials, the error is surfaced immediately rather than falling back to DCR.
</Note>
