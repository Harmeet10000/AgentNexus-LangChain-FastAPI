# Project Setup and Quickstart

Source lines: 5087-6312 from the original FastMCP documentation dump.

Project configuration, installation, quickstart, intro material, and Anthropic/OpenAI style remote calling examples.

---

# Project Configuration
Source: https://gofastmcp.com/deployment/server-configuration

Use fastmcp.json for portable, declarative project configuration

<VersionBadge />

FastMCP supports declarative configuration through `fastmcp.json` files. This is the canonical and preferred way to configure FastMCP projects, providing a single source of truth for server settings, dependencies, and deployment options that replaces complex command-line arguments.

The `fastmcp.json` file is designed to be a portable description of your server configuration that can be shared across environments and teams. When running from a `fastmcp.json` file, you can override any configuration values using CLI arguments.

## Overview

The `fastmcp.json` configuration file allows you to define all aspects of your FastMCP server in a structured, shareable format. Instead of remembering command-line arguments or writing shell scripts, you declare your server's configuration once and use it everywhere.

When you have a `fastmcp.json` file, running your server becomes as simple as:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Run the server using the configuration
fastmcp run fastmcp.json

# Or if fastmcp.json exists in the current directory
fastmcp run
```

This configuration approach ensures reproducible deployments across different environments, from local development to production servers. It works seamlessly with Claude Desktop, VS Code extensions, and any MCP-compatible client.

## File Structure

The `fastmcp.json` configuration answers three fundamental questions about your server:

* **Source** = WHERE does your server code live?
* **Environment** = WHAT environment setup does it require?
* **Deployment** = HOW should the server run?

This conceptual model helps you understand the purpose of each configuration section and organize your settings effectively. The configuration file maps directly to these three concerns:

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
{
  "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
  "source": {
    // WHERE: Location of your server code
    "type": "filesystem",  // Optional, defaults to "filesystem"
    "path": "server.py",
    "entrypoint": "mcp"
  },
  "environment": {
    // WHAT: Environment setup and dependencies
    "type": "uv",  // Optional, defaults to "uv"
    "python": ">=3.10",
    "dependencies": ["pandas", "numpy"]
  },
  "deployment": {
    // HOW: Runtime configuration
    "transport": "stdio",
    "log_level": "INFO"
  }
}
```

Only the `source` field is required. The `environment` and `deployment` sections are optional and provide additional configuration when needed.

### JSON Schema Support

FastMCP provides JSON schemas for IDE autocomplete and validation. Add the schema reference to your `fastmcp.json` for enhanced developer experience:

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
{
  "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
  "source": {
    "path": "server.py",
    "entrypoint": "mcp"
  }
}
```

Two schema URLs are available:

* **Version-specific**: `https://gofastmcp.com/public/schemas/fastmcp.json/v1.json`
* **Latest version**: `https://gofastmcp.com/public/schemas/fastmcp.json/latest.json`

Modern IDEs like VS Code will automatically provide autocomplete suggestions, validation, and inline documentation when the schema is specified.

### Source Configuration

The source configuration determines **WHERE** your server code lives. It tells FastMCP how to find and load your server, whether it's a local Python file, a remote repository, or hosted in the cloud. This section is required and forms the foundation of your configuration.

<Card icon="code" title="Source">
  <ParamField type="object">
    The server source configuration that determines where your server code lives.

    <ParamField type="string">
      The source type identifier that determines which implementation to use. Currently supports `"filesystem"` for local files. Future releases will add support for `"git"` and `"cloud"` source types.
    </ParamField>

    <Expandable title="FileSystemSource">
      When `type` is `"filesystem"` (or omitted), the source points to a local Python file containing your FastMCP server:

      <ParamField type="string">
        Path to the Python file containing your FastMCP server.
      </ParamField>

      <ParamField type="string">
        Name of the server instance or factory function within the module:

        * Can be a FastMCP server instance (e.g., `mcp = FastMCP("MyServer")`)
        * Can be a function with no arguments that returns a FastMCP server
        * If not specified, FastMCP searches for common names: `mcp`, `server`, or `app`
      </ParamField>

      **Example:**

      ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
      "source": {
        "type": "filesystem",
        "path": "src/server.py",
        "entrypoint": "mcp"
      }
      ```

      Note: File paths are resolved relative to the configuration file's location.
    </Expandable>
  </ParamField>
</Card>

<Note>
  **Future Source Types**

  Future releases will support additional source types:

  * **Git repositories** (`type: "git"`) for loading server code directly from version control
  * **Prefect Horizon** (`type: "cloud"`) for hosted servers with automatic scaling and management
</Note>

### Environment Configuration

The environment configuration determines **WHAT** environment setup your server requires. It controls the build-time setup of your Python environment, ensuring your server runs with the exact Python version and dependencies it requires. This section creates isolated, reproducible environments across different systems.

FastMCP uses an extensible environment system with a base `Environment` class that can be implemented by different environment providers. Currently, FastMCP supports the `UVEnvironment` for Python environment management using `uv`'s powerful dependency resolver.

<Card icon="code" title="Environment">
  <ParamField type="object">
    Optional environment configuration. When specified, FastMCP uses the appropriate environment implementation to set up your server's runtime.

    <ParamField type="string">
      The environment type identifier that determines which implementation to use. Currently supports `"uv"` for Python environments managed by uv. If omitted, defaults to `"uv"`.
    </ParamField>

    <Expandable title="UVEnvironment">
      When `type` is `"uv"` (or omitted), the environment uses uv to manage Python dependencies:

      <ParamField type="string">
        Python version constraint. Examples:

        * Exact version: `"3.12"`
        * Minimum version: `">=3.10"`
        * Version range: `">=3.10,<3.13"`
      </ParamField>

      <ParamField type="list[str]">
        List of pip packages with optional version specifiers (PEP 508 format).

        ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
        "dependencies": ["pandas>=2.0", "requests", "httpx"]
        ```
      </ParamField>

      <ParamField type="string">
        Path to a requirements.txt file, resolved relative to the config file location.

        ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
        "requirements": "requirements.txt"
        ```
      </ParamField>

      <ParamField type="string">
        Path to a project directory containing pyproject.toml for uv project management.

        ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
        "project": "."
        ```
      </ParamField>

      <ParamField type="list[string]">
        List of paths to packages to install in editable/development mode. Useful for local development when you want changes to be reflected immediately. Supports multiple packages for monorepo setups or shared libraries.

        ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
        "editable": ["."]
        ```

        Or with multiple packages:

        ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
        "editable": [".", "../shared-lib", "/path/to/another-package"]
        ```
      </ParamField>

      **Example:**

      ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
      "environment": {
        "type": "uv",
        "python": ">=3.10",
        "dependencies": ["pandas", "numpy"],
        "editable": ["."]
      }
      ```

      Note: When any UVEnvironment field is specified, FastMCP automatically creates an isolated environment using `uv` before running your server.
    </Expandable>
  </ParamField>
</Card>

When environment configuration is provided, FastMCP:

1. Detects the environment type (defaults to `"uv"` if not specified)
2. Creates an isolated environment using the appropriate provider
3. Installs the specified dependencies
4. Runs your server in this clean environment

This build-time setup ensures your server always has the dependencies it needs, without polluting your system Python or conflicting with other projects.

<Note>
  **Future Environment Types**

  Similar to source types, future releases may support additional environment types for different runtime requirements, such as Docker containers or language-specific environments beyond Python.
</Note>

### Deployment Configuration

The deployment configuration controls **HOW** your server runs. It defines the runtime behavior including network settings, environment variables, and execution context. These settings determine how your server operates when it executes, from transport protocols to logging levels.

Environment variables are included in this section because they're runtime configuration that affects how your server behaves when it executes, not how its environment is built. The deployment configuration is applied every time your server starts, controlling its operational characteristics.

<Card icon="code" title="Deployment Fields">
  <ParamField type="object">
    Optional runtime configuration for the server.

    <Expandable title="Deployment Fields">
      <ParamField type="string">
        Protocol for client communication:

        * `"stdio"`: Standard input/output for desktop clients
        * `"http"`: Network-accessible HTTP server
        * `"sse"`: Server-sent events
      </ParamField>

      <ParamField type="string">
        Network interface to bind (HTTP transport only):

        * `"127.0.0.1"`: Local connections only
        * `"0.0.0.0"`: All network interfaces
      </ParamField>

      <ParamField type="integer">
        Port number for HTTP transport.
      </ParamField>

      <ParamField type="string">
        URL path for the MCP endpoint when using HTTP transport.
      </ParamField>

      <ParamField type="string">
        Server logging verbosity. Options:

        * `"DEBUG"`: Detailed debugging information
        * `"INFO"`: General informational messages
        * `"WARNING"`: Warning messages
        * `"ERROR"`: Error messages only
        * `"CRITICAL"`: Critical errors only
      </ParamField>

      <ParamField type="object">
        Environment variables to set when running the server. Supports `${VAR_NAME}` syntax for runtime interpolation.

        ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
        "env": {
          "API_KEY": "secret-key",
          "DATABASE_URL": "postgres://${DB_USER}@${DB_HOST}/mydb"
        }
        ```
      </ParamField>

      <ParamField type="string">
        Working directory for the server process. Relative paths are resolved from the config file location.
      </ParamField>

      <ParamField type="list[str]">
        Command-line arguments to pass to the server, passed after `--` to the server's argument parser.

        ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
        "args": ["--config", "server-config.json"]
        ```
      </ParamField>
    </Expandable>
  </ParamField>
</Card>

#### Environment Variable Interpolation

The `env` field in deployment configuration supports runtime interpolation of environment variables using `${VAR_NAME}` syntax. This enables dynamic configuration based on your deployment environment:

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
{
  "deployment": {
    "env": {
      "API_URL": "https://api.${ENVIRONMENT}.example.com",
      "DATABASE_URL": "postgres://${DB_USER}:${DB_PASS}@${DB_HOST}/myapp",
      "CACHE_KEY": "myapp_${ENVIRONMENT}_${VERSION}"
    }
  }
}
```

When the server starts, FastMCP replaces `${ENVIRONMENT}`, `${DB_USER}`, etc. with values from your system's environment variables. If a variable doesn't exist, the placeholder is preserved as-is.

**Example**: If your system has `ENVIRONMENT=production` and `DB_HOST=db.example.com`:

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
// Configuration
{
  "deployment": {
    "env": {
      "API_URL": "https://api.${ENVIRONMENT}.example.com",
      "DB_HOST": "${DB_HOST}"
    }
  }
}

// Result at runtime
{
  "API_URL": "https://api.production.example.com",
  "DB_HOST": "db.example.com"
}
```

This feature is particularly useful for:

* Deploying the same configuration across development, staging, and production
* Keeping sensitive values out of configuration files
* Building dynamic URLs and connection strings
* Creating environment-specific prefixes or suffixes

## Usage with CLI Commands

FastMCP automatically detects and uses a file specifically named `fastmcp.json` in the current directory, making server execution simple and consistent. Files with FastMCP configuration format but different names are not auto-detected and must be specified explicitly:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Auto-detect fastmcp.json in current directory
cd my-project
fastmcp run  # No arguments needed!

# Or specify a configuration file explicitly
fastmcp run prod.fastmcp.json

# Skip environment setup when already in a uv environment
fastmcp run fastmcp.json --skip-env

# Skip source preparation when source is already prepared
fastmcp run fastmcp.json --skip-source

# Skip both environment and source preparation
fastmcp run fastmcp.json --skip-env --skip-source
```

### Pre-building Environments

You can use `fastmcp project prepare` to create a persistent uv project with all dependencies pre-installed:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Create a persistent environment
fastmcp project prepare fastmcp.json --output-dir ./env

# Use the pre-built environment to run the server
fastmcp run fastmcp.json --project ./env
```

This pattern separates environment setup (slow) from server execution (fast), useful for deployment scenarios.

### Using an Existing Environment

By default, FastMCP creates an isolated environment with `uv` based on your configuration. When you already have a suitable Python environment, use the `--skip-env` flag to skip environment creation:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run fastmcp.json --skip-env
```

**When you already have an environment:**

* You're in an activated virtual environment with all dependencies installed
* You're inside a Docker container with pre-installed dependencies
* You're in a CI/CD pipeline that pre-builds the environment
* You're using a system-wide installation with all required packages
* You're in a uv-managed environment (prevents infinite recursion)

This flag tells FastMCP: "I already have everything installed, just run the server."

### Using an Existing Source

When working with source types that require preparation (future support for git repositories or cloud sources), use the `--skip-source` flag when you already have the source code available:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run fastmcp.json --skip-source
```

**When you already have the source:**

* You've previously cloned a git repository and don't need to re-fetch
* You have a cached copy of a cloud-hosted server
* You're in a CI/CD pipeline where source checkout is a separate step
* You're iterating locally on already-downloaded code

This flag tells FastMCP: "I already have the source code, skip any download/clone steps."

Note: For filesystem sources (local Python files), this flag has no effect since they don't require preparation.

The configuration file works with all FastMCP commands:

* **`run`** - Start the server in production mode
* **`dev`** - Launch with the Inspector UI for development
* **`inspect`** - View server capabilities and configuration
* **`install`** - Install to Claude Desktop, Cursor, or other MCP clients

When no file argument is provided, FastMCP searches the current directory for `fastmcp.json`. This means you can simply navigate to your project directory and run `fastmcp run` to start your server with all its configured settings.

### CLI Override Behavior

Command-line arguments take precedence over configuration file values, allowing ad-hoc adjustments without modifying the file:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Config specifies port 3000, CLI overrides to 8080
fastmcp run fastmcp.json --port 8080

# Config specifies stdio, CLI overrides to HTTP
fastmcp run fastmcp.json --transport http

# Add extra dependencies not in config
fastmcp run fastmcp.json --with requests --with httpx
```

This precedence order enables:

* Quick testing of different settings
* Environment-specific overrides in deployment scripts
* Debugging with increased log levels
* Temporary configuration changes

### Custom Naming Patterns

You can use different configuration files for different environments:

* `fastmcp.json` - Default configuration
* `dev.fastmcp.json` - Development settings
* `prod.fastmcp.json` - Production settings
* `test_fastmcp.json` - Test configuration

Any file with "fastmcp.json" in the name is recognized as a configuration file.

## Examples

<Tabs>
  <Tab title="Basic Configuration">
    A minimal configuration for a simple server:

    ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
    {
      "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
      "source": {
        "path": "server.py",
        "entrypoint": "mcp"
      }
    }
    ```

    This configuration explicitly specifies the server entrypoint (`mcp`), making it clear which server instance or factory function to use. Uses all defaults: STDIO transport, no special dependencies, standard logging.
  </Tab>

  <Tab title="Development Configuration">
    A configuration optimized for local development:

    ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
    {
      "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
      // WHERE does the server live?
      "source": {
        "path": "src/server.py",
        "entrypoint": "app"
      },
      // WHAT dependencies does it need?
      "environment": {
        "type": "uv",
        "python": "3.12",
        "dependencies": ["fastmcp[dev]"],
        "editable": "."
      },
      // HOW should it run?
      "deployment": {
        "transport": "http",
        "host": "127.0.0.1",
        "port": 8000,
        "log_level": "DEBUG",
        "env": {
          "DEBUG": "true",
          "ENV": "development"
        }
      }
    }
    ```
  </Tab>

  <Tab title="Production Configuration">
    A production-ready configuration with full dependency management:

    ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
    {
      "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
      // WHERE does the server live?
      "source": {
        "path": "app/main.py",
        "entrypoint": "mcp_server"
      },
      // WHAT dependencies does it need?
      "environment": {
        "python": "3.11",
        "requirements": "requirements/production.txt",
        "project": "."
      },
      // HOW should it run?
      "deployment": {
        "transport": "http",
        "host": "0.0.0.0",
        "port": 3000,
        "path": "/api/mcp/",
        "log_level": "INFO",
        "env": {
          "ENV": "production",
          "API_BASE_URL": "https://api.example.com",
          "DATABASE_URL": "postgresql://user:pass@db.example.com/prod"
        },
        "cwd": "/app",
        "args": ["--workers", "4"]
      }
    }
    ```
  </Tab>

  <Tab title="Data Science Server">
    Configuration for a data analysis server with scientific packages:

    ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
    {
      "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
      "source": {
        "path": "analysis_server.py",
        "entrypoint": "mcp"
      },
      "environment": {
        "python": "3.11",
        "dependencies": [
          "pandas>=2.0",
          "numpy",
          "scikit-learn",
          "matplotlib",
          "jupyterlab"
        ]
      },
      "deployment": {
        "transport": "stdio",
        "env": {
          "MATPLOTLIB_BACKEND": "Agg",
          "DATA_PATH": "./datasets"
        }
      }
    }
    ```
  </Tab>

  <Tab title="Multi-Environment Setup">
    You can maintain multiple configuration files for different environments:

    **dev.fastmcp.json**:

    ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
    {
      "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
      "source": {
        "path": "server.py",
        "entrypoint": "mcp"
      },
      "deployment": {
        "transport": "http",
        "log_level": "DEBUG"
      }
    }
    ```

    **prod.fastmcp.json**:

    ```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
    {
      "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
      "source": {
        "path": "server.py",
        "entrypoint": "mcp"
      },
      "environment": {
        "requirements": "requirements/production.txt"
      },
      "deployment": {
        "transport": "http",
        "host": "0.0.0.0",
        "log_level": "WARNING"
      }
    }
    ```

    Run different configurations:

    ```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
    fastmcp run dev.fastmcp.json   # Development
    fastmcp run prod.fastmcp.json  # Production
    ```
  </Tab>
</Tabs>

## Migrating from CLI Arguments

If you're currently using command-line arguments or shell scripts, migrating to `fastmcp.json` simplifies your workflow. Here's how common CLI patterns map to configuration:

**CLI Command**:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
uv run --with pandas --with requests \
  fastmcp run server.py \
  --transport http \
  --port 8000 \
  --log-level INFO
```

**Equivalent fastmcp.json**:

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
{
  "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
  "source": {
    "path": "server.py",
    "entrypoint": "mcp"
  },
  "environment": {
    "dependencies": ["pandas", "requests"]
  },
  "deployment": {
    "transport": "http",
    "port": 8000,
    "log_level": "INFO"
  }
}
```

Now simply run:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run  # Automatically finds and uses fastmcp.json
```

The configuration file approach provides better documentation, easier sharing, and consistent execution across different environments while maintaining the flexibility to override settings when needed.


# Installation
Source: https://gofastmcp.com/getting-started/installation

Install FastMCP and verify your setup

## Install FastMCP

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to install and manage FastMCP.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install fastmcp
```

Or with uv:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
uv add fastmcp
```

### Optional Dependencies

FastMCP provides optional extras for specific features. For example, to install the background tasks extra:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install "fastmcp[tasks]"
```

See [Background Tasks](/servers/tasks) for details on the task system.

### Verify Installation

To verify that FastMCP is installed correctly, you can run the following command:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp version
```

You should see output like the following:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
$ fastmcp version

FastMCP version:                           3.0.0
MCP version:                               1.25.0
Python version:                            3.12.2
Platform:            macOS-15.3.1-arm64-arm-64bit
FastMCP root path:            ~/Developer/fastmcp
```

### Dependency Licensing

<Info>
  FastMCP depends on Cyclopts for CLI functionality. Cyclopts v4 includes docutils as a transitive dependency, which has complex licensing that may trigger compliance reviews in some organizations.

  If this is a concern, you can install Cyclopts v5 alpha which removes this dependency:

  ```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  pip install "cyclopts>=5.0.0a1"
  ```

  Alternatively, wait for the stable v5 release. See [this issue](https://github.com/BrianPugh/cyclopts/issues/672) for details.
</Info>

## Upgrading

### From FastMCP 2.0

See the [Upgrade Guide](/getting-started/upgrading/from-fastmcp-2) for a complete list of breaking changes and migration steps.

### From the MCP SDK

#### From FastMCP 1.0

If you're using FastMCP 1.0 via the `mcp` package (meaning you import FastMCP as  `from mcp.server.fastmcp import FastMCP`), upgrading is straightforward — for most servers, it's a single import change. See the [full upgrade guide](/getting-started/upgrading/from-mcp-sdk) for details.

#### From the Low-Level Server API

If you built your server directly on the `mcp` package's `Server` class — with `list_tools()`/`call_tool()` handlers and hand-written JSON Schema — see the [migration guide](/getting-started/upgrading/from-low-level-sdk) for a full walkthrough.

## Versioning Policy

FastMCP follows semantic versioning with pragmatic adaptations for the rapidly evolving MCP ecosystem. Breaking changes may occur in minor versions (e.g., 2.3.x to 2.4.0) when necessary to stay current with the MCP Protocol.

For production use, always pin to exact versions:

```
fastmcp==3.0.0  # Good
fastmcp>=3.0.0  # Bad - may install breaking changes
```

See the full [versioning and release policy](/development/releases#versioning-policy) for details on our public API, deprecation practices, and breaking change philosophy.

## Contributing to FastMCP

Interested in contributing to FastMCP? See the [Contributing Guide](/development/contributing) for details on:

* Setting up your development environment
* Running tests and pre-commit hooks
* Submitting issues and pull requests
* Code standards and review process


# Quickstart
Source: https://gofastmcp.com/getting-started/quickstart



Welcome! This guide will help you quickly set up FastMCP, run your first MCP server, and deploy a server to Prefect Horizon.

If you haven't already installed FastMCP, follow the [installation instructions](/getting-started/installation).

## Create a FastMCP Server

A FastMCP server is a collection of tools, resources, and other MCP components. To create a server, start by instantiating the `FastMCP` class.

Create a new file called `my_server.py` and add the following code:

```python my_server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")
```

That's it! You've created a FastMCP server, albeit a very boring one. Let's add a tool to make it more interesting.

## Add a Tool

To add a tool that returns a simple greeting, write a function and decorate it with `@mcp.tool` to register it with the server:

```python my_server.py {5-7} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

## Run the Server

The simplest way to run your FastMCP server is to call its `run()` method. You can choose between different transports, like `stdio` for local servers, or `http` for remote access:

<CodeGroup>
  ```python my_server.py (stdio) {9, 10} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  from fastmcp import FastMCP

  mcp = FastMCP("My MCP Server")

  @mcp.tool
  def greet(name: str) -> str:
      return f"Hello, {name}!"

  if __name__ == "__main__":
      mcp.run()
  ```

  ```python my_server.py (HTTP) {9, 10} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  from fastmcp import FastMCP

  mcp = FastMCP("My MCP Server")

  @mcp.tool
  def greet(name: str) -> str:
      return f"Hello, {name}!"

  if __name__ == "__main__":
      mcp.run(transport="http", port=8000)
  ```
</CodeGroup>

This lets us run the server with `python my_server.py`. The stdio transport is the traditional way to connect MCP servers to clients, while the HTTP transport enables remote connections.

<Tip>
  Why do we need the `if __name__ == "__main__":` block?

  The `__main__` block is recommended for consistency and compatibility, ensuring your server works with all MCP clients that execute your server file as a script. Users who will exclusively run their server with the FastMCP CLI can omit it, as the CLI imports the server object directly.
</Tip>

### Using the FastMCP CLI

You can also use the `fastmcp run` command to start your server. Note that the FastMCP CLI **does not** execute the `__main__` block of your server file. Instead, it imports your server object and runs it with whatever transport and options you provide.

For example, to run this server with the default stdio transport (no matter how you called `mcp.run()`), you can use the following command:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run my_server.py:mcp
```

To run this server with the HTTP transport, you can use the following command:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp run my_server.py:mcp --transport http --port 8000
```

## Call Your Server

Once your server is running with HTTP transport, you can connect to it with a FastMCP client or any LLM client that supports the MCP protocol:

```python my_client.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import asyncio
from fastmcp import Client

client = Client("http://localhost:8000/mcp")

async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)

asyncio.run(call_tool("Ford"))
```

Note that:

* FastMCP clients are asynchronous, so we need to use `asyncio.run` to run the client
* We must enter a client context (`async with client:`) before using the client
* You can make multiple client calls within the same context

## Deploy to Prefect Horizon

[Prefect Horizon](https://horizon.prefect.io) is the enterprise MCP platform built by the FastMCP team at [Prefect](https://www.prefect.io). It provides managed hosting, authentication, access control, and observability for MCP servers.

<Info>
  Horizon is **free for personal projects** and offers enterprise governance for teams.
</Info>

To deploy your server, you'll need a [GitHub account](https://github.com). Once you have one, you can deploy your server in three steps:

1. Push your `my_server.py` file to a GitHub repository
2. Sign in to [Prefect Horizon](https://horizon.prefect.io) with your GitHub account
3. Create a new project from your repository and enter `my_server.py:mcp` as the server entrypoint

That's it! Horizon will build and deploy your server, making it available at a URL like `https://your-project.fastmcp.app/mcp`. You can chat with it to test its functionality, or connect to it from any LLM client that supports the MCP protocol.

For more details, see the [Prefect Horizon guide](/deployment/prefect-horizon).


# Welcome to FastMCP
Source: https://gofastmcp.com/getting-started/welcome

The fast, Pythonic way to build MCP servers, clients, and applications.

<video />

<video />

**FastMCP is the standard framework for building MCP applications.** The [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) connects LLMs to tools and data. FastMCP gives you everything you need to go from prototype to production — build servers that expose capabilities, connect clients to any MCP service, and give your tools interactive UIs:

```python {1} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()
```

## Move Fast and Make Things

The [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) lets you give agents access to your tools and data. But building an effective MCP application is harder than it looks.

FastMCP handles all of it. Declare a tool with a Python function, and the schema, validation, and documentation are generated automatically. Connect to a server with a URL, and transport negotiation, authentication, and protocol lifecycle are managed for you. You focus on your logic, and the MCP part just works: **with FastMCP, best practices are built in.**

**That's why FastMCP is the standard framework for working with MCP.** FastMCP 1.0 was incorporated into the official MCP Python SDK in 2024. Today, the actively maintained standalone project is downloaded a million times a day, and some version of FastMCP powers 70% of MCP servers across all languages.

FastMCP has three pillars:

<CardGroup>
  <Card title="Servers" href="/servers/server">
    Expose tools, resources, and prompts to LLMs.
  </Card>

  <Card title="Apps" href="/apps/overview">
    Give your tools interactive UIs rendered directly in the conversation.
  </Card>

  <Card title="Clients" href="/clients/client">
    Connect to any MCP server — local or remote, programmatic or CLI.
  </Card>
</CardGroup>

**[Servers](/servers/server)** wrap your Python functions into MCP-compliant tools, resources, and prompts. **[Clients](/clients/client)** connect to any server with full protocol support. And **[Apps](/apps/overview)** give your tools interactive UIs rendered directly in the conversation.

Ready to build? Start with the [installation guide](/getting-started/installation) or jump straight to the [quickstart](/getting-started/quickstart). When you're ready to deploy, [Prefect Horizon](https://www.prefect.io/horizon) offers free hosting for FastMCP users.

FastMCP is made with 💙 by [Prefect](https://www.prefect.io/).

<Tip>
  **This documentation reflects FastMCP's `main` branch**, meaning it always reflects the latest development version. Features are generally marked with version badges (e.g. `New in version: 3.0.0`) to indicate when they were introduced. Note that this may include features that are not yet released.
</Tip>

## LLM-Friendly Docs

The FastMCP documentation is available in multiple LLM-friendly formats:

### MCP Server

The FastMCP docs are accessible via MCP! The server URL is `https://gofastmcp.com/mcp`.

In fact, you can use FastMCP to search the FastMCP docs:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import asyncio
from fastmcp import Client

async def main():
    async with Client("https://gofastmcp.com/mcp") as client:
        result = await client.call_tool(
            name="SearchFastMcp",
            arguments={"query": "deploy a FastMCP server"}
        )
    print(result)

asyncio.run(main())
```

### Text Formats

The docs are also available in [llms.txt format](https://llmstxt.org/):

* [llms.txt](https://gofastmcp.com/llms.txt) - A sitemap listing all documentation pages
* [llms-full.txt](https://gofastmcp.com/llms-full.txt) - The entire documentation in one file (may exceed context windows)

Any page can be accessed as markdown by appending `.md` to the URL. For example, this page becomes `https://gofastmcp.com/getting-started/welcome.md`.

You can also copy any page as markdown by pressing "Cmd+C" (or "Ctrl+C" on Windows) on your keyboard.


# Anthropic API 🤝 FastMCP
Source: https://gofastmcp.com/integrations/anthropic

Connect FastMCP servers to the Anthropic API

Anthropic's [Messages API](https://docs.anthropic.com/en/api/messages) supports MCP servers as remote tool sources. This tutorial will show you how to create a FastMCP server and deploy it to a public URL, then how to call it from the Messages API.

<Tip>
  Currently, the MCP connector only accesses **tools** from MCP servers—it queries the `list_tools` endpoint and exposes those functions to Claude. Other MCP features like resources and prompts are not currently supported. You can read more about the MCP connector in the [Anthropic documentation](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector).
</Tip>

## Create a Server

First, create a FastMCP server with the tools you want to expose. For this example, we'll create a server with a single tool that rolls dice.

```python server.py theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import random
from fastmcp import FastMCP

mcp = FastMCP(name="Dice Roller")

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

## Deploy the Server

Your server must be deployed to a public URL in order for Anthropic to access it. The MCP connector supports both SSE and Streamable HTTP transports.

For development, you can use tools like `ngrok` to temporarily expose a locally-running server to the internet. We'll do that for this example (you may need to install `ngrok` and create a free account), but you can use any other method to deploy your server.

Assuming you saved the above code as `server.py`, you can run the following two commands in two separate terminals to deploy your server and expose it to the internet:

<CodeGroup>
  ```bash FastMCP server theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  python server.py
  ```

  ```bash ngrok theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  ngrok http 8000
  ```
</CodeGroup>

<Warning>
  This exposes your unauthenticated server to the internet. Only run this command in a safe environment if you understand the risks.
</Warning>

## Call the Server

To use the Messages API with MCP servers, you'll need to install the Anthropic Python SDK (not included with FastMCP):

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install anthropic
```

You'll also need to authenticate with Anthropic. You can do this by setting the `ANTHROPIC_API_KEY` environment variable. Consult the Anthropic SDK documentation for more information.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
export ANTHROPIC_API_KEY="your-api-key"
```

Here is an example of how to call your server from Python. Note that you'll need to replace `https://your-server-url.com` with the actual URL of your server. In addition, we use `/mcp/` as the endpoint because we deployed a streamable-HTTP server with the default path; you may need to use a different endpoint if you customized your server's deployment. **At this time you must also include the `extra_headers` parameter with the `anthropic-beta` header.**

```python {5, 13-22} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import anthropic
from rich import print

# Your server URL (replace with your actual URL)
url = 'https://your-server-url.com'

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Roll a few dice!"}],
    mcp_servers=[
        {
            "type": "url",
            "url": f"{url}/mcp/",
            "name": "dice-server",
        }
    ],
    extra_headers={
        "anthropic-beta": "mcp-client-2025-04-04"
    }
)

print(response.content)
```

If you run this code, you'll see something like the following output:

```text theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
I'll roll some dice for you! Let me use the dice rolling tool.

I rolled 3 dice and got: 4, 2, 6

The results were 4, 2, and 6. Would you like me to roll again or roll a different number of dice?
```

## Authentication

<VersionBadge />

The MCP connector supports OAuth authentication through authorization tokens, which means you can secure your server while still allowing Anthropic to access it.

### Server Authentication

The simplest way to add authentication to the server is to use a bearer token scheme.

For this example, we'll quickly generate our own tokens with FastMCP's `RSAKeyPair` utility, but this may not be appropriate for production use. For more details, see the complete server-side [Token Verification](/servers/auth/token-verification) documentation.

We'll start by creating an RSA key pair to sign and verify tokens.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.auth.providers.jwt import RSAKeyPair

key_pair = RSAKeyPair.generate()
access_token = key_pair.create_token(audience="dice-server")
```

<Warning>
  FastMCP's `RSAKeyPair` utility is for development and testing only.
</Warning>

Next, we'll create a `JWTVerifier` to authenticate the server.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth import JWTVerifier

auth = JWTVerifier(
    public_key=key_pair.public_key,
    audience="dice-server",
)

mcp = FastMCP(name="Dice Roller", auth=auth)
```

Here is a complete example that you can copy/paste. For simplicity and the purposes of this example only, it will print the token to the console. **Do NOT do this in production!**

```python server.py [expandable] theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.auth import JWTVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair
import random

key_pair = RSAKeyPair.generate()
access_token = key_pair.create_token(audience="dice-server")

auth = JWTVerifier(
    public_key=key_pair.public_key,
    audience="dice-server",
)

mcp = FastMCP(name="Dice Roller", auth=auth)

@mcp.tool
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

if __name__ == "__main__":
    print(f"\n---\n\n🔑 Dice Roller access token:\n\n{access_token}\n\n---\n")
    mcp.run(transport="http", port=8000)
```

### Client Authentication

If you try to call the authenticated server with the same Anthropic code we wrote earlier, you'll get an error indicating that the server rejected the request because it's not authenticated.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
Error code: 400 - {
    "type": "error", 
    "error": {
        "type": "invalid_request_error", 
        "message": "MCP server 'dice-server' requires authentication. Please provide an authorization_token.",
    },
}
```

To authenticate the client, you can pass the token using the `authorization_token` parameter in your MCP server configuration:

```python {8, 21} theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import anthropic
from rich import print

# Your server URL (replace with your actual URL)
url = 'https://your-server-url.com'

# Your access token (replace with your actual token)
access_token = 'your-access-token'

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Roll a few dice!"}],
    mcp_servers=[
        {
            "type": "url",
            "url": f"{url}/mcp/",
            "name": "dice-server",
            "authorization_token": access_token
        }
    ],
    extra_headers={
        "anthropic-beta": "mcp-client-2025-04-04"
    }
)

print(response.content)
```

You should now see the dice roll results in the output.
