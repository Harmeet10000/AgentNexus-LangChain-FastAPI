# Contributing and Tests

Source lines: 28336-29117 from the original FastMCP documentation dump.

Contribution workflow, release process, and test guidance.

---

# Generate CLI
Source: https://gofastmcp.com/cli/generate-cli

Scaffold a standalone typed CLI from any MCP server

<VersionBadge />

`fastmcp list` and `fastmcp call` are general-purpose — you always specify the server, the tool name, and the arguments from scratch. `fastmcp generate-cli` goes further: it connects to a server, reads its tool schemas, and writes a standalone Python script where every tool is a proper subcommand with typed flags, help text, and tab completion. The result is a CLI that feels hand-written for that specific server.

MCP tool schemas already contain everything a CLI framework needs — parameter names, types, descriptions, required/optional status, and defaults. `generate-cli` maps that into [cyclopts](https://cyclopts.readthedocs.io/) commands, so JSON Schema types become Python type annotations, descriptions become `--help` text, and required parameters become mandatory flags.

## Generating a Script

Point the command at any [server target](/cli/overview#server-targets) and it writes a CLI script:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp generate-cli weather
fastmcp generate-cli http://localhost:8000/mcp
fastmcp generate-cli server.py my_weather_cli.py
```

The second positional argument sets the output path (defaults to `cli.py`). If the file already exists, pass `-f` to overwrite:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp generate-cli weather -f
```

## What You Get

The generated script is a regular Python file — executable, editable, and yours:

```
$ python cli.py call-tool --help
Usage: weather-cli call-tool COMMAND

Call a tool on the server

Commands:
  get_forecast  Get the weather forecast for a city.
  search_city   Search for a city by name.
```

Each tool has typed parameters with help text pulled directly from the server's schema:

```
$ python cli.py call-tool get_forecast --help
Usage: weather-cli call-tool get_forecast [OPTIONS]

Get the weather forecast for a city.

Options:
  --city    [str]  City name (required)
  --days    [int]  Number of forecast days (default: 3)
```

Beyond tool commands, the script includes generic MCP operations — `list-tools`, `list-resources`, `read-resource`, `list-prompts`, and `get-prompt` — that always reflect the server's current state, even if tools have changed since generation.

## Parameter Handling

Parameters are mapped based on their JSON Schema type:

**Simple types** (`string`, `integer`, `number`, `boolean`) become typed flags:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
python cli.py call-tool get_forecast --city London --days 3
```

**Arrays of simple types** become repeatable flags:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
python cli.py call-tool tag_items --tags python --tags fastapi --tags mcp
```

**Complex types** (objects, nested arrays, unions) accept JSON strings. The `--help` output shows the full schema so you know what structure to pass:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
python cli.py call-tool create_user \
  --name John \
  --metadata '{"role": "admin", "dept": "engineering"}'
```

## Agent Skill

Alongside the CLI script, `generate-cli` writes a `SKILL.md` file — a [Claude Code agent skill](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/skills) that documents every tool's exact invocation syntax, parameter flags, types, and descriptions. An agent can pick up the CLI immediately without running `--help` or experimenting with flag names.

To skip skill generation:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
fastmcp generate-cli weather --no-skill
```

## How It Works

The generated script is a *client*, not a server — it connects to the server on every invocation rather than bundling it. A `CLIENT_SPEC` variable at the top holds the resolved transport (a URL string or `StdioTransport` with baked-in command and arguments).

The most common edit is changing `CLIENT_SPEC` — for example, pointing a script generated from a dev server at production. Beyond that, the helper functions (`_call_tool`, `_print_tool_result`) are thin wrappers around `fastmcp.Client` that are easy to adapt.

The script requires `fastmcp` as a dependency. If it lives outside a project that already has FastMCP installed:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
uv run --with fastmcp python cli.py call-tool get_forecast --city London
```


# Contributing
Source: https://gofastmcp.com/development/contributing

Development workflow for FastMCP contributors

Contributing to FastMCP means joining a community that values clean, maintainable code and thoughtful API design. All contributions are valued - from fixing typos in documentation to implementing major features.

## Design Principles

Every contribution should advance these principles:

* 🚀 **Fast** — High-level interfaces mean less code and faster development
* 🍀 **Simple** — Minimal boilerplate; the obvious way should be the right way
* 🐍 **Pythonic** — Feels natural to Python developers; no surprising patterns
* 🔍 **Complete** — Everything needed for production: auth, testing, deployment, observability

PRs are evaluated against these principles. Code that makes FastMCP slower, harder to reason about, less Pythonic, or less complete will be rejected.

## Issues

### Issue First, Code Second

**Every pull request requires a corresponding issue - no exceptions.** This requirement creates a collaborative space where approach, scope, and alignment are established before code is written. Issues serve as design documents where maintainers and contributors discuss implementation strategy, identify potential conflicts with existing patterns, and ensure proposed changes advance FastMCP's vision.

**FastMCP is an opinionated framework, not a kitchen sink.** The maintainers have strong beliefs about what FastMCP should and shouldn't do. Just because something takes N lines of code and you want it in fewer lines doesn't mean FastMCP should take on the maintenance burden or endorse that pattern. This is judged at the maintainers' discretion.

Use issues to understand scope BEFORE opening PRs. The issue discussion determines whether a feature belongs in core, contrib, or not at all.

### Writing Good Issues

FastMCP is an extremely highly-trafficked repository maintained by a very small team. Issues that appear to transfer burden to maintainers without any effort to validate the problem will be closed. Please help the maintainers help you by always providing a minimal reproducible example and clearly describing the problem.

**LLM-generated issues will be closed immediately.** Issues that contain paragraphs of unnecessary explanation, verbose problem descriptions, or obvious LLM authorship patterns obfuscate the actual problem and transfer burden to maintainers.

Write clear, concise issues that:

* State the problem directly
* Provide a minimal reproducible example
* Skip unnecessary background or context
* Take responsibility for clear communication

Issues may be labeled "Invalid" simply due to confusion caused by verbosity or not adhering to the guidelines outlined here.

## Pull Requests

PRs that deviate from FastMCP's core principles will be rejected regardless of implementation quality. **PRs are NOT for iterating on ideas** - they should only be opened for ideas that already have a bias toward acceptance based on issue discussion.

### Development Environment

#### Installation

To contribute to FastMCP, you'll need to set up a development environment with all necessary tools and dependencies.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Clone the repository
git clone https://github.com/PrefectHQ/fastmcp.git
cd fastmcp

# Install all dependencies including dev tools
uv sync

# Install prek hooks
uv run prek install
```

In addition, some development commands require [just](https://github.com/casey/just) to be installed.

Prek hooks will run automatically on every commit to catch issues before they reach CI. If you see failures, fix them before committing - never commit broken code expecting to fix it later.

### Development Standards

#### Scope

Large pull requests create review bottlenecks and quality risks. Unless you're fixing a discrete bug or making an incredibly well-scoped change, keep PRs small and focused.

A PR that changes 50 lines across 3 files can be thoroughly reviewed in minutes. A PR that changes 500 lines across 20 files requires hours of careful analysis and often hides subtle issues.

Breaking large features into smaller PRs:

* Creates better review experiences
* Makes git history clear
* Simplifies debugging with bisect
* Reduces merge conflicts
* Gets your code merged faster

#### Code Quality

FastMCP values clarity over cleverness. Every line you write will be maintained by someone else - possibly years from now, possibly without context about your decisions.

**PRs can be rejected for two opposing reasons:**

1. **Insufficient quality** - Code that doesn't meet our standards for clarity, maintainability, or idiomaticity
2. **Overengineering** - Code that is overbearing, unnecessarily complex, or tries to be too clever

The focus is on idiomatic, high-quality Python. FastMCP uses patterns like `NotSet` type as an alternative to `None` in certain situations - follow existing patterns.

#### Required Practices

**Full type annotations** on all functions and methods. They catch bugs before runtime and serve as inline documentation.

**Async/await patterns** for all I/O operations. Even if your specific use case doesn't need concurrency, consistency means users can compose features without worrying about blocking operations.

**Descriptive names** make code self-documenting. `auth_token` is clear; `tok` requires mental translation.

**Specific exception types** make error handling predictable. Catching `ValueError` tells readers exactly what error you expect. Never use bare `except` clauses.

#### Anti-Patterns to Avoid

**Complex one-liners** are hard to debug and modify. Break operations into clear steps.

**Mutable default arguments** cause subtle bugs. Use `None` as the default and create the mutable object inside the function.

**Breaking established patterns** confuses readers. If you must deviate, discuss in the issue first.

### Prek Checks

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Runs automatically on commit, or manually:
uv run prek run --all-files
```

This runs three critical tools:

* **Ruff**: Linting and formatting
* **Prettier**: Code formatting
* **ty**: Static type checking

Pytest runs separately as a distinct workflow step after prek checks pass. CI will reject PRs that fail these checks. Always run them locally first.

### Testing

Tests are documentation that shows how features work. Good tests give reviewers confidence and help future maintainers understand intent.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Run specific test directory
uv run pytest tests/server/ -v

# Run all tests before submitting PR
uv run pytest
```

Every new feature needs tests. See the [Testing Guide](/development/tests) for patterns and requirements.

### Documentation

A feature doesn't exist unless it's documented. Note that FastMCP's hosted documentation always tracks the main branch - users who want historical documentation can clone the repo, checkout a specific tag, and host it themselves.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Preview documentation locally
just docs
```

Documentation requirements:

* **Explain concepts in prose first** - Code without context is just syntax
* **Complete, runnable examples** - Every code block should be copy-pasteable
* **Register in docs.json** - Makes pages appear in navigation
* **Version badges** - Mark when features were added using `<VersionBadge />`

#### SDK Documentation

FastMCP's SDK documentation is auto-generated from the source code docstrings and type annotations. It is automatically updated on every merge to main by a GitHub Actions workflow, so users are *not* responsible for keeping the documentation up to date. However, to generate it proactively, you can use the following command:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
just api-ref-all
```

### Submitting Your PR

#### Before Submitting

1. **Run all checks**: `uv run prek run --all-files && uv run pytest`
2. **Keep scope small**: One feature or fix per PR
3. **Write clear description**: Your PR description becomes permanent documentation
4. **Update docs**: Include documentation for API changes

#### PR Description

Write PR descriptions that explain:

* What problem you're solving
* Why you chose this approach
* Any trade-offs or alternatives considered
* Migration path for breaking changes

Focus on the "why" - the code shows the "what". Keep it concise but complete.

#### What We Look For

**Framework Philosophy**: FastMCP is NOT trying to do all things or provide all shortcuts. Features are rejected when they don't align with the framework's vision, even if perfectly implemented. The burden of proof is on the PR to demonstrate value.

**Code Quality**: We verify code follows existing patterns. Consistency reduces cognitive load. When every module works similarly, developers understand new code quickly.

**Test Coverage**: Not every line needs testing, but every behavior does. Tests document intent and protect against regressions.

**Breaking Changes**: May be acceptable in minor versions but must be clearly documented. See the [versioning policy](/development/releases#versioning-policy).

## Special Modules

**`contrib`**: Community-maintained patterns and utilities. Original authors maintain their contributions. Not representative of the core framework.

**`experimental`**: Maintainer-developed features that may preview future functionality. Can break or be deleted at any time without notice. Pin your FastMCP version when using these features.


# Releases
Source: https://gofastmcp.com/development/releases

FastMCP versioning and release process

FastMCP releases frequently to deliver features quickly in the rapidly evolving MCP ecosystem. We use semantic versioning pragmatically - the Model Context Protocol is young, patterns are still emerging, and waiting for perfect stability would mean missing opportunities to empower developers with better tools.

## Versioning Policy

### Semantic Versioning

**Major (x.0.0)**: Complete API redesigns

Major versions represent fundamental shifts. FastMCP 2.x is entirely different from 1.x in both implementation and design philosophy.

**Minor (2.x.0)**: New features and evolution

<Warning>
  Unlike traditional semantic versioning, minor versions **may** include [breaking changes](#breaking-changes) when necessary for the ecosystem's evolution. This flexibility is essential in a young ecosystem where perfect backwards compatibility would prevent important improvements.
</Warning>

FastMCP always targets the most current MCP Protocol version. Breaking changes in the MCP spec or MCP SDK automatically flow through to FastMCP - we prioritize staying current with the latest features and conventions over maintaining compatibility with older protocol versions.

**Patch (2.0.x)**: Bug fixes and refinements

Patch versions contain only bug fixes without breaking changes. These are safe updates you can apply with confidence.

### Breaking Changes

We permit breaking changes in minor versions because the MCP ecosystem is rapidly evolving. Refusing to break problematic APIs would accumulate design debt that eventually makes the framework unusable. Each breaking change represents a deliberate decision to keep FastMCP aligned with the ecosystem's evolution.

When breaking changes occur:

* They only happen in minor versions (e.g., 2.3.x to 2.4.0)
* Release notes explain what changed and how to migrate
* We provide deprecation warnings at least 1 minor version in advance when possible
* Changes must substantially benefit users to justify disruption

The public API is what's covered by our compatibility guarantees - these are the parts of FastMCP you can rely on to remain stable within a minor version. The public API consists of:

* `FastMCP` server class, `Client` class, and FastMCP `Context`
* Core MCP components: `Tool`, `Prompt`, `Resource`, `ResourceTemplate`, and transports
* Their public methods and documented behaviors

Everything else (utilities, private methods, internal modules) may change without notice. This boundary lets us refactor internals and improve implementation details without breaking your code. For production stability, pin to specific versions.

<Warning>
  The `fastmcp.server.auth` module was introduced in 2.12.0 and is exempted from this policy temporarily, meaning it is *expected* to have breaking changes even on patch versions. This is because auth is a rapidly evolving part of the MCP spec and it would be dangerous to be beholden to old decisions. Please pin your FastMCP version if using authentication in production.

  We expect this exemption to last through at least the 2.12.x and 2.13.x release series.
</Warning>

### Production Use

Pin to exact versions:

```
fastmcp==2.11.0  # Good
fastmcp>=2.11.0  # Bad - will install breaking changes
```

## Creating Releases

Our release process is intentionally simple:

1. Create GitHub release with tag `vMAJOR.MINOR.PATCH` (e.g., `v2.11.0`)
2. Generate release notes automatically, and curate or add additional editorial information as needed
3. GitHub releases automatically trigger PyPI deployments

This automation lets maintainers focus on code quality rather than release mechanics.

### Release Cadence

We follow a feature-driven release cadence rather than a fixed schedule. Minor versions ship approximately every 3-4 weeks when significant functionality is ready.

Patch releases ship promptly for:

* Critical bug fixes
* Security updates (immediate release)
* Regression fixes

This approach means you get improvements as soon as they're ready rather than waiting for arbitrary release dates.


# Tests
Source: https://gofastmcp.com/development/tests

Testing patterns and requirements for FastMCP

Good tests are the foundation of reliable software. In FastMCP, we treat tests as first-class documentation that demonstrates how features work while protecting against regressions. Every new capability needs comprehensive tests that demonstrate correctness.

## FastMCP Tests

### Running Tests

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/server/test_auth.py

# Run with coverage
uv run pytest --cov=fastmcp

# Skip integration tests for faster runs
uv run pytest -m "not integration"

# Skip tests that spawn processes
uv run pytest -m "not integration and not client_process"
```

Tests should complete in under 1 second unless marked as integration tests. This speed encourages running them frequently, catching issues early.

### Test Organization

Our test organization mirrors the `src/` directory structure, creating a predictable mapping between code and tests. When you're working on `src/fastmcp/server/auth.py`, you'll find its tests in `tests/server/test_auth.py`. In rare cases tests are split further - for example, the OpenAPI tests are so comprehensive they're split across multiple files.

### Test Markers

We use pytest markers to categorize tests that require special resources or take longer to run:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@pytest.mark.integration
async def test_github_api_integration():
    """Test GitHub API integration with real service."""
    token = os.getenv("FASTMCP_GITHUB_TOKEN")
    if not token:
        pytest.skip("FASTMCP_GITHUB_TOKEN not available")
    
    # Test against real GitHub API
    client = GitHubClient(token)
    repos = await client.list_repos("prefecthq")
    assert "fastmcp" in [repo.name for repo in repos]

@pytest.mark.client_process
async def test_stdio_transport():
    """Test STDIO transport with separate process."""
    # This spawns a subprocess
    async with Client("python examples/simple_echo.py") as client:
        result = await client.call_tool("echo", {"message": "test"})
        assert result.content[0].text == "test"
```

## Writing Tests

### Test Requirements

Following these practices creates maintainable, debuggable test suites that serve as both documentation and regression protection.

#### Single Behavior Per Test

Each test should verify exactly one behavior. When it fails, you need to know immediately what broke. A test that checks five things gives you five potential failure points to investigate. A test that checks one thing points directly to the problem.

<CodeGroup>
  ```python Good: Atomic Test theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  async def test_tool_registration():
      """Test that tools are properly registered with the server."""
      mcp = FastMCP("test-server")
      
      @mcp.tool
      def add(a: int, b: int) -> int:
          return a + b
      
      tools = mcp.list_tools()
      assert len(tools) == 1
      assert tools[0].name == "add"
  ```

  ```python Bad: Multi-Behavior Test theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  async def test_server_functionality():
      """Test multiple server features at once."""
      mcp = FastMCP("test-server")
      
      # Tool registration
      @mcp.tool
      def add(a: int, b: int) -> int:
          return a + b
      
      # Resource creation
      @mcp.resource("config://app")
      def get_config():
          return {"version": "1.0"}
      
      # Authentication setup
      mcp.auth = BearerTokenProvider({"token": "user"})
      
      # What exactly are we testing? If this fails, what broke?
      assert mcp.list_tools()
      assert mcp.list_resources()
      assert mcp.auth is not None
  ```
</CodeGroup>

#### Self-Contained Setup

Every test must create its own setup. Tests should be runnable in any order, in parallel, or in isolation. When a test fails, you should be able to run just that test to reproduce the issue.

<CodeGroup>
  ```python Good: Self-Contained theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  async def test_tool_execution_with_error():
      """Test that tool errors are properly handled."""
      mcp = FastMCP("test-server")
      
      @mcp.tool
      def divide(a: int, b: int) -> float:
          if b == 0:
              raise ValueError("Cannot divide by zero")
          return a / b
      
      async with Client(mcp) as client:
          with pytest.raises(Exception):
              await client.call_tool("divide", {"a": 10, "b": 0})
  ```

  ```python Bad: Test Dependencies theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  # Global state that tests depend on
  test_server = None

  def test_setup_server():
      """Setup for other tests."""
      global test_server
      test_server = FastMCP("shared-server")

  def test_server_works():
      """Test server functionality."""
      # Depends on test_setup_server running first
      assert test_server is not None
  ```
</CodeGroup>

#### Clear Intent

Test names and assertions should make the verified behavior obvious. A developer reading your test should understand what feature it validates and how that feature should behave.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def test_authenticated_tool_requires_valid_token():
    """Test that authenticated users can access protected tools."""
    mcp = FastMCP("test-server")
    mcp.auth = BearerTokenProvider({"secret-token": "test-user"})
    
    @mcp.tool
    def protected_action() -> str:
        return "success"
    
    async with Client(mcp, auth=BearerAuth("secret-token")) as client:
        result = await client.call_tool("protected_action", {})
        assert result.content[0].text == "success"
```

#### Using Fixtures

Use fixtures to create reusable data, server configurations, or other resources for your tests. Note that you should **not** open FastMCP clients in your fixtures as it can create hard-to-diagnose issues with event loops.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import pytest
from fastmcp import FastMCP, Client

@pytest.fixture
def weather_server():
    server = FastMCP("WeatherServer")
    
    @server.tool
    def get_temperature(city: str) -> dict:
        temps = {"NYC": 72, "LA": 85, "Chicago": 68}
        return {"city": city, "temp": temps.get(city, 70)}
    
    return server

async def test_temperature_tool(weather_server):
    async with Client(weather_server) as client:
        result = await client.call_tool("get_temperature", {"city": "LA"})
        assert result.data == {"city": "LA", "temp": 85}
```

#### Effective Assertions

Assertions should be specific and provide context on failure. When a test fails during CI, the assertion message should tell you exactly what went wrong.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Basic assertion - minimal context on failure
assert result.status == "success"

# Better - explains what was expected
assert result.status == "success", f"Expected successful operation, got {result.status}: {result.error}"
```

Try not to have too many assertions in a single test unless you truly need to check various aspects of the same behavior. In general, assertions of different behaviors should be in separate tests.

#### Inline Snapshots

FastMCP uses `inline-snapshot` for testing complex data structures. On first run of `pytest --inline-snapshot=create` with an empty `snapshot()`, pytest will auto-populate the expected value. To update snapshots after intentional changes, run `pytest --inline-snapshot=fix`. This is particularly useful for testing JSON schemas and API responses.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from inline_snapshot import snapshot

async def test_tool_schema_generation():
    """Test that tool schemas are generated correctly."""
    mcp = FastMCP("test-server")
    
    @mcp.tool
    def calculate_tax(amount: float, rate: float = 0.1) -> dict:
        """Calculate tax on an amount."""
        return {"amount": amount, "tax": amount * rate, "total": amount * (1 + rate)}
    
    tools = mcp.list_tools()
    schema = tools[0].inputSchema
    
    # First run: snapshot() is empty, gets auto-populated
    # Subsequent runs: compares against stored snapshot
    assert schema == snapshot({
        "type": "object", 
        "properties": {
            "amount": {"type": "number"}, 
            "rate": {"type": "number", "default": 0.1}
        }, 
        "required": ["amount"]
    })
```

### In-Memory Testing

FastMCP uses in-memory transport for testing, where servers and clients communicate directly. The majority of functionality can be tested in a deterministic fashion this way. We use more complex setups only when testing transports themselves.

The in-memory transport runs the real MCP protocol implementation without network overhead. Instead of deploying your server or managing network connections, you pass your server instance directly to the client. Everything runs in the same Python process - you can set breakpoints anywhere and step through with your debugger.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP, Client

# Create your server
server = FastMCP("WeatherServer")

@server.tool
def get_temperature(city: str) -> dict:
    """Get current temperature for a city"""
    temps = {"NYC": 72, "LA": 85, "Chicago": 68}
    return {"city": city, "temp": temps.get(city, 70)}

async def test_weather_operations():
    # Pass server directly - no deployment needed
    async with Client(server) as client:
        result = await client.call_tool("get_temperature", {"city": "NYC"})
        assert result.data == {"city": "NYC", "temp": 72}
```

This pattern makes tests deterministic and fast - typically completing in milliseconds rather than seconds.

### Mocking External Dependencies

FastMCP servers are standard Python objects, so you can mock external dependencies using your preferred approach:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from unittest.mock import AsyncMock

async def test_database_tool():
    server = FastMCP("DataServer")
    
    # Mock the database
    mock_db = AsyncMock()
    mock_db.fetch_users.return_value = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
    ]
    
    @server.tool
    async def list_users() -> list:
        return await mock_db.fetch_users()
    
    async with Client(server) as client:
        result = await client.call_tool("list_users", {})
        assert len(result.data) == 2
        assert result.data[0]["name"] == "Alice"
        mock_db.fetch_users.assert_called_once()
```

### Testing Network Transports

While in-memory testing covers most unit testing needs, you'll occasionally need to test actual network transports like HTTP or SSE. FastMCP provides two approaches: in-process async servers (preferred), and separate subprocess servers (for special cases).

#### In-Process Network Testing (Preferred)

<VersionBadge />

For most network transport tests, use `run_server_async` as an async context manager. This runs the server as a task in the same process, providing fast, deterministic tests with full debugger support:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import pytest
from fastmcp import FastMCP, Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.tests import run_server_async

def create_test_server() -> FastMCP:
    """Create a test server instance."""
    server = FastMCP("TestServer")

    @server.tool
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    return server

@pytest.fixture
async def http_server() -> str:
    """Start server in-process for testing."""
    server = create_test_server()
    async with run_server_async(server) as url:
        yield url

async def test_http_transport(http_server: str):
    """Test actual HTTP transport behavior."""
    async with Client(
        transport=StreamableHttpTransport(http_server)
    ) as client:
        result = await client.ping()
        assert result is True

        greeting = await client.call_tool("greet", {"name": "World"})
        assert greeting.data == "Hello, World!"
```

The `run_server_async` context manager automatically handles server lifecycle and cleanup. This approach is faster than subprocess-based testing and provides better error messages.

#### Subprocess Testing (Special Cases)

For tests that require complete process isolation (like STDIO transport or testing subprocess behavior), use `run_server_in_process`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import pytest
from fastmcp.utilities.tests import run_server_in_process
from fastmcp import FastMCP, Client
from fastmcp.client.transports import StreamableHttpTransport

def run_server(host: str, port: int) -> None:
    """Function to run in subprocess."""
    server = FastMCP("TestServer")
    
    @server.tool
    def greet(name: str) -> str:
        return f"Hello, {name}!"
    
    server.run(host=host, port=port)

@pytest.fixture
async def http_server():
    """Fixture that runs server in subprocess."""
    with run_server_in_process(run_server, transport="http") as url:
        yield f"{url}/mcp"

async def test_http_transport(http_server: str):
    """Test actual HTTP transport behavior."""
    async with Client(
        transport=StreamableHttpTransport(http_server)
    ) as client:
        result = await client.ping()
        assert result is True
```

The `run_server_in_process` utility handles server lifecycle, port allocation, and cleanup automatically. Use this only when subprocess isolation is truly necessary, as it's slower and harder to debug than in-process testing. FastMCP uses the `client_process` marker to isolate these tests in CI.

### Documentation Testing

Documentation requires the same validation as code. The `just docs` command launches a local Mintlify server that renders your documentation exactly as users will see it:

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Start local documentation server with hot reload
just docs

# Or run Mintlify directly
mintlify dev
```

The local server watches for changes and automatically refreshes. This preview catches formatting issues and helps you see documentation as users will experience it.
