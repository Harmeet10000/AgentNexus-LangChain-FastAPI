# Client Core

Source lines: 2046-3388 from the original FastMCP documentation dump.

Client creation, callbacks, elicitation, logging, notifications, prompts, resources, roots, sampling, and background task primitives.

---

# The FastMCP Client
Source: https://gofastmcp.com/clients/client

Programmatic client for interacting with MCP servers through a well-typed, Pythonic interface.

<VersionBadge />

The `fastmcp.Client` class provides a programmatic interface for interacting with any MCP server. It handles protocol details and connection management automatically, letting you focus on the operations you want to perform.

The FastMCP Client is designed for deterministic, controlled interactions rather than autonomous behavior, making it ideal for testing MCP servers during development, building deterministic applications that need reliable MCP interactions, and creating the foundation for agentic or LLM-based clients with structured, type-safe operations.

<Note>
  This is a programmatic client that requires explicit function calls and provides direct control over all MCP operations. Use it as a building block for higher-level systems.
</Note>

## Creating a Client

You provide a server source and the client automatically infers the appropriate transport mechanism.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import asyncio
from fastmcp import Client, FastMCP

# In-memory server (ideal for testing)
server = FastMCP("TestServer")
client = Client(server)

# HTTP server
client = Client("https://example.com/mcp")

# Local Python script
client = Client("my_mcp_server.py")

async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()

        # Execute operations
        result = await client.call_tool("example_tool", {"param": "value"})
        print(result)

asyncio.run(main())
```

All client operations require using the `async with` context manager for proper connection lifecycle management.

## Choosing a Transport

The client automatically selects a transport based on what you pass to it, but different transports have different characteristics that matter for your use case.

**In-memory transport** connects directly to a FastMCP server instance within the same Python process. Use this for testing and development where you want to eliminate subprocess and network complexity. The server shares your process's environment and memory space.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client, FastMCP

server = FastMCP("TestServer")
client = Client(server)  # In-memory, no network or subprocess
```

**STDIO transport** launches a server as a subprocess and communicates through stdin/stdout pipes. This is the standard mechanism used by desktop clients like Claude Desktop. The subprocess runs in an isolated environment, so you must explicitly pass any environment variables the server needs.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

# Simple inference from file path
client = Client("my_server.py")

# With explicit environment configuration
client = Client("my_server.py", env={"API_KEY": "secret"})
```

**HTTP transport** connects to servers running as web services. Use this for production deployments where the server runs independently and manages its own lifecycle.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

client = Client("https://api.example.com/mcp")
```

See [Transports](/clients/transports) for detailed configuration options including authentication headers, session persistence, and multi-server configurations.

## Configuration-Based Clients

<VersionBadge />

Create clients from MCP configuration dictionaries, which can include multiple servers. While there is no official standard for MCP configuration format, FastMCP follows established conventions used by tools like Claude Desktop.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather-api.example.com/mcp"
        },
        "assistant": {
            "command": "python",
            "args": ["./assistant_server.py"]
        }
    }
}

client = Client(config)

async with client:
    # Tools are prefixed with server names
    weather_data = await client.call_tool("weather_get_forecast", {"city": "London"})
    response = await client.call_tool("assistant_answer_question", {"question": "What's the capital of France?"})

    # Resources use prefixed URIs
    icons = await client.read_resource("weather://weather/icons/sunny")
```

## Connection Lifecycle

The client uses context managers for connection management. When you enter the context, the client establishes a connection and performs an MCP initialization handshake with the server. This handshake exchanges capabilities, server metadata, and instructions.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client, FastMCP

mcp = FastMCP(name="MyServer", instructions="Use the greet tool to say hello!")

@mcp.tool
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

async with Client(mcp) as client:
    # Initialization already happened automatically
    print(f"Server: {client.initialize_result.serverInfo.name}")
    print(f"Instructions: {client.initialize_result.instructions}")
    print(f"Capabilities: {client.initialize_result.capabilities.tools}")
```

For advanced scenarios where you need precise control over when initialization happens, disable automatic initialization and call `initialize()` manually:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

client = Client("my_mcp_server.py", auto_initialize=False)

async with client:
    # Connection established, but not initialized yet
    print(f"Connected: {client.is_connected()}")
    print(f"Initialized: {client.initialize_result is not None}")  # False

    # Initialize manually with custom timeout
    result = await client.initialize(timeout=10.0)
    print(f"Server: {result.serverInfo.name}")

    # Now ready for operations
    tools = await client.list_tools()
```

## Operations

FastMCP clients interact with three types of server components.

**Tools** are server-side functions that the client can execute with arguments. Call them with `call_tool()` and receive structured results.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    tools = await client.list_tools()
    result = await client.call_tool("multiply", {"a": 5, "b": 3})
    print(result.data)  # 15
```

See [Tools](/clients/tools) for detailed documentation including version selection, error handling, and structured output.

**Resources** are data sources that the client can read, either static or templated. Access them with `read_resource()` using URIs.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    resources = await client.list_resources()
    content = await client.read_resource("file:///config/settings.json")
    print(content[0].text)
```

See [Resources](/clients/resources) for detailed documentation including templates and binary content.

**Prompts** are reusable message templates that can accept arguments. Retrieve rendered prompts with `get_prompt()`.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    prompts = await client.list_prompts()
    messages = await client.get_prompt("analyze_data", {"data": [1, 2, 3]})
    print(messages.messages)
```

See [Prompts](/clients/prompts) for detailed documentation including argument serialization.

## Callback Handlers

The client supports callback handlers for advanced server interactions. These let you respond to server-initiated requests and receive notifications.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.logging import LogMessage

async def log_handler(message: LogMessage):
    print(f"Server log: {message.data}")

async def progress_handler(progress: float, total: float | None, message: str | None):
    print(f"Progress: {progress}/{total} - {message}")

async def sampling_handler(messages, params, context):
    # Integrate with your LLM service here
    return "Generated response"

client = Client(
    "my_mcp_server.py",
    log_handler=log_handler,
    progress_handler=progress_handler,
    sampling_handler=sampling_handler,
    timeout=30.0
)
```

Each handler type has its own documentation:

* **[Sampling](/clients/sampling)** - Respond to server LLM requests
* **[Elicitation](/clients/elicitation)** - Handle server requests for user input
* **[Progress](/clients/progress)** - Monitor long-running operations
* **[Logging](/clients/logging)** - Handle server log messages
* **[Roots](/clients/roots)** - Provide local context to servers

<Tip>
  The FastMCP Client is designed as a foundational tool. Use it directly for deterministic operations, or build higher-level agentic systems on top of its reliable, type-safe interface.
</Tip>


# User Elicitation
Source: https://gofastmcp.com/clients/elicitation

Handle server requests for structured user input.

<VersionBadge />

Use this when you need to respond to server requests for user input during tool execution.

Elicitation allows MCP servers to request structured input from users during operations. Instead of requiring all inputs upfront, servers can interactively ask for missing parameters, request clarification, or gather additional context.

## Handler Template

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult, ElicitRequestParams, RequestContext

async def elicitation_handler(
    message: str,
    response_type: type | None,
    params: ElicitRequestParams,
    context: RequestContext
) -> ElicitResult | object:
    """
    Handle server requests for user input.

    Args:
        message: The prompt to display to the user
        response_type: Python dataclass type for the response (None if no data expected)
        params: Original MCP elicitation parameters including raw JSON schema
        context: Request context with metadata

    Returns:
        - Data directly (implicitly accepts the elicitation)
        - ElicitResult for explicit control over the action
    """
    # Present the message and collect input
    user_input = input(f"{message}: ")

    if not user_input:
        return ElicitResult(action="decline")

    # Create response using the provided dataclass type
    return response_type(value=user_input)

client = Client(
    "my_mcp_server.py",
    elicitation_handler=elicitation_handler,
)
```

## How It Works

When a server needs user input, it sends an elicitation request with a message prompt and a JSON schema describing the expected response structure. FastMCP automatically converts this schema into a Python dataclass type, making it easy to construct properly typed responses without manually parsing JSON schemas.

The handler receives four parameters:

<Card icon="code" title="Handler Parameters">
  <ResponseField name="message" type="str">
    The prompt message to display to the user
  </ResponseField>

  <ResponseField name="response_type" type="type | None">
    A Python dataclass type that FastMCP created from the server's JSON schema. Use this to construct your response with proper typing. If the server requests an empty object, this will be `None`.
  </ResponseField>

  <ResponseField name="params" type="ElicitRequestParams">
    The original MCP elicitation parameters, including the raw JSON schema in `params.requestedSchema`
  </ResponseField>

  <ResponseField name="context" type="RequestContext">
    Request context containing metadata about the elicitation request
  </ResponseField>
</Card>

## Response Actions

You can return data directly, which implicitly accepts the elicitation:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def elicitation_handler(message, response_type, params, context):
    user_input = input(f"{message}: ")
    return response_type(value=user_input)  # Implicit accept
```

Or return an `ElicitResult` for explicit control over the action:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.client.elicitation import ElicitResult

async def elicitation_handler(message, response_type, params, context):
    user_input = input(f"{message}: ")

    if not user_input:
        return ElicitResult(action="decline")  # User declined

    if user_input == "cancel":
        return ElicitResult(action="cancel")   # Cancel entire operation

    return ElicitResult(
        action="accept",
        content=response_type(value=user_input)
    )
```

**Action types:**

* **`accept`**: User provided valid input. Include the data in the `content` field.
* **`decline`**: User chose not to provide the requested information. Omit `content`.
* **`cancel`**: User cancelled the entire operation. Omit `content`.

## Example

A file management tool might ask which directory to create:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult

async def elicitation_handler(message, response_type, params, context):
    print(f"Server asks: {message}")

    user_response = input("Your response: ")

    if not user_response:
        return ElicitResult(action="decline")

    # Use the response_type dataclass to create a properly structured response
    return response_type(value=user_response)

client = Client(
    "my_mcp_server.py",
    elicitation_handler=elicitation_handler
)
```


# Server Logging
Source: https://gofastmcp.com/clients/logging

Receive and handle log messages from MCP servers.

<VersionBadge />

Use this when you need to capture or process log messages sent by the server.

MCP servers can emit log messages to clients. The client handles these through a log handler callback.

## Log Handler

Provide a `log_handler` function when creating the client:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import logging
from fastmcp import Client
from fastmcp.client.logging import LogMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()

async def log_handler(message: LogMessage):
    """Forward MCP server logs to Python's logging system."""
    msg = message.data.get('msg')
    extra = message.data.get('extra')

    level = LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)
    logger.log(level, msg, extra=extra)

client = Client(
    "my_mcp_server.py",
    log_handler=log_handler,
)
```

The handler receives a `LogMessage` object:

<Card icon="code" title="LogMessage">
  <ResponseField name="level" type="Literal[&#x22;debug&#x22;, &#x22;info&#x22;, &#x22;notice&#x22;, &#x22;warning&#x22;, &#x22;error&#x22;, &#x22;critical&#x22;, &#x22;alert&#x22;, &#x22;emergency&#x22;]">
    The log level
  </ResponseField>

  <ResponseField name="logger" type="str | None">
    The logger name (may be None)
  </ResponseField>

  <ResponseField name="data" type="dict">
    The log payload, containing `msg` and `extra` keys
  </ResponseField>
</Card>

## Structured Logs

The `message.data` attribute is a dictionary containing the log payload. This enables structured logging with rich contextual information.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async def detailed_log_handler(message: LogMessage):
    msg = message.data.get('msg')
    extra = message.data.get('extra')

    if message.level == "error":
        print(f"ERROR: {msg} | Details: {extra}")
    elif message.level == "warning":
        print(f"WARNING: {msg} | Details: {extra}")
    else:
        print(f"{message.level.upper()}: {msg}")
```

This structure is preserved even when logs are forwarded through a FastMCP proxy, making it useful for debugging multi-server applications.

## Default Behavior

If you do not provide a custom `log_handler`, FastMCP's default handler routes server logs to Python's logging system at the appropriate severity level. The MCP levels map as follows: `notice` becomes INFO; `alert` and `emergency` become CRITICAL.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
client = Client("my_mcp_server.py")

async with client:
    # Server logs are forwarded at proper severity automatically
    await client.call_tool("some_tool")
```


# Notifications
Source: https://gofastmcp.com/clients/notifications

Handle server-sent notifications for list changes and other events.

<VersionBadge />

Use this when you need to react to server-side changes like tool list updates or resource modifications.

MCP servers can send notifications to inform clients about state changes. The message handler provides a unified way to process these notifications.

## Handling Notifications

The simplest approach is a function that receives all messages and filters for the notifications you care about:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

async def message_handler(message):
    """Handle MCP notifications from the server."""
    if hasattr(message, 'root'):
        method = message.root.method

        if method == "notifications/tools/list_changed":
            print("Tools have changed - refresh tool cache")
        elif method == "notifications/resources/list_changed":
            print("Resources have changed")
        elif method == "notifications/prompts/list_changed":
            print("Prompts have changed")

client = Client(
    "my_mcp_server.py",
    message_handler=message_handler,
)
```

## MessageHandler Class

For fine-grained targeting, subclass `MessageHandler` to use specific hooks:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.messages import MessageHandler
import mcp.types

class MyMessageHandler(MessageHandler):
    async def on_tool_list_changed(
        self, notification: mcp.types.ToolListChangedNotification
    ) -> None:
        """Handle tool list changes."""
        print("Tool list changed - refreshing available tools")

    async def on_resource_list_changed(
        self, notification: mcp.types.ResourceListChangedNotification
    ) -> None:
        """Handle resource list changes."""
        print("Resource list changed")

    async def on_prompt_list_changed(
        self, notification: mcp.types.PromptListChangedNotification
    ) -> None:
        """Handle prompt list changes."""
        print("Prompt list changed")

client = Client(
    "my_mcp_server.py",
    message_handler=MyMessageHandler(),
)
```

### Handler Template

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.client.messages import MessageHandler
import mcp.types

class MyMessageHandler(MessageHandler):
    async def on_message(self, message) -> None:
        """Called for ALL messages (requests and notifications)."""
        pass

    async def on_notification(
        self, notification: mcp.types.ServerNotification
    ) -> None:
        """Called for notifications (fire-and-forget)."""
        pass

    async def on_tool_list_changed(
        self, notification: mcp.types.ToolListChangedNotification
    ) -> None:
        """Called when the server's tool list changes."""
        pass

    async def on_resource_list_changed(
        self, notification: mcp.types.ResourceListChangedNotification
    ) -> None:
        """Called when the server's resource list changes."""
        pass

    async def on_prompt_list_changed(
        self, notification: mcp.types.PromptListChangedNotification
    ) -> None:
        """Called when the server's prompt list changes."""
        pass

    async def on_progress(
        self, notification: mcp.types.ProgressNotification
    ) -> None:
        """Called for progress updates during long-running operations."""
        pass

    async def on_logging_message(
        self, notification: mcp.types.LoggingMessageNotification
    ) -> None:
        """Called for log messages from the server."""
        pass
```

## List Change Notifications

A practical example of maintaining a tool cache that refreshes when tools change:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.messages import MessageHandler
import mcp.types

class ToolCacheHandler(MessageHandler):
    def __init__(self):
        self.cached_tools = []

    async def on_tool_list_changed(
        self, notification: mcp.types.ToolListChangedNotification
    ) -> None:
        """Clear tool cache when tools change."""
        print("Tools changed - clearing cache")
        self.cached_tools = []  # Force refresh on next access

client = Client("server.py", message_handler=ToolCacheHandler())
```

## Server Requests

While the message handler receives server-initiated requests, you should use dedicated callback parameters for most interactive scenarios:

* **Sampling requests**: Use [`sampling_handler`](/clients/sampling)
* **Elicitation requests**: Use [`elicitation_handler`](/clients/elicitation)
* **Progress updates**: Use [`progress_handler`](/clients/progress)
* **Log messages**: Use [`log_handler`](/clients/logging)

The message handler is primarily for monitoring and handling notifications rather than responding to requests.


# Progress Monitoring
Source: https://gofastmcp.com/clients/progress

Handle progress notifications from long-running server operations.

<VersionBadge />

Use this when you need to track progress of long-running operations.

MCP servers can report progress during operations. The client receives these updates through a progress handler.

## Progress Handler

Set a handler when creating the client:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

async def progress_handler(
    progress: float,
    total: float | None,
    message: str | None
) -> None:
    if total is not None:
        percentage = (progress / total) * 100
        print(f"Progress: {percentage:.1f}% - {message or ''}")
    else:
        print(f"Progress: {progress} - {message or ''}")

client = Client(
    "my_mcp_server.py",
    progress_handler=progress_handler
)
```

The handler receives three parameters:

<Card icon="code" title="Handler Parameters">
  <ResponseField name="progress" type="float">
    Current progress value
  </ResponseField>

  <ResponseField name="total" type="float | None">
    Expected total value (may be None if unknown)
  </ResponseField>

  <ResponseField name="message" type="str | None">
    Optional status message
  </ResponseField>
</Card>

## Per-Call Handler

Override the client-level handler for specific tool calls:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.call_tool(
        "long_running_task",
        {"param": "value"},
        progress_handler=my_progress_handler
    )
```


# Getting Prompts
Source: https://gofastmcp.com/clients/prompts

Retrieve rendered message templates with automatic argument serialization.

<VersionBadge />

Use this when you need to retrieve server-defined message templates for LLM interactions.

Prompts are reusable message templates exposed by MCP servers. They can accept arguments to generate personalized message sequences for LLM interactions.

## Basic Usage

Request a rendered prompt with `get_prompt()`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    # Simple prompt without arguments
    result = await client.get_prompt("welcome_message")
    # result -> mcp.types.GetPromptResult

    # Access the generated messages
    for message in result.messages:
        print(f"Role: {message.role}")
        print(f"Content: {message.content}")
```

Pass arguments to customize the prompt:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.get_prompt("user_greeting", {
        "name": "Alice",
        "role": "administrator"
    })

    for message in result.messages:
        print(f"Generated message: {message.content}")
```

## Argument Serialization

<VersionBadge />

FastMCP automatically serializes complex arguments to JSON strings as required by the MCP specification. You can pass typed objects directly:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from dataclasses import dataclass

@dataclass
class UserData:
    name: str
    age: int

async with client:
    result = await client.get_prompt("analyze_user", {
        "user": UserData(name="Alice", age=30),     # Automatically serialized
        "preferences": {"theme": "dark"},           # Dict serialized
        "scores": [85, 92, 78],                     # List serialized
        "simple_name": "Bob"                        # Strings unchanged
    })
```

The client handles serialization using `pydantic_core.to_json()` for consistent formatting. FastMCP servers automatically deserialize these JSON strings back to the expected types.

## Working with Results

The `get_prompt()` method returns a `GetPromptResult` containing a list of messages:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.get_prompt("conversation_starter", {"topic": "climate"})

    for i, message in enumerate(result.messages):
        print(f"Message {i + 1}:")
        print(f"  Role: {message.role}")
        print(f"  Content: {message.content.text if hasattr(message.content, 'text') else message.content}")
```

Prompts can generate different message types. System messages configure LLM behavior:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.get_prompt("system_configuration", {
        "role": "helpful assistant",
        "expertise": "python programming"
    })

    # Access the returned messages
    message = result.messages[0]
    print(f"Prompt: {message.content}")
```

Conversation templates generate multi-turn flows:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.get_prompt("interview_template", {
        "candidate_name": "Alice",
        "position": "Senior Developer"
    })

    # Multiple messages for a conversation flow
    for message in result.messages:
        print(f"{message.role}: {message.content}")
```

## Version Selection

<VersionBadge />

When a server exposes multiple versions of a prompt, you can request a specific version:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    # Get the highest version (default)
    result = await client.get_prompt("summarize", {"text": "..."})

    # Get a specific version
    result_v1 = await client.get_prompt("summarize", {"text": "..."}, version="1.0")
```

See [Metadata](/servers/versioning#version-discovery) for how to discover available versions.

## Multi-Server Clients

When using multi-server clients, prompts are accessible directly without prefixing:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:  # Multi-server client
    result1 = await client.get_prompt("weather_prompt", {"city": "London"})
    result2 = await client.get_prompt("assistant_prompt", {"query": "help"})
```

## Raw Protocol Access

For complete control, use `get_prompt_mcp()` which returns the full MCP protocol object:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.get_prompt_mcp("example_prompt", {"arg": "value"})
    # result -> mcp.types.GetPromptResult
```


# Reading Resources
Source: https://gofastmcp.com/clients/resources

Access static and templated data sources from MCP servers.

<VersionBadge />

Use this when you need to read data from server-exposed resources like configuration files, generated content, or external data sources.

Resources are data sources exposed by MCP servers. They can be static files with fixed content, or dynamic templates that generate content based on parameters in the URI.

## Reading Resources

Read a resource using its URI:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    content = await client.read_resource("file:///path/to/README.md")
    # content -> list[TextResourceContents | BlobResourceContents]

    # Access text content
    if hasattr(content[0], 'text'):
        print(content[0].text)

    # Access binary content
    if hasattr(content[0], 'blob'):
        print(f"Binary data: {len(content[0].blob)} bytes")
```

Resource templates generate content based on URI parameters. The template defines a pattern like `weather://{{city}}/current`, and you fill in the parameters when reading:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    # Read from a resource template
    weather_content = await client.read_resource("weather://london/current")
    print(weather_content[0].text)
```

## Content Types

Resources return different content types depending on what they expose.

Text resources include configuration files, JSON data, and other human-readable content:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    content = await client.read_resource("resource://config/settings.json")

    for item in content:
        if hasattr(item, 'text'):
            print(f"Text content: {item.text}")
            print(f"MIME type: {item.mimeType}")
```

Binary resources include images, PDFs, and other non-text data:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    content = await client.read_resource("resource://images/logo.png")

    for item in content:
        if hasattr(item, 'blob'):
            print(f"Binary content: {len(item.blob)} bytes")
            print(f"MIME type: {item.mimeType}")

            # Save to file
            with open("downloaded_logo.png", "wb") as f:
                f.write(item.blob)
```

## Multi-Server Clients

When using multi-server clients, resource URIs are prefixed with the server name:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:  # Multi-server client
    weather_icons = await client.read_resource("weather://weather/icons/sunny")
    templates = await client.read_resource("resource://assistant/templates/list")
```

## Version Selection

<VersionBadge />

When a server exposes multiple versions of a resource, you can request a specific version:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    # Read the highest version (default)
    content = await client.read_resource("data://config")

    # Read a specific version
    content_v1 = await client.read_resource("data://config", version="1.0")
```

See [Metadata](/servers/versioning#version-discovery) for how to discover available versions.

## Raw Protocol Access

For complete control, use `read_resource_mcp()` which returns the full MCP protocol object:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async with client:
    result = await client.read_resource_mcp("resource://example")
    # result -> mcp.types.ReadResourceResult
```


# Client Roots
Source: https://gofastmcp.com/clients/roots

Provide local context and resource boundaries to MCP servers.

<VersionBadge />

Use this when you need to tell servers what local resources the client has access to.

Roots inform servers about resources the client can provide. Servers can use this information to adjust behavior or provide more relevant responses.

## Static Roots

Provide a list of roots when creating the client:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

client = Client(
    "my_mcp_server.py",
    roots=["/path/to/root1", "/path/to/root2"]
)
```

## Dynamic Roots

Use a callback to compute roots dynamically when the server requests them:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.roots import RequestContext

async def roots_callback(context: RequestContext) -> list[str]:
    print(f"Server requested roots (Request ID: {context.request_id})")
    return ["/path/to/root1", "/path/to/root2"]

client = Client(
    "my_mcp_server.py",
    roots=roots_callback
)
```


# LLM Sampling
Source: https://gofastmcp.com/clients/sampling

Handle server-initiated LLM completion requests.

<VersionBadge />

Use this when you need to respond to server requests for LLM completions.

MCP servers can request LLM completions from clients during tool execution. This enables servers to delegate AI reasoning to the client, which controls which LLM is used and how requests are made.

## Handler Template

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.sampling import SamplingMessage, SamplingParams, RequestContext

async def sampling_handler(
    messages: list[SamplingMessage],
    params: SamplingParams,
    context: RequestContext
) -> str:
    """
    Handle server requests for LLM completions.

    Args:
        messages: Conversation messages to send to the LLM
        params: Sampling parameters (temperature, max_tokens, etc.)
        context: Request context with metadata

    Returns:
        Generated text response from your LLM
    """
    # Extract message content
    conversation = []
    for message in messages:
        content = message.content.text if hasattr(message.content, 'text') else str(message.content)
        conversation.append(f"{message.role}: {content}")

    # Use the system prompt if provided
    system_prompt = params.systemPrompt or "You are a helpful assistant."

    # Integrate with your LLM service here
    return "Generated response based on the messages"

client = Client(
    "my_mcp_server.py",
    sampling_handler=sampling_handler,
)
```

## Handler Parameters

<Card icon="code" title="SamplingMessage">
  <ResponseField name="role" type="Literal[&#x22;user&#x22;, &#x22;assistant&#x22;]">
    The role of the message
  </ResponseField>

  <ResponseField name="content" type="TextContent | ImageContent | AudioContent">
    The content of the message. TextContent has a `.text` attribute.
  </ResponseField>
</Card>

<Card icon="code" title="SamplingParams">
  <ResponseField name="systemPrompt" type="str | None">
    Optional system prompt the server wants to use
  </ResponseField>

  <ResponseField name="modelPreferences" type="ModelPreferences | None">
    Server preferences for model selection (hints, cost/speed/intelligence priorities)
  </ResponseField>

  <ResponseField name="temperature" type="float | None">
    Sampling temperature
  </ResponseField>

  <ResponseField name="maxTokens" type="int">
    Maximum tokens to generate
  </ResponseField>

  <ResponseField name="stopSequences" type="list[str] | None">
    Stop sequences for sampling
  </ResponseField>

  <ResponseField name="tools" type="list[Tool] | None">
    Tools the LLM can use during sampling
  </ResponseField>

  <ResponseField name="toolChoice" type="ToolChoice | None">
    Tool usage behavior (`auto`, `required`, or `none`)
  </ResponseField>
</Card>

## Built-in Handlers

FastMCP provides built-in handlers for OpenAI, Anthropic, and Google Gemini APIs that support the full sampling API including tool use.

### OpenAI Handler

<VersionBadge />

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.sampling.handlers.openai import OpenAISamplingHandler

client = Client(
    "my_mcp_server.py",
    sampling_handler=OpenAISamplingHandler(default_model="gpt-4o"),
)
```

For OpenAI-compatible APIs (like local models):

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from openai import AsyncOpenAI

client = Client(
    "my_mcp_server.py",
    sampling_handler=OpenAISamplingHandler(
        default_model="llama-3.1-70b",
        client=AsyncOpenAI(base_url="http://localhost:8000/v1"),
    ),
)
```

<Note>
  Install the OpenAI handler with `pip install fastmcp[openai]`.
</Note>

### Anthropic Handler

<VersionBadge />

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.sampling.handlers.anthropic import AnthropicSamplingHandler

client = Client(
    "my_mcp_server.py",
    sampling_handler=AnthropicSamplingHandler(default_model="claude-sonnet-4-5"),
)
```

<Note>
  Install the Anthropic handler with `pip install fastmcp[anthropic]`.
</Note>

### Google Gemini Handler

<VersionBadge />

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client
from fastmcp.client.sampling.handlers.google_genai import GoogleGenAISamplingHandler

client = Client(
    "my_mcp_server.py",
    sampling_handler=GoogleGenAISamplingHandler(default_model="gemini-2.0-flash"),
)
```

<Note>
  Install the Google Gemini handler with `pip install fastmcp[gemini]`.
</Note>

## Sampling Capabilities

When you provide a `sampling_handler`, FastMCP automatically advertises full sampling capabilities to the server, including tool support. To disable tool support for simpler handlers:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from mcp.types import SamplingCapability

client = Client(
    "my_mcp_server.py",
    sampling_handler=basic_handler,
    sampling_capabilities=SamplingCapability(),  # No tool support
)
```

## Tool Execution

Tool execution happens on the server side. The client's role is to pass tools to the LLM and return the LLM's response (which may include tool use requests). The server then executes the tools and may send follow-up sampling requests with tool results.

<Tip>
  To implement a custom sampling handler, see the [handler source code](https://github.com/PrefectHQ/fastmcp/tree/main/src/fastmcp/client/sampling/handlers) as a reference.
</Tip>


# Background Tasks
Source: https://gofastmcp.com/clients/tasks

Execute operations asynchronously and track their progress.

<VersionBadge />

Use this when you need to run long operations asynchronously while doing other work.

The MCP task protocol lets you request operations to run in the background. The call returns a Task object immediately, letting you track progress, cancel operations, or await results.

## Requesting Background Execution

Pass `task=True` to run an operation as a background task:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

async with Client(server) as client:
    # Start a background task
    task = await client.call_tool("slow_computation", {"duration": 10}, task=True)

    print(f"Task started: {task.task_id}")

    # Do other work while it runs...

    # Get the result when ready
    result = await task.result()
```

This works with tools, resources, and prompts:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
tool_task = await client.call_tool("my_tool", args, task=True)
resource_task = await client.read_resource("file://large.txt", task=True)
prompt_task = await client.get_prompt("my_prompt", args, task=True)
```

## Task API

All task types share a common interface.

### Getting Results

Call `await task.result()` or simply `await task` to block until the task completes:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
task = await client.call_tool("analyze", {"text": "hello"}, task=True)

# Wait for result (blocking)
result = await task.result()
# or: result = await task
```

### Checking Status

Check the current status without blocking:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
status = await task.status()
print(f"{status.status}: {status.statusMessage}")
# status.status is "working", "completed", "failed", or "cancelled"
```

### Waiting with Control

Use `task.wait()` for more control over waiting:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Wait up to 30 seconds for completion
status = await task.wait(timeout=30.0)

# Wait for a specific state
status = await task.wait(state="completed", timeout=30.0)
```

### Cancellation

Cancel a running task:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
await task.cancel()
```

## Status Updates

Register callbacks to receive real-time status updates as the server reports progress:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
def on_status_change(status):
    print(f"Task {status.taskId}: {status.status} - {status.statusMessage}")

task.on_status_change(on_status_change)

# Async callbacks work too
async def on_status_async(status):
    await log_status(status)

task.on_status_change(on_status_async)
```

### Handler Template

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

def status_handler(status):
    """
    Handle task status updates.

    Args:
        status: Task status object with:
            - taskId: Unique task identifier
            - status: "working", "completed", "failed", or "cancelled"
            - statusMessage: Optional progress message from server
    """
    if status.status == "working":
        print(f"Progress: {status.statusMessage}")
    elif status.status == "completed":
        print("Task completed")
    elif status.status == "failed":
        print(f"Task failed: {status.statusMessage}")

task.on_status_change(status_handler)
```

## Graceful Degradation

You can always pass `task=True` regardless of whether the server supports background tasks. Per the MCP specification, servers without task support execute the operation immediately and return the result inline.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
task = await client.call_tool("my_tool", args, task=True)

if task.returned_immediately:
    print("Server executed immediately (no background support)")
else:
    print("Running in background")

# Either way, this works
result = await task.result()
```

This lets you write task-aware client code without worrying about server capabilities.

## Example

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import asyncio
from fastmcp import Client

async def main():
    async with Client(server) as client:
        # Start background task
        task = await client.call_tool(
            "slow_computation",
            {"duration": 10},
            task=True,
        )

        # Subscribe to updates
        def on_update(status):
            print(f"Progress: {status.statusMessage}")

        task.on_status_change(on_update)

        # Do other work while task runs
        print("Doing other work...")
        await asyncio.sleep(2)

        # Wait for completion and get result
        result = await task.result()
        print(f"Result: {result.content}")

asyncio.run(main())
```

See [Server Background Tasks](/servers/tasks) for how to enable background task support on the server side.
