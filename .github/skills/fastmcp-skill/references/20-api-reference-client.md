# API Reference: Client

Source lines: 31470-34657 from the original FastMCP documentation dump.

Package-level API reference for fastmcp.client, transports, sampling handlers, prompts, resources, and decorators.

---

# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-client-__init__



# `fastmcp.client`

*This module is empty or contains only private/internal implementations.*


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-client-auth-__init__



# `fastmcp.client.auth`

*This module is empty or contains only private/internal implementations.*


# bearer
Source: https://gofastmcp.com/python-sdk/fastmcp-client-auth-bearer



# `fastmcp.client.auth.bearer`

## Classes

### `BearerAuth` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/bearer.py#L11"><Icon icon="github" /></a></sup>

**Methods:**

#### `auth_flow` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/bearer.py#L15"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
auth_flow(self, request)
```


# oauth
Source: https://gofastmcp.com/python-sdk/fastmcp-client-auth-oauth



# `fastmcp.client.auth.oauth`

## Functions

### `check_if_auth_required` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L41"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
check_if_auth_required(mcp_url: str, httpx_kwargs: dict[str, Any] | None = None) -> bool
```

Check if the MCP endpoint requires authentication by making a test request.

**Returns:**

* True if auth appears to be required, False otherwise

## Classes

### `ClientNotFoundError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L37"><Icon icon="github" /></a></sup>

Raised when OAuth client credentials are not found on the server.

### `TokenStorageAdapter` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L71"><Icon icon="github" /></a></sup>

**Methods:**

#### `clear` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L99"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
clear(self) -> None
```

#### `get_tokens` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L104"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tokens(self) -> OAuthToken | None
```

#### `set_tokens` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L108"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_tokens(self, tokens: OAuthToken) -> None
```

#### `get_client_info` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L119"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_client_info(self) -> OAuthClientInformationFull | None
```

#### `set_client_info` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L125"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_client_info(self, client_info: OAuthClientInformationFull) -> None
```

### `OAuth` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L138"><Icon icon="github" /></a></sup>

OAuth client provider for MCP servers with browser-based authentication.

This class provides OAuth authentication for FastMCP clients by opening
a browser for user authorization and running a local callback server.

**Methods:**

#### `redirect_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L290"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
redirect_handler(self, authorization_url: str) -> None
```

Open browser for authorization, with pre-flight check for invalid client.

#### `callback_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L311"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
callback_handler(self) -> tuple[str, str | None]
```

Handle OAuth callback and return (auth\_code, state).

#### `async_auth_flow` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/auth/oauth.py#L350"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]
```

HTTPX auth flow with automatic retry on stale cached credentials.

If the OAuth flow fails due to invalid/stale client credentials,
clears the cache and retries once with fresh registration.


# client
Source: https://gofastmcp.com/python-sdk/fastmcp-client-client



# `fastmcp.client.client`

## Classes

### `ClientSessionState` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L95"><Icon icon="github" /></a></sup>

Holds all session-related state for a Client instance.

This allows clean separation of configuration (which is copied) from
session state (which should be fresh for each new client instance).

### `CallToolResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L112"><Icon icon="github" /></a></sup>

Parsed result from a tool call.

### `Client` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L122"><Icon icon="github" /></a></sup>

MCP client that delegates connection management to a Transport instance.

The Client class is responsible for MCP protocol logic, while the Transport
handles connection establishment and management. Client provides methods for
working with resources, prompts, tools and other MCP capabilities.

This client supports reentrant context managers (multiple concurrent
`async with client:` blocks) using reference counting and background session
management. This allows efficient session reuse in any scenario with
nested or concurrent client usage.

MCP SDK 1.10 introduced automatic list\_tools() calls during call\_tool()
execution. This created a race condition where events could be reset while
other tasks were waiting on them, causing deadlocks. The issue was exposed
in proxy scenarios but affects any reentrant usage.

The solution uses reference counting to track active context managers,
a background task to manage the session lifecycle, events to coordinate
between tasks, and ensures all session state changes happen within a lock.
Events are only created when needed, never reset outside locks.

This design prevents race conditions where tasks wait on events that get
replaced by other tasks, ensuring reliable coordination in concurrent scenarios.

**Args:**

* `transport`:
  Connection source specification, which can be:

  * ClientTransport: Direct transport instance
  * FastMCP: In-process FastMCP server
  * AnyUrl or str: URL to connect to
  * Path: File path for local socket
  * MCPConfig: MCP server configuration
  * dict: Transport configuration
* `roots`: Optional RootsList or RootsHandler for filesystem access
* `sampling_handler`: Optional handler for sampling requests
* `log_handler`: Optional handler for log messages
* `message_handler`: Optional handler for protocol messages
* `progress_handler`: Optional handler for progress notifications
* `timeout`: Optional timeout for requests (seconds or timedelta)
* `init_timeout`: Optional timeout for initial connection (seconds or timedelta).
  Set to 0 to disable. If None, uses the value in the FastMCP global settings.

**Examples:**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Connect to FastMCP server
client = Client("http://localhost:8080")

async with client:
    # List available resources
    resources = await client.list_resources()

    # Call a tool
    result = await client.call_tool("my_tool", {"param": "value"})
```

**Methods:**

#### `session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L340"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
session(self) -> ClientSession
```

Get the current active session. Raises RuntimeError if not connected.

#### `initialize_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L350"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
initialize_result(self) -> mcp.types.InitializeResult | None
```

Get the result of the initialization request.

#### `set_roots` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L354"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_roots(self, roots: RootsList | RootsHandler) -> None
```

Set the roots for the client. This does not automatically call `send_roots_list_changed`.

#### `set_sampling_callback` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L358"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_sampling_callback(self, sampling_callback: SamplingHandler, sampling_capabilities: mcp.types.SamplingCapability | None = None) -> None
```

Set the sampling callback for the client.

#### `set_elicitation_callback` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L373"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_elicitation_callback(self, elicitation_callback: ElicitationHandler) -> None
```

Set the elicitation callback for the client.

#### `is_connected` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L381"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
is_connected(self) -> bool
```

Check if the client is currently connected.

#### `new` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L385"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
new(self) -> Client[ClientTransportT]
```

Create a new client instance with the same configuration but fresh session state.

This creates a new client with the same transport, handlers, and configuration,
but with no active session. Useful for creating independent sessions that don't
share state with the original client.

**Returns:**

* A new Client instance with the same configuration but disconnected state.

#### `initialize` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L430"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
initialize(self, timeout: datetime.timedelta | float | int | None = None) -> mcp.types.InitializeResult
```

Send an initialize request to the server.

This method performs the MCP initialization handshake with the server,
exchanging capabilities and server information. It is idempotent - calling
it multiple times returns the cached result from the first call.

The initialization happens automatically when entering the client context
manager unless `auto_initialize=False` was set during client construction.
Manual calls to this method are only needed when auto-initialization is disabled.

**Args:**

* `timeout`: Optional timeout for the initialization request (seconds or timedelta).
  If None, uses the client's init\_timeout setting.

**Returns:**

* The server's initialization response containing server info,
  capabilities, protocol version, and optional instructions.

**Raises:**

* `RuntimeError`: If the client is not connected or initialization times out.

#### `close` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L731"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
close(self)
```

#### `ping` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L737"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
ping(self) -> bool
```

Send a ping request.

#### `cancel` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L742"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
cancel(self, request_id: str | int, reason: str | None = None) -> None
```

Send a cancellation notification for an in-progress request.

#### `progress` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L759"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
progress(self, progress_token: str | int, progress: float, total: float | None = None, message: str | None = None) -> None
```

Send a progress notification.

#### `set_logging_level` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L771"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_logging_level(self, level: mcp.types.LoggingLevel) -> None
```

Send a logging/setLevel request.

#### `send_roots_list_changed` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L775"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
send_roots_list_changed(self) -> None
```

Send a roots/list\_changed notification.

#### `complete_mcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L781"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
complete_mcp(self, ref: mcp.types.ResourceTemplateReference | mcp.types.PromptReference, argument: dict[str, str], context_arguments: dict[str, Any] | None = None) -> mcp.types.CompleteResult
```

Send a completion request and return the complete MCP protocol result.

**Args:**

* `ref`: The reference to complete.
* `argument`: Arguments to pass to the completion request.
* `context_arguments`: Optional context arguments to
  include with the completion request. Defaults to None.

**Returns:**

* mcp.types.CompleteResult: The complete response object from the protocol,
  containing the completion and any additional metadata.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `complete` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L812"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
complete(self, ref: mcp.types.ResourceTemplateReference | mcp.types.PromptReference, argument: dict[str, str], context_arguments: dict[str, Any] | None = None) -> mcp.types.Completion
```

Send a completion request to the server.

**Args:**

* `ref`: The reference to complete.
* `argument`: Arguments to pass to the completion request.
* `context_arguments`: Optional context arguments to
  include with the completion request. Defaults to None.

**Returns:**

* mcp.types.Completion: The completion object.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `generate_name` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/client.py#L839"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
generate_name(cls, name: str | None = None) -> str
```


# elicitation
Source: https://gofastmcp.com/python-sdk/fastmcp-client-elicitation



# `fastmcp.client.elicitation`

## Functions

### `create_elicitation_callback` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/elicitation.py#L38"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_elicitation_callback(elicitation_handler: ElicitationHandler) -> ElicitationFnT
```

## Classes

### `ElicitResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/elicitation.py#L22"><Icon icon="github" /></a></sup>


# logging
Source: https://gofastmcp.com/python-sdk/fastmcp-client-logging



# `fastmcp.client.logging`

## Functions

### `default_log_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/logging.py#L17"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_log_handler(message: LogMessage) -> None
```

Default handler that properly routes server log messages to appropriate log levels.

### `create_log_callback` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/logging.py#L47"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_log_callback(handler: LogHandler | None = None) -> LoggingFnT
```


# messages
Source: https://gofastmcp.com/python-sdk/fastmcp-client-messages



# `fastmcp.client.messages`

## Classes

### `MessageHandler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L16"><Icon icon="github" /></a></sup>

This class is used to handle MCP messages sent to the client. It is used to handle all messages,
requests, notifications, and exceptions. Users can override any of the hooks

**Methods:**

#### `dispatch` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L30"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
dispatch(self, message: Message) -> None
```

#### `on_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L76"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_message(self, message: Message) -> None
```

#### `on_request` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L79"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_request(self, message: RequestResponder[mcp.types.ServerRequest, mcp.types.ClientResult]) -> None
```

#### `on_ping` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L84"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_ping(self, message: mcp.types.PingRequest) -> None
```

#### `on_list_roots` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L87"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_list_roots(self, message: mcp.types.ListRootsRequest) -> None
```

#### `on_create_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L90"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_create_message(self, message: mcp.types.CreateMessageRequest) -> None
```

#### `on_notification` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L93"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_notification(self, message: mcp.types.ServerNotification) -> None
```

#### `on_exception` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L96"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_exception(self, message: Exception) -> None
```

#### `on_progress` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L99"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_progress(self, message: mcp.types.ProgressNotification) -> None
```

#### `on_logging_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L102"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_logging_message(self, message: mcp.types.LoggingMessageNotification) -> None
```

#### `on_tool_list_changed` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L107"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_tool_list_changed(self, message: mcp.types.ToolListChangedNotification) -> None
```

#### `on_resource_list_changed` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L112"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_resource_list_changed(self, message: mcp.types.ResourceListChangedNotification) -> None
```

#### `on_prompt_list_changed` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L117"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_prompt_list_changed(self, message: mcp.types.PromptListChangedNotification) -> None
```

#### `on_resource_updated` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L122"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_resource_updated(self, message: mcp.types.ResourceUpdatedNotification) -> None
```

#### `on_cancelled` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/messages.py#L127"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_cancelled(self, message: mcp.types.CancelledNotification) -> None
```


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-client-mixins-__init__



# `fastmcp.client.mixins`

Client mixins for FastMCP.


# prompts
Source: https://gofastmcp.com/python-sdk/fastmcp-client-mixins-prompts



# `fastmcp.client.mixins.prompts`

Prompt-related methods for FastMCP Client.

## Classes

### `ClientPromptsMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/prompts.py#L29"><Icon icon="github" /></a></sup>

Mixin providing prompt-related methods for Client.

**Methods:**

#### `list_prompts_mcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/prompts.py#L34"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts_mcp(self: Client) -> mcp.types.ListPromptsResult
```

Send a prompts/list request and return the complete MCP protocol result.

**Args:**

* `cursor`: Optional pagination cursor from a previous request's nextCursor.

**Returns:**

* mcp.types.ListPromptsResult: The complete response object from the protocol,
  containing the list of prompts and any additional metadata.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/prompts.py#L57"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self: Client) -> list[mcp.types.Prompt]
```

Retrieve all prompts available on the server.

This method automatically fetches all pages if the server paginates results,
returning the complete list. For manual pagination control (e.g., to handle
large result sets incrementally), use list\_prompts\_mcp() with the cursor parameter.

**Returns:**

* list\[mcp.types.Prompt]: A list of all Prompt objects.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `get_prompt_mcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/prompts.py#L92"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt_mcp(self: Client, name: str, arguments: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> mcp.types.GetPromptResult
```

Send a prompts/get request and return the complete MCP protocol result.

**Args:**

* `name`: The name of the prompt to retrieve.
* `arguments`: Arguments to pass to the prompt. Defaults to None.
* `meta`: Request metadata (e.g., for SEP-1686 tasks). Defaults to None.

**Returns:**

* mcp.types.GetPromptResult: The complete response object from the protocol,
  containing the prompt messages and any additional metadata.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/prompts.py#L161"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self: Client, name: str, arguments: dict[str, Any] | None = None) -> mcp.types.GetPromptResult
```

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/prompts.py#L172"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self: Client, name: str, arguments: dict[str, Any] | None = None) -> PromptTask
```

#### `get_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/prompts.py#L184"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt(self: Client, name: str, arguments: dict[str, Any] | None = None) -> mcp.types.GetPromptResult | PromptTask
```

Retrieve a rendered prompt message list from the server.

**Args:**

* `name`: The name of the prompt to retrieve.
* `arguments`: Arguments to pass to the prompt. Defaults to None.
* `version`: Specific prompt version to get. If None, gets highest version.
* `meta`: Optional request-level metadata.
* `task`: If True, execute as background task (SEP-1686). Defaults to False.
* `task_id`: Optional client-provided task ID (auto-generated if not provided).
* `ttl`: Time to keep results available in milliseconds (default 60s).

**Returns:**

* mcp.types.GetPromptResult | PromptTask: The complete response object if task=False,
  or a PromptTask object if task=True.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError


# resources
Source: https://gofastmcp.com/python-sdk/fastmcp-client-mixins-resources



# `fastmcp.client.mixins.resources`

Resource-related methods for FastMCP Client.

## Classes

### `ClientResourcesMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/resources.py#L28"><Icon icon="github" /></a></sup>

Mixin providing resource-related methods for Client.

**Methods:**

#### `list_resources_mcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/resources.py#L33"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources_mcp(self: Client) -> mcp.types.ListResourcesResult
```

Send a resources/list request and return the complete MCP protocol result.

**Args:**

* `cursor`: Optional pagination cursor from a previous request's nextCursor.

**Returns:**

* mcp.types.ListResourcesResult: The complete response object from the protocol,
  containing the list of resources and any additional metadata.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/resources.py#L56"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self: Client) -> list[mcp.types.Resource]
```

Retrieve all resources available on the server.

This method automatically fetches all pages if the server paginates results,
returning the complete list. For manual pagination control (e.g., to handle
large result sets incrementally), use list\_resources\_mcp() with the cursor parameter.

**Returns:**

* list\[mcp.types.Resource]: A list of all Resource objects.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `list_resource_templates_mcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/resources.py#L90"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates_mcp(self: Client) -> mcp.types.ListResourceTemplatesResult
```

Send a resources/listResourceTemplates request and return the complete MCP protocol result.

**Args:**

* `cursor`: Optional pagination cursor from a previous request's nextCursor.

**Returns:**

* mcp.types.ListResourceTemplatesResult: The complete response object from the protocol,
  containing the list of resource templates and any additional metadata.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/resources.py#L113"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates(self: Client) -> list[mcp.types.ResourceTemplate]
```

Retrieve all resource templates available on the server.

This method automatically fetches all pages if the server paginates results,
returning the complete list. For manual pagination control (e.g., to handle
large result sets incrementally), use list\_resource\_templates\_mcp() with the
cursor parameter.

**Returns:**

* list\[mcp.types.ResourceTemplate]: A list of all ResourceTemplate objects.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `read_resource_mcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/resources.py#L149"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource_mcp(self: Client, uri: AnyUrl | str, meta: dict[str, Any] | None = None) -> mcp.types.ReadResourceResult
```

Send a resources/read request and return the complete MCP protocol result.

**Args:**

* `uri`: The URI of the resource to read. Can be a string or an AnyUrl object.
* `meta`: Request metadata (e.g., for SEP-1686 tasks). Defaults to None.

**Returns:**

* mcp.types.ReadResourceResult: The complete response object from the protocol,
  containing the resource contents and any additional metadata.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/resources.py#L205"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource(self: Client, uri: AnyUrl | str) -> list[mcp.types.TextResourceContents | mcp.types.BlobResourceContents]
```

#### `read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/resources.py#L215"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource(self: Client, uri: AnyUrl | str) -> ResourceTask
```

#### `read_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/resources.py#L226"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read_resource(self: Client, uri: AnyUrl | str) -> list[mcp.types.TextResourceContents | mcp.types.BlobResourceContents] | ResourceTask
```

Read the contents of a resource or resolved template.

**Args:**

* `uri`: The URI of the resource to read. Can be a string or an AnyUrl object.
* `version`: Specific version to read. If None, reads highest version.
* `meta`: Optional request-level metadata.
* `task`: If True, execute as background task (SEP-1686). Defaults to False.
* `task_id`: Optional client-provided task ID (auto-generated if not provided).
* `ttl`: Time to keep results available in milliseconds (default 60s).

**Returns:**

* list\[mcp.types.TextResourceContents | mcp.types.BlobResourceContents] | ResourceTask:
  A list of content objects if task=False, or a ResourceTask object if task=True.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError


# task_management
Source: https://gofastmcp.com/python-sdk/fastmcp-client-mixins-task_management



# `fastmcp.client.mixins.task_management`

Task management methods for FastMCP Client.

## Classes

### `ClientTaskManagementMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/task_management.py#L30"><Icon icon="github" /></a></sup>

Mixin providing task management methods for Client.

**Methods:**

#### `get_task_status` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/task_management.py#L33"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_task_status(self: Client, task_id: str) -> GetTaskResult
```

Query the status of a background task.

Sends a 'tasks/get' MCP protocol request over the existing transport.

**Args:**

* `task_id`: The task ID returned from call\_tool\_as\_task

**Returns:**

* Status information including taskId, status, pollInterval, etc.

**Raises:**

* `RuntimeError`: If client not connected
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `get_task_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/task_management.py#L56"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_task_result(self: Client, task_id: str) -> Any
```

Retrieve the raw result of a completed background task.

Sends a 'tasks/result' MCP protocol request over the existing transport.
Returns the raw result - callers should parse it appropriately.

**Args:**

* `task_id`: The task ID returned from call\_tool\_as\_task

**Returns:**

* The raw result (could be tool, prompt, or resource result)

**Raises:**

* `RuntimeError`: If client not connected, task not found, or task failed
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `list_tasks` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/task_management.py#L85"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tasks(self: Client, cursor: str | None = None, limit: int = 50) -> dict[str, Any]
```

List background tasks.

Sends a 'tasks/list' MCP protocol request to the server. If the server
returns an empty list (indicating client-side tracking), falls back to
querying status for locally tracked task IDs.

**Args:**

* `cursor`: Optional pagination cursor
* `limit`: Maximum number of tasks to return (default 50)

**Returns:**

* Response with structure:
* tasks: List of task status dicts with taskId, status, etc.
* nextCursor: Optional cursor for next page

**Raises:**

* `RuntimeError`: If client not connected
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `cancel_task` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/task_management.py#L135"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
cancel_task(self: Client, task_id: str) -> mcp.types.CancelTaskResult
```

Cancel a task, transitioning it to cancelled state.

Sends a 'tasks/cancel' MCP protocol request. Task will halt execution
and transition to cancelled state.

**Args:**

* `task_id`: The task ID to cancel

**Returns:**

* The task status showing cancelled state

**Raises:**

* `RuntimeError`: If task doesn't exist
* `McpError`: If the request results in a TimeoutError | JSONRPCError


# tools
Source: https://gofastmcp.com/python-sdk/fastmcp-client-mixins-tools



# `fastmcp.client.mixins.tools`

Tool-related methods for FastMCP Client.

## Classes

### `ClientToolsMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/tools.py#L32"><Icon icon="github" /></a></sup>

Mixin providing tool-related methods for Client.

**Methods:**

#### `list_tools_mcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/tools.py#L37"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools_mcp(self: Client) -> mcp.types.ListToolsResult
```

Send a tools/list request and return the complete MCP protocol result.

**Args:**

* `cursor`: Optional pagination cursor from a previous request's nextCursor.

**Returns:**

* mcp.types.ListToolsResult: The complete response object from the protocol,
  containing the list of tools and any additional metadata.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/tools.py#L60"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self: Client) -> list[mcp.types.Tool]
```

Retrieve all tools available on the server.

This method automatically fetches all pages if the server paginates results,
returning the complete list. For manual pagination control (e.g., to handle
large result sets incrementally), use list\_tools\_mcp() with the cursor parameter.

**Returns:**

* list\[mcp.types.Tool]: A list of all Tool objects.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the request results in a TimeoutError | JSONRPCError

#### `call_tool_mcp` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/tools.py#L96"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_tool_mcp(self: Client, name: str, arguments: dict[str, Any], progress_handler: ProgressHandler | None = None, timeout: datetime.timedelta | float | int | None = None, meta: dict[str, Any] | None = None) -> mcp.types.CallToolResult
```

Send a tools/call request and return the complete MCP protocol result.

This method returns the raw CallToolResult object, which includes an isError flag
and other metadata. It does not raise an exception if the tool call results in an error.

**Args:**

* `name`: The name of the tool to call.
* `arguments`: Arguments to pass to the tool.
* `timeout`: The timeout for the tool call. Defaults to None.
* `progress_handler`: The progress handler to use for the tool call. Defaults to None.
* `meta`: Additional metadata to include with the request.
  This is useful for passing contextual information (like user IDs, trace IDs, or preferences)
  that shouldn't be tool arguments but may influence server-side processing. The server
  can access this via `context.request_context.meta`. Defaults to None.

**Returns:**

* mcp.types.CallToolResult: The complete response object from the protocol,
  containing the tool result and any additional metadata.

**Raises:**

* `RuntimeError`: If called while the client is not connected.
* `McpError`: If the tool call requests results in a TimeoutError | JSONRPCError

#### `call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/tools.py#L176"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_tool(self: Client, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult
```

#### `call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/tools.py#L190"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_tool(self: Client, name: str, arguments: dict[str, Any] | None = None) -> ToolTask
```

#### `call_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/mixins/tools.py#L205"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_tool(self: Client, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult | ToolTask
```

Call a tool on the server.

Unlike call\_tool\_mcp, this method raises a ToolError if the tool call results in an error.

**Args:**

* `name`: The name of the tool to call.
* `arguments`: Arguments to pass to the tool. Defaults to None.
* `version`: Specific tool version to call. If None, calls highest version.
* `timeout`: The timeout for the tool call. Defaults to None.
* `progress_handler`: The progress handler to use for the tool call. Defaults to None.
* `raise_on_error`: Whether to raise an exception if the tool call results in an error. Defaults to True.
* `meta`: Additional metadata to include with the request.
  This is useful for passing contextual information (like user IDs, trace IDs, or preferences)
  that shouldn't be tool arguments but may influence server-side processing. The server
  can access this via `context.request_context.meta`. Defaults to None.
* `task`: If True, execute as background task (SEP-1686). Defaults to False.
* `task_id`: Optional client-provided task ID (auto-generated if not provided).
* `ttl`: Time to keep results available in milliseconds (default 60s).

**Returns:**

* CallToolResult | ToolTask: The content returned by the tool if task=False,
  or a ToolTask object if task=True. If the tool returns structured
  outputs, they are returned as a dataclass (if an output schema
  is available) or a dictionary; otherwise, a list of content
  blocks is returned. Note: to receive both structured and
  unstructured outputs, use call\_tool\_mcp instead and access the
  raw result object.

**Raises:**

* `ToolError`: If the tool call results in an error.
* `McpError`: If the tool call request results in a TimeoutError | JSONRPCError
* `RuntimeError`: If called while the client is not connected.


# oauth_callback
Source: https://gofastmcp.com/python-sdk/fastmcp-client-oauth_callback



# `fastmcp.client.oauth_callback`

OAuth callback server for handling authorization code flows.

This module provides a reusable callback server that can handle OAuth redirects
and display styled responses to users.

## Functions

### `create_callback_html` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/oauth_callback.py#L34"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_callback_html(message: str, is_success: bool = True, title: str = 'FastMCP OAuth', server_url: str | None = None) -> str
```

Create a styled HTML response for OAuth callbacks.

### `create_oauth_callback_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/oauth_callback.py#L103"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_oauth_callback_server(port: int, callback_path: str = '/callback', server_url: str | None = None, result_container: OAuthCallbackResult | None = None, result_ready: anyio.Event | None = None) -> Server
```

Create an OAuth callback server.

**Args:**

* `port`: The port to run the server on
* `callback_path`: The path to listen for OAuth redirects on
* `server_url`: Optional server URL to display in success messages
* `result_container`: Optional container to store callback results
* `result_ready`: Optional event to signal when callback is received

**Returns:**

* Configured uvicorn Server instance (not yet running)

## Classes

### `CallbackResponse` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/oauth_callback.py#L80"><Icon icon="github" /></a></sup>

**Methods:**

#### `from_dict` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/oauth_callback.py#L87"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_dict(cls, data: dict[str, str]) -> CallbackResponse
```

#### `to_dict` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/oauth_callback.py#L90"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_dict(self) -> dict[str, str]
```

### `OAuthCallbackResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/oauth_callback.py#L95"><Icon icon="github" /></a></sup>

Container for OAuth callback results, used with anyio.Event for async coordination.


# progress
Source: https://gofastmcp.com/python-sdk/fastmcp-client-progress



# `fastmcp.client.progress`

## Functions

### `default_progress_handler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/progress.py#L12"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_progress_handler(progress: float, total: float | None, message: str | None) -> None
```

Default handler for progress notifications.

Logs progress updates at debug level, properly handling missing total or message values.

**Args:**

* `progress`: Current progress value
* `total`: Optional total expected value
* `message`: Optional status message


# roots
Source: https://gofastmcp.com/python-sdk/fastmcp-client-roots



# `fastmcp.client.roots`

## Functions

### `convert_roots_list` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/roots.py#L19"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
convert_roots_list(roots: RootsList) -> list[mcp.types.Root]
```

### `create_roots_callback` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/roots.py#L33"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_roots_callback(handler: RootsList | RootsHandler) -> ListRootsFnT
```


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-client-sampling-__init__



# `fastmcp.client.sampling`

## Functions

### `create_sampling_callback` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/sampling/__init__.py#L44"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_sampling_callback(sampling_handler: SamplingHandler) -> SamplingFnT
```


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-client-sampling-handlers-__init__



# `fastmcp.client.sampling.handlers`

*This module is empty or contains only private/internal implementations.*


# anthropic
Source: https://gofastmcp.com/python-sdk/fastmcp-client-sampling-handlers-anthropic



# `fastmcp.client.sampling.handlers.anthropic`

Anthropic sampling handler for FastMCP.

## Classes

### `AnthropicSamplingHandler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/sampling/handlers/anthropic.py#L46"><Icon icon="github" /></a></sup>

Sampling handler that uses the Anthropic API.


# google_genai
Source: https://gofastmcp.com/python-sdk/fastmcp-client-sampling-handlers-google_genai



# `fastmcp.client.sampling.handlers.google_genai`

Google GenAI sampling handler with tool support for FastMCP 3.0.

## Classes

### `GoogleGenaiSamplingHandler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/sampling/handlers/google_genai.py#L54"><Icon icon="github" /></a></sup>

Sampling handler that uses the Google GenAI API with tool support.


# openai
Source: https://gofastmcp.com/python-sdk/fastmcp-client-sampling-handlers-openai



# `fastmcp.client.sampling.handlers.openai`

OpenAI sampling handler for FastMCP.

## Classes

### `OpenAISamplingHandler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/sampling/handlers/openai.py#L45"><Icon icon="github" /></a></sup>

Sampling handler that uses the OpenAI API.


# tasks
Source: https://gofastmcp.com/python-sdk/fastmcp-client-tasks



# `fastmcp.client.tasks`

SEP-1686 client Task classes.

## Classes

### `TaskNotificationHandler` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L26"><Icon icon="github" /></a></sup>

MessageHandler that routes task status notifications to Task objects.

**Methods:**

#### `dispatch` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L33"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
dispatch(self, message: Message) -> None
```

Dispatch messages, including task status notifications.

### `Task` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L47"><Icon icon="github" /></a></sup>

Abstract base class for MCP background tasks (SEP-1686).

Provides a uniform API whether the server accepts background execution
or executes synchronously (graceful degradation per SEP-1686).

**Methods:**

#### `task_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L105"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
task_id(self) -> str
```

Get the task ID.

#### `returned_immediately` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L110"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
returned_immediately(self) -> bool
```

Check if server executed the task immediately.

**Returns:**

* True if server executed synchronously (graceful degradation or no task support)
* False if server accepted background execution

#### `on_status_change` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L145"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
on_status_change(self, callback: Callable[[GetTaskResult], None | Awaitable[None]]) -> None
```

Register callback for status change notifications.

The callback will be invoked when a notifications/tasks/status is received
for this task (optional server feature per SEP-1686 lines 436-444).

Supports both sync and async callbacks (auto-detected).

**Args:**

* `callback`: Function to call with GetTaskResult when status changes.
  Can return None (sync) or Awaitable\[None] (async).

#### `status` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L171"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
status(self) -> GetTaskResult
```

Get current task status.

If server executed immediately, returns synthetic completed status.
Otherwise queries the server for current status.

#### `result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L202"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
result(self) -> TaskResultT
```

Wait for and return the task result.

Must be implemented by subclasses to return the appropriate result type.

#### `wait` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L209"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wait(self) -> GetTaskResult
```

Wait for task to reach a specific state or complete.

Uses event-based waiting when notifications are available (fast),
with fallback to polling (reliable). Optimally wakes up immediately
on status changes when server sends notifications/tasks/status.

**Args:**

* `state`: Desired state ('submitted', 'working', 'completed', 'failed').
  If None, waits for any terminal state (completed/failed)
* `timeout`: Maximum time to wait in seconds

**Returns:**

* Final task status

**Raises:**

* `TimeoutError`: If desired state not reached within timeout

#### `cancel` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L272"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
cancel(self) -> None
```

Cancel this task, transitioning it to cancelled state.

Sends a tasks/cancel protocol request. The server will attempt to halt
execution and move the task to cancelled state.

Note: If server executed immediately (graceful degradation), this is a no-op
as there's no server-side task to cancel.

### `ToolTask` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L294"><Icon icon="github" /></a></sup>

Represents a tool call that may execute in background or immediately.

Provides a uniform API whether the server accepts background execution
or executes synchronously (graceful degradation per SEP-1686).

**Methods:**

#### `result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L336"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
result(self) -> CallToolResult
```

Wait for and return the tool result.

If server executed immediately, returns the immediate result.
Otherwise waits for background task to complete and retrieves result.

**Returns:**

* The parsed tool result (same as call\_tool returns)

### `PromptTask` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L396"><Icon icon="github" /></a></sup>

Represents a prompt call that may execute in background or immediately.

Provides a uniform API whether the server accepts background execution
or executes synchronously (graceful degradation per SEP-1686).

**Methods:**

#### `result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L427"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
result(self) -> mcp.types.GetPromptResult
```

Wait for and return the prompt result.

If server executed immediately, returns the immediate result.
Otherwise waits for background task to complete and retrieves result.

**Returns:**

* The prompt result with messages and description

### `ResourceTask` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L461"><Icon icon="github" /></a></sup>

Represents a resource read that may execute in background or immediately.

Provides a uniform API whether the server accepts background execution
or executes synchronously (graceful degradation per SEP-1686).

**Methods:**

#### `result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/tasks.py#L497"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
result(self) -> list[mcp.types.TextResourceContents | mcp.types.BlobResourceContents]
```

Wait for and return the resource contents.

If server executed immediately, returns the immediate result.
Otherwise waits for background task to complete and retrieves result.

**Returns:**

* list\[ReadResourceContents]: The resource contents


# telemetry
Source: https://gofastmcp.com/python-sdk/fastmcp-client-telemetry



# `fastmcp.client.telemetry`

Client-side telemetry helpers.

## Functions

### `client_span` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/telemetry.py#L12"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
client_span(name: str, method: str, component_key: str, session_id: str | None = None, resource_uri: str | None = None) -> Generator[Span, None, None]
```

Create a CLIENT span with standard MCP attributes.

Automatically records any exception on the span and sets error status.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-client-transports-__init__



# `fastmcp.client.transports`

*This module is empty or contains only private/internal implementations.*


# base
Source: https://gofastmcp.com/python-sdk/fastmcp-client-transports-base



# `fastmcp.client.transports.base`

## Classes

### `SessionKwargs` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/base.py#L23"><Icon icon="github" /></a></sup>

Keyword arguments for the MCP ClientSession constructor.

### `ClientTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/base.py#L36"><Icon icon="github" /></a></sup>

Abstract base class for different MCP client transport mechanisms.

A Transport is responsible for establishing and managing connections
to an MCP server, and providing a ClientSession within an async context.

**Methods:**

#### `connect_session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/base.py#L47"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
connect_session(self, **session_kwargs: Unpack[SessionKwargs]) -> AsyncIterator[ClientSession]
```

Establishes a connection and yields an active ClientSession.

The ClientSession is *not* expected to be initialized in this context manager.

The session is guaranteed to be valid only within the scope of the
async context manager. Connection setup and teardown are handled
within this context.

**Args:**

* `**session_kwargs`: Keyword arguments to pass to the ClientSession
  constructor (e.g., callbacks, timeouts).

#### `close` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/base.py#L73"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
close(self)
```

Close the transport.

#### `get_session_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/base.py#L76"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_session_id(self) -> str | None
```

Get the session ID for this transport, if available.


# config
Source: https://gofastmcp.com/python-sdk/fastmcp-client-transports-config



# `fastmcp.client.transports.config`

## Classes

### `MCPConfigTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/config.py#L22"><Icon icon="github" /></a></sup>

Transport for connecting to one or more MCP servers defined in an MCPConfig.

This transport provides a unified interface to multiple MCP servers defined in an MCPConfig
object or dictionary matching the MCPConfig schema. It supports two key scenarios:

1. If the MCPConfig contains exactly one server, it creates a direct transport to that server.
2. If the MCPConfig contains multiple servers, it creates a composite client by mounting
   all servers on a single FastMCP instance, with each server's name, by default, used as its mounting prefix.

In the multiserver case, tools are accessible with the prefix pattern `{server_name}_{tool_name}`
and resources with the pattern `protocol://{server_name}/path/to/resource`.

This is particularly useful for creating clients that need to interact with multiple specialized
MCP servers through a single interface, simplifying client code.

**Examples:**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Client

# Create a config with multiple servers
config = {
    "mcpServers": {
        "weather": {
            "url": "https://weather-api.example.com/mcp",
            "transport": "http"
        },
        "calendar": {
            "url": "https://calendar-api.example.com/mcp",
            "transport": "http"
        }
    }
}

# Create a client with the config
client = Client(config)

async with client:
    # Access tools with prefixes
    weather = await client.call_tool("weather_get_forecast", {"city": "London"})
    events = await client.call_tool("calendar_list_events", {"date": "2023-06-01"})

    # Access resources with prefixed URIs
    icons = await client.read_resource("weather://weather/icons/sunny")
```

**Methods:**

#### `connect_session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/config.py#L85"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
connect_session(self, **session_kwargs: Unpack[SessionKwargs]) -> AsyncIterator[ClientSession]
```

#### `close` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/config.py#L198"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
close(self)
```


# http
Source: https://gofastmcp.com/python-sdk/fastmcp-client-transports-http



# `fastmcp.client.transports.http`

Streamable HTTP transport for FastMCP Client.

## Classes

### `StreamableHttpTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/http.py#L25"><Icon icon="github" /></a></sup>

Transport implementation that connects to an MCP server via Streamable HTTP Requests.

**Methods:**

#### `connect_session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/http.py#L92"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
connect_session(self, **session_kwargs: Unpack[SessionKwargs]) -> AsyncIterator[ClientSession]
```

#### `get_session_id` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/http.py#L138"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_session_id(self) -> str | None
```

#### `close` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/http.py#L146"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
close(self)
```


# inference
Source: https://gofastmcp.com/python-sdk/fastmcp-client-transports-inference



# `fastmcp.client.transports.inference`

## Functions

### `infer_transport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/inference.py#L61"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
infer_transport(transport: ClientTransport | FastMCP | FastMCP1Server | AnyUrl | Path | MCPConfig | dict[str, Any] | str) -> ClientTransport
```

Infer the appropriate transport type from the given transport argument.

This function attempts to infer the correct transport type from the provided
argument, handling various input types and converting them to the appropriate
ClientTransport subclass.

The function supports these input types:

* ClientTransport: Used directly without modification
* FastMCP or FastMCP1Server: Creates an in-memory FastMCPTransport
* Path or str (file path): Creates PythonStdioTransport (.py) or NodeStdioTransport (.js)
* AnyUrl or str (URL): Creates StreamableHttpTransport (default) or SSETransport (for /sse endpoints)
* MCPConfig or dict: Creates MCPConfigTransport, potentially connecting to multiple servers

For HTTP URLs, they are assumed to be Streamable HTTP URLs unless they end in `/sse`.

For MCPConfig with multiple servers, a composite client is created where each server
is mounted with its name as prefix. This allows accessing tools and resources from multiple
servers through a single unified client interface, using naming patterns like
`servername_toolname` for tools and `protocol://servername/path` for resources.
If the MCPConfig contains only one server, a direct connection is established without prefixing.

**Examples:**

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
# Connect to a local Python script
transport = infer_transport("my_script.py")

# Connect to a remote server via HTTP
transport = infer_transport("http://example.com/mcp")

# Connect to multiple servers using MCPConfig
config = {
    "mcpServers": {
        "weather": {"url": "http://weather.example.com/mcp"},
        "calendar": {"url": "http://calendar.example.com/mcp"}
    }
}
transport = infer_transport(config)
```


# memory
Source: https://gofastmcp.com/python-sdk/fastmcp-client-transports-memory



# `fastmcp.client.transports.memory`

## Classes

### `FastMCPTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/memory.py#L14"><Icon icon="github" /></a></sup>

In-memory transport for FastMCP servers.

This transport connects directly to a FastMCP server instance in the same
Python process. It works with both FastMCP 2.x servers and FastMCP 1.0
servers from the low-level MCP SDK. This is particularly useful for unit
tests or scenarios where client and server run in the same runtime.

**Methods:**

#### `connect_session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/memory.py#L33"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
connect_session(self, **session_kwargs: Unpack[SessionKwargs]) -> AsyncIterator[ClientSession]
```


# sse
Source: https://gofastmcp.com/python-sdk/fastmcp-client-transports-sse



# `fastmcp.client.transports.sse`

Server-Sent Events (SSE) transport for FastMCP Client.

## Classes

### `SSETransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/sse.py#L24"><Icon icon="github" /></a></sup>

Transport implementation that connects to an MCP server via Server-Sent Events.

**Methods:**

#### `connect_session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/sse.py#L64"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
connect_session(self, **session_kwargs: Unpack[SessionKwargs]) -> AsyncIterator[ClientSession]
```


# stdio
Source: https://gofastmcp.com/python-sdk/fastmcp-client-transports-stdio



# `fastmcp.client.transports.stdio`

## Classes

### `StdioTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L22"><Icon icon="github" /></a></sup>

Base transport for connecting to an MCP server via subprocess with stdio.

This is a base class that can be subclassed for specific command-based
transports like Python, Node, Uvx, etc.

**Methods:**

#### `connect_session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L72"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
connect_session(self, **session_kwargs: Unpack[SessionKwargs]) -> AsyncIterator[ClientSession]
```

#### `connect` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L84"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
connect(self, **session_kwargs: Unpack[SessionKwargs]) -> ClientSession | None
```

#### `disconnect` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L120"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
disconnect(self)
```

#### `close` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L135"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
close(self)
```

### `PythonStdioTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L205"><Icon icon="github" /></a></sup>

Transport for running Python scripts.

### `FastMCPStdioTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L258"><Icon icon="github" /></a></sup>

Transport for running FastMCP servers using the FastMCP CLI.

### `NodeStdioTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L287"><Icon icon="github" /></a></sup>

Transport for running Node.js scripts.

### `UvStdioTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L340"><Icon icon="github" /></a></sup>

Transport for running commands via the uv tool.

### `UvxStdioTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L419"><Icon icon="github" /></a></sup>

Transport for running commands via the uvx tool.

### `NpxStdioTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/stdio.py#L484"><Icon icon="github" /></a></sup>

Transport for running commands via the npx tool.


# decorators
Source: https://gofastmcp.com/python-sdk/fastmcp-decorators



# `fastmcp.decorators`

Shared decorator utilities for FastMCP.

## Functions

### `resolve_task_config` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/decorators.py#L17"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resolve_task_config(task: bool | TaskConfig | None) -> bool | TaskConfig
```

Resolve task config, defaulting None to False.

### `get_fastmcp_meta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/decorators.py#L29"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_fastmcp_meta(fn: Any) -> Any | None
```

Extract FastMCP metadata from a function, handling bound methods and wrappers.

## Classes

### `HasFastMCPMeta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/decorators.py#L23"><Icon icon="github" /></a></sup>

Protocol for callables decorated with FastMCP metadata.


# dependencies
Source: https://gofastmcp.com/python-sdk/fastmcp-dependencies



# `fastmcp.dependencies`

Dependency injection exports for FastMCP.

This module re-exports dependency injection symbols to provide a clean,
centralized import location for all dependency-related functionality.

DI features (Depends, CurrentContext, CurrentFastMCP) work without pydocket
using the uncalled-for DI engine. Only task-related dependencies (CurrentDocket,
CurrentWorker) and background task execution require fastmcp\[tasks].


# exceptions
Source: https://gofastmcp.com/python-sdk/fastmcp-exceptions



# `fastmcp.exceptions`

Custom exceptions for FastMCP.

## Classes

### `FastMCPError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L6"><Icon icon="github" /></a></sup>

Base error for FastMCP.

### `ValidationError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L10"><Icon icon="github" /></a></sup>

Error in validating parameters or return values.

### `ResourceError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L14"><Icon icon="github" /></a></sup>

Error in resource operations.

### `ToolError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L18"><Icon icon="github" /></a></sup>

Error in tool operations.

### `PromptError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L22"><Icon icon="github" /></a></sup>

Error in prompt operations.

### `InvalidSignature` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L26"><Icon icon="github" /></a></sup>

Invalid signature for use with FastMCP.

### `ClientError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L30"><Icon icon="github" /></a></sup>

Error in client operations.

### `NotFoundError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L34"><Icon icon="github" /></a></sup>

Object not found.

### `DisabledError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L38"><Icon icon="github" /></a></sup>

Object is disabled.

### `AuthorizationError` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/exceptions.py#L42"><Icon icon="github" /></a></sup>

Error when authorization check fails.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-experimental-__init__



# `fastmcp.experimental`

*This module is empty or contains only private/internal implementations.*


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-experimental-sampling-__init__



# `fastmcp.experimental.sampling`

*This module is empty or contains only private/internal implementations.*


# handlers
Source: https://gofastmcp.com/python-sdk/fastmcp-experimental-sampling-handlers



# `fastmcp.experimental.sampling.handlers`

*This module is empty or contains only private/internal implementations.*


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-experimental-transforms-__init__



# `fastmcp.experimental.transforms`

*This module is empty or contains only private/internal implementations.*


# code_mode
Source: https://gofastmcp.com/python-sdk/fastmcp-experimental-transforms-code_mode



# `fastmcp.experimental.transforms.code_mode`

## Classes

### `SandboxProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L73"><Icon icon="github" /></a></sup>

Interface for executing LLM-generated Python code in a sandbox.

WARNING: The `code` parameter passed to `run` contains untrusted,
LLM-generated Python.  Implementations MUST execute it in an isolated
sandbox — never with plain `exec()`.  Use `MontySandboxProvider`
(backed by `pydantic-monty`) for production workloads.

**Methods:**

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L82"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, code: str) -> Any
```

### `MontySandboxProvider` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L91"><Icon icon="github" /></a></sup>

Sandbox provider backed by `pydantic-monty`.

**Args:**

* `limits`: Resource limits for sandbox execution. Supported keys:
  `max_duration_secs` (float), `max_allocations` (int),
  `max_memory` (int), `max_recursion_depth` (int),
  `gc_interval` (int).  All are optional; omit a key to
  leave that limit uncapped.

**Methods:**

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L109"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, code: str) -> Any
```

### `Search` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L180"><Icon icon="github" /></a></sup>

Discovery tool factory that searches the catalog by query.

**Args:**

* `search_fn`: Async callable `(tools, query) -> matching_tools`.
  Defaults to BM25 ranking.
* `name`: Name of the synthetic tool exposed to the LLM.
* `default_detail`: Default detail level for search results.
  `"brief"` returns tool names and descriptions only.
  `"detailed"` returns compact markdown with parameter schemas.
  `"full"` returns complete JSON tool definitions.
* `default_limit`: Maximum number of results to return.
  The LLM can override this per call.  `None` means no limit.

### `GetSchemas` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L262"><Icon icon="github" /></a></sup>

Discovery tool factory that returns schemas for tools by name.

**Args:**

* `name`: Name of the synthetic tool exposed to the LLM.
* `default_detail`: Default detail level for schema results.
  `"brief"` returns tool names and descriptions only.
  `"detailed"` renders compact markdown with parameter names,
  types, and required markers.
  `"full"` returns the complete JSON schema.

### `GetTags` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L323"><Icon icon="github" /></a></sup>

Discovery tool factory that lists tool tags from the catalog.

Reads `tool.tags` from the catalog and groups tools by tag. Tools
without tags appear under `"untagged"`.

**Args:**

* `name`: Name of the synthetic tool exposed to the LLM.
* `default_detail`: Default detail level.
  `"brief"` returns tag names with tool counts.
  `"full"` lists all tools under each tag.

### `ListTools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L390"><Icon icon="github" /></a></sup>

Discovery tool factory that lists all tools in the catalog.

**Args:**

* `name`: Name of the synthetic tool exposed to the LLM.
* `default_detail`: Default detail level.
  `"brief"` returns tool names and one-line descriptions.
  `"detailed"` returns compact markdown with parameter schemas.
  `"full"` returns the complete JSON schema.

### `CodeMode` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L439"><Icon icon="github" /></a></sup>

Transform that collapses all tools into discovery + execute meta-tools.

Discovery tools are composable via the `discovery_tools` parameter.
Each is a callable that receives catalog access and returns a `Tool`.
By default, `Search` and `GetSchemas` are included for
progressive disclosure: search finds candidates, get\_schema retrieves
parameter details, and execute runs code.

The `execute` tool is always present and provides a sandboxed Python
environment with `call_tool(name, params)` in scope.

**Methods:**

#### `transform_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L489"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

#### `get_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/experimental/transforms/code_mode.py#L492"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool(self, name: str, call_next: GetToolNext) -> Tool | None
```


# mcp_config
Source: https://gofastmcp.com/python-sdk/fastmcp-mcp_config



# `fastmcp.mcp_config`

Canonical MCP Configuration Format.

This module defines the standard configuration format for Model Context Protocol (MCP) servers.
It provides a client-agnostic, extensible format that can be used across all MCP implementations.

The configuration format supports both stdio and remote (HTTP/SSE) transports, with comprehensive
field definitions for server metadata, authentication, and execution parameters.

Example configuration:

```json theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
{
    "mcpServers": {
        "my-server": {
            "command": "npx",
            "args": ["-y", "@my/mcp-server"],
            "env": {"API_KEY": "secret"},
            "timeout": 30000,
            "description": "My MCP server"
        }
    }
}
```

## Functions

### `infer_transport_type_from_url` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L56"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
infer_transport_type_from_url(url: str | AnyUrl) -> Literal['http', 'sse']
```

Infer the appropriate transport type from the given URL.

### `update_config_file` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L345"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
update_config_file(file_path: Path, server_name: str, server_config: CanonicalMCPServerTypes) -> None
```

Update an MCP configuration file from a server object, preserving existing fields.

This is used for updating the mcpServer configurations of third-party tools so we do not
worry about transforming server objects here.

## Classes

### `StdioMCPServer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L155"><Icon icon="github" /></a></sup>

MCP server configuration for stdio transport.

This is the canonical configuration format for MCP servers using stdio transport.

**Methods:**

#### `to_transport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L188"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_transport(self) -> StdioTransport
```

### `TransformingStdioMCPServer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L200"><Icon icon="github" /></a></sup>

A Stdio server with tool transforms.

### `RemoteMCPServer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L204"><Icon icon="github" /></a></sup>

MCP server configuration for HTTP/SSE transport.

This is the canonical configuration format for MCP servers using remote transports.

**Methods:**

#### `to_transport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L240"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_transport(self) -> StreamableHttpTransport | SSETransport
```

### `TransformingRemoteMCPServer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L265"><Icon icon="github" /></a></sup>

A Remote server with tool transforms.

### `MCPConfig` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L276"><Icon icon="github" /></a></sup>

A configuration object for MCP Servers that conforms to the canonical MCP configuration format
while adding additional fields for enabling FastMCP-specific features like tool transformations
and filtering by tags.

For an MCPConfig that is strictly canonical, see the `CanonicalMCPConfig` class.

**Methods:**

#### `wrap_servers_at_root` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L290"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
wrap_servers_at_root(cls, values: dict[str, Any]) -> dict[str, Any]
```

If there's no mcpServers key but there are server configs at root, wrap them.

#### `add_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L303"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_server(self, name: str, server: MCPServerTypes) -> None
```

Add or update a server in the configuration.

#### `from_dict` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L308"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_dict(cls, config: dict[str, Any]) -> Self
```

Parse MCP configuration from dictionary format.

#### `to_dict` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L312"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_dict(self) -> dict[str, Any]
```

Convert MCPConfig to dictionary format, preserving all fields.

#### `write_to_file` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L316"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
write_to_file(self, file_path: Path) -> None
```

Write configuration to JSON file.

#### `from_file` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L322"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_file(cls, file_path: Path) -> Self
```

Load configuration from JSON file.

### `CanonicalMCPConfig` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L330"><Icon icon="github" /></a></sup>

Canonical MCP configuration format.

This defines the standard configuration format for Model Context Protocol servers.
The format is designed to be client-agnostic and extensible for future use cases.

**Methods:**

#### `add_server` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/mcp_config.py#L340"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_server(self, name: str, server: CanonicalMCPServerTypes) -> None
```

Add or update a server in the configuration.


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-prompts-__init__



# `fastmcp.prompts`

*This module is empty or contains only private/internal implementations.*


# function_prompt
Source: https://gofastmcp.com/python-sdk/fastmcp-prompts-function_prompt



# `fastmcp.prompts.function_prompt`

Standalone @prompt decorator for FastMCP.

## Functions

### `prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/function_prompt.py#L398"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
prompt(name_or_fn: str | Callable[..., Any] | None = None) -> Any
```

Standalone decorator to mark a function as an MCP prompt.

Returns the original function with metadata attached. Register with a server
using mcp.add\_prompt().

## Classes

### `DecoratedPrompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/function_prompt.py#L49"><Icon icon="github" /></a></sup>

Protocol for functions decorated with @prompt.

### `PromptMeta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/function_prompt.py#L58"><Icon icon="github" /></a></sup>

Metadata attached to functions by the @prompt decorator.

### `FunctionPrompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/function_prompt.py#L74"><Icon icon="github" /></a></sup>

A prompt that is a function.

**Methods:**

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/function_prompt.py#L80"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any]) -> FunctionPrompt
```

Create a Prompt from a function.

**Args:**

* `fn`: The function to wrap
* `metadata`: PromptMeta object with all configuration. If provided,
  individual parameters must not be passed.
* `name, title, etc.`: Individual parameters for backwards compatibility.
  Cannot be used together with metadata parameter.

The function can return:

* str: wrapped as single user Message
* list\[Message | str]: converted to list\[Message]
* PromptResult: used directly

#### `render` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/function_prompt.py#L281"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
render(self, arguments: dict[str, Any] | None = None) -> PromptResult
```

Render the prompt with arguments.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/function_prompt.py#L331"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this prompt with docket for background execution.

FunctionPrompt registers the underlying function, which has the user's
Depends parameters for docket to resolve.

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/function_prompt.py#L341"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, arguments: dict[str, Any] | None, **kwargs: Any) -> Execution
```

Schedule this prompt for background execution via docket.

FunctionPrompt splats the arguments dict since .fn expects \*\*kwargs.

**Args:**

* `docket`: The Docket instance
* `arguments`: Prompt arguments
* `fn_key`: Function lookup key in Docket registry (defaults to self.key)
* `task_key`: Redis storage key for the result
* `**kwargs`: Additional kwargs passed to docket.add()


# prompt
Source: https://gofastmcp.com/python-sdk/fastmcp-prompts-prompt



# `fastmcp.prompts.prompt`

Base classes for FastMCP prompts.

## Classes

### `Message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L41"><Icon icon="github" /></a></sup>

Wrapper for prompt message with auto-serialization.

Accepts any content - strings pass through, other types
(dict, list, BaseModel) are JSON-serialized to text.

**Methods:**

#### `to_mcp_prompt_message` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L91"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_prompt_message(self) -> PromptMessage
```

Convert to MCP PromptMessage.

### `PromptArgument` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L96"><Icon icon="github" /></a></sup>

An argument that can be passed to a prompt.

### `PromptResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L108"><Icon icon="github" /></a></sup>

Canonical result type for prompt rendering.

Provides explicit control over prompt responses: multiple messages,
roles, and metadata at both the message and result level.

**Methods:**

#### `to_mcp_prompt_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L180"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_prompt_result(self) -> GetPromptResult
```

Convert to MCP GetPromptResult.

### `Prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L190"><Icon icon="github" /></a></sup>

A prompt template that can be rendered with parameters.

**Methods:**

#### `to_mcp_prompt` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L202"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_prompt(self, **overrides: Any) -> SDKPrompt
```

Convert the prompt to an MCP prompt.

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L228"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any]) -> FunctionPrompt
```

Create a Prompt from a function.

The function can return:

* str: wrapped as single user Message
* list\[Message | str]: converted to list\[Message]
* PromptResult: used directly

#### `render` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L264"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
render(self, arguments: dict[str, Any] | None = None) -> str | list[Message | str] | PromptResult
```

Render the prompt with arguments.

Subclasses must implement this method. Return one of:

* str: Wrapped as single user Message
* list\[Message | str]: Converted to list\[Message]
* PromptResult: Used directly

#### `convert_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L277"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
convert_result(self, raw_value: Any) -> PromptResult
```

Convert a raw return value to PromptResult.

**Raises:**

* `TypeError`: for unsupported types

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L367"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this prompt with docket for background execution.

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L373"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, arguments: dict[str, Any] | None, **kwargs: Any) -> Execution
```

Schedule this prompt for background execution via docket.

**Args:**

* `docket`: The Docket instance
* `arguments`: Prompt arguments
* `fn_key`: Function lookup key in Docket registry (defaults to self.key)
* `task_key`: Redis storage key for the result
* `**kwargs`: Additional kwargs passed to docket.add()

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/prompts/prompt.py#L396"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```


# __init__
Source: https://gofastmcp.com/python-sdk/fastmcp-resources-__init__



# `fastmcp.resources`

*This module is empty or contains only private/internal implementations.*


# function_resource
Source: https://gofastmcp.com/python-sdk/fastmcp-resources-function_resource



# `fastmcp.resources.function_resource`

Standalone @resource decorator for FastMCP.

## Functions

### `resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/function_resource.py#L236"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
resource(uri: str) -> Callable[[F], F]
```

Standalone decorator to mark a function as an MCP resource.

Returns the original function with metadata attached. Register with a server
using mcp.add\_resource().

## Classes

### `DecoratedResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/function_resource.py#L36"><Icon icon="github" /></a></sup>

Protocol for functions decorated with @resource.

### `ResourceMeta` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/function_resource.py#L45"><Icon icon="github" /></a></sup>

Metadata attached to functions by the @resource decorator.

### `FunctionResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/function_resource.py#L64"><Icon icon="github" /></a></sup>

A resource that defers data loading by wrapping a function.

The function is only called when the resource is read, allowing for lazy loading
of potentially expensive data. This is particularly useful when listing resources,
as the function won't be called until the resource is actually accessed.

The function can return:

* str for text content (default)
* bytes for binary content
* other types will be converted to JSON

**Methods:**

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/function_resource.py#L80"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any], uri: str | AnyUrl | None = None) -> FunctionResource
```

Create a FunctionResource from a function.

**Args:**

* `fn`: The function to wrap
* `uri`: The URI for the resource (required if metadata not provided)
* `metadata`: ResourceMeta object with all configuration. If provided,
  individual parameters must not be passed.
* `name, title, etc.`: Individual parameters for backwards compatibility.
  Cannot be used together with metadata parameter.

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/function_resource.py#L204"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> str | bytes | ResourceResult
```

Read the resource by calling the wrapped function.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/function_resource.py#L225"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this resource with docket for background execution.

FunctionResource registers the underlying function, which has the user's
Depends parameters for docket to resolve.


# resource
Source: https://gofastmcp.com/python-sdk/fastmcp-resources-resource



# `fastmcp.resources.resource`

Base classes and interfaces for FastMCP resources.

## Classes

### `ResourceContent` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L37"><Icon icon="github" /></a></sup>

Wrapper for resource content with optional MIME type and metadata.

Accepts any value for content - strings and bytes pass through directly,
other types (dict, list, BaseModel, etc.) are automatically JSON-serialized.

**Methods:**

#### `to_mcp_resource_contents` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L92"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_resource_contents(self, uri: AnyUrl | str) -> mcp.types.TextResourceContents | mcp.types.BlobResourceContents
```

Convert to MCP resource contents type.

**Args:**

* `uri`: The URI of the resource (required by MCP types)

**Returns:**

* TextResourceContents for str content, BlobResourceContents for bytes

### `ResourceResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L119"><Icon icon="github" /></a></sup>

Canonical result type for resource reads.

Provides explicit control over resource responses: multiple content items,
per-item MIME types, and metadata at both the item and result level.

**Methods:**

#### `to_mcp_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L194"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_result(self, uri: AnyUrl | str) -> mcp.types.ReadResourceResult
```

Convert to MCP ReadResourceResult.

**Args:**

* `uri`: The URI of the resource (required by MCP types)

**Returns:**

* MCP ReadResourceResult with converted contents

### `Resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L210"><Icon icon="github" /></a></sup>

Base class for all resources.

**Methods:**

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L235"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any], uri: str | AnyUrl) -> FunctionResource
```

#### `set_default_mime_type` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L274"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_default_mime_type(cls, mime_type: str | None) -> str
```

Set default MIME type if not provided.

#### `set_default_name` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L281"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_default_name(self) -> Self
```

Set default name from URI if not provided.

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L291"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> str | bytes | ResourceResult
```

Read the resource content.

Subclasses implement this to return resource data. Supported return types:

* str: Text content
* bytes: Binary content
* ResourceResult: Full control over contents and result-level meta

#### `convert_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L303"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
convert_result(self, raw_value: Any) -> ResourceResult
```

Convert a raw result to ResourceResult.

This is used in two contexts:

1. In \_read() to convert user function return values to ResourceResult
2. In tasks\_result\_handler() to convert Docket task results to ResourceResult

Handles ResourceResult passthrough and converts raw values using
ResourceResult's normalization.  When the raw value is a plain
string or bytes, the resource's own `mime_type` is forwarded so
that `ui://` resources (and others with non-default MIME types)
don't fall back to `text/plain`.

The resource's component-level `meta` (e.g. `ui` metadata for
MCP Apps CSP/permissions) is propagated to each content item so
that hosts can read it from the `resources/read` response.

#### `to_mcp_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L374"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_resource(self, **overrides: Any) -> SDKResource
```

Convert the resource to an SDKResource.

#### `key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L397"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
key(self) -> str
```

The globally unique lookup key for this resource.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L402"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this resource with docket for background execution.

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L408"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, **kwargs: Any) -> Execution
```

Schedule this resource for background execution via docket.

**Args:**

* `docket`: The Docket instance
* `fn_key`: Function lookup key in Docket registry (defaults to self.key)
* `task_key`: Redis storage key for the result
* `**kwargs`: Additional kwargs passed to docket.add()

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/resource.py#L429"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```


# template
Source: https://gofastmcp.com/python-sdk/fastmcp-resources-template



# `fastmcp.resources.template`

Resource template functionality.

## Functions

### `extract_query_params` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L38"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
extract_query_params(uri_template: str) -> set[str]
```

Extract query parameter names from RFC 6570 `{?param1,param2}` syntax.

### `build_regex` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L46"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
build_regex(template: str) -> re.Pattern
```

Build regex pattern for URI template, handling RFC 6570 syntax.

Supports:

* `{var}` - simple path parameter
* `{var*}` - wildcard path parameter (captures multiple segments)
* `{?var1,var2}` - query parameters (ignored in path matching)

### `match_uri_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L72"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
match_uri_template(uri: str, uri_template: str) -> dict[str, str] | None
```

Match URI against template and extract both path and query parameters.

Supports RFC 6570 URI templates:

* Path params: `{var}`, `{var*}`
* Query params: `{?var1,var2}`

## Classes

### `ResourceTemplate` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L103"><Icon icon="github" /></a></sup>

A template for dynamically creating resources.

**Methods:**

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L130"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(fn: Callable[..., Any], uri_template: str, name: str | None = None, version: str | int | None = None, title: str | None = None, description: str | None = None, icons: list[Icon] | None = None, mime_type: str | None = None, tags: set[str] | None = None, annotations: Annotations | None = None, meta: dict[str, Any] | None = None, task: bool | TaskConfig | None = None, auth: AuthCheck | list[AuthCheck] | None = None) -> FunctionResourceTemplate
```

#### `set_default_mime_type` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L163"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_default_mime_type(cls, mime_type: str | None) -> str
```

Set default MIME type if not provided.

#### `matches` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L169"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
matches(self, uri: str) -> dict[str, Any] | None
```

Check if URI matches template and extract parameters.

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L173"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self, arguments: dict[str, Any]) -> str | bytes | ResourceResult
```

Read the resource content.

#### `convert_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L179"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
convert_result(self, raw_value: Any) -> ResourceResult
```

Convert a raw result to ResourceResult.

This is used in two contexts:

1. In \_read() to convert user function return values to ResourceResult
2. In tasks\_result\_handler() to convert Docket task results to ResourceResult

Handles ResourceResult passthrough and converts raw values using
ResourceResult's normalization.

#### `create_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L243"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_resource(self, uri: str, params: dict[str, Any]) -> Resource
```

Create a resource from the template with the given parameters.

The base implementation does not support background tasks.
Use FunctionResourceTemplate for task support.

#### `to_mcp_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L254"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_template(self, **overrides: Any) -> SDKResourceTemplate
```

Convert the resource template to an SDKResourceTemplate.

#### `from_mcp_template` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L274"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_mcp_template(cls, mcp_template: SDKResourceTemplate) -> ResourceTemplate
```

Creates a FastMCP ResourceTemplate from a raw MCP ResourceTemplate object.

#### `key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L287"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
key(self) -> str
```

The globally unique lookup key for this template.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L292"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this template with docket for background execution.

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L298"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, params: dict[str, Any], **kwargs: Any) -> Execution
```

Schedule this template for background execution via docket.

**Args:**

* `docket`: The Docket instance
* `params`: Template parameters
* `fn_key`: Function lookup key in Docket registry (defaults to self.key)
* `task_key`: Redis storage key for the result
* `**kwargs`: Additional kwargs passed to docket.add()

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L321"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```

### `FunctionResourceTemplate` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L328"><Icon icon="github" /></a></sup>

A template for dynamically creating resources.

**Methods:**

#### `create_resource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L374"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
create_resource(self, uri: str, params: dict[str, Any]) -> Resource
```

Create a resource from the template with the given parameters.

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L393"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self, arguments: dict[str, Any]) -> str | bytes | ResourceResult
```

Read the resource content.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L424"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this template with docket for background execution.

FunctionResourceTemplate registers the underlying function, which has the
user's Depends parameters for docket to resolve.

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L434"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, params: dict[str, Any], **kwargs: Any) -> Execution
```

Schedule this template for background execution via docket.

FunctionResourceTemplate splats the params dict since .fn expects \*\*kwargs.

**Args:**

* `docket`: The Docket instance
* `params`: Template parameters
* `fn_key`: Function lookup key in Docket registry (defaults to self.key)
* `task_key`: Redis storage key for the result
* `**kwargs`: Additional kwargs passed to docket.add()

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/template.py#L460"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any], uri_template: str, name: str | None = None, version: str | int | None = None, title: str | None = None, description: str | None = None, icons: list[Icon] | None = None, mime_type: str | None = None, tags: set[str] | None = None, annotations: Annotations | None = None, meta: dict[str, Any] | None = None, task: bool | TaskConfig | None = None, auth: AuthCheck | list[AuthCheck] | None = None) -> FunctionResourceTemplate
```

Create a template from a function.


# types
Source: https://gofastmcp.com/python-sdk/fastmcp-resources-types



# `fastmcp.resources.types`

Concrete resource implementations.

## Classes

### `TextResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L21"><Icon icon="github" /></a></sup>

A resource that reads from a string.

**Methods:**

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L26"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> ResourceResult
```

Read the text content.

### `BinaryResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L37"><Icon icon="github" /></a></sup>

A resource that reads from bytes.

**Methods:**

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L42"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> ResourceResult
```

Read the binary content.

### `FileResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L53"><Icon icon="github" /></a></sup>

A resource that reads from a file.

Set is\_binary=True to read file as binary data instead of text.

**Methods:**

#### `validate_absolute_path` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L75"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_absolute_path(cls, path: Path) -> Path
```

Ensure path is absolute.

#### `set_binary_from_mime_type` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L83"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
set_binary_from_mime_type(cls, is_binary: bool, info: ValidationInfo) -> bool
```

Set is\_binary based on mime\_type if not explicitly set.

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L91"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> ResourceResult
```

Read the file content.

### `HttpResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L105"><Icon icon="github" /></a></sup>

A resource that reads from an HTTP endpoint.

**Methods:**

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L114"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> ResourceResult
```

Read the HTTP content.

### `DirectoryResource` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L126"><Icon icon="github" /></a></sup>

A resource that lists files in a directory.

**Methods:**

#### `validate_absolute_path` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L146"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
validate_absolute_path(cls, path: Path) -> Path
```

Ensure path is absolute.

#### `list_files` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L152"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_files(self) -> list[Path]
```

List files in the directory.

#### `read` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/resources/types.py#L168"><Icon icon="github" /></a></sup>

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
read(self) -> ResourceResult
```

Read the directory listing.
