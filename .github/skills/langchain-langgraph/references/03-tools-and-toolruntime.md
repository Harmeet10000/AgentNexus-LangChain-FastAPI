# Tools And ToolRuntime

## Tool Design Rules

- Each tool needs a clear name, description, argument names, and argument descriptions.
- These are not just metadata. They guide the model's reasoning about when and how to use the tool.
- Too many tools can overwhelm the model. Too few can block completion.

## Tool Selection Examples

Filter tools by State:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    is_authenticated = request.state.get("authenticated", False)
    message_count = len(request.state["messages"])

    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)
```

Filter tools by Store:

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@wrap_model_call
def store_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        tools = [t for t in request.tools if t.name in enabled_features]
        request = request.override(tools=tools)

    return handler(request)
```

Filter tools by Runtime Context permissions:

```python
from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@dataclass
class Context:
    user_role: str

@wrap_model_call
def context_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        pass
    elif user_role == "editor":
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)

    return handler(request)
```

## ToolRuntime Reads

Read from State:

```python
from langchain.tools import tool, ToolRuntime

@tool
def check_authentication(runtime: ToolRuntime) -> str:
    current_state = runtime.state
    is_authenticated = current_state.get("authenticated", False)
    return "User is authenticated" if is_authenticated else "User is not authenticated"
```

Read from Store:

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str

@tool
def get_preference(preference_key: str, runtime: ToolRuntime[Context]) -> str:
    user_id = runtime.context.user_id
    store = runtime.store
    existing_prefs = store.get(("preferences",), user_id)

    if existing_prefs:
        value = existing_prefs.value.get(preference_key)
        return f"{preference_key}: {value}" if value else f"No preference set for {preference_key}"
    return "No preferences found"
```

Read from Runtime Context:

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str
    api_key: str
    db_connection: str

@tool
def fetch_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    user_id = runtime.context.user_id
    api_key = runtime.context.api_key
    db_connection = runtime.context.db_connection
    results = perform_database_query(db_connection, query, api_key)
    return f"Found {len(results)} results for user {user_id}"
```

## ToolRuntime Writes

Write to State with `Command`:

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

@tool
def authenticate_user(password: str, runtime: ToolRuntime) -> Command:
    if password == "correct":
        return Command(update={"authenticated": True})
    return Command(update={"authenticated": False})
```

Write to Store:

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str

@tool
def save_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime[Context]
) -> str:
    user_id = runtime.context.user_id
    store = runtime.store
    existing_prefs = store.get(("preferences",), user_id)
    prefs = existing_prefs.value if existing_prefs else {}
    prefs[preference_key] = preference_value
    store.put(("preferences",), user_id, prefs)
    return f"Saved preference: {preference_key} = {preference_value}"
```

## Streaming From Tools

58. To stream updates from tools as they are executed, you can use `get_stream_writer`.

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"
```

ToolRuntime stream writer:

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    writer = runtime.stream_writer
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"
```

## Tool Error Handling

56. Error handling via `ToolNode`:

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
tool_node = ToolNode(tools, handle_tool_errors=True)
tool_node = ToolNode(tools, handle_tool_errors="Something went wrong, please try again.")

def handle_error(e: ValueError) -> str:
    return f"Invalid input: {e}"

tool_node = ToolNode(tools, handle_tool_errors=handle_error)
tool_node = ToolNode(tools, handle_tool_errors=(ValueError, TypeError))
```

## Tool Naming And Schema Rules

54. Prefer `snake_case` for tool names such as `web_search` instead of `Web Search`. Some model providers reject names containing spaces or special characters.

```python
@tool(name="", description="", args_schema=WeatherInput)
```

ToolRuntime provides access to:

- State
- Context
- Store
- Stream Writer
- Config
- Tool Call ID
