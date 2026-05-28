# Middleware And Guardrails

## Supported Middleware Types

60. The following middleware work with any LLM provider:

- Summarization
- Human-in-the-loop
- Model call limit
- Tool call limit
- Model fallback
- PII detection
- To-do list
- LLM tool selector
- Tool retry
- Model retry
- LLM tool emulator
- Context editing
- Shell tool
- File search
- Filesystem
- Subagent

## Composing Guardrails

61. Combine multiple guardrails by stacking them in the middleware array. They execute in order.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, send_email_tool],
    middleware=[
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
        SafetyGuardrailMiddleware(),
    ],
)
```

## Runtime Access In Middleware

62. LangChain `create_agent` runs on LangGraph's runtime under the hood.

The runtime exposes:

- Context
- Store
- Stream writer

Use the Runtime parameter for node-style hooks. For wrap-style hooks, access runtime through `ModelRequest`.

## Custom Middleware Hooks

63. Middleware provides two styles of hooks:

- Node-style hooks: `before_agent`, `before_model`, `after_model`, `after_agent`
- Wrap-style hooks: `wrap_model_call`, `wrap_tool_call`

Node-style example:

```python
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 50:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None
```

Wrap-style example:

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")
```

## State Updates From Middleware

Node-style hooks return a dict directly.

```python
from langchain.agents.middleware import after_model, AgentState
from langgraph.runtime import Runtime
from typing import Any
from typing_extensions import NotRequired

class TrackingState(AgentState):
    model_call_count: NotRequired[int]

@after_model(state_schema=TrackingState)
def increment_after_model(state: TrackingState, runtime: Runtime) -> dict[str, Any] | None:
    return {"model_call_count": state.get("model_call_count", 0) + 1}
```

Wrap-style model hook returning `ExtendedModelResponse`:

```python
from typing import Callable
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    AgentState,
    ExtendedModelResponse,
)
from langgraph.types import Command
from typing_extensions import NotRequired

class UsageTrackingState(AgentState):
    last_model_call_tokens: NotRequired[int]

@wrap_model_call(state_schema=UsageTrackingState)
def track_usage(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ExtendedModelResponse:
    response = handler(request)
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"last_model_call_tokens": 150}),
    )
```

68. Quick reminder:

- `wrap_model_call`
- `wrap_tool_call`
- `before_model`
- `after_model`
- `before_agent`
- `after_agent`
