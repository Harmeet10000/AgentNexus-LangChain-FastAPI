# Runtime State Store Context

## What You Can Control

To build reliable agents, control what happens at each step of the agent loop and what happens between steps.

Context Type | What You Control | Transient or Persistent
--- | --- | ---
Model Context | instructions, message history, tools, response format | Transient
Tool Context | what tools can access and produce | Persistent
Life-cycle Context | summarization, guardrails, logging, processing between steps | Persistent

Transient context is what the LLM sees for a single call.

Persistent context is what gets saved in state across turns.

## Data Sources

Data Source | Also Known As | Scope | Examples
--- | --- | --- | ---
Runtime Context | Static configuration | Conversation-scoped | user ID, API keys, DB connections, permissions, environment settings
State | Short-term memory | Conversation-scoped | current messages, uploaded files, auth status, tool results
Store | Long-term memory | Cross-conversation | user preferences, extracted insights, memories, historical data

## Model Context

Control what goes into each model call: instructions, available tools, which model to use, and output format.

These model-context choices can draw from:

- State
- Store
- Runtime Context

## Tool Context

Tools are special because they both read and write context.

Most real-world tools need more than the LLM's explicit parameters. They often need:

- user IDs
- API keys
- session state
- feature flags
- long-term memories

## Practical Separation Rules

- State is for active working memory.
- Store is for cross-thread durable memory.
- Runtime Context is immutable invocation-scoped configuration.
- Model Context is the exact prompt-time view for one call.
- Life-cycle Context is where middleware and guardrails shape execution over time.

## Important Notes

62. LangChain agents run on LangGraph runtime under the hood.

67. As of LangChain 1.0, custom state schemas must be `TypedDict` types. Pydantic models and dataclasses are no longer supported for custom state schemas.

Defining custom state via middleware is preferred over defining it via `state_schema` on `create_agent`, because it scopes state extensions to the relevant middleware and tools.

```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware

class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]
```
