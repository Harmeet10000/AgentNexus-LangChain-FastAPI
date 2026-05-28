# Memory

## Quick Reference

Memory type | Scope | Stored in | Source notes
--- | --- | --- | ---
Short-term memory | One thread or conversation | Graph or agent state via checkpointer | 55, 57
Long-term memory | Across conversations or threads | LangGraph store namespace/key JSON docs | 30, 55, 57, 66
Semantic memory | Facts and concepts | Profile doc or collection | 66
Episodic memory | Past actions and experiences | Few-shot examples or retrieved examples | 66
Procedural memory | Rules and instructions | Prompts, code, weights, or editable instruction memory | 66

## Long-Term Memory API

Long-term memory is built on LangGraph stores, stored as JSON documents by namespace and key. Pass `store=...` to `create_agent`, then access it through `runtime.store` inside tools.

`InMemoryStore` is for development and testing. Use DB-backed stores for production.

Configure store indexing with embeddings and dimensions to support semantic memory search.

Namespace guidance:

- `(user_id, "memories")`
- `(org_id, "preferences")`
- `(user_id, app_context)`

## Preserved Source Notes

30. The recommended way to access the store is through the Runtime object.

55. Tools can access the current conversation state using `runtime.state`.

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    messages = runtime.state["messages"]
```

Update state with `Command`:

```python
from langgraph.types import Command
from langchain.tools import tool

@tool
def set_user_name(new_name: str) -> Command:
    return Command(update={"user_name": new_name})
```

Access context through `runtime.context`.

Access store through `runtime.store`.

57. By default, agents use `AgentState` for short-term memory, especially conversation history through `messages`.

```python
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel

class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomState]) -> Command:
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })
```

## File Memory Conventions

If memory has to be stored on disk, source notes suggest two file patterns:

1. `memory-dd-mm-yyyy.md`
2. `Memory.md`

The first is a daily running memory log.

The second is curated long-term facts, decisions, and preferences.

## Semantic Episodic Procedural Memory

66. Semantic memory stores facts.

Episodic memory stores experiences and past actions.

Procedural memory stores instructions and task rules.

Profile vs collection trade-off:

- profile: unified JSON document, easier centralization, harder updates at scale
- collection: higher recall and smaller atomic memories, but more search and deduplication complexity

Procedural memory can be refined by reflection or meta-prompting, where an agent updates its own instructions based on conversation history and feedback.
