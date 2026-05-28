# LangGraph State Nodes Edges

## Quick Reference

- Use reducers such as `add_messages` or `operator.add` for mergeable state.
- Use async graph, model, and tool paths where I/O is involved.
- Set recursion limits and LLM timeouts.
- Prefer explicit message or tool-call handoffs for multi-agent workflows.

## State Rules

14. Use this:

```python
from langgraph.graph.message import add_messages

state["messages"] = add_messages(state["messages"], [new_msg])
```

17. Get conversation state:

```python
state = graph.get_state(config)
```

67. Use `TypedDict` for LangGraph state schemas.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class MyCustomState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    step_count: int
```

## Async Guidance

15. `SqliteSaver` is sync, `AsyncSqliteSaver` is async. In production, use the async version to avoid blocking your pool.

16. Use async methods:

- `ainvoke`
- `astream`
- `abatch`
- `atransform`

20. Always set a `recursion_limit` in your LangGraph and a timeout on your LLM calls.

## Explicit Orchestration Example

13. Example graph orchestration:

```python
from langgraph.graph import StateGraph
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

class LegalState(TypedDict):
    messages: list

def research_node(state: LegalState) -> LegalState:
    researcher = create_agent(
        model="gpt-4o",
        tools=[search_caselaw, validate_cite],
        middleware=[SummarizationMiddleware()],
        checkpointer=MemorySaver(),
        system_prompt="Legal researcher. Cite sources.",
    )
    result = researcher.invoke(state)
    return {"messages": result["messages"]}

def review_node(state: LegalState) -> LegalState:
    reviewer = create_agent(
        model="claude-3-sonnet",
        tools=[analyze_risk, flag_clause],
        system_prompt="Risk reviewer. Conservative analysis.",
    )
    result = reviewer.invoke(state)
    return {"messages": result["messages"]}

graph = StateGraph(LegalState)
graph.add_node("research", research_node)
graph.add_node("review", review_node)
graph.add_edge("research", "review")
app = graph.compile()
```

Note: this source example is preserved, but elsewhere in the reference there is a stronger recommendation to keep full agent loops out of nodes when explicit graph-native orchestration is the better fit.
