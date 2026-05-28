# LangChain Overview

## Production Reminders

- Add `Field` descriptions for tool inputs instead of relying only on docstrings.
- Use structured output for LLM output, tool output, and MCP output when typed boundaries matter.
- Use async functions, methods, and packages across LangChain and LangGraph integrations.
- Trim or remove large tool outputs in multi-step agent conversations.
- Prefer dedicated package imports over deprecated community import paths.
- Do not rebuild model instances or agent instances on every call.

## Preserved Source Notes

1. add Field description for tool instead of simple docstrings

2. use structured ouput everywhere for llm output, tool output, MCP output

3. use toons for serialisation before sending to LLM

4. use toons for deserialisation after receiving from LLM

5. use toons for serialisation before sending to tools

6. use toons for deserialisation after receiving from tools / should i use chains for repeatable action for toon conversion

"Communicate data using TOON format. Declaring uniform arrays as key[N]{field1, field2}: val1, val2. Minimal punctuation. No braces."

7. use async functions, methods and packages in langchain and langGraph

8. trim/remove tool output in a multi step agent conversation

9. CORRECT — use dedicated package imports

```python
from langchain_tavily import TavilySearch

# WRONG — deprecated community import path
from langchain_community.tools.tavily_search import TavilySearchResults
```

Creating an agent instance (`create_agent`) inside a node is an anti-pattern because it forces the agent to be rebuilt on every single step, which is inefficient, prevents effective caching, and complicates testing.

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent  # or langchain.agents.create_agent
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

# Central model instances
cheap_model = init_chat_model("gpt-4o-mini", temperature=0)
expensive_model = init_chat_model("gpt-4o", temperature=0)

# Single agent instance
tools = [...]
agent = create_react_agent(cheap_model, tools)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def agent_node(state: AgentState) -> AgentState:
    result = agent.invoke(state)
    return {"messages": [result]}

workflow = StateGraph(state_schema=AgentState)
workflow.add_node("agent", agent_node)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
graph = workflow.compile()
```

State Management: Nodes should focus on how to process the state, not how to configure an agent. By keeping the agent logic outside the node, your nodes become cleaner and easier to unit test.

Using `langchain.agents.create_agent` inside a LangGraph node is generally an anti-pattern.

Why? `create_agent` returns a Runnable. While you can call a Runnable inside a node, LangGraph is designed to be the orchestrator. If you build a complex agent using LangChain's `create_agent` and put it inside a node, you bury an entire agent loop inside a single node, which makes state, streaming, and human-in-the-loop behavior harder to manage.

The LangGraph way in modern LangGraph is to define your own nodes and explicitly implement the agent loop (`LLM call -> check for tool calls -> execute tools -> repeat`) using native features like `ToolNode` and conditional edges.

10. all methods, functions, model and agent invocation should have langsmith decorator for proper obervability

11. always normalise agent state after fetching from checkpointer so that there is no version mismatch

12. have proper retry mechanism for tools with idenpotent execution as mention in langchain docs

24. Model instances are rebuilt on every call. `build_chat_model()` constructs a new model every time it is called. The model object should be a module-level singleton or per-spec singleton since it is stateless.

## Design Categories

Sync vs Async Execution:

- Synchronous: the main agent waits for subagent results. Simpler, but blocks the conversation.
- Asynchronous: subagents run in the background. More complex, but better for responsiveness when tasks are independent.

Tool Design:

- Tool per Agent: dedicated tool to each subagent, more configuration, more control.
- Single Dispatch Tool: one tool routes tasks to any subagent, good when the system has many agents.

Context Engineering:

- Give the main agent clear tool names, descriptive docstrings, and well-defined schemas.
- Customize subagent inputs and returned history to improve performance.
