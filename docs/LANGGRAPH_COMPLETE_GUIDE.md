# LangGraph Complete Documentation Guide

**Last Updated**: November 28, 2025  
**Source**: MCP Context7 - Official LangGraph Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [StateGraph & Workflows](#stategraph--workflows)
5. [Nodes & Edges](#nodes--edges)
6. [State Management](#state-management)
7. [Streaming](#streaming)
8. [Common Patterns](#common-patterns)
9. [Advanced Topics](#advanced-topics)
10. [Best Practices](#best-practices)

---

## Introduction

### What is LangGraph?

LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents. It provides:

- **Durable execution** with persistence
- **Human-in-the-loop** capabilities
- **Comprehensive memory** management
- **Production-ready** deployment
- **Flexible workflows** for complex agent logic
- **Multi-agent systems** support

### Key Features

- **Stateful graphs** with TypedDict-based state
- **Checkpointing** for persistence and resumability
- **Streaming** for real-time updates
- **Conditional routing** for dynamic workflows
- **Tool calling** with automatic handlers
- **Graph visualization** and debugging

---

## Installation & Setup

### Install LangGraph

```bash
pip install -U langgraph
```

### Install with Additional Tools

```bash
pip install -U langgraph "langchain[anthropic]"
```

### JavaScript/TypeScript

```bash
npm install @langchain/langgraph
```

### LangGraph CLI Setup

```bash
pip install -U "langgraph-cli[inmem]"
```

### Create New Project

```bash
# Python
langgraph new --template=new-langgraph-project-python my-agent
cd my-agent

# TypeScript
npx @langchain/langgraph-cli new --template=new-langgraph-project-typescript my-agent
cd my-agent
```

### LangGraph Development Server

```bash
# Start development server with auto-reload
langgraph dev

# Start with debugging
langgraph dev --debug-port 5678
```

---

## Core Concepts

### Graph Structure

```
START
  ↓
Node 1
  ↓
Node 2 (conditional)
  ├─ condition → Node 3
  └─ condition → Node 4
  ↓
END
```

### State Graph

A `StateGraph` is the foundation of LangGraph applications. It manages:

- **State schema**: Data structure for graph state
- **Nodes**: Individual processing units
- **Edges**: Connections between nodes
- **Routing logic**: Conditional and fixed edges

### Checkpointing

Enables persistence and resumability:

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)
```

---

## StateGraph & Workflows

### Define Graph State

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_info: str
    count: int
```

### Create StateGraph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Create builder
builder = StateGraph(State)

# Add nodes
def node1(state: State):
    return {"messages": ["Node 1 executed"]}

def node2(state: State):
    return {"count": state["count"] + 1}

builder.add_node("node1", node1)
builder.add_node("node2", node2)

# Add edges
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

# Compile with checkpointer
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)
```

### Invoke Graph

```python
# Execute with thread_id for persistence
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config=config
)

# Continue in same thread
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What did we discuss?"}]},
    config=config
)
```

---

## Nodes & Edges

### Node Definition

```python
def my_node(state: State) -> dict:
    """Process state and return updates."""
    # Access state
    messages = state["messages"]
    user_info = state["user_info"]
    
    # Process
    result = llm.invoke(messages)
    
    # Return partial update
    return {"messages": [result]}
```

### Fixed Edges

Connect nodes in a fixed sequence:

```python
builder.add_edge("node1", "node2")
builder.add_edge("node2", "node3")
```

### Conditional Edges

Route to different nodes based on state:

```python
from typing import Literal

def should_continue(state: State) -> Literal["node1", "node2", "end"]:
    """Routing function."""
    if condition1:
        return "node1"
    elif condition2:
        return "node2"
    else:
        return "end"

builder.add_conditional_edges(
    "start_node",
    should_continue,
    {
        "node1": "node1",
        "node2": "node2",
        "end": END
    }
)
```

### Node with Tools

```python
from langgraph.prebuilt import ToolNode

# Create tool node that handles tool calls
tool_node = ToolNode(tools=[tool1, tool2])

builder.add_node("call_tools", tool_node)
```

---

## State Management

### Annotated State with Reducers

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages are merged (deduplicated by ID)
    messages: Annotated[list, add_messages]
    # Custom fields
    user_id: str
    conversation_count: int
```

### Message Reducer

The `add_messages` reducer automatically:
- Merges new messages with existing ones
- Deduplicates by message ID
- Maintains message order

```python
from langgraph.graph.message import add_messages

# Add new message
state["messages"] = add_messages(
    state["messages"],
    [new_message]
)
```

### Custom State

```python
from typing_extensions import TypedDict

class CustomState(TypedDict):
    messages: list
    extra_field: int
    metadata: dict

def node(state: CustomState):
    # Partial update
    return {
        "extra_field": state["extra_field"] + 1,
        "metadata": {"updated": True}
    }
```

---

## Streaming

### Stream Mode: Values

Stream complete state after each step:

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="values",
):
    print(chunk)
```

### Stream Mode: Updates

Stream only state changes:

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates",
):
    print(chunk)
```

### JavaScript Streaming

```javascript
for await (const chunk of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "values" }
)) {
  console.log(chunk);
}
```

---

## Common Patterns

### Pattern 1: Simple Sequential Workflow

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    data: str

builder = StateGraph(State)

def step1(state: State):
    return {"data": "Step 1: " + state["data"]}

def step2(state: State):
    return {"data": state["data"] + " → Step 2"}

def step3(state: State):
    return {"data": state["data"] + " → Step 3"}

builder.add_node("step1", step1)
builder.add_node("step2", step2)
builder.add_node("step3", step3)

builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", "step3")
builder.add_edge("step3", END)

graph = builder.compile()
result = graph.invoke({"data": "Input"})
```

### Pattern 2: Conditional Routing (Reflexion)

```python
from typing import Literal

class State(TypedDict):
    messages: Annotated[list, add_messages]

def draft(state: State):
    """Generate draft response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_revise(state: State) -> Literal["revise", "end"]:
    """Check if revision needed."""
    last_message = state["messages"][-1]
    if needs_revision(last_message):
        return "revise"
    return "end"

def revise(state: State):
    """Refine response."""
    response = llm.invoke(state["messages"] + ["Please revise your response"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("draft", draft)
builder.add_node("revise", revise)

builder.add_edge(START, "draft")
builder.add_conditional_edges(
    "draft",
    should_revise,
    {"revise": "revise", "end": END}
)
builder.add_edge("revise", END)

graph = builder.compile()
```

### Pattern 3: Tool-Using Agent

```python
from langgraph.prebuilt import ToolNode
from typing import Literal

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def should_continue(state: AgentState) -> Literal["call_tools", "end"]:
    """Check if tools should be called."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tools"
    return "end"

builder = StateGraph(AgentState)

# Agent node
def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools=[tool1, tool2]))

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"call_tools": "tools", "end": END}
)
builder.add_edge("tools", "agent")  # Loop back

graph = builder.compile()
```

### Pattern 4: Multi-Agent Orchestration

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

def supervisor(state: State) -> dict:
    """Route to appropriate agent."""
    response = llm.invoke(state["messages"])
    agent_choice = extract_agent(response)
    return {"next_agent": agent_choice}

def agent1(state: State):
    """Agent 1 specialized for task A."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def agent2(state: State):
    """Agent 2 specialized for task B."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("agent1", agent1)
builder.add_node("agent2", agent2)

builder.add_edge(START, "supervisor")

def route_agent(state: State):
    if state["next_agent"] == "agent1":
        return "agent1"
    else:
        return "agent2"

builder.add_conditional_edges(
    "supervisor",
    route_agent
)
builder.add_edge("agent1", END)
builder.add_edge("agent2", END)

graph = builder.compile()
```

### Pattern 5: Document Processing Workflow

```python
class State(TypedDict):
    documents: list[str]
    processed: list[dict]
    summary: str

def load_documents(state: State):
    """Load documents."""
    docs = load_from_source()
    return {"documents": docs}

def process_each(state: State):
    """Process each document."""
    processed = [process_doc(doc) for doc in state["documents"]]
    return {"processed": processed}

def summarize(state: State):
    """Generate summary."""
    summary = llm.invoke(f"Summarize: {state['processed']}")
    return {"summary": summary}

builder = StateGraph(State)
builder.add_node("load", load_documents)
builder.add_node("process", process_each)
builder.add_node("summarize", summarize)

builder.add_edge(START, "load")
builder.add_edge("load", "process")
builder.add_edge("process", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
```

---

## Advanced Topics

### Graph Visualization

```python
import matplotlib.pyplot as plt

# Get graph visualization
graph_image = graph.get_graph().draw_mermaid_png()

# Save to file
with open("graph.png", "wb") as f:
    f.write(graph_image)

# Display
plt.imshow(graph_image)
plt.show()
```

### Persistence & Resumability

```python
from langgraph.checkpoint.memory import InMemorySaver

# Create checkpointer
memory = InMemorySaver()

# Compile with persistence
graph = builder.compile(checkpointer=memory)

# Execute with thread_id
config = {"configurable": {"thread_id": "conversation-1"}}
result1 = graph.invoke(input1, config)

# Resume later
result2 = graph.invoke(input2, config)  # Continues from last state
```

### Human-in-the-Loop

```python
from langgraph.types import Interrupt

def node_with_approval(state: State):
    """Node that waits for human approval."""
    action = generate_action(state)
    
    # Interrupt and wait for human input
    human_input = yield Interrupt(f"Approve action: {action}?")
    
    if human_input == "approved":
        return {"actions": [action]}
    else:
        return {"actions": []}
```

### Custom Checkpointing

```python
from langgraph.checkpoint.base import BaseCheckpointSaver

class CustomCheckpointer(BaseCheckpointSaver):
    def put(self, values, metadata):
        # Save to database
        db.save(values, metadata)
    
    def get(self, checkpoint_id):
        # Load from database
        return db.load(checkpoint_id)

checkpointer = CustomCheckpointer()
graph = builder.compile(checkpointer=checkpointer)
```

---

## Best Practices

### 1. State Design

✅ **Good:**
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    context: list[str]
```

❌ **Avoid:**
```python
class State(TypedDict):
    huge_nested_object: dict  # Difficult to update partially
    everything: str  # Too generic
```

### 2. Node Functions

✅ **Good:**
```python
def my_node(state: State) -> dict:
    """Process and return partial update."""
    result = process(state["data"])
    return {"processed": result}  # Partial update
```

❌ **Avoid:**
```python
def my_node(state: State):
    state["data"] = process(state["data"])  # Mutating state
    # Missing return statement
```

### 3. Conditional Edges

✅ **Good:**
```python
def route(state: State) -> Literal["path1", "path2"]:
    if condition:
        return "path1"
    return "path2"

builder.add_conditional_edges("node", route)
```

❌ **Avoid:**
```python
builder.add_conditional_edges("node", lambda s: ["path1", "path2"])  # Invalid
```

### 4. Error Handling

```python
def node_with_error_handling(state: State) -> dict:
    try:
        result = risky_operation(state)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}
```

### 5. Streaming Usage

```python
# Use streaming for real-time feedback
for event in graph.stream(input_data, config, stream_mode="values"):
    # Process each state update
    print(event)
    
    # Can stop streaming early if needed
    if should_stop(event):
        break
```

### 6. Thread Management

```python
# Use meaningful thread IDs
thread_id = f"conversation-{user_id}-{timestamp}"

config = {"configurable": {"thread_id": thread_id}}
result = graph.invoke(input_data, config)

# Resume conversation
result = graph.invoke(new_input, config)
```

---

## Examples

### Example 1: ReAct Agent

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

def agent(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools=[search_tool, calc_tool]))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
builder.add_edge("tools", "agent")

graph = builder.compile()
```

### Example 2: Document Q&A with Persistence

```python
from langgraph.checkpoint.memory import InMemorySaver

class DocQAState(TypedDict):
    document: str
    question: str
    answer: str
    messages: Annotated[list, add_messages]

def retrieve_doc(state: DocQAState):
    doc = load_document(state["document"])
    return {"messages": [f"Loaded: {doc[:100]}..."]}

def answer_question(state: DocQAState):
    context = state["document"]
    question = state["question"]
    response = llm.invoke(f"Using: {context}\nQuestion: {question}")
    return {"answer": response.content, "messages": [response]}

builder = StateGraph(DocQAState)
builder.add_node("retrieve", retrieve_doc)
builder.add_node("answer", answer_question)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)

graph = builder.compile(checkpointer=InMemorySaver())
```

---

## Resources

- **Official Docs**: https://langchain-ai.github.io/langgraph/
- **GitHub**: https://github.com/langchain-ai/langgraph
- **API Reference**: https://langchain-ai.github.io/langgraph/reference/
- **Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples
- **Community**: https://github.com/langchain-ai/langgraph/discussions

