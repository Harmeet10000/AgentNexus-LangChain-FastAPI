# LangChain & LangGraph Quick Reference Cheat Sheet

---

## 🚀 LangChain Quick Start

### Installation
```bash
pip install langchain langchain-openai langchain-community
```

### Initialize Chat Model
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4", api_key="sk-...")
```

### Create a Tool
```python
from langchain.tools import tool

@tool
def my_tool(input: str) -> str:
    """Tool description."""
    return result
```

### Build an Agent
```python
from langchain.agents import create_agent

agent = create_agent(
    model,
    tools=[my_tool],
    system_prompt="You are helpful"
)

result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

### RAG System (3 Steps)
```python
# 1. Load & chunk
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(["https://example.com"])
docs = loader.load()
splits = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(docs)

# 2. Index
vector_store.add_documents(splits)

# 3. Retrieve
@tool
def retrieve(query: str):
    return vector_store.similarity_search(query, k=2)

agent = create_agent(model, tools=[retrieve])
```

---

## 🔗 LangGraph Quick Start

### Installation
```bash
pip install langgraph langchain langchain-openai
```

### Basic Workflow
```python
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# 1. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Create Graph
builder = StateGraph(State)

# 3. Add Nodes
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder.add_node("chatbot", chatbot)

# 4. Add Edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 5. Compile
graph = builder.compile(checkpointer=InMemorySaver())

# 6. Invoke
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config=config
)
```

### Conditional Routing
```python
from typing import Literal

def should_continue(state: State) -> Literal["tool_node", "end"]:
    if needs_tool(state["messages"][-1]):
        return "tool_node"
    return "end"

builder.add_conditional_edges("agent", should_continue)
```

### Tool Node
```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([tool1, tool2])
builder.add_node("tools", tool_node)
```

---

## 📦 Vector Store Cheat Sheet

### Pinecone
```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("index-name")
vs = PineconeVectorStore(index=index, embedding=embeddings)
```

### Qdrant
```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")
vs = QdrantVectorStore(client=client, collection_name="docs", embedding=embeddings)
```

### Chroma
```python
from langchain_chroma import Chroma

vs = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="./data"
)
```

### FAISS
```python
import faiss
from langchain_community.vectorstores import FAISS

index = faiss.IndexFlatL2(embedding_dim)
vs = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
```

---

## 🛠️ Common Patterns

### Pattern: ReAct Agent
```python
from langgraph.prebuilt import ToolNode

def should_continue(state):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "end"

builder.add_edge(START, "agent")
builder.add_node("agent", agent_func)
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")
```

### Pattern: Reflexion
```python
def should_revise(state):
    if needs_improvement(state["messages"][-1]):
        return "revise"
    return "end"

builder.add_edge(START, "draft")
builder.add_conditional_edges("draft", should_revise)
builder.add_edge("revise", END)
```

### Pattern: Multi-Agent
```python
def supervisor(state):
    route = llm.invoke(state["messages"])
    return {"next": extract_agent(route)}

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", lambda s: s["next"])
builder.add_edge("agent1", END)
builder.add_edge("agent2", END)
```

---

## 💾 State Management Cheat Sheet

### Define State
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Auto-merges messages
    user_id: str  # Simple field
    context: list[str]  # Simple list
```

### Update State in Node
```python
def my_node(state: State) -> dict:
    # Return partial update
    return {"messages": [new_message]}
```

### Reducer Functions
```python
from langgraph.graph.message import add_messages

# Messages are merged by ID (deduplication)
state["messages"] = add_messages(state["messages"], [new_msg])
```

---

## 🔄 Streaming Cheat Sheet

### Stream Values (Full State)
```python
for chunk in graph.stream(input_data, config, stream_mode="values"):
    print(chunk)  # Full state at each step
```

### Stream Updates (Only Changes)
```python
for chunk in graph.stream(input_data, config, stream_mode="updates"):
    print(chunk)  # Only changed fields
```

### Stream LLM Tokens
```python
for chunk in graph.stream(input_data, config):
    if isinstance(chunk, str):
        print(chunk, end="")  # Print tokens
```

---

## 🔐 Persistence Cheat Sheet

### In-Memory Checkpointer
```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())
```

### Invoke with Thread ID
```python
config = {"configurable": {"thread_id": "conversation-123"}}
result = graph.invoke(input_data, config)

# Continue later with same thread_id
result2 = graph.invoke(new_input, config)  # Resumes from last state
```

### Get Conversation History
```python
from langgraph.graph import get_state

state = graph.get_state(config)
print(state.values)  # Full conversation history
```

---

## 🎯 Debugging Cheat Sheet

### Visualize Graph
```python
# Save PNG visualization
graph_image = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)
```

### Print Graph Structure
```python
print(graph.get_graph().to_mermaid())
```

### Debug Node
```python
def debug_node(state: State):
    print(f"State keys: {state.keys()}")
    print(f"Messages: {state['messages']}")
    return {}

builder.add_node("debug", debug_node)
```

### Stream with Details
```python
for event in graph.stream(input_data, config):
    print(f"Event: {event}")
```

---

## ⚡ Performance Tips

### LangChain
- ✅ Use vector store batching for multiple queries
- ✅ Cache embeddings for repeated documents
- ✅ Use async operations with `ainvoke()`
- ✅ Stream responses for faster feedback
- ❌ Don't re-embed the same documents

### LangGraph
- ✅ Use checkpointing to save partial progress
- ✅ Return only changed state fields
- ✅ Use streaming for real-time updates
- ✅ Cache LLM responses with caching decorators
- ❌ Don't process huge documents in single nodes

---

## 🐛 Common Errors & Solutions

### Error: "No tool found"
**Solution**: Ensure tool name matches exactly in tool_calls

### Error: "State schema mismatch"
**Solution**: Ensure node return dict matches State keys

### Error: "Checkpointer not set"
**Solution**: Pass checkpointer to `compile()`:
```python
graph = builder.compile(checkpointer=InMemorySaver())
```

### Error: "Thread ID required"
**Solution**: Pass config with thread_id:
```python
config = {"configurable": {"thread_id": "..."}}
```

---

## 📚 File Organization

```
project/
├── src/
│   └── agents/
│       ├── rag_agent.py
│       ├── tools/
│       │   ├── search.py
│       │   ├── calculator.py
│       │   └── database.py
│       └── graphs/
│           ├── react_graph.py
│           └── multi_agent_graph.py
├── docs/
│   ├── LANGCHAIN_COMPLETE_GUIDE.md
│   ├── LANGGRAPH_COMPLETE_GUIDE.md
│   └── INDEX_LANGCHAIN_LANGGRAPH.md
└── tests/
    ├── test_agents.py
    └── test_graphs.py
```

---

## 🔗 Import Reference

### LangChain Core
```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

### LangGraph
```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Interrupt
```

### Vector Stores
```python
from langchain_pinecone import PineconeVectorStore
from langchain_qdrant import QdrantVectorStore
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_milvus import Milvus
```

---

## ✅ Checklist for Production

- [ ] Use environment variables for API keys
- [ ] Implement error handling in agents
- [ ] Enable logging and monitoring
- [ ] Use persistent checkpointing
- [ ] Implement rate limiting
- [ ] Add input validation
- [ ] Test with multiple models
- [ ] Monitor token usage
- [ ] Set up alerts for failures
- [ ] Document custom tools
- [ ] Version your graphs
- [ ] Implement rollback strategy

---

**Last Updated**: November 28, 2025  
**Quick Reference Version**: 1.0

