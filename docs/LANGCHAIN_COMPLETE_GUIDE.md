# LangChain Complete Documentation Guide
Perfect! Now I have comprehensive information. Let me compile a structured learning path for building models with LangChain and LangGraph.

---

## **LangChain & LangGraph Learning Path for Building AI Models**

Here's an ordered list of all essential topics to learn for building AI models, organized from foundational to advanced:

### **Foundation (Start Here)**

1. **Models & Chat Models** - Understand how to use language models and their interfaces
2. **Messages & Prompts** - Learn message formats, prompt templates, and dynamic prompting
3. **Tools** - Define and create tools for your models to use
4. **Tool Calling** - Enable models to decide which tools to invoke

### **Building Agents (Core Development)**

5. **Agents** - Create agents using `create_agent` for autonomous task execution
6. **Middleware** - Add customization hooks (prebuilt and custom middleware)
7. **Context Engineering** - Dynamically manage information flow to models
8. **Structured Output** - Generate responses in specific formats

### **Knowledge & Data (RAG & Retrieval)**

9. **Document Loaders** - Load data from various sources
10. **Embeddings** - Convert text to vector representations
11. **Vector Stores** - Store and search embeddings
12. **Retrievers** - Build retrieval components for fetching relevant documents
13. **Retrieval (RAG)** - Implement retrieval-augmented generation architectures
14. **2-Step RAG** - Simple, predictable retrieval and generation
15. **Agentic RAG** - Agent-driven retrieval with dynamic tool use
16. **Hybrid RAG** - Combine 2-Step and Agentic approaches

### **State & Memory (Conversation Management)**

17. **Messages** - Understand message types and formats
18. **Short-term Memory** - Maintain conversation state
19. **Long-term Memory (Store)** - Persist data across sessions
20. **State Management** - Define and manage custom state schemas

### **Advanced Agent Features**

21. **Streaming** - Stream tokens, tool calls, and updates in real-time
22. **Human-in-the-Loop** - Add approval gates for sensitive actions
23. **Error Handling** - Implement tool error handling with middleware

### **Orchestration & Workflows (LangGraph)**

24. **LangGraph Overview** - Understand graph-based agent orchestration
25. **State Graph** - Define nodes and edges for agent workflows
26. **Graph API** - Build custom workflows with nodes and conditionals
27. **Persistence** - Save and resume agent execution (checkpointing)
28. **Durable Execution** - Build agents resilient to failures
29. **Interrupts** - Pause execution for human oversight

### **Production & Observability**

30. **Tracing** - Track agent execution with LangSmith
31. **Evaluation** - Measure and validate model outputs
32. **Deployment** - Deploy agents to production
33. **Middleware Patterns** - Implement middleware for production concerns
34. **Summarization** - Manage long conversations with context summarization
35. **PII Redaction** - Protect sensitive information in prompts

### **Optional Advanced Topics**

36. **Runnables** - Chain components together
37. **Document Transformers** - Process and transform documents
38. **Text Splitters** - Split documents into retrievable chunks
39. **Callbacks** - Add custom logging and monitoring
40. **Prompt Management** - Version and manage prompts in LangSmith

---

## **Recommended Learning Sequence by Use Case**

**Simple Chatbot:** 1 → 2 → 3 → 4 → 5 → 30

**RAG Application:** 1 → 2 → 9 → 10 → 11 → 12 → 13 → 14 → 30

**Intelligent Agent:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 21 → 30

**Production Agent:** All items in order (1-40)

**Relevant docs:**

- [Agents Documentation](https://docs.langchain.com/oss/python/langchain/agents)
- [Tools Documentation](https://docs.langchain.com/oss/python/langchain/tools)
- [Retrieval & RAG](https://docs.langchain.com/oss/python/langchain/retrieval)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Prompts & Messages](https://docs.langchain.com/oss/python/langchain/messages)
- [Streaming](https://docs.langchain.com/oss/python/langchain/streaming)

## Core Concepts

### LangChain Overview

LangChain is a framework for developing applications powered by large language models (LLMs). It simplifies every stage of the LLM application lifecycle by offering:

- **Open-source components** for building LLM applications
- **Third-party integrations** with various LLM providers
- **Standardized interfaces** for consistent model interactions
- **Tools and memory** for building complex AI applications
- **Streaming support** for real-time LLM interactions

### Key Components

- **Chat Models**: Language models optimized for conversation
- **Tools**: Functions that agents can call to perform actions
- **Agents**: LLM-powered decision makers that use tools
- **Chains**: Sequences of operations on LLM outputs
- **Memory**: Persistent state management for conversations
- **Vector Stores**: Semantic search and embedding storage

---

## Installation & Setup

### Install Project Dependencies

You can use either `pip` or the modern `uv` package manager:

```bash
# Using pip
pip install -e .

# Using uv (faster and recommended)
uv sync
```

### Install LangChain with OpenAI Support

```bash
pip install -U "langchain[openai]"
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Vector Stores

Vector stores are critical for RAG systems, enabling semantic search and similarity matching.

### Pinecone Vector Store

**Installation:**
```bash
pip install -qU langchain-pinecone
```

**Usage:**
```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key=...)
index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```

### Qdrant Vector Store

**Installation:**
```bash
pip install -qU langchain-qdrant
```

**Setup:**
```python
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Initialize Qdrant client
client = QdrantClient(...)

# Create vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=embeddings,
)
```

### Chroma Vector Store

**Installation:**
```bash
pip install -qU langchain-chroma
```

**Usage:**
```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
```

### FAISS Vector Store

**Installation:**
```bash
pip install -qU langchain-community
```

**Usage:**
```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
```

### Milvus Vector Store

**Installation:**
```bash
pip install -qU langchain-milvus
```

**Usage:**
```python
from langchain_milvus import Milvus

URI = "./milvus_example.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)
```

### Fake Embeddings (for Testing)

```bash
pip install -qU langchain-core
```

```python
from langchain_core.embeddings import DeterministicFakeEmbedding

embeddings = DeterministicFakeEmbedding(size=4096)
```

---

## Agents & Tools

### Creating Tools

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
```

### Creating Agents with create_agent

```python
from langchain.agents import create_agent

tools = [retrieve_context]
system_prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)

agent = create_agent(model, tools, system_prompt=system_prompt)
```

### Streaming Agent Execution

```python
query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

### Agent with Fallback Models

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="gpt-4o",  # Primary model
    tools=[search_tool, calculator_tool],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ),
    ],
)
```

---

## RAG (Retrieval-Augmented Generation)

### Complete RAG Agent Example

```python
import bs4
from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load and chunk web content
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Step 2: Index chunks in vector store
_ = vector_store.add_documents(documents=all_splits)

# Step 3: Create retrieval tool
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 4: Create RAG agent
tools = [retrieve_context]
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

# Step 5: Query the agent
query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

### Agentic RAG with Documentation Fetching

```python
import requests
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from markdownify import markdownify

ALLOWED_DOMAINS = ["https://langchain-ai.github.io/"]
LLMS_TXT = 'https://langchain-ai.github.io/langgraph/llms.txt'

@tool
def fetch_documentation(url: str) -> str:
    """Fetch and convert documentation from a URL"""
    if not any(url.startswith(domain) for domain in ALLOWED_DOMAINS):
        return (
            "Error: URL not allowed. "
            f"Must start with one of: {', '.join(ALLOWED_DOMAINS)}"
        )
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return markdownify(response.text)

# Fetch llms.txt content ahead of time
llms_txt_content = requests.get(LLMS_TXT).text

# Create system prompt with approved documentation
system_prompt = f"""
You are an expert Python developer and technical assistant.
Your primary role is to help users with questions about LangGraph and related tools.

Instructions:

1. If a user asks a question you're unsure about — or one that likely involves API usage,
   behavior, or configuration — you MUST use the `fetch_documentation` tool.
2. When citing documentation, summarize clearly and include relevant context.
3. Do not use any URLs outside of the allowed domain.
4. If a documentation fetch fails, tell the user and proceed with your best understanding.

You can access official documentation from the following approved sources:

{llms_txt_content}

Your answers should be clear, concise, and technically accurate.
"""

tools = [fetch_documentation]
model = init_chat_model("claude-sonnet-4-0", max_tokens=32_000)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    name="Agentic RAG",
)

# Query the agent
response = agent.invoke({
    'messages': [
        HumanMessage(content=(
            "Write a short example of a langgraph agent using the "
            "prebuilt create react agent. the agent should be able "
            "to look up stock pricing information."
        ))
    ]
})

print(response['messages'][-1].content)
```

### RAG with Middleware

```python
from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState

class State(AgentState):
    context: list[Document]

class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = vector_store.similarity_search(last_message.text)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return {"context": retrieved_docs}

# Create agent with middleware
agent = create_agent(
    model,
    tools=[],
    middleware=[RetrieveDocumentsMiddleware()],
)
```

### Dynamic Prompt with Context Injection

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message

agent = create_agent(model, tools=[], middleware=[prompt_with_context])
```

---

## Chat Models

### Initialize OpenAI Chat Model

**Method 1: Using init_chat_model**
```python
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model("gpt-4.1")
```

**Method 2: Direct instantiation**
```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "sk-..."

model = ChatOpenAI(model="gpt-4.1")
```

---

## SQL Agents

### Create SQL Agent

```python
from langchain.agents import create_agent

# Define SQL tools and system prompt
system_prompt = """
You are a helpful assistant that can query a SQL database.
Use the provided tools to execute queries and analyze results.
"""

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)
```

---

## Best Practices

### 1. Vector Store Selection

| Vector Store | Use Case | Advantages |
|---|---|---|
| **Pinecone** | Cloud-hosted semantic search | Managed service, scalable, easy to use |
| **Qdrant** | Self-hosted semantic search | Open source, fast, flexible |
| **Chroma** | Development & testing | In-memory, simple API |
| **FAISS** | Local/offline use | Fast, no external deps |
| **Milvus** | Enterprise scale | Open source, distributed |

### 2. Document Loading

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load documents
loader = WebBaseLoader(web_paths=["https://example.com"])
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# Add to vector store
vector_store.add_documents(splits)
```

### 3. Tool Design

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def my_tool(input_str: str) -> tuple[str, Any]:
    """Tool description for the LLM."""
    result = process(input_str)
    return str(result), result  # (display, artifact)
```

### 4. Error Handling in Agents

```python
try:
    result = agent.invoke({"messages": [user_message]})
except Exception as e:
    print(f"Agent error: {e}")
    # Fallback behavior
```

### 5. Streaming for Better UX

```python
# Stream intermediate steps
for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    print(event["messages"][-1].content)
```

### 6. Memory Management

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)

# Use in agent
agent = create_agent(
    model,
    tools,
    memory=memory
)
```

---

## Common Patterns

### Pattern 1: Question Answering with RAG

```
User Question
    ↓
Retrieve relevant documents
    ↓
Inject into LLM context
    ↓
Generate answer
    ↓
Return to user
```

### Pattern 2: Agentic RAG

```
User Question
    ↓
Agent decides: need more info?
    ├─ Yes → Use retrieval tool
    │         ↓
    │         More info retrieved?
    │         ├─ Yes → Generate answer
    │         └─ No → Loop back
    └─ No → Generate answer
            ↓
            Return to user
```

### Pattern 3: Multi-Step Reasoning

```
Query
    ↓
Decompose into steps
    ↓
Execute each step with tools
    ↓
Aggregate results
    ↓
Generate final answer
```

---

## Resources

- **Official Docs**: https://docs.langchain.com/
- **GitHub**: https://github.com/langchain-ai/langchain
- **Community**: https://github.com/langchain-ai/langchain/discussions
- **API Reference**: https://api.python.langchain.com/

