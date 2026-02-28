# Production Agent System

> LangChain 1.0 · LangGraph 1.0 · LangSmith · Gemini · FastAPI

A production-grade, multi-agent AI system designed to compete with the best coding and reasoning agents. Built on the latest LangChain 1.0 primitives (`create_agent`, `middleware`, `context_schema`).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                         │
│   /agents/invoke  /agents/stream  /agents/batch  /embed     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      Agent Runtime                           │
│   create_production_agent(AgentSpec)  →  ProductionAgent    │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              LangChain 1.0: create_agent            │  │
│   │                                                     │  │
│   │  model ──► [middleware stack] ──► tools ──► output  │  │
│   │             │                                       │  │
│   │  Middleware (before_model order):                   │  │
│   │  1. SummarizationMiddleware  (context window)       │  │
│   │  2. LLMToolSelectorMiddleware (reduce tool noise)   │  │
│   │  3. ToolRetryMiddleware      (resilience)           │  │
│   │  4. ModelRetryMiddleware     (resilience)           │  │
│   │  5. HumanInTheLoopMiddleware (HITL - optional)      │  │
│   │  6. GuardrailMiddleware      (safety - after_model) │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   context_schema → RichContext (user_id, role, flags...)    │
│   checkpointer   → InMemory / Postgres / Redis              │
└───────────────────────────┬─────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
┌─────────▼──────────┐             ┌──────────▼──────────┐
│  Short-term Memory  │             │  Long-term Memory    │
│  LangGraph          │             │  Mem0               │
│  Checkpointer       │             │  (semantic search)  │
│  (per-thread)       │             │  (cross-session)    │
└────────────────────┘             └─────────────────────┘
```

## File Structure

```
agents/
  factory.py          # create_production_agent() + AgentSpec + ProductionAgent runtime
  registry.py         # Concrete agent instances (research, code, general)
  api.py              # FastAPI router (mount into your app)
  memory/
    manager.py        # MemoryManager: checkpointer + Mem0
  orchestration/
    supervisor.py     # MultiAgentSystem, LLMRouter, Skill, Handoff
  tools/
    base.py           # ToolOutput, ToolRegistry, @register_tool
    shell.py          # shell_tool, read_file, write_file, list_directory, file_search
    subagent.py       # make_subagent_tool() — agents as tools

langchain_layer/      # ⚠ Rename to avoid shadowing installed `langchain` package
  callback.py         # LangSmith setup, LatencyCallback, TokenUsageCallback, AsyncStreamingCallback
  chains.py           # LCEL chains: summarization, router, guardrail, extraction, parallel
  messages.py         # trim_by_token_count, summarize_history, manage_context
  models.py           # build_chat_model, ainvoke_text, abatch_text, astream_text,
                      # ainvoke_multimodal, abatch_multimodal, aembed_text, aembed_batch
  prompts.py          # SystemPromptParts, inject_context, few_shot_block

langgraph_layer/      # ⚠ Rename to avoid shadowing installed `langgraph` package
  state.py            # AgentState, SupervisorState, BaseContext, RichContext
  nodes.py            # agent_node, guardrail_node, supervisor_node, synthesizer_node
  edges.py            # should_continue, after_guardrail, supervisor_route
  graph.py            # build_single_agent_graph, build_supervisor_graph,
                      # build_sequential_workflow, build_parallel_fanout_graph

middleware/
  __init__.py         # ALL middleware: Summarization, HITL, ToolSelector, ToolRetry,
                      # ModelRetry, TodoList, ContextEditing, Guardrail, DynamicPrompt
  guardrails.py       # DeterministicGuardrails, evaluate_response (model-based)

schemas/
  agent.py            # AgentInvokeRequest, AgentResponse, EmbedRequest/Response, etc.

config/
  settings.py         # Typed settings via pydantic-settings
```

## Quick Start

### 1. Install

```bash
pip install langchain>=1.0.0 langgraph>=1.0.0 langsmith langchain-google-genai mem0ai
```

### 2. Configure

```env
# .env
GOOGLE_API_KEY=your_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=my-agent
MEM0_API_KEY=your_mem0_key          # Optional
POSTGRES_URI=postgresql://...       # Optional (for production checkpointer)
```

### 3. Bootstrap (in your FastAPI main.py)

```python
from langchain_layer.callback import configure_langsmith
configure_langsmith()   # Call before anything else

from agents.api import router as agent_router
app.include_router(agent_router, prefix="/agents")
```

### 4. Invoke an agent

```python
from agents.registry import get_research_agent
from langgraph_layer.state import ResearchAgentContext

agent = get_research_agent()

# Single turn
result = await agent.ainvoke(
    "What is the current state of LLM reasoning?",
    thread_id="session-123",
    user_id="user-456",
    context=ResearchAgentContext(user_id="user-456", depth="deep"),
)

# Streaming
async for chunk in agent.astream("Explain transformers", thread_id="session-123"):
    print(chunk)

# Batch
results = await agent.abatch(
    ["Q1", "Q2", "Q3"],
    thread_ids=["t1", "t2", "t3"],
)
```

### 5. Create a custom agent

```python
from agents.factory import AgentSpec, create_production_agent
from langgraph_layer.state import RichContext
from middleware import TodoListMiddleware, DynamicSystemPromptMiddleware

@dataclass
class MyContext(RichContext):
    department: str = "engineering"

spec = AgentSpec(
    name="my_agent",
    description="My custom agent",
    tools=["web_search_tool"],           # tool names from registry
    context_schema=MyContext,
    system_prompt="You are a ${department} specialist.",
    extra_middleware=[
        TodoListMiddleware().build(),
        DynamicSystemPromptMiddleware(
            prompt_fn=lambda state, ctx: f"You work in {ctx.department}."
        ).build(),
    ],
    enable_guardrails=True,
    enable_long_term_memory=True,
)
agent = create_production_agent(spec)
```

### 6. Multi-agent system

```python
from agents.orchestration.supervisor import MultiAgentSystem, Skill

system = MultiAgentSystem()
system.register_agent(get_research_agent())
system.register_agent(get_code_agent())
system.register_skill(Skill("summarize", "Summarize text", my_summarize_fn))
system.build()

result = await system.ainvoke("Research and implement a Redis cache", thread_id="t1")
```

---

## Key Concepts

### `AgentSpec` + `create_production_agent`
Declarative agent definition. Handles all wiring: model, tools, middleware, memory, context.

### `context_schema`
Typed, non-persisted runtime context. Available in middleware and dynamic prompts.
Not stored in checkpoints — passed fresh on each `ainvoke` call.

```python
result = await agent.ainvoke(
    "Hello",
    thread_id="t1",
    context=MyContext(user_role="admin", department="sales"),
)
```

### Middleware
Composable hooks around the agent loop. Applied in order for `before_model`,
reverse order for `after_model`. Mix built-in and custom.

### Memory
- **Short-term**: LangGraph checkpointer (automatic per `thread_id`)
- **Long-term**: Mem0 (semantic search across sessions, injected into system prompt)
- **Context window management**: `SummarizationMiddleware` (default) or `manage_context()`

### Guardrails
Two layers:
1. `DeterministicGuardrails.check_input/output()` — zero latency, keyword/regex blocking
2. `GuardrailMiddleware` + `evaluate_response()` — LLM-as-judge (~200ms)

---

## ⚠ Important: Naming Conflicts

The folders `langchain/` and `langgraph/` in this project **shadow** the installed
packages of the same name if they live at the Python import root.

**Solution**: Either:
- Put everything inside your app package: `src/myapp/langchain_layer/`
- Use the `langchain_layer/` / `langgraph_layer/` naming used in this project
- Or use namespace packages carefully

This project uses `langchain_layer/` and `langgraph_layer/` to avoid conflicts.
