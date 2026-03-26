# Production Agent System

> LangChain 1.0 · LangGraph 1.0 · LangSmith · Gemini · FastAPI

A production-grade, multi-agent AI system designed to compete with the best coding and reasoning agents. Built on the latest LangChain 1.0 primitives (`create_agent`, `middleware`, `context_schema`).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                        │
│   /agents/invoke  /agents/stream  /agents/batch  /embed     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│                      Agent Runtime                         │
│   create_production_agent(AgentSpec)  →  ProductionAgent   │
│                                                            │
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
│                                                            │
│   context_schema → RichContext (user_id, role, flags...)   │
│   checkpointer   → InMemory / Postgres / Redis             │
└───────────────────────────┬────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
┌─────────▼──────────┐             ┌──────────▼──────────┐
│  Short-term Memory  │             │  Long-term Memory  │
│  LangGraph          │             │  cognee              │
│  Checkpointer       │             │  (semantic search) │
│  (per-thread)       │             │  (cross-session)   │
└────────────────────┘             └─────────────────────┘
```

## File Structure

```
agents/
  factory.py          # create_production_agent() + AgentSpec + ProductionAgent runtime
  registry.py         # Concrete agent instances (research, code, general)
  api.py              # FastAPI router (mount into your app)
  memory/
    manager.py        # MemoryManager: checkpointer + cognee
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
# Moto of this project
AI Agents should never be replacing Humans. They should be your devoted digital companions, ever-ready to absorb the soul-crushing repetition and mindless grunt work that slowly poisons the very profession you once chased with youthful fire. Niether should you use AI Agents to do the job for you completely end-to-end. Instead you should be doing AI-Assisted work that should feel like a very ergonomic office/work chair that doesnt break your back and a bent neck making you lose your time to dread-filled procrastination, existential second-guessing, and the quiet terror of “what if I chose wrong,”. If your agents frees your time that would rather have wasted dreading over life choices, you can spend that time to be more human lifting the invisible weight from your shoulders and not see your hair getting white prematurly.
this is a graph-backed, human-verified legal intelligence platform for Indian contracts.

# i am building a stateful, resumable, memory-aware reasoning pipeline
A distributed, resumable, schema-driven cognitive workflow engine with controlled reasoning surfaces Which has three layers:
1. Memory shaping (filters, trimming)
2. Runtime control (dynamic agents, routing)
3. Execution durability (pause/resume)
a stateless compute unit inside a deterministic workflow engine
The real architecture:
LLM = stateless reasoning engine
State = source of truth
Memory = indexed projections of state
The deepest insight: If your system cannot deterministically replay a run, you do not control your agent.
Final Mental model: Plan → deterministic execution → validated output → persisted state
Not: LLM → decide → act → hope it works

# workflow choice and reasoning
Orchestrator 
why:	Main agent plans → Delegates to workers → Synthesize.  Perfect: Plan multi-step, coordinate agents, reflect/recover
My requirements match Orchestrator exactly:

Deterministic workflows: Explicit plan → worker → reflect loop
Multi-step planning: Main orchestrator generates plan: list[str]
Repeated tool calls: Workers loop until plan complete
Self-reflection: Orchestrator reviews worker outputs
Error recovery: Orchestrator handles failures, re-plans
Coordinate agents: Workers = specialized sub-agents
Shared state: Orchestrator owns LegalState, workers read/write

# Performance Considerations
Do NOT init_chat_model + create_agent inside LangGraph nodes. Performance killer + anti-pattern. Pre-compile agents outside graph, pass as node functions.
Why NOT Inside Nodes? ❌
❌ BAD - Inside node (N+1 models per request)
def research_node(state):
    model = init_chat_model("gpt-4o")  # ❌ 100ms+ cold start per call
    agent = create_agent(model, tools)  # ❌ 50ms+ compilation per call
    return agent.invoke(state)          # ❌ Model init + agent build EVERY TIME

Correct Pattern: Pre-Compile Outside ✅
✅ GOOD - Pre-compile once
# OUTSIDE GRAPH - Compile once at startup
research_agent = create_agent(
    init_chat_model("gpt-4o"),
    tools=[search_caselaw]
)

def research_node(state):
    return research_agent.invoke(state)  # <5ms, cached model
Copy
Benefits:

Instant startup: Models/agents ready in memory
Zero recompilation: Compiled once at server start
Memory efficient: Reuse across all requests/threads
Scales perfectly: 10K req/s = 1 model instance
LangSmith optimized: Single trace per agent

LangGraph Node Strategy:
├─ ✅ Pre-compile create_agent OUTSIDE graph (global)
├─ ✅ Pass as node function: graph.add_node("research", research_agent.invoke)
├─ ✅ Per-node checkpointers for isolation
├─ ✅ Dynamic model selection via dict lookup
└─ ✅ Startup event for initialization

Result:
├─ Latency: 30ms (vs 500ms)
├─ Memory: 100MB (vs 20GB)
├─ Scale: 10K req/s
└─ Debug: LangSmith traces per agent
# Result: 500ms+ latency per node, scales poorly
Copy
Problems:

Cold starts: init_chat_model downloads model config/metadata (~100-500ms)
Recompilation: create_agent rebuilds agent executor (~50ms)
Memory leaks: Creates new model/agent objects per invocation
Scalability: 10 nodes × 10 req/s = 100 model inits/second
Token waste: Repeated system prompts/metadata


# Whats in this
3. Why are they looking for a solution (actual pain)

Not “time-saving”. That’s shallow.

Real reasons in India:
Small startups might not get best legal teams
individuals specially in India are not aware about there legal rights and are afraid of fine printed legal notice
need a Saul Goodmen like personality that can think outside of box and is not afraid to get into grey area
Grunt work that should be automated (big relief)

This is your goldmine.

High-volume, low-intelligence tasks:
Clause tagging
Risk flagging (one-sided indemnity, unlimited liability)
Deadline tracking
Comparing against standard templates
Stamp duty & jurisdiction sanity checks
“Is this clause enforceable in India?”

Lawyers hate this work. Clients pay for it unwillingly. Automate this, not “summaries”.
Existing players from companies Kira, Luminance, Evisort, Ironclad are trained on US/UK contracts
Poor understanding of Indian statutes
No concept of:
Stamp duty
Arbitration peculiarities
Indian limitation periods
Sectoral regulators
Junior associates miss risk → partner gets blamed
Businesses sign boilerplate contracts without understanding downside
MSMEs cannot afford continuous legal review
Inconsistent interpretation across regions

Time is a symptom. Risk and uncertainty are the disease.

7. Impact across domains (realistic, not hype)
Insurance
Detect claim-rejection loopholes
Policyholder vs insurer asymmetry reduced
Healthcare
Hospital contracts, consent forms, vendor SLAs
Compliance with clinical establishment norms
Prenup / Family settlements
Indian enforceability analysis
Asset disclosure risks
Business contracts
MSMEs finally understand what they sign
Vendor lock-in risks exposed
Wills & land
Title risk flags
Ambiguous beneficiary clauses
State-specific succession nuances

This directly reduces information asymmetry, not lawyers.

8. Judgements misinterpretation problem

In India:

Same judgement interpreted differently across High Courts
Older precedents ignored or revived suddenly
Context matters (facts > ratio)

Your system must:

Store judgement context
Highlight conflicting rulings
Surface latest binding authority

For example:

What binds a District Court vs High Court vs Supreme Court of India

This is not optional. This is core.

Why this DAG is correct
No cycles before human review → avoids compounding hallucinations
Risk + Compliance separated → legal correctness > linguistic fluency
Human gate before persistence → memory is trusted memory

Human-in-the-Loop Design (Mandatory, Not Optional)
Why humans are required
Legal liability
Continuous improvement
Trust formation
What humans do
Approve / reject risks
Correct clauses
Annotate reasoning
What you store
Overrides
Comments
Reviewer role

This becomes training + audit data.
