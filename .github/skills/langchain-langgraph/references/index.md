# LangChain LangGraph Reference Index

This skill is split into bounded reference files so opencode can load only the relevant material.

## File Selection

- Use `01-langchain-overview.md` for:
  - top-level defaults
  - anti-patterns
  - dedicated imports
  - agent reuse guidance
- Use `02-prompts-and-messages.md` for:
  - prompt templates
  - message prompts
  - multimodal inputs
  - signature preservation
- Use `03-tools-and-toolruntime.md` for:
  - `@tool`
  - args schemas
  - ToolRuntime reads and writes
  - tool streaming
  - tool errors
  - tool naming
- Use `04-model-selection-and-structured-output.md` for:
  - model routing
  - model profiles
  - prompt caching
  - structured output
  - dynamic response formats
- Use `05-middleware-and-guardrails.md` for:
  - node-style hooks
  - wrap-style hooks
  - state updates from middleware
  - guardrail composition
- Use `06-runtime-state-store-context.md` for:
  - model context
  - tool context
  - lifecycle context
  - runtime context
  - state and store separation
- Use `07-langgraph-state-nodes-edges.md` for:
  - `StateGraph`
  - reducers
  - nodes
  - edges
  - recursion limits
  - explicit orchestration
- Use `08-checkpointing-persistence-durability.md` for:
  - checkpointers
  - thread IDs
  - checkpoint IDs
  - snapshots
  - serializers
  - durability modes
- Use `09-interrupts-hitl-resume.md` for:
  - interrupts
  - resume semantics
  - idempotency around pauses
  - ordering hazards
- Use `10-subgraphs-and-streaming.md` for:
  - subgraph patterns
  - per-thread vs stateless subgraphs
  - stream `version="v2"`
  - namespace behavior
- Use `11-multi-agent-patterns.md` for:
  - subagents
  - handoffs
  - skills pattern
  - routers
  - custom workflows
- Use `12-memory.md` for:
  - short-term memory
  - long-term memory
  - semantic, episodic, and procedural memory
  - file-memory conventions
- Use `13-retrieval-rag.md` for:
  - loaders
  - splitters
  - embeddings
  - vector stores
  - retrievers
  - RAG architecture

## Quick Reminders

- Do not create agent instances inside LangGraph nodes unless there is a narrow, well-justified reason.
- Keep model context, tool context, runtime context, state, and store conceptually separate.
- Use reducers for append-style state fields or they will be overwritten.
- Prefer the smallest reference file that answers the current question.

## Preservation Notes

- These files preserve most of the original numbered notes and code examples from `.github/LangChain-LangGraph_organized_reference.md`.
- The split is editorial, not canonical. Cross-check the preserved source doc if exact original ordering matters.
