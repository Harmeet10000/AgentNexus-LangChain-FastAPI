---
name: langchain-langgraph
description: Use when working with LangChain or LangGraph in this repo, especially for agent construction, middleware, state graphs, persistence, memory, retrieval, runtime context, or durable execution patterns.
---

## Workflow

1. Open `references/index.md` first.
2. Pick the smallest matching reference file instead of loading everything.
3. Prefer the LangChain-side files for agent construction, prompts, tools, ToolRuntime, middleware, model routing, and structured output.
4. Prefer the LangGraph-side files for graph design, reducers, checkpoints, interrupts, subgraphs, streaming, and durable execution.
5. Prefer the memory and retrieval files for long-term memory, state/store separation, vector stores, loaders, splitters, embeddings, and RAG architecture.
6. Preserve the original source doc as the canonical reference for cross-checking; this skill is a bounded access layer, not a replacement source.

## High-Value Repo Rules

- Prefer explicit LangGraph orchestration for complex workflows instead of burying a full agent loop inside a graph node.
- Reuse model and agent instances instead of rebuilding them inside each call path.
- Use structured output consistently for LLM and tool-facing boundaries when typed data matters.
- Keep async usage end-to-end across LangChain and LangGraph integrations.
- Preserve provider-specific message metadata when the model requires it across turns.
- Normalize persisted agent state when loading from checkpointers if schema or version drift is possible.
- Keep code examples close to the topic-specific references instead of collapsing them into abstract summaries.

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
