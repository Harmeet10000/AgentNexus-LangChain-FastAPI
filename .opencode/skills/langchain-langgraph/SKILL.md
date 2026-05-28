---
name: langchain-langgraph
description: Use when working with LangChain or LangGraph in this repo, especially for agent construction, middleware, state graphs, persistence, memory, retrieval, runtime context, or durable execution patterns.
---

# LangChain LangGraph

Use this skill as the project-local dispatcher for LangChain and LangGraph guidance derived from the curated reference in `.github/LangChain-LangGraph_organized_reference.md`.

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

## Reference Files

- `references/index.md`
- `references/01-langchain-overview.md`
- `references/02-prompts-and-messages.md`
- `references/03-tools-and-toolruntime.md`
- `references/04-model-selection-and-structured-output.md`
- `references/05-middleware-and-guardrails.md`
- `references/06-runtime-state-store-context.md`
- `references/07-langgraph-state-nodes-edges.md`
- `references/08-checkpointing-persistence-durability.md`
- `references/09-interrupts-hitl-resume.md`
- `references/10-subgraphs-and-streaming.md`
- `references/11-multi-agent-patterns.md`
- `references/12-memory.md`
- `references/13-retrieval-rag.md`

## Source Preservation

The source reference remains in `.github/LangChain-LangGraph_organized_reference.md`. This skill is an opencode-native access layer and does not replace the preserved source document.
