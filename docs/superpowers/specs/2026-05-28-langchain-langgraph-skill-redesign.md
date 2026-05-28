# LangChain LangGraph Skill Redesign

> Approved for implementation on 2026-05-28.

## Goal

Rewrite the repo-local `langchain-langgraph` opencode skill so it preserves far more of the source material from `.github/LangChain-LangGraph_organized_reference.md`, including code examples, while splitting the content into focused bounded reference files that are easier for opencode to load selectively.

## Problem With The First Version

The initial skill split was too compressed. It summarized the source into three short files and dropped many of the original code examples, dynamic context patterns, middleware examples, checkpointing details, interrupt rules, streaming notes, and retrieval examples. That made the resulting skill easier to load but less useful as a working reference.

## Scope

- Replace the current shallow skill routing and references.
- Preserve the existing skill name and directory so current config does not need to change.
- Keep `.github/LangChain-LangGraph_organized_reference.md` untouched.
- Create a richer `SKILL.md` and `references/index.md`.
- Create a near-complete split of the source into focused topic files.

## Architecture

The rewritten skill remains a dispatcher skill with a small `SKILL.md`, but the reference content moves into many smaller files organized by usage questions instead of by one broad LangChain/LangGraph summary. The new split will preserve original notes and code blocks wherever practical, while adding only enough editorial structure to make the material navigable.

## File Plan

- Modify `.opencode/skills/langchain-langgraph/SKILL.md`
  - Replace the shallow dispatcher with a richer routing guide.
- Modify `.opencode/skills/langchain-langgraph/references/index.md`
  - Add a fuller topic map and usage guidance.
- Create `.opencode/skills/langchain-langgraph/references/01-langchain-overview.md`
  - Core defaults, anti-patterns, and top-level LangChain notes.
- Create `.opencode/skills/langchain-langgraph/references/02-prompts-and-messages.md`
  - Prompt types, message handling, multimodal content, and signature preservation.
- Create `.opencode/skills/langchain-langgraph/references/03-tools-and-toolruntime.md`
  - Tool schemas, ToolRuntime reads/writes, streaming, tool error handling, and naming rules.
- Create `.opencode/skills/langchain-langgraph/references/04-model-selection-and-structured-output.md`
  - Dynamic model selection, profiles, caching, response formats, and structured output.
- Create `.opencode/skills/langchain-langgraph/references/05-middleware-and-guardrails.md`
  - Hook styles, examples, guardrail composition, and state updates.
- Create `.opencode/skills/langchain-langgraph/references/06-runtime-state-store-context.md`
  - Model context, tool context, lifecycle context, state, store, and runtime context patterns.
- Create `.opencode/skills/langchain-langgraph/references/07-langgraph-state-nodes-edges.md`
  - Graph design method, reducers, nodes, edges, and explicit orchestration patterns.
- Create `.opencode/skills/langchain-langgraph/references/08-checkpointing-persistence-durability.md`
  - Checkpointers, thread IDs, snapshots, serializers, durability modes, and replay rules.
- Create `.opencode/skills/langchain-langgraph/references/09-interrupts-hitl-resume.md`
  - Interrupt semantics, ordering hazards, idempotency constraints, and resume behavior.
- Create `.opencode/skills/langchain-langgraph/references/10-subgraphs-and-streaming.md`
  - Subgraph patterns, persistence choices, stream v2 format, and namespace behavior.
- Create `.opencode/skills/langchain-langgraph/references/11-multi-agent-patterns.md`
  - Subagents, handoffs, skills, routers, custom workflows, and architecture trade-offs.
- Create `.opencode/skills/langchain-langgraph/references/12-memory.md`
  - Short-term memory, long-term memory, semantic/episodic/procedural memory, and file-memory conventions.
- Create `.opencode/skills/langchain-langgraph/references/13-retrieval-rag.md`
  - Loaders, splitting, embeddings, caching, vector stores, retrievers, and RAG architecture choices.

## Editorial Rules

- Preserve almost all original notes and examples.
- Keep important opinionated guidance even when the source wording is rough.
- Fix only structural issues that improve navigation.
- Avoid rewriting the content into abstract summaries when a concrete example already exists.

## Testing

- Read the rewritten `SKILL.md` and `references/index.md` for routing quality.
- Read several reference files to confirm code examples and important source notes survived.
- Ensure the new file layout covers LangChain, LangGraph, memory, and retrieval without major omissions.
