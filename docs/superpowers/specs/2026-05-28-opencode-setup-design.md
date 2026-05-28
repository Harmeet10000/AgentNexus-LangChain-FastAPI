# Opencode Setup Design

> Approved for implementation on 2026-05-28.

## Goal

Set up repo-local opencode configuration that reuses the current `.github/` project guidance, adds a minimal core set of opencode-native agents, creates an empty `.opencode/plugins/` directory, and turns `.github/LangChain-LangGraph_organized_reference.md` into a proper opencode skill without deleting or mutating the existing `.github/` sources.

## Scope

- Add a root `opencode.json` as the primary project entrypoint.
- Keep `.opencode/opencode.json` as a project-local overlay and preserve the existing Graphify plugin wiring.
- Reuse `.github/copilot-instructions.md` and `.opencode/AGENTS.md` as instruction sources.
- Add three opencode-native agents based on the current `.github/agents` content:
  - `principal-software-engineer`
  - `backend-developer`
  - `architect-reviewer`
- Create `.opencode/plugins/` but do not add new plugin files.
- Create a new `.opencode/skills/langchain-langgraph/` skill backed by split reference files derived from the organized reference document.

## Non-Goals

- Do not delete or rewrite anything in `.github/`.
- Do not mirror the full `.github/skills` tree into `.opencode/skills`.
- Do not invent keybindings or hooks that are not already represented in repo-local config.
- Do not add new plugins beyond preserving the existing Graphify plugin reference.

## Architecture

The root `opencode.json` will be the main project config and will declare shared instructions, the local skills path, a small curated agent set, and conservative permissions for common repo workflows. `.opencode/opencode.json` will remain a thin overlay for project-local plugin configuration so the existing Graphify integration remains intact and future plugin changes can stay scoped under `.opencode/`.

The LangChain/LangGraph skill will use a compact `SKILL.md` as a dispatcher and move most heavy material into `references/` markdown files so opencode can load the skill efficiently. The source `.github/LangChain-LangGraph_organized_reference.md` will remain the canonical preserved source document, while the new skill acts as an opencode-native access layer over that material.

## File Plan

- Create `opencode.json`
  - Root project opencode config.
- Modify `.opencode/opencode.json`
  - Keep plugin wiring and align with root config layering.
- Create `.opencode/agents/principal-software-engineer.md`
  - Principal-level engineering review and implementation guidance.
- Create `.opencode/agents/backend-developer.md`
  - Backend-focused delivery agent aligned to this FastAPI and LangGraph stack.
- Create `.opencode/agents/architect-reviewer.md`
  - Architecture review agent for design and systems trade-off work.
- Create `.opencode/plugins/.gitkeep`
  - Preserve an empty plugins directory in git.
- Create `.opencode/skills/langchain-langgraph/SKILL.md`
  - Entry skill and dispatcher.
- Create `.opencode/skills/langchain-langgraph/references/index.md`
  - Reference map.
- Create `.opencode/skills/langchain-langgraph/references/langchain-core.md`
  - LangChain agent, tool, middleware, and runtime notes.
- Create `.opencode/skills/langchain-langgraph/references/langgraph-core.md`
  - LangGraph state, nodes, edges, persistence, and execution notes.
- Create `.opencode/skills/langchain-langgraph/references/memory-and-retrieval.md`
  - Memory and retrieval notes from the organized reference.

## Data Flow

1. Opencode loads `opencode.json` from the repo root.
2. Root config loads project instructions from `.opencode/AGENTS.md` and `.github/copilot-instructions.md`.
3. Root config discovers the new local skill via `.opencode/skills`.
4. Root config exposes the curated repo agents for primary or subagent use.
5. `.opencode/opencode.json` overlays the existing Graphify plugin entry.

## Permissions

The repo config should default to safe, productive local usage:

- Allow normal read, search, and edit operations in the repo.
- Allow common local commands such as `uv`, `pytest`, `ruff`, `ty`, `git status`, `git diff`, and `graphify`.
- Require `ask` for broad or potentially destructive shell patterns.
- Deny clearly dangerous commands like `rm -rf *` and `git reset --hard *`.

## Error Handling

- Keep config small and schema-valid to reduce startup failures.
- Preserve the Graphify plugin path so current behavior does not regress.
- Use references files for the large LangChain/LangGraph content to avoid oversized skill bodies.

## Testing

- Read the resulting `opencode.json` and `.opencode/opencode.json` for shape verification.
- Read the generated agent and skill files for correctness.
- If available, run a JSON parse check over the root config.
- Summarize any restart requirement because opencode only reloads config on startup.

## Suggestions

After the base setup lands, good follow-up additions would be:

1. A repo health plugin or command wrapper for `uv run ruff`, `uv run ty`, and `uv run pytest`.
2. A LangSmith-focused skill or agent if tracing, datasets, and evaluation become frequent workflows.
3. Additional curated skills for `fastapi`, `fastmcp`, `crawl4ai`, or `pgvector` if you want more of `.github/skills` promoted into opencode later.
