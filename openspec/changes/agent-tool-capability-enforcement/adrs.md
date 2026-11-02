# ADRs — agent-tool-capability-enforcement

## ADR-1: Enforce capability grants inside the tool executor

**Decision:** the grant check runs in each tool body against injected, hidden `ToolRuntime` context — not in the registry, the binding step, or the selector middleware.
**Alternatives:** binding-time filtering (rejected: binding is per-compile and silent; a rebind silently re-enables), selector-based authorization (rejected: visibility is relevance, never authority), middleware wrapper (rejected: middleware stacks are per-agent configuration and refactor-prone).
**Consequence:** binding a dangerous tool without a grant is a no-op at every call; the mistake becomes safe by construction.

## ADR-2: Workspace jail defaults to deny-unset

**Decision:** with `TOOL_WORKSPACE_ROOTS` unset, path-taking tools refuse with an operator-directed message.
**Alternatives:** default to a temp dir (rejected: silently defines a sandbox the operator never chose), default to cwd (rejected: the process working directory is exactly what the jail exists to fence off).
**Consequence:** first use requires one configuration decision, made explicitly.

## ADR-3: Authority metadata as tags until the catalog lands

**Decision:** `side-effect:*` / `capability:*` tags now; structured registry fields and a role → grant policy file are D4's.
**Alternatives:** design the structured model now (rejected: one consumer, premature policy vocabulary), skip metadata entirely (rejected: the catalog needs seed data and reviewers need the tags for diffing).
**Consequence:** D4 inherits a working vocabulary; nothing here becomes load-bearing selection logic.

## ADR-4: Argument schemas are schema models, never tool types

**Decision:** `WebSearchInput` is a plain `BaseModel`.
**Alternatives:** keep the `BaseTool` inheritance (rejected: schema-as-tool confusion — a nameless unusable tool object in the type system).
**Consequence:** the model-facing schema is exactly the declared fields; no phantom tool exists.
