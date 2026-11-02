> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — confined to the tool subsystem (`app/ai/tools/`, one settings field, one TypedDict field) plus its unit suite. It is a security-posture change on a latent surface, which is why it is not S, but it changes no live agent behaviour: no tool in this module is bound to any agent today.

## Why

The system/filesystem tools' module docstring claimed "Shell and filesystem tools require explicit capability grants via context", but no tool body inspected any context: `shell_tool` executed arbitrary model-supplied command strings with a caller-controlled working directory, and the file tools read or wrote any path on the host. The tools are bound to no agent (latent, not live), which made the gap invisible — and made the fix cheap to land before any binding exists. `PLAN-agent-tools.md` § D0 specified the remedy; it is implemented in the working tree. This change carries its spec delta so the enforcement becomes a contractual requirement on `agent-tool-registry` rather than a code detail, and records the verification that was run.

## What Changes

- **Deny-by-default capability enforcement at the executor.** Every tool in the system/filesystem band requires its capability (`system.shell.execute`, `filesystem.read`, `filesystem.write`) to be granted in the server-owned `ToolRuntime` context (`ToolRuntimeContext.capabilities`, new field). No runtime, or a runtime without the grant, produces a typed refusal — never execution. Grants can never arrive as model-visible arguments: the runtime is injected and hidden from the tool schema.
- **Workspace jail.** A `TOOL_WORKSPACE_ROOTS` setting names the only roots path-taking tools may touch. Containment is checked on the *resolved* path, so a symlink from inside the jail to outside it fails rather than escaping. An unset jail refuses everything with an operator-directed message (deny-by-default at the deployment level too).
- **Shell requires an explicit jailed `cwd`.** A shell without a confined working directory is an unconfined process, so absence of `cwd` is a refusal, not a default.
- **Schema hygiene:** `WebSearchInput` becomes a plain `BaseModel` (it is an argument schema, not a tool — inheriting `BaseTool` made it a nameless, unusable tool in its own right).
- **Authority metadata at registration.** The five tools register with `side-effect:*` and `capability:*` tags alongside their existing category tags; tag-based selection by the legacy tags is unchanged.
- **Refusals name their fix.** Deny messages name the missing capability or the setting to configure, because a refusal that explains itself gets fixed while a bare "denied" gets worked around by widening grants.

## Scope / Non-Goals

In scope: executor-side enforcement for the five system/filesystem tools, the jail setting, the schema fix, authority tags, and the enforcement test suite.

Out of scope (explicit): web-search grant enforcement (the class-based `BaseTool` needs the runtime seam migrated first — owned by the `tool-capability-catalog` change, D4); structured authority fields replacing tags (same change); any wiring of these tools into Agent Saul (P: "no shell/filesystem tools for Saul" — the grants exist so the global registry is safe if another consumer ever binds them); the capability policy file mapping roles to grants (D4).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `agent-tool-registry`: tool execution is denied unless the server-owned runtime context grants the tool's capability; path-taking tools are confined to a configured workspace jail; registration carries authority metadata; refusals name their fix.

## Impact

Already implemented in the working tree (no commit): `src/app/ai/tools/shell.py` (enforcement, jail, explicit-cwd), `src/app/ai/tools/registry.py` (authority tags), `src/app/ai/tools/web_search.py` (schema fix), `src/app/features/documents/tools/authorization.py` (`capabilities` on `ToolRuntimeContext`), `src/app/config/settings.py` (`TOOL_WORKSPACE_ROOTS`), `tests/unit/ai/test_system_tool_capabilities.py` (12 enforcement tests). Normative input: `PLAN-agent-tools.md` § D0 and § 3 (catalog → assignment → projection). Gates at implementation time: ruff clean on touched files, `ty` unchanged for the subsystem, 12/12 new tests green, full suite failures attributed (all pre-existing or environmental — see `review.md`).
