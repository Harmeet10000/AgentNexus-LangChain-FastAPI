> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — confined to the tool subsystem: the registry's metadata model, one policy file, the selector's scoping rule, and one new grant on an existing tool. It is the security-relevant half of D4 (egress control on web search), which is why it is not S, but it touches no agent graph and no HTTP surface.

## Why

`agent-tool-capability-enforcement` made grants executable but left three gaps: authority metadata lives in tags (strings a selector can ignore), nothing maps roles to grants (the grant tuple has no authoritative source), and `web_search` — a model-chosen channel that ships client text to a third party — still has no egress gate. `PLAN-agent-tools.md` § D4 names the remedy: a capability catalog where authority is structured data, the projection a policy decision, and egress a grant like any other. The interim tags become seed data rather than the contract.

## What Changes

- **Structured authority fields on registration.** `register` gains typed authority metadata (enforced capability, side-effect class, egress flag) alongside tags; the `agent-tool-capability-enforcement` tags are migrated into these fields and retained as derived views for selection.
- **A role → grants policy file.** One declarative source maps agent role (and unbound/system context) to granted capabilities. The policy is the only place a grant is authored; `ToolRuntimeContext.capabilities` is populated from it, never assembled ad hoc at call sites.
- **Selector scoped within the authorized subset.** `LLMToolSelectorMiddleware` may only *shrink* the projection a role's policy already authorized — it can never surface a tool the policy did not grant. Selection is relevance; the policy is authority; the executor re-checks (unchanged).
- **Web-search egress grant.** `web_search` requires the `web.search` capability like any other tool; its class-based construction gains the runtime seam so the executor check is reachable. Without the grant, web search refuses — model-chosen text stops being a standing exfiltration channel.
- **Toolset diff discipline.** A test asserts that no role's model-visible toolset can contain a tool its policy does not authorize — the projection cap (≤7) becomes checkable data.

## Scope / Non-Goals

In scope: structured registry metadata, the policy file and its loading into runtime context, selector scoping, the web-search grant and seam migration, and the projection/authorization consistency tests.

Out of scope (explicit): JS/TS parity (deferred until agents exist on that side, per reference 03); changes to which tools exist; any Saul toolset change (D2/D5 own theirs); executor-side enforcement changes (this change authorizes; the executor semantics of `agent-tool-capability-enforcement` are consumed unchanged).

## Capabilities

### New Capabilities

- `tool-capability-catalog`: the declarative authority layer over the tool registry — structured metadata per tool, one policy source mapping roles to grants, a selector that may shrink but never widen, and egress-classified tools gated like any other.

### Modified Capabilities

- `agent-tool-registry`: registration carries structured authority metadata; the authority tags become derived views; the web-search tool requires its grant at execution.

## Impact

Affected code: `src/app/ai/tools/base.py` + `registry.py` (metadata model, policy loading), `src/app/ai/tools/web_search.py` (runtime seam + grant check), `src/app/ai/guardrails.py` (selector scoping), a new policy module/file (role → grants), `ToolRuntimeContext` population path (lifespan/graph factory), tests: policy-loading, selector-shrink-only, web-search grant, projection-consistency. Normative inputs: `PLAN-agent-tools.md` § D4 and P1/P5; `agent-tool-capability-enforcement` (executor semantics consumed, tags migrated).
