> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

The enforcement landed in the working tree before this change existed (D0 of `PLAN-agent-tools.md`); this document records the decisions that shaped it so the spec delta is reviewable. The tools in this band are registered in the single authoritative registry but bound to no agent — the enforcement changes what a *future mistake* costs, not any live behaviour.

## Goals / Non-Goals

**Goals:**
- Enforcement that survives every layer above it: binding, middleware, and prompt refactors cannot re-open the hole.
- Deny-by-default at two levels: per-call (no grant → refuse) and per-deployment (no jail configured → refuse).
- Refusals that are operator documentation, not just errors.

**Non-Goals:**
- Role → grant policy mapping and structured authority fields (D4, `tool-capability-catalog`).
- Web-search egress gating (class-based tool; needs the runtime seam migrated — same D4 change).
- Making these tools available to any agent.

## Decisions

### Decision 1 — enforcement at the executor, not at binding or selection
The grant check lives inside each tool body, behind the same `trusted_tool_context` seam the evidence tools already use. A registry, a tool selector, or a binding step can all be bypassed or misconfigured; the executor cannot be skipped on the way to the operation. Selector-based visibility is explicitly *not* authorization (P1 of the plan).

### Decision 2 — containment on the resolved path, checked against resolved roots
`Path.resolve()` on both the requested path and every configured root before comparison. A symlink from inside the jail to `/etc` must fail containment, not pass the string check and escape through the link. The symlink-escape test is the one that matters; it is in the suite.

### Decision 3 — shell requires an explicit jailed `cwd` rather than defaulting one
A default ("first configured root") would make the safe-looking case silently wide: any relative path in the command then resolves against an attacker-chosen base. Requiring an explicit `cwd` keeps the process confined and makes the operator's intent visible in the transcript.

### Decision 4 — `total=False` TypedDict grant field, with tests as the real gate
`ToolRuntimeContext.capabilities` is optional-by-shape (the context is a `total=False` TypedDict), so the type checker cannot see a missing grant. The deny paths for every tool are asserted by name in the enforcement suite — the tests hold the line the type system cannot.

### Decision 5 — authority as tags for now, structured fields in D4
`side-effect:*` / `capability:*` tags are additive and keep legacy tag selection intact. The `tool-capability-catalog` change (D4) replaces them with structured registry fields and a policy file; committing to the structured shape now would design the policy model before its second consumer exists.

## Risks / Trade-offs

- **Deny-by-default may surprise a future consumer** → every refusal names the capability or setting; the enforcement suite doubles as executable documentation of the contract.
- **Jail as a single global setting** → adequate while the only consumers are host-side utility tools; multi-root or per-role jails are a D4 policy concern, not an executor one.

## Migration Plan

None required — no tool in this band is bound to any agent. The change is verification-and-spec for code already in the working tree.

## Open Questions

None. (Web-search egress gating is deliberately deferred, with its reason recorded in the proposal's non-goals.)
