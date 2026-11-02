> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

The executor contract (denial on missing grant) already exists and is tested; this change builds the layer *above* it that decides what the grant tuple contains. Today that tuple is assembled where contexts are built, the authority story lives in tags, and the selector middleware can rank anything registered. The registry already resolves tools by name and tag through one authoritative surface — the catalog extends that surface, it does not add a second one.

## Goals / Non-Goals

**Goals:**
- One declarative answer to "why did role R have capability C?" — the policy file, greppable and diffable.
- The selector's scope made structural: it cannot widen, by construction, not by convention.
- Web search stops being a standing egress channel the moment the policy omits `web.search`.

**Non-Goals:**
- Rewriting the executor contract (consumed as-is from `agent-tool-capability-enforcement`).
- Per-request dynamic policies, per-tenant tool catalogs, or a model-visible catalog browser (P: the model sees its projection, period).
- JS/TS parity (no consumers on that side yet).

## Decisions

### Decision 1 — policy is a static, versioned declaration, not computed rules
A role → grants mapping in one file, loaded at startup, fail-fast on unknown capability identifiers (a grant naming a capability no tool enforces is a config bug, not a no-op). Rationale: the catalog's value is auditability; dynamic rules would recreate the ad-hoc assembly this change exists to remove. Per-tenant dynamics, if ever needed, extend the policy model later.

### Decision 2 — structured metadata replaces tags at the same registration sites
`register(...)` gains typed fields; the five system/filesystem tools and web search migrate in the same commit that introduces the fields. Tags remain as derived views so tag-based selection (handoff routing, the `web` tag) is untouched — the delta spec makes tag/metadata disagreement a registration error, which keeps the views honest.

### Decision 3 — selector scoping by intersection, not by post-filter
The selector's candidate set is the *intersection* of the policy-authorized set and the relevance ranking, computed before ranking is applied — not a post-ranking removal. Post-filtering would let an unauthorized tool influence ranking telemetry and ordering of what remains. The executor re-check stays (defense at the executor survives every layer above it — ADR-1 of the enforcement change).

### Decision 4 — web search migrates to the standard injection seam
The class-based `BaseTool` gains the `ToolRuntime` injection the `@tool`-built tools already have (the evidence/system tools' idiom), converting the D0 deferral into a normal grant. Its schema stays model-clean per the registry's schema requirement.

### Decision 5 — projection consistency is a test, not a convention
A test walks every role's policy entry and its model-visible toolset and asserts the subset invariant plus the ≤7 projection cap. The cap becomes checkable data; a future toolset edit that breaks either fails CI with the policy row named.

## Risks / Trade-offs

- **A policy file can drift from reality** → fail-fast loading plus the consistency test make drift loud at startup and in CI rather than at incident time.
- **Selector intersection may over-restrict when policy is coarser than relevance** → that is the intended direction: widening is a policy edit, which is reviewable; shrinking is runtime, which is not.

## Migration Plan

One commit introduces metadata + policy; a second migrates web search's seam; the selector scoping lands with its shrink-only test. No behaviour change for any role whose policy equals its current effective grants — the initial policy file is written to encode exactly today's effective sets, verified by the consistency test before any narrowing.

## Open Questions

None. (Whether the policy later gains per-tenant overrides is a D-next concern and deliberately not designed here.)
