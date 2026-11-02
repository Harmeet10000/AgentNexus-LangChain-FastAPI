> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — one change that gives the five deferred S-tier roadmap items (G2, G4, G5, G6-remainder, G7) named contracts and unambiguous unblocking triggers. It defers all of them behind explicit gates: nothing here ships behaviour today. It exists so the deferrals recorded in other changes' non-goal sections become owned, triggered contracts instead of prose that only the non-goals remember.

## Why

The audit of the three plans (2026-11-01) found the only genuine spec gaps: five S-tier roadmap items that every active change explicitly excludes from its own scope, and that no other change owns. `mount-retrieval-graph` reserves G4's registration point and names G5/G7 as out of scope; `retrieval-measurement-gate` names G2, G5, G6-remainder, G7 as out of scope; the harden change owns the auth/budget *events* and pilot-kill alerts but not the retrieval loop's own budget machinery. Deferred is a decision only while someone remembers it — this change is the remembering.

## What Changes

- **G2 — contextual chunk retrieval (new capability, gated).** The `preamble` column on chunks has been empty since migration 0003 and the generated search text concatenates `clause_type ‖ preamble ‖ content`. This change owns the contract: ingest-time context summaries (amortized, prompt-cached document context per the contextual-retrievals approach) written to `preamble`, landing in the searchable text. **Gate:** the `retrieval-measurement-gate` baseline exists, and the win is measured on it before the feature is default-on.
- **G4 — Graphiti as the fourth fused branch (modified, gated).** The registration point is already reserved at the fusion call site by `mount-retrieval-graph`. This change owns the contract: graph-traversal results fuse into the weighted RRF as a fourth branch with a measured weight. **Gate:** relationship edges exist in the corpus (harden 4.1 landed) **and** the G1 baseline can attribute the branch's contribution.
- **G5 — small-to-big retrieval (modified, gated).** Index at clause level, hydrate the parent section for generation using the stored structural tree. **Gate:** the G1 baseline shows recall gaps attributable to context starvation — it is adopted on evidence, not on fashion.
- **G6-remainder — retrieval-loop budgets and degradation alerts (modified, gated).** `mount-retrieval-graph` ships per-stage spans and the grade-decision log; the harden change ships auth/budget events and pilot-kill criteria. What nobody owns is the loop's own budget enforcement (cost/latency caps per ask, attributed per stage) and the standing degradation alert on those budgets. **Gate:** `mount-retrieval-graph` lands (spans exist to attribute against).
- **G7 — retrieval feedback capture (new capability, gated).** Captured user signals (accept/reject, citation clicks) promoted into the golden set under review — the compounding engine. **Gate:** a UI surface emitting the signals exists (the `ag-ui-copilotkit-console` change) and the G1 harness can ingest promoted queries.
- **Ledger rows added** for the two surface-mappable deferrals (the empty `preamble` column; the reserved fourth-branch registration point), per the quarantine ledger's standing rule 4 — each with a concrete revisit trigger.
- **G8 niceties (late chunking, multi-query expansion, semantic cache) are named as permanently out of scope until an eval demands them** — recorded once, here, so no future audit re-discovers them.

## Scope / Non-Goals

In scope: the gated contracts above, their trigger definitions, the two ledger rows, and the requirement that every gate is checkable.

Out of scope (explicit): implementing any G-item (each task here is deferred behind its named trigger); the measurement harness itself (`retrieval-measurement-gate`); the graph edges (harden 4.1, landed); the UI signals (`ag-ui-copilotkit-console`); any behaviour change to the served retrieval path.

## Capabilities

### New Capabilities

- `contextual-chunk-retrieval`: the ingest-time context-summary contract for the empty `preamble` column — amortized generation, search-text integration, measured-before-default-on.
- `retrieval-feedback-loop`: the capture-and-promote contract turning user signals into reviewed golden-set queries.

### Modified Capabilities

- `hybrid-retrieval-ranking`: a fourth (graph-traversal) fusion branch admitted only when its weight can be measured, at the reserved registration point.
- `agentic-retrieval-loop`: per-stage budget enforcement on the retrieval loop with a standing degradation alert, and (gated) section-level context hydration for generation.

## Impact

Affected code at implementation time (none today): chunk upsert path + `preamble` generation (G2), fusion call site (`application/queries/documents.py`, G4), retrieval loop hydration + budgets (G5/G6), signal capture endpoints + golden-set promotion tooling (G7). Normative inputs: the S-tier roadmap in the audit record; `PLAN-agent-tools.md` § P2; `retrieval-measurement-gate` (the harness every gate reads); `mount-retrieval-graph` (the spans slice and reserved registration point); `docs/quarantine-ledger.md` (two rows added by this change).
