> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

Deep research is currently one scoped `EvidenceService` adapter call inside `make_deep_research_node`. The evidence seam (`search_evidence` / `read_evidence_by_id` / `get_ingestion_state`) already enforces server-owned scoping via `trusted_tool_context`; the deep-research rebuild composes those calls instead of bypassing them. `mount-retrieval-graph` establishes the repo precedent for wrapping a compiled subgraph (its qna engine) — this change applies the same pattern at tool granularity, where reference 03's compiled-subgraph-tool rule governs the boundary.

## Goals / Non-Goals

**Goals:**
- Deep research gains a bounded plan→search→read→synthesize loop with zero toolset growth.
- V6 (ToolRuntime inside sub-agents) closes *by construction* — the only path into the subgraph is explicit state construction from server-owned context.
- Partial progress is always truthfully labelled (`budget_exhausted` with handles), never dressed as an answer.

**Non-Goals:**
- Any model-visible toolset growth (the projection stays at one research tool + transfer).
- Interrupts, memory writes, or LLM-judge semantics inside the subgraph.
- Changing the evidence contracts themselves — the subgraph is a consumer.

## Decisions

### Decision 1 — compile once at graph build, invoke per call, no internal checkpointer
The subgraph is compiled when Saul's graph factory runs and invoked inside the tool call. No checkpointer inside: the run is bounded and short-lived, per-call state dies with the call, and this matches the mounted retrieval graph's no-checkpointer decision for hot-path invocations. Resume across a deep-research call is a re-invocation, not a checkpoint replay.

### Decision 2 — the wrapper is the only door, and it builds the state
`DeepResearchState` is constructed by the wrapper from `ToolRuntimeContext` (tenant, `allowed_doc_ids`, run/thread ids). The subgraph's nodes read scope only from state the wrapper wrote. This makes V6's propagation test a *design consequence*, not a hope: there is no ambient path for model-supplied scope to travel.

### Decision 3 — the step budget draws from Saul's ledger (harden 6.11)
The subgraph's hard step/recursion bound is backed by the same per-run ledger the harden change introduces; exhaustion mid-research yields `budget_exhausted` with partial handles. No fresh allowance is minted inside the subgraph — the ledger is handed in, not recreated.

### Decision 4 — handles stay run-scoped because the context already carries `run_id`
`ToolRuntimeContext.run_id` scopes the evidence-handle membership the subgraph inherits from the evidence seam. The change adds no new scoping mechanism; its tests prove the inherited scoping holds across the tool boundary, including on resumed threads.

### Decision 5 — D3 is a binding change, not a capability change
Risk's `get_ingestion_state` reuses the compliance branch's existing factory pattern verbatim; the only new behaviour is the disclosure requirement on risk output, which the delegation spec carries.

### Decision 6 — refine, don't fork, harden task 6.9
6.9's "document-scoped adapter" becomes this subgraph's search stage; the argument-mismatch fix it demands is subsumed by the wrapper's typed `args_schema`. The harden change's task remains the umbrella; this change delivers its refined form and notes the supersession in its tasks.

## Risks / Trade-offs

- **Subgraph-in-tool is the newest pattern in the repo** → the V6 propagation test, the no-interrupt test, and the budget test are the three failure modes made executable; all three are first-class acceptance cases.
- **Synthesis quality is unchanged by this change** → the bounded loop changes *how* evidence is gathered and bounded, not how it is judged; quality work belongs to the validation-layer changes.

## Migration Plan

Single consumer (`make_deep_research_node` + the factory binding). The old adapter call is replaced in the same commit that lands the wrapper; no deprecation window is needed pre-launch.

## Open Questions

None — the 2026-10-25 review decision plus reference 03 settle the shape.
