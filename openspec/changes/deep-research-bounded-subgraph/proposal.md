> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

**M** — confined to Agent Saul's tooling (`features/agent_saul/` + the evidence seam it already consumes). It changes one role's toolset composition and rebuilds one capability behind an existing boundary; no HTTP surface, no migration, no new dependency. It refines `harden-agent-saul-evidence-pipeline` task 6.9 (which specced a plain adapter) into the compiled-subgraph decision.

## Why

Deep research today is a single scoped `EvidenceService` adapter call: one search, no bounded read-back loop, no step budget, and a partial result that reads as a complete answer. The 2026-10-25 review decision rebuilds it as a *plan-conditional adapter over `search_evidence` / `read_evidence_by_id`* — which is a subgraph, not a bigger prompt — exposed as exactly **one** tool so the deep_research role's projection stays flat (P4/§3 of `PLAN-agent-tools.md`). The same change closes check **V6** (ToolRuntime propagation into sub-agents) by construction: the wrapper builds the subgraph's initial state from server-owned runtime context, so ambient or model-supplied identity cannot leak in. D3 rides along: risk conclusions must disclose incomplete documents, and compliance already has that disclosure.

## What Changes

- **`deep_research` becomes a compiled bounded subgraph exposed as one tool.** `plan → search_evidence → read_evidence_by_id (handles only) → synthesize`, with a hard step/recursion budget. One `@tool` wrapper with an explicit Pydantic `args_schema` (`ConfigDict(extra="forbid")`): `objective` plus optional `focus` — nothing else is model-visible.
- **Context by explicit construction, never ambient.** `allowed_doc_ids`, tenant, and identity come from `ToolRuntime[ToolRuntimeContext]`, injected and hidden; the wrapper constructs the subgraph's initial state from them. A model-supplied tenant/scope never reaches the subgraph (V6's propagation test becomes this change's first acceptance case).
- **No HITL inside the subgraph.** Interrupts do not cross a tool boundary; approval stays on Saul's orchestration graph.
- **Bounded envelope with typed status.** Returns synthesized summary + evidence handles + coverage + `status: completed | budget_exhausted | no_evidence | ingestion_incomplete`. Budget exhaustion is a typed result carrying partial handles — never a confident partial answer.
- **Run-scoped handles.** An evidence handle issued in run N fails handle-membership in run N+1.
- **Registry + projection.** The wrapper registers in the single registry with read-only authority tags (per `agent-tool-capability-enforcement`); the deep_research role's projection stays at one visible tool.
- **Risk parity (D3).** `get_ingestion_state` joins the risk role's toolset so risk conclusions disclose incomplete documents exactly as compliance's do.

## Scope / Non-Goals

In scope: the subgraph and its wrapper, the V6 propagation test, the budget/status envelope, handle run-scoping, registry registration, risk's `get_ingestion_state`, and the plan-level execution test.

Out of scope (explicit): any memory write inside deep research (read-only by P3; writers are the harden change's task 6.10); HITL inside the subgraph; portfolio/cross-document traversal (D2 — gated on graph edges, separate change); qna's retrieval-graph engine (`mount-retrieval-graph`); changes to search/read evidence contracts themselves (the `saul-memory-prefetch-and-retrieval` / evidence seam is consumed, not modified).

## Capabilities

### New Capabilities

- `deep-research-subgraph`: the bounded plan→search→read→synthesize machine behind one tool — server-owned context construction, hard step budget, typed terminal statuses, run-scoped handles, no interrupts.

### Modified Capabilities

- `agent-saul-delegation`: the deep_research role's toolset is exactly one research tool (the subgraph wrapper) plus its transfer; the risk role gains `get_ingestion_state` so its conclusions disclose ingestion completeness identically to compliance.

## Impact

Affected code: `src/app/features/agent_saul/graph/nodes.py` (`make_deep_research_node` internals → subgraph build), `src/app/features/agent_saul/graph/factory.py` (binding: one wrapper replaces the adapter call), a new `DeepResearchState` + graph module under the agent_saul graph package, `src/app/features/documents/tools/` (wrapper registration + `get_ingestion_state` in the risk binding), tests: V6 propagation, budget, handle-scope, plan-level execution. Normative inputs: `PLAN-agent-tools.md` § D5 and D3; reference `03-tools-and-toolruntime.md` (compiled-subgraph-tool rule); harden change task 6.9 (refined by this change) and 6.11 (ledger the step budget draws from).
