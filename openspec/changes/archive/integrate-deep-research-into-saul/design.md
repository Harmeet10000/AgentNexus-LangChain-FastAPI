## Context

Agent Saul is a compiled LangGraph with 14 pipeline nodes and a structured-output orchestrator. The Open Deep Search (ODS) subgraph (`deep_researcher`) already exists as a standalone compiled graph in `open_deep_search/` and is wrapped as a `StructuredTool` via `make_deep_research_tool()`. Currently Agent Saul's `create_agent` instances (orchestrator, risk, compliance) have `tools=[]` — no web research capability.

The orchestrator already supports `CONTINUE(target_node=...)` for dynamic routing, and the planner already defines `PlanActionType.SEARCH_PRECEDENTS`. The infrastructure to inject research is 80% scaffolded — what's missing is the graph node, state field, and wiring.

## Graph topology (before vs after)

```
BEFORE:                              AFTER:
                                      ┌──────────────┐
                                      │ deep_research │
                                      └┬─────────────┬┘
                                       │             │
                                       ▼             │
orchestrator ──CONTINUE──→ ingestion  orchestrator ──┼──→ ingestion
       │                             │   ^           │
       ├──→ finalization             ├───┘           │
       └──→ END                      ├──→ finalization
                                      └──→ END
```

## Goals / Non-Goals

**Goals:**
- Add `deep_research` as a routable LangGraph node activated by `OrchestratorActionType.CONTINUE(target_node="deep_research")`
- Store research output as `LegalAgentState.deep_research_results: str | None`
- Batch all remaining `SEARCH_PRECEDENTS` plan steps in one node invocation using `asyncio.gather`
- Advance `current_step` past consumed research steps before returning to orchestrator
- Log + skip on ODS failure — research is additive, not pipeline-critical

**Non-Goals:**
- Adding ODS as a tool on risk/compliance `create_agent` instances (those get dedicated tools in a separate change)
- Uncommenting Agent Saul lifespan wiring (out of scope)
- Modifying the ODS subgraph itself (it works as-is)
- Replacing the existing `ToolRegistry` pattern

## Decisions

### Dedicated graph node over ToolNode

The orchestrator is a structured-output LLM (`with_structured_output`), not a `create_agent` — it cannot call tools directly. A dedicated node keeps the research step explicit, checkpoint-visible, and testable. This follows the skill guidance: "Prefer explicit LangGraph orchestration for complex workflows instead of burying a full agent loop inside a graph node."

### Tool built at graph construction time

`make_deep_research_tool(http_client=...)` is called once in `build_saul_graph()` and stored on `SaulGraphNodes`. The HTTP client is baked into the tool closure — the node never needs to see it. This avoids passing HTTP clients through `RunnableConfig` at node execution time and keeps the node's signature clean.

**Alternative considered:** Passing `http_client` via `RunnableConfig` from the outer graph. Rejected because the tool already encapsulates this pattern internally (the tool calls `deep_researcher.ainvoke(..., config={...})` with the baked-in client).

### Batch all SEARCH_PRECEDENTS steps in one node call

The node reads `plan`, filters to `SEARCH_PRECEDENTS` steps, runs `asyncio.gather` across all of them, concatenates results, and sets `current_step` past the last consumed step. This is faster than looping back to the orchestrator between each research call and produces fewer LangGraph transitions.

**Alternative considered:** One step per invocation via orchestrator loop. Rejected — adds N extra graph transitions and checkpoint operations for no benefit, since research steps have no intra-step dependencies.

### State field over working_memory

`deep_research_results: str | None` is added to `LegalAgentState` as a typed field. Unlike `working_memory` (ephemeral, untyped), this is visible in checkpoint snapshots, survives replay, and can be referenced by downstream nodes with full type safety.

### Orchestrator routing unchanged

`route_from_orchestrator` already handles `CONTINUE` with `action.target_node`. Adding `"deep_research": "deep_research"` to the conditional edges map is the only change needed. After research completes, a static edge `deep_research → orchestrator` returns control.

## Risks / Trade-offs

- **ODS is expensive** (multiple LLM calls per research question). Batching all steps concurrently could cause token rate limits. Mitigation: ODS internally caps concurrent research units (`max_concurrent_research_units`) — the batched calls still respect this per-invocation.
- **Plan step dependencies ignored**: `SEARCH_PRECEDENTS` steps may list `depends_on` pointing to other steps. Mitigation: web research is inherently independent. If cross-step dependencies are needed later, the node can be changed to sequential execution.
- **Research results size**: ODS reports can be long. `deep_research_results` stores concatenated text. Mitigation: if size becomes an issue, add truncation at the node boundary.
