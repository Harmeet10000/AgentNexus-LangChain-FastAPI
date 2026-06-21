## Why

Agent Saul's risk and compliance agents have empty tool lists (`# TODO`) and the orchestrator cannot dispatch web research at all. The Open Deep Search (ODS) compiled subgraph already exists in the repo — wrapped as a `StructuredTool` via `make_deep_research_tool()` — but nothing wires it into Agent Saul. Without it, the pipeline cannot ground legal analysis in current external sources (precedents, statutes, news).

## What Changes

- Add a `deep_research` node to the Agent Saul LangGraph, routed from the orchestrator via `CONTINUE(target_node="deep_research")`.
- Add `deep_research_results: str | None` field to `LegalAgentState` for typed, checkpoint-visible storage.
- Batch all remaining `SEARCH_PRECEDENTS` plan steps in one invocation and advance `current_step` past them.
- Accept `deep_research_tool` as a parameter to `build_saul_graph()` / `SaulGraphNodes` so the node gets the pre-built tool.
- Update lifespan wiring docs (and uncommented wiring when ready) to construct the tool with the lifespan-managed HTTP client.
- Failures: log + skip + continue (research is additive, not critical).

## Capabilities

### New Capabilities
- `deep-research`: Deep web research from within Agent Saul. The orchestrator dispatches to a dedicated graph node that runs the ODS compiled subgraph against all pending `SEARCH_PRECEDENTS` plan steps concurrently. Results stored in `LegalAgentState.deep_research_results` for downstream nodes (risk analysis, compliance) to reference.

### Modified Capabilities
- *(none — Agent Saul is not yet live in lifespan)*

## Impact

- `src/app/shared/langgraph_layer/agent_saul/state.py` — new state field
- `src/app/shared/langgraph_layer/agent_saul/nodes.py` — new node factory
- `src/app/shared/langgraph_layer/agent_saul/factory.py` — add to `SaulGraphNodes`
- `src/app/shared/langgraph_layer/agent_saul/graph.py` — wire node + edge; accept tool in `build_saul_graph`
- `src/app/shared/rag/graphiti/registry.py` — docstring update with tool construction pattern
- `src/app/shared/langgraph_layer/open_deep_search/tools.py` — no changes needed (tool already works)
