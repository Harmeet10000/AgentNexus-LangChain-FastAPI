## 1. State changes

- [ ] 1.1 Add `deep_research_results: str | None` field to `LegalAgentState` in `state.py`
- [ ] 1.2 Add `"deep_research"` to `_VALID_WORKER_NODES` in `nodes.py`
- [ ] 1.3 Add `"deep_research"` to `GRAPH_NODE_NAMES` in `state.py`

## 2. Node implementation

- [ ] 2.1 Create `make_deep_research_node(deep_research_tool: BaseTool) -> StateNode` factory in `nodes.py`
- [ ] 2.2 Node reads `plan` from state, filters to `SEARCH_PRECEDENTS` steps at and after `current_step`
- [ ] 2.3 Node runs all research steps concurrently via `asyncio.gather(*[tool.ainvoke({"question": step.description}) for step in research_steps])`
- [ ] 2.4 Node concatenates results (join with `\n\n---\n\n`) and stores in `deep_research_results`
- [ ] 2.5 Node advances `current_step` past the last consumed SEARCH_PRECEDENTS step
- [ ] 2.6 Wrap in try/except, log failures with `logger.bind(...).warning("deep_research_step_failed")`, continue with partial results

## 3. Graph wiring

- [ ] 3.1 Add `deep_research: Any` field to `SaulGraphNodes` in `factory.py`
- [ ] 3.2 Wire `deep_research=make_deep_research_node(deep_research_tool)` in `_build_graph_nodes()`
- [ ] 3.3 Add `deep_research_tool: BaseTool` parameter to `build_saul_graph()` in `graph.py`
- [ ] 3.4 Pass `deep_research_tool` through to `_build_graph_nodes()`
- [ ] 3.5 Add `graph.add_node("deep_research", nodes.deep_research)` in `_wire_graph()`
- [ ] 3.6 Add `"deep_research": "deep_research"` to the `route_from_orchestrator` conditional edges map
- [ ] 3.7 Add `graph.add_edge("deep_research", "orchestrator")` — route back after research

## 4. Tool construction in registry

- [ ] 4.1 Update `registry.py` docstring to show `make_deep_research_tool(http_client=...)` construction + passing to `build_saul_graph()`
- [ ] 4.2 Ensure `build_saul_graph` in registry.py call site passes the tool

## 5. Verification

- [ ] 5.1 Run `uv run ruff check src/app/shared/langgraph_layer/agent_saul/`
- [ ] 5.2 Run `uv run ty check src/app/shared/langgraph_layer/agent_saul/`
- [ ] 5.3 Verify no new lint/type errors introduced
