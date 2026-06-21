## ADDED Requirements

### Requirement: Dedicated deep research graph node

Agent Saul SHALL have a `deep_research` graph node that invokes the ODS compiled subgraph to answer web research questions.

#### Scenario: Orchestrator routes to deep research node

- **WHEN** the orchestrator emits `OrchestratorAction(action_type=CONTINUE, target_node="deep_research")`
- **THEN** the graph transitions to the `deep_research` node

#### Scenario: Node runs all pending SEARCH_PRECEDENTS steps

- **WHEN** the `deep_research` node executes
- **THEN** it reads `plan` from state, filters steps where `action == SEARCH_PRECEDENTS`, and runs ODS concurrently for each
- **THEN** it advances `current_step` to the first non-SEARCH_PRECEDENTS step
- **THEN** it routes back to the orchestrator

#### Scenario: Research results stored in state

- **WHEN** ODS completes for all research steps
- **THEN** the concatenated final reports are stored in `LegalAgentState.deep_research_results`
- **THEN** downstream nodes (risk_analysis, compliance) MAY reference `deep_research_results` from state

### Requirement: ODS tool built at graph construction time

The ODS compiled subgraph SHALL be wrapped via `make_deep_research_tool()` at graph construction time, not inside the node.

#### Scenario: Tool built in factory

- **WHEN** `build_saul_graph()` is called
- **THEN** it accepts a pre-built `deep_research_tool` parameter
- **THEN** the tool is stored on `SaulGraphNodes.deep_research`
- **THEN** the `deep_research` node calls the tool, not the raw subgraph

### Requirement: Research failures are non-fatal

If ODS fails for any research step, the node SHALL log the error and continue rather than failing the pipeline.

#### Scenario: ODS fails gracefully

- **WHEN** `deep_research_tool.ainvoke()` raises for one or more steps
- **THEN** the error is logged with `logger.bind(...).warning("deep_research_step_failed")`
- **THEN** the node continues with partial results
- **THEN** the pipeline continues to the next node

### Requirement: Research results referenceable by risk and compliance nodes

Risk analysis and compliance nodes SHALL have access to the research results stored in `LegalAgentState.deep_research_results`.

#### Scenario: Risk analysis references research

- **WHEN** the risk analysis node runs after deep_research has completed
- **THEN** `state["deep_research_results"]` contains the research string
- **THEN** the risk agent's system prompt includes instructions to reference these findings
