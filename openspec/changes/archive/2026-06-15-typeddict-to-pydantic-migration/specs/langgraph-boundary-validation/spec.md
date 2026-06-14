## ADDED Requirements

### Requirement: Pydantic input/output schemas for LegalAgentState

The `LegalAgentState` graph (`agent_saul/graph.py`) SHALL use distinct `input_schema` and `output_schema` Pydantic models on its `StateGraph(...)` definition. The interior state SHALL remain `LegalAgentState(TypedDict)`.

#### Scenario: Input validation at graph boundary
- **WHEN** a caller invokes the graph with input that does not match the input schema
- **THEN** a validation error is raised at the boundary before any node executes

#### Scenario: Output schema filters internal state
- **WHEN** the graph completes
- **THEN** the returned dict contains only fields matching the output schema

### Requirement: Pydantic output schema for SupervisorState

The `SupervisorState` graph (`open_deep_search/graph.py`) SHALL use a Pydantic `output_schema` on its `StateGraph(...)` definition. The interior state SHALL remain `SupervisorState(TypedDict)`.

#### Scenario: Supervisor output is validated
- **WHEN** the supervisor graph completes
- **THEN** the output is validated against the Pydantic output schema

### Requirement: Formalize existing ResearcherOutputState pattern

The `ResearcherState` graph (`open_deep_search/graph.py`) already uses `ResearcherOutputState(BaseModel)` as `output_schema`. This requirement formalizes that the pattern SHALL be documented and consistent across all graphs.

#### Scenario: Pattern is documented
- **WHEN** a developer reads the state boundary code
- **THEN** the TypedDict-interior / Pydantic-boundary pattern is clear and consistent across all three graphs

### Requirement: Node-level Pydantic for structured I/O

Node functions that accept or return complex nested data structures (tool calls, LLM structured output, API responses) MAY use Pydantic models for internal node I/O. These are node-internal and NOT part of the graph state schema.

#### Scenario: Pydantic model for node I/O
- **WHEN** a node function processes a complex data payload
- **THEN** it MAY define a Pydantic model local to that node or shared across the module for validation
