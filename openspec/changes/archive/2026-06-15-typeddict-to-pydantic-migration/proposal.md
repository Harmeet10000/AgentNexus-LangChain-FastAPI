## Why

The codebase has 8 TypedDicts across two distinct contexts: Celery/Redis persistence models that cross serialization boundaries without runtime validation, and LangGraph state schemas that follow the recommended interior-state pattern. The persistence models (`IdempotencyRecord`, `CircuitBreakerState`, `RawCircuitBreakerState`) are manually parsed and constructed with dict literals — error-prone when Redis data drifts or corrupts. The LangGraph state schemas should stay TypedDict per LangGraph best practice, but benefit from Pydantic at graph input/output boundaries, which is already partially modeled (`ResearcherOutputState`).

## What Changes

- **Redis persistence models**: Convert `IdempotencyRecord`, `CircuitBreakerState` from TypedDict to Pydantic `BaseModel`. Delete `RawCircuitBreakerState` (it exists only because the current TypedDict can't handle raw deserialization). Replace all manual builder/parse functions with `model_validate()` and direct construction.
- **Celery task return types**: Keep `DecayStats`, `ReconciliationSummary` as TypedDict — they are purely internal return types with no serialization boundary. No change.
- **LangGraph state schemas**: Keep `SupervisorState`, `ResearcherState`, `LegalAgentState` as TypedDict. Add Pydantic `input_schema`/`output_schema` to graph boundaries where missing. Formalize the pattern already started with `ResearcherOutputState`.
- **Combine with CODE-QUALITY-PATTERNS.md**: Also apply `functools.cache`, `itertools.pairwise`, `match`/guards, `pathlib` standardization as secondary improvements where they naturally fit alongside the migration.

## Capabilities

### New Capabilities
- `redis-persistence-models`: Convert CircuitBreakerState, IdempotencyRecord to Pydantic BaseModel with model_validate(), remove RawCircuitBreakerState and manual builder/parse functions
- `langgraph-boundary-validation`: Add Pydantic input/output schemas to LangGraph state boundaries, formalize pattern across all graph definitions

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- `src/app/connections/celery_reliability.py`: Core changes — 3 TypedDicts, ~5 builder/parse functions replaced
- `src/app/shared/langgraph_layer/agent_saul/state.py` + `graph.py`: Add Pydantic boundary schemas
- `src/app/shared/langgraph_layer/open_deep_search/state.py` + `graph.py`: Formalize existing ResearcherOutputState pattern
- No API changes, no dependency changes, no breaking changes to consumers
