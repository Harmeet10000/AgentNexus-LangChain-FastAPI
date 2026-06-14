## Context

The codebase uses TypedDict in two distinct ways:

1. **Redis persistence models** (`celery_reliability.py`): `IdempotencyRecord`, `CircuitBreakerState`, and `RawCircuitBreakerState` are serialized to/from Redis JSON. Currently uses manual dict construction (4 builder functions) and manual parsing with raw object casts (2 parse functions). `RawCircuitBreakerState` exists purely as an intermediate "unvalidated" type because TypedDict can't do proper runtime coercion.

2. **LangGraph state schemas** (`agent_saul/state.py`, `open_deep_search/state.py`): `SupervisorState`, `ResearcherState`, `LegalAgentState` are passed to `StateGraph(...)`. The skill reference explicitly recommends TypedDict for this use case. The `ResearcherOutputState(BaseModel)` pattern already exists as an example of Pydantic at boundaries.

The secondary improvements (`functools.cache`, `itertools.pairwise`, `match`/guards, `pathlib`) are opportunistic cleanups identified in `CODE-QUALITY-PATTERNS.md`.

## Goals / Non-Goals

**Goals:**
- Replace `CircuitBreakerState`, `IdempotencyRecord` with Pydantic `BaseModel` with runtime validation
- Delete `RawCircuitBreakerState` and all manual builder/parse functions
- Add Pydantic `input_schema`/`output_schema` at LangGraph state boundaries for `LegalAgentState`, `SupervisorState`, `ResearcherState`
- Apply `functools.cache` to no-arg factories, `itertools.pairwise` to adjacent-element loops, `match`/guards to mode dispatchers, `pathlib` to remaining `os.path` usage
- All changes must pass `ruff check` and `ty check` with zero new issues

**Non-Goals:**
- NOT converting `DecayStats`, `ReconciliationSummary` (Celery task return types — internal only, no boundary)
- NOT converting LangGraph state TypedDicts to Pydantic (against LangGraph best practice)
- NOT changing the `state.py` files' schema fields or reducer signatures
- NOT adding Pydantic to the entire codebase — targeted to the specific units identified

## Decisions

### Decision 1: Pydantic BaseModel for Redis models (not dataclass)
- **Chosen:** Pydantic `BaseModel` with `model_validate()` for Redis deserialization, direct construction for serialization
- **Alternatives considered:** Dataclass + manual validation (fragments validation logic); staying with TypedDict (status quo)
- **Rationale:** Pydantic gives free runtime validation when loading from Redis, catching data corruption. `model_validate_json()` directly replaces `json.loads()` + manual field-by-field parsing. Builder functions collapse to simple `Model(...)` calls.

### Decision 2: Pydantic boundary schemas for LangGraph, not interior state
- **Chosen:** Add distinct Pydantic `InputState`/`OutputState` models to `StateGraph(..., input_schema=..., output_schema=...)` where missing; keep interior TypedDict state
- **Alternatives considered:** Convert entire state to Pydantic (slower, output is not Pydantic per docs); keep everything TypedDict (no boundary validation)
- **Rationale:** LangGraph docs explicitly support this split — input/output get runtime validation, interior state stays fast. Already modeled by `ResearcherOutputState`.

### Decision 3: Keep DecayStats/ReconciliationSummary as TypedDict
- **Chosen:** No change
- **Rationale:** These are strictly internal Celery task return types. They are constructed once and consumed locally — no serialization boundary, no data corruption risk, no consumer that benefits from validation.

### Decision 4: Apply secondary patterns during migration
- **Chosen:** Apply `functools.cache`, `itertools.pairwise`, `match`/guards, `pathlib` improvements alongside the primary migration
- **Rationale:** These touch the same files or adjacent code. Doing them together reduces churn and lint cycles.

## Risks / Trade-offs

- **Pydantic performance overhead on Redis hot path** → Mitigation: CircuitBreakerState and IdempotencyRecord are checked once per task invocation, not in tight loops. Negligible impact.
- **LangGraph boundary schemas may need alignment with existing graph callers** → Mitigation: Callers already pass dicts that match the schema; Pydantic accepts dicts. No breaking change expected.
- **`functools.cache` on async factories** → Mitigation: `@cache` works on sync functions only. Async no-arg factories should use `@lru_cache` or a manual `functools.cache` on a sync wrapper. Verify each target.
- **Secondary patterns may introduce style inconsistencies** → Mitigation: Run `ruff check` and `ty check` after all changes. Keep secondary changes per-file alongside primary migration commits.
