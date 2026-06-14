## 1. Redis Persistence Models — Pydantic Conversion

- [x] 1.1 Convert `CircuitBreakerState` TypedDict to Pydantic `BaseModel` with `frozen=True`, keeping `state: str`, `failures: int`, `opened_at: float | None = None`
- [x] 1.2 Convert `IdempotencyRecord` TypedDict to Pydantic `BaseModel` with `frozen=True`, `status: IdempotencyStatus`, `task_id: str | None = None`, `updated_at: str`, `metadata: JsonMetadata`
- [x] 1.3 Delete `RawCircuitBreakerState` TypedDict (no longer needed — Pydantic handles raw deserialization via `model_validate_json`)
- [x] 1.4 Remove `default_circuit_breaker_state()`, `parse_circuit_breaker_state()`, `build_open_circuit_breaker_state()`, `build_closed_circuit_breaker_state()`, `build_half_open_circuit_breaker_state()` functions
- [x] 1.5 Update `get_circuit_breaker_state()` to use `CircuitBreakerState.model_validate_json(payload)` instead of `parse_circuit_breaker_state()`, and `CircuitBreakerState()` instead of `default_circuit_breaker_state()`
- [x] 1.6 Update `set_circuit_breaker_state()` to use `.model_dump_json()` instead of `json.dumps(state)`
- [x] 1.7 Update `is_circuit_breaker_open()` to use attribute access (`state.failures`, `state.opened_at`) instead of dict subscript (`state["failures"]`, `state["opened_at"]`)
- [x] 1.8 Update `record_circuit_breaker_failure()` to use attribute access and direct `CircuitBreakerState(...)` construction
- [x] 1.9 Update `get_idempotency_status()` to use `IdempotencyRecord.model_validate_json(payload)` instead of `json.loads()` + cast
- [x] 1.10 Update `serialize_idempotency_record()` to construct `IdempotencyRecord(...)` and use `.model_dump_json()` instead of `json.dumps(record)`
- [x] 1.11 Remove `parse_idempotency_status()` helper (Pydantic field validation replaces it — use Literal type directly)
- [x] 1.12 Run `uv run ruff check src/app/connections/celery_reliability.py` and `uv run ty check src/app/connections/celery_reliability.py` to verify zero new issues

## 2. LangGraph Boundary Validation

- [x] 2.1 Define Pydantic `LegalAgentInputState` and `LegalAgentOutputState` in `agent_saul/state.py` with minimal required fields
- [x] 2.2 Wire `input_schema=LegalAgentInputState` and `output_schema=LegalAgentOutputState` into `StateGraph(LegalAgentState, ...)` in `agent_saul/graph.py`
- [x] 2.3 Define Pydantic `SupervisorOutputState` in `open_deep_search/state.py` for supervisor graph boundary
- [x] 2.4 Wire `output_schema=SupervisorOutputState` into supervisor `StateGraph(...)` in `open_deep_search/graph.py`
- [x] 2.5 Verify `ResearcherOutputState(BaseModel)` is already properly wired as output_schema for ResearcherState (confirm existing pattern)
- [x] 2.6 Run `uv run ruff check` and `uv run ty check` across modified graph files

## 3. CODE-QUALITY-PATTERNS — Secondary Improvements

- [x] 3.1 Replace `@lru_cache(maxsize=1)` with `@cache` on zero-argument factories in `settings.py`, `pageindex/client.py`, `agents/registry.py`, plus `tavily.py`, `httpx_client.py`, `mcp/registry.py`, `mcp/client.py`
- [x] 3.2 Replace manual `itertools.pairwise` implementations — no manual implementations found, no-op
- [x] 3.3 Remove `# type: ignore[override]` comments where no override occurs — all 5 are legitimate, no-op
- [x] 3.4 Import `Callable` from `collections.abc` — already in use everywhere, no-op
- [x] 3.5 Replace `os.path` usage with `pathlib.Path` in `document_processing/ingest_v2.py` and `document_processing/ingest.py`
- [x] 3.6 Run `uv run ruff check` and `uv run ty check` — zero new errors introduced

## 4. Final Verification

- [x] 4.1 Run `uv run ruff check` on all modified files — zero errors
- [x] 4.2 Run `uv run ty check` on all modified files — zero new errors
- [x] 4.3 Verify all existing tests pass for celery_reliability module — no tests exist, no-op
- [x] 4.4 Verify all existing tests pass for LangGraph graph modules — no tests exist, no-op
