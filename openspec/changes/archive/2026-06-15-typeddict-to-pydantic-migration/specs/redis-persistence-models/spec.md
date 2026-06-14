## ADDED Requirements

### Requirement: CircuitBreakerState as Pydantic BaseModel

The `CircuitBreakerState` SHALL be a Pydantic `BaseModel` with `state: str`, `failures: int`, `opened_at: float | None = None`. It MUST support `model_validate()` from a dict and `model_validate_json()` from a Redis JSON string. The `RawCircuitBreakerState` TypedDict SHALL be removed.

#### Scenario: Construct closed state
- **WHEN** constructing a closed circuit breaker state
- **THEN** `CircuitBreakerState(state="closed", failures=0)` creates a valid instance with `opened_at=None`

#### Scenario: Construct open state
- **WHEN** constructing an open circuit breaker state
- **THEN** `CircuitBreakerState(state="open", failures=3, opened_at=1234567890.0)` creates a valid instance

#### Scenario: Deserialize from Redis JSON
- **WHEN** loading the JSON string `{"state": "open", "failures": 3, "opened_at": 1234567890.0}` from Redis
- **THEN** `CircuitBreakerState.model_validate_json(...)` returns a valid `CircuitBreakerState`

#### Scenario: Deserialize partial data from Redis
- **WHEN** loading the JSON string `{}` from Redis (missing keys)
- **THEN** `CircuitBreakerState.model_validate_json(...)` raises a validation error for missing required fields

#### Scenario: Invalid failures type
- **WHEN** loading `{"state": "closed", "failures": "abc"}` from Redis
- **THEN** `CircuitBreakerState.model_validate_json(...)` raises a validation error

### Requirement: IdempotencyRecord as Pydantic BaseModel

The `IdempotencyRecord` SHALL be a Pydantic `BaseModel` with `status: IdempotencyStatus`, `task_id: str | None = None`, `updated_at: str`, `metadata: JsonMetadata = {}`. It MUST support `model_validate()` from a dict and `model_validate_json()` from a Redis JSON string.

#### Scenario: Construct processing record
- **WHEN** constructing a processing idempotency record
- **THEN** `IdempotencyRecord(status="processing", task_id="abc-123", updated_at="2025-01-01T00:00:00", metadata={"source": "webhook"})` creates a valid instance

#### Scenario: Deserialize from Redis JSON
- **WHEN** loading valid JSON from Redis
- **THEN** `IdempotencyRecord.model_validate_json(...)` returns a valid `IdempotencyRecord`

#### Scenario: Invalid status value
- **WHEN** loading `{"status": "unknown", "updated_at": "...", "metadata": {}}` from Redis
- **THEN** validation error is raised

### Requirement: Remove manual builder and parse functions

The following helper functions in `celery_reliability.py` SHALL be removed: `default_circuit_breaker_state()`, `parse_circuit_breaker_state()`, `build_open_circuit_breaker_state()`, `build_closed_circuit_breaker_state()`, `build_half_open_circuit_breaker_state()`. Their callers MUST be updated to use Pydantic `model_validate()` / direct construction instead.

#### Scenario: default state replaced by constructor defaults
- **WHEN** a default state is needed
- **THEN** use `CircuitBreakerState(state="closed", failures=0)` instead of `default_circuit_breaker_state()`

#### Scenario: parse replaced by model_validate_json
- **WHEN** parsing a circuit breaker state from Redis JSON
- **THEN** use `CircuitBreakerState.model_validate_json(payload)` instead of `parse_circuit_breaker_state(payload)`

#### Scenario: builder functions replaced by direct construction
- **WHEN** building open/closed/half-open states
- **THEN** use `CircuitBreakerState(...)` with appropriate field values instead of builder functions
