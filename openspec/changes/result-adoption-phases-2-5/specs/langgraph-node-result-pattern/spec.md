## ADDED Requirements

### Requirement: Ingestion node helpers return AppResult
Sync helper functions in `ingestion_kb/nodes.py` SHALL return `AppResult[T]` using `Success` / `Failure` instead of raising exceptions or returning plain error values. The `_ingestion_failure` helper SHALL return `AppError` instances wrapped in `Failure`. Node entrypoints SHALL remain ordinary async functions returning plain dicts.

#### Scenario: Validation helper returns Success on valid input
- **WHEN** a validation helper receives valid input
- **THEN** it returns `Success(validated_data)`

#### Scenario: Validation helper returns ValidationAppError on invalid input
- **WHEN** a validation helper receives invalid input
- **THEN** it returns `Failure(ValidationAppError(...))`

#### Scenario: _ingestion_failure returns Failure(AppError)
- **WHEN** a node encounters a recoverable error during processing
- **THEN** `_ingestion_failure` returns `Failure(AppError(...))` with appropriate subtype

#### Scenario: Node entrypoint unwraps helper Results to plain dicts
- **WHEN** a node entrypoint calls a Result-returning helper and receives `Success(data)`
- **THEN** it returns the data embedded in a plain dict for state update
- **WHEN** a node entrypoint receives `Failure(error)`
- **THEN** it returns `{"failure": error}` (existing state contract preserved)

### Requirement: Reconciliation node helpers return AppResult
Sync helper functions in `reconciliation/nodes.py` SHALL return `AppResult[T]`. The existing pattern of accumulating `failures: list[AppError]` in state SHALL remain unchanged. Node entrypoints SHALL remain ordinary async functions returning plain dicts.

#### Scenario: Reconciliation helper returns Success on completion
- **WHEN** a reconciliation step completes successfully
- **THEN** it returns `Success(result_data)`

#### Scenario: Reconciliation helper returns InfrastructureAppError on failure
- **WHEN** a reconciliation step encounters a DB or network error
- **THEN** it returns `Failure(InfrastructureAppError(...))`

#### Scenario: Node entrypoint maps Failure to list[AppError] for state
- **WHEN** a node entrypoint receives `Failure(error)`
- **THEN** it adds `error` to the `failures` list in the returned state dict, preserving the `operator.add` accumulation pattern

### Requirement: log_expected_failure called at Result boundary
Every `Failure` that is mapped back to a plain dict at the node boundary SHALL be logged via `log_expected_failure`.

#### Scenario: Failure is logged at node boundary
- **WHEN** a node entrypoint receives `Failure(error)` from a helper
- **THEN** `log_expected_failure(error, operation="<node_name>")` is called before returning the state dict
