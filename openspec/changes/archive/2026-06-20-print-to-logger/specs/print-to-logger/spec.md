## ADDED Requirements

### Requirement: No print() calls in application code
All `print()` calls within `src/app/` SHALL be replaced with structured loguru logger calls. The project SHALL use `from loguru import logger` as the logging interface.

#### Scenario: CLI script uses logger for user-facing output
- **WHEN** a CLI script (e.g., `rag_agent_advanced.py`, `ingest.py`) displays status to the user
- **THEN** it SHALL use `logger.info()` instead of `print()`

#### Scenario: Debug/progress output uses logger.debug
- **WHEN** a CLI script displays detailed progress (chunk counts, iteration status, timing)
- **THEN** it SHALL use `logger.debug()` instead of `print()`

#### Scenario: Error output uses logger.error
- **WHEN** a CLI script displays error messages
- **THEN** it SHALL use `logger.error()` instead of `print()`

### Requirement: Streaming output is buffered before logging
For code that streams tokens with `end=""` and `flush=True`, the response SHALL be buffered and logged as a complete message rather than token-by-token.

#### Scenario: Streaming LLM response is logged as complete message
- **WHEN** a streaming response finishes
- **THEN** the full response text SHALL be logged via `logger.info()` as a single call

### Requirement: No new dependencies introduced
This change SHALL NOT introduce any new package dependencies. loguru is already a project dependency.

#### Scenario: Dependency list unchanged
- **WHEN** `uv sync` is run after the change
- **THEN** no new packages appear in `pyproject.toml` or lock file
