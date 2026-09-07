## MODIFIED Requirements

### Requirement: Agent tools SHALL catch OS-level and library-specific exceptions

Agent tool operations SHALL catch specific exceptions instead of bare `except Exception`:
- `OSError` (and subclasses `FileNotFoundError`, `PermissionError`) for filesystem operations
- `redis.exceptions.RedisError` for Redis operations
- `langchain_core.exceptions.LangChainException` for LLM operations
- `subprocess.SubprocessError` for subprocess execution failures

Each catch site SHALL add `exc.add_note()` with the command, path, or operation context.

The result these catch sites return SHALL be the single tool-result envelope defined by the `agent-tool-contract`
capability, constructed through its failure constructor. The previously-named `ToolOutput` envelope
(`shared/langchain_layer/agents/tools/base.py:30`) is one of four competing envelope definitions and is removed by
the `agent-tool-contract` collapse, so scenarios below name the surviving envelope's failure constructor instead of
`ToolOutput.fail()`.

A catch site SHALL return that envelope as a value. It SHALL NOT render the envelope to a human-readable string
before returning — the removed `ToolOutput.to_agent_string()` returned `f"ERROR: {self.error}"`, which is the
string-as-error anti-pattern the `agent-tool-contract` capability forbids.

Where the failure means the tool's backing corpus, index, or graph could not be reached, the catch site SHALL use
the envelope's unavailability constructor rather than its generic failure constructor, so that unavailability is
never reported as absence.

#### Scenario: Shell command failure catches OSError
- **WHEN** a shell command execution fails due to an OS error
- **THEN** the code catches `OSError`, adds a note with the command and working directory, and returns a failure result in the single tool-result envelope

#### Scenario: File not found catches FileNotFoundError
- **WHEN** a file read operation fails because the file doesn't exist
- **THEN** the code catches `FileNotFoundError`, adds a note with the file path, and returns a failure result in the single tool-result envelope

#### Scenario: Permission denied catches PermissionError
- **WHEN** a file or directory operation fails because of insufficient permissions
- **THEN** the code catches `PermissionError`, adds a note with the path and required permission, and returns a failure result in the single tool-result envelope

#### Scenario: Redis cache failure catches RedisError
- **WHEN** a tool's Redis cache operation fails
- **THEN** the code catches `redis.exceptions.RedisError`, adds a note with the key and operation, and continues without cache

#### Scenario: LLM call failure catches LangChainException
- **WHEN** a tool's LLM call fails
- **THEN** the code catches `langchain_core.exceptions.LangChainException`, adds a note with the model and operation, and returns a failure result in the single tool-result envelope

#### Scenario: Subprocess failure catches SubprocessError
- **WHEN** a subprocess spawned by a tool fails
- **THEN** the code catches `subprocess.SubprocessError`, adds a note with the command and return code, and returns a failure result in the single tool-result envelope

#### Scenario: A catch site returns the envelope rather than a rendered string
- **WHEN** any of the catch sites above returns its result to the caller
- **THEN** it SHALL return the envelope value itself
- **AND** it SHALL NOT return a rendered error sentence in place of the envelope
