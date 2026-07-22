## ADDED Requirements

### Requirement: Redis operations SHALL catch specific redis.exceptions types

All Redis operations in the codebase SHALL catch `redis.exceptions.RedisError` or its subclasses instead of bare `except Exception`. Each catch site SHALL add `exc.add_note()` with relevant context (key, operation, URL) before logging or returning fallback.

Import aliasing is REQUIRED for exceptions that shadow Python builtins: `from redis.exceptions import ConnectionError as RedisConnectionError` and `from redis.exceptions import TimeoutError as RedisTimeoutError`.

#### Scenario: Cache read failure catches RedisConnectionError
- **WHEN** a Redis cache read operation fails due to a connection error
- **THEN** the code catches `redis.exceptions.ConnectionError` (aliased as `RedisConnectionError`), adds a note with the cache key and operation, and returns `None` or fallback

#### Scenario: Cache read/write failure catches RedisTimeoutError
- **WHEN** a Redis operation fails due to a timeout
- **THEN** the code catches `redis.exceptions.TimeoutError` (aliased as `RedisTimeoutError`), adds a note with the cache key and timeout context, and logs the failure without crashing

#### Scenario: Redis response error catches ResponseError
- **WHEN** a Redis operation fails because the server returned an error reply (OOM, wrong type on key, read-only replica)
- **THEN** the code catches `redis.exceptions.ResponseError`, adds a note with the key and operation, and handles the specific failure (e.g., skip cache on OOM)

#### Scenario: Redis data error catches DataError
- **WHEN** a Redis operation fails due to a bad argument or type mismatch
- **THEN** the code catches `redis.exceptions.DataError`, adds a note with the key and value type, and logs the programming error

#### Scenario: Idempotency Redis write catches RedisError
- **WHEN** an idempotency state write to Redis fails
- **THEN** the code catches `redis.exceptions.RedisError`, adds a note with the idempotency key and tool name, and continues to Postgres fallback

### Requirement: HTTP operations SHALL catch specific httpx and crawl4ai exceptions

httpx operations SHALL catch `httpx.HTTPError` or its subclasses instead of bare `except Exception`. Crawl4AI/Playwright operations SHALL catch `playwright.async_api.Error` or `crawl4ai.utils.InvalidCSSSelectorError` instead of bare `except Exception`. Each catch site SHALL add `exc.add_note()` with the URL and operation context.

#### Scenario: HTTP client timeout catches httpx.TimeoutException
- **WHEN** an httpx request times out
- **THEN** the code catches `httpx.TimeoutException`, adds a note with the URL and timeout value, and returns fallback or retries

#### Scenario: HTTP connection failure catches httpx.ConnectError
- **WHEN** an httpx request fails to connect
- **THEN** the code catches `httpx.ConnectError`, adds a note with the URL and target host, and returns fallback

#### Scenario: HTTP status error catches httpx.HTTPStatusError
- **WHEN** an httpx request returns a 4xx/5xx status and `raise_for_status()` is called
- **THEN** the code catches `httpx.HTTPStatusError`, adds a note with the URL and status code, and returns fallback or raises a project exception

#### Scenario: Crawl4AI browser failure catches PlaywrightError
- **WHEN** a Crawl4AI crawl operation fails due to a Playwright browser error (page crash, navigation timeout, etc.)
- **THEN** the code catches `playwright.async_api.Error` (aliased as `PlaywrightError`), adds a note with the URL, and returns a failure CrawlResult

#### Scenario: Crawl4AI invalid selector catches InvalidCSSSelectorError
- **WHEN** a Crawl4AI operation fails due to an invalid CSS selector
- **THEN** the code catches `crawl4ai.utils.InvalidCSSSelectorError`, adds a note with the selector, and returns a failure CrawlResult

### Requirement: Graphiti operations SHALL catch graphiti_core.errors.GraphitiError

All Graphiti operations SHALL catch `graphiti_core.errors.GraphitiError` or its subclasses instead of bare `except Exception`. Each catch site SHALL add `exc.add_note()` with document_id, chunk_id, or query context.

Note: Graphiti operations internally call Neo4j, so `GraphitiError` may wrap Neo4j connection errors. Catching `GraphitiError` covers both.

#### Scenario: Graphiti write failure catches GraphitiError
- **WHEN** a `graphiti.add_episode()` call fails
- **THEN** the code catches `GraphitiError`, adds a note with document_id and chunk_id, and returns a verification result with `verified=False`

#### Scenario: Graphiti search failure catches GraphitiError
- **WHEN** a `graphiti.search()` call fails
- **THEN** the code catches `GraphitiError`, adds a note with the query and group_ids, and returns an empty list or fallback results

#### Scenario: Graphiti close failure catches GraphitiError
- **WHEN** `graphiti.close()` fails during shutdown
- **THEN** the code catches `GraphitiError`, adds a note with the service name, and logs a warning without crashing

#### Scenario: Graphiti edge lookup catches EdgeNotFoundError
- **WHEN** a `graphiti.get_edge()` call fails because the edge doesn't exist
- **THEN** the code catches `graphiti_core.errors.EdgeNotFoundError`, adds a note with the edge UUID, and returns None or raises NotFoundException

#### Scenario: Graphiti node lookup catches NodeNotFoundError
- **WHEN** a `graphiti.get_node()` call fails because the node doesn't exist
- **THEN** the code catches `graphiti_core.errors.NodeNotFoundError`, adds a note with the node UUID, and returns None or raises NotFoundException

### Requirement: Cognee operations SHALL catch cognee.exceptions.CogneeApiError

All Cognee operations SHALL catch `cognee.exceptions.CogneeApiError` or its subclasses instead of bare `except Exception`. Each catch site SHALL add `exc.add_note()` with query, user_id, and dataset context.

Note: `CogneeApiError` auto-logs on construction, so catching it silently is safe. The ad-hoc exceptions (`MigrationError`, `FileTypeException`, `FileContentHashingError`) are NOT under `CogneeApiError` and must be caught separately if they can occur.

#### Scenario: Cognee recall failure catches CogneeApiError
- **WHEN** `cognee.recall()` fails
- **THEN** the code catches `CogneeApiError`, adds a note with the query and user_id, and returns an empty list

#### Scenario: Cognee transient error catches CogneeTransientError
- **WHEN** a Cognee operation fails due to a temporary condition (timeout, connection lost)
- **THEN** the code catches `cognee.exceptions.CogneeTransientError`, adds a note with the operation, and retries or returns fallback

#### Scenario: Cognee validation error catches CogneeValidationError
- **WHEN** a Cognee operation fails because of invalid input data
- **THEN** the code catches `cognee.exceptions.CogneeValidationError`, adds a note with the invalid field, and raises ValidationException or returns fallback

#### Scenario: Cognee configuration error catches CogneeConfigurationError
- **WHEN** a Cognee operation fails because of missing or invalid configuration
- **THEN** the code catches `cognee.exceptions.CogneeConfigurationError`, adds a note with the missing config key, and raises ConfigurationException or returns fallback

### Requirement: LLM provider operations SHALL catch provider-specific exceptions

LLM operations SHALL catch provider-specific exceptions instead of bare `except Exception`. The following SHALL be caught:
- `openai.OpenAIError` (base for all OpenAI errors)
- `google.api_core.exceptions.GoogleAPIError` (base for Google API errors)
- `google.genai.errors.APIError` (new google-genai SDK)
- `langchain_core.exceptions.LangChainException` (base for LangChain errors)

Each catch site SHALL add `exc.add_note()` with the model name, prompt context, and operation.

#### Scenario: OpenAI API failure catches OpenAIError
- **WHEN** an OpenAI API call fails
- **THEN** the code catches `openai.OpenAIError`, adds a note with the model name and operation, and returns fallback or retries

#### Scenario: OpenAI rate limit catches RateLimitError
- **WHEN** an OpenAI API call fails with HTTP 429
- **THEN** the code catches `openai.RateLimitError`, adds a note with the model name and retry-after header, and implements backoff

#### Scenario: OpenAI auth failure catches AuthenticationError
- **WHEN** an OpenAI API call fails with HTTP 401
- **THEN** the code catches `openai.AuthenticationError`, adds a note with the model name, and raises UnauthorizedException (do NOT retry)

#### Scenario: Google API failure catches GoogleAPIError
- **WHEN** a Google API call fails
- **THEN** the code catches `google.api_core.exceptions.GoogleAPIError`, adds a note with the model name and operation, and returns fallback

#### Scenario: Google GenAI failure catches APIError
- **WHEN** a google-genai SDK call fails
- **THEN** the code catches `google.genai.errors.APIError`, adds a note with the model name and operation, and returns fallback

#### Scenario: LangChain output parser failure catches OutputParserException
- **WHEN** a LangChain output parser fails to parse LLM output
- **THEN** the code catches `langchain_core.exceptions.OutputParserException`, adds a note with the raw output and expected format, and returns fallback

#### Scenario: LangChain context overflow catches ContextOverflowError
- **WHEN** an LLM call fails because the input exceeds the model's context window
- **THEN** the code catches `langchain_core.exceptions.ContextOverflowError`, adds a note with the token count and model limit, and truncates input or switches to a longer-context model

#### Scenario: Unknown provider exception falls back to Exception
- **WHEN** a provider raises an exception type not in the known hierarchy
- **THEN** the code catches `Exception` as a final fallback, adds a note with the provider name and exception type, and returns fallback with `# noqa: BLE001 — provider-specific catch exhausted`

### Requirement: Database operations SHALL catch asyncpg.exceptions.PostgresError

All asyncpg operations SHALL catch `asyncpg.exceptions.PostgresError` or its subclasses instead of bare `except Exception`. Each catch site SHALL add `exc.add_note()` with the query, table, and operation context.

Client-side errors (`asyncpg.InterfaceError`, `asyncpg.InternalClientError`) SHALL be caught separately when they indicate programming errors rather than database failures.

#### Scenario: Reconciliation fetch failure catches PostgresError
- **WHEN** a reconciliation database query fails
- **THEN** the code catches `asyncpg.exceptions.PostgresError`, adds a note with the user_id and query, and returns a failure result

#### Scenario: Outbox publish failure catches PostgresError
- **WHEN** an outbox event publish fails at the database level
- **THEN** the code catches `asyncpg.exceptions.PostgresError`, adds a note with the event_id and event_type, and marks the event as failed

#### Scenario: Unique violation catches UniqueViolationError
- **WHEN** an INSERT/UPDATE violates a UNIQUE constraint
- **THEN** the code catches `asyncpg.exceptions.UniqueViolationError`, adds a note with the constraint name, and raises ConflictException

#### Scenario: Connection failure catches ConnectionDoesNotExistError
- **WHEN** a query fails because the connection was closed/pooled away
- **THEN** the code catches `asyncpg.exceptions.ConnectionDoesNotExistError`, adds a note with the operation, and retries with a new connection

#### Scenario: Deadlock detected catches DeadlockDetectedError
- **WHEN** a query fails because of a deadlock
- **THEN** the code catches `asyncpg.exceptions.DeadlockDetectedError`, adds a note with the query, and retries the transaction

#### Scenario: Client misuse catches InterfaceError
- **WHEN** an asyncpg API is used incorrectly (closed connection, wrong call order)
- **THEN** the code catches `asyncpg.exceptions.InterfaceError`, adds a note with the operation, and raises DatabaseException (programming error, not retryable)

### Requirement: Document processing SHALL catch library-specific exceptions

Document processing operations SHALL catch specific exceptions from each library instead of bare `except Exception`:
- `docling.exceptions.BaseError` (or its subclass `ConversionError`) for docling operations
- `google.genai.errors.APIError` for Gemini operations
- `yaml.YAMLError` for YAML parsing operations
- `asyncpg.exceptions.PostgresError` for database operations during ingestion

Each catch site SHALL add `exc.add_note()` with the document path, operation, and relevant identifiers.

#### Scenario: Docling table extraction catches BaseError
- **WHEN** docling fails to extract a table from a document
- **THEN** the code catches `docling.exceptions.BaseError`, adds a note with the table index and document path, and skips the table

#### Scenario: Docling conversion catches ConversionError
- **WHEN** docling fails to convert a document to markdown
- **THEN** the code catches `docling.exceptions.ConversionError`, adds a note with the file path and format, and returns a fallback or raises

#### Scenario: Embedding generation catches APIError
- **WHEN** Gemini embedding generation fails
- **THEN** the code catches `google.genai.errors.APIError`, adds a note with the model and text count, and retries or raises

#### Scenario: YAML frontmatter parsing catches YAMLError
- **WHEN** YAML frontmatter in a document fails to parse
- **THEN** the code catches `yaml.YAMLError`, adds a note with the file path, and skips frontmatter extraction

#### Scenario: Ingestion database write catches PostgresError
- **WHEN** a document ingestion fails at the database write step
- **THEN** the code catches `asyncpg.exceptions.PostgresError`, adds a note with the document_id and table, and returns an error result

#### Scenario: PageIndex function catches specific exceptions
- **WHEN** a PageIndex operation fails
- **THEN** the code catches the specific exception from the PageIndex library, adds a note with the page ID and operation, and returns fallback

### Requirement: Agent tools SHALL catch OS-level and library-specific exceptions

Agent tool operations SHALL catch specific exceptions instead of bare `except Exception`:
- `OSError` (and subclasses `FileNotFoundError`, `PermissionError`) for filesystem operations
- `redis.exceptions.RedisError` for Redis operations
- `langchain_core.exceptions.LangChainException` for LLM operations
- `subprocess.SubprocessError` for subprocess execution failures

Each catch site SHALL add `exc.add_note()` with the command, path, or operation context.

#### Scenario: Shell command failure catches OSError
- **WHEN** a shell command execution fails due to an OS error
- **THEN** the code catches `OSError`, adds a note with the command and working directory, and returns a `ToolOutput.fail()` result

#### Scenario: File not found catches FileNotFoundError
- **WHEN** a file read operation fails because the file doesn't exist
- **THEN** the code catches `FileNotFoundError`, adds a note with the file path, and returns a `ToolOutput.fail()` result

#### Scenario: Permission denied catches PermissionError
- **WHEN** a file or directory operation fails because of insufficient permissions
- **THEN** the code catches `PermissionError`, adds a note with the path and required permission, and returns a `ToolOutput.fail()` result

#### Scenario: Redis cache failure catches RedisError
- **WHEN** a tool's Redis cache operation fails
- **THEN** the code catches `redis.exceptions.RedisError`, adds a note with the key and operation, and continues without cache

#### Scenario: LLM call failure catches LangChainException
- **WHEN** a tool's LLM call fails
- **THEN** the code catches `langchain_core.exceptions.LangChainException`, adds a note with the model and operation, and returns a `ToolOutput.fail()` result

#### Scenario: Subprocess failure catches SubprocessError
- **WHEN** a subprocess spawned by a tool fails
- **THEN** the code catches `subprocess.SubprocessError`, adds a note with the command and return code, and returns a `ToolOutput.fail()` result

### Requirement: Celery operations SHALL catch celery.exceptions.CeleryError

Celery operations SHALL catch `celery.exceptions.CeleryError` or its subclasses instead of bare `except Exception`. Each catch site SHALL add `exc.add_note()` with the task name, event type, and operation context.

#### Scenario: Outbox relay publish catches CeleryError
- **WHEN** an outbox relay fails to publish an event via Celery
- **THEN** the code catches `celery.exceptions.CeleryError`, adds a note with the event_id and event_type, and marks the event as failed in the dead letter queue

#### Scenario: Script replay catches CeleryError
- **WHEN** a dead-letter replay script fails to send a task
- **THEN** the code catches `celery.exceptions.CeleryError`, adds a note with the event_type, and continues to the next event

### Requirement: Degradation boundaries SHALL keep except Exception with add_note

The following locations SHALL keep `except Exception` because they are genuine degradation boundaries where:
1. The exception types are unknown at catch time (optional dependencies with opaque internals)
2. Too many exception types from multiple libraries could be thrown in a single block
3. The operation MUST succeed or degrade gracefully — crashing is never acceptable

Each of these sites SHALL add `exc.add_note()` with context and SHALL have a `# noqa: BLE001` comment with an explanatory reason.

#### Scenario: Optional dependency startup failure degrades gracefully
- **WHEN** an optional dependency (Neo4j, Graphiti, Crawl4AI, object storage) fails to initialize during app startup
- **THEN** the code catches `Exception`, adds a note with the dependency name and error details, sets the dependency to `None`, and continues startup

#### Scenario: LangGraph node failure returns fallback state
- **WHEN** a LangGraph node encounters an exception from an LLM, database, or external service
- **THEN** the code catches `Exception`, adds a note with the node name and operation, and returns a fallback state dict without crashing the graph

#### Scenario: OTEL instrumentation failure logs warning
- **WHEN** an OTEL instrumentor fails to register
- **THEN** the code catches `Exception`, adds a note with the instrumentor name, logs a warning, and continues without that instrumentor

#### Scenario: OTEL provider shutdown failure logs warning
- **WHEN** an OTEL provider fails to flush or shutdown
- **THEN** the code catches `Exception`, adds a note with the provider name, logs a warning, and continues shutdown

#### Scenario: Outbox relay dead-letters on any failure
- **WHEN** an outbox relay publish fails for any reason (Celery, database, serialization)
- **THEN** the code catches `Exception`, adds a note with the event_id and exception type, marks the event as failed, and continues to the next event

#### Scenario: Reranker model load failure falls back to default
- **WHEN** a CrossEncoder model fails to load (OSError, ValueError, RuntimeError from torch/transformers)
- **THEN** the code catches `Exception`, adds a note with the model name, loads the fallback model, and continues

#### Scenario: Reranker inference failure returns unranked results
- **WHEN** CrossEncoder reranking fails at inference time
- **THEN** the code catches `Exception`, adds a note with the model name and chunk count, returns chunks without reranking, and logs a warning

#### Scenario: Cognee recall failure returns empty list
- **WHEN** `cognee.recall()` fails for any reason (ad-hoc exceptions outside CogneeApiError hierarchy)
- **THEN** the code catches `Exception`, adds a note with the query and user_id, returns an empty list, and logs the failure

### Requirement: Import aliasing SHALL prevent builtin shadowing

When importing exceptions that shadow Python builtins, the code SHALL alias them on import. The following pairs SHALL use aliases:
- `redis.exceptions.ConnectionError` → `RedisConnectionError`
- `redis.exceptions.TimeoutError` → `RedisTimeoutError`

#### Scenario: Redis import uses alias
- **WHEN** a file imports `redis.exceptions.ConnectionError` or `redis.exceptions.TimeoutError`
- **THEN** the import uses an alias (e.g., `from redis.exceptions import ConnectionError as RedisConnectionError`) and the `except` clause uses the alias

#### Scenario: Playwright import uses alias
- **WHEN** a file imports `playwright.async_api.Error`
- **THEN** the import uses an alias (e.g., `from playwright.async_api import Error as PlaywrightError`) and the `except` clause uses the alias
