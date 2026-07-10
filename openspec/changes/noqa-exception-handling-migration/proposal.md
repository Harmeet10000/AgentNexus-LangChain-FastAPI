## Why

The codebase has 120 `# noqa` comments suppressing ruff lint rules. The largest category (~55) suppresses `BLE001` (blind except), which means `except Exception` is used instead of catching specific exception types. This is a problem because:

1. **Debuggability is destroyed**: When `except Exception` catches everything, the actual root cause (a Redis timeout, a Graphiti connection error, a LangChain output parser failure) is buried in a generic log message. There's no way to programmatically distinguish "Redis is down" from "the LLM returned garbage" from "the database connection was closed."

2. **Recovery strategies are impossible**: Different failure modes need different recovery. A Redis cache miss should fall through to a fresh fetch. A Graphiti write failure should degrade gracefully. A database connection error should trigger a retry. When everything is caught as `Exception`, all recovery paths look the same — log and return a fallback.

3. **The project already has a typed exception hierarchy**: `APIException` and its subclasses (`NotFoundException`, `ExternalServiceException`, `DatabaseException`, etc.) exist precisely for this purpose. The `app_error_to_exception()` mapper in `shared/result/mappers.py` converts typed `AppError` instances to typed `APIException` instances. But the service and infrastructure layers bypass this by catching `Exception` directly.

4. **ruff BLE001 is enabled in pyproject.toml**: The rule is active. Every `# noqa: BLE001` is a reminder that the code doesn't meet the project's own standards. The same applies to `PLC0415` (import outside top level), `TC003`/`TC002` (type-checking imports), `TRY300` (return in try), and others.

5. **The suppression comments are inconsistent**: Some have explanatory text (`# noqa: BLE001 — cache read can fail for any reason`), some don't (`# noqa: BLE001`). Some are in files that already have per-file-ignores for the same rule. The inconsistency makes it hard to tell which suppressions are intentional design decisions and which are accidental omissions.

The goal is to replace every `# noqa` with either (a) proper exception handling that catches specific types, or (b) a documented, intentional suppression that explains why the broad catch is correct for that specific case.

## What Changes

### BLE001 Migration (~55 occurrences)

Replace `except Exception` with specific exception types from each library:

- **Redis operations** (`crawler.py`, `idempotency.py`): Catch `redis.exceptions.RedisError` and its subclasses (`ConnectionError`, `TimeoutError`, `ResponseError`)
- **HTTP/Crawl4AI operations** (`crawler.py`, `open_deep_search/utils.py`): Catch `httpx.HTTPError`, `playwright.async_api.Error`, `crawl4ai.utils.InvalidCSSSelectorError`
- **Graphiti operations** (`graphiti_verifier.py`, `graphiti/client.py`): Catch `graphiti_core.errors.GraphitiError` and its subclasses
- **Cognee operations** (`cognee_client.py`): Catch `cognee.exceptions.CogneeApiError` and its subclasses
- **LLM/Model provider operations** (`ingestion_kb/nodes.py`, `retrieval_kb/nodes.py`, `rag_agent_advanced.py`, `open_deep_search/graph.py`): Catch `openai.OpenAIError`, `google.api_core.exceptions.GoogleAPIError`, `langchain_core.exceptions.LangChainException`
- **asyncpg operations** (`reconciliation/nodes.py`, `outbox/relay.py`): Catch `asyncpg.exceptions.PostgresError` and its subclasses
- **Celery operations** (`scripts/replay_outbox.py`): Catch `celery.exceptions.CeleryError`
- **OpenTelemetry operations** (`otel/instrument.py`, `otel/__init__.py`, `otel/metrics.py`): Keep `except Exception` with `add_note()` — OTel instrumentation is optional and must not crash the app
- **Lifespan optional dependencies** (`lifespan.py`): Keep `except Exception` with `add_note()` — optional deps must degrade gracefully
- **LangGraph node fallbacks** (`ingestion_kb/nodes.py`, `retrieval_kb/nodes.py`, `reconciliation/nodes.py`): Keep `except Exception` with `add_note()` — graph nodes must return fallback state, not crash the graph
- **Agent tools** (`shell.py`, `idempotency.py`, `guardrails.py`): Catch `OSError`, `FileNotFoundError`, `PermissionError` for filesystem; `redis.asyncio.RedisError` for Redis; `langchain_core.exceptions.LangChainException` for LLM calls
- **MCP** (`mcp_core/client/manager.py`, `mcp_core/server/middleware.py`, `mcp_core/server/tools.py`): Catch specific MCP client errors and rate limit errors
- **Document processing** (`docling_enhanced.py`, `embedder.py`, `ingest.py`, `ingest_v2.py`, `chunker.py`, `entity_extractor.py`): Catch `docling.exceptions.BaseError`, `google.genai.errors.APIError`, `asyncpg.exceptions.PostgresError`
- **RAG agent** (`rag_agent_advanced.py`): Catch `openai.OpenAIError`, `google.api_core.exceptions.GoogleAPIError`

For every catch site, add `exc.add_note(f"context={value}")` to attach debugging context before logging or returning fallback.

### PLC0415 Migration (~15 occurrences)

Most PLC0415 suppressions are for legitimate lazy imports (optional dependencies, circular import avoidance). These will be:

- **Kept as noqa** with improved explanatory comments for genuinely lazy imports
- **Removed** where the import can be moved to the top of the file without circular dependency issues

### TC003/TC002/TC001 Migration (~9 occurrences)

These suppress imports that ruff thinks are type-only but are actually resolved at runtime by Pydantic or SQLAlchemy. These will be:

- **Kept as noqa** with comments explaining the runtime resolution (e.g., `# noqa: TC001 — Pydantic resolves this field at runtime via model_config`)

### TRY300 Migration (~4 occurrences)

Return statements inside try blocks that are intentional (idempotency patterns, logger trace layer). These will be:

- **Kept as noqa** with comments explaining the pattern (e.g., `# noqa: TRY300 — return must be inside try for idempotency completion`)

### F401 Migration (~2 occurrences)

Unused imports that are actually used for side effects (model registration for Alembic autogenerate). These will be:

- **Kept as noqa** with comments explaining the side effect (e.g., `# noqa: F401 — registers model with Base.metadata for Alembic autogenerate`)

### S104/S105 Migration (~6 occurrences)

Hardcoded IPs and passwords that are actually configuration defaults and error codes. These will be:

- **Kept as noqa** with comments explaining why they're not secrets (e.g., `# noqa: S105 — error code string, not a password`)

### ARG002 Migration (~5 occurrences)

Unused arguments in protocol implementations (cognee_client.py). These will be:

- **Kept as noqa** with comments explaining the protocol constraint (e.g., `# noqa: ARG002 — protocol-mandated signature`)

### PLW0603 Migration (~3 occurrences)

Global statements for module-level OTEL state. These will be:

- **Kept as noqa** with comments explaining the pattern (e.g., `# noqa: PLW0603 — intentional module-level state for OTEL providers`)

### PLR0915/PLR0912 Migration (~1 occurrence)

Lifespan function complexity. This will be:

- **Kept as noqa** with comment explaining the inherent complexity (e.g., `# noqa: PLR0915 — lifespan initializes 15+ optional dependencies`)

### RET503 Migration (~2 occurrences)

Missing return in try/except/else pattern. These will be:

- **Kept as noqa** with comments explaining the pattern (e.g., `# noqa: RET503 — implicit return in else branch`)

### A002 Migration (~1 occurrence)

Builtin shadowing (`filter` parameter in cognee_client.py). This will be:

- **Renamed** to `filter_query` to eliminate the suppression entirely

### B039 Migration (~2 occurrences)

Mutable defaults in ContextVar (logger.py). These will be:

- **Kept as noqa** with comments explaining safety (e.g., `# noqa: B039 — ContextVar default is evaluated once at module load, safe`)

### ANN001/E402 Migration (~3 occurrences)

Test file annotations and import ordering. These will be:

- **Kept as noqa** in test files (already covered by per-file-ignores, but inline comments improve readability)

## Capabilities

### New Capabilities

- `typed-exception-handling`: Replace generic `except Exception` catches with library-specific exception types across the codebase, using `add_note()` for debug context and the project's `APIException` hierarchy for user-facing errors
- `noqa-documentation`: Add explanatory comments to every remaining `# noqa` suppression, documenting the specific reason the suppression is intentional and correct

### Modified Capabilities

- None. This change does not modify any existing spec-level behavior. It improves internal code quality without changing external contracts.

## Impact

### Files Modified (~45 files)

**BLE001 fixes (highest impact):**
- `src/app/shared/crawler/crawler.py` — 3 catches → Redis + httpx exceptions
- `src/app/shared/crawler/processor.py` — 2 catches → specific extraction exceptions
- `src/app/features/documents/graphiti_verifier.py` — 2 catches → GraphitiError
- `src/app/shared/rag/graphiti/client.py` — 4 catches → GraphitiError
- `src/app/shared/rag/rag_agent_advanced.py` — 9 catches → OpenAI/Google API errors
- `src/app/shared/rag/document_processing/docling_enhanced.py` — 6 catches → docling BaseError
- `src/app/shared/rag/document_processing/embedder.py` — 5 catches → google.genai errors
- `src/app/shared/rag/document_processing/ingest.py` — 5 catches → docling + asyncpg errors
- `src/app/shared/rag/document_processing/ingest_v2.py` — 4 catches → docling + yaml errors
- `src/app/shared/rag/document_processing/chunker.py` — 1 catch → specific processing error
- `src/app/shared/rag/document_processing/entity_extractor.py` — 1 catch → extraction error
- `src/app/shared/rag/langextract/langextract_batch_processor.py` — 1 catch → extraction error
- `src/app/shared/rag/pageindex/functions.py` — 3 catches → pageindex errors
- `src/app/shared/langchain_layer/agents/tools/shell.py` — 6 catches → OSError, PermissionError
- `src/app/shared/langchain_layer/agents/tools/idempotency.py` — 4 catches → Redis + asyncpg errors
- `src/app/shared/langchain_layer/agents/middlewares/guardrails.py` — 1 catch → LangChainException
- `src/app/shared/langchain_layer/agents/memory/cognee_client.py` — 1 catch → CogneeApiError
- `src/app/shared/langgraph_layer/ingestion_kb/nodes.py` — 5 catches → provider-specific + add_note
- `src/app/shared/langgraph_layer/retrieval_kb/nodes.py` — 4 catches → provider-specific + add_note
- `src/app/shared/langgraph_layer/retrieval_kb/reranker.py` — 2 catches → sentence-transformers errors
- `src/app/shared/langgraph_layer/open_deep_search/graph.py` — 3 catches → LangChain + provider errors
- `src/app/shared/langgraph_layer/open_deep_search/utils.py` — 1 catch → httpx + crawl4ai errors
- `src/app/shared/langgraph_layer/reconciliation/nodes.py` — 4 catches → asyncpg errors
- `src/app/shared/otel/instrument.py` — 4 catches → ImportError + add_note
- `src/app/shared/otel/__init__.py` — 3 catches → shutdown errors + add_note
- `src/app/shared/otel/metrics.py` — 1 catch → metrics setup + add_note
- `src/app/shared/outbox/relay.py` — 1 catch → CeleryError
- `src/app/lifecycle/lifespan.py` — 5 catches → optional dep + add_note
- `src/mcp_core/client/manager.py` — 2 catches → MCP client errors
- `src/mcp_core/server/middleware.py` — 1 catch → rate limit errors
- `src/mcp_core/server/tools.py` — 1 catch → tool execution errors
- `src/app/utils/logger.py` — 1 TRY300 + 2 B039 noqa documentation
- `src/app/utils/codes.py` — 3 S105 noqa documentation
- `src/app/utils/embedding.py` — 1 PLC0415 noqa documentation
- `src/app/config/settings.py` — 2 S104 noqa documentation
- `src/app/features/auth/dependencies.py` — 1 S105 noqa documentation
- `src/app/features/auth/service.py` — 3 PLC0415 noqa documentation
- `src/app/features/auth/websocket_security.py` — 1 TC003 noqa documentation
- `src/app/features/documents/service.py` — 1 PLC0415 noqa documentation
- `src/app/features/search/model.py` — 2 TC003/TC002 noqa documentation
- `src/app/features/search/service.py` — 1 PLC0415 noqa documentation
- `src/app/shared/langchain_layer/callback.py` — 1 TC003 noqa documentation
- `src/app/shared/langgraph_layer/ingestion_kb/state.py` — 1 TC001 noqa documentation
- `src/app/shared/langgraph_layer/retrieval_kb/state.py` — 1 TC003 noqa documentation
- `src/app/shared/langgraph_layer/open_deep_search/utils.py` — 2 TC003/TC002 noqa documentation
- `src/app/shared/otel/__init__.py` — 2 PLW0603 noqa documentation
- `src/app/shared/otel/metrics.py` — 1 PLW0603 noqa documentation
- `src/app/lifecycle/lifespan.py` — 1 PLR0915/PLR0912 noqa documentation
- `src/alembic/env.py` — 1 F401 noqa documentation
- `src/app/shared/outbox/model.py` — 1 F401 noqa documentation
- `src/app/connections/crawl4ai.py` — 1 PLC0415 noqa documentation
- `src/mcp_core/testing.py` — 1 PLC0415 noqa documentation
- `src/database/schemas/memory_schema.py` — 1 TC002 noqa documentation
- `src/tasks/auth_email_tasks_typed.py` — 2 TRY300 noqa documentation
- `src/tasks/memory_decay_reconciliation_tasks.py` — 1 BLE001 noqa documentation
- `src/app/shared/langchain_layer/agents/memory/cognee_client.py` — 1 A002 rename + 5 ARG002 noqa documentation
- `src/app/middleware/server_middleware.py` — 1 PLC0415 noqa documentation
- `src/app/examples/redis_examples.py` — 6 BLE001 noqa documentation (examples, keep as-is)
- `scripts/replay_outbox.py` — 1 catch → CeleryError
- `tests/integration/test_api_deprecation.py` — 1 ANN001 noqa documentation
- `tests/integration/test_health.py` — 1 ANN001 noqa documentation
- `tests/unit/test_outbox.py` — 1 E402 noqa documentation

### No Dependencies Added

This change uses only exception types from libraries already in `pyproject.toml`. No new packages required.

### No API Changes

External contracts (HTTP responses, task signatures, WebSocket frames) are unchanged. The only visible change is that error responses may carry more specific `error_code` values when the underlying exception is properly typed.

### Risk

- **Low risk**: Every catch site is being made MORE specific, not less. If a library raises an exception type we don't catch, it will propagate up to the existing global exception handler (which already handles `Exception` as a fallback).
- **Testing needed**: Each modified catch site should be tested with both the specific exception type AND a generic `Exception` fallback to verify the global handler still works.
