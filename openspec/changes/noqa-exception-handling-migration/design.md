## Context

The codebase has 120 `# noqa` comments. 55 suppress `BLE001` (blind except), meaning `except Exception` is used instead of catching specific exception types. The rest suppress `PLC0415` (lazy imports), `TC003`/`TC002` (runtime-resolved type imports), `TRY300` (return in try), `F401` (side-effect imports), `S104`/`S105` (false positive secrets), `ARG002` (protocol arguments), `PLW0603` (module globals), `PLR0915` (complexity), `RET503` (implicit return), `A002` (builtin shadowing), `B039` (ContextVar defaults), `ANN001`/`E402` (test file rules).

**Current state:**
- `except Exception` is used as a catch-all in ~55 locations across service, infrastructure, and tool layers
- The project has a typed exception hierarchy (`APIException` → `NotFoundException`, `ExternalServiceException`, `DatabaseException`, etc.) in `src/app/utils/exceptions.py`
- The Result bridge pattern (`app_error_to_exception()` in `shared/result/mappers.py`) converts typed `AppError` to typed `APIException`
- ruff BLE001 is enabled in `pyproject.toml` — every `# noqa: BLE001` is a known suppression
- No consistent pattern for adding debugging context to caught exceptions

**Libraries with known exception hierarchies:**
- `redis.exceptions.RedisError` → `ConnectionError`, `TimeoutError`, `ResponseError`
- `httpx.HTTPError` → `ConnectError`, `TimeoutException`, `HTTPStatusError`
- `graphiti_core.errors.GraphitiError` → `EdgeNotFoundError`, `NodeNotFoundError`, etc.
- `cognee.exceptions.CogneeApiError` → `CogneeSystemError`, `CogneeValidationError`, `CogneeTransientError`
- `openai.OpenAIError` → `APIError`, `APIConnectionError`, `RateLimitError`
- `google.api_core.exceptions.GoogleAPIError` → `ClientError`, `ServerError`, `ResourceExhausted`
- `langchain_core.exceptions.LangChainException` → `OutputParserException`, `ContextOverflowError`
- `asyncpg.exceptions.PostgresError` → `ConnectionDoesNotExistError`, `UniqueViolationError`, etc.
- `celery.exceptions.CeleryError` → `TaskRevokedError`, `MaxRetriesExceededError`
- `docling.exceptions.BaseError` → `ConversionError`
- `crawl4ai` — no custom exceptions; raises Playwright errors
- `opentelemetry.sdk` — no custom exceptions; uses stdlib `ValueError`, `RuntimeError`
- `sentence_transformers.CrossEncoder` — no custom exceptions; raises `OSError`, `ValueError`

## Goals / Non-Goals

**Goals:**
- Replace every `except Exception` with the most specific exception type available from each library
- Add `exc.add_note(f"context={value}")` to every catch site for debugging context
- Document every remaining `# noqa` with an explanatory comment explaining why the suppression is correct
- Zero behavioral changes — all error handling paths produce the same outcomes, just with better typing
- Zero new dependencies — use only exception types from libraries already in `pyproject.toml`

**Non-Goals:**
- Changing the project's `APIException` hierarchy
- Adding new exception types to the project
- Modifying the global exception handler behavior
- Changing HTTP response formats or error codes visible to clients
- Refactoring the lifespan function complexity (PLR0915)
- Moving imports to `TYPE_CHECKING` blocks (TC003/TC002 — these are runtime-resolved by Pydantic/SQLAlchemy)
- Changing lazy import patterns (PLC0415 — these are intentional for optional deps and circular import avoidance)

## Decisions

### 1. Specific exceptions with `add_note()` over generic `except Exception`

**Decision:** Replace `except Exception` with library-specific exception types wherever possible. Add `exc.add_note()` to attach debugging context before logging or returning fallback.

**Rationale:** Specific catches enable (a) differentiated logging ("Redis timeout" vs "Graphiti connection error"), (b) targeted recovery strategies (retry on timeout, degrade on connection error), and (c) proper exception chaining with `raise ... from exc` when re-raising. The `add_note()` pattern (Python 3.11+) attaches context without losing the original traceback.

**Alternatives considered:**
- Keep `except Exception` everywhere and just add `add_note()` — simpler but doesn't enable differentiated recovery
- Create project-specific exception wrappers for each library — adds unnecessary abstraction layer

### 2. Keep `except Exception` at genuine degradation boundaries

**Decision:** Some catch sites legitimately need `except Exception` because:
- **Optional dependency initialization** (lifespan.py): If Neo4j, Graphiti, Crawl4AI, or object storage fails to start, the app must continue without them. The specific exception types are unknown at startup time.
- **LangGraph node fallbacks** (ingestion_kb, retrieval_kb, reconciliation): Graph nodes must return fallback state dicts, never crash the graph. The exceptions come from LLM providers, databases, and external services — too many types to enumerate.
- **OTEL instrumentation** (otel/instrument.py, otel/__init__.py): Optional instrumentation must not crash the app. Import errors, SDK errors, and exporter errors are all possible.
- **Agent tools** (shell.py): Shell execution can fail with any `OSError`, `PermissionError`, `FileNotFoundError`, or subprocess errors.

For these sites, the `# noqa: BLE001` stays but gets an explanatory comment and `add_note()`.

**Rationale:** These are genuine "catch everything and degrade" boundaries. The alternative (enumerating 20+ exception types per site) would be fragile and need updating every time a dependency changes.

### 3. Explanatory comments on all remaining `# noqa`

**Decision:** Every remaining `# noqa` gets a comment explaining WHY the suppression is correct.

**Pattern:**
```python
except Exception:  # noqa: BLE001 — [specific reason]
    logger.warning("...")
```

**Examples:**
- `# noqa: BLE001 — optional dep, must not crash app startup`
- `# noqa: BLE001 — graph node must return fallback, not crash graph`
- `# noqa: PLC0415 — lazy import to avoid circular dependency`
- `# noqa: TC001 — Pydantic resolves this field at runtime via model_config`
- `# noqa: TRY300 — return must be inside try for idempotency completion`
- `# noqa: F401 — registers model with Base.metadata for Alembic autogenerate`
- `# noqa: S105 — error code string, not a password`
- `# noqa: ARG002 — protocol-mandated signature`
- `# noqa: PLW0603 — intentional module-level state for OTEL providers`
- `# noqa: PLR0915 — lifespan initializes 15+ optional dependencies`
- `# noqa: RET503 — implicit return in else branch`
- `# noqa: B039 — ContextVar default evaluated once at module load, safe`

### 4. Rename `filter` → `filter_query` in cognee_client.py

**Decision:** Rename the `filter` parameter to `filter_query` to eliminate the `A002` (builtin shadowing) suppression entirely.

**Rationale:** This is the only noqa that can be eliminated by a simple rename. All others require the suppression because the underlying reason is structural (lazy imports, runtime resolution, protocol constraints).

### 5. Import aliasing for library exceptions that shadow builtins

**Decision:** When importing exceptions that shadow Python builtins (e.g., `redis.exceptions.ConnectionError`), alias them on import.

**Pattern:**
```python
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
```

**Rationale:** Avoids confusion with `builtins.ConnectionError` and `builtins.TimeoutError`. Makes `except` blocks self-documenting.

### 6. Group-by-library migration order

**Decision:** Migrate by library, not by file. This ensures consistency — all Redis catches use the same exception types, all Graphiti catches use the same types, etc.

**Order:**
1. Redis operations (2 files, 8 catches) — well-defined exception hierarchy
2. HTTP/Crawl4AI operations (2 files, 5 catches) — httpx has clean hierarchy
3. Graphiti operations (2 files, 6 catches) — GraphitiError hierarchy
4. Cognee operations (1 file, 1 catch) — CogneeApiError hierarchy
5. LLM/Model providers (6 files, ~15 catches) — OpenAI + Google + LangChain
6. asyncpg operations (2 files, 5 catches) — PostgresError hierarchy
7. Document processing (8 files, ~18 catches) — docling + genai
8. Agent tools (3 files, ~10 catches) — OS + Redis + LangChain
9. MCP (3 files, 4 catches) — MCP client errors
10. OTEL + Lifespan (4 files, ~12 catches) — keep BLE001 with add_note
11. LangGraph nodes (4 files, ~13 catches) — keep BLE001 with add_note
12. Non-BLE001 noqa documentation (remaining files)

## Risks / Trade-offs

- **[Library version changes exception hierarchy]** → Mitigated by catching the base exception class (e.g., `RedisError`, `GraphitiError`, `CogneeApiError`) rather than leaf classes. New subclasses are automatically caught.

- **[Some libraries don't expose specific exceptions]** → Crawl4AI, sentence-transformers, and OpenTelemetry SDK don't have custom exception hierarchies. For these, keep `except Exception` with `add_note()` and document why.

- **[Increased code verbosity at catch sites]** → Each catch site goes from 1 line to 2-3 lines (specific exception type + add_note + log). This is intentional — the verbosity is debugging information.

- **[Risk of missing an exception type]** → If a library raises an exception type we don't catch, it propagates to the global exception handler (which already handles `Exception`). No data loss, no crashes. The worst case is a less-specific error message until we add the missing type.

- **[add_note() requires Python 3.11+]** → The project targets Python 3.12 (`requires-python = ">=3.12,<3.14"`), so `add_note()` is available.

## Migration Plan

### Phase 1: Redis operations (2 files)
- `src/app/shared/crawler/crawler.py` — 3 catches
- `src/app/shared/langchain_layer/agents/tools/idempotency.py` — 4 catches

### Phase 2: HTTP/Crawl4AI operations (2 files)
- `src/app/shared/crawler/crawler.py` — 1 catch (already modified in Phase 1)
- `src/app/shared/langgraph_layer/open_deep_search/utils.py` — 1 catch

### Phase 3: Graphiti operations (2 files)
- `src/app/features/documents/graphiti_verifier.py` — 2 catches
- `src/app/shared/rag/graphiti/client.py` — 4 catches

### Phase 4: Cognee operations (1 file)
- `src/app/shared/langchain_layer/agents/memory/cognee_client.py` — 1 catch + rename `filter` → `filter_query`

### Phase 5: LLM/Model providers (6 files)
- `src/app/shared/langgraph_layer/ingestion_kb/nodes.py` — 5 catches
- `src/app/shared/langgraph_layer/retrieval_kb/nodes.py` — 4 catches
- `src/app/shared/langgraph_layer/retrieval_kb/reranker.py` — 2 catches
- `src/app/shared/langgraph_layer/open_deep_search/graph.py` — 3 catches
- `src/app/shared/rag/rag_agent_advanced.py` — 9 catches
- `src/app/shared/langchain_layer/agents/middlewares/guardrails.py` — 1 catch

### Phase 6: asyncpg operations (2 files)
- `src/app/shared/langgraph_layer/reconciliation/nodes.py` — 4 catches
- `src/app/shared/outbox/relay.py` — 1 catch
- `scripts/replay_outbox.py` — 1 catch

### Phase 7: Document processing (8 files)
- `src/app/shared/rag/document_processing/docling_enhanced.py` — 6 catches
- `src/app/shared/rag/document_processing/embedder.py` — 5 catches
- `src/app/shared/rag/document_processing/ingest.py` — 5 catches
- `src/app/shared/rag/document_processing/ingest_v2.py` — 4 catches
- `src/app/shared/rag/document_processing/chunker.py` — 1 catch
- `src/app/shared/rag/document_processing/entity_extractor.py` — 1 catch
- `src/app/shared/rag/langextract/langextract_batch_processor.py` — 1 catch
- `src/app/shared/rag/pageindex/functions.py` — 3 catches

### Phase 8: Agent tools (3 files)
- `src/app/shared/langchain_layer/agents/tools/shell.py` — 6 catches
- `src/app/shared/langchain_layer/agents/tools/idempotency.py` — already modified in Phase 1
- `src/app/shared/langchain_layer/agents/middlewares/guardrails.py` — already modified in Phase 5

### Phase 9: MCP (3 files)
- `src/mcp_core/client/manager.py` — 2 catches
- `src/mcp_core/server/middleware.py` — 1 catch
- `src/mcp_core/server/tools.py` — 1 catch

### Phase 10: OTEL + Lifespan (4 files) — keep BLE001 with add_note
- `src/app/shared/otel/instrument.py` — 4 catches
- `src/app/shared/otel/__init__.py` — 3 catches
- `src/app/shared/otel/metrics.py` — 1 catch
- `src/app/lifecycle/lifespan.py` — 5 catches

### Phase 11: LangGraph nodes (4 files) — keep BLE001 with add_note
- `src/app/shared/langgraph_layer/ingestion_kb/nodes.py` — already modified in Phase 5
- `src/app/shared/langgraph_layer/retrieval_kb/nodes.py` — already modified in Phase 5
- `src/app/shared/langgraph_layer/reconciliation/nodes.py` — already modified in Phase 6
- `src/tasks/memory_decay_reconciliation_tasks.py` — 1 catch

### Phase 12: Non-BLE001 noqa documentation (remaining files)
- All PLC0415, TC003/TC002/TC001, TRY300, F401, S104/S105, ARG002, PLW0603, PLR0915, RET503, B039, ANN001, E402 — add explanatory comments

### Phase 13: Verification
- Run `uv run ruff check src/` — confirm zero new violations
- Run `uv run ty check src/` — confirm no type errors from changed exception types
- Run test suite — confirm no regressions

## Open Questions

- None. All exception hierarchies have been researched and the migration path is clear.
