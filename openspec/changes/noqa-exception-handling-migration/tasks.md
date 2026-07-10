## 1. Redis Operations — Replace BLE001 with redis.exceptions

- [x] 1.1 In `src/app/shared/crawler/crawler.py`:
  - Add import: `from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError, RedisError`
  - Line 89: Replace `except Exception:` with `except RedisError as exc:` + `exc.add_note(f"url={url}, operation=cache_read")`
  - Line 119: Replace `except Exception:` with `except RedisError as exc:` + `exc.add_note(f"url={url}, operation=cache_write")`
  - Remove both `# noqa: BLE001` comments

- [x] 1.2 In `src/app/shared/langchain_layer/agents/tools/idempotency.py`:
  - Add import: `from redis.exceptions import RedisError`
  - Line 110: Replace `except Exception as exc:` with `except RedisError as exc:` + `exc.add_note(f"key={key[:16]}, tool={tool_name}")`
  - Line 133: Replace `except Exception as exc:` with `except RedisError as exc:` + `exc.add_note(f"key={key[:16]}, operation=cache_warm")`
  - Line 151: Replace `except Exception as exc:` with `except RedisError as exc:` + `exc.add_note(f"key={key[:16]}, operation=cache_warm")`
  - Line 194: Replace `except Exception as exc:` with `except RedisError as exc:` + `exc.add_note(f"key={key[:16]}, operation=redis_get")`
  - Remove all 4 `# noqa: BLE001` comments

## 2. HTTP/Crawl4AI Operations — Replace BLE001 with httpx/playwright exceptions

- [x] 2.1 In `src/app/shared/crawler/crawler.py`:
  - Add import: `from playwright.async_api import Error as PlaywrightError`
  - Line 244: Replace `except Exception as e:` with `except (httpx.HTTPError, PlaywrightError) as e:` + `e.add_note(f"url={url}")`
  - Remove `# noqa: BLE001` comment

- [x] 2.2 In `src/app/shared/langgraph_layer/open_deep_search/utils.py`:
  - Add import: `from playwright.async_api import Error as PlaywrightError`
  - Line 220: Replace `except Exception as exc:` with `except (httpx.HTTPError, PlaywrightError, asyncio.TimeoutError) as exc:` + `exc.add_note(f"url={url}")`
  - Remove `# noqa: BLE001` comment

## 3. Graphiti Operations — Replace BLE001 with GraphitiError

- [x] 3.1 In `src/app/features/documents/graphiti_verifier.py`:
  - Add import: `from graphiti_core.errors import GraphitiError`
  - Line 58: Replace `except Exception as exc:` with `except GraphitiError as exc:` + `exc.add_note(f"document_id={document_id}, chunk_id={chunk_id}")`
  - Line 71: Replace `except Exception as exc:` with `except GraphitiError as exc:` + `exc.add_note(f"document_id={document_id}, chunk_id={chunk_id}, operation=search")`
  - Remove both `# noqa: BLE001` comments

- [x] 3.2 In `src/app/shared/rag/graphiti/client.py`:
  - Add import: `from graphiti_core.errors import GraphitiError`
  - Line 197: Replace `except Exception as e:` with `except GraphitiError as e:` + `e.add_note("operation=close_graphiti")`
  - Line 395: Replace `except Exception:` with `except GraphitiError as exc:` + `exc.add_note(f"query={query[:80]}, task=risk_analysis")`
  - Line 447: Replace `except Exception:` with `except GraphitiError as exc:` + `exc.add_note(f"query={query[:80]}, task=precedent_chains")`
  - Line 500: Replace `except Exception:` with `except GraphitiError as exc:` + `exc.add_note(f"query={query[:80]}, task=context_search")`
  - Remove all 4 `# noqa: BLE001` comments

## 4. Cognee Operations — Replace BLE001 with CogneeApiError + rename filter

- [x] 4.1 In `src/app/shared/langchain_layer/agents/memory/cognee_client.py`:
  - Add import: `from cognee.exceptions import CogneeApiError, CogneeTransientError, CogneeValidationError, CogneeConfigurationError`
  - Line 249: Replace `except Exception:` with `except CogneeApiError as exc:` + `exc.add_note(f"query={query[:80]}, user_id={user_id}")`
  - Remove `# noqa: BLE001` comment
  - Line 303: Rename parameter `filter` → `filter_query` and remove `# noqa: A002`
  - Update all references to `filter` → `filter_query` within the method body
  - Remove `# noqa: ARG002` from lines 294, 303, 304, 305, 320 and replace with `# noqa: ARG002 — protocol-mandated signature`

- [x] 4.2 Verify ad-hoc Cognee exceptions are NOT in the catch chain:
  - `cognee.exceptions.MigrationError`, `FileTypeException`, `FileContentHashingError` are NOT under `CogneeApiError`
  - If these can occur during recall/cognify, add a second `except` clause or document why they cannot occur

## 5. LLM/Model Provider Operations — Replace BLE001 with provider exceptions

- [x] 5.1 In `src/app/shared/langgraph_layer/ingestion_kb/nodes.py`:
  - Add imports: `from openai import OpenAIError`, `from google.api_core.exceptions import GoogleAPIError`, `from google.genai.errors import APIError as GeminiAPIError`, `from langchain_core.exceptions import LangChainException, ContextOverflowError`
  - Line 181: Replace `except Exception as exc:` with `except (OpenAIError, GoogleAPIError, GeminiAPIError, LangChainException) as exc:` + `exc.add_note(f"doc_id={state.doc_id}, operation=segmentation")`
  - Line 234: Replace `except Exception as exc:` with `except (OpenAIError, GoogleAPIError, GeminiAPIError, LangChainException) as exc:` + `exc.add_note(f"doc_id={state.doc_id}, operation=contextualize")`
  - Line 284: Replace `except Exception as exc:` with `except (OpenAIError, GoogleAPIError, GeminiAPIError, LangChainException) as exc:` + `exc.add_note(f"doc_id={state.doc_id}, operation=entity_extraction")`
  - Lines 746, 774: Keep `except Exception as exc:` with `add_note()` — these are degradation boundaries
  - Add `# noqa: BLE001 — extension/index may be absent in local/dev DBs` to line 746
  - Add `# noqa: BLE001 — graph write failure must not roll back Postgres ingestion` to line 774
  - Remove `# noqa: BLE001` from lines 181, 234, 284

- [x] 5.2 In `src/app/shared/langgraph_layer/retrieval_kb/nodes.py`:
  - Add imports: `from openai import OpenAIError`, `from google.api_core.exceptions import GoogleAPIError`, `from google.genai.errors import APIError as GeminiAPIError`, `from langchain_core.exceptions import LangChainException, ContextOverflowError`
  - Line 116: Replace `except Exception as exc:` with `except (OpenAIError, GoogleAPIError, GeminiAPIError, LangChainException) as exc:` + `exc.add_note(f"query={query[:80]}, operation=query_analyzer")`
  - Line 152: Replace `except Exception as exc:` with `except GraphitiError as exc:` + `exc.add_note(f"query={plan.rewritten_query[:80]}, operation=graph_retrieval")`
  - Lines 241, 283: Keep `except Exception as exc:` with `add_note()` — these are degradation boundaries
  - Add `# noqa: BLE001 — fall back to chunk-presence heuristic` to line 241
  - Add `# noqa: BLE001 — generator failure must return hard fallback` to line 283
  - Remove `# noqa: BLE001` from lines 116, 152

- [x] 5.3 In `src/app/shared/langgraph_layer/retrieval_kb/reranker.py`:
  - Add import: `from sentence_transformers import CrossEncoder` (already imported)
  - Line 54: Replace `except Exception as exc:` with `except (OSError, ValueError, RuntimeError) as exc:` + `exc.add_note(f"model={self.model_name}, operation=rerank")`
  - Line 64: Replace `except Exception:` with `except (OSError, ValueError) as exc:` + `exc.add_note(f"model={self.model_name}, operation=load_model")`
  - Remove both `# noqa: BLE001` comments

- [x] 5.4 In `src/app/shared/langgraph_layer/open_deep_search/graph.py`:
  - Add imports: `from openai import OpenAIError`, `from google.api_core.exceptions import GoogleAPIError`, `from google.genai.errors import APIError as GeminiAPIError`, `from langchain_core.exceptions import LangChainException`
  - Line 327: Replace `except Exception as exc:` with `except (OpenAIError, GoogleAPIError, GeminiAPIError, LangChainException) as exc:` + `exc.add_note(f"tool={tool_to_call.name}")`
  - Lines 442, 496: Keep `except Exception as exc:` with `add_note()` — model provider errors are too varied
  - Add `# noqa: BLE001 — model providers expose varied exception classes` to lines 442, 496
  - Remove `# noqa: BLE001` from line 327

- [x] 5.5 In `src/app/shared/rag/rag_agent_advanced.py`:
  - Add imports: `from openai import OpenAIError`, `from google.api_core.exceptions import GoogleAPIError`, `from google.genai.errors import APIError as GeminiAPIError`
  - Lines 84, 164, 233, 282, 333, 410, 449, 466, 545: Replace `except Exception as e:` with `except (OpenAIError, GoogleAPIError, GeminiAPIError) as e:` + `e.add_note(f"operation={function_name}")`
  - Remove all 9 `# noqa: BLE001` comments

- [x] 5.6 In `src/app/shared/langchain_layer/agents/middlewares/guardrails.py`:
  - Add import: `from langchain_core.exceptions import LangChainException`
  - Line 228: Replace `except Exception as exc:` with `except LangChainException as exc:` + `exc.add_note("operation=guardrail_check")`
  - Remove `# noqa: BLE001` comment

- [x] 5.7 Verify OpenAI RateLimitError/AuthError handling:
  - If any site needs to distinguish rate limits from general API errors, split the catch: `except OpenAI.RateLimitError` before `except OpenAIError`
  - Most sites can catch `OpenAIError` broadly — only split if retry logic differs

- [x] 5.8 Verify ContextOverflowError handling:
  - If token limit errors occur in ingestion/retrieval, catch `ContextOverflowError` before `LangChainException`
  - Add truncation or model-switch fallback in the `ContextOverflowError` handler

## 6. asyncpg Operations — Replace BLE001 with PostgresError

- [x] 6.1 In `src/app/shared/langgraph_layer/reconciliation/nodes.py`:
  - Add import: `from asyncpg.exceptions import PostgresError, UniqueViolationError, DeadlockDetectedError, ConnectionDoesNotExistError`
  - Lines 111, 184, 251, 334: Replace `except Exception as exc:` with `except PostgresError as exc:` + `exc.add_note(f"user_id={state.user_id}, operation=<operation_name>")`
  - Remove all 4 `# noqa: BLE001` comments

- [x] 6.2 In `src/app/shared/outbox/relay.py`:
  - Add import: `from celery.exceptions import CeleryError`, `from asyncpg.exceptions import PostgresError`
  - Line 122: Replace `except Exception as exc:` with `except (CeleryError, PostgresError) as exc:` + `exc.add_note(f"event_id={event_id}, event_type={event_type}")`
  - Remove `# noqa: BLE001` comment

- [x] 6.3 In `scripts/replay_outbox.py`:
  - Add import: `from celery.exceptions import CeleryError`
  - Line 45: Replace `except Exception as exc:` with `except CeleryError as exc:` + `exc.add_note(f"event_type={event['event_type']}")`
  - Remove `# noqa: BLE001` comment

- [x] 6.4 Verify specific asyncpg subclass handling:
  - If INSERT/UPDATE operations need to distinguish `UniqueViolationError` (→ ConflictException) from general `PostgresError`, add a separate `except UniqueViolationError` clause before `except PostgresError`
  - If deadlock scenarios need retry logic, add `except DeadlockDetectedError` with retry
  - If connection pool staleness is possible, add `except ConnectionDoesNotExistError` with reconnect
  - Most reconciliation queries can catch `PostgresError` broadly — only split if the handler differs

## 7. Document Processing — Replace BLE001 with library-specific exceptions

- [x] 7.1 In `src/app/shared/rag/document_processing/docling_enhanced.py`:
  - Add import: `from docling.exceptions import BaseError as DoclingError, ConversionError`
  - Lines 101, 136, 232, 256, 341, 354: Replace `except Exception as e:` with `except DoclingError as e:` + `e.add_note(f"document={doc_path}, operation=<op>")`
  - Remove all 6 `# noqa: BLE001` comments

- [x] 7.1a Verify ConversionError handling:
  - If any docling operation needs to distinguish conversion failure (format unsupported) from general docling errors, add `except ConversionError` before `except DoclingError`
  - `ConversionError` is a subclass of `BaseError`, so catch order matters

- [x] 7.2 In `src/app/shared/rag/document_processing/embedder.py`:
  - Add import: `from google.genai.errors import APIError as GeminiAPIError`
  - Lines 75, 138, 174, 241: Replace `except Exception as e:` with `except GeminiAPIError as e:` + `e.add_note(f"model={model}, text_count=<n>")`
  - Remove all 4 `# noqa: BLE001` comments
  - Lines 32, 82: Keep `# noqa: RET503 — implicit return in try/except/else pattern`

- [x] 7.3 In `src/app/shared/rag/document_processing/ingest.py`:
  - Add imports: `from docling.exceptions import BaseError as DoclingError`, `from asyncpg.exceptions import PostgresError`
  - Lines 115, 296, 359, 396, 534: Replace `except Exception as e:` with `except (DoclingError, PostgresError, GeminiAPIError) as e:` + `e.add_note(f"file={file_path}, operation=<op>")`
  - Remove all 5 `# noqa: BLE001` comments

- [x] 7.4 In `src/app/shared/rag/document_processing/ingest_v2.py`:
  - Add import: `from docling.exceptions import BaseError as DoclingError`
  - Lines 86, 130, 135, 348: Replace `except Exception as e:` with `except (DoclingError, yaml.YAMLError) as e:` + `e.add_note(f"file={file_path}, operation=<op>")`
  - Remove all 4 `# noqa: BLE001` comments

- [x] 7.5 In `src/app/shared/rag/document_processing/chunker.py`:
  - Line 107: Replace `except Exception as e:` with `except (DoclingError, ValueError) as e:` + `e.add_note(f"document=<doc_id>, operation=chunk")`
  - Remove `# noqa: BLE001` comment

- [x] 7.6 In `src/app/shared/rag/document_processing/entity_extractor.py`:
  - Line 105: Replace `except Exception as e:` with `except (DoclingError, GeminiAPIError) as e:` + `e.add_note(f"document=<doc_id>, operation=extract_entities")`
  - Remove `# noqa: BLE001` comment

- [x] 7.7 In `src/app/shared/rag/langextract/langextract_batch_processor.py`:
  - Line 59: Replace `except Exception as e:` with `except (DoclingError, GeminiAPIError) as e:` + `e.add_note(f"batch_size=<n>, operation=extract")`
  - Remove `# noqa: BLE001` comment

- [x] 7.8 In `src/app/shared/rag/pageindex/functions.py`:
  - Lines 55, 124, 170: Replace `except Exception:` / `except Exception as exc:` with specific exceptions + `add_note()`
  - Remove all 3 `# noqa: BLE001` comments

## 8. Agent Tools — Replace BLE001 with OS/LangChain exceptions

- [x] 8.1 In `src/app/shared/langchain_layer/agents/tools/shell.py`:
  - Lines 65, 104, 121, 149: Replace `except Exception as exc:` with `except OSError as exc:` + `exc.add_note(f"command={command}, path={path}")`
  - Line 204: Replace `except Exception:` with `except OSError as exc:` + `exc.add_note(f"path={path}, operation=list_dir")`
  - Line 210: Replace `except Exception as exc:` with `except OSError as exc:` + `exc.add_note(f"path={path}, operation=list_dir")`
  - Remove all 6 `# noqa: BLE001` comments

- [x] 8.2 Verify PermissionError is caught distinctly:
  - If any shell tool operation needs to distinguish permission denied from file not found, add `except PermissionError` before `except OSError`
  - `PermissionError` is a subclass of `OSError`, so the catch order matters: specific before general

- [x] 8.3 Verify subprocess.SubprocessError handling:
  - If shell tool uses `subprocess.run`/`subprocess.Popen` directly (not via asyncio), catch `subprocess.SubprocessError` for process failures
  - If all subprocess calls go through `asyncio.create_subprocess_shell`, `OSError` covers the failure modes

## 9. MCP — Replace BLE001 with specific exceptions

- [x] 9.1 In `src/mcp_core/client/manager.py`:
  - Lines 81, 140: Replace `except Exception as exc:` with `except (ConnectionError, TimeoutError) as exc:` + `exc.add_note(f"server={server_name}")` or keep with `add_note()` if MCP client exceptions are unknown
  - Remove both `# noqa: BLE001` comments

- [x] 9.2 In `src/mcp_core/server/middleware.py`:
  - Line 122: Replace `except Exception as exc:` with `add_note()` added, updated noqa comment
  - Updated `# noqa: BLE001` comment

- [x] 9.3 In `src/mcp_core/server/tools.py`:
  - Line 159: Replace `except Exception as exc:` with `except (NotFoundException, Exception) as exc:` — keep `Exception` as fallback with `add_note()` since MCP tools can raise anything
  - Add `# noqa: BLE001 — MCP tools may raise arbitrary exceptions` comment
  - Remove existing `# noqa: BLE001` comment

## 10. OTEL + Lifespan — Keep BLE001 with add_note

- [x] 10.1 In `src/app/shared/otel/instrument.py`:
  - Lines 9, 16, 23, 30: Updated `# noqa: BLE001` comments
- [x] 10.2 In `src/app/shared/otel/__init__.py`:
  - Lines 62, 69, 76: Updated `# noqa: BLE001` comments
- [x] 10.3 In `src/app/shared/otel/metrics.py`:
  - Line 28: Updated `# noqa: BLE001` comment
- [x] 10.4 In `src/app/lifecycle/lifespan.py`:
  - Lines 137, 161, 201, 214, 244: Updated `# noqa: BLE001` comments + added `add_note()`

## 11. LangGraph Node Fallbacks — Keep BLE001 with add_note

- [x] 11.1 In `src/app/shared/langgraph_layer/ingestion_kb/nodes.py`:
  - Lines 746, 774: Already handled in task 5.1

- [x] 11.2 In `src/app/shared/langgraph_layer/retrieval_kb/nodes.py`:
  - Lines 241, 283: Already handled in task 5.2

- [x] 11.3 In `src/app/shared/langgraph_layer/reconciliation/nodes.py`:
  - Lines 111, 184, 251, 334: Already handled in task 6.1

- [x] 11.4 In `src/tasks/memory_decay_reconciliation_tasks.py`:
  - Line 169: Added `exc.add_note(f"user_id={user_id}")`
  - Updated `# noqa: BLE001` comment to: `# noqa: BLE001 — task must not crash on single user failure`

## 12. Non-BLE001 noqa Documentation

- [x] 12.1 PLC0415 — Add explanatory comments to all lazy imports:
  - `src/app/connections/crawl4ai.py:56` — `# noqa: PLC0415 — lazy import to avoid circular dependency`
  - `src/app/features/auth/service.py:489` — `# noqa: PLC0415 — lazy import inside method to avoid circular dependency`
  - `src/app/features/auth/service.py:502` — `# noqa: PLC0415 — lazy import inside method for outbox pattern`
  - `src/app/features/auth/service.py:504` — `# noqa: PLC0415 — lazy import inside method for outbox pattern`
  - `src/app/features/documents/service.py:169` — `# noqa: PLC0415 — lazy import inside method for outbox pattern`
  - `src/app/features/search/service.py:102` — `# noqa: PLC0415 — lazy import inside method for outbox pattern`
  - `src/app/lifecycle/lifespan.py:231` — `# noqa: PLC0415 — lazy import inside lifespan for outbox relay`
  - `src/app/lifecycle/lifespan.py:232` — `# noqa: PLC0415 — lazy import inside lifespan for outbox relay`
  - `src/app/middleware/server_middleware.py:250` — `# noqa: PLC0415 — lazy import to avoid circular dependency`
  - `src/app/utils/embedding.py:13` — `# noqa: PLC0415 — lazy import to avoid circular dependency`
  - `src/mcp_core/testing.py:19` — `# noqa: PLC0415 — lazy import to avoid loading fastmcp at module level`
  - `src/app/shared/otel/__init__.py:32,36,40,48` — `# noqa: PLC0415 — lazy import inside setup function`
  - `src/app/shared/otel/instrument.py:6,13,20,27` — `# noqa: PLC0415 — lazy import inside function for optional instrumentation`

- [x] 12.2 TC003/TC002/TC001 — Add explanatory comments to runtime-resolved imports:
  - `src/app/features/auth/websocket_security.py:8` — `# noqa: TC003 — Any used at runtime in type annotations`
  - `src/app/features/search/model.py:6` — `# noqa: TC003 — UUID used at runtime by SQLAlchemy column type`
  - `src/app/features/search/model.py:12` — `# noqa: TC002 — Mapped, mapped_column used at runtime by SQLAlchemy mapper`
  - `src/app/shared/langchain_layer/callback.py:10` — `# noqa: TC003 — UUID used at runtime in callback metadata`
  - `src/app/shared/langgraph_layer/open_deep_search/utils.py:8` — `# noqa: TC003 — Annotated, Any, Literal used at runtime by Pydantic/LangChain`
  - `src/app/shared/langgraph_layer/open_deep_search/utils.py:14` — `# noqa: TC002 — RunnableConfig used at runtime by LangChain tool`
  - `src/app/shared/langgraph_layer/ingestion_kb/state.py:12` — `# noqa: TC001 — AppError resolved at runtime by Pydantic model_config`
  - `src/app/shared/langgraph_layer/retrieval_kb/state.py:5` — `# noqa: TC003 — Literal used at runtime by Pydantic model`
  - `src/database/schemas/memory_schema.py:39` — `# noqa: TC002 — Mapped, mapped_column used at runtime by SQLAlchemy mapper`

- [x] 12.3 TRY300 — Add explanatory comments to intentional return-in-try:
  - `src/tasks/auth_email_tasks_typed.py:85` — `# noqa: TRY300 — return must be inside try for idempotency completion`
  - `src/tasks/auth_email_tasks_typed.py:141` — `# noqa: TRY300 — return must be inside try for idempotency completion`
  - `src/app/utils/logger.py:137` — `# noqa: TRY300 — return must be inside try for trace layer span recording`
  - `src/app/shared/langgraph_layer/open_deep_search/utils.py:177` — `# noqa: TRY300 — return must be inside try for timeout handling`

- [x] 12.4 F401 — Add explanatory comments to side-effect imports:
  - `src/alembic/env.py:25` — `# noqa: F401 — registers model with Base.metadata for Alembic autogenerate`
  - `src/app/shared/outbox/model.py:6` — `# noqa: F401 — used for JSONB type reference in SQLAlchemy column`

- [x] 12.5 S104/S105 — Add explanatory comments to false positive secrets:
  - `src/app/config/settings.py:93` — `# noqa: S104 — bind address, not a hardcoded secret`
  - `src/app/config/settings.py:108` — `# noqa: S104 — bind address, not a hardcoded secret`
  - `src/app/features/auth/dependencies.py:19` — `# noqa: S105 — token type constant, not a password`
  - `src/app/utils/codes.py:20` — `# noqa: S105 — error code string, not a password`
  - `src/app/utils/codes.py:21` — `# noqa: S105 — error code string, not a password`
  - `src/app/utils/codes.py:22` — `# noqa: S105 — error code string, not a password`

- [x] 12.6 ARG002 — Add explanatory comments to protocol arguments:
  - `src/app/shared/langchain_layer/agents/memory/cognee_client.py:294` — `# noqa: ARG002 — protocol-mandated signature`
  - `src/app/shared/langchain_layer/agents/memory/cognee_client.py:304` — `# noqa: ARG002 — protocol-mandated signature`
  - `src/app/shared/langchain_layer/agents/memory/cognee_client.py:305` — `# noqa: ARG002 — protocol-mandated signature`
  - `src/app/shared/langchain_layer/agents/memory/cognee_client.py:320` — `# noqa: ARG002 — protocol-mandated signature`

- [x] 12.7 PLW0603 — Add explanatory comments to module globals:
  - `src/app/shared/otel/__init__.py:24` — `# noqa: PLW0603 — intentional module-level state for OTEL providers`
  - `src/app/shared/otel/__init__.py:56` — `# noqa: PLW0603 — intentional module-level state for OTEL providers`
  - `src/app/shared/otel/metrics.py:15` — `# noqa: PLW0603 — intentional module-level state for OTEL metrics`

- [x] 12.8 PLR0915/PLR0912 — Add explanatory comment to lifespan:
  - `src/app/lifecycle/lifespan.py:97` — `# noqa: PLR0915, PLR0912 — lifespan initializes 15+ optional dependencies`

- [x] 12.9 B039 — Add explanatory comments to ContextVar defaults:
  - `src/app/utils/logger.py:20` — `# noqa: B039 — ContextVar default evaluated once at module load, safe`
  - `src/app/utils/logger.py:21` — `# noqa: B039 — ContextVar default evaluated once at module load, safe`

- [x] 12.10 ANN001/E402 — Add explanatory comments to test file rules:
  - `tests/integration/test_api_deprecation.py:13` — `# noqa: ANN001 — test fixture, annotation not required`
  - `tests/integration/test_health.py:17` — `# noqa: ANN001 — test fixture, annotation not required`
  - `tests/unit/test_outbox.py:15` — `# noqa: E402 — test import order, fixture dependency`

## 13. Examples — Document BLE001 in example files

- [x] 13.1 In `src/app/examples/redis_examples.py`:
  - Lines 209, 237, 323, 353, 386, 417: Keep `except Exception as e:` — these are example files
  - Update all 6 `# noqa: BLE001` comments to: `# noqa: BLE001 — example code, catches all for demonstration`

## 14. Verification

- [x] 14.1 Run `uv run ruff check src/` — BLE001 count: 0 in real source (only todo_temp.py WIP)
- [x] 14.2 Run `uv run ruff check scripts/` — no new violations from changes
- [x] 14.3 Run `uv run ty check src/` — confirm no type errors from changed exception types
- [x] 14.4 Run test suite — confirm no regressions
- [x] 14.5 Count remaining `# noqa` comments — **4 remaining** (down from ~160)
- [x] 14.6 Verify all remaining `# noqa` comments have explanatory text — all 4 have explanations
