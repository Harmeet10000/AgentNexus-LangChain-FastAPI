# Graph Report - langchain-fastapi-production  (2026-09-27)

## Corpus Check
- 1223 files · ~1,223,260 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2372 nodes · 4483 edges · 135 communities (124 shown, 10 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 260 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fd89007e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- SimpleNamespace
- lifespan.py
- nodes.py
- graphiti/client.py
- test_unprovisioned_graph_fails_closed.py
- redis_func.py
- WebCrawler
- CrawlJobStore
- document_worker.py
- Open Knowledge Format (OKF)
- CeleryTaskPayload
- init_db
- logger.py
- trace_layer
- Failure
- celery.py
- connections/__init__.py
- test_exception_reachability.py
- render.py
- billing_tasks.py
- build_chat_model
- graph.py
- ResilientTask
- AsyncExtractionService
- AsyncStreamingCallbackHandler
- test_ingestion_persistence_retarget.py
- HealthService
- langchain_layer/__init__.py
- CircuitBreakerService
- storage.py
- tavily.py
- SystemPromptParts
- test_statute_lookup_order.py
- test_throwaway_graph_resilience.py
- load_golden_set
- embedder.py
- S3ClientWrapper
- setup_cognee
- prompts.py
- functions.py
- structural_navigator.py
- validate
- Agent Saul
- health/service.py
- utils.py
- OutboxRelay
- get_deep_researcher
- RateLimiter
- crawler_tasks.py
- write_clause_episodes_to_graphiti
- .get_health
- router.py
- pageindex/client.py
- test_outbox_relay.py
- HumanMessage
- env.py
- Performance and agent prompt audit — 2026-09-09
- preprocess_legal_document
- validate_okf.py
- Checker
- dependencies.py
- ProductionAgent
- .fixture_manifest
- knowledge/index.md
- logger_usage_example.py
- test_phrase_search.py
- test_alembic_memory_filter.py
- Inter-Process Communication Architectures in Rust
- ._do_extract_structured
- test_registry_adoption.py
- reindex_okf.py
- auth.py
- CrawlerService
- neo4j.py
- DocumentWorkerResources
- langextract_to_graph.py
- _consolidate_async
- credit_tasks.py
- SKILL.md
- Open Knowledge Format (OKF)
- processor.py
- build_tool_bundle
- lifespan_mcp.py
- .crawl
- test_repository_exception_notes.py
- Open Knowledge Format
- CircuitBreakerOpenError
- GeminiProcessor
- audit
- Configuring Android Network Security Config for Certificate Pinning
- test_auth_documents_feature_errors.py
- AI Gateway
- Compiler Code Generation Pipeline
- TestSettingsProductionValidation
- langchain-fastapi-production
- Lossless authoring and validation
- Add, retrieve, and revise knowledge
- Public Key Certificate Pinning in Android
- Register Allocation via Graph Coloring and Linear Scan
- Engineering Notes: Deep Internals in Security, Compilers, and Rust IPC
- Requirements
- Requirement: A deliberate degradation SHALL name the reason it degrades
- Requirement: A shared module that wraps a third-party service SHALL own an error union, not raise an APIException
- Requirement: Example code SHALL NOT demonstrate a pattern the project forbids
- initialize_crawler_worker
- _invoice_service
- Source Summary
- Requirement: An exception family rooted outside the project base SHALL be caught by name or re-rooted
- Requirement: The cache layer SHALL classify a cache-backend failure as a cache failure
- AgentMemoryCheck
- test_embedder_no_substitution.py
- _MappingResult
- overview.mdx
- copilot-instructions.md
- 4. Concept Documents
- Technical Guide: Compiler Code Generation Pipeline
- Source Summary
- Requirement: A raise that satisfies a framework contract SHALL be exempt from the union rules
- AGENTS.md
- 5. Cross-linking
- authored_body
- Knowledge Index
- Requirement: A generation adapter behind a circuit breaker SHALL classify by name, not relabel a broad catch
- Requirement: Inheritance among the shared spine's exception families SHALL be ordered narrowest-first at every catch site
- Requirement: The global exception handler's isinstance dispatch SHALL be exempt from the union rules
- Requirement: The request-scoped session dependency SHALL remain the only committer and SHALL NOT inspect Results
- Requirement: The startup degradation boundary SHALL name every family it survives
- test_rag_agent_embedder_import.py
- 1. Motivation
- scripts/__init__.py
- Knowledge Activity Log
- playbooks/index.md
- refresh-code-graphs.sh
- langchain-fastapi-production

## God Nodes (most connected - your core abstractions)
1. `StorageService` - 47 edges
2. `HealthService` - 33 edges
3. `validate()` - 32 edges
4. `CrawlJobStore` - 27 edges
5. `_run()` - 27 edges
6. `trace_layer()` - 27 edges
7. `build_chat_model()` - 26 edges
8. `ComponentCheck` - 25 edges
9. `DocumentRepository` - 22 edges
10. `WebCrawler` - 22 edges

## Surprising Connections (you probably didn't know these)
- `test_the_psutil_memory_field_is_not_collided_with()` --uses--> `HealthChecksDTO`  [INFERRED]
  tests/unit/test_agent_memory_health.py → src/app/features/health/dto.py
- `test_cognee_setup_base_caught_not_just_subclass()` --uses--> `CogneeSetupError`  [INFERRED]
  tests/unit/test_exception_reachability.py → src/app/connections/cognee.py
- `test_cognee_setup_base_caught_not_just_subclass()` --uses--> `CogneeDimensionMismatchError`  [INFERRED]
  tests/unit/test_exception_reachability.py → src/app/connections/cognee.py
- `test_worker_compiles_once_and_two_service_invocations_reuse_the_graph()` --indirect_call--> `init_db()`  [INFERRED]
  tests/unit/lifecycle/test_document_worker_lifecycle.py → src/app/connections/postgres.py
- `_service()` --uses--> `HealthService`  [INFERRED]
  tests/unit/test_agent_memory_health.py → src/app/features/health/service.py

## Import Cycles
- None detected.

## Communities (135 total, 10 thin omitted)

### Community 0 - "SimpleNamespace"
Cohesion: 0.07
Nodes (52): SimpleNamespace, check_cognee(), check_graphiti(), check_mongodb(), check_neo4j(), check_neo4j_plugins(), check_postgres(), check_redis() (+44 more)

### Community 1 - "lifespan.py"
Cohesion: 0.06
Nodes (57): NamedTuple, close_neo4j_driver(), Close Neo4j driver and cleanup connections. Args: driver: Neo4j async driver…, _init_object_storage(), _init_outbox_relay(), lifespan(), Any, AsyncIOMotorClient (+49 more)

### Community 2 - "nodes.py"
Cohesion: 0.07
Nodes (54): ComplianceOutput, FinalReport, GroundingVerificationOutput, LegalAgentState, NormalizedDocument, OrchestratorAction, RiskAnalysisOutput, Runnable (+46 more)

### Community 3 - "graphiti/client.py"
Cohesion: 0.07
Nodes (40): GraphitiSearchResult, Knowledge-graph connection establishment. Thin re-export: the implementation…, BoundGraphitiService, close_graphiti(), get_obligation_chain(), GraphitiService, _has_jurisdiction(), Any (+32 more)

### Community 4 - "test_unprovisioned_graph_fails_closed.py"
Cohesion: 0.06
Nodes (48): SecurityConfig, SecurityMiddleware, apply_fastapi_guard_response_modifier(), build_fastapi_guard_config(), _client_ip(), get_metrics(), get_security_middleware(), initialize_fastapi_guard() (+40 more)

### Community 5 - "redis_func.py"
Cohesion: 0.15
Nodes (47): CacheError, CacheKey, CacheKeyPart, CacheResult, RedisCommandArgs, RedisError, add_to_bloom_filter(), _bloom_filter_exists() (+39 more)

### Community 6 - "WebCrawler"
Cohesion: 0.08
Nodes (28): CrawlerConfig, CrawlerRunConfig, MemoryAdaptiveDispatcher, _crawl4ai_version(), CrawlerBrowser, CrawlResult, Any, BaseModel (+20 more)

### Community 7 - "CrawlJobStore"
Cohesion: 0.09
Nodes (21): CrawlJob, CrawlJobChunk, CrawlJobPageList, CrawlJobSearchResponse, CrawlResultItem, CrawlJobStore, CrawlResponse, Redis (+13 more)

### Community 8 - "document_worker.py"
Cohesion: 0.07
Nodes (38): AgentMemoryService, AsyncPostgresSaver, CompiledStateGraph, IngestDocumentFn, LangGraph checkpointer connection establishment. Thin re-export: the…, _close_document_worker_resources(), initialize_document_worker(), _provision_document_worker() (+30 more)

### Community 9 - "Open Knowledge Format (OKF)"
Cohesion: 0.05
Nodes (41): 10.1 A computation is its own concept, 10.2 Contract fields, 10.3 The computation, 10.4 Concepts that use a computation, 10.5 How a consumer uses it (informative), 10.6 Verification versus attestation, 10. Attested computations concept, 11. Conformance (+33 more)

### Community 10 - "CeleryTaskPayload"
Cohesion: 0.08
Nodes (29): MailerResult, CeleryTaskPayload, CeleryTaskRegistry, Base for all typed Celery task payloads., Maps task names → Pydantic payload models for validation., config_from_settings(), MailerConfig, BaseModel (+21 more)

### Community 11 - "init_db"
Cohesion: 0.08
Nodes (34): DatabaseUrlFlavour, SplitResult, create_outbox_relay(), async_sessionmaker, AsyncSession, Celery, Outbox-relay connection establishment., Build the outbox relay against the plain-DSN database and run its startup scan.… (+26 more)

### Community 12 - "logger.py"
Cohesion: 0.08
Nodes (30): LogRecord, initialize_celery_observability(), Initialize observability inside each Celery worker process., console_format(), _install_stdlib_bridge(), _InterceptHandler, _is_sensitive_key(), Any (+22 more)

### Community 13 - "trace_layer"
Cohesion: 0.12
Nodes (19): DocumentEmbeddingWidthError, DocumentResult, Insert, build_chunk_rows(), build_chunk_upsert_statement(), build_search_filter_params(), DocumentRepository, Any (+11 more)

### Community 14 - "Failure"
Cohesion: 0.15
Nodes (16): Failure, abort_multipart_upload(), copy_object(), create_multipart_upload(), delete_by_uri(), delete_object(), get_by_uri(), get_object() (+8 more)

### Community 15 - "celery.py"
Cohesion: 0.14
Nodes (32): IdempotencyStatus, JsonMetadata, acquire_idempotency_lock(), build_circuit_breaker_key(), build_idempotency_key(), CircuitBreakerSnapshot, create_celery_app(), _default_circuit_snapshot() (+24 more)

### Community 16 - "connections/__init__.py"
Cohesion: 0.10
Nodes (29): close_httpx_client(), create_httpx_client(), get_httpx_client(), get_shared_httpx_client(), AsyncClient, HTTPConnection, HTTPX client with optimal performance settings., Create production-grade HTTPX client with HTTP/2 and connection pooling. Key… (+21 more)

### Community 17 - "test_exception_reachability.py"
Cohesion: 0.09
Nodes (28): CeleryError, Base for the refusals the typed registry raises before a send., A dispatch named a task that has no registered payload model., A dispatched payload did not match the model its task declares., TaskDispatchError, TaskPayloadValidationError, UnregisteredTaskError, _cognee_policy() (+20 more)

### Community 18 - "render.py"
Cohesion: 0.11
Nodes (30): Result, APIResponse, _build_request_meta(), _error_code_value(), ErrorDetail, _ExceptionError, HealthResponse, http_error() (+22 more)

### Community 19 - "billing_tasks.py"
Cohesion: 0.16
Nodes (30): AuditLogRepository, BillingOperation, PlanRepository, independent_session(), async_sessionmaker, Give one unit of work an independent transaction., _audit_repo(), billing_dunning() (+22 more)

### Community 20 - "build_chat_model"
Cohesion: 0.12
Nodes (29): BaseMessage, abatch_multimodal(), abatch_text(), acreate_gemini_context_cache(), aget_chat_model(), ainvoke_multimodal(), ainvoke_text(), astream_text() (+21 more)

### Community 21 - "graph.py"
Cohesion: 0.12
Nodes (29): Command, MessageLikeRepresentation, ResearcherState, compress_research(), execute_tool_safely(), get_researcher_subgraph(), RunnableConfig, LangGraph implementation for Tavily-backed deep research. (+21 more)

### Community 22 - "ResilientTask"
Cohesion: 0.13
Nodes (21): _celery_meters(), idempotency_manager(), _inject_trace_context(), log_task_failure(), log_task_postrun(), log_task_prerun(), log_task_published(), log_task_retry() (+13 more)

### Community 23 - "AsyncExtractionService"
Cohesion: 0.10
Nodes (21): ExtractionOutcome, AsyncExtractionService, ExtractionFailed, ExtractionFailureCode, ExtractionRequest, ExtractionSucceeded, LangExtractClient, LangExtractSettings (+13 more)

### Community 24 - "AsyncStreamingCallbackHandler"
Cohesion: 0.10
Nodes (13): AsyncCallbackHandler, BaseCallbackHandler, AsyncStreamingCallbackHandler, configure_langsmith(), LatencyCallbackHandler, Any, LangSmith observability bootstrap and custom callbacks. Must be imported before…, Bootstrap LangSmith tracing by setting env vars. Call this at application… (+5 more)

### Community 25 - "test_ingestion_persistence_retarget.py"
Cohesion: 0.16
Nodes (20): ContextualizedChunk, ContractMetadata, IngestionState, ParsedDocument, _chunk(), _FakeResult, _FakeSession, _metadata() (+12 more)

### Community 26 - "HealthService"
Cohesion: 0.20
Nodes (11): ComponentCheck, Any, Self, One dependency check in its explicit state. Replaces stringly-typed ``dict``…, HealthService, T, Service for system and dependency health checks., Probe the graph-memory layer with a bounded, read-only query. Absence resolves… (+3 more)

### Community 27 - "langchain_layer/__init__.py"
Cohesion: 0.14
Nodes (20): build_default_middleware_stack(), build_minimal_middleware_stack(), ContextEditingMiddleware, DynamicSystemPromptMiddleware, GuardrailMiddleware, MiddlewareConfig, ModelRetryMiddleware, Any (+12 more)

### Community 28 - "CircuitBreakerService"
Cohesion: 0.14
Nodes (14): IntEnum, post, generate_text(), get_circuit_breaker(), Any, Depends, Request, AcquireStatus (+6 more)

### Community 29 - "storage.py"
Cohesion: 0.14
Nodes (22): build_s3_key(), build_s3_uri(), complete_multipart_upload(), get_signed_put_url(), list_multipart_upload_parts(), list_objects(), MultipartPartURL, MultipartUploadPlan (+14 more)

### Community 30 - "tavily.py"
Cohesion: 0.13
Nodes (24): _build_search_result(), get_context(), get_tavily_client(), AsyncClient, BaseModel, Tavily search service integration., Build a SearchResult from API response data., Search the web using Tavily. Args: query: The search query max_results: Maximum… (+16 more)

### Community 31 - "SystemPromptParts"
Cohesion: 0.10
Nodes (18): ChatPromptTemplate, field_validator, model_validator, AgentSpec, create_production_agent(), MemoryManager, BaseModel, Agent factory — the main entry point for creating production agents. Uses… (+10 more)

### Community 32 - "test_statute_lookup_order.py"
Cohesion: 0.11
Nodes (14): _fetch_statute_section(), make_retrieve_statute_section_tool(), Any, AsyncEngine, BaseTool, IdempotencyGuard, Tool: retrieve_statute_section Compliance agent tool. Point lookup — NOT…, _ConnectionContext (+6 more)

### Community 33 - "test_throwaway_graph_resilience.py"
Cohesion: 0.17
Nodes (24): _build_graph(), _config(), _make_nodes(), _make_tool_seam(), PermanentConfigError, Any, BaseTool, Exception (+16 more)

### Community 34 - "load_golden_set"
Cohesion: 0.11
Nodes (20): GoldenSetLoadResult, InfrastructureException, MalformedGoldenRowError, NotFoundException, OSError, GoldenQuery, GoldenSet, GoldenSetNotFoundException (+12 more)

### Community 35 - "embedder.py"
Cohesion: 0.13
Nodes (23): Never, _attach_embeddings_to_chunks(), create_embedder(), embed_chunks(), _Embedder, generate_embedding(), generate_embeddings_batch(), get_embedding_dimension() (+15 more)

### Community 36 - "S3ClientWrapper"
Cohesion: 0.11
Nodes (9): S3Client, create_object_store(), Settings, Object-storage connection establishment., Build the object store from settings and verify bucket access. Returns None…, Any, Settings, Thin synchronous wrapper around the boto3 S3 client. Owns the raw S3 API calls… (+1 more)

### Community 37 - "setup_cognee"
Cohesion: 0.17
Nodes (22): CogneeDimensionMismatchError, CogneeSetupConfig, CogneeSetupError, _export_cognee_subprocess_env(), BaseModel, RuntimeError, Settings, Cognee connection establishment: long-term episodic + procedural memory. Cognee… (+14 more)

### Community 38 - "prompts.py"
Cohesion: 0.10
Nodes (19): Serialize structured prompt context with TOON for lower token overhead than…, serialize_to_toon(), assemble_kinded_sections(), AssembledPrompt, build_assembled_prompt(), PromptSection, BaseModel, StrEnum (+11 more)

### Community 39 - "functions.py"
Cohesion: 0.14
Nodes (21): PageIndexBatchConfig, PageIndexChatConfig, PageIndexConfig, BaseModel, Configuration for indexing operations., Concurrency settings for batch indexing., Configuration for chat completion calls., abatch_page_index() (+13 more)

### Community 40 - "structural_navigator.py"
Cohesion: 0.20
Nodes (17): NodePath, _Candidate, _children(), _heading_level(), _match_score(), navigate_tree(), _node_text(), BaseModel (+9 more)

### Community 41 - "validate"
Cohesion: 0.24
Nodes (3): FormatTests, Any, validate()

### Community 42 - "Agent Saul"
Cohesion: 0.09
Nodes (21): 1. Clone the repository, 3. Install dependencies, 4. Run the app, Acknowledgments, Agent Saul, Architecture, Common commands, Context window discipline (+13 more)

### Community 43 - "health/service.py"
Cohesion: 0.11
Nodes (17): GraphMemoryClient, GraphQueryDriver, _null_probe(), _null_probe_sync(), Any, async_sessionmaker, AsyncDriver, AsyncIOMotorClient (+9 more)

### Community 44 - "utils.py"
Cohesion: 0.15
Nodes (20): InjectedToolArg, get_all_tools(), _get_httpx_client_from_config(), AsyncClient, BaseModel, BaseTool, RunnableConfig, tool (+12 more)

### Community 45 - "OutboxRelay"
Cohesion: 0.13
Nodes (12): Notification, OutboxRelay, async_sessionmaker, AsyncSession, Celery, Transactional outbox relay using PostgreSQL NOTIFY/LISTEN., Stop accepting notifications and wait for active publishes to finish., Listens for outbox events and publishes them to Celery. Startup scan (one-… (+4 more)

### Community 46 - "get_deep_researcher"
Cohesion: 0.15
Nodes (18): ResearchExecutionGate, Shared LangGraph layer exports., get_deep_researcher(), get_supervisor_subgraph(), Any, Build and cache the supervisor graph on first use in this process., Build and cache the complete deep-research graph on first use., Open Deep Search package exports. (+10 more)

### Community 47 - "RateLimiter"
Cohesion: 0.11
Nodes (8): BaseModel, Settings, RateLimiter, RateLimitResult, Unified reliability base for Celery tasks., Result of rate limit check., Redis-based rate limiter with config embedded in keys., ReliabilitySystem

### Community 48 - "crawler_tasks.py"
Cohesion: 0.13
Nodes (20): get_processor(), Get a Gemini processor instance., crawl_job(), CrawlerJobPayload, CrawlerWorkerResources, get_crawler_worker_resources(), _provision_crawler_worker(), Any (+12 more)

### Community 49 - "write_clause_episodes_to_graphiti"
Cohesion: 0.18
Nodes (17): ClauseSegment, LegalRelationship, Semaphore, ClauseWriteResult, GraphitiService, BaseModel, ClauseEpisodeMetadata, IdempotencyGuard (+9 more)

### Community 50 - ".get_health"
Cohesion: 0.14
Nodes (14): Probe, HealthChecksDTO, HealthDataDTO, HealthResultDTO, BaseModel, DTOs for health feature responses., Aggregated health payload., Service result consumed by router response wrapper. (+6 more)

### Community 51 - "router.py"
Cohesion: 0.17
Nodes (16): get, get_health_service, Basic service metadata., SelfInfoDTO, get_deep_health(), get_health(), get_self(), Depends (+8 more)

### Community 52 - "pageindex/client.py"
Cohesion: 0.18
Nodes (13): FeatureError, StrEnum, RagCode, RagProviderError, RagServiceError, RagValidationError, RAG provider-boundary typed errors., _get_sdk_client() (+5 more)

### Community 53 - "test_outbox_relay.py"
Cohesion: 0.20
Nodes (8): _CeleryRecorder, Outbox relay collaborator ownership and graceful-drain behavior., _relay(), _Session, _SessionContext, _SessionFactory, test_drain_waits_for_an_in_flight_publish_and_rejects_new_work(), test_publish_uses_the_injected_celery_application()

### Community 54 - "HumanMessage"
Cohesion: 0.17
Nodes (17): AgentState, HumanMessage, _build_model(), clarify_with_user(), final_report_generation(), Transform user messages into a structured research brief., Generate the final research report from compressed findings., Build a shared Gemini model for deep research nodes. (+9 more)

### Community 55 - "env.py"
Cohesion: 0.18
Nodes (15): Connection, do_run_migrations(), include_name(), include_object(), _is_memory_schema(), Run migrations with the provided database connection., Run migrations in async mode using init_db() to get the engine., Run migrations in 'online' mode. (+7 more)

### Community 56 - "Performance and agent prompt audit — 2026-09-09"
Cohesion: 0.12
Nodes (16): 1. Scope caches before making them faster, 2. Shorten ingestion transactions and avoid rebuilding infrastructure per document, 3. Fix blocking work in enhanced Docling; retain offloading already present elsewhere, 4. Synchronize reranker initialization and bound inference, 5. Make query plans control execution, 6. Batch cache I/O and preserve ingestion progress, Agent prompts: repair the contract before refining wording, Assessment of the pasted pooling and value-object advice (+8 more)

### Community 57 - "preprocess_legal_document"
Cohesion: 0.21
Nodes (15): ExampleData, CleanLegalDocument, DoclingProcessingContext, preprocess_legal_document(), BaseModel, RagResult, Narrow context for document preprocessing., Structured output from preprocessing. (+7 more)

### Community 58 - "validate_okf.py"
Cohesion: 0.18
Nodes (14): check_preservation(), frontmatter(), local_target(), main(), datetime, Path, Safe YAML loading without silently overwritten duplicate keys., reconstruct() (+6 more)

### Community 59 - "Checker"
Cohesion: 0.37
Nodes (4): actor(), Checker, links(), nonempty()

### Community 60 - "dependencies.py"
Cohesion: 0.17
Nodes (15): CurrentClaims, dependency, DocumentCommandService, DocumentQueryService, get_current_user_id(), get_document_command_service(), _get_document_llm(), get_document_query_service() (+7 more)

### Community 61 - "ProductionAgent"
Cohesion: 0.20
Nodes (9): ProductionAgent, Any, Wraps a compiled LangGraph agent with production runtime behaviour: - Long-term…, Single async invocation. Args: user_message: The user's input message.…, Stream the agent's response token by token. stream_mode options: "messages"…, Batch invoke the agent concurrently on multiple messages. Each message gets its…, Resume a paused (human-in-the-loop) agent after human approval. Call this after…, Get current checkpoint state for a thread. (+1 more)

### Community 62 - ".fixture_manifest"
Cohesion: 0.22
Nodes (4): FixtureCase, PreservationTests, Path, Behavior tests for format permissiveness, producer checks, and preservation.

### Community 64 - "logger_usage_example.py"
Cohesion: 0.20
Nodes (14): PaymentResult, db_create_payment(), _demo(), _FakeDriverError, PaymentBackendError, PaymentCode, PaymentValidationError, process_payment() (+6 more)

### Community 65 - "test_phrase_search.py"
Cohesion: 0.22
Nodes (12): _phrase_like_pattern(), Build an escaped `LIKE %phrase%` pattern matching the phrase literally. The…, _CapturingSession, Any, Phrase post-filter lives in the keyword leg (retrieval-sql task 6.1). The…, Fake session: records the statement and params, returns canned rows., _repository(), test_chunk_hydration_reuses_the_escaped_database_phrase_predicate() (+4 more)

### Community 66 - "test_alembic_memory_filter.py"
Cohesion: 0.22
Nodes (14): configured_calls(), env(), _load_env(), Any, fixture, MonkeyPatch, Band F group 3: the alembic autogenerate filter. Ordering is load-bearing…, Static check: both branches name the filter. (A dynamic count of recorded… (+6 more)

### Community 67 - "Inter-Process Communication Architectures in Rust"
Cohesion: 0.14
Nodes (14): 1. Remote IPC Protocols: Transport & Framing Models, 2. Local IPC Protocols: Kernel Bypass & In-Memory Channels, 3. Comprehensive Protocol Comparison Matrix, 4. Architectural Selection Guide, Deep Internals, HTTP Request/Response (Unary RPC), Inter-Process Communication Architectures in Rust, QUIC Transport (Streams and Datagrams) (+6 more)

### Community 68 - "._do_extract_structured"
Cohesion: 0.23
Nodes (11): ExtractionResult, _parse_extraction_json(), Any, BaseModel, CrawlerProcessingResult, Result from Gemini extraction., Extract structured data from content using Gemini. Args: content: Content to…, Extract structured data AND create a summary. Args: content: Content to process… (+3 more)

### Community 69 - "test_registry_adoption.py"
Cohesion: 0.20
Nodes (9): _FakeModel, populated_registry(), Any, fixture, Band: agent-tools-unification group 3 — registry adoption in the factory. The…, Minimal stand-in: the factory only binds tools to it., The factory's string branch resolves through the registry — proven at the…, _spec() (+1 more)

### Community 70 - "reindex_okf.py"
Cohesion: 0.26
Nodes (9): directories(), documents(), index_content(), link_graph(), outgoing(), Path, rebuild(), Check regeneration preserves manually maintained material. (+1 more)

### Community 71 - "auth.py"
Cohesion: 0.19
Nodes (12): McpError, exchange_subject_token_for_mcp_token(), _find_mcp_error(), get_stored_mcp_tokens(), Any, AsyncClient, BaseException, RunnableConfig (+4 more)

### Community 72 - "CrawlerService"
Cohesion: 0.17
Nodes (8): RateLimitScope, CrawlerService, Any, Redis, Close all connections., Service for web crawling and searching., Check if rate limit is exceeded., Increment rate limit counter.

### Community 73 - "neo4j.py"
Cohesion: 0.18
Nodes (12): get_neo4j_driver(), get_neo4j_session(), init_neo4j(), AsyncDriver, AsyncSession, HTTPConnection, Neo4j database configuration with driver management., Initialize Neo4j driver and test connection. Returns: AsyncDriver: Configured… (+4 more)

### Community 74 - "DocumentWorkerResources"
Cohesion: 0.18
Nodes (13): DocumentWorkerResources, get_document_worker_resources(), Any, BaseModel, T, Return provisioned resources or a typed, capability-naming failure., Run task I/O on the same loop that created the process resources., Resources bound to one worker child and its persistent event loop. (+5 more)

### Community 75 - "langextract_to_graph.py"
Cohesion: 0.17
Nodes (11): GraphIngestionContext, ingest_extractions_to_graph(), Neo4jClient, AnnotatedDocument, BaseModel, Protocol, RagResult, Minimal protocol for Neo4j operations. (+3 more)

### Community 76 - "_consolidate_async"
Cohesion: 0.18
Nodes (13): agent_memory_consolidation(), _connect_graph_driver(), _consolidate_async(), Neo4jCredentials, Any, AsyncDriver, BaseModel, Settings (+5 more)

### Community 77 - "credit_tasks.py"
Cohesion: 0.21
Nodes (11): NoKwargsPayload, Payload for a task that takes no keyword arguments., credits_expire(), credits_reconcile(), _expire_credits_job(), task, Scheduled credit jobs: daily expiration, weekly reconciliation., Daily job to expire past-due credits (Requirement 51). (+3 more)

### Community 78 - "SKILL.md"
Cohesion: 0.18
Nodes (4): Applied Agent Skills guidance, Refinements from the completed crawl and execution, Evaluation and regression checks, Executed independent workflow check

### Community 79 - "Open Knowledge Format (OKF)"
Cohesion: 0.18
Nodes (11): 10. Relationship to other formats, 11. Versioning, 2. Terminology, 3.1 Reserved filenames, 3. Bundle Structure, 6. Index Files, 7. Log Files (optional), 8. Citations (+3 more)

### Community 80 - "processor.py"
Cohesion: 0.22
Nodes (9): Crawler feature service., get_schema_for_type(), StrEnum, Gemini processing for content extraction and summarization., Predefined schema types for structured extraction., Get predefined schema for a type., Bound prompt input with a deterministic tokenizer-independent estimate., SchemaType (+1 more)

### Community 81 - "build_tool_bundle"
Cohesion: 0.20
Nodes (9): AgentToolBundle, build_tool_bundle(), AsyncEngine, BaseModel, BaseTool, IdempotencyGuard, AgentToolBundle: all LangChain tools assembled once at graph-build time.…, Immutable collection of all pre-built LangChain tools. Tool assignment to… (+1 more)

### Community 82 - "lifespan_mcp.py"
Cohesion: 0.24
Nodes (10): initialize_mcp_observability(), MCPServerHandle, BaseModel, FastAPI, Running MCP HTTP server and its serve task., Configure logging and telemetry for a standalone MCP process., Flush telemetry for a standalone MCP process., serve_mcp() (+2 more)

### Community 83 - ".crawl"
Cohesion: 0.22
Nodes (7): CrawlerResult, CrawlRequest, SearchRequest, CrawlResponse, Process a single crawl result with optional Gemini processing., Search the web using Tavily. Args: request: Search request parameters Returns:…, Crawl a URL or URLs based on request. Args: request: Crawl request parameters…

### Community 84 - "test_repository_exception_notes.py"
Cohesion: 0.33
Nodes (9): ExceptHandler, stmt, _contains_call(), _exception_names(), _handlers(), Path, Every relational database catcher preserves bounded driver diagnostics., test_relational_handlers_note_before_rollback_and_failure() (+1 more)

### Community 85 - "Open Knowledge Format"
Cohesion: 0.20
Nodes (10): Apply the format, Execution checklist, Gotchas observed in real runs, Ongoing knowledge operations, Open Knowledge Format, Preserve and enrich, Provenance and trust in v0.2, Script interfaces and portability (+2 more)

### Community 86 - "CircuitBreakerOpenError"
Cohesion: 0.20
Nodes (7): CircuitBreakerOpenError, IdempotencyLockError, RuntimeError, Base Celery task that validates kwargs against a registered Pydantic model., Raised when idempotency lock cannot be acquired., Raised when the circuit breaker is open., TypedCeleryTask

### Community 87 - "GeminiProcessor"
Cohesion: 0.22
Nodes (6): GeminiProcessor, BaseChatModel, Processor for Gemini-based content extraction and summarization., Summarize content using Gemini. Args: content: Content to summarize max_length:…, Invoke the model without blocking the event loop., _response_text()

### Community 88 - "audit"
Cohesion: 0.31
Nodes (6): MigrationTests, Migration checks must detect lost PDF text and edited enhancement prefixes., asset(), audit(), main(), Any

### Community 89 - "Configuring Android Network Security Config for Certificate Pinning"
Cohesion: 0.22
Nodes (9): Configuring Android Network Security Config for Certificate Pinning, Deep Internals, Key Elements, Prerequisites, Related Topics, Step 1: Create the Network Security Configuration XML, Step 2: Bind Configuration in AndroidManifest.xml, Step 3: Generating SPKI Hashes from Server Certificates (+1 more)

### Community 90 - "test_auth_documents_feature_errors.py"
Cohesion: 0.22
Nodes (5): parametrize, MonkeyPatch, Contract tests for the documents and auth feature error migrations., test_mongo_failure_is_retryable_without_rollback(), test_render_result_preserves_security_and_store_statuses()

### Community 91 - "AI Gateway"
Cohesion: 0.22
Nodes (8): **0:00 - 5:00: Introduction & The Skill Checklist Framework**, **10:00 - 15:00: Steering with Leading Words & Leg Work**, **15:00 - 20:43: Pruning & Final Summary**, **5:00 - 10:00: Structure & Minimizing Skill.md**, AI Gateway, Context Engineering, To-Do List, Upgrades

### Community 92 - "Compiler Code Generation Pipeline"
Cohesion: 0.25
Nodes (8): 1. Instruction Selection: The Tree Tiling Problem, 2. Instruction Scheduling and Latency Hiding, 3. Stack Frames and Calling Conventions (ABIs), 4. Peephole Optimization, Compiler Code Generation Pipeline, Deep Internals, Related Concepts & References, Tree Covering Algorithms

### Community 94 - "langchain-fastapi-production"
Cohesion: 0.29
Nodes (6): Commands, Detailed rules, langchain-fastapi-production, Relay, Response priority, Search strategy

### Community 95 - "Lossless authoring and validation"
Cohesion: 0.29
Nodes (7): Agent handoff, Concurrent source revisions, Lossless authoring and validation, Planning, Read and audit before enrichment, Standalone enrichment, Validation interpretation

### Community 96 - "Add, retrieve, and revise knowledge"
Cohesion: 0.29
Nodes (7): Add, retrieve, and revise knowledge, Archive and recovery, Completion checks, Ingestion, Locate the right workflow, Revision and correction, Search and retrieval

### Community 97 - "Public Key Certificate Pinning in Android"
Cohesion: 0.29
Nodes (7): Architectural Trade-offs, Deep Internals, First-Principles Mechanics: Full Certificate vs. Public Key (SPKI) Pinning, Implementation Guide & References, Operational Risk: The Lockout Vulnerability, Public Key Certificate Pinning in Android, Threat Model and Applicability

### Community 98 - "Register Allocation via Graph Coloring and Linear Scan"
Cohesion: 0.29
Nodes (7): 1. Theoretical Complexity: NP-Completeness, 2. The Chaitin-Briggs Graph Coloring Heuristic, 3. Linear Scan Allocation for JIT Compilers, 4. Architectural Comparison, Deep Internals, Register Allocation via Graph Coloring and Linear Scan, Related Concepts & References

### Community 99 - "Engineering Notes: Deep Internals in Security, Compilers, and Rust IPC"
Cohesion: 0.29
Nodes (7): 1. Android Network Security Deep Internals, 2. Compiler Code Generation Deep Internals, 3. Rust Inter-Process Communication (IPC) Protocol Comparison Matrix & Deep Internals, Comprehensive Protocol Comparison, Deep Internals, Engineering Notes: Deep Internals in Security, Compilers, and Rust IPC, Linked Concepts

### Community 100 - "Requirements"
Cohesion: 0.29
Nodes (6): Purpose, Requirement: A seeder SHALL survive one failing seeder without reporting success, Requirements, Scenario: A failing seeder is named and the run does not report success, Scenario: The ORM schema modules need no rule, Shared Infrastructure Errors Specification

### Community 101 - "Requirement: A deliberate degradation SHALL name the reason it degrades"
Cohesion: 0.29
Nodes (7): Requirement: A deliberate degradation SHALL name the reason it degrades, Scenario: A bare suppression is not a reason, Scenario: A documented broad catch is preserved, Scenario: A logging pass-through is not a degradation, Scenario: A task converts a Result it consumes at its own edge, Scenario: An undocumented broad catch is a violation, Scenario: The duplicated task module does not carry a divergent copy of the rule

### Community 102 - "Requirement: A shared module that wraps a third-party service SHALL own an error union, not raise an APIException"
Cohesion: 0.29
Nodes (7): Requirement: A shared module that wraps a third-party service SHALL own an error union, not raise an APIException, Scenario: A feature consuming a shared module narrows rather than catches, Scenario: A mixed subtree is converted at its provider boundary only, Scenario: A shared module is not exempted for being shared, Scenario: A shared module owns its codes, Scenario: A shared third-party wrapper returns a typed failure, Scenario: An optional-dependency guard owes no union

### Community 103 - "Requirement: Example code SHALL NOT demonstrate a pattern the project forbids"
Cohesion: 0.29
Nodes (7): Requirement: Example code SHALL NOT demonstrate a pattern the project forbids, Scenario: A correct example is left alone, Scenario: A green lint run over an exempted directory is not evidence, Scenario: An example is held to the production gates, Scenario: The error-handling ignores are removed from the ignore list, Scenario: The example's catches follow the cache reclassification, Scenario: The raw HTTPException raises are corrected, not exempted

### Community 104 - "initialize_crawler_worker"
Cohesion: 0.29
Nodes (7): initialize_crawler_worker(), connect, Release the browser once and close the persistent loop., Return whether this worker command consumes ``queue_name``. Celery does not…, Provision once in each forked child, never in the Celery parent., shutdown_crawler_worker(), _worker_consumes_queue()

### Community 105 - "_invoice_service"
Cohesion: 0.33
Nodes (6): InvoiceRepository, InvoiceService, PaymentRepository, _invoice_repo(), _invoice_service(), _payment_repo()

### Community 106 - "Source Summary"
Cohesion: 0.33
Nodes (6): 1. Understanding Certificates (0:17 - 4:47), 2. What is Certificate Pinning? (4:47 - 9:09), 3. Practical Implementation (9:09 - 21:24), Derived Artifacts, Source Summary, Technical Guide: Certificate Pinning in Android Applications

### Community 107 - "Requirement: An exception family rooted outside the project base SHALL be caught by name or re-rooted"
Cohesion: 0.33
Nodes (6): Requirement: An exception family rooted outside the project base SHALL be caught by name or re-rooted, Scenario: A family reachable through its root needs no dedicated catch, Scenario: A family with no catch site is closed, Scenario: A worker-path family reaches the worker's boundary, Scenario: An unraised abstract base is not a defect, Scenario: The retry-boundary pattern is the reference

### Community 108 - "Requirement: The cache layer SHALL classify a cache-backend failure as a cache failure"
Cohesion: 0.33
Nodes (6): Requirement: The cache layer SHALL classify a cache-backend failure as a cache failure, Scenario: A backend outage is not a database error, Scenario: A defect in the helper is not laundered into an outage, Scenario: Broad catching to degrade is still permitted, Scenario: The example is corrected with the pattern, Scenario: The own-family re-raise is preserved

### Community 109 - "AgentMemoryCheck"
Cohesion: 0.33
Nodes (4): AgentMemoryCheck, Agent-memory (cognee) check with its named procedure precondition. Absent graph…, Probe agent memory (cognee), mirroring the middleware probe's three states. The…, Run the agent-memory probe, preserving its typed shape. The shared runner…

### Community 110 - "test_embedder_no_substitution.py"
Cohesion: 0.47
Nodes (5): _chunks(), Chunk, The Docling ingestion path delegates one batch to the shared embedder., test_docling_uses_one_shared_document_embedding_batch(), test_provider_failure_is_not_replaced_with_vectors()

### Community 112 - "overview.mdx"
Cohesion: 0.40
Nodes (4): A request's path, Stack, Three layers, Where to go next

### Community 113 - "copilot-instructions.md"
Cohesion: 0.40
Nodes (4): Detailed rules, Matt Pocock skills, Response Priority & Tone, Search strategy

### Community 114 - "4. Concept Documents"
Cohesion: 0.40
Nodes (5): 4.1 Frontmatter, 4.2 Body, 4.3 Example: a concept bound to a resource, 4.4 Example: a concept not bound to a resource, 4. Concept Documents

### Community 115 - "Technical Guide: Compiler Code Generation Pipeline"
Cohesion: 0.40
Nodes (5): Derived Concepts, Key Stages of Code Generation, Source Summary, Technical Guide: Compiler Code Generation Pipeline, Theoretical Complexity

### Community 116 - "Source Summary"
Cohesion: 0.40
Nodes (5): Derived Architecture & Concept Documents, Local Protocols, Remote Protocols, Source Summary, Technical Guide: Inter-Process Communication (IPC) Patterns in Rust

### Community 117 - "Requirement: A raise that satisfies a framework contract SHALL be exempt from the union rules"
Cohesion: 0.40
Nodes (5): Requirement: A raise that satisfies a framework contract SHALL be exempt from the union rules, Scenario: A module __getattr__ keeps raising AttributeError, Scenario: A settings validator keeps raising ValueError, Scenario: An unimplemented task body is not an unclassified raise, Scenario: The same builtin in project code is not exempt

### Community 118 - "AGENTS.md"
Cohesion: 0.50
Nodes (3): Detailed rules, Response Priority & Tone, Search strategy

### Community 119 - "5. Cross-linking"
Cohesion: 0.50
Nodes (4): 5.1 Absolute (bundle-relative) links, 5.2 Relative links, 5.3 Link semantics, 5. Cross-linking

### Community 121 - "Knowledge Index"
Cohesion: 0.50
Nodes (4): Concepts, Knowledge Index, Playbooks, References

### Community 122 - "Requirement: A generation adapter behind a circuit breaker SHALL classify by name, not relabel a broad catch"
Cohesion: 0.50
Nodes (4): Requirement: A generation adapter behind a circuit breaker SHALL classify by name, not relabel a broad catch, Scenario: A local defect is not reported as an upstream outage, Scenario: A provider failure is named, Scenario: The version routers need no rule

### Community 123 - "Requirement: Inheritance among the shared spine's exception families SHALL be ordered narrowest-first at every catch site"
Cohesion: 0.50
Nodes (4): Requirement: Inheritance among the shared spine's exception families SHALL be ordered narrowest-first at every catch site, Scenario: A broader member does not shadow a narrower one, Scenario: A migrating member flattens, Scenario: An exception family is not forced to flatten prematurely

### Community 124 - "Requirement: The global exception handler's isinstance dispatch SHALL be exempt from the union rules"
Cohesion: 0.50
Nodes (4): Requirement: The global exception handler's isinstance dispatch SHALL be exempt from the union rules, Scenario: An except-based gate does not claim to cover the dispatcher, Scenario: The dispatcher keeps its isinstance chain, Scenario: The registration is not simplified

### Community 125 - "Requirement: The request-scoped session dependency SHALL remain the only committer and SHALL NOT inspect Results"
Cohesion: 0.50
Nodes (4): Requirement: The request-scoped session dependency SHALL remain the only committer and SHALL NOT inspect Results, Scenario: A swallowed failure still reaches this commit, Scenario: The dependency is not widened, Scenario: The escaping-exception path is unchanged

### Community 126 - "Requirement: The startup degradation boundary SHALL name every family it survives"
Cohesion: 0.50
Nodes (4): Requirement: The startup degradation boundary SHALL name every family it survives, Scenario: A degraded start is logged, not silent, Scenario: A new startup step names its own failures, Scenario: A required subsystem still fails the start

### Community 128 - "1. Motivation"
Cohesion: 0.67
Nodes (3): 1. Motivation, Goals, Non-goals

## Knowledge Gaps
- **236 isolated node(s):** `Search strategy`, `Matt Pocock skills`, `Response Priority & Tone`, `Detailed rules`, `Ongoing knowledge operations` (+231 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 977 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_chat_model()` connect `build_chat_model` to `document_worker.py`, `utils.py`, `processor.py`, `connections/__init__.py`, `graph.py`, `HumanMessage`, `GeminiProcessor`, `langchain_layer/__init__.py`, `dependencies.py`, `SystemPromptParts`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `create_production_agent()` connect `SystemPromptParts` to `test_registry_adoption.py`, `langchain_layer/__init__.py`, `build_chat_model`, `ProductionAgent`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `trace_layer()` connect `trace_layer` to `logger_usage_example.py`, `CrawlerService`, `logger.py`, `.get_health`, `router.py`, `.crawl`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 54 inferred relationships involving `Failure` (e.g. with `db_create_payment()` and `.crawl()`) actually correct?**
  _`Failure` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `StorageService` (e.g. with `create_object_store()` and `get_document_command_service()`) actually correct?**
  _`StorageService` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `HealthService` (e.g. with `get_health()` and `get_self()`) actually correct?**
  _`HealthService` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `validate()` (e.g. with `.test_attested_computation_is_not_executed()` and `.test_bare_and_list_verification_are_equivalent()`) actually correct?**
  _`validate()` has 26 INFERRED edges - model-reasoned connections that need verification._