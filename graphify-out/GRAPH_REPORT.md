# Graph Report - langchain-fastapi-production  (2026-06-22)

## Corpus Check
- 342 files · ~261,380 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1434 nodes · 2994 edges · 81 communities (74 shown, 7 thin omitted)
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 681 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bdb96644`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]

## God Nodes (most connected - your core abstractions)
1. `AuthService` - 61 edges
2. `DocumentRepository` - 43 edges
3. `RegisterRequest` - 34 edges
4. `UserResponse` - 34 edges
5. `LoginRequest` - 33 edges
6. `TokenResponse` - 33 edges
7. `SessionResponse` - 33 edges
8. `SearchRepository` - 33 edges
9. `Settings` - 33 edges
10. `WebCrawler` - 28 edges

## Surprising Connections (you probably didn't know these)
- `SearchTaskStatusResponse` --uses--> `SearchRepository`  [INFERRED]
  src/app/features/search/service.py → src/app/features/search/repository.py
- `RagSearchRequest` --uses--> `SearchRepository`  [INFERRED]
  src/app/features/search/service.py → src/app/features/search/repository.py
- `RagSearchResponse` --uses--> `SearchRepository`  [INFERRED]
  src/app/features/search/service.py → src/app/features/search/repository.py
- `AsyncSession` --uses--> `SearchRepository`  [INFERRED]
  src/app/features/search/service.py → src/app/features/search/repository.py
- `RankedResultRow` --uses--> `SearchRepository`  [INFERRED]
  src/app/features/search/service.py → src/app/features/search/repository.py

## Import Cycles
- 1-file cycle: `src/app/main.py -> src/app/main.py`
- 1-file cycle: `src/app/lifecycle/lifespan.py -> src/app/lifecycle/lifespan.py`
- 1-file cycle: `src/app/shared/mcp/registry.py -> src/app/shared/mcp/registry.py`
- 1-file cycle: `src/app/middleware/health_check.py -> src/app/middleware/health_check.py`

## Communities (81 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (38): BaseModel, AgentContext, AgentDependencies, ChatResponse, DoclingEnhancementConfig, DoclingExtractionResult, Document, DocumentMetadata (+30 more)

### Community 1 - "Community 1"
Cohesion: 0.23
Nodes (16): get_current_user_id(), _get_search_llm(), get_search_repository(), get_search_service(), Dependency wiring for the search feature., Request-scoped orchestration for the search feature., SearchService, SearchService (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.16
Nodes (20): ContextGrade, _build_answer_cache_key(), _build_query_plan(), _cached_embedding(), _flatten_warnings(), _generate_answer(), _grade_context(), _merge_warning_lists() (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.24
Nodes (20): DocumentCommandService, DocumentQueryService, get_current_user_id(), get_document_command_service(), _get_document_llm(), get_document_query_service(), get_document_repository(), Dependency wiring for unified document feature. (+12 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (61): AgentRegistry, build_agent_registry(), _build_graph_nodes(), Startup-time Agent Saul graph composition helpers., Instantiate all agents + LLM chains once.     Called from build_saul_graph — nev, Holds all pre-built agents and structured-output LLM chains.     Created once at, SaulGraphNodes, _build_analysis_context() (+53 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (21): DocumentIngestionPipeline, main(), Main ingestion script for processing markdown documents into vector DB and knowl, Ingest a single document.          Args:             file_path: Path to the docu, Find all supported document files in the documents folder., Pipeline for ingesting documents into vector DB and knowledge graph., Read document content from file - supports multiple formats via Docling., Transcribe audio file using Whisper ASR via Docling. (+13 more)

### Community 6 - "Community 6"
Cohesion: 0.24
Nodes (6): AppResult, User, UserRole, Admin-scoped user queries.      Directly uses the Beanie User document. No Motor, Return (items, total_count) for the requested page., UserAdminRepository

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (32): create_app(), Create and configure FastAPI application with proper middleware order., BaseHTTPMiddleware, DependencyHealth, ApiDeprecationMiddleware, Middleware to inject API deprecation headers on v1 routes.  Adds ``Deprecation``, Inject ``Deprecation``, ``Sunset``, and ``Link`` headers on v1 routes., check_graphiti() (+24 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (20): close_db(), expand_query_variations(), initialize_db(), initialize_reranker(), main(), Advanced RAG CLI Agent with Multiple Strategies ================================, Two-stage retrieval: Fast vector search + precise cross-encoder re-ranking., Standard semantic search over chunks.      Args:         query: The search query (+12 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (21): CeleryTaskPayload, CeleryTaskRegistry, LegacyTaskPayload, Typed Celery task registry.  Provides a ``CeleryTaskRegistry`` that maps task na, Base for all typed Celery task payloads.      Subclass this for each task to def, Fallback payload for untyped tasks during incremental migration.      Accepts an, Maps task names → Pydantic payload models for validation., Validate kwargs against registered model, or fall back to LegacyTaskPayload. (+13 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (38): get_crawler_service(), get_rate_limiter(), Crawler feature dependencies., Get crawler service instance with Redis from app.state., Get rate limiter instance with Redis from app.state., crawl_url(), get_client_identifier(), get_rate_limit_info() (+30 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (13): CrawlerService, Crawler feature service., Search the web using Tavily.          Args:             request: Search request, Close all connections., Service for web crawling and searching., Check if rate limit is exceeded., Increment rate limit counter., GeminiProcessor (+5 more)

### Community 13 - "Community 13"
Cohesion: 0.08
Nodes (48): ClauseSegment, ContractMetadata, EmbeddingFunction, _build_preamble(), _cached_embedding(), _call_embedding_fn(), _chunk_metadata_json(), _contract_events() (+40 more)

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (18): CrossEncoderReranker, _cached_embedding(), _call_embedding_fn(), make_context_grader_node(), make_generator_node(), make_graph_retrieval_node(), make_hybrid_retrieval_node(), make_query_analyzer_node() (+10 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (32): MessageRole, SearchType, load_mcp_client_server_configs(), MCPClientAuthMode, MCPClientCircuitState, MCPClientServerConfig, MCPClientTransport, MCPToolCatalogEntry (+24 more)

### Community 16 - "Community 16"
Cohesion: 0.26
Nodes (11): CompiledStateGraph, build_document_ingestion_graph(), DocumentIngestionState, _make_ingest_document_node(), Document ingestion graph wrapper., Build the per-job ingestion graph., IngestDocumentFn, BaseChatModel (+3 more)

### Community 17 - "Community 17"
Cohesion: 0.07
Nodes (28): BaseTool, Any, BaseTool, Any, StructuredTool, build_validation_error_handler(), format_tool_validation_error(), make_structured_tool() (+20 more)

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (21): 5.1 Core Model: Cyclic State Machine, 5.2 Node Definitions, 5.3 Execution Pipeline (Legal Processing), 5.4 Parallelization Strategy (CRITICAL), 5.5 Evaluator Node (QA Gate), 5.6 HITL (Interrupt + Resume), 5. Phase 2: LangGraph Orchestration Pipeline, Behavior: (+13 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (15): 10.1 Rules, 10.2 Execution Model, 10. State & Execution Model, 11.1 Recovery Model, 11. Persistence Layer, 12. Observability, 13. Security Model, 14. Failure Modes & Recovery (+7 more)

### Community 21 - "Community 21"
Cohesion: 0.17
Nodes (12): 16.3.1 Structure of a System Prompt, 16.4 Rationale, ## Dos and Don'ts (Operational Guardrails), ## Output Formatting & Interoperability, Recommended Structure:, ## Security & Injection Defense (The "Shield"), ## Strategic Goals & Acceptance Criteria, ## System Prompt Architecture (The Bone Structure) (+4 more)

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (11): 18.1 Memory Model, 18.2 Processor Pipeline (Per Layer), 18.3 Advanced Memory Processing (Per Layer), 18.4 Memory Types, 18.5 Key Insight, 18.6 Memory Retrieval Strategy, 18.7 Graph Memory (Primary Reasoning Layer), 18.8 Memory Router Agent (+3 more)

### Community 24 - "Community 24"
Cohesion: 0.18
Nodes (11): 8.10 PersistMemoryAgent Agent (MANDATORY), 8.1 Ingestion Agent, 8.2 Structure Normalization, 8.3 Clause Segmentation, 8.4 Entity Extraction, 8.5 Relationship Mapping, 8.6 Risk Analysis Agent, 8.7 Compliance & Precedent Agent (+3 more)

### Community 25 - "Community 25"
Cohesion: 0.20
Nodes (10): 16.1 Objective, 16.2 System Prompt Design (High-Pressure Expert Mode), 16.3 System Prompt Template (Canonical), **16. System Prompt Governance & Behavioral Contract**, 1. Persona Definition, 2. Motivation Layer (Negative Pressure Injection), 3. Response Guidelines, 4. Compliance Rules (+2 more)

### Community 26 - "Community 26"
Cohesion: 0.20
Nodes (10): 1. Executive Summary, 2.1 Determinism Over Intelligence, 2.2 State-Centric Design, 2.3 Agents as Orchestrated Components (Not Autonomous Systems), 2.4 Separation of Concerns, 2. Architectural Principles, 3.1 High-Level Architecture, 3. System Overview (+2 more)

### Community 27 - "Community 27"
Cohesion: 0.20
Nodes (10): 4.1 Edge Firewall (DLP + Guardrails), 4.2 Semantic Caching Layer, 4.3 Auth & Session Management, 4. Phase 1: Macro Architecture (Infrastructure Layer), Behavior:, Cache Key:, Design:, Insight: (+2 more)

### Community 28 - "Community 28"
Cohesion: 0.22
Nodes (8): Agent Sandbox, Guardrails, Key architectural features that make ModernBERT efficient include, Key Takeaways for Secure Sandboxing, Practical Implementation, The 4-Step Maturity Model, The Zero Trust Gap in LLMs, Why Encoders Models

### Community 29 - "Community 29"
Cohesion: 0.22
Nodes (9): 17.1 Retry Limits, 17.2 Retry Risks, 17.3 Idempotency Layer (MANDATORY), 17.4 Durable Execution (LangGraph Tasks), 17.5 Tool Output Normalization Layer, 17.6 Tool Design Rules, **17. Tool Execution & Retry Policy**, Execution Contract (+1 more)

### Community 30 - "Community 30"
Cohesion: 0.29
Nodes (7): 24.1 Legal-Specific Metrics, 24.2 System Metrics, **24. Accuracy, Evaluation & Reliability**, Clause Detection, Compliance, Entity Extraction, Risk Analysis

### Community 31 - "Community 31"
Cohesion: 0.33
Nodes (6): 19.1 Core Principle, 19.2 Structured Context Assembly, 19.3 Context Optimization Techniques, 19.4 Observed Impact, 19.5 Implementation Pattern, **19. Context Engineering & Token Discipline**

### Community 32 - "Community 32"
Cohesion: 0.33
Nodes (6): 1. Your "Plan" is actually a **transaction log**, 2. Graph Memory is not for retrieval — it’s for **constraint propagation**, 3. Your Evaluator Node is secretly your **control plane**, 4. The real bottleneck is NOT LLM latency, 5. If you ever allow an agent to mutate state outside schema:, 🔥 CHOSEN-ONES INSIGHT

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (6): 1. Your System Prompt is NOT a prompt, 2. Idempotency is your real “agent memory”, 3. The biggest hidden bug in your system will be:, 4. Your system already has the shape of a distributed database, 5. Final edge most people miss, 🔥 CHOSEN-ONES INSIGHT

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (6): 20.1 Design Constraint, 20.2 Execution Model, 20.3 Key Requirement, **20. HITL (Human-in-the-Loop) Execution Model**, 21.3 System Signals, 21.4 Logging

### Community 35 - "Community 35"
Cohesion: 0.40
Nodes (5): 9.1 Critical Optimization, 9.2 Correct Pattern, 9.3 Impact, 9. Performance Architecture, NEVER do inside node:

### Community 36 - "Community 36"
Cohesion: 0.50
Nodes (4): 23.1 Design Principles, 23.2 MCP Alignment (Optional Layer), 23.3 Tool Context Optimization, **23. Tool Calling Best Practices (Enforced Standard)**

### Community 37 - "Community 37"
Cohesion: 0.50
Nodes (4): 6.1 State Schema, 6.2 Memory Processing Pipeline, 6.3 Long-Term Memory, 6. Phase 3: Memory & Context OS

### Community 38 - "Community 38"
Cohesion: 0.50
Nodes (4): 7.1 Structured Outputs, 7.2 Validation Pipeline, 7. Phase 4: Tooling & Schema Enforcement, BEST PRACTICES for tool calling:

### Community 39 - "Community 39"
Cohesion: 0.18
Nodes (11): _build_cache_key(), _build_search_items(), _graphiti_filter_chunk_ids(), _to_ranked_rows(), DocumentSearchResultItem, RankedChunk, RankedResultRow, UnifiedRagRequest (+3 more)

### Community 40 - "Community 40"
Cohesion: 0.39
Nodes (7): DocumentRepository, build_chunk_rows(), process_document_ingestion(), _verify_legal_chunks(), BaseChatModel, Graphiti, StorageService

### Community 41 - "Community 41"
Cohesion: 0.43
Nodes (6): _extract_postgres_chunk_ids(), _extract_search_blob(), GraphitiVerificationResult, Graphiti write and verification helpers for legal chunks., write_and_verify_chunk(), Graphiti

### Community 43 - "Community 43"
Cohesion: 0.08
Nodes (44): _extract_raw_token(), extract_raw_token_from_connection(), get_auth_service(), get_current_active_user(), get_current_user(), get_current_verified_user(), get_refresh_token_repository(), get_token_claims() (+36 more)

### Community 44 - "Community 44"
Cohesion: 0.16
Nodes (9): build_search_filter_params(), DocumentRepository, Persistence and retrieval operations for unified documents/chunks., Repository for unified document lifecycle and retrieval., _vector_literal(), Any, AppResult, AsyncSession (+1 more)

### Community 45 - "Community 45"
Cohesion: 0.13
Nodes (18): _infrastructure_failure(), make_apply_changes_node(), make_fetch_existing_node(), make_reconcile_node(), make_write_versions_node(), _parse_reconciliation_decision(), Reconciliation graph nodes for background entity deduplication and conflict reso, _read_llm_content() (+10 more)

### Community 46 - "Community 46"
Cohesion: 0.21
Nodes (9): ImpersonateResponse, PaginatedData, RefreshTokenRepository, User, UserRole, UserAdminRepository, UserAdminResponse, _to_admin_response() (+1 more)

### Community 47 - "Community 47"
Cohesion: 0.06
Nodes (35): async_sessionmaker, AsyncDriver, AsyncIOMotorClient, AsyncIOMotorDatabase, Base, Celery, lifespan(), Application lifespan management. (+27 more)

### Community 48 - "Community 48"
Cohesion: 0.13
Nodes (13): LegalAskRequest, LegalAskResponse, Database operations for the search feature., SearchRepository, Run the clauses-backed retrieval graph and return a grounded answer., Create or deduplicate a document, then enqueue chunk/embed work., SearchIngestRequest, SearchIngestResponse (+5 more)

### Community 49 - "Community 49"
Cohesion: 0.19
Nodes (12): _build_bm25_statement(), _build_trigram_statement(), _build_vector_statement(), _rank_rows(), Persistence layer for search ingestion and retrieval., Bulk upsert chunk rows using the document/chunk unique key., Run weighted in-database RRF over canonical clauses., _vector_literal() (+4 more)

### Community 50 - "Community 50"
Cohesion: 0.11
Nodes (37): ActivityLog, add_activity(), cache_with_fallback(), delete_user_endpoint(), get_activity_endpoint(), get_cached_user(), get_multiple_users(), get_user_activity() (+29 more)

### Community 51 - "Community 51"
Cohesion: 0.17
Nodes (13): HybridSearchRequest, RagSearchRequest, RagSearchResponse, _build_cache_key(), _build_search_items(), Run BM25 and vector retrieval, fuse the results, and hydrate ranked chunks., Return ranked hits plus ordered context sections for RAG use., _run_parallel_search() (+5 more)

### Community 52 - "Community 52"
Cohesion: 0.24
Nodes (10): build_chunk_rows(), Create row payloads for repository upsert calls., _batched(), process_ingestion_document(), Service layer for hybrid search and ingestion., Chunk, embed, and upsert search chunks for a document., Create a task-local async DB session and execute search ingestion., run_ingestion_task() (+2 more)

### Community 53 - "Community 53"
Cohesion: 0.19
Nodes (29): ChangePasswordRequest, File, get_token_claims, AvatarResponse, ChangePasswordRequest, UpdateProfileRequest, change_password(), get_profile() (+21 more)

### Community 54 - "Community 54"
Cohesion: 0.11
Nodes (27): _get_sdk_client(), PageIndexBatchConfig, PageIndexChatConfig, PageIndexClient, PageIndexConfig, Configuration for indexing operations., Concurrency settings for batch indexing., Configuration for chat completion calls. (+19 more)

### Community 55 - "Community 55"
Cohesion: 0.16
Nodes (25): AutoTokenizer, Chunk, DoclingDocument, chunk_document(), chunk_document_simple(), create_hybrid_chunker(), get_tokenizer(), initialize_chunking() (+17 more)

### Community 56 - "Community 56"
Cohesion: 0.12
Nodes (15): BaseSettings, get_settings(), Application settings loaded from environment variables., Returns a cached instance of the application settings., Settings, _batched(), _embed_chunks(), run_document_ingestion_task() (+7 more)

### Community 57 - "Community 57"
Cohesion: 0.13
Nodes (15): AgentSpec, create_production_agent(), ProductionAgent, Agent factory — the main entry point for creating production agents.  Uses LangC, Build a fully configured production agent from a spec.      Returns a Production, Wraps a compiled LangGraph agent with production runtime behaviour:     - Long-t, Single async invocation.          Args:             user_message: The user's inp, Stream the agent's response token by token.          stream_mode options: "messa (+7 more)

### Community 59 - "Community 59"
Cohesion: 0.15
Nodes (11): build_default_middleware_stack(), build_minimal_middleware_stack(), ContextEditingMiddleware, GuardrailMiddleware, ModelRetryMiddleware, Allows runtime editing of context: inject variables, redact PII,     or transfor, Model-based guardrails: evaluates the AI's response before returning it.     On, Production-ready default middleware stack.      Order matters — middleware runs (+3 more)

### Community 60 - "Community 60"
Cohesion: 0.26
Nodes (12): ExampleData, CleanLegalDocument, DoclingProcessingContext, preprocess_legal_document(), Narrow context for document preprocessing., Structured output from preprocessing., Async wrapper around Docling (CPU-heavy)., BatchExtractionResult (+4 more)

### Community 61 - "Community 61"
Cohesion: 0.23
Nodes (11): get_health_service, get_health(), get_self(), Health feature API router., HealthDataDTO, HealthService, SelfInfoDTO, APIResponse (+3 more)

### Community 62 - "Community 62"
Cohesion: 0.22
Nodes (9): do_run_migrations(), Run migrations in 'offline' mode.      This is used for generating migration scr, Run migrations with the provided database connection.      Args:         connect, Run migrations in async mode using init_db() to get the engine., Run migrations in 'online' mode.      This connects to the database and applies, run_async_migrations(), run_migrations_offline(), run_migrations_online() (+1 more)

### Community 63 - "Community 63"
Cohesion: 0.29
Nodes (9): close_mcp_client_manager(), get_mcp_client_manager_dep(), get_shared_mcp_client_manager(), MCP client manager initialization and dependency injection., Return a process-wide MCP client manager for non-request runtimes., Close all MCP upstream connections during lifespan shutdown., Dependency to inject MCP client manager from lifespan., MCPClientManager (+1 more)

### Community 64 - "Community 64"
Cohesion: 0.10
Nodes (83): ForgotPasswordRequest, LoginRequest, LogoutRequest, OAuthAuthorizeResponse, RefreshRequest, RegisterRequest, ResendVerificationRequest, ResetPasswordRequest (+75 more)

### Community 65 - "Community 65"
Cohesion: 0.47
Nodes (5): generate_text(), get_circuit_breaker(), CircuitBreakerService, Depends, Request

### Community 66 - "Community 66"
Cohesion: 0.50
Nodes (3): ChunkResult, Chunk search result model., Ensure score is between 0 and 1.

### Community 82 - "Community 82"
Cohesion: 0.14
Nodes (14): CrawlerConfig, get_crawler_config(), Crawler configuration and settings., Convert to Crawl4AI CrawlerRunConfig kwargs., Configuration for the web crawler., Get DefaultMarkdownGenerator with configured options and optional content filter, Get crawler configuration from settings., Get proxy configuration for Crawl4AI. (+6 more)

### Community 88 - "Community 88"
Cohesion: 0.24
Nodes (8): CrawlResult, Result from crawling a URL., Shared crawler module., Process a single crawl result with optional Gemini processing., Crawl a URL or URLs based on request.          Args:             request: Crawl, CrawlRequest, CrawlResponse, CrawlResultItem

### Community 89 - "Community 89"
Cohesion: 0.23
Nodes (12): AsyncWebCrawler, close_crawl4ai_crawler(), create_crawl4ai_crawler(), get_crawl4ai_crawler(), get_crawler(), Crawl4AI browser initialization and dependency injection., Create and start a Crawl4AI browser for lifespan management.      Uses full Craw, Close the Crawl4AI browser during lifespan shutdown. (+4 more)

### Community 90 - "Community 90"
Cohesion: 0.13
Nodes (14): 1. Current State Summary, 2. Missing Specs, 3. Environment Variables to Add, 4. Bug: Lifespan Singleton vs Per-Call Browser, 5. Priority Order, Crawl4AI Configuration Spec, SPEC-01: Content Noise Filtering, SPEC-02: Lifespan Crawler Config Parity (+6 more)

### Community 91 - "Community 91"
Cohesion: 0.14
Nodes (12): Convert crawl4ai result to domain CrawlResult., Discover URLs for a domain via sitemap without full crawl., Check if URL points to a PDF., Build dispatcher with optional CrawlerMonitor., Recursively crawl internal links using native BFS deep crawl strategy., Web crawler using Crawl4AI with caching., Generate cache key for URL., Get cached crawl result. (+4 more)

## Knowledge Gaps
- **188 isolated node(s):** `Connection`, `Request`, `Depends`, `URLSafeTimedSerializer`, `Request` (+183 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CrawlResult` connect `Community 88` to `Community 0`, `Community 11`, `Community 82`, `Community 56`, `Community 89`, `Community 91`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Why does `AuthService` connect `Community 64` to `Community 43`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `Settings` connect `Community 56` to `Community 2`, `Community 3`, `Community 39`, `Community 40`, `Community 82`, `Community 88`, `Community 91`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 41 inferred relationships involving `AuthService` (e.g. with `LoginRequest` and `RegisterRequest`) actually correct?**
  _`AuthService` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `DocumentRepository` (e.g. with `ContextGrade` and `DocumentRepository`) actually correct?**
  _`DocumentRepository` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `RegisterRequest` (e.g. with `AuthService` and `AuthServiceDep`) actually correct?**
  _`RegisterRequest` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `UserResponse` (e.g. with `AuthService` and `AuthServiceDep`) actually correct?**
  _`UserResponse` has 29 INFERRED edges - model-reasoned connections that need verification._