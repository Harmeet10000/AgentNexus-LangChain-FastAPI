# Graph Report - src  (2026-05-28)

## Corpus Check
- 250 files · ~96,542 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3311 nodes · 10299 edges · 141 communities (129 shown, 12 thin omitted)
- Extraction: 67% EXTRACTED · 33% INFERRED · 0% AMBIGUOUS · INFERRED: 3369 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b337613d`
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
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
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
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 100|Community 100]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 102|Community 102]]
- [[_COMMUNITY_Community 103|Community 103]]
- [[_COMMUNITY_Community 106|Community 106]]
- [[_COMMUNITY_Community 107|Community 107]]
- [[_COMMUNITY_Community 108|Community 108]]
- [[_COMMUNITY_Community 109|Community 109]]
- [[_COMMUNITY_Community 110|Community 110]]
- [[_COMMUNITY_Community 111|Community 111]]
- [[_COMMUNITY_Community 112|Community 112]]
- [[_COMMUNITY_Community 113|Community 113]]

## God Nodes (most connected - your core abstractions)
1. `BaseModel` - 217 edges
2. `get_settings()` - 79 edges
3. `AuthService` - 67 edges
4. `BaseMessage` - 67 edges
5. `SearchRepository` - 64 edges
6. `RefreshTokenRepository` - 63 edges
7. `UserRepository` - 62 edges
8. `IdempotencyGuard` - 60 edges
9. `ToolResult` - 54 edges
10. `BaseTool` - 50 edges

## Surprising Connections (you probably didn't know these)
- `int` --uses--> `ResilientTask`  [INFERRED]
  tasks/example.py → app/connections/celery.py
- `str` --uses--> `ResilientTask`  [INFERRED]
  tasks/example.py → app/connections/celery.py
- `object` --uses--> `ResilientTask`  [INFERRED]
  tasks/search_tasks.py → app/connections/celery.py
- `ResilientTask` --uses--> `ResilientTask`  [INFERRED]
  tasks/search_tasks.py → app/connections/celery.py
- `str` --uses--> `ResilientTask`  [INFERRED]
  tasks/search_tasks.py → app/connections/celery.py

## Communities (141 total, 12 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (153): int, str, AsyncSession, Depends, Redis, Request, SearchRepository, str (+145 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (130): AgentRegistry, build_agent_registry(), _build_graph_nodes(), Startup-time Agent Saul graph composition helpers., Holds all pre-built agents and structured-output LLM chains.     Created once at, Instantiate all agents + LLM chains once.     Called from build_saul_graph — nev, SaulGraphNodes, build_saul_graph() (+122 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (73): CreateSessionRequest, CreateSessionResponse, WebSocket protocol for Agent Saul.  Inbound  (client → server): discriminated on, Graph paused at interrupt(). Client must send WSResumeMessage to continue., Coarse-grained status transition between pipeline stages., Terminal frame.  Client should close the WS after receiving this., POST /agent-saul/sessions — pre-flight before WS connection., First message on a new WS connection.  Kicks off the graph. (+65 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (85): Any, AsyncEngine, CompiledStateGraph, EmbeddingFunction, Redis, Any, AsyncEngine, AsyncSession (+77 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (62): Any, bool, str, bool, int, object, str, AsyncDriver (+54 more)

### Community 5 - "Community 5"
Cohesion: 0.10
Nodes (61): ADMIN, Depends, get_refresh_token_repository, RefreshTokenRepository, UserAdminRepository, bool, int, str (+53 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (63): BaseModel, AgentContext, AgentDependencies, ChatResponse, ChunkResult, Document, DocumentMetadata, Message (+55 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (49): Depends, HTTPConnection, Permission, RefreshTokenRepository, Request, str, TokenClaims, User (+41 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (36): CompiledStateGraph, Request, str, APIResponse, DocumentUploadResponse, str, UploadFile, UserIdDep (+28 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (49): Any, CompiledStateGraph, CrossEncoderReranker, Any, CrossEncoderReranker, EmbeddingFunction, float, object (+41 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (42): Any, BaseTool, bool, str, StructuredTool, bool, int, str (+34 more)

### Community 11 - "Community 11"
Cohesion: 0.13
Nodes (52): bool, bytes, int, Redis, str, Cache utilities using Redis., add_to_bloom_filter(), _build_database_exception() (+44 more)

### Community 12 - "Community 12"
Cohesion: 0.27
Nodes (52): HTTPConnection, APIResponse, bool, CurrentClaims, CurrentVerifiedUser, LoginRequest, Path, Query (+44 more)

### Community 13 - "Community 13"
Cohesion: 0.08
Nodes (39): Any, AsyncEngine, datetime, Redis, str, BaseTool, GraphitiService, IdempotencyGuard (+31 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (50): AgentState, Any, Command, int, object, RunnableConfig, str, MessagesState (+42 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (39): str, APIResponse, AvatarResponse, CurrentVerifiedUser, Depends, Request, TokenClaims, UpdateProfileRequest (+31 more)

### Community 16 - "Community 16"
Cohesion: 0.10
Nodes (38): AgentSaulDeps, get_agent_saul_deps(), get_agent_saul_ws_security_context(), get_current_user_id(), get_redis(), get_saul_checkpointer(), get_saul_graph(), get_websocket_security_service() (+30 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (49): bool, bytes, callable, DoclingDocument, DoclingExtractionResult, int, str, DoclingEnhancementConfig (+41 more)

### Community 18 - "Community 18"
Cohesion: 0.08
Nodes (33): bool, CrawlRequest, CrawlResponse, RateLimiter, Redis, SearchResponse, str, bool (+25 more)

### Community 19 - "Community 19"
Cohesion: 0.06
Nodes (38): create_app(), FastAPI, Create and configure FastAPI application with proper middleware order., Exception, int, Request, str, bytes (+30 more)

### Community 20 - "Community 20"
Cohesion: 0.08
Nodes (35): Any, str, int, str, BaseTool, CrawlUrlTool, CrawlUrlInput, CrawlUrlTool (+27 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (28): LoginRequest, RefreshTokenRepository, RegisterRequest, SessionResponse, str, TokenResponse, User, UserRepository (+20 more)

### Community 22 - "Community 22"
Cohesion: 0.06
Nodes (37): do_run_migrations(), Run migrations in 'offline' mode.      This is used for generating migration scr, Run migrations with the provided database connection.      Args:         connect, Run migrations in async mode using init_db() to get the engine., Run migrations in 'online' mode.      This connects to the database and applies, run_async_migrations(), run_migrations_offline(), run_migrations_online() (+29 more)

### Community 23 - "Community 23"
Cohesion: 0.16
Nodes (39): bool, float, int, object, str, T, acquire_idempotency_lock(), build_circuit_breaker_key() (+31 more)

### Community 24 - "Community 24"
Cohesion: 0.11
Nodes (43): Any, BaseChatModel, BaseMessage, BaseTool, bool, float, GoogleGenerativeAIEmbeddings, int (+35 more)

### Community 25 - "Community 25"
Cohesion: 0.12
Nodes (36): bool, bytes, str, bytes, int, object, str, _get_sdk_client() (+28 more)

### Community 26 - "Community 26"
Cohesion: 0.11
Nodes (42): int, Redis, str, Exception, DatabaseException, ActivityLog, add_activity(), cache_with_fallback() (+34 more)

### Community 27 - "Community 27"
Cohesion: 0.10
Nodes (40): Any, AsyncClient, BaseTool, bool, Exception, int, RunnableConfig, SearchResponse (+32 more)

### Community 28 - "Community 28"
Cohesion: 0.14
Nodes (35): main(), Any, bool, FastAPI, int, str, get_settings(), Returns a cached instance of the application settings. (+27 more)

### Community 29 - "Community 29"
Cohesion: 0.10
Nodes (37): Any, datetime, float, GraphitiSearchResult, int, str, Graphiti, close_graphiti() (+29 more)

### Community 30 - "Community 30"
Cohesion: 0.08
Nodes (36): str, AsyncClient, bool, float, int, object, str, get_shared_tavily_http_client() (+28 more)

### Community 31 - "Community 31"
Cohesion: 0.14
Nodes (35): AsyncEngine, CompiledStateGraph, ReconciliationRunnable, AsyncEngine, object, ReconciliationRunnable, str, Prompts for contract KB ingestion nodes. (+27 more)

### Community 32 - "Community 32"
Cohesion: 0.11
Nodes (38): Any, bool, callable, DoclingExtractionResult, int, str, create_converter(), Factory function to create enhancement config. (+30 more)

### Community 33 - "Community 33"
Cohesion: 0.09
Nodes (23): Any, bool, IngestionConfig, IngestionResult, str, DocumentIngestionPipeline, main(), Main ingestion script for processing markdown documents into vector DB and knowl (+15 more)

### Community 34 - "Community 34"
Cohesion: 0.10
Nodes (28): BaseTool, GraphitiSearchResult, GraphitiService, IdempotencyGuard, object, str, ClauseEpisodeMetadata, FinalReportEpisodeMetadata (+20 more)

### Community 35 - "Community 35"
Cohesion: 0.12
Nodes (32): bool, bytes, int, Permission, str, UserRole, create_access_token(), create_impersonation_token() (+24 more)

### Community 36 - "Community 36"
Cohesion: 0.12
Nodes (30): Any, callable, Chunk, float, int, str, create_cached_embedder(), create_embedding_cache() (+22 more)

### Community 37 - "Community 37"
Cohesion: 0.08
Nodes (30): AsyncClient, HTTPConnection, AsyncDriver, Celery, FastAPI, Redis, Settings, get_shared_httpx_client() (+22 more)

### Community 38 - "Community 38"
Cohesion: 0.17
Nodes (29): APIResponse, HealthService, Request, Response, SelfInfoDTO, async_sessionmaker, AsyncDriver, AsyncIOMotorClient (+21 more)

### Community 39 - "Community 39"
Cohesion: 0.16
Nodes (12): bool, Any, bool, str, bool, float, str, MCPClientManager (+4 more)

### Community 40 - "Community 40"
Cohesion: 0.16
Nodes (28): bool, str, bool, ClauseEpisodeMetadata, ClauseSegment, IdempotencyGuard, LegalEdgeInput, LegalRelationship (+20 more)

### Community 41 - "Community 41"
Cohesion: 0.10
Nodes (31): CrawlerService, RateLimiter, Request, bool, CrawlerService, CrawlRequest, CrawlResponse, int (+23 more)

### Community 42 - "Community 42"
Cohesion: 0.17
Nodes (32): Any, bool, callable, float, str, create_extractor(), extract(), extract_entities_batch() (+24 more)

### Community 43 - "Community 43"
Cohesion: 0.11
Nodes (29): IdempotencyRecord, Serialized idempotency record stored in Redis., Partially validated state loaded from Redis., RawCircuitBreakerState, _compute_decay(), DecayStats, float, int (+21 more)

### Community 44 - "Community 44"
Cohesion: 0.07
Nodes (29): 1. MCP server foundation, 2. Remote HTTP integration, 3. MCP security and protection, 4. MCP metrics, 5. Internal MCP client manager, App-state reuse, Assumptions Made, Chosen Ones (+21 more)

### Community 45 - "Community 45"
Cohesion: 0.07
Nodes (27): Basic Task, Celery Usage, Circuit Breaker, code:env (RABBITMQ_URL=amqp://guest:guest@localhost:5672//), code:bash (uv run celery -A app.connections.celery:celery_app inspect a), code:python (from app.connections import celery_app), code:bash (uv run celery -A app.connections.celery:celery_app worker --), code:bash (uv run celery -A app.connections.celery:celery_app worker --) (+19 more)

### Community 46 - "Community 46"
Cohesion: 0.12
Nodes (24): Redis, bool, str, CrawlerConfig, get_crawler_config(), Crawler configuration and settings., Configuration for the web crawler., Get crawler configuration from settings. (+16 more)

### Community 47 - "Community 47"
Cohesion: 0.16
Nodes (27): Any, bool, callable, IngestionConfig, IngestionResult, str, clean_databases(), create_ingestion_pipeline() (+19 more)

### Community 48 - "Community 48"
Cohesion: 0.12
Nodes (21): Any, int, str, Any, bool, str, ExtractionResult, GeminiProcessor (+13 more)

### Community 49 - "Community 49"
Cohesion: 0.09
Nodes (18): Any, bool, int, str, build_default_middleware_stack(), build_minimal_middleware_stack(), ContextEditingMiddleware, DynamicSystemPromptMiddleware (+10 more)

### Community 50 - "Community 50"
Cohesion: 0.14
Nodes (25): Any, _append_search_filter_args(), _append_search_highlight_args(), _append_search_summarize_args(), _build_create_search_index_args(), _build_search_index_query_args(), create_search_index(), execute_pipeline() (+17 more)

### Community 51 - "Community 51"
Cohesion: 0.17
Nodes (16): Any, BaseException, bool, str, create_celery_app(), log_task_failure(), log_task_postrun(), log_task_prerun() (+8 more)

### Community 52 - "Community 52"
Cohesion: 0.16
Nodes (22): AgentSpec, create_production_agent(), ProductionAgent, Agent factory — the main entry point for creating production agents.  Uses LangC, Build a fully configured production agent from a spec.      Returns a Production, Wraps a compiled LangGraph agent with production runtime behaviour:     - Long-t, Declarative specification for a production agent.     Pass to `create_production, CodeAgentContext (+14 more)

### Community 53 - "Community 53"
Cohesion: 0.16
Nodes (20): CrawlMode, Crawler feature constants., Predefined schema types., SchemaType, CrawlRequest, CrawlResponse, CrawlResultItem, RateLimitInfo (+12 more)

### Community 54 - "Community 54"
Cohesion: 0.20
Nodes (24): Any, Chunk, DoclingDocument, IngestionConfig, str, AutoTokenizer, chunk_document(), chunk_document_simple() (+16 more)

### Community 55 - "Community 55"
Cohesion: 0.11
Nodes (20): ResilientTask, str, Deliver email verification link. Wire your mailer of choice here., Deliver password reset link. Wire your mailer of choice here., send_password_reset_email(), send_verification_email(), add(), process_document() (+12 more)

### Community 56 - "Community 56"
Cohesion: 0.11
Nodes (19): Any, Request, RunnableConfig, str, AsyncClient, str, Shared LangGraph layer exports., build_open_deep_search_config() (+11 more)

### Community 57 - "Community 57"
Cohesion: 0.16
Nodes (16): float, int, str, Any, bool, bytes, int, str (+8 more)

### Community 58 - "Community 58"
Cohesion: 0.18
Nodes (22): int, str, close_db(), expand_query_variations(), initialize_db(), initialize_reranker(), main(), Advanced RAG CLI Agent with Multiple Strategies ================================ (+14 more)

### Community 59 - "Community 59"
Cohesion: 0.18
Nodes (21): Celery, async_sessionmaker, AsyncDriver, AsyncIOMotorClient, AsyncSession, Celery, HealthService, Redis (+13 more)

### Community 60 - "Community 60"
Cohesion: 0.14
Nodes (17): async_sessionmaker, AsyncEngine, AsyncSession, HTTPConnection, str, BaseSettings, Environment, Application environment. (+9 more)

### Community 61 - "Community 61"
Cohesion: 0.15
Nodes (20): Any, int, str, Chunk, clean_markdown(), extract_headers(), extract_title_from_markdown(), get_chunk_summary() (+12 more)

### Community 62 - "Community 62"
Cohesion: 0.23
Nodes (18): Any, APIResponse, int, str, T, RequestMeta, _build_request_meta(), http_error() (+10 more)

### Community 63 - "Community 63"
Cohesion: 0.17
Nodes (7): AsyncIOMotorDatabase, bool, str, User, Returns only sessions still active in Redis; lazily cleans up expired IDs., Return (user, was_created). Links OAuth account if user already exists., UserRepository

### Community 64 - "Community 64"
Cohesion: 0.18
Nodes (17): str, str, ExampleData, CleanLegalDocument, DoclingProcessingContext, preprocess_legal_document(), Narrow context for document preprocessing., Structured output from preprocessing. (+9 more)

### Community 65 - "Community 65"
Cohesion: 0.16
Nodes (13): Any, str, Any, str, BaseOutputParser, ChatPromptTemplate, Assemble the full system prompt string with plain labeled sections., Convert to LangChain ChatPromptTemplate.          Args:             **extra_runt (+5 more)

### Community 66 - "Community 66"
Cohesion: 0.13
Nodes (18): Any, AsyncClient, BaseException, RunnableConfig, str, StructuredTool, exchange_subject_token_for_mcp_token(), _find_mcp_error() (+10 more)

### Community 67 - "Community 67"
Cohesion: 0.22
Nodes (15): bool, FinalReport, FinalReportEpisodeMetadata, LegalRelationship, str, FinalReportEpisodeMetadata, Final report metadata for high-trust episode storage., CogneeService (+7 more)

### Community 68 - "Community 68"
Cohesion: 0.11
Nodes (11): Any, str, AsyncIOMotorCollection, APIFeatures, Execute the query and return results., Apply filtering to query., Apply sorting to query., Apply field limiting (projection). (+3 more)

### Community 69 - "Community 69"
Cohesion: 0.32
Nodes (6): Any, str, HealthService, Service for system and dependency health checks., Return basic service metadata., Run all health checks and return aggregated status.

### Community 70 - "Community 70"
Cohesion: 0.20
Nodes (13): int, Request, int, Redis, str, get_rate_limiter(), Build a FastAPI dependency that enforces a Redis-backed rate limit.      Args:, Rate limiting helpers. (+5 more)

### Community 71 - "Community 71"
Cohesion: 0.19
Nodes (10): Single async invocation.          Args:             user_message: The user's inp, Stream the agent's response token by token.          stream_mode options: "messa, Batch invoke the agent concurrently on multiple messages.         Each message g, Resume a paused (human-in-the-loop) agent after human approval.         Call thi, Get current checkpoint state for a thread., Get full checkpoint history for time-travel debugging., Any, bool (+2 more)

### Community 72 - "Community 72"
Cohesion: 0.15
Nodes (15): float, int, str, Any, str, db_create_payment(), process_payment(), console_format() (+7 more)

### Community 73 - "Community 73"
Cohesion: 0.15
Nodes (16): Any, AsyncEngine, BaseTool, GraphitiService, IdempotencyGuard, int, str, Neo4jSubgraphExpander (+8 more)

### Community 74 - "Community 74"
Cohesion: 0.21
Nodes (4): Any, Exception, int, str

### Community 75 - "Community 75"
Cohesion: 0.19
Nodes (7): Any, bool, int, Redis, str, Adapt a project Redis client to FastAPI Guard's Redis handler protocol., RedisProtocolAdapter

### Community 76 - "Community 76"
Cohesion: 0.21
Nodes (9): bool, Enum, RateLimitScope, Rate limiting using fastapi-limiter and Redis., Increment rate limit counter., Rate limit scope types., Get remaining rate limit quota., Check if rate limit is exceeded.          Args:             identifier: User ide (+1 more)

### Community 77 - "Community 77"
Cohesion: 0.15
Nodes (13): code:python (from app.utils.cache import push_to_list), code:python (from app.utils.cache import get_list_items), code:python (from app.utils.cache import get_list_length), code:python (from app.utils.cache import remove_from_list), code:python (from app.utils.cache import update_list_item), code:python (from app.utils.cache import trim_list), Get List Items, List Length (+5 more)

### Community 78 - "Community 78"
Cohesion: 0.28
Nodes (12): AsyncDriver, AsyncSession, HTTPConnection, str, close_neo4j_driver(), get_neo4j_driver(), get_neo4j_session(), init_neo4j() (+4 more)

### Community 79 - "Community 79"
Cohesion: 0.20
Nodes (6): Global API response envelope enforcement for routers., APIRouter that validates `response_model` uses `APIResponse[T]`., StrictEnvelopeAPIRouter, APIRouter, bool, str

### Community 80 - "Community 80"
Cohesion: 0.18
Nodes (10): Access Redis Client, Best Practices, code:python (from fastapi import Depends, Request), code:python (from app.utils.cache import execute_pipeline), code:block27 ({object_type}:{key_component}:{optional_sub_keys}), code:python (from app.utils.exceptions import DatabaseException), Error Handling, Key Naming Convention (+2 more)

### Community 81 - "Community 81"
Cohesion: 0.18
Nodes (11): code:python (from app.utils.cache import set_hash), code:python (from app.utils.cache import get_hash), code:python (from app.utils.cache import update_hash), code:python (from app.utils.cache import delete_hash_field), code:python (from app.utils.cache import delete_hash), Delete Hash, Delete Hash Field, Get Hash (+3 more)

### Community 82 - "Community 82"
Cohesion: 0.22
Nodes (9): Add Items, Bloom Filters (For membership testing), Check Items, code:python (from app.utils.cache import create_bloom_filter), code:python (from app.utils.cache import add_to_bloom_filter), code:python (from app.utils.cache import check_bloom_filter), code:python (from app.utils.cache import get_bloom_filter_info), Create Bloom Filter (+1 more)

### Community 83 - "Community 83"
Cohesion: 0.25
Nodes (8): float, _bloom_filter_exists(), create_bloom_filter(), Create a Bloom filter (BF.RESERVE).      Args:         redis: Redis client insta, Case-insensitive Redis error matching for command capability checks., Return whether a Redis Bloom filter already exists., _redis_error_contains(), RedisError

### Community 84 - "Community 84"
Cohesion: 0.29
Nodes (7): AsyncClient, HTTPConnection, create_httpx_client(), get_httpx_client(), HTTPX client with optimal performance settings., Create production-grade HTTPX client with HTTP/2 and connection pooling.      Ke, Dependency to inject HTTPX client.

### Community 85 - "Community 85"
Cohesion: 0.36
Nodes (5): Any, str, Get proxy configuration for Crawl4AI., Convert to Crawl4AI BrowserConfig kwargs., Convert to Crawl4AI CrawlerRunConfig kwargs.

### Community 86 - "Community 86"
Cohesion: 0.29
Nodes (7): AsyncPostgresSaver, str, Async PostgreSQL checkpointer for LangGraph persistence.  Uses the existing Post, Initialize AsyncPostgresSaver using existing PostgreSQL connection.      AsyncPo, Graceful shutdown of AsyncPostgresSaver connection pool.      Closes all active, setup_langgraph_checkpointer(), teardown_langgraph_checkpointer()

### Community 87 - "Community 87"
Cohesion: 0.29
Nodes (7): AnnotatedDocument, int, GraphIngestionContext, ingest_extractions_to_graph(), Narrow context for graph mapping., Convert grounded LangExtract output → property graph nodes/rels., # TODO: Design a multi-pass LangExtract prompting strategy to first extract enti

### Community 88 - "Community 88"
Cohesion: 0.33
Nodes (6): HTTPConnection, Redis, str, create_redis_client(), get_redis(), Create and return a configured async Redis client.     The client is connection-

### Community 89 - "Community 89"
Cohesion: 0.29
Nodes (6): AsyncIOMotorClient, AsyncIOMotorDatabase, str, create_mongo_client(), MongoDB connection and database management., Initialize database connection using Beanie's recommended approach.

### Community 90 - "Community 90"
Cohesion: 0.29
Nodes (7): Caching User Sessions, code:python (from fastapi import APIRouter, Depends), code:python (from app.utils.cache import push_to_list, get_list_items, tr), code:python (from app.utils.cache import set_hash, get_hash, update_hash), Real-World Examples, Recent Activity Feed, User Profile with Selective Updates

### Community 91 - "Community 91"
Cohesion: 0.29
Nodes (7): code:python (from app.utils.cache import create_search_index), code:python (from app.utils.cache import search_index), code:python (from app.utils.cache import delete_search_index), Create Index, Delete Index, Search Index, Search Indexes (For full-text search on hashes)

### Community 92 - "Community 92"
Cohesion: 0.29
Nodes (7): code:python (from app.utils.cache import set_cache), code:python (from app.utils.cache import get_cache), code:python (from app.utils.cache import delete_cache), Delete Cache, Get Cache, Set Cache, String Caching (Recommended for simple values & JSON)

### Community 93 - "Community 93"
Cohesion: 0.40
Nodes (3): float, Validate embedding is not empty if provided., Ensure score is between 0 and 1.

### Community 94 - "Community 94"
Cohesion: 0.50
Nodes (3): __getattr__(), str, LangChain FastAPI Production - Main application package.

### Community 97 - "Community 97"
Cohesion: 0.67
Nodes (3): get_key_name(), Generate a namespaced cache key from object type and args.      Args:         ob, CacheKeyPart

### Community 99 - "Community 99"
Cohesion: 0.67
Nodes (3): CircuitBreakerOpenError, Raised when the circuit breaker is open., RuntimeError

### Community 101 - "Community 101"
Cohesion: 0.67
Nodes (3): object, override_reducer(), Reducer function that allows overriding values in state.

## Knowledge Gaps
- **159 isolated node(s):** `str`, `int`, `str`, `int`, `str` (+154 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseModel` connect `Community 6` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 8`, `Community 9`, `Community 10`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 17`, `Community 18`, `Community 21`, `Community 24`, `Community 25`, `Community 26`, `Community 27`, `Community 28`, `Community 30`, `Community 31`, `Community 32`, `Community 34`, `Community 38`, `Community 39`, `Community 40`, `Community 42`, `Community 46`, `Community 47`, `Community 48`, `Community 52`, `Community 53`, `Community 54`, `Community 56`, `Community 61`, `Community 62`, `Community 64`, `Community 65`, `Community 67`, `Community 87`?**
  _High betweenness centrality (0.478) - this node is a cross-community bridge._
- **Why does `get_settings()` connect `Community 28` to `Community 0`, `Community 1`, `Community 8`, `Community 12`, `Community 18`, `Community 19`, `Community 21`, `Community 24`, `Community 25`, `Community 27`, `Community 29`, `Community 30`, `Community 35`, `Community 37`, `Community 39`, `Community 46`, `Community 48`, `Community 51`, `Community 55`, `Community 57`, `Community 60`, `Community 62`, `Community 76`, `Community 78`, `Community 84`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Why does `BaseTool` connect `Community 24` to `Community 6`, `Community 71`, `Community 10`, `Community 48`, `Community 52`, `Community 20`, `Community 56`, `Community 27`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `BaseModel` (e.g. with `BaseMessage` and `BaseTool`) actually correct?**
  _`BaseModel` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 48 inferred relationships involving `AuthService` (e.g. with `Depends` and `HTTPConnection`) actually correct?**
  _`AuthService` has 48 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `BaseMessage` (e.g. with `AgentError` and `Citation`) actually correct?**
  _`BaseMessage` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `SearchRepository` (e.g. with `AsyncSession` and `Depends`) actually correct?**
  _`SearchRepository` has 46 INFERRED edges - model-reasoned connections that need verification._