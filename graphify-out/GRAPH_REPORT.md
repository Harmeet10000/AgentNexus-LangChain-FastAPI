# Graph Report - langchain-fastapi-production  (2026-05-28)

## Corpus Check
- 257 files · ~224,309 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2811 nodes · 6233 edges · 80 communities detected
- Extraction: 57% EXTRACTED · 43% INFERRED · 0% AMBIGUOUS · INFERRED: 2700 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

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
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 105|Community 105]]
- [[_COMMUNITY_Community 106|Community 106]]
- [[_COMMUNITY_Community 111|Community 111]]
- [[_COMMUNITY_Community 112|Community 112]]
- [[_COMMUNITY_Community 113|Community 113]]
- [[_COMMUNITY_Community 114|Community 114]]
- [[_COMMUNITY_Community 115|Community 115]]

## God Nodes (most connected - your core abstractions)
1. `DatabaseException` - 63 edges
2. `WorkflowStatus` - 54 edges
3. `HITLInterruptType` - 46 edges
4. `LegalAgentState` - 44 edges
5. `Configuration` - 44 edges
6. `get_settings()` - 44 edges
7. `LegalEdgeInput` - 43 edges
8. `ClauseEpisodeMetadata` - 42 edges
9. `FinalReportEpisodeMetadata` - 42 edges
10. `ResearchComplete` - 42 edges

## Surprising Connections (you probably didn't know these)
- `Celery Application` --semantically_similar_to--> `AOT Cron Scheduler`  [INFERRED] [semantically similar]
  src/app/examples/CELERY.md → docs/Lynk_Plan/api_server_plan.md
- `WebSocketSecurityService` --uses--> `Application lifespan management.`  [INFERRED]
  src/app/features/auth/websocket_security.py → app/lifecycle/lifespan.py
- `WebSocketSecurityService` --uses--> `Initialize Redis with health check.`  [INFERRED]
  src/app/features/auth/websocket_security.py → app/lifecycle/lifespan.py
- `WebSocketSecurityService` --uses--> `Initialize MongoDB with health check.`  [INFERRED]
  src/app/features/auth/websocket_security.py → app/lifecycle/lifespan.py
- `WebSocketSecurityService` --uses--> `Initialize Neo4j with connectivity verification.`  [INFERRED]
  src/app/features/auth/websocket_security.py → app/lifecycle/lifespan.py

## Hyperedges (group relationships)
- **Production MCP Surface** — fastmcp_guide_mcp_server_foundation, fastmcp_guide_mcp_security, fastmcp_guide_mcp_metrics, fastmcp_guide_internal_mcp_client_manager [EXTRACTED 1.00]
- **Advanced RAG Strategy Set** — rag_readme_reranking, rag_readme_contextual_retrieval, rag_readme_hierarchical_rag, rag_readme_self_reflective_rag, rag_readme_fine_tuned_embeddings [EXTRACTED 1.00]
- **Legal Agent Control Architecture** — cap_ai_agents_legal_agent_architecture, readme_langgraph_orchestrator, readme_human_in_loop, ai_agent_engineering_verification_hierarchy [INFERRED 0.86]
- **Write Path Memory Persistence Flow** — memory_stack_architecture_relationship_mapping, memory_stack_architecture_write_clause_episodes, memory_stack_architecture_graphiti_write_store, memory_stack_architecture_postgres_write_store [EXTRACTED 1.00]
- **Read Path Agent Context Flow** — memory_stack_architecture_risk_analysis, memory_stack_architecture_compliance, memory_stack_architecture_build_agent_context, memory_stack_architecture_graphiti_risk_context, memory_stack_architecture_pgvector, memory_stack_architecture_postgres_read_store [EXTRACTED 1.00]
- **Tool Call Idempotency And Durable Audit Flow** — memory_stack_architecture_sha256_tool_call_key, memory_stack_architecture_redis_idempotency, memory_stack_architecture_postgres_tool_executions [EXTRACTED 1.00]
- **Public Ingress Flow** — agent_saul_deployment_topology_users, agent_saul_deployment_topology_cloudflare, agent_saul_deployment_topology_frontend, agent_saul_deployment_topology_backend_api, agent_saul_deployment_topology_docs_frontend, agent_saul_deployment_topology_keycloak [EXTRACTED 1.00]
- **Railway Compute Plane Services** — agent_saul_deployment_topology_railway_compute_plane, agent_saul_deployment_topology_public_internet_facing, agent_saul_deployment_topology_private_railway_internal_only, agent_saul_deployment_topology_frontend, agent_saul_deployment_topology_backend_api, agent_saul_deployment_topology_openclaw, agent_saul_deployment_topology_worker_ops, agent_saul_deployment_topology_worker_llm, agent_saul_deployment_topology_scheduler [EXTRACTED 1.00]
- **API And Agents To Managed Stores** — agent_saul_deployment_topology_backend_api, agent_saul_deployment_topology_openclaw, agent_saul_deployment_topology_external_managed_data_plane, agent_saul_deployment_topology_timescale, agent_saul_deployment_topology_neo4j_aura, agent_saul_deployment_topology_mongodb, agent_saul_deployment_topology_storage [EXTRACTED 1.00]
- **Hybrid Query Cache Lookup Flow** — hybrid_query_pipeline_incoming_query, hybrid_query_pipeline_metadata_filters, hybrid_query_pipeline_redis_cache_check, hybrid_query_pipeline_query_filter_hash, hybrid_query_pipeline_cached_result [EXTRACTED 1.00]
- **Parallel BM25 and Vector Retrieval Flow** — hybrid_query_pipeline_embed_query, hybrid_query_pipeline_bm25_search, hybrid_query_pipeline_vector_ann_search, hybrid_query_pipeline_top_50 [EXTRACTED 1.00]
- **RRF Fusion Result Delivery Flow** — hybrid_query_pipeline_bm25_search, hybrid_query_pipeline_vector_ann_search, hybrid_query_pipeline_rrf_fusion, hybrid_query_pipeline_top_n_fused_results, hybrid_query_pipeline_fts_ranked_chunks, hybrid_query_pipeline_rag_context_assembly [EXTRACTED 1.00]
- **API to Storage Access Flow** — search_system_architecture_overview_api_layer, search_system_architecture_overview_pgbouncer, search_system_architecture_overview_postgresql [INFERRED 0.80]
- **Ingestion Persistence Flow** — search_system_architecture_overview_api_layer, search_system_architecture_overview_ingestion_workers, search_system_architecture_overview_pgbouncer, search_system_architecture_overview_postgresql, search_system_architecture_overview_chunks [EXTRACTED 1.00]
- **PostgreSQL Search Capabilities** — search_system_architecture_overview_postgresql, search_system_architecture_overview_pgvector, search_system_architecture_overview_pgvectorscale, search_system_architecture_overview_pg_textsearch, search_system_architecture_overview_pg_trgm, search_system_architecture_overview_chunks [EXTRACTED 1.00]
- **HTTP Upload Ingestion Flow** — agent_saul_full_architecture_client, agent_saul_full_architecture_ingestion_router_service, agent_saul_full_architecture_ingestiongraph, agent_saul_full_architecture_doc_id_returned [EXTRACTED 1.00]
- **WebSocket Agent Saul Analysis Flow** — agent_saul_full_architecture_client, agent_saul_full_architecture_ws_router_service, agent_saul_full_architecture_agent_saul_langgraph, agent_saul_full_architecture_ws_stream_tokens_events [EXTRACTED 1.00]
- **Background Reconciliation and Memory Decay Flow** — agent_saul_full_architecture_celery_beat, agent_saul_full_architecture_reconciliationgraph, agent_saul_full_architecture_memory_decay, agent_saul_full_architecture_shared_infrastructure [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (126): _extract_raw_token(), extract_raw_token_from_connection(), get_auth_service(), get_current_active_user(), get_current_user(), get_current_verified_user(), get_refresh_token_repository(), get_token_claims() (+118 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (160): AgentRegistry, build_agent_registry(), build_saul_graph(), Graph factory for Agent Saul.  Entry point: build_saul_graph(checkpointer, pro_l, Instantiate all agents + LLM chains once.     Called from build_saul_graph — nev, Build and compile the Agent Saul LangGraph.      Graph topology:       START → g, Holds all pre-built agents and structured-output LLM chains.     Created once at, SaulGraphNodes (+152 more)

### Community 2 - "Community 2"
Cohesion: 0.02
Nodes (173): BaseModel, Chunk, A chunk of content with metadata., ExtractionResult, Result from Gemini extraction., DeclarativeBase, chunk_document(), chunk_document_simple() (+165 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (151): add_to_bloom_filter(), _append_search_filter_args(), _append_search_highlight_args(), _append_search_summarize_args(), _bloom_filter_exists(), _build_create_search_index_args(), _build_database_exception(), _build_search_index_query_args() (+143 more)

### Community 4 - "Community 4"
Cohesion: 0.03
Nodes (106): Truncate content to maximum length with warning., truncate_content(), CrawlerConfig, get_crawler_config(), load_from_settings(), Crawler configuration and settings., Configuration for the web crawler., Get proxy configuration for Crawl4AI. (+98 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (95): _extract_compliance_output(), _extract_risk_output(), make_compliance_node(), make_finalization_node(), make_grounding_verification_node(), make_human_review_node(), make_persist_memory_node(), make_risk_analysis_node() (+87 more)

### Community 6 - "Community 6"
Cohesion: 0.03
Nodes (75): generate_text(), get_circuit_breaker(), create_app(), Create and configure FastAPI application with proper middleware order., main(), Cache utilities using Redis., Adapt a project Redis client to FastAPI Guard's Redis handler protocol., RedisProtocolAdapter (+67 more)

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (94): chunk_text(), Token-based chunking helpers for search ingestion., Split text into overlapping token windows while preserving order., Normalized text chunk plus its ordinal position., TextChunk, Search feature constants., get_search_repository(), get_search_service() (+86 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (112): MessagesState, Configuration, MCPConfig, Configuration management for the Open Deep Research system., Enumeration of available search API providers., Configuration for Model Context Protocol (MCP) servers., Main configuration class for the Deep Research agent., SearchAPI (+104 more)

### Community 9 - "Community 9"
Cohesion: 0.03
Nodes (83): create_celery_app(), get_redis_client(), Celery connection and production reliability configuration., Create and configure Celery application., Base Celery task with retries, observability, and reliability helpers., acquire_idempotency_lock(), build_circuit_breaker_key(), build_closed_circuit_breaker_state() (+75 more)

### Community 10 - "Community 10"
Cohesion: 0.02
Nodes (91): AI Agent Control Loop, Context Engineering, Agent Memory Architecture, Multi-agent Patterns, Agent Observability, Agent Safety Primitives, Agent Tool Design, Agent Verification Hierarchy (+83 more)

### Community 11 - "Community 11"
Cohesion: 0.04
Nodes (58): BaseStore, BoundLogger, create_subgraph_expander(), detect_conflicts(), expand_from_seeds(), get_obligation_chain(), Neo4jSubgraphConfig, _parse_subgraph_records() (+50 more)

### Community 12 - "Community 12"
Cohesion: 0.03
Nodes (72): create_httpx_client(), get_httpx_client(), get_shared_httpx_client(), HTTPX client with optimal performance settings., Create production-grade HTTPX client with HTTP/2 and connection pooling.      Ke, Return a process-wide async HTTPX client for non-request runtimes., Dependency to inject HTTPX client., build_extraction_chain() (+64 more)

### Community 13 - "Community 13"
Cohesion: 0.04
Nodes (56): forgot_password(), get_me(), list_sessions(), login(), logout(), oauth_authorize(), refresh_token(), register() (+48 more)

### Community 14 - "Community 14"
Cohesion: 0.06
Nodes (46): AgentSpec, create_production_agent(), ProductionAgent, Agent factory — the main entry point for creating production agents.  Uses LangC, Build a fully configured production agent from a spec.      Returns a Production, Wraps a compiled LangGraph agent with production runtime behaviour:     - Long-t, Single async invocation.          Args:             user_message: The user's inp, Stream the agent's response token by token.          stream_mode options: "messa (+38 more)

### Community 15 - "Community 15"
Cohesion: 0.09
Nodes (65): check_gpu_available(), convert_document(), create_converter(), create_document_converter(), _detect_language(), _encode_base64(), extract_code_blocks(), _extract_code_fallback() (+57 more)

### Community 16 - "Community 16"
Cohesion: 0.06
Nodes (52): Exception, build_ingestion_graph(), Contract KB ingestion graph factory., Build the contract ingestion graph once during application startup., _structured(), Contract KB ingestion LangGraph package., _cached_embedding(), _call_embedding_fn() (+44 more)

### Community 17 - "Community 17"
Cohesion: 0.06
Nodes (32): build_validation_error_handler(), format_tool_validation_error(), make_structured_tool(), Tool base classes and registry.  All agent tools use structured Pydantic input/o, Return a schema-first validation error message the model can retry against., Create a stable validation error formatter bound to a specific schema., Decorator to register a @tool-decorated function in the global registry.      Us, Wrap an async function as a StructuredTool with typed input.     Automatically r (+24 more)

### Community 18 - "Community 18"
Cohesion: 0.06
Nodes (43): Agent Runtime, Alloy OTLP, Analytics Flags, Atlas Docs, Auth DB, Auth Enforcement, Backend API, Beat Cron (+35 more)

### Community 19 - "Community 19"
Cohesion: 0.11
Nodes (23): get_health_service(), Dependency wiring for health feature., HealthChecksDTO, HealthDataDTO, HealthResultDTO, DTOs for health feature responses., Per-component health checks., Aggregated health payload. (+15 more)

### Community 20 - "Community 20"
Cohesion: 0.06
Nodes (42): agent_saul LangGraph, Celery beat, Client, Cognee, comply, doc_id Returned, extract, Extract Validate Embed Store (+34 more)

### Community 21 - "Community 21"
Cohesion: 0.1
Nodes (29): BaseTool, CrawlUrlInput, CrawlUrlTool, LangChain tool for web crawling., Input schema for crawl URL tool., Tool for crawling a URL and extracting content., Agent tools for web search, crawling and document processing., get_all_tools() (+21 more)

### Community 22 - "Community 22"
Cohesion: 0.11
Nodes (26): build_retrieval_graph(), Retrieval graph factory for canonical clauses KB., Build a request-scoped retrieval graph over clauses., _structured(), Clauses-backed legal retrieval graph., _cached_embedding(), _call_embedding_fn(), make_context_grader_node() (+18 more)

### Community 23 - "Community 23"
Cohesion: 0.07
Nodes (35): API Layer, Up to 500 App Clients, Async SQLAlchemy, BM25 Full-text Search, Celery, Chunk Embed Upsert Pipeline, Chunk Content Embedding BM25 Index Fields, chunks Table (+27 more)

### Community 24 - "Community 24"
Cohesion: 0.13
Nodes (24): _get_sdk_client(), PageIndexBatchConfig, PageIndexChatConfig, PageIndexClient, PageIndexConfig, Configuration for indexing operations., Concurrency settings for batch indexing., Configuration for chat completion calls. (+16 more)

### Community 25 - "Community 25"
Cohesion: 0.09
Nodes (17): DocumentIngestionPipeline, main(), Main ingestion script for processing markdown documents into vector DB and knowl, Ingest a single document.          Args:             file_path: Path to the docu, Find all supported document files in the documents folder., Pipeline for ingesting documents into vector DB and knowledge graph., Read document content from file - supports multiple formats via Docling., Initialize ingestion pipeline.          Args:             config: Ingestion conf (+9 more)

### Community 26 - "Community 26"
Cohesion: 0.1
Nodes (26): close_tavily_http_client(), create_tavily_http_client(), get_shared_tavily_http_client(), get_tavily_http_client(), Tavily search client initialization and dependency injection., Return a process-wide async HTTPX client for Tavily requests.      Used for non-, Dependency to inject Tavily HTTP client from request lifespan., Create Tavily HTTP client for FastAPI lifespan management. (+18 more)

### Community 27 - "Community 27"
Cohesion: 0.08
Nodes (12): AsyncCallbackHandler, BaseCallbackHandler, AsyncStreamingCallbackHandler, configure_langsmith(), LatencyCallbackHandler, LangSmith observability bootstrap and custom callbacks. Must be imported before, Bootstrap LangSmith tracing by setting env vars.     Call this at application st, Tracks per-run latency for structured logging. (+4 more)

### Community 28 - "Community 28"
Cohesion: 0.12
Nodes (17): build_reconciliation_graph(), Build the reconciliation graph once during application startup., make_apply_changes_node(), make_fetch_existing_node(), make_reconcile_node(), make_write_versions_node(), _parse_reconciliation_decision(), Reconciliation graph nodes for background entity deduplication and conflict reso (+9 more)

### Community 29 - "Community 29"
Cohesion: 0.1
Nodes (20): do_run_migrations(), Run migrations in 'offline' mode.      This is used for generating migration scr, Run migrations with the provided database connection.      Args:         connect, Run migrations in async mode using init_db() to get the engine., Run migrations in 'online' mode.      This connects to the database and applies, run_async_migrations(), run_migrations_offline(), run_migrations_online() (+12 more)

### Community 30 - "Community 30"
Cohesion: 0.14
Nodes (16): AsyncAttrs, Base, Database package with Base and all schemas., ChatMessage, ChatSession, Chat messages schema for storing user-LLM conversations., Store chat messages between user and LLM., Store chat session metadata. (+8 more)

### Community 31 - "Community 31"
Cohesion: 0.11
Nodes (20): _compute_decay(), DecayStats, Background helpers for memory decay and reconciliation workflows., Run reconciliation sequentially for each user id., Run memory decay from a synchronous task runner., Run reconciliation for a single user., Run reconciliation for an already-resolved set of active users., Summary returned by the decay workflow. (+12 more)

### Community 32 - "Community 32"
Cohesion: 0.11
Nodes (22): 768-Dimensional Embedding, BM25 Search, Cache Hit, Cache Miss, Cached Result, Embed Query, FastAPI, FTS Ranked Chunks (+14 more)

### Community 33 - "Community 33"
Cohesion: 0.12
Nodes (20): Agentic RAG Router Pattern, Hybrid Retrieval, HyDE Rejected for Legal RAG V1, Knowledge Graph Retrieval, Long Context versus RAG, Multi-modal RAG, Reciprocal Rank Fusion, Agentic RAG (+12 more)

### Community 34 - "Community 34"
Cohesion: 0.19
Nodes (8): DocumentUploadResponse, Ingestion feature: HTTP upload endpoint that runs IngestionGraph before WS.  Flo, Ingestion router: POST /ingestion/documents/upload  Accepts multipart form: file, Upload flow:       1. Read raw bytes from uploaded file.       2. Run IngestionG, Upload flow:       1. Read raw bytes from uploaded file.       2. Run IngestionG, upload_document(), IngestionService, IngestionService: runs IngestionGraph for a given uploaded document.  Called by

### Community 35 - "Community 35"
Cohesion: 0.2
Nodes (13): CleanLegalDocument, DoclingProcessingContext, preprocess_legal_document(), Narrow context for document preprocessing., Structured output from preprocessing., Async wrapper around Docling (CPU-heavy)., BatchExtractionResult, LangExtractBatchContext (+5 more)

### Community 36 - "Community 36"
Cohesion: 0.14
Nodes (15): Celery decay nightly, Cognee episodic procedural memory, entities.decay_score and clauses.decay_score, events immutable audit log, events dual write, Graphiti Neo4j episodes and entity edges, memory_versions CRDT snapshots, persist_memory node (+7 more)

### Community 37 - "Community 37"
Cohesion: 0.16
Nodes (13): clean_markdown(), extract_headers(), extract_title_from_markdown(), get_chunk_summary(), Content chunking utilities for crawled content., Extract title from markdown content., Clean and normalize markdown content., Get a summary of a chunk. (+5 more)

### Community 38 - "Community 38"
Cohesion: 0.15
Nodes (14): app.state.neo4j_driver, build_agent_context(), compliance, COMPLIANCE: +CONTRACT depth=1, Graphiti search_for_risk_context, memory_pipeline.py, MemoryScope, MemoryScope enforcement (+6 more)

### Community 39 - "Community 39"
Cohesion: 0.18
Nodes (8): AgentSaulDeps, get_agent_saul_deps(), get_agent_saul_ws_security_context(), get_current_user_id(), get_saul_checkpointer(), FastAPI dependencies for agent_saul.  All infra clients are read from request.ap, Stub — replace with your project's JWT/session auth dependency.     The user_id, Narrow context object for Agent Saul dependencies.      Typed against infra prot

### Community 40 - "Community 40"
Cohesion: 0.33
Nodes (5): _is_envelope_violation(), Global API response envelope enforcement for routers., APIRouter that validates `response_model` uses `APIResponse[T]`., StrictEnvelopeAPIRouter, APIRouter

### Community 41 - "Community 41"
Cohesion: 0.5
Nodes (3): build_open_deep_search_config(), Open Deep Search package exports., Build graph config with the lifespan-owned HTTPX client attached.

### Community 42 - "Community 42"
Cohesion: 0.5
Nodes (2): BaseOutputParser, ToonParser

### Community 43 - "Community 43"
Cohesion: 0.5
Nodes (1): Contract KB parent documents and pg_textsearch clauses.  Revision ID: 9f4a1b7c6d

### Community 44 - "Community 44"
Cohesion: 0.5
Nodes (1): Add search documents and chunks schema  Revision ID: 8a7d9b1c2e3f Revises: 2bc77

### Community 45 - "Community 45"
Cohesion: 0.5
Nodes (1): Initial schema: document_vectors and chat tables  Revision ID: c0c17c6eb1cc Revi

### Community 46 - "Community 46"
Cohesion: 0.5
Nodes (1): rename_metadata_to_meta_data  Revision ID: 2bc7726317f6 Revises: c0c17c6eb1cc Cr

### Community 47 - "Community 47"
Cohesion: 0.5
Nodes (4): Private railway.internal Only Zone, Public Internet-Facing Zone, Railway Compute Plane, railway.internal Private Networking

### Community 48 - "Community 48"
Cohesion: 0.67
Nodes (1): LangChain FastAPI Production - Main application package.

### Community 49 - "Community 49"
Cohesion: 0.67
Nodes (2): Run all seeders in order., run_all_seeders()

### Community 50 - "Community 50"
Cohesion: 0.67
Nodes (3): Database Connection Pool Sizing, Kingman's Law, Little's Law

### Community 51 - "Community 51"
Cohesion: 0.67
Nodes (3): auto_explain, EXPLAIN ANALYZE BUFFERS, pg_stat_statements

### Community 52 - "Community 52"
Cohesion: 0.67
Nodes (3): LLM Attack Vectors, ModernBERT Safety Discriminator, LLM Zero Trust Gap

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (3): Postgres tool_executions durable audit, Redis idempotency hot path, SHA-256(step_id + input + user_id) tool call key

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Application features/domain modules.

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Knowledge base feature.

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Web scraping feature.

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): RAG (Retrieval-Augmented Generation) utilities.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): LangExtract async utilities.

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Agent definitions and utilities.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Vector store utilities.

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Alembic revision modules.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (2): postgresql.conf Tuning, work_mem Risk

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (2): PostgreSQL Index Design, Linux perf PostgreSQL CPU Analysis

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (2): Agent Saul, Production Deployment Topology

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (2): Backstage, Dev Portal Catalog

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (2): CI/CD Control Plane, Jenkins

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (1): Ensure score is between 0 and 1.

### Community 92 - "Community 92"
Cohesion: 1.0
Nodes (1): Validate embedding is not empty if provided.

### Community 93 - "Community 93"
Cohesion: 1.0
Nodes (1): Ensure overlap is less than chunk size.

### Community 97 - "Community 97"
Cohesion: 1.0
Nodes (1): Create a Configuration instance from a RunnableConfig.

### Community 98 - "Community 98"
Cohesion: 1.0
Nodes (1): Strip leading/trailing whitespace from optional fields.

### Community 99 - "Community 99"
Cohesion: 1.0
Nodes (1): Cross-field validation (e.g., role minimum length).

### Community 105 - "Community 105"
Cohesion: 1.0
Nodes (1): Build a deterministic SHA-256 key for a tool invocation.

### Community 106 - "Community 106"
Cohesion: 1.0
Nodes (1): Load configuration from settings.

### Community 111 - "Community 111"
Cohesion: 1.0
Nodes (1): Immutable collection of all pre-built LangChain tools.      Tool assignment to a

### Community 112 - "Community 112"
Cohesion: 1.0
Nodes (1): Build all tools once.  Call at lifespan startup only.

### Community 113 - "Community 113"
Cohesion: 1.0
Nodes (1): Main configuration class for the Deep Research agent.

### Community 114 - "Community 114"
Cohesion: 1.0
Nodes (1): Create a Configuration instance from a RunnableConfig.

### Community 115 - "Community 115"
Cohesion: 1.0
Nodes (1): Pydantic configuration.

## Ambiguous Edges - Review These
- `norm` → `rel_map`  [AMBIGUOUS]
  docs/diagrams/agent_saul_full_architecture.svg · relation: calls
- `risk` → `review HITL`  [AMBIGUOUS]
  docs/diagrams/agent_saul_full_architecture.svg · relation: calls

## Knowledge Gaps
- **565 isolated node(s):** `LangChain FastAPI Production - Main application package.`, `Create and configure FastAPI application with proper middleware order.`, `Application features/domain modules.`, `FastAPI dependencies for agent_saul.  All infra clients are read from request.ap`, `Stub — replace with your project's JWT/session auth dependency.     The user_id` (+560 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 42`** (4 nodes): `BaseOutputParser`, `toon_parser.py`, `ToonParser`, `.parse()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (4 nodes): `9f4a1b7c6d2e_contract_kb_clauses_pg_textsearch.py`, `downgrade()`, `Contract KB parent documents and pg_textsearch clauses.  Revision ID: 9f4a1b7c6d`, `upgrade()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (4 nodes): `8a7d9b1c2e3f_add_search_documents_and_chunks.py`, `downgrade()`, `Add search documents and chunks schema  Revision ID: 8a7d9b1c2e3f Revises: 2bc77`, `upgrade()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (4 nodes): `c0c17c6eb1cc_initial_schema_document_vectors_and_.py`, `downgrade()`, `Initial schema: document_vectors and chat tables  Revision ID: c0c17c6eb1cc Revi`, `upgrade()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (4 nodes): `2bc7726317f6_rename_metadata_to_meta_data.py`, `downgrade()`, `rename_metadata_to_meta_data  Revision ID: 2bc7726317f6 Revises: c0c17c6eb1cc Cr`, `upgrade()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (3 nodes): `__getattr__()`, `LangChain FastAPI Production - Main application package.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (3 nodes): `Run all seeders in order.`, `run_all_seeders()`, `run_seeders.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (2 nodes): `Application features/domain modules.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (2 nodes): `Knowledge base feature.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (2 nodes): `__init__.py`, `Web scraping feature.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (2 nodes): `RAG (Retrieval-Augmented Generation) utilities.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (2 nodes): `LangExtract async utilities.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (2 nodes): `Agent definitions and utilities.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (2 nodes): `__init__.py`, `Vector store utilities.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (2 nodes): `__init__.py`, `Alembic revision modules.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (2 nodes): `postgresql.conf Tuning`, `work_mem Risk`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (2 nodes): `PostgreSQL Index Design`, `Linux perf PostgreSQL CPU Analysis`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (2 nodes): `Agent Saul`, `Production Deployment Topology`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (2 nodes): `Backstage`, `Dev Portal Catalog`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (2 nodes): `CI/CD Control Plane`, `Jenkins`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (1 nodes): `Ensure score is between 0 and 1.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 92`** (1 nodes): `Validate embedding is not empty if provided.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 93`** (1 nodes): `Ensure overlap is less than chunk size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 97`** (1 nodes): `Create a Configuration instance from a RunnableConfig.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 98`** (1 nodes): `Strip leading/trailing whitespace from optional fields.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 99`** (1 nodes): `Cross-field validation (e.g., role minimum length).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 105`** (1 nodes): `Build a deterministic SHA-256 key for a tool invocation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 106`** (1 nodes): `Load configuration from settings.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 111`** (1 nodes): `Immutable collection of all pre-built LangChain tools.      Tool assignment to a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 112`** (1 nodes): `Build all tools once.  Call at lifespan startup only.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 113`** (1 nodes): `Main configuration class for the Deep Research agent.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 114`** (1 nodes): `Create a Configuration instance from a RunnableConfig.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 115`** (1 nodes): `Pydantic configuration.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `norm` and `rel_map`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **What is the exact relationship between `risk` and `review HITL`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **Why does `get_settings()` connect `Community 6` to `Community 0`, `Community 1`, `Community 4`, `Community 7`, `Community 12`, `Community 13`, `Community 24`, `Community 26`, `Community 27`, `Community 29`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Why does `create_production_agent()` connect `Community 14` to `Community 12`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Why does `SystemPromptParts` connect `Community 14` to `Community 2`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Are the 170 inferred relationships involving `str` (e.g. with `create_session()` and `saul_ws_endpoint()`) actually correct?**
  _`str` has 170 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `DatabaseException` (e.g. with `UserProfile` and `UserUpdate`) actually correct?**
  _`DatabaseException` has 59 INFERRED edges - model-reasoned connections that need verification._