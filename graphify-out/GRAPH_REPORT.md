# Graph Report - src, docs  (2026-04-29)

## Corpus Check
- 276 files · ~188,293 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2751 nodes · 6073 edges · 68 communities detected
- Extraction: 58% EXTRACTED · 42% INFERRED · 0% AMBIGUOUS · INFERRED: 2572 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Graph State Search|Graph State Search]]
- [[_COMMUNITY_Token User Oauth|Token User Oauth]]
- [[_COMMUNITY_Redis Args Search|Redis Args Search]]
- [[_COMMUNITY_Graphiti Neo4j Episode|Graphiti Neo4j Episode]]
- [[_COMMUNITY_Agent Build Chain|Agent Build Chain]]
- [[_COMMUNITY_Crawler Crawl Url|Crawler Crawl Url]]
- [[_COMMUNITY_Client Tavily Run|Client Tavily Run]]
- [[_COMMUNITY_Research Deep State|Research Deep State]]
- [[_COMMUNITY_Circuit Breaker Idempotency|Circuit Breaker Idempotency]]
- [[_COMMUNITY_Args Embedding Chunk|Args Embedding Chunk]]
- [[_COMMUNITY_Search Chunk Rag|Search Chunk Rag]]
- [[_COMMUNITY_Node Make Graph|Node Make Graph]]
- [[_COMMUNITY_Upload User Router|Upload User Router]]
- [[_COMMUNITY_Saul Redis Agent|Saul Redis Agent]]
- [[_COMMUNITY_Agent Mcp Legal|Agent Mcp Legal]]
- [[_COMMUNITY_Connection Websocket Saul|Connection Websocket Saul]]
- [[_COMMUNITY_Cognee Memory Subgraph|Cognee Memory Subgraph]]
- [[_COMMUNITY_Node Make Graph|Node Make Graph]]
- [[_COMMUNITY_Extract Extraction Code|Extract Extraction Code]]
- [[_COMMUNITY_Extract Entity Args|Extract Entity Args]]
- [[_COMMUNITY_Auth Frontend External|Auth Frontend External]]
- [[_COMMUNITY_Health Check Feature|Health Check Feature]]
- [[_COMMUNITY_Hitl Phase Extract|Hitl Phase Extract]]
- [[_COMMUNITY_Tool Web Search|Tool Web Search]]
- [[_COMMUNITY_Search Endpoint Bm25|Search Endpoint Bm25]]
- [[_COMMUNITY_Client Indexing Chat|Client Indexing Chat]]
- [[_COMMUNITY_Llm End Langsmith|Llm End Langsmith]]
- [[_COMMUNITY_Chat Database All|Chat Database All]]
- [[_COMMUNITY_Tool Validation Registry|Tool Validation Registry]]
- [[_COMMUNITY_Query Cache Search|Query Cache Search]]
- [[_COMMUNITY_Rag Retrieval Agentic|Rag Retrieval Agentic]]
- [[_COMMUNITY_Write Decay Memory|Write Decay Memory]]
- [[_COMMUNITY_Risk Context Compliance|Risk Context Compliance]]
- [[_COMMUNITY_Envelope Apirouter Violation|Envelope Apirouter Violation]]
- [[_COMMUNITY_Build Open Deep|Build Open Deep]]
- [[_COMMUNITY_Toon Parser Baseoutputparser|Toon Parser Baseoutputparser]]
- [[_COMMUNITY_Contract Clauses Textsearch|Contract Clauses Textsearch]]
- [[_COMMUNITY_A7d9b1c2e3f Add Search|A7d9b1c2e3f Add Search]]
- [[_COMMUNITY_C0c17c6eb1cc Initial Vectors|C0c17c6eb1cc Initial Vectors]]
- [[_COMMUNITY_Bc7726317f6 Rename Metadata|Bc7726317f6 Rename Metadata]]
- [[_COMMUNITY_Railway Private Internal|Railway Private Internal]]
- [[_COMMUNITY_Getattr Langchain Fastapi|Getattr Langchain Fastapi]]
- [[_COMMUNITY_Run Seeders All|Run Seeders All]]
- [[_COMMUNITY_Law Database Connection|Law Database Connection]]
- [[_COMMUNITY_Explain Auto Analyze|Explain Auto Analyze]]
- [[_COMMUNITY_Llm Attack Vectors|Llm Attack Vectors]]
- [[_COMMUNITY_Tool Postgres Executions|Tool Postgres Executions]]
- [[_COMMUNITY_Application Features Domain|Application Features Domain]]
- [[_COMMUNITY_Knowledge Feature|Knowledge Feature]]
- [[_COMMUNITY_Web Scraping Feature|Web Scraping Feature]]
- [[_COMMUNITY_Rag Retrieval Augmented|Rag Retrieval Augmented]]
- [[_COMMUNITY_Langextract Utilities|Langextract Utilities]]
- [[_COMMUNITY_Agent Definitions Utilities|Agent Definitions Utilities]]
- [[_COMMUNITY_Vector Store Utilities|Vector Store Utilities]]
- [[_COMMUNITY_Alembic Revision Modules|Alembic Revision Modules]]
- [[_COMMUNITY_Postgresql Conf Tuning|Postgresql Conf Tuning]]
- [[_COMMUNITY_Postgresql Index Design|Postgresql Index Design]]
- [[_COMMUNITY_Agent Saul Production|Agent Saul Production]]
- [[_COMMUNITY_Backstage Dev Portal|Backstage Dev Portal]]
- [[_COMMUNITY_Control Plane Jenkins|Control Plane Jenkins]]
- [[_COMMUNITY_Ensure Score Between|Ensure Score Between]]
- [[_COMMUNITY_Validate Embedding Not|Validate Embedding Not]]
- [[_COMMUNITY_Ensure Overlap Less|Ensure Overlap Less]]
- [[_COMMUNITY_Configuration Instance Runnableconfig|Configuration Instance Runnableconfig]]
- [[_COMMUNITY_Strip Leading Trailing|Strip Leading Trailing]]
- [[_COMMUNITY_Cross Field Validation|Cross Field Validation]]
- [[_COMMUNITY_Build Deterministic Sha|Build Deterministic Sha]]
- [[_COMMUNITY_Load Configuration Settings|Load Configuration Settings]]

## God Nodes (most connected - your core abstractions)
1. `DatabaseException` - 63 edges
2. `WorkflowStatus` - 54 edges
3. `HITLInterruptType` - 46 edges
4. `Configuration` - 44 edges
5. `get_settings()` - 44 edges
6. `ResearchComplete` - 42 edges
7. `LegalAgentState` - 38 edges
8. `AuthService` - 37 edges
9. `DoclingExtractionResult` - 37 edges
10. `SearchResponse` - 36 edges

## Surprising Connections (you probably didn't know these)
- `AOT Cron Scheduler` --semantically_similar_to--> `Celery Application`  [INFERRED] [semantically similar]
  docs/Lynk_Plan/api_server_plan.md → src/app/examples/CELERY.md
- `MCP Tool Design Best Practices` --semantically_similar_to--> `Curated Read Only MCP Tools`  [INFERRED] [semantically similar]
  docs/MCP_things.md → src/app/examples/FastMCP-guide.md
- `Broker-backed Background Tasks` --semantically_similar_to--> `RabbitMQ Queue Topology`  [INFERRED] [semantically similar]
  docs/Lynk_Plan/api_server_plan.md → src/app/examples/CELERY.md
- `ingest_search_document()` --calls--> `run_ingestion_task()`  [INFERRED]
  tasks/search_tasks.py → app/features/search/service.py
- `Celery tasks for search ingestion.` --uses--> `ResilientTask`  [INFERRED]
  tasks/search_tasks.py → app/connections/celery.py

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

### Community 0 - "Graph State Search"
Cohesion: 0.02
Nodes (145): Citation, ClauseSegment, ClauseType, ComplianceFinding, DocumentSection, EntityType, PlanActionType, RelationshipType (+137 more)

### Community 1 - "Token User Oauth"
Cohesion: 0.03
Nodes (97): _extract_raw_token(), extract_raw_token_from_connection(), get_auth_service(), get_current_active_user(), get_current_user(), get_current_verified_user(), get_refresh_token_repository(), get_token_claims() (+89 more)

### Community 2 - "Redis Args Search"
Cohesion: 0.03
Nodes (139): add_to_bloom_filter(), _append_search_filter_args(), _append_search_highlight_args(), _append_search_summarize_args(), _bloom_filter_exists(), _build_create_search_index_args(), _build_database_exception(), _build_search_index_query_args() (+131 more)

### Community 3 - "Graphiti Neo4j Episode"
Cohesion: 0.03
Nodes (115): close_neo4j_driver(), get_neo4j_session(), init_neo4j(), Neo4j database configuration with driver management., Initialize Neo4j driver and test connection.      Returns:         AsyncDriver:, Context manager for Neo4j sessions.      Args:         driver: Neo4j async drive, Close Neo4j driver and cleanup connections.      Args:         driver: Neo4j asy, close_graphiti() (+107 more)

### Community 4 - "Agent Build Chain"
Cohesion: 0.02
Nodes (111): AgentSpec, create_production_agent(), ProductionAgent, Agent factory — the main entry point for creating production agents.  Uses LangC, Build a fully configured production agent from a spec.      Returns a Production, Wraps a compiled LangGraph agent with production runtime behaviour:     - Long-t, Single async invocation.          Args:             user_message: The user's inp, Stream the agent's response token by token.          stream_mode options: "messa (+103 more)

### Community 5 - "Crawler Crawl Url"
Cohesion: 0.04
Nodes (101): Truncate content to maximum length with warning., truncate_content(), CrawlerConfig, get_crawler_config(), load_from_settings(), Crawler configuration and settings., Configuration for the web crawler., Get proxy configuration for Crawl4AI. (+93 more)

### Community 6 - "Client Tavily Run"
Cohesion: 0.02
Nodes (97): do_run_migrations(), Run migrations in 'offline' mode.      This is used for generating migration scr, Run migrations with the provided database connection.      Args:         connect, Run migrations in async mode using init_db() to get the engine., Run migrations in 'online' mode.      This connects to the database and applies, run_async_migrations(), run_migrations_offline(), run_migrations_online() (+89 more)

### Community 7 - "Research Deep State"
Cohesion: 0.06
Nodes (114): MessagesState, Config, Configuration, MCPConfig, Configuration management for the Open Deep Research system., Enumeration of available search API providers., Configuration for Model Context Protocol (MCP) servers., Pydantic configuration. (+106 more)

### Community 8 - "Circuit Breaker Idempotency"
Cohesion: 0.03
Nodes (90): create_celery_app(), get_redis_client(), Celery connection and production reliability configuration., Create and configure Celery application., Base Celery task with retries, observability, and reliability helpers., acquire_idempotency_lock(), build_circuit_breaker_key(), build_closed_circuit_breaker_state() (+82 more)

### Community 9 - "Args Embedding Chunk"
Cohesion: 0.03
Nodes (94): Chunk, clean_markdown(), extract_headers(), extract_title_from_markdown(), get_chunk_summary(), Content chunking utilities for crawled content., A chunk of content with metadata., Extract title from markdown content. (+86 more)

### Community 10 - "Search Chunk Rag"
Cohesion: 0.06
Nodes (83): Base, chunk_text(), Token-based chunking helpers for search ingestion., Split text into overlapping token windows while preserving order., Normalized text chunk plus its ordinal position., TextChunk, Search feature constants., get_search_repository() (+75 more)

### Community 11 - "Node Make Graph"
Cohesion: 0.06
Nodes (88): _extract_compliance_output(), _extract_risk_output(), make_compliance_node(), make_finalization_node(), make_grounding_verification_node(), make_human_review_node(), make_persist_memory_node(), make_risk_analysis_node() (+80 more)

### Community 12 - "Upload User Router"
Cohesion: 0.03
Nodes (71): forgot_password(), get_me(), list_sessions(), login(), logout(), oauth_authorize(), refresh_token(), register() (+63 more)

### Community 13 - "Saul Redis Agent"
Cohesion: 0.03
Nodes (47): AgentSaulDeps, get_agent_saul_deps(), get_agent_saul_ws_security_context(), get_current_user_id(), get_saul_checkpointer(), FastAPI dependencies for agent_saul.  All infra clients are read from request.ap, Stub — replace with your project's JWT/session auth dependency.     The user_id, Narrow context object for Agent Saul dependencies.      Typed against infra prot (+39 more)

### Community 14 - "Agent Mcp Legal"
Cohesion: 0.02
Nodes (91): AI Agent Control Loop, Context Engineering, Agent Memory Architecture, Multi-agent Patterns, Agent Observability, Agent Safety Primitives, Agent Tool Design, Agent Verification Hierarchy (+83 more)

### Community 15 - "Connection Websocket Saul"
Cohesion: 0.07
Nodes (55): CreateSessionRequest, CreateSessionResponse, WebSocket protocol for Agent Saul.  Inbound  (client → server): discriminated on, Graph paused at interrupt(). Client must send WSResumeMessage to continue., Coarse-grained status transition between pipeline stages., Terminal frame.  Client should close the WS after receiving this., POST /agent-saul/sessions — pre-flight before WS connection., First message on a new WS connection.  Kicks off the graph. (+47 more)

### Community 16 - "Cognee Memory Subgraph"
Cohesion: 0.03
Nodes (58): BaseStore, create_subgraph_expander(), detect_conflicts(), expand_from_seeds(), get_obligation_chain(), Neo4jSubgraphConfig, _parse_subgraph_records(), Neo4j Cypher subgraph expander for depth-N traversal.  Two-layer strategy:   Lay (+50 more)

### Community 17 - "Node Make Graph"
Cohesion: 0.04
Nodes (60): Exception, build_ingestion_graph(), _structured(), _cached_embedding(), _call_embedding_fn(), _chunk_metadata_json(), _contract_metadata_json(), _ensure_chunk_enrichment() (+52 more)

### Community 18 - "Extract Extraction Code"
Cohesion: 0.09
Nodes (67): check_gpu_available(), convert_document(), create_converter(), create_document_converter(), _detect_language(), _encode_base64(), extract_code_blocks(), _extract_code_fallback() (+59 more)

### Community 19 - "Extract Entity Args"
Cohesion: 0.08
Nodes (43): ExtractionResult, GeminiProcessor, Result from Gemini extraction., Processor for Gemini-based content extraction and summarization., Summarize content using Gemini.          Args:             content: Content to s, Extract structured data from content using Gemini.          Args:             co, Extract structured data AND create a summary.          Args:             content, create_extractor() (+35 more)

### Community 20 - "Auth Frontend External"
Cohesion: 0.06
Nodes (43): Agent Runtime, Alloy OTLP, Analytics Flags, Atlas Docs, Auth DB, Auth Enforcement, Backend API, Beat Cron (+35 more)

### Community 21 - "Health Check Feature"
Cohesion: 0.11
Nodes (23): get_health_service(), Dependency wiring for health feature., HealthChecksDTO, HealthDataDTO, HealthResultDTO, DTOs for health feature responses., Per-component health checks., Aggregated health payload. (+15 more)

### Community 22 - "Hitl Phase Extract"
Cohesion: 0.06
Nodes (42): agent_saul LangGraph, Celery beat, Client, Cognee, comply, doc_id Returned, extract, Extract Validate Embed Store (+34 more)

### Community 23 - "Tool Web Search"
Cohesion: 0.1
Nodes (29): BaseTool, CrawlUrlInput, CrawlUrlTool, LangChain tool for web crawling., Input schema for crawl URL tool., Tool for crawling a URL and extracting content., Agent tools for web search, crawling and document processing., get_all_tools() (+21 more)

### Community 24 - "Search Endpoint Bm25"
Cohesion: 0.07
Nodes (35): API Layer, Up to 500 App Clients, Async SQLAlchemy, BM25 Full-text Search, Celery, Chunk Embed Upsert Pipeline, Chunk Content Embedding BM25 Index Fields, chunks Table (+27 more)

### Community 25 - "Client Indexing Chat"
Cohesion: 0.13
Nodes (24): _get_sdk_client(), PageIndexBatchConfig, PageIndexChatConfig, PageIndexClient, PageIndexConfig, Configuration for indexing operations., Concurrency settings for batch indexing., Configuration for chat completion calls. (+16 more)

### Community 26 - "Llm End Langsmith"
Cohesion: 0.08
Nodes (12): AsyncCallbackHandler, BaseCallbackHandler, AsyncStreamingCallbackHandler, configure_langsmith(), LatencyCallbackHandler, LangSmith observability bootstrap and custom callbacks. Must be imported before, Bootstrap LangSmith tracing by setting env vars.     Call this at application st, Tracks per-run latency for structured logging. (+4 more)

### Community 27 - "Chat Database All"
Cohesion: 0.13
Nodes (16): AsyncAttrs, Database package with Base and all schemas., DeclarativeBase, ChatMessage, ChatSession, Chat messages schema for storing user-LLM conversations., Store chat messages between user and LLM., Store chat session metadata. (+8 more)

### Community 28 - "Tool Validation Registry"
Cohesion: 0.11
Nodes (12): build_validation_error_handler(), format_tool_validation_error(), make_structured_tool(), Tool base classes and registry.  All agent tools use structured Pydantic input/o, Return a schema-first validation error message the model can retry against., Create a stable validation error formatter bound to a specific schema., Decorator to register a @tool-decorated function in the global registry.      Us, Wrap an async function as a StructuredTool with typed input.     Automatically r (+4 more)

### Community 29 - "Query Cache Search"
Cohesion: 0.11
Nodes (22): 768-Dimensional Embedding, BM25 Search, Cache Hit, Cache Miss, Cached Result, Embed Query, FastAPI, FTS Ranked Chunks (+14 more)

### Community 30 - "Rag Retrieval Agentic"
Cohesion: 0.12
Nodes (20): Agentic RAG Router Pattern, Hybrid Retrieval, HyDE Rejected for Legal RAG V1, Knowledge Graph Retrieval, Long Context versus RAG, Multi-modal RAG, Reciprocal Rank Fusion, Agentic RAG (+12 more)

### Community 31 - "Write Decay Memory"
Cohesion: 0.14
Nodes (15): Celery decay nightly, Cognee episodic procedural memory, entities.decay_score and clauses.decay_score, events immutable audit log, events dual write, Graphiti Neo4j episodes and entity edges, memory_versions CRDT snapshots, persist_memory node (+7 more)

### Community 32 - "Risk Context Compliance"
Cohesion: 0.15
Nodes (14): app.state.neo4j_driver, build_agent_context(), compliance, COMPLIANCE: +CONTRACT depth=1, Graphiti search_for_risk_context, memory_pipeline.py, MemoryScope, MemoryScope enforcement (+6 more)

### Community 33 - "Envelope Apirouter Violation"
Cohesion: 0.33
Nodes (5): _is_envelope_violation(), Global API response envelope enforcement for routers., APIRouter that validates `response_model` uses `APIResponse[T]`., StrictEnvelopeAPIRouter, APIRouter

### Community 34 - "Build Open Deep"
Cohesion: 0.5
Nodes (3): build_open_deep_search_config(), Open Deep Search package exports., Build graph config with the lifespan-owned HTTPX client attached.

### Community 35 - "Toon Parser Baseoutputparser"
Cohesion: 0.5
Nodes (2): BaseOutputParser, ToonParser

### Community 36 - "Contract Clauses Textsearch"
Cohesion: 0.5
Nodes (1): Contract KB parent documents and pg_textsearch clauses.  Revision ID: 9f4a1b7c6d

### Community 37 - "A7d9b1c2e3f Add Search"
Cohesion: 0.5
Nodes (1): Add search documents and chunks schema  Revision ID: 8a7d9b1c2e3f Revises: 2bc77

### Community 38 - "C0c17c6eb1cc Initial Vectors"
Cohesion: 0.5
Nodes (1): Initial schema: document_vectors and chat tables  Revision ID: c0c17c6eb1cc Revi

### Community 39 - "Bc7726317f6 Rename Metadata"
Cohesion: 0.5
Nodes (1): rename_metadata_to_meta_data  Revision ID: 2bc7726317f6 Revises: c0c17c6eb1cc Cr

### Community 40 - "Railway Private Internal"
Cohesion: 0.5
Nodes (4): Private railway.internal Only Zone, Public Internet-Facing Zone, Railway Compute Plane, railway.internal Private Networking

### Community 41 - "Getattr Langchain Fastapi"
Cohesion: 0.67
Nodes (1): LangChain FastAPI Production - Main application package.

### Community 42 - "Run Seeders All"
Cohesion: 0.67
Nodes (2): Run all seeders in order., run_all_seeders()

### Community 43 - "Law Database Connection"
Cohesion: 0.67
Nodes (3): Database Connection Pool Sizing, Kingman's Law, Little's Law

### Community 44 - "Explain Auto Analyze"
Cohesion: 0.67
Nodes (3): auto_explain, EXPLAIN ANALYZE BUFFERS, pg_stat_statements

### Community 45 - "Llm Attack Vectors"
Cohesion: 0.67
Nodes (3): LLM Attack Vectors, ModernBERT Safety Discriminator, LLM Zero Trust Gap

### Community 46 - "Tool Postgres Executions"
Cohesion: 1.0
Nodes (3): Postgres tool_executions durable audit, Redis idempotency hot path, SHA-256(step_id + input + user_id) tool call key

### Community 47 - "Application Features Domain"
Cohesion: 1.0
Nodes (1): Application features/domain modules.

### Community 48 - "Knowledge Feature"
Cohesion: 1.0
Nodes (1): Knowledge base feature.

### Community 49 - "Web Scraping Feature"
Cohesion: 1.0
Nodes (1): Web scraping feature.

### Community 50 - "Rag Retrieval Augmented"
Cohesion: 1.0
Nodes (1): RAG (Retrieval-Augmented Generation) utilities.

### Community 51 - "Langextract Utilities"
Cohesion: 1.0
Nodes (1): LangExtract async utilities.

### Community 52 - "Agent Definitions Utilities"
Cohesion: 1.0
Nodes (1): Agent definitions and utilities.

### Community 53 - "Vector Store Utilities"
Cohesion: 1.0
Nodes (1): Vector store utilities.

### Community 54 - "Alembic Revision Modules"
Cohesion: 1.0
Nodes (1): Alembic revision modules.

### Community 56 - "Postgresql Conf Tuning"
Cohesion: 1.0
Nodes (2): postgresql.conf Tuning, work_mem Risk

### Community 57 - "Postgresql Index Design"
Cohesion: 1.0
Nodes (2): PostgreSQL Index Design, Linux perf PostgreSQL CPU Analysis

### Community 58 - "Agent Saul Production"
Cohesion: 1.0
Nodes (2): Agent Saul, Production Deployment Topology

### Community 59 - "Backstage Dev Portal"
Cohesion: 1.0
Nodes (2): Backstage, Dev Portal Catalog

### Community 60 - "Control Plane Jenkins"
Cohesion: 1.0
Nodes (2): CI/CD Control Plane, Jenkins

### Community 80 - "Ensure Score Between"
Cohesion: 1.0
Nodes (1): Ensure score is between 0 and 1.

### Community 81 - "Validate Embedding Not"
Cohesion: 1.0
Nodes (1): Validate embedding is not empty if provided.

### Community 82 - "Ensure Overlap Less"
Cohesion: 1.0
Nodes (1): Ensure overlap is less than chunk size.

### Community 86 - "Configuration Instance Runnableconfig"
Cohesion: 1.0
Nodes (1): Create a Configuration instance from a RunnableConfig.

### Community 87 - "Strip Leading Trailing"
Cohesion: 1.0
Nodes (1): Strip leading/trailing whitespace from optional fields.

### Community 88 - "Cross Field Validation"
Cohesion: 1.0
Nodes (1): Cross-field validation (e.g., role minimum length).

### Community 94 - "Build Deterministic Sha"
Cohesion: 1.0
Nodes (1): Build a deterministic SHA-256 key for a tool invocation.

### Community 95 - "Load Configuration Settings"
Cohesion: 1.0
Nodes (1): Load configuration from settings.

## Ambiguous Edges - Review These
- `norm` → `rel_map`  [AMBIGUOUS]
  docs/diagrams/agent_saul_full_architecture.svg · relation: calls
- `risk` → `review HITL`  [AMBIGUOUS]
  docs/diagrams/agent_saul_full_architecture.svg · relation: calls

## Knowledge Gaps
- **553 isolated node(s):** `LangChain FastAPI Production - Main application package.`, `Create and configure FastAPI application with proper middleware order.`, `Application features/domain modules.`, `FastAPI dependencies for agent_saul.  All infra clients are read from request.ap`, `Stub — replace with your project's JWT/session auth dependency.     The user_id` (+548 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Toon Parser Baseoutputparser`** (4 nodes): `toon_parser.py`, `BaseOutputParser`, `ToonParser`, `.parse()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Contract Clauses Textsearch`** (4 nodes): `9f4a1b7c6d2e_contract_kb_clauses_pg_textsearch.py`, `downgrade()`, `Contract KB parent documents and pg_textsearch clauses.  Revision ID: 9f4a1b7c6d`, `upgrade()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `A7d9b1c2e3f Add Search`** (4 nodes): `8a7d9b1c2e3f_add_search_documents_and_chunks.py`, `downgrade()`, `Add search documents and chunks schema  Revision ID: 8a7d9b1c2e3f Revises: 2bc77`, `upgrade()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C0c17c6eb1cc Initial Vectors`** (4 nodes): `c0c17c6eb1cc_initial_schema_document_vectors_and_.py`, `downgrade()`, `Initial schema: document_vectors and chat tables  Revision ID: c0c17c6eb1cc Revi`, `upgrade()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Bc7726317f6 Rename Metadata`** (4 nodes): `2bc7726317f6_rename_metadata_to_meta_data.py`, `downgrade()`, `rename_metadata_to_meta_data  Revision ID: 2bc7726317f6 Revises: c0c17c6eb1cc Cr`, `upgrade()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Getattr Langchain Fastapi`** (3 nodes): `__getattr__()`, `__init__.py`, `LangChain FastAPI Production - Main application package.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Run Seeders All`** (3 nodes): `run_seeders.py`, `Run all seeders in order.`, `run_all_seeders()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Application Features Domain`** (2 nodes): `__init__.py`, `Application features/domain modules.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Knowledge Feature`** (2 nodes): `__init__.py`, `Knowledge base feature.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Web Scraping Feature`** (2 nodes): `__init__.py`, `Web scraping feature.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Rag Retrieval Augmented`** (2 nodes): `__init__.py`, `RAG (Retrieval-Augmented Generation) utilities.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Langextract Utilities`** (2 nodes): `__init__.py`, `LangExtract async utilities.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Agent Definitions Utilities`** (2 nodes): `Agent definitions and utilities.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Vector Store Utilities`** (2 nodes): `__init__.py`, `Vector store utilities.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Alembic Revision Modules`** (2 nodes): `__init__.py`, `Alembic revision modules.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Postgresql Conf Tuning`** (2 nodes): `postgresql.conf Tuning`, `work_mem Risk`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Postgresql Index Design`** (2 nodes): `PostgreSQL Index Design`, `Linux perf PostgreSQL CPU Analysis`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Agent Saul Production`** (2 nodes): `Agent Saul`, `Production Deployment Topology`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Backstage Dev Portal`** (2 nodes): `Backstage`, `Dev Portal Catalog`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Control Plane Jenkins`** (2 nodes): `CI/CD Control Plane`, `Jenkins`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ensure Score Between`** (1 nodes): `Ensure score is between 0 and 1.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Validate Embedding Not`** (1 nodes): `Validate embedding is not empty if provided.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ensure Overlap Less`** (1 nodes): `Ensure overlap is less than chunk size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Configuration Instance Runnableconfig`** (1 nodes): `Create a Configuration instance from a RunnableConfig.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Strip Leading Trailing`** (1 nodes): `Strip leading/trailing whitespace from optional fields.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cross Field Validation`** (1 nodes): `Cross-field validation (e.g., role minimum length).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Build Deterministic Sha`** (1 nodes): `Build a deterministic SHA-256 key for a tool invocation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Load Configuration Settings`** (1 nodes): `Load configuration from settings.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `norm` and `rel_map`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **What is the exact relationship between `risk` and `review HITL`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **Why does `get_settings()` connect `Client Tavily Run` to `Token User Oauth`, `Graphiti Neo4j Episode`, `Crawler Crawl Url`, `Circuit Breaker Idempotency`, `Search Chunk Rag`, `Upload User Router`, `Saul Redis Agent`, `Extract Entity Args`, `Client Indexing Chat`, `Llm End Langsmith`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `_encode_file()` connect `Agent Build Chain` to `Redis Args Search`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Why does `Chunk` connect `Args Embedding Chunk` to `Graph State Search`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 170 inferred relationships involving `str` (e.g. with `create_session()` and `saul_ws_endpoint()`) actually correct?**
  _`str` has 170 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `DatabaseException` (e.g. with `UserProfile` and `UserUpdate`) actually correct?**
  _`DatabaseException` has 59 INFERRED edges - model-reasoned connections that need verification._