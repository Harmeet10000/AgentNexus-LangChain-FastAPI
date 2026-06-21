# Graph Report - langchain-fastapi-production  (2026-06-21)

## Corpus Check
- 375 files · ~276,471 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 853 nodes · 1475 edges · 41 communities (39 shown, 2 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 158 edges (avg confidence: 0.52)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e4d019e0`
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
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]

## God Nodes (most connected - your core abstractions)
1. `Settings` - 30 edges
2. `UserRepository` - 28 edges
3. `RefreshTokenRepository` - 23 edges
4. `_build_graph_nodes()` - 22 edges
5. `WebCrawler` - 21 edges
6. `CrawlResult` - 20 edges
7. `SessionData` - 20 edges
8. `AuthService` - 19 edges
9. `DocumentQueryService` - 19 edges
10. `SearchService` - 19 edges

## Surprising Connections (you probably didn't know these)
- `WebCrawler` --uses--> `AsyncWebCrawler`  [INFERRED]
  src/app/features/crawler/service.py → src/app/connections/crawl4ai.py
- `Any` --uses--> `Settings`  [INFERRED]
  src/app/connections/celery_registry.py → src/app/config/settings.py
- `DocumentSearchResultItem` --uses--> `Settings`  [INFERRED]
  src/app/features/documents/service.py → src/app/config/settings.py
- `DocumentStatusResponse` --uses--> `Settings`  [INFERRED]
  src/app/features/documents/service.py → src/app/config/settings.py
- `DocumentUploadResponse` --uses--> `Settings`  [INFERRED]
  src/app/features/documents/service.py → src/app/config/settings.py

## Import Cycles
- 1-file cycle: `src/app/lifecycle/lifespan.py -> src/app/lifecycle/lifespan.py`
- 1-file cycle: `src/app/main.py -> src/app/main.py`
- 1-file cycle: `src/app/middleware/health_check.py -> src/app/middleware/health_check.py`

## Communities (41 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (52): BaseModel, AgentContext, AgentDependencies, ChatResponse, Chunk, ChunkResult, DoclingEnhancementConfig, DoclingExtractionResult (+44 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (47): HybridSearchRequest, LegalAskRequest, LegalAskResponse, RagSearchRequest, RagSearchResponse, get_current_user_id(), _get_search_llm(), get_search_repository() (+39 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (69): BaseSettings, get_settings(), Application settings loaded from environment variables., Returns a cached instance of the application settings., Settings, ContextGrade, DocumentCommandService, DocumentQueryService (+61 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (19): Redis-serializable session record. Frozen for safe pipeline use., Redis-primary, MongoDB-audit session store., RefreshTokenRepository, SessionData, UserRepository, AuthService, Return (authorization_url, signed_state_for_cookie)., _to_user_response() (+11 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (61): AgentRegistry, build_agent_registry(), _build_graph_nodes(), Startup-time Agent Saul graph composition helpers., Instantiate all agents + LLM chains once.     Called from build_saul_graph — nev, Holds all pre-built agents and structured-output LLM chains.     Created once at, SaulGraphNodes, _build_analysis_context() (+53 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (20): DocumentIngestionPipeline, main(), Main ingestion script for processing markdown documents into vector DB and knowl, Ingest a single document.          Args:             file_path: Path to the docu, Find all supported document files in the documents folder., Pipeline for ingesting documents into vector DB and knowledge graph., Read document content from file - supports multiple formats via Docling., Transcribe audio file using Whisper ASR via Docling. (+12 more)

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
Nodes (22): Any, CeleryTaskPayload, CeleryTaskRegistry, LegacyTaskPayload, Typed Celery task registry.  Provides a ``CeleryTaskRegistry`` that maps task na, Base for all typed Celery task payloads.      Subclass this for each task to def, Fallback payload for untyped tasks during incremental migration.      Accepts an, Maps task names → Pydantic payload models for validation. (+14 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (39): AsyncDriver, AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncWebCrawler, Celery, close_crawl4ai_crawler(), create_crawl4ai_crawler(), get_crawl4ai_crawler() (+31 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (33): CrawlResult, Core crawler module using Crawl4AI., Crawl a single URL.          Args:             url: URL to crawl             use, Recursively crawl internal links from starting URLs.          Args:, Result from crawling a URL., Web crawler using Crawl4AI with caching., Generate cache key for URL., Get cached crawl result. (+25 more)

### Community 13 - "Community 13"
Cohesion: 0.08
Nodes (48): AppError, AsyncSession, ClauseSegment, ContractMetadata, Failure, _build_preamble(), _cached_embedding(), _call_embedding_fn() (+40 more)

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (18): CrossEncoderReranker, _cached_embedding(), _call_embedding_fn(), make_context_grader_node(), make_generator_node(), make_graph_retrieval_node(), make_hybrid_retrieval_node(), make_query_analyzer_node() (+10 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (29): APIResponse, forgot_password(), get_me(), list_sessions(), login(), logout(), oauth_authorize(), oauth_callback() (+21 more)

### Community 16 - "Community 16"
Cohesion: 0.27
Nodes (10): CompiledStateGraph, build_document_ingestion_graph(), DocumentIngestionState, _make_ingest_document_node(), Document ingestion graph wrapper., Build the per-job ingestion graph., IngestDocumentFn, BaseChatModel (+2 more)

### Community 17 - "Community 17"
Cohesion: 0.22
Nodes (8): BaseTool, Any, CrawlUrlInput, CrawlUrlTool, LangChain tool for web crawling., Input schema for crawl URL tool., Tool for crawling a URL and extracting content., Run the crawl tool.          Args:             url: URL to crawl             ext

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (21): 5.1 Core Model: Cyclic State Machine, 5.2 Node Definitions, 5.3 Execution Pipeline (Legal Processing), 5.4 Parallelization Strategy (CRITICAL), 5.5 Evaluator Node (QA Gate), 5.6 HITL (Interrupt + Resume), 5. Phase 2: LangGraph Orchestration Pipeline, Behavior: (+13 more)

### Community 20 - "Community 20"
Cohesion: 0.15
Nodes (12): 11.1 Recovery Model, 11. Persistence Layer, 12. Observability, 13. Security Model, 14. Failure Modes & Recovery, 15. Final System Guarantees, 22.1 Schema, 22.2 Identifiers (+4 more)

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

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (3): 10.1 Rules, 10.2 Execution Model, 10. State & Execution Model

## Knowledge Gaps
- **171 isolated node(s):** `LogoutRequest`, `RefreshRequest`, `VerifyEmailRequest`, `ResendVerificationRequest`, `ForgotPasswordRequest` (+166 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CrawlResult` connect `Community 11` to `Community 0`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `SessionData` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `Any` connect `Community 9` to `Community 2`, `Community 5`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Are the 25 inferred relationships involving `Settings` (e.g. with `Any` and `ContextGrade`) actually correct?**
  _`Settings` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `UserRepository` (e.g. with `AuthService` and `LoginRequest`) actually correct?**
  _`UserRepository` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `RefreshTokenRepository` (e.g. with `AuthService` and `LoginRequest`) actually correct?**
  _`RefreshTokenRepository` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `LogoutRequest`, `RefreshRequest`, `VerifyEmailRequest` to the rest of the system?**
  _314 weakly-connected nodes found - possible documentation gaps or missing edges._