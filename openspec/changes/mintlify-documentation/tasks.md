## 1. Scaffold and Configuration

- [x] 1.1 Create `docs-site/` directory with full directory structure (all navigation groups as subdirectories, `images/`, `scripts/`)
- [x] 1.2 Write `mint.json` with complete 9-group navigation (Get Started, Core Concepts, Architecture, Guides, API Reference, Configuration, Deployment, Development, Resources), colors, favicon, logo, topbar, footer, SEO, integrations, feedback
- [x] 1.3 Create light and dark mode logo SVGs in `docs-site/images/`
- [x] 1.4 Create favicon SVG in `docs-site/images/`
- [x] 1.5 Write `404.mdx` with navigation help and search bar
- [x] 1.6 Configure redirects in `mint.json` (at minimum: `/security` → `/resources/security`)
- [x] 1.7 Write `.gitignore` entries for `docs-site/.mintlify/` and `docs-site/node_modules/`
      **Executed 2026-08-24 (first tranche).** Scaffold complete: 9-group mint.json (colors, logos, favicon,
      topbar, footer, SEO, redirects incl. /security -> /resources/security), 14 MDX pages + 404.mdx seeded,
      SVG logos/favicon. `docs-mint.json` deleted (content absorbed; its page slugs intentionally reorganised
      under group directories — Mintlify requires file paths to match navigation).
      1.9 (`npx mintlify dev`) NOT verified: requires an interactive browser session; left open.

- [x] 1.8 Delete `docs-mint.json` from repo root (moved into this change)
- [ ] 1.9 Verify `npx mintlify dev` starts and renders the empty site without errors

## 2. OpenAPI Spec Extraction

- [x] 2.1 Write `scripts/extract_openapi.py` — starts FastAPI app in subprocess, fetches `/openapi.json`, validates against OpenAPI 3.1 schema, saves to `docs-site/openapi.json`
- [x] 2.2 Run extraction script and commit the initial `openapi.json` snapshot
      **Executed 2026-08-24.** Script rewritten after measuring two environment facts: (a) the app serves its
      schema at `/swagger.json` under production hardening (main.py:62), not `/openapi.json`; (b) uvicorn-under-
      subprocess made readiness polling flaky and lifespan blocks on a graph host that does not resolve here.
      Final form builds the real app in-process and fetches through an ASGI transport with lifespan never run;
      snapshot committed at **74 paths**. Extraction surfaced and fixed SEVEN latent not-fully-defined defects
      across plans/subscriptions/payments/invoices/agent_saul DTOs and dependencies — OpenAPI generation walks
      every response model, which no test had ever done.
      2.4/2.5 (browser rendering + playground) need a human with a browser; left open.

- [x] 2.3 Configure `mint.json` API Reference navigation group with `"openapi": "openapi.json"`
- [ ] 2.4 Manually verify that Mintlify renders all 30+ endpoints grouped by tag
- [ ] 2.5 Verify interactive API playground works (users can test endpoints from browser)

## 3. AI Agent Generation — Phase 1 (parallel)

### 3.1 API Harvester Agent
**Input:** FastAPI OpenAPI spec, all `features/*/router.py` and `service.py` files, `src/app/utils/response_type.py`, `src/app/utils/exceptions.py`, `pyproject.toml`
**Focus:** API consumers. Emphasize auth flow first. Every page leads with the answer.
- [ ] 3.1.1 Read FastAPI OpenAPI spec and all feature router files
- [ ] 3.1.2 Generate `api-reference/overview.mdx` — API organization, base URL (`/api/v1`), versioning (v1 deprecated, sunset 2027-06-15), content type (`application/json`), `APIResponse[T]` envelope structure
- [ ] 3.1.3 Generate `api-reference/authentication.mdx` — JWT access + refresh tokens, cookie-based auth (`access_token` cookie), Authorization header alternative, OAuth2 providers (Google, GitHub), RBAC via JWT claims
- [ ] 3.1.4 Generate `api-reference/rate-limiting.mdx` — per-endpoint rate limit table, `Retry-After` header, 429 response format, client-side backoff strategy
- [ ] 3.1.5 Generate `api-reference/errors.mdx` — `APIResponse[T]` envelope (success, statusCode, error, request fields), `APIException` hierarchy table (all subclasses with status codes), example error JSON bodies
- [ ] 3.1.6 Each page SHALL include `<CodeGroup>` with `curl` and `python` tabs for key operations
- [ ] 3.1.7 Generate navigation fragment for API Reference group

### 3.2 Config Indexer Agent
**Input:** `src/app/config/settings.py`
**Focus:** DevOps engineers and self-hosters. Every setting includes env var name, type, default, plain-language description. Grouped by functional area.
- [ ] 3.2.1 Read `src/app/config/settings.py` — extract all 400+ fields grouped by functional category
- [ ] 3.2.2 Generate `configuration/environment-variables.mdx` — full table of settings by group (App, Databases, Auth/JWT, Gemini, Redis, Celery/RabbitMQ, Crawl4AI, OTel, Rate Limiting, S3/R2, File Upload, WebSocket, Email, Logging, API Versioning, MCP, FastAPI Guard)
- [ ] 3.2.3 Generate `configuration/llm-models.mdx` — Gemini model configuration (Flash, Pro, Vision, Embedding), model name env vars, temperature/max-token defaults
- [ ] 3.2.4 Generate `configuration/rate-limits.mdx` — rate limit configuration fields, per-endpoint default limits
- [ ] 3.2.5 Generate `configuration/caching.mdx` — Redis cache configuration, TTLs, idempotency key config
- [ ] 3.2.6 Generate `configuration/observability.mdx` — OTel config (tracing, metrics, logs), LangSmith config, logging level/format/rotation
- [ ] 3.2.7 Generate navigation fragment for Configuration group

### 3.3 Architecture Distiller Agent
**Input:** `README.md`, `docs/Agent_thingies/*.md` (all 8), `docs/diagrams/*.svg` (all 5), `src/app/shared/outbox/`, `src/app/shared/result/errors.py` + `mappers.py`, `src/app/utils/exceptions.py`, `src/app/shared/langgraph_layer/agent_saul/`, `src/app/middleware/`, `pyproject.toml`
**Additional docs:** `RESULT-PATTERN.md`, `EXCEPTION-RULES.md`, `ARCHITECTURE-RULES.md` from `.opencode/instructions/`
**Focus:** Technical decision-makers and new contributors. Preserve engineering insight, strip planning artifacts. Each architecture page references exactly one SVG via `<Frame caption="...">`.
- [ ] 3.3.1 Read all source material, extract key architectural decisions
- [ ] 3.3.2 Generate `overview.mdx` — project purpose (legal intelligence for Indian contracts), philosophy (AI-assisted, not AI-replacing), high-level capabilities, link to Quickstart
- [ ] 3.3.3 Generate `quickstart.mdx` — from zero to running in 5 minutes: clone → uv sync → env → `uv run uvicorn`. Include a "try it" section with one curl command. Preserve the philosophical voice from the README.
- [ ] 3.3.4 Generate `installation.mdx` — prerequisites (Python 3.12, uv, PostgreSQL, Neo4j, Redis, MongoDB, Gemini API key), clone, uv venv + sync, environment setup (`.env.development`), verify installation
- [ ] 3.3.5 Generate `project-structure.mdx` — directory tree using `<FileTree>`, explain each top-level directory, link to relevant architecture pages
- [ ] 3.3.6 Generate `architecture/overview.mdx` — system architecture overview with `agent_saul_full_architecture.svg` in `<Frame>`, three-layer model summary, stack diagram (FastAPI → LangGraph → Gemini → PostgreSQL → Neo4j → Redis)
- [ ] 3.3.7 Generate `architecture/system-design.mdx` — three-layer model deep-dive (memory shaping, runtime control, execution durability), with `agent_saul_deployment_topology.svg` in `<Frame>`
- [ ] 3.3.8 Generate `architecture/langgraph-orchestration.mdx` — graph design pattern (precompile outside graph, pass callables into nodes), orchestrator pattern (plan → workers → synthesize), node strategy, with `search_system_architecture_overview.svg` in `<Frame>`
- [ ] 3.3.9 Generate `architecture/rag-pipeline.mdx` — hybrid search architecture (vector + keyword), fusion scoring (RRF + cross-encoder), pgvector/pgvectorscale/pg_textsearch, with `hybrid_query_pipeline.svg` in `<Frame>`
- [ ] 3.3.10 Generate `architecture/graph-memory.mdx` — Neo4j + Graphiti integration, graph model (nodes: Party, Person, Contract, Clause, etc.; relationships: SIGNED_BY, OBLIGATED_TO, etc.), graph-backed retrieval for legal Q&A, with `memory_stack_architecture.svg` in `<Frame>`
- [ ] 3.3.11 Generate `architecture/document-ingestion.mdx` — ingestion pipeline (Docling → classify → extract → segment → contextualize → embed → store), LangGraph ingestion graph, document types supported
- [ ] 3.3.12 Generate `architecture/outbox-pattern.mdx` — the dual-write problem, outbox solution, `with_outbox()` helper, `outbox_events` table, PostgreSQL NOTIFY/LISTEN relay, dead-letter after 5 failures, `uv run replay-outbox` CLI
- [ ] 3.3.13 Generate `architecture/error-and-result-pattern.mdx` — `returns.Result` rationale, `AppError` hierarchy (all 5 subclasses), standard unwrapping pattern with `isinstance(result, Failure)`, `app_error_to_exception()` bridge, `APIException` hierarchy table
- [ ] 3.3.14 Generate `architecture/security-architecture.mdx` — JWT auth flow (access + refresh token), RBAC model with roles and permissions (zero-DB per-request checks), OAuth2 provider integration (Google, GitHub), rate limiting architecture, FastAPI Guard integration (IP blocking, pen-test detection), secret management (Pydantic SecretStr), CORS, security headers
- [ ] 3.3.15 Generate `concepts/stateful-agents.mdx` — resumability via checkpointing, deterministic replay ("if your system cannot deterministically replay a run, you do not control your agent"), LangGraph checkpointer, session-based execution
- [ ] 3.3.16 Generate `concepts/human-in-the-loop.mdx` — why HITL is mandatory for legal liability, what humans do (approve/reject risks, correct clauses, annotate reasoning), what gets stored (overrides, comments, reviewer role), audit trail
- [ ] 3.3.17 Generate `concepts/memory-architecture.mdx` — Cognee for long-term recall, Graphiti for knowledge graph, memory shaping (filters, trimming, bounded context), context budget discipline
- [ ] 3.3.18 Generate `concepts/context-discipline.mdx` — context window budget breakdown (system prompt, bootstrap files, memory, skills, conversation history, tool output, compaction summaries), rules for keeping context under control
- [ ] 3.3.19 Generate `concepts/deterministic-execution.mdx` — "Plan → deterministic execution → validated output → persisted state" mental model, NOT "LLM → decide → act → hope"
- [ ] 3.3.20 Copy SVGs from `docs/diagrams/` to `docs-site/images/`
- [ ] 3.3.21 Generate navigation fragments for Get Started, Core Concepts, and Architecture groups

### 3.4 Feature Surveyor Agent
**Input:** Each `features/*/` module (router, service, models, dependencies, Pydantic schemas), `src/app/features/agent_saul/` WebSocket handler code, `src/app/middleware/`
**Focus:** End users completing specific tasks. Each guide starts with a concrete goal and walks through every step with code examples.
- [ ] 3.4.1 Read each `features/*/` module + WebSocket handler code
- [ ] 3.4.2 Generate `guides/upload-analyze-contract.mdx` — upload via `POST /documents/upload`, check status via `GET /documents/{doc_id}/status`, review and approve via human gate, handle errors (invalid format, too large, unsupported type). Include `<CodeGroup>` with `curl` and Python examples.
- [ ] 3.4.3 Generate `guides/hybrid-search.mdx` — search interface (`POST /search`), query parameters, filters, fusion scoring explanation, result format, pagination. Include `<CodeGroup>` with example queries.
- [ ] 3.4.4 Generate `guides/legal-qa.mdx` — RAG query (`POST /search/rag` vs `POST /search/ask` vs `POST /legal/ask`), Graphiti-verified answers vs standard RAG, citation format, confidence scores
- [ ] 3.4.5 Generate `guides/agent-saul-workflow.mdx` — orchestrator pattern: main agent plans → delegates to workers → synthesizes, node pipeline (planner → ingestion → retrieval → reconciliation → synthesis → finalization), human gate before persistence
- [ ] 3.4.6 Generate `guides/web-crawl-research.mdx` — crawling URLs (`POST /crawler/crawl`), Tavily search (`GET /crawler/search`), Crawl4AI integration, rate limits
- [ ] 3.4.7 Generate `guides/human-review.mdx` — approval flow, risk review, clause correction, annotated reasoning, what gets stored (overrides, comments, reviewer role), audit trail for compliance
- [ ] 3.4.8 Generate navigation fragment for Guides group

## 4. Phase 1.5 — Content UX Agent (sequential, after fragment merge)

**Input:** Full page inventory from Phase 1 nav fragments, `git log --oneline -100`, `openspec/changes/archive/`, `SECURITY.md`, `src/database/schemas/`, `src/app/shared/langgraph_layer/ingestion_kb/state.py` (entity types), `src/app/shared/outbox/`
**Focus:** Self-service and orientation. These pages fill gaps that the source-extraction agents don't cover.

### 4.1 Glossary and Reference Pages
- [ ] 4.1.1 Scan all generated `.mdx` files for domain terms not yet defined in the glossary
- [ ] 4.1.2 Generate `concepts/glossary.mdx` — 30+ terms alphabetized with `<AccordionGroup>`, each with definition and "See also" links
- [ ] 4.1.3 Generate `resources/data-model-reference.mdx` — document User (MongoDB/Beanie), outbox_events/dead_letter_events (PostgreSQL/SQLAlchemy), Neo4j graph model (entity types + relationship types from `state.py`)
- [ ] 4.1.4 Generate `resources/changelog.mdx` — distilled from git log and OpenSpec archive: all archived changes with dates, summaries, and links

### 4.2 User Support Pages
- [ ] 4.2.1 Generate `guides/faq.mdx` — 12+ questions (model support, Neo4j requirement, India-only, vs generic RAG chatbot, production readiness, Cognee vs Graphiti, Pinecone migration, Docker requirement, cost, HITL mandate, adding document types, monitoring)
- [ ] 4.2.2 Generate `guides/troubleshooting.mdx` — infrastructure errors (Neo4j, Redis, Celery, PostgreSQL, MongoDB, OTel) + application errors (Gemini API key, JWT expiry, rate limit 429, WebSocket disconnect, upload failures, empty search, human review timeout)
- [ ] 4.2.3 Verify FAQ and troubleshooting pages cross-reference the relevant architecture and API reference pages

### 4.3 WebSocket Guide
- [ ] 4.3.1 Read WebSocket handler code and Agent Saul session management
- [ ] 4.3.2 Generate `guides/websocket-guide.mdx` — full protocol documentation: create session (`POST /agent-saul/sessions`), connect (`WS /agent-saul/ws/{thread_id}`), message format (client→server and server→client JSON), heartbeat/ping-pong, reconnection strategy, error codes, rate limiting
- [ ] 4.3.3 Include `<CodeGroup>` with `curl` (session creation), `python` (websockets library full example), and `python-sdk` tabs

### 4.4 Migration and Onboarding
- [ ] 4.4.1 Generate `deployment/migration-guide-v1-to-v2.mdx` — endpoint equivalence table, header changes (Deprecation/Sunset), base URL change, schema differences, compatibility timeline (sunset 2027-06-15)
- [ ] 4.4.2 Generate `development/contributing.mdx` — setup workflow, pre-commit hooks, test conventions, PR workflow, branch naming, CI checks, security reporting
- [ ] 4.4.3 Generate `resources/security.mdx` — expand SECURITY.md into full guide: auth flows (JWT, OAuth2, session management), RBAC model, rate limiting, FastAPI Guard, CORS, security headers, secret management (SecretStr), vulnerability reporting
- [ ] 4.4.4 Generate navigation fragments for Resources, Deployment, and Development groups

## 5. Phase 2 — Composition and Review

- [ ] 5.1 Run mint.json composer — merge all 6 navigation fragments, validate against `https://mintlify.com/docs.json` schema
- [ ] 5.2 Run review agent check 1: frontmatter completeness — every `.mdx` has `title` and `description`
- [ ] 5.3 Run review agent check 2: broken links — no dead internal or external links
- [ ] 5.4 Run review agent check 3: orphan pages — every `.mdx` is in nav, every nav path has a file
- [ ] 5.5 Run review agent check 4: tone consistency — all pages match the project voice (philosophical but precise, human-first)
- [ ] 5.6 Run review agent check 5: cross-reference validity — every `/concepts/glossary`, `/api-reference/`, `/architecture/` link resolves
- [ ] 5.7 Run review agent check 6: glossary coverage — every distinctive domain term used across pages has a glossary entry
- [ ] 5.8 Manually review generated content for accuracy, tone, and readability
- [ ] 5.9 Fix issues found during review

## 6. Verification Tooling

- [ ] 6.1 Write `scripts/validate_frontmatter.py` — iterates all `.mdx` files, checks for required frontmatter fields (`title`, `description`), reports missing/invalid
- [ ] 6.2 Write `scripts/check_nav_completeness.py` — parses `mint.json` navigation, verifies every page path maps to an existing `.mdx` file, verifies every `.mdx` file is referenced at least once in navigation
- [ ] 6.3 Add `.markdownlint.jsonc` config for consistent `.mdx` formatting (heading levels, list indentation, no bare URLs)
- [ ] 6.4 Create `.github/workflows/docs-ci.yml` — runs frontmatter validation, nav completeness check, broken link checker, markdownlint on PRs to main
- [ ] 6.5 Create `.github/workflows/deploy-docs.yml` — builds Mintlify site, deploys to GitHub Pages (alternative to Mintlify Cloud hosting)

## 7. Final Verification

- [ ] 7.1 Run `npx mintlify dev` and verify all pages render without errors
- [ ] 7.2 Click through all navigation links — verify no dead ends
- [ ] 7.3 API Reference: verify all 30+ endpoints render with correct schemas
- [ ] 7.4 API Reference: verify interactive playground works (try an endpoint)
- [ ] 7.5 API Reference: verify code examples show correctly in `<CodeGroup>` tabs
- [ ] 7.6 Verify all 5 diagrams render correctly in their pages with `<Frame>` captions
- [ ] 7.7 Verify glossary accordions work (expand/collapse)
- [ ] 7.8 Verify SEO metatags render (open any page, inspect `<meta>` tags for og:title, og:description, canonical)
- [ ] 7.9 Verify `llms.txt` accessible at `/llms.txt` and populated
- [ ] 7.10 Verify `skills.md` accessible at `/skills.md`
- [ ] 7.11 Verify search works (type domain queries, observe results)
- [ ] 7.12 Verify thumbs feedback widget appears on content pages
- [ ] 7.13 Verify 404 page renders for non-existent URL
- [ ] 7.14 Verify redirect works (`/security` → `/resources/security`)
- [ ] 7.15 Verify mobile responsiveness (narrow viewport, sidebar collapse, readable text, scrollable code blocks)
- [ ] 7.16 Verify favicon appears in browser tab

## 8. Documentation Follow-up

- [ ] 8.1 Add `docs-site/README.md` explaining how to add new pages (create `.mdx`, add nav entry, verify locally), update the OpenAPI spec (run extraction script, verify rendering), and deploy
- [ ] 8.2 Update root `README.md` with a link to the deployed docs site
- [ ] 8.3 Archive this OpenSpec change
