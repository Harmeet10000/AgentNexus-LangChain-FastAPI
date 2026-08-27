## Overview

This spec defines the AI agent pipeline that extracts documentation source material from the codebase and generates Mintlify-compatible `.mdx` files. The pipeline runs once to produce the initial docs site; it is not a permanent system.

## Agent Focus and Consistency Guidelines

Every agent MUST follow these shared guidelines. These apply to ALL generated pages regardless of agent role. The review agent in Phase 2 checks conformance.

### Tone and Voice
- Match the project README's voice: philosophical but precise, confident but not arrogant, human-first. The project has strong opinions about deterministic execution, human-in-the-loop being mandatory, and treating context as an operating budget. Preserve these opinions.
- Lead with the answer — first sentence of every page should tell the reader what they'll get (GEO optimization for LLM consumption). Bad: "Authentication is an important part...". Good: "Include your API key in the Authorization header of every request."
- Use direct active language: "The orchestrator routes to a sub-agent" not "Routing to a sub-agent is done by the orchestrator"
- One idea per paragraph. Max 5 sentences per paragraph.
- Code examples on every concept — never state a rule without showing it in code.

### Cross-Referencing Rules
Every agent MUST produce cross-reference links using Mintlify path format (no `.mdx` extension):

| Concept | Link Target |
|---|---|
| Any endpoint | `/api-reference#method-path` or the auto-generated Mintlify page |
| "stateful agent", "deterministic replay", "checkpointing" | `/concepts/stateful-agents` |
| "human-in-the-loop", "HITL" | `/concepts/human-in-the-loop` |
| "memory architecture", "Cognee", "memory shaping", "context budget" | `/concepts/memory-architecture` |
| "context window", "context discipline", "token budget" | `/concepts/context-discipline` |
| "deterministic execution", "plan execute validate persist" | `/concepts/deterministic-execution` |
| "RAG", "hybrid search", "fusion scoring", "vector search", "keyword search" | `/architecture/rag-pipeline` |
| "Graphiti", "graph memory", "Neo4j", "relationship traversal" | `/architecture/graph-memory` |
| "LangGraph", "orchestrator", "sub-agent", "graph node" | `/architecture/langgraph-orchestration` |
| "Docling", "document ingestion", "parsing pipeline" | `/architecture/document-ingestion` |
| "outbox", "NOTIFY/LISTEN", "dead letter", "transactional outbox" | `/architecture/outbox-pattern` |
| "Result", "AppResult", "Failure", "Success", "app_error_to_exception" | `/architecture/error-and-result-pattern` |
| "JWT", "RBAC", "OAuth", "rate limit", "API key", "permission" | `/architecture/security-architecture` |
| "WebSocket", "session", "thread_id", "heartbeat", "reconnect" | `/guides/websocket-guide` |
| Any error code or exception class | `/api-reference/errors` |
| Any config/env setting | `/configuration/environment-variables` |
| Any glossary term | `/concepts/glossary` |

### Page Structure by Type
| Page Type | Required Sections | Max Prose | Min Examples |
|---|---|---|---|
| Concept | What → Why → How → Example → Related | 6 paragraphs | 1 code block |
| Guide | Goal → Prerequisites → Steps (`<Steps>`) → Result → Next steps | 3 paragraphs intro | 1 code block per step |
| Reference | Overview → Table/Schema → Details → Examples → Related | 2 paragraphs | 2 code blocks |
| Tutorial | Problem → Solution → Walkthrough → Verification → Cleanup | 4 paragraphs | 5+ code blocks |
| Troubleshooting | Symptom → Cause → Solution → Prevention | 1 paragraph per entry | 1 code block per solution |
| FAQ | Question → Direct answer → Details → Related links | 2 paragraphs per entry | 0–1 code block |
| Glossary | Term → Definition → Context → Related terms | 2 sentences per term | 0 |

### Terminology Standardization
Agents MUST use these exact spellings and capitalizations:

| Correct | Incorrect |
|---|---|
| Agent Saul | agent Saul, Agent saul, saul |
| human-in-the-loop (adj), human in the loop (noun) | human in the loop (adj) |
| LangGraph | Langgraph, langgraph |
| LangChain | Langchain, langchain |
| Graphiti | graphiti |
| Cognee | cognee |
| Gemini | gemini |
| FastAPI | Fastapi, fastapi |
| OpenAPI | openapi, Open api |
| Mintlify | mintlify |
| Pydantic | pydantic |
| pgvector | pgVector, pg_vector |
| pgvectorscale | pgVectorScale, pg_vector_scale |
| pg_textsearch | pg-textsearch, pgTextSearch |
| Docling | docling |
| LangExtract | Langextract, lang_extract |
| Tavily | tavily |
| Crawl4AI | Crawl4ai, crawl4ai |
| GraphRAG | Graph Rag, graph-rag |
| HITL | hitl |
| MCP | mcp |
| RBAC | rbac |
| OTel | otel, OpenTelemetry (only full name on first mention) |
| SQLAlchemy | Sqlalchemy, sqlalchemy |
| Beanie | beanie |
| Celery | celery |
| Redis | redis |
| RabbitMQ | Rabbitmq, rabbitmq |
| Neo4j | neo4j |
| PostHog | posthog |
| LangSmith | Langsmith, langsmith |

## Requirements

### Agent Scope and Focus

Each agent has a specific scope. It MUST NOT generate pages outside its scope. If content from another agent's scope is needed for context, the agent SHALL link to that agent's pages rather than duplicating content.

| Agent | Generates These Groups | Pages Count (approx) |
|---|---|---|
| API Harvester | API Reference | 4 hand-written + auto-generated endpoints |
| Config Indexer | Configuration | 5 |
| Architecture Distiller | Get Started, Core Concepts, Architecture | 18 |
| Feature Surveyor | Guides (feature-based) | 5 |
| Content UX Agent (Phase 1.5) | Guides (auxiliary), Development (contributing), Resources | 8 |
| Deployment + Dev Agent | Deployment, Development (minus contributing) | 11 |

### Requirement: API Harvester agent extracts OpenAPI spec and router metadata
The agent SHALL start the FastAPI app in a non-blocking way (or run a script that hits `/openapi.json`) and capture the full OpenAPI spec. It SHALL also read every `features/*/router.py` file to extract route descriptions, parameter metadata, and response schemas.

**Focus:** API consumers — developers integrating with Agent Saul from their own applications. Emphasize authentication flow first (everything else requires it), then the most common operations (upload document, search, start agent session).

#### Scenario: OpenAPI spec is captured
- **GIVEN** the FastAPI app is running
- **WHEN** the agent fetches `/openapi.json`
- **THEN** the agent SHALL save the spec to `docs-site/openapi.json`

#### Scenario: Router docstrings are extracted
- **GIVEN** a router file (e.g., `features/auth/router.py`)
- **WHEN** the agent parses the file
- **THEN** it SHALL extract route paths, HTTP methods, summary descriptions, and any non-standard status codes or error responses

### Requirement: Config Indexer agent extracts settings documentation
The agent SHALL parse `src/app/config/settings.py` and group the 400+ Pydantic settings fields into documented categories (App, Databases, Auth, Gemini, Redis, Celery, Crawl4AI, OTel, Rate Limiting, etc.).

**Focus:** DevOps engineers and self-hosters — people who need to configure the system without reading source code. Every setting MUST include its environment variable name, Pydantic type, default value, and a plain-language description of what it controls and why you'd change it. Group settings by functional area (not by Python class hierarchy).

#### Scenario: Settings are grouped by category
- **GIVEN** the `Settings` model with nested groups
- **WHEN** the agent processes the fields
- **THEN** each group SHALL become a section in `configuration/environment-variables.mdx`

#### Scenario: Each setting includes type and default
- **GIVEN** a Pydantic field with type annotation and default value
- **WHEN** the agent documents it
- **THEN** the output SHALL include the field name, type, default value, and the original comment (if any)

### Requirement: Architecture Distiller agent condenses internal docs
The agent SHALL read `README.md`, all files in `docs/Agent_thingies/`, and the 5 SVGs in `docs/diagrams/`. It SHALL produce user-facing pages for Get Started, Core Concepts, and Architecture groups that preserve engineering depth without planning-context noise.

**Additional source material beyond the existing docs:**
- `src/app/shared/outbox/` — outbox helper, model, relay implementation for the Outbox Pattern page
- `src/app/shared/result/errors.py`, `mappers.py`, `RESULT-PATTERN.md`, `EXCEPTION-RULES.md` — for the Error and Result Pattern page
- `src/app/utils/exceptions.py` — APIException hierarchy
- `src/app/shared/langgraph_layer/agent_saul/` — graph structure for the orchestration page
- `pyproject.toml` (ruff, ty, pytest sections) — inline into coding standards guide

**Focus:** Technical decision-makers and new contributors who need to understand WHY the system is designed this way, not just WHAT it does. Preserve the engineering insight (deterministic replay, memory discipline, HITL necessity) while stripping planning-format artifacts (agenda items, TODO markers, personal notes, unresolved discussion threads).

Each architecture page SHALL reference exactly one SVG diagram using `<Frame caption="...">`. The captions SHALL explain what the diagram shows and point out the most important data flow or decision point.

#### Scenario: Key concepts are extracted
- **GIVEN** internal docs containing planning artifacts, design discussions, and decision records
- **WHEN** the agent generates pages for `Core Concepts` and `Architecture` groups
- **THEN** it SHALL distill the content to user-facing explanations, removing agenda items, TODO markers, and personal notes

#### Scenario: Diagrams are preserved
- **GIVEN** SVG diagram files
- **WHEN** the agent creates the architecture pages
- **THEN** it SHALL reference the SVGs via `<Frame caption="...">` components with explanatory text

### Requirement: Feature Surveyor agent documents each feature module
The agent SHALL read each directory under `features/` and produce one guide page per feature, covering its router, service layer, models, and key dependencies. The agent SHALL also read the WebSocket handler code (`features/agent_saul/` and any WebSocket middleware) to produce the WebSocket guide.

**Additional source material:**
- `src/app/features/agent_saul/service.py` — Agent Saul session management and WS handler
- `src/app/middleware/` — middleware code for request context, rate limiting, deprecation headers
- `src/app/features/agent_saul/websocket_*.py` or equivalent WebSocket handler files

**Focus:** End users who want to accomplish specific tasks with Agent Saul. Each guide SHALL start with a concrete goal ("Upload a contract and get a risk analysis") and walk through every step with code examples.

**WebSocket Guide specific requirements:**
- Document the session lifecycle: create session (`POST /agent-saul/sessions`) → connect (`WS /agent-saul/ws/{thread_id}`) → message exchange → heartbeat → disconnect
- Show the message format (client→server and server→client JSON schemas)
- Document reconnection strategy, error codes, rate limiting on WS connections
- Mention that OpenAPI does NOT auto-generate WebSocket docs — this page fills that gap

#### Scenario: Feature pages include endpoint tables
- **GIVEN** a feature like `auth`
- **WHEN** the agent generates the page
- **THEN** it SHALL include a table of all endpoints with method, path, description, authentication requirement, and rate limits

#### Scenario: Feature pages link to API reference
- **GIVEN** a feature page referencing endpoints
- **WHEN** the agent places cross-references
- **THEN** it SHALL link to the auto-generated API reference pages (e.g., `POST /auth/login` links to the Mintlify API page)

### Requirement: Generated pages follow Mintlify page template
Every generated `.mdx` file SHALL include YAML frontmatter with `title` and `description`. Mintlify components (`<Tabs>`, `<CodeGroup>`, `<Accordion>`, `<Frame>`, `<Card>`, `<Steps>`, `<FileTree>`) SHALL be used where structurally appropriate.

#### Scenario: Frontmatter is present on all pages
- **GIVEN** any `.mdx` file in the generated site
- **WHEN** the file is opened
- **THEN** it SHALL have `title` and `description` in its frontmatter

#### Scenario: Code examples use CodeGroup
- **GIVEN** an API endpoint page
- **WHEN** code examples are shown
- **THEN** they SHALL use `<CodeGroup>` with tabs for `curl`, `python`, and `python-sdk` variants

#### Scenario: Architecture pages embed SVGs
- **GIVEN** an architecture page
- **WHEN** a diagram is relevant
- **THEN** it SHALL use `<Frame caption="..."><img src="..." /></Frame>` with a descriptive caption

### Requirement: Navigation fragments are merged correctly
Each agent SHALL produce a navigation fragment (a JSON snippet defining its pages and their group). The composer agent SHALL merge all fragments into a single valid `mint.json`.

#### Scenario: No duplicate page paths
- **GIVEN** navigation fragments from multiple agents
- **WHEN** the composer merges them
- **THEN** duplicate page paths SHALL be detected and flagged

#### Scenario: All .mdx files have a nav entry
- **GIVEN** the final set of `.mdx` files
- **WHEN** the composer validates completeness
- **THEN** every file SHALL be referenced in at least one navigation group
