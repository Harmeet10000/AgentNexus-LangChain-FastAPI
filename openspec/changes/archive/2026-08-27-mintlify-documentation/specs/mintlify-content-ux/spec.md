## Overview

This spec defines the auxiliary content pages that support the core documentation: glossary, FAQ, troubleshooting, changelog, 404 page, migration guide, security guide, contributing guide, WebSocket guide, and data model reference. These pages depend on knowing the full navigation structure and page inventory, so they are generated in Phase 1.5 after the main agent fragments are merged.

## Requirements

### Requirement: Glossary provides centralized term definitions
A `concepts/glossary.mdx` page SHALL define every domain-specific term used across the docs site. Terms SHALL be alphabetized and use the `<AccordionGroup>` component for scannability.

#### Scenario: Glossary covers all domain terms
- **GIVEN** the full page inventory from Phase 1
- **WHEN** the Content UX Agent scans all generated `.mdx` files for distinctive domain terms
- **THEN** the glossary SHALL include entries for at minimum: Agent Saul, Celery, checkpointing, Cognee, context budget, Crawl4AI, dead letter, deterministic execution, deterministic replay, Docling, FastAPI, Gemini, Graphiti, GraphRAG, human-in-the-loop, idempotency, LangChain, LangExtract, LangGraph, LangSmith, MCP, memory architecture, memory shaping, Neo4j, NOTIFY/LISTEN, OpenAPI, OpenTelemetry (OTel), outbox pattern, pgvector, pgvectorscale, pg_textsearch, Pydantic, RAG, rate limiting, RBAC, resumability, SSE, stateful agent, Tavily, TigerData, transactional outbox
- **AND** each term SHALL have a 1–3 sentence definition plus a "See also" link to related terms

#### Scenario: Glossary uses AccordionGroup
- **GIVEN** the glossary page
- **WHEN** rendered
- **THEN** each term SHALL use `<Accordion title="Term">definition</Accordion>`
- **AND** all accordions SHALL be wrapped in `<AccordionGroup>`

### Requirement: FAQ answers the top developer questions
A `guides/faq.mdx` page SHALL answer the questions developers most commonly ask when first encountering the project.

#### Scenario: FAQ covers at least 12 questions
- **GIVEN** the `faq.mdx` page
- **WHEN** inspected
- **THEN** it SHALL include answers to at minimum:
  1. "What models do I need? Can I use OpenAI instead of Gemini?"
  2. "Do I need Neo4j/Graphiti? What if I skip it?"
  3. "Can I use this for contracts outside India?"
  4. "How is this different from a generic RAG chatbot?"
  5. "Is this production-ready?"
  6. "What's the difference between Cognee and Graphiti?"
  7. "How do I migrate from Pinecone to pgvector?"
  8. "Can I run this without Docker?"
  9. "How much does it cost to run?"
  10. "Why is human-in-the-loop mandatory?"
  11. "How do I add a new document type?"
  12. "What monitoring should I set up?"

### Requirement: Troubleshooting page covers common errors
A `guides/troubleshooting.mdx` page SHALL document the most common errors and their solutions.

#### Scenario: Troubleshooting covers infrastructure errors
- **GIVEN** the troubleshooting page
- **WHEN** inspected
- **THEN** it SHALL include entries for: Neo4j connection refused, Redis connection refused, Celery worker not processing tasks, PostgreSQL connection timeout, MongoDB authentication failure, OTel exporter not receiving spans

#### Scenario: Troubleshooting covers application errors
- **GIVEN** the troubleshooting page
- **WHEN** inspected
- **THEN** it SHALL include entries for: Gemini API key not set or invalid, JWT token expired, rate limit exceeded (429), WebSocket disconnection with error code, document upload fails validation, search returns no results, human review timeout

### Requirement: Changelog captures release history
A `resources/changelog.mdx` page SHALL document the project's release history, distilled from the git log and OpenSpec change archive.

#### Scenario: Changelog covers major changes
- **GIVEN** the `changelog.mdx` page
- **WHEN** inspected
- **THEN** it SHALL include entries for all archived OpenSpec changes (outbox, MCP split, result pattern standardization, settings fail-fast, OTel migration, test coverage, etc.)
- **AND** each entry SHALL include: date, change title, one-sentence summary, and link to the OpenSpec change archive (if available)

### Requirement: 404 page exists
A `404.mdx` page SHALL provide a helpful error page when users navigate to a non-existent URL.

#### Scenario: 404 page has navigation help
- **GIVEN** the `404.mdx` page
- **WHEN** a user hits a broken link
- **THEN** it SHALL display a message like "Page not found" with links to: Overview, Quickstart, API Reference, Search
- **AND** it SHALL include the Mintlify search bar

### Requirement: Migration Guide documents v1→v2 path
A `deployment/migration-guide-v1-to-v2.mdx` page SHALL document how to migrate from the deprecated v1 API to v2.

#### Scenario: Migration guide covers endpoint changes
- **GIVEN** the migration guide page
- **WHEN** a user reads it
- **THEN** it SHALL list every v1 endpoint and its v2 equivalent (or migration status)
- **AND** it SHALL document header changes (Deprecation, Sunset), base URL changes, and any schema differences

#### Scenario: Migration guide has a compatibility timeline
- **GIVEN** the migration guide page
- **WHEN** a user reads it
- **THEN** it SHALL include the sunset date (2027-06-15) and a timeline showing when v1 stops receiving updates

### Requirement: Security Guide documents auth architecture
A `resources/security.mdx` page SHALL expand the existing `SECURITY.md` into a full architecture reference covering authentication, authorization, and defense-in-depth.

#### Scenario: Security guide covers authentication flows
- **GIVEN** the security page
- **WHEN** inspected
- **THEN** it SHALL document: JWT access token format and claims, refresh token rotation, cookie-based auth vs Authorization header, OAuth2 flows (Google, GitHub), session management (list, revoke)

#### Scenario: Security guide covers authorization model
- **GIVEN** the security page
- **WHEN** inspected
- **THEN** it SHALL document: RBAC roles and their permissions, how JWT claims encode role info (zero DB hits), permission checks in middleware

#### Scenario: Security guide covers defense-in-depth
- **GIVEN** the security page
- **WHEN** inspected
- **THEN** it SHALL document: rate limiting per endpoint, IP blocking and pen-test detection (FastAPI Guard), CORS configuration, security headers, secret management (SecretStr pattern), HTTPS enforcement

### Requirement: Contributing guide documents PR workflow
A `development/contributing.mdx` page SHALL lower the barrier for external contributors by documenting the full development and PR workflow.

#### Scenario: Contributing guide covers setup
- **GIVEN** the contributing page
- **WHEN** inspected
- **THEN** it SHALL cover: cloning, uv venv setup, pre-commit hooks, running tests locally, coding standards (ruff, ty), commit message conventions

#### Scenario: Contributing guide covers PR workflow
- **GIVEN** the contributing page
- **WHEN** inspected
- **THEN** it SHALL cover: branch naming, PR template, what CI checks run, how to request review, how to report security issues (link to SECURITY.md)

### Requirement: WebSocket Guide documents real-time communication
A `guides/websocket-guide.mdx` page SHALL document the Agent Saul WebSocket protocol. This is critical because OpenAPI auto-generation does NOT cover WebSocket endpoints.

#### Scenario: WebSocket guide covers session lifecycle
- **GIVEN** the WebSocket guide page
- **WHEN** inspected
- **THEN** it SHALL document: creating a session via `POST /agent-saul/sessions`, connecting via `WS /agent-saul/ws/{thread_id}`, the JSON message format for client→server and server→client, session teardown

#### Scenario: WebSocket guide covers operational concerns
- **GIVEN** the WebSocket guide page
- **WHEN** inspected
- **THEN** it SHALL document: heartbeat/ping-pong mechanism, reconnection strategy with exponential backoff, rate limiting on WS connections, error codes during WS communication, origin checking

#### Scenario: WebSocket guide includes code examples
- **GIVEN** the WebSocket guide page
- **WHEN** inspected
- **THEN** it SHALL include a `<CodeGroup>` with `curl` (for session creation), `python` (websockets library full example), and `python-sdk` (if an SDK exists) tabs

### Requirement: Data Model Reference documents all schemas
A `resources/data-model-reference.mdx` page SHALL document the Pydantic models, SQLAlchemy tables, MongoDB collections, and Neo4j graph model used by the system.

#### Scenario: Data model covers the User model
- **GIVEN** the data model page
- **WHEN** inspected
- **THEN** it SHALL document the `User` Beanie/MongoDB document with all fields, types, and constraints

#### Scenario: Data model covers the outbox schema
- **GIVEN** the data model page
- **WHEN** inspected
- **THEN** it SHALL document the `outbox_events` and `dead_letter_events` SQLAlchemy tables with all columns, types, indexes, and constraints

#### Scenario: Data model covers the Neo4j graph model
- **GIVEN** the data model page
- **WHEN** inspected
- **THEN** it SHALL document the node types (Party, Person, Organization, Contract, Clause, Obligation) and relationship types (SIGNED_BY, OBLIGATED_TO, GOVERNED_BY, SUPERSEDES, REFERENCES_CLAUSE) used by Graphiti

### Requirement: Outbox Pattern page documents the transactional outbox
An `architecture/outbox-pattern.mdx` page SHALL document the transactional outbox architecture that replaced direct Celery task calls.

#### Scenario: Outbox page explains the problem
- **GIVEN** the outbox pattern page
- **WHEN** inspected
- **THEN** it SHALL explain the dual-write problem (DB write + Celery task) and why the outbox pattern solves it

#### Scenario: Outbox page documents the implementation
- **GIVEN** the outbox pattern page
- **WHEN** inspected
- **THEN** it SHALL document: the `with_outbox()` helper and how it wraps DB transactions, the `outbox_events` table schema, the NOTIFY/LISTEN relay mechanism, dead-letter handling after 5 failures, the replay CLI (`uv run replay-outbox`)

### Requirement: Error and Result Pattern page documents the error architecture
An `architecture/error-and-result-pattern.mdx` page SHALL document the `returns.Result` pattern and how it bridges to the `APIException` hierarchy.

#### Scenario: Result pattern page explains the pattern
- **GIVEN** the error and result pattern page
- **WHEN** inspected
- **THEN** it SHALL explain: why `returns.Result` is used in repositories (expected failures vs exceptions), the `AppError` hierarchy (ValidationAppError, NotFoundAppError, ConflictAppError, InfrastructureAppError, ExternalServiceAppError), and the standard unwrapping pattern (`isinstance(result, Failure)` + `raise app_error_to_exception(error)`)

#### Scenario: Result pattern page documents the exception hierarchy
- **GIVEN** the error and result pattern page
- **WHEN** inspected
- **THEN** it SHALL list all `APIException` subclasses with their HTTP status codes: ValidationException (422), NotFoundException (404), UnauthorizedException (401), ForbiddenException (403), ConflictException (409), TooManyRequestsException (429), ServiceUnavailableException (503), DatabaseException (500), ExternalServiceException (502), etc.
