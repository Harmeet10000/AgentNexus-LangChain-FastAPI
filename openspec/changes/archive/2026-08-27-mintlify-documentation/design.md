# Mintlify Documentation — Design

## Overview

This change creates a professional documentation site for Agent Saul using Mintlify. The site is generated primarily by AI agents that extract content from the existing codebase and internal docs, then format it into Mintlify-compatible `.mdx` files.

The site lives in a new `docs-site/` directory at the repo root, entirely separate from the existing `docs/` (which contains internal planning docs). The existing `docs/` is left untouched.

## Source Material Inventory

| Source | Format | Pages Produced |
|---|---|---|
| FastAPI OpenAPI spec | Auto-generated `/openapi.json` at runtime | ~30–40 auto-generated API pages |
| `README.md` | Markdown | 2 pages (Overview, Project Structure) |
| `src/app/config/settings.py` | 400+ Pydantic settings fields | 1 page (Environment Variables reference, grouped by category) |
| `src/app/features/*/router.py`, `service.py` | Router + service code, docstrings, Pydantic schemas | 1 page per feature (Auth, Users, Documents, Search, Agent Saul, Ingestion, Crawler, Health) |
| `docs/Agent_thingies/*.md` | 8 internal docs | 3 condensed guide pages (Architecture, Memory, RAG) |
| `docs/diagrams/*.svg` | 5 architecture SVGs | Embedded via `<Frame>` in relevant pages |
| `docs/superpowers/` | 12 design docs | 0 — these are planning artifacts, not user-facing |
| `src/app/utils/exceptions.py` | Exception class hierarchy | Inlined into the Errors page |
| `src/app/shared/result/errors.py` | Result error types | Inlined into the Errors page |
| `src/app/shared/result/mappers.py` | Result→exception bridge | Inlined into the Errors page |
| Existing guide docs | `tests/performance/Saul_agent_Arch.md`, `SECURITY.md` | 2 pages (Agent Arch Deep Dive, Security) |
| `src/app/shared/outbox/` | Outbox implementation | 1 page (Outbox Pattern — architectural decision) |
| `src/app/shared/langgraph_layer/agent_saul/nodes.py` | Agent Saul graph nodes + WebSocket handler | 2 pages (WebSocket Guide, Agent Saul Deep Dive) |
| `src/mcp_core/` | MCP server + client implementation | 2 pages (MCP Architecture, MCP Development) |
| `src/database/` | DB schemas, models | 1 page (Data Model Reference) |
| `src/app/utils/response_type.py` | APIResponse[T] envelope | Inlined into the Errors page |
| `openspec/changes/archive/` | Change proposals + ADRs | 1 page (Changelog — distilled from git log + OpenSpec archive) |
| `pyproject.toml` | Tool config (ruff, ty, pytest) | Inlined into Development guides |
| `AGENTS.md`, `.opencode/instructions/` | Project rules | 0 — internal AI workspace config, not user-facing |

## Navigation Structure

```
Get Started
├── Overview
├── Quickstart
├── Installation
└── Project Structure

Core Concepts
├── Stateful Agents
├── Human-in-the-Loop
├── Memory Architecture
├── Context Discipline
├── Deterministic Execution
└── Glossary                          ← NEW — domain glossary

Architecture
├── System Design
├── LangGraph Orchestration
├── RAG Pipeline
├── Graph Memory (Neo4j + Graphiti)
├── Document Ingestion Pipeline
├── Outbox Pattern                     ← NEW — transactional outbox
├── Error and Result Pattern           ← NEW — returns.Result architecture
└── Security Architecture              ← NEW — auth, RBAC, rate limiting, defense in depth

Guides
├── Upload and Analyze a Contract
├── Hybrid Search
├── Legal Q&A
├── Agent Saul Workflow
├── WebSocket Guide                    ← NEW — WebSocket session lifecycle, reconnection, heartbeat
├── Web Crawl and Research
├── Human Review Workflow
├── Troubleshooting                    ← NEW — common errors, solutions
└── FAQ                                ← NEW — frequently asked questions

API Reference (auto-generated from OpenAPI)
├── Authentication
├── Rate Limiting
├── Error Handling
├── Interactive Playground             ← NEW — enabled in mint.json
└── [all endpoints]

Configuration
├── Environment Variables
├── LLM Models
├── Rate Limits
├── Caching
└── Observability

Deployment
├── Docker
├── GCP
├── Environment Setup
├── Monitoring
└── Migration Guide (v1 → v2)         ← NEW — v1 deprecation path

Development
├── Local Setup
├── Coding Standards
├── Testing
├── Database Migrations
├── Celery Tasks
├── MCP Integration
└── Contributing                       ← NEW — PR workflow, commit conventions, review process

Resources
├── Changelog                          ← NEW — release history
├── Security                           ← NEW — vulnerability reporting, auth architecture doc
└── Data Model Reference               ← NEW — Pydantic models, DB schemas, Neo4j graph model
```

### Navigation rationale for new pages

| New Page | Why it exists | Which agent owns it |
|---|---|---|
| Glossary | 15+ domain terms (RAG, Graphiti, Cognee, HITL, MCP, etc.) need centralized definitions; every other page should link here | Content UX Agent (Phase 2) |
| Outbox Pattern | Architectural decision with production implications; NOTIFY/LISTEN, dead-letter replay | Architecture Distiller |
| Error and Result Pattern | returns.Result is the project's most distinctive architectural pattern; developers must understand it before contributing | Architecture Distiller |
| Security Architecture | Auth flow, RBAC, JWT handling, OAuth, rate limiting, IP blocking, secret management span multiple features | Architecture Distiller |
| WebSocket Guide | Agent Saul's core UX runs over WebSocket; OpenAPI auto-generation does NOT cover WS endpoints | Feature Surveyor |
| Troubleshooting | Developer self-service; reduces support burden | Content UX Agent (Phase 2) |
| FAQ | Developer self-service; covers "Do I need Neo4j?", "Can I use this outside India?" etc. | Content UX Agent (Phase 2) |
| Interactive Playground | Mintlify feature allowing users to try API endpoints from the browser; needs explicit config | API Harvester |
| Migration Guide (v1 → v2) | v1 is deprecated with Sunset header (2027-06-15); users need a migration path | Content UX Agent (Phase 2) |
| Contributing | Lowers barrier for external contributors | Content UX Agent (Phase 2) |
| Changelog | Professional docs always have a changelog; distilled from git log + OpenSpec archive | Content UX Agent (Phase 2) |
| Security | Expands SECURITY.md into a full architecture guide with auth flows and defense-in-depth | Content UX Agent (Phase 2) |
| Data Model Reference | Pydantic models, SQLAlchemy tables, MongoDB collections, Neo4j graph model — not covered by OpenAPI | Content UX Agent (Phase 2) |

## AI Agent Pipeline

The documentation is generated by parallel AI agents, not a single monolithic agent. Each agent has a narrow scope and produces a batch of `.mdx` files plus a `mint.json` navigation fragment.

### Phase 1 — Source Extraction (parallel)

```
                    ┌──────────────────────┐
                    │   Codebase Context    │
                    │  (codegraph explore)  │
                    └──────┬───────┬───────┘
                           │       │
            ┌──────────────┘       └──────────────┐
            │                                      │
            ▼                                      ▼
    ┌───────────────┐                     ┌───────────────┐
    │ API Harvester │                     │Config Indexer │
    │               │                     │               │
    │ Reads OpenAPI │                     │ Reads         │
    │ spec + router │                     │ settings.py   │
    │ docstrings    │                     │ Pydantic      │
    └───────┬───────┘                     │ fields        │
            │                             └───────┬───────┘
            │                                     │
            ▼                                     ▼
    ┌───────────────┐                     ┌───────────────┐
    │Architecture   │                     │Feature        │
    │Distiller      │                     │Surveyor       │
    │               │                     │               │
    │ Reads README  │                     │ Reads each    │
    │ + Agent_thing │                     │ features/*/   │
    │ + diagrams    │                     │ module        │
    │ + outbox/     │                     │ + WebSocket   │
    │ + result/     │                     │ handlers      │
    └───────┬───────┘                     └───────┬───────┘
            │                                     │
            └────────────────┬────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Phase 1 Output       │
                  │  .mdx batches +       │
                  │  nav fragments        │
                  └──────────────────────┘
```

### Phase 1.5 — Content UX Agent (parallel to Phase 1, runs after fragment merge)

After the 4 source-extraction agents finish, a Content UX Agent generates auxiliary pages that depend on knowing the full nav structure:

- Glossary
- FAQ
- Troubleshooting
- Changelog
- 404 page
- Migration Guide (v1 → v2)
- Contributing Guide
- Security Guide (expanded from SECURITY.md)
- WebSocket Guide (if not covered by Feature Surveyor)

These pages require awareness of the complete page inventory to avoid duplicating content and to cross-reference correctly.

### Phase 2 — Composition (sequential)

1. **mint.json composer** — Merges navigation fragments from Phase 1 + Phase 1.5, validates against `https://mintlify.com/docs.json` schema, adds topbar/footer/integrations/seo/blocks
2. **Review agent** — Runs 5 checks: frontmatter completeness, broken links, orphan pages, tone consistency, cross-reference validity

### Phase 3 — Polish

- Branding colors extracted from existing project
- Light/dark code themes configured
- Logo SVG created (or text-based fallback)
- Favicon created from logo
- Analytics integration (PostHog)
- SEO meta tags
- Feedback widget (thumbs rating)
- 404 page configured
- Redirects configured for any renamed pages
- Interactive API Playground enabled in mint.json

### Agent Focus and Consistency Guidelines

Every agent receives these shared guidelines before generating:

**Tone:**
- Match the README's voice: philosophical but precise, confident but not arrogant, human-first
- Lead with the answer (GEO optimization for LLM consumption)
- Use direct, active language. "The agent does X" not "X is done by the agent"
- Keep paragraphs short (3–5 sentences max)
- Use code examples liberally — every concept needs a concrete example

**Cross-referencing:**
- Any mention of an endpoint MUST link to the API reference page for that endpoint
- Any mention of a concept in the Glossary (stateful agent, deterministic replay, HITL, RAG, Graphiti, Cognee, MCP, etc.) MUST link to the glossary entry
- Any mention of an error type MUST link to the Errors page
- Any mention of a config setting MUST link to the Environment Variables page
- Use Mintlify path references: `/concepts/stateful-agents` not `concepts/stateful-agents.mdx`

**Page structure per type:**

| Page Type | Required Structure |
|---|---|
| Concept | What → Why → How → Example → Related |
| Guide | Goal → Prerequisites → Steps (Mintlify `<Steps>`) → Result → Next steps |
| Reference | Overview → Table/Schema → Details → Examples → Related |
| Tutorial | Problem → Solution → Walkthrough → Verification → Cleanup |
| Troubleshooting | Symptom → Cause → Solution → Prevention |
| FAQ | Question (title) → Direct answer → Details → Related links |

**Depth:**
- Overview/landing pages: 2–3 paragraphs, bullet links to child pages
- Concept pages: 3–6 paragraphs, 1–2 code/diagram examples
- Guide pages: full walkthrough with 5–15 steps, code blocks at every step
- Reference pages: exhaustive tables, minimal prose
- Glossary: one paragraph per term, 1–2 sentences per term minimum

**Terminology standardization:**
- "Agent Saul" (capitalized, not "agent Saul" or "Agent saul" or "saul")
- "human-in-the-loop" (hyphenated when adjective, "human in the loop" when noun)
- "LangGraph" not "Langgraph" or "langgraph"
- "LangChain" not "Langchain" or "langchain"
- "Graphiti" not "graphiti"
- "Cognee" not "cognee"
- "Gemini" not "gemini"
- Use the Glossary as the single source of truth for term definitions — never define the same term differently on two pages

## Mintlify Features Used

| Feature | Where Used |
|---|---|
| OpenAPI auto-reference | `API Reference` nav group — reads `/openapi.json` |
| `llms.txt` auto-generation | Free — Mintlify generates it; no config needed |
| `skills.md` auto-generation | Free — Mintlify generates it |
| SEO metatags | `mint.json` — `og:site_name`, `og:title`, `og:description`, canonical URL |
| Code groups (`<CodeGroup>`) | Every endpoint page — `curl` / `python` / `python SDK` tabs |
| Tabs (`<Tabs>`) | Multi-language SDK examples, comparison pages |
| Accordion (`<Accordion>`) | FAQ, troubleshooting, deep-dive details |
| Frame (`<Frame>`) | Architecture diagrams with captions |
| Cards (`<Card>`) | Overview landing pages linking to key sections |
| FileTree (`<FileTree>`) | Project structure page |
| Steps | Quickstart, guide pages |
| Insert analytics | PostHog integration |
| Search | Built-in Mintlify search |
| Feedback | Thumbs rating |

## Page Template

Every `.mdx` page follows this structure:

```yaml
---
title: "Page Title — Keyword Rich"
description: "One-line description for SEO and nav preview"
sidebarTitle: "Short Name"  # optional, for sidebar
icon: "icon-name"            # optional, from Lucide icons
---

Content goes here, using Mintlify components where appropriate.

Code blocks use CodeGroup if multiple languages are involved.
Diagrams use Frame with caption.
```

## Mintlify Config Structure

```json
{
  "$schema": "https://mintlify.com/docs.json",
  "theme": "mint",
  "name": "Agent Saul",
  "colors": { "primary": "#1a56db", "light": "#3b82f6", "dark": "#1d4ed8" },
  "logo": { "light": "images/logo-light.svg", "dark": "images/logo-dark.svg" },
  "favicon": "images/favicon.svg",
  "topbarLinks": [{ "name": "API Reference", "url": "/api-reference" }],
  "topbarCtaButton": {
    "name": "GitHub",
    "url": "https://github.com/Harmeet10000/langchain-fastapi-production"
  },
  "navigation": [
    { "group": "Get Started", "pages": ["overview", "quickstart", "installation", "project-structure"] },
    {
      "group": "Core Concepts",
      "pages": [
        "concepts/stateful-agents",
        "concepts/human-in-the-loop",
        "concepts/memory-architecture",
        "concepts/context-discipline",
        "concepts/deterministic-execution",
        "concepts/glossary"
      ]
    },
    {
      "group": "Architecture",
      "pages": [
        "architecture/overview",
        "architecture/system-design",
        "architecture/langgraph-orchestration",
        "architecture/rag-pipeline",
        "architecture/graph-memory",
        "architecture/document-ingestion",
        "architecture/outbox-pattern",
        "architecture/error-and-result-pattern",
        "architecture/security-architecture"
      ]
    },
    {
      "group": "Guides",
      "pages": [
        "guides/upload-analyze-contract",
        "guides/hybrid-search",
        "guides/legal-qa",
        "guides/agent-saul-workflow",
        "guides/websocket-guide",
        "guides/web-crawl-research",
        "guides/human-review",
        "guides/troubleshooting",
        "guides/faq"
      ]
    },
    {
      "group": "API Reference",
      "openapi": "openapi.json",
      "pages": [
        "api-reference/overview",
        "api-reference/authentication",
        "api-reference/rate-limiting",
        "api-reference/errors"
      ]
    },
    {
      "group": "Configuration",
      "pages": [
        "configuration/environment-variables",
        "configuration/llm-models",
        "configuration/rate-limits",
        "configuration/caching",
        "configuration/observability"
      ]
    },
    {
      "group": "Deployment",
      "pages": [
        "deployment/docker",
        "deployment/gcp",
        "deployment/environment",
        "deployment/monitoring",
        "deployment/migration-guide-v1-to-v2"
      ]
    },
    {
      "group": "Development",
      "pages": [
        "development/setup",
        "development/coding-standards",
        "development/testing",
        "development/migrations",
        "development/celery-tasks",
        "development/mcp-integration",
        "development/contributing"
      ]
    },
    {
      "group": "Resources",
      "pages": [
        "resources/changelog",
        "resources/security",
        "resources/data-model-reference"
      ]
    }
  ],
  "redirects": [
    {
      "source": "/security",
      "destination": "/resources/security"
    }
  ],
  "footer": { "socials": { "github": "https://github.com/Harmeet10000/langchain-fastapi-production" } },
  "seo": { "indexing": "navigable", "metatags": { ... } },
  "integrations": { "analytics": { "provider": "posthog" } },
  "feedback": { "thumbsRating": true }
}
```

### Key config decisions

| Field | Value | Why |
|---|---|---|
| `favicon` | `images/favicon.svg` | Professional branding; generated from logo |
| `redirects` | `security → resources/security` | Existing SECURITY.md users expect a `/security` path |
| API playground | Enabled by default in Mintlify | Users can try endpoints from browser |
| Search | Built-in | No config needed |
| `llms.txt` | Auto-generated | AI discoverability — no config needed |
| `skills.md` | Auto-generated | Agent capability discovery — no config needed |

## Color Palette

| Token | Value | Usage |
|---|---|---|
| Primary | `#1a56db` (blue-700) | Headers, links, buttons, active nav |
| Primary light | `#3b82f6` (blue-500) | Hover states, light-mode accent |
| Primary dark | `#1d4ed8` (blue-800) | Dark-mode accent |

These are derived from the existing blue accent in the project's FastAPI Swagger UI and the general legal/professional tone.

## File Structure

```
docs-site/
├── mint.json                      # Mintlify configuration
├── openapi.json                   # Snapshot of FastAPI OpenAPI spec
├── 404.mdx                        # Custom 404 page
├── overview.mdx
├── quickstart.mdx
├── installation.mdx
├── project-structure.mdx
├── concepts/
│   ├── stateful-agents.mdx
│   ├── human-in-the-loop.mdx
│   ├── memory-architecture.mdx
│   ├── context-discipline.mdx
│   ├── deterministic-execution.mdx
│   └── glossary.mdx               ← NEW — domain glossary
├── architecture/
│   ├── overview.mdx
│   ├── system-design.mdx
│   ├── langgraph-orchestration.mdx
│   ├── rag-pipeline.mdx
│   ├── graph-memory.mdx
│   ├── document-ingestion.mdx
│   ├── outbox-pattern.mdx         ← NEW — transactional outbox
│   ├── error-and-result-pattern.mdx ← NEW — returns.Result pattern
│   └── security-architecture.mdx  ← NEW — auth, RBAC, defense in depth
├── guides/
│   ├── upload-analyze-contract.mdx
│   ├── hybrid-search.mdx
│   ├── legal-qa.mdx
│   ├── agent-saul-workflow.mdx
│   ├── websocket-guide.mdx        ← NEW — WebSocket session lifecycle
│   ├── web-crawl-research.mdx
│   ├── human-review.mdx
│   ├── troubleshooting.mdx        ← NEW — common errors + solutions
│   └── faq.mdx                    ← NEW — frequently asked questions
├── api-reference/
│   ├── overview.mdx
│   ├── authentication.mdx
│   ├── rate-limiting.mdx
│   └── errors.mdx
├── configuration/
│   ├── environment-variables.mdx
│   ├── llm-models.mdx
│   ├── rate-limits.mdx
│   ├── caching.mdx
│   └── observability.mdx
├── deployment/
│   ├── docker.mdx
│   ├── gcp.mdx
│   ├── environment.mdx
│   ├── monitoring.mdx
│   └── migration-guide-v1-to-v2.mdx ← NEW — v1 deprecation path
├── development/
│   ├── setup.mdx
│   ├── coding-standards.mdx
│   ├── testing.mdx
│   ├── migrations.mdx
│   ├── celery-tasks.mdx
│   ├── mcp-integration.mdx
│   └── contributing.mdx           ← NEW — PR workflow, conventions
├── resources/
│   ├── changelog.mdx              ← NEW — release history
│   ├── security.mdx               ← NEW — expanded security guide
│   └── data-model-reference.mdx   ← NEW — Pydantic models, DB schemas, Neo4j graph
├── scripts/                       ← NEW — extraction + validation scripts
│   ├── extract_openapi.py
│   ├── validate_frontmatter.py
│   └── check_nav_completeness.py
└── images/
    ├── favicon.svg                ← NEW
    ├── logo-light.svg
    ├── logo-dark.svg
    ├── agent_saul_full_architecture.svg
    ├── agent_saul_deployment_topology.svg
    ├── hybrid_query_pipeline.svg
    ├── memory_stack_architecture.svg
    └── search_system_architecture_overview.svg
```

## Verification & Quality

| Check | Tool | What It Verifies |
|---|---|---|
| Link integrity | `broken-link-checker` or `hyperlink` | No dead internal or external links |
| Frontmatter completeness | `scripts/validate_frontmatter.py` | Every `.mdx` has `title`, `description` |
| Nav completeness | `scripts/check_nav_completeness.py` | Every file referenced in `mint.json` navigation exists; every `.mdx` is referenced (no orphan pages) |
| Tone consistency | Manual review | All pages follow the voice guidelines (philosophical but precise, human-first) |
| Cross-reference validity | Manual check + script | Every `/concepts/glossary` link resolves; every API endpoint link resolves |
| OpenAPI integration | Manual check | API reference renders all endpoints, schemas link correctly |
| Mintlify dev build | `mintlify dev` | Local preview renders without errors |
| Markdown linting | `markdownlint` | Consistent heading levels, no broken syntax |
| Diagram rendering | Manual check | All SVGs display correctly in `<Frame>` components |
| Mintlify config validation | `https://mintlify.com/docs.json` schema | `mint.json` is valid against the Mintlify schema |
| Glossary coverage | Manual check | Every term used across pages has a glossary entry |
| Mobile responsiveness | Manual check | Sidebar collapses, text reflows, code blocks scroll horizontally |
| AI discoverability | Manual check | `/llms.txt` and `/skills.md` are accessible and populated |
