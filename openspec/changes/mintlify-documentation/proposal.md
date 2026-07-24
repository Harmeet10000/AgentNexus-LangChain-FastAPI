## Why

Agent Saul has no professional documentation site. What exists is:

- A dense `README.md` that buries setup steps under philosophical essays
- 29 internal design docs in `docs/` written in planning-format, not user-facing format
- 5 architecture SVGs with no surrounding explanatory text
- No API reference, no quickstart, no guides, no search, no SEO

This means:
- New contributors spend 2+ hours understanding the project before they can contribute
- Users cannot self-serve — they must read source code or ask the maintainer
- The project has no discoverable web presence for its capabilities
- AI coding assistants (Cursor, Copilot) have no `llms.txt` to reference

A Mintlify documentation site solves all of these in one shot: beautiful, searchable, SEO-optimized, with auto-generated API reference from the FastAPI OpenAPI spec, built-in AI discoverability via `llms.txt`, and a structure that guides users from zero to productive in under 10 minutes.

## What Changes

- Create a `docs-site/` directory with a full Mintlify project
- Generate ~60 `.mdx` pages across 8 navigation groups using AI agents in a parallel pipeline
- Configure `mint.json` with branding, navigation, SEO, analytics, and integrations
- Wire the FastAPI OpenAPI spec into Mintlify's auto-generated API reference
- Add a deployment pipeline (GitHub Pages or Mintlify hosting)
- Add verification tooling (link checker, frontmatter validator, nav completeness checker)

## Capabilities

### New Capabilities
- `mintlify-mdx-generation` — AI agent pipeline that extracts source material from the codebase and generates `.mdx` pages with correct frontmatter, Mintlify components, and navigation placement
- `mintlify-configuration-and-styling` — `mint.json` with branding, navigation groups, SEO metatags, analytics integration, and feedback widgets
- `mintlify-api-reference` — OpenAPI spec extraction from the running FastAPI app, configured for Mintlify's auto-rendered API reference pages
- `mintlify-deployment-and-testing` — Deployment pipeline, link checker, frontmatter validator, nav completeness checker

### Modified Capabilities
- None.

## Impact

- New `docs-site/` directory at repo root (separate from existing `docs/`)
- No modification to existing source code
- AI agent pipeline is one-shot generation, not a permanent system
- SVG diagrams in `docs/diagrams/` are copied/referenced, not moved
