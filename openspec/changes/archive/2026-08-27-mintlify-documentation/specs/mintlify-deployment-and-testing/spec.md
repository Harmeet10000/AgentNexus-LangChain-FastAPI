## Overview

This spec defines how the Mintlify documentation site is deployed, verified, and kept in sync with the codebase. The site is a static Mintlify project that can be hosted via Mintlify's built-in hosting, GitHub Pages, or any static host.

## Requirements

### Requirement: Deployment pipeline exists
The docs site SHALL be deployable via a single command. Two hosting options SHALL be supported: Mintlify Cloud (connected to GitHub repo) or GitHub Pages (via a GitHub Actions workflow).

#### Scenario: Mintlify Cloud deployment
- **GIVEN** the `docs-site/` directory pushed to GitHub
- **WHEN** Mintlify Cloud is configured to watch the repo
- **THEN** changes pushed to `main` SHALL auto-deploy
- **AND** preview deployments SHALL be available for PRs

#### Scenario: GitHub Pages deployment (alternative)
- **GIVEN** a GitHub Actions workflow in `.github/workflows/deploy-docs.yml`
- **WHEN** changes are pushed to `main`
- **THEN** the workflow SHALL build the Mintlify site and deploy to GitHub Pages
- **AND** the workflow SHALL accept `workflow_dispatch` for manual triggering

### Requirement: Local preview works
Developers SHALL be able to preview the docs site locally with hot reload.

#### Scenario: Mintlify dev server starts
- **GIVEN** Node.js is installed
- **WHEN** `npx mintlify dev` is run in `docs-site/`
- **THEN** a local server SHALL start at `http://localhost:3000`
- **AND** changes to `.mdx` files SHALL hot-reload

#### Scenario: Dev server handles OpenAPI spec
- **GIVEN** `docs-site/openapi.json` exists
- **WHEN** the dev server runs
- **THEN** the API Reference pages SHALL render correctly from the spec

### Requirement: Verification tooling is automated
The docs site SHALL have automated checks that run in CI.

#### Scenario: Link checker runs in CI
- **GIVEN** the docs site is built
- **WHEN** CI runs
- **THEN** a link checker (`broken-link-checker` or `hyperlink`) SHALL verify all internal and external links
- **AND** the check SHALL fail if any dead links are found

#### Scenario: Frontmatter validation runs in CI
- **GIVEN** all `.mdx` files
- **WHEN** CI runs
- **THEN** a script SHALL verify that every file has `title` and `description` in its frontmatter
- **AND** the check SHALL fail if any page is missing required frontmatter

#### Scenario: Navigation completeness check runs in CI
- **GIVEN** the `mint.json` navigation array and the set of `.mdx` files
- **WHEN** CI runs
- **THEN** a script SHALL verify that every `.mdx` file is referenced in `navigation`
- **AND** verify that every navigation path maps to an existing `.mdx` file
- **AND** the check SHALL fail if orphan pages or broken nav references exist

#### Scenario: Markdown linting runs in CI
- **GIVEN** `.mdx` files
- **WHEN** CI runs
- **THEN** `markdownlint` SHALL check for consistent heading levels, proper list formatting, and other markdown best practices
- **AND** the check SHALL NOT be blocking (warnings only)

### Requirement: OpenAPI spec is refreshed on release
The OpenAPI spec snapshot SHALL be updated when the API changes. A manual validation step SHALL be documented in the deployment guide.

#### Scenario: Refresh script is documented
- **GIVEN** the extraction script
- **WHEN** a developer needs to refresh the API reference
- **THEN** running `python scripts/extract_openapi.py` SHALL overwrite `docs-site/openapi.json`
- **AND** the developer SHALL manually verify that the API reference renders correctly in the dev server

### Requirement: .gitignore is configured
The `docs-site/` directory SHALL NOT check in Mintlify build artifacts.

#### Scenario: Build artifacts are ignored
- **GIVEN** the `.gitignore` at repo root
- **WHEN** inspected
- **THEN** it SHALL include `docs-site/.mintlify/` and `docs-site/node_modules/`
