## Overview

This spec defines the `mint.json` configuration, visual branding, SEO setup, analytics integration, and layout components for the Agent Saul Mintlify site.

## Requirements

### Requirement: mint.json has complete configuration
The `mint.json` file SHALL include all required fields: `$schema`, `theme`, `name`, `colors`, `navigation`. It SHALL also include optional fields: `logo`, `topbarLinks`, `topbarCtaButton`, `footer`, `seo`, `integrations`, `feedback`.

#### Scenario: Schema reference is present
- **GIVEN** `mint.json`
- **WHEN** validated
- **THEN** it SHALL have `"$schema": "https://mintlify.com/docs.json"`

#### Scenario: Navigation covers all groups
- **GIVEN** the navigation array
- **WHEN** inspected
- **THEN** it SHALL have 8 groups: Get Started, Core Concepts, Architecture, Guides, API Reference, Configuration, Deployment, Development
- **AND** each group SHALL have a `pages` array with paths to the relevant `.mdx` files

#### Scenario: OpenAPI spec is configured
- **GIVEN** the API Reference group
- **WHEN** configured
- **THEN** it SHALL have `"openapi": "openapi.json"` at the group level

### Requirement: Branding is professional and consistent
The site SHALL use the color palette defined in the design doc. Logo images SHALL be provided for both light and dark modes.

#### Scenario: Colors match the palette
- **GIVEN** `mint.json` colors
- **WHEN** inspected
- **THEN** `primary` SHALL be `#1a56db`, `light` SHALL be `#3b82f6`, `dark` SHALL be `#1d4ed8`

#### Scenario: Logo SVGs exist
- **GIVEN** the `images/` directory
- **WHEN** inspected
- **THEN** it SHALL contain `logo-light.svg` and `logo-dark.svg`

### Requirement: SEO is configured
The site SHALL have SEO metatags and indexing policy configured for maximum discoverability.

#### Scenario: Default metatags are set
- **GIVEN** the `seo` block in `mint.json`
- **WHEN** applied
- **THEN** `og:site_name` SHALL be "Agent Saul"
- **AND** `og:description` SHALL summarize the platform's purpose
- **AND** a canonical URL SHALL be set

#### Scenario: Indexing policy is explicit
- **GIVEN** the `seo.indexing` field
- **WHEN** deployed
- **THEN** it SHALL be `"navigable"` (only navigable pages are indexed)

### Requirement: Analytics and feedback are integrated
The site SHALL include analytics (PostHog) and a feedback widget.

#### Scenario: PostHog analytics is configured
- **GIVEN** the `integrations` block
- **WHEN** the site loads
- **THEN** it SHALL send page views to PostHog (API key set as Mintlify project setting, not in `mint.json`)

#### Scenario: Thumbs feedback is enabled
- **GIVEN** the `feedback` block
- **WHEN** any page renders
- **THEN** a thumbs up/down widget SHALL appear at the bottom of the page content

### Requirement: Layout elements are polished
The site SHALL have a topbar with links, a CTA button, and a footer with social links.

#### Scenario: Topbar has an API Reference link
- **GIVEN** the `topbarLinks` array
- **WHEN** rendered
- **THEN** a "API Reference" link SHALL appear in the topbar pointing to `/api-reference`

#### Scenario: Topbar has a GitHub CTA
- **GIVEN** the `topbarCtaButton` object
- **WHEN** rendered
- **THEN** a "GitHub" button SHALL appear linking to `https://github.com/Harmeet10000/langchain-fastapi-production`

#### Scenario: Footer has GitHub social link
- **GIVEN** the `footer.socials` object
- **WHEN** rendered
- **THEN** a GitHub icon SHALL appear in the footer
