## Overview

This spec defines how the FastAPI OpenAPI specification is extracted, formatted, and integrated into Mintlify's auto-generated API reference pages. This is the most technically critical part of the docs — if the OpenAPI integration works, 30–40 API endpoint pages are auto-generated with zero manual effort.

## Requirements

### Requirement: OpenAPI spec is extracted from the running app
The spec SHALL be extracted from the FastAPI application at runtime, not hand-written. A script SHALL start the application, fetch `/openapi.json`, and save the result to `docs-site/openapi.json`.

#### Scenario: Script extracts valid OpenAPI 3.1 spec
- **GIVEN** the FastAPI app factory (`src/app/main.py`)
- **WHEN** the extraction script runs
- **THEN** it SHALL produce `docs-site/openapi.json` that passes OpenAPI 3.1 schema validation

#### Scenario: Spec is a snapshot, not a live proxy
- **GIVEN** the extracted `openapi.json`
- **WHEN** the docs site serves it
- **THEN** it SHALL be a static file, not a proxy to the running app

### Requirement: Mintlify auto-generates API reference pages
The `mint.json` SHALL reference the OpenAPI spec in the API Reference navigation group. Mintlify SHALL auto-generate endpoint pages for all paths.

#### Scenario: All endpoints appear in the API reference
- **GIVEN** the OpenAPI spec with 30+ endpoints
- **WHEN** `mint.json` has `"openapi": "openapi.json"` in the API Reference group
- **THEN** Mintlify SHALL generate one page per endpoint path
- **AND** group them by tag (Auth, Users, Documents, Search, Agent Saul, etc.)

#### Scenario: Schemas are rendered correctly
- **GIVEN** Pydantic request/response models in the OpenAPI spec
- **WHEN** Mintlify renders the API pages
- **THEN** request body schemas, response schemas, and parameter schemas SHALL be displayed with proper formatting

### Requirement: API reference landing pages exist
Alongside the auto-generated endpoint pages, hand-written `.mdx` pages SHALL provide orientation for the API Reference section.

#### Scenario: API Reference overview page exists
- **GIVEN** `api-reference/overview.mdx`
- **WHEN** a user visits the API Reference section
- **THEN** they SHALL see an overview of how the API is organized, base URL, and authentication approach

#### Scenario: Authentication page exists
- **GIVEN** `api-reference/authentication.mdx`
- **WHEN** a user wants to understand auth
- **THEN** they SHALL see a guide to JWT access tokens, refresh tokens, cookie-based auth, OAuth2 providers, and permission model
- **AND** it SHALL include code examples in `<CodeGroup>` for `curl` and `python`

#### Scenario: Rate limiting page exists
- **GIVEN** `api-reference/rate-limiting.mdx`
- **WHEN** a user needs to understand rate limits
- **THEN** they SHALL see per-endpoint rate limits, retry-after headers, and how to handle 429 responses

#### Scenario: Errors page exists
- **GIVEN** `api-reference/errors.mdx`
- **WHEN** a user encounters an error
- **THEN** they SHALL see the `APIResponse[T]` envelope structure, the `APIException` hierarchy (ValidationException 422, NotFoundException 404, UnauthorizedException 401, ConflictException 409, TooManyRequestsException 429, etc.), and example error JSON bodies

### Requirement: Code examples are provided for key endpoints
For the most important endpoints (login, register, upload document, search, agent-saul session), hand-written code examples SHALL supplement the auto-generated documentation.

#### Scenario: Login endpoint has curl and Python examples
- **GIVEN** the auto-generated `POST /auth/login` page
- **WHEN** a dev reads it
- **THEN** the page SHALL include a `<CodeGroup>` with `curl` and `python-sdk` tabs showing authentication flow

#### Scenario: Document upload has multipart example
- **GIVEN** the auto-generated `POST /documents/upload` page
- **WHEN** a dev reads it
- **THEN** the page SHALL show how to upload a file with multipart/form-data
