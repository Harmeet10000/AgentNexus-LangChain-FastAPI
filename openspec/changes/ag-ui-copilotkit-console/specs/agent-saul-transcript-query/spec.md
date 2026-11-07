## Purpose

Lets an authorized client discover a caller's Agent Saul sessions and read a session's persisted transcript as paginated JSON, so a console can render history without a second chat store and without weakening the redaction and erasure guarantees of the export path.

## ADDED Requirements

### Requirement: Session listing

The system SHALL expose `GET /api/v2/agent-saul/sessions` returning only sessions the caller is authorized to read, scoped to the selected tenant, ordered by most recently updated first, and paginated with a bounded page size.

Each entry MUST include an opaque session identifier, the document scope summary, the terminal or current workflow status, the last-updated timestamp, and whether reviewer attention is required. When attention is required it MUST include the interrupt reason but no interrupt response payload or transcript content.

The endpoint SHALL accept bounded filters for workflow status, attention-required state, and authorized document id. Filters MUST be applied server-side before pagination so that a page is not silently shortened by client-side filtering.

#### Scenario: Caller lists own sessions

- **WHEN** an authenticated caller requests the session list for their tenant
- **THEN** the response contains only their authorized sessions, newest first, with status and document scope but no transcript content

#### Scenario: Other tenant is invisible

- **WHEN** a caller requests the session list while selecting a tenant they do not belong to
- **THEN** the request is refused and no session identifiers are returned

#### Scenario: Filter sessions requiring attention

- **WHEN** an authorized caller filters sessions to attention-required for one document
- **THEN** every returned summary belongs to that document, has an open interrupt, and exposes only the interrupt reason

### Requirement: Paginated transcript read

The system SHALL expose `GET /api/v2/agent-saul/sessions/{session_id}/transcript` returning transcript rows in display order `(turn_seq, ordinal)` as JSON, with a keyset cursor and a bounded page size.

The response MUST report the cursor needed to fetch the next page and MUST report whether more rows exist. Rows MUST be the same rows, in the same order, as the NDJSON export for that session.

#### Scenario: Read first page

- **WHEN** an authorized caller reads a session transcript without a cursor
- **THEN** the response returns the first page in `(turn_seq, ordinal)` order together with a next cursor and a more-rows flag

#### Scenario: Export and read agree

- **WHEN** the same session is read through the JSON endpoint and exported as NDJSON
- **THEN** the two sequences of rows and their ordering are identical

### Requirement: Redaction is represented, never reconstructed

A redacted transcript row SHALL be returned with `content` null and `redacted` true. The system MUST NOT substitute a placeholder body, summary, embedding-derived reconstruction, or prior version for erased content.

#### Scenario: Erased turn remains visible as erased

- **WHEN** a caller reads a transcript containing an erased turn
- **THEN** that row is returned with null content and `redacted` true, and ordering still reflects its original position

### Requirement: Transcript authorization and non-disclosure

Transcript and session reads SHALL require an authenticated caller and SHALL authorize on session ownership or an explicitly granted review capability for that tenant.

A session in another tenant MUST be reported as not found. A session in the caller's tenant that exists but is not readable by the caller MUST be refused without revealing its content or metadata.

#### Scenario: Cross-tenant session is not found

- **WHEN** a caller requests a transcript id belonging to another tenant
- **THEN** the response is a not-found error and reveals no session metadata

#### Scenario: Same-tenant unauthorized read

- **WHEN** a caller in the same tenant lacks ownership and review authority for the session
- **THEN** the read is refused and no rows are returned

### Requirement: Schema gating before streaming rows

Before returning any transcript rows the system SHALL verify that the session's persisted schema version is supported. An unsupported version MUST fail the request as a whole rather than returning a partial or silently downgraded transcript.

#### Scenario: Unsupported schema version

- **WHEN** a session was persisted under a schema version the service no longer reads
- **THEN** the request fails with an explicit unsupported-schema error and no rows are returned

### Requirement: Pagination stability under append

Pagination SHALL be keyset-based on the full ordering key, including the row identity, so that rows appended after a page is served cannot duplicate or skip rows on subsequent pages.

#### Scenario: Rows appended between pages

- **WHEN** new transcript rows are persisted after the first page is served and the caller requests the next page with the returned cursor
- **THEN** the second page contains no duplicate of a previously returned row and no gap in the earlier ordering
