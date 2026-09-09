## Purpose

Define how structured knowledge is extracted from a document and made available: extraction runs before
chunking so both derive from one parse, extracted entities reach the knowledge graph idempotently,
extraction failure degrades rather than aborting ingestion, structural navigation reasons over the
document tree this system stores rather than one an external service holds, and no document content is
sent to a third-party structural service.

## ADDED Requirements

### Requirement: Extraction precedes chunking

Structured extraction SHALL run before chunking, and its output SHALL be available to the stage that
writes chunks.

#### Scenario: Extraction output reaches the chunk writer

- **WHEN** a document is ingested
- **THEN** extraction SHALL have completed before chunking begins, and its output SHALL be available to
  the chunk writer

#### Scenario: Both derive from one parse

- **WHEN** extraction and chunking run for one document
- **THEN** both SHALL operate on the same parse of that document, so extracted entities and the chunks
  describing them do not have to be reconciled afterward

### Requirement: Extracted entities become graph episodes idempotently

Extraction output SHALL be written to the knowledge graph through the application's existing canonical
writer.

#### Scenario: Clause entities are written as episodes

- **WHEN** extraction yields clause entities
- **THEN** they SHALL be written to the knowledge graph as episodes

#### Scenario: Re-ingestion creates no duplicates

- **WHEN** a document is ingested a second time
- **THEN** no duplicate episode SHALL be created for an entity already present

#### Scenario: Writing uses the existing canonical writer

- **WHEN** episodes are written
- **THEN** they SHALL be written through the application's existing canonical graph writer rather than
  a second write path

### Requirement: Extraction failure is non-fatal and visible

An unavailable extraction provider SHALL NOT fail ingestion, and SHALL NOT be indistinguishable from a
document with nothing to extract.

#### Scenario: Ingestion completes without extraction

- **WHEN** the extraction provider is unavailable
- **THEN** ingestion SHALL complete and the document SHALL be stored

#### Scenario: The document is flagged

- **WHEN** extraction fails for a document
- **THEN** that document SHALL be flagged extraction-incomplete

#### Scenario: An empty extraction is not a failure

- **WHEN** extraction succeeds and yields no entities
- **THEN** the document SHALL NOT be flagged extraction-incomplete, because yielding nothing and
  failing are distinct outcomes

#### Scenario: Failure is typed, not swallowed

- **WHEN** extraction fails
- **THEN** the failure SHALL be represented as a typed value the ingestion path handles explicitly,
  rather than an exception caught and discarded

### Requirement: Structural navigation reasons over the stored document tree

Navigation over a document's structure SHALL use the tree this system persists.

#### Scenario: Navigation uses the persisted tree

- **WHEN** a query requires structural navigation of a document
- **THEN** retrieval SHALL reason over the persisted document tree

#### Scenario: No external structural service is called

- **WHEN** structural navigation runs
- **THEN** no request SHALL be made to an external document-structure service, and no document content
  SHALL be sent to one

#### Scenario: The navigator is pure

- **WHEN** a document tree is navigated
- **THEN** the navigation SHALL be a function from a tree and a query to node paths, performing no
  input or output, so it is testable without a database

#### Scenario: Navigation is reached through the data-access layer

- **WHEN** structural navigation is exposed as a retrieval branch
- **THEN** it SHALL be reached through the repository and the service layer, and SHALL NOT be reached
  directly from a route handler

### Requirement: The external structural-service surface is retired

No symbol for the external document-structure client SHALL remain exported or constructible.

#### Scenario: No client symbol is exported

- **WHEN** the shared retrieval package is imported
- **THEN** no external document-structure client symbol SHALL be exported from it

#### Scenario: No construction site remains

- **WHEN** application startup is inspected
- **THEN** no construction of an external document-structure client SHALL be present, whether active or
  commented

#### Scenario: The application still imports

- **WHEN** the package is removed
- **THEN** the application SHALL import successfully and no reference to the removed package SHALL
  remain
