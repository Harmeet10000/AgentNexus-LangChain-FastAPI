## Purpose

Define the externally visible contract for document ingestion: an accepted upload becomes durable, retrievable
chunk records and a reported terminal status, through one multi-stage pipeline whose stages are individually
recoverable and whose failures short-circuit rather than cascade.

## ADDED Requirements

### Requirement: Ingestion runs as one multi-stage recoverable pipeline
The system SHALL process every accepted document through a single multi-stage ingestion pipeline in which each
stage is an independently recoverable unit. The system SHALL NOT retain a second ingestion implementation, and
SHALL NOT expose a single-stage wrapper that forwards an entire ingestion to one opaque routine.

#### Scenario: An accepted document reaches a terminal status
- **WHEN** a document is accepted for ingestion
- **THEN** the system SHALL advance it through the pipeline stages and SHALL record a terminal status of either completed or failed

#### Scenario: Resubmitting the same document resumes it rather than duplicating it
- **WHEN** the same document identity is submitted for ingestion a second time
- **THEN** the system SHALL resume that document's ingestion at its first incomplete stage under the same identity, and SHALL NOT create a second document record

#### Scenario: Exactly one ingestion entry point exists
- **WHEN** the repository is inspected for paths that begin an ingestion
- **THEN** exactly one SHALL exist, and it SHALL be the multi-stage pipeline

### Requirement: Ingestion executes outside the request path
The system SHALL perform document ingestion in a queue worker process, not inside the HTTP request that accepts
the upload. The accepting request SHALL return once the document is durably recorded and the ingestion work is
enqueued.

#### Scenario: Upload returns before ingestion completes
- **WHEN** a client uploads a document
- **THEN** the response SHALL report an accepted, non-terminal status and SHALL NOT wait for parsing, embedding, or graph writes

#### Scenario: Enqueue failure is reported, not swallowed
- **WHEN** the ingestion work cannot be enqueued
- **THEN** the system SHALL report the failure to the client and SHALL NOT report the document as accepted for processing

### Requirement: Synchronous ingestion surfaces fail closed when their shared pipeline is not provisioned
Where a synchronous ingestion surface depends on a shared pipeline instance that the deployment deliberately does
not provision, that surface SHALL respond with a typed service-unavailable error. It SHALL NOT raise an
attribute or type error, and it SHALL NOT report success.

#### Scenario: Unprovisioned pipeline yields service unavailable
- **WHEN** a request reaches a synchronous ingestion surface whose shared pipeline instance is absent
- **THEN** the system SHALL respond with a service-unavailable error in the standard error envelope naming the missing capability

#### Scenario: Absent pipeline never surfaces as an internal error
- **WHEN** the shared pipeline instance is absent
- **THEN** no request SHALL produce an unhandled attribute error, and no response SHALL carry a success status

### Requirement: A failed stage short-circuits the remaining stages
When a stage records an unrecoverable failure, the system SHALL terminate the pipeline for that document without
executing any further processing stage.

#### Scenario: Guard failure stops the pipeline
- **WHEN** an early stage rejects a document as unprocessable and records a failure
- **THEN** no later processing stage SHALL execute for that document, and the document SHALL reach a terminal failure status

#### Scenario: External model calls are not spent after a failure
- **WHEN** a document has recorded an unrecoverable failure
- **THEN** the system SHALL issue no further external model or knowledge-graph calls for that document

#### Scenario: A recoverable stage failure continues the pipeline
- **WHEN** a stage encounters a recoverable external-model failure and produces a documented degraded result
- **THEN** the pipeline SHALL continue to the next stage and the degradation SHALL be recorded

### Requirement: Failures are recorded as serialisable failure records that preserve the original diagnostic
The pipeline SHALL represent a stage failure as a serialisable record carrying a stable failure code, a message,
and its context. The system SHALL NOT place an exception instance in a persisted pipeline channel, and error
handling for a failing stage SHALL NOT replace the original diagnostic with a secondary error raised by the
handler itself.

#### Scenario: Failure record is serialisable
- **WHEN** a stage records a failure
- **THEN** the failure record SHALL round-trip through the pipeline's persistence serialiser without relying on an arbitrary-object fallback

#### Scenario: Original diagnostic survives degraded handling
- **WHEN** a stage's recoverable-failure handler runs
- **THEN** the recorded diagnostic SHALL identify the original failure cause, and the handler SHALL NOT raise a new error of its own

#### Scenario: Degraded fan-out item keeps its identity
- **WHEN** a fanned-out per-chunk stage degrades for one item
- **THEN** the recorded degradation SHALL carry that item's document and chunk identity

### Requirement: Document status transitions remain observable through terminal state
The system SHALL record a document status transition at each externally meaningful stage boundary, and SHALL
expose the current status to the document's owner. Adding pipeline persistence SHALL NOT remove the status
surface.

#### Scenario: Status advances through processing
- **WHEN** a document is parsed, then persisted, then completed
- **THEN** the document's recorded status SHALL advance through the corresponding values in that order

#### Scenario: Failure is reported as a terminal status
- **WHEN** a document's ingestion fails unrecoverably
- **THEN** the document's recorded status SHALL be a terminal failure value and SHALL remain queryable by its owner

#### Scenario: Status is queryable while processing
- **WHEN** the owner queries a document that is still being processed
- **THEN** the system SHALL report the current non-terminal status rather than an error

### Requirement: Chunk records are the sole persisted retrieval unit
The pipeline's persistence stages SHALL write document records and chunk records as the only relational
retrieval truth. They SHALL NOT write clause records, parent-document records, or relational entity or
relationship records as retrieval truth. Extracted entities and relationships SHALL be written to the
knowledge-graph store.

#### Scenario: Persistence writes chunk records
- **WHEN** a document completes parsing and chunking
- **THEN** the pipeline SHALL persist one chunk record per chunk, each carrying its document identity, ordinal position, text, and embedding

#### Scenario: No legacy clause records are produced
- **WHEN** a document completes ingestion
- **THEN** no clause records and no parent-document records SHALL have been written

#### Scenario: Entities and relationships go to the graph store
- **WHEN** the pipeline extracts entities and relationships
- **THEN** they SHALL be written to the knowledge-graph store and SHALL NOT be written to a relational retrieval table

### Requirement: The pipeline fetches document bytes by reference from the object store
The pipeline SHALL receive an object-store reference for the document and SHALL fetch its bytes inside the stage
that needs them. Document bytes SHALL NOT be carried across stage boundaries in pipeline state.

#### Scenario: Bytes are fetched inside the parsing stage
- **WHEN** the pipeline begins parsing
- **THEN** it SHALL fetch the document bytes from the object store using the reference it was given

#### Scenario: Missing object yields a terminal failure with no partial writes
- **WHEN** the referenced object cannot be fetched
- **THEN** the document SHALL reach a terminal failure status and no chunk records SHALL have been written for it

### Requirement: Each chunk is persisted once per ingestion
The pipeline SHALL write each chunk's full record exactly once per ingestion. A subsequent knowledge-graph
verification pass SHALL update only that chunk's verification fields, and its per-chunk graph calls SHALL be
issued as a bounded concurrent fan-out rather than serially.

#### Scenario: Chunk payload is written once
- **WHEN** a document producing many chunks is ingested and verified
- **THEN** each chunk's text and embedding SHALL be written exactly once

#### Scenario: Verification updates only verification fields
- **WHEN** knowledge-graph verification completes for a chunk
- **THEN** only that chunk's verification state and graph reference SHALL be updated

#### Scenario: Per-chunk graph calls are concurrent and bounded
- **WHEN** verification runs over many chunks
- **THEN** the graph calls SHALL be issued concurrently under a stated concurrency bound rather than one after another

### Requirement: The pipeline fails closed when its target schema is absent
The pipeline SHALL NOT define or apply database schema changes. When a relational table it writes is absent, the
pipeline SHALL fail with a diagnostic naming the missing relation.

The contract splits by *which* table is missing, because the terminal status the pipeline records lives in the
document table itself:

- Where the document table exists and a downstream table it writes does not, the document SHALL reach a terminal
  failure status whose diagnostic names the missing relation, and SHALL NOT be left in a non-terminal status.
- Where the document table itself does not exist, there is no row on which to record a status. The pipeline SHALL
  fail with a diagnostic naming the missing relation, surfaced through the task result and the log, and SHALL NOT
  imply, report, or require a document row. No requirement here SHALL be read as demanding a status transition that
  has nowhere to be written.

Both tables are created by the foundation change's single migration; this change depends on that migration and ships
no revision of its own.

#### Scenario: A missing downstream table produces a terminal failure on the document
- **WHEN** ingestion runs against a database where the document table exists and the chunk table does not
- **THEN** the document SHALL reach a terminal failure status whose diagnostic names the missing relation, and SHALL NOT remain in a non-terminal status

#### Scenario: A missing document table fails through the task result, with no document row implied
- **WHEN** ingestion runs against a database where the document table itself does not exist
- **THEN** the pipeline SHALL fail with a diagnostic naming the missing relation, reported through the task result and the log, and SHALL NOT attempt to record a status or report a document row

#### Scenario: No schema changes ship with the pipeline
- **WHEN** the change is applied to a database at the expected schema version
- **THEN** applying it SHALL introduce no new schema revision of its own
