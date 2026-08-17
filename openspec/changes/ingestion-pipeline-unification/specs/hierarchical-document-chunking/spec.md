## Purpose

Define the quality contract for turning a parsed document into chunks: structure-aware for every document kind,
token-bounded against a declared counter, lossless with respect to document content, and produced without
blocking the event loop or rebuilding parser models per call.

## ADDED Requirements

### Requirement: Every document kind is chunked structure-aware
The system SHALL chunk all document kinds, including legal documents, using the same structure-aware chunking
behaviour. Chunking SHALL derive boundaries from the parsed document structure, SHALL retain each chunk's heading
path, and SHALL merge adjacent peer sections that fit within the token bound. Splitting a document by blank-line
pattern matching SHALL NOT be used for any document kind.

#### Scenario: Legal document chunks carry their heading path
- **WHEN** a legal document with a nested heading hierarchy is chunked
- **THEN** each chunk SHALL carry the heading path of the section it came from

#### Scenario: Peer sections are merged within the bound
- **WHEN** two adjacent sibling sections together fit within the token bound
- **THEN** they SHALL be emitted as one chunk rather than two

#### Scenario: Clause boundaries are respected for legal documents
- **WHEN** a legal document contains numbered clauses
- **THEN** a chunk SHALL NOT begin or end mid-clause where the clause fits within the token bound

### Requirement: Chunking never silently discards document content
The system SHALL represent the whole parsed document in its emitted chunks. If any cap on the number of chunks or
sections is retained, exceeding it SHALL emit a quality warning that names the count of discarded content, and
the document SHALL NOT be reported as fully processed.

#### Scenario: A long document is chunked in full
- **WHEN** a document containing more than two hundred sections is chunked
- **THEN** the emitted chunks SHALL cover the whole document rather than a fixed prefix of it

#### Scenario: A retained cap emits a warning naming the loss
- **WHEN** a cap on emitted chunks is reached
- **THEN** the system SHALL emit a quality warning naming the number of sections not represented

#### Scenario: A degenerate parse is still reported
- **WHEN** a parsed document yields one section or none
- **THEN** the system SHALL emit a quality warning identifying the document

### Requirement: Token bounds are enforced against a declared counter
Every emitted chunk SHALL be within the configured token bound as measured by an explicitly declared token
counter. Where that counter is not the embedding model's own counter, the divergence SHALL be recorded together
with the safety margin it implies.

#### Scenario: Chunks respect the bound
- **WHEN** a document is chunked with a configured token bound
- **THEN** every emitted chunk's token count under the declared counter SHALL be less than or equal to that bound

#### Scenario: The counter in force is discoverable
- **WHEN** a chunk's token bound is enforced
- **THEN** the identity of the counter used SHALL be recorded so a reader can tell whether it matches the embedding model

### Requirement: Token counter acquisition is cached at process scope
The system SHALL acquire its token counter once per process and reuse it. Acquiring the counter SHALL NOT perform
disk or network loading on each chunking operation, and the first acquisition SHALL NOT block the event loop.

#### Scenario: Repeated chunking loads the counter once
- **WHEN** two chunking operations run in the same process
- **THEN** the underlying counter SHALL be loaded exactly once

#### Scenario: First acquisition yields to the event loop
- **WHEN** the counter is acquired for the first time inside asynchronous code
- **THEN** the acquisition SHALL be offloaded so other tasks continue to run

### Requirement: Parsing does not block the event loop and reuses its converter
Document parsing and its export steps SHALL be offloaded so they do not block the event loop, and the parser
SHALL be constructed once per process and reused across documents.

#### Scenario: A concurrent task progresses during a parse
- **WHEN** a document parse is in progress inside asynchronous code
- **THEN** other scheduled asynchronous tasks SHALL continue to make progress

#### Scenario: The parser is constructed once
- **WHEN** two documents are parsed in the same process
- **THEN** the parser SHALL be constructed exactly once

### Requirement: Parsed table structures are retained
Parsing SHALL retain the table structures the parser extracts and expose them to downstream stages. It SHALL NOT
return an empty table set for a document containing tables.

#### Scenario: A document with a table exposes it
- **WHEN** a document containing at least one table is parsed
- **THEN** the parse result SHALL expose at least one table structure

#### Scenario: A document without tables exposes none
- **WHEN** a document containing no tables is parsed
- **THEN** the parse result SHALL expose an empty table set without error

### Requirement: A declared content type is honoured or not accepted
Where the parsing contract accepts a content type, that value SHALL influence how the document is parsed. A
content type that is accepted and ignored SHALL NOT remain in the contract.

#### Scenario: Content type selects the parse behaviour
- **WHEN** a caller supplies a content type that the parser supports
- **THEN** the parse SHALL use the behaviour for that content type

#### Scenario: An unsupported content type is rejected
- **WHEN** a caller supplies a content type the parser cannot handle
- **THEN** the system SHALL fail with a diagnostic naming the unsupported content type rather than parsing it as another type
