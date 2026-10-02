## Purpose

Define the quality contract for turning a parsed document into chunks: structure-aware for every document kind,
token-bounded against a declared counter, lossless with respect to document content, and produced without
blocking the event loop or rebuilding parser models per call.

## MODIFIED Requirements

### Requirement: Every document kind is chunked structure-aware
The system SHALL chunk all document kinds, including legal documents, using the same structure-aware chunking
behaviour. Chunking SHALL derive boundaries from the parsed document structure, SHALL retain each chunk's heading
path, and SHALL merge adjacent peer sections that fit within the token bound. Splitting a document by blank-line
pattern matching SHALL NOT be used for any document kind. Chunking SHALL further resolve a chunk policy from the
document's classified kind, such that each of contract, statute, judgment, and filing resolves a policy distinct
from the others, and the identity of the resolved policy SHALL be recorded on every emitted chunk. Policy
resolution SHALL be a pure function of the classified kind and SHALL perform no input or output.

#### Scenario: Legal document chunks carry their heading path
- **WHEN** a legal document with a nested heading hierarchy is chunked
- **THEN** each chunk SHALL carry the heading path of the section it came from

#### Scenario: Peer sections are merged within the bound
- **WHEN** two adjacent sibling sections together fit within the token bound
- **THEN** they SHALL be emitted as one chunk rather than two

#### Scenario: Clause boundaries are respected for legal documents
- **WHEN** a legal document contains numbered clauses
- **THEN** a chunk SHALL NOT begin or end mid-clause where the clause fits within the token bound

#### Scenario: Each legal family resolves a distinct policy
- **WHEN** a document classified as a contract, a statute, a judgment, and a filing are each chunked
- **THEN** four distinct chunk policies SHALL be resolved, one per kind

#### Scenario: The resolved policy is recorded on the chunk
- **WHEN** a chunk is emitted
- **THEN** it SHALL carry the identity of the policy that produced it, so a reader can tell which policy
  was in force without re-running classification

#### Scenario: An unclassified document still chunks
- **WHEN** a document's kind cannot be classified
- **THEN** a default policy SHALL be resolved and recorded, and chunking SHALL proceed rather than fail

#### Scenario: Policy resolution performs no input or output
- **WHEN** a chunk policy is resolved for a document kind
- **THEN** no database, network, or filesystem access SHALL occur during resolution
