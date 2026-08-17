## Purpose

Define the retrieval-quality contract: lexical, vector, and fuzzy branches fused by one documented rule, re-ranked
before results are returned, reading the same chunk records the ingestion pipeline writes, with database extension
availability as a declared, per-environment precondition rather than an assumption.

## ADDED Requirements

### Requirement: Retrieval fuses lexical, vector, and fuzzy branches under one fusion rule
The system SHALL combine a lexical relevance branch, a vector similarity branch, and a fuzzy text-similarity
branch into one ranked candidate set using a single documented rank-fusion rule with a stated constant. There
SHALL be exactly one fusion implementation in the system.

#### Scenario: All available branches contribute to the fused set
- **WHEN** a retrieval query runs with every branch available
- **THEN** the fused candidate set SHALL contain contributions from each branch, ranked by the documented fusion rule

#### Scenario: One fusion implementation exists
- **WHEN** rank fusion is performed anywhere in the system
- **THEN** it SHALL be performed by the same fusion behaviour with the same stated constant

#### Scenario: The fusion constant is stated
- **WHEN** a reader inspects the retrieval contract
- **THEN** the fusion constant in force SHALL be stated explicitly rather than embedded in a query

### Requirement: Fused results are re-ranked before they are returned
The system SHALL re-rank the fused candidate set with a cross-encoding relevance model before returning results to
a caller. Every retrieval path that returns ranked results to a caller SHALL apply re-ranking.

#### Scenario: Re-ranking changes the returned order
- **WHEN** a query's fused candidates are re-ranked
- **THEN** the returned order SHALL reflect the re-ranker's relevance judgement rather than the fusion order alone

#### Scenario: Every ranked retrieval path re-ranks
- **WHEN** any retrieval path returns ranked results to a caller
- **THEN** that path SHALL have applied re-ranking

### Requirement: Re-ranker unavailability degrades to the fused order and is observable
When the re-ranking model is unavailable or fails, the system SHALL return the fused order rather than raising, and
SHALL report the degradation as an observable signal, not only as a log line.

#### Scenario: Model load failure returns fused order
- **WHEN** the re-ranking model cannot be loaded
- **THEN** the system SHALL return results in fused order and SHALL NOT raise to the caller

#### Scenario: Degradation is reported by the health surface
- **WHEN** the re-ranking model is unavailable
- **THEN** the system's health surface SHALL report the re-ranker as unavailable

### Requirement: Retrieval reads the same chunk records the pipeline writes
Retrieval SHALL read the document and chunk records that the ingestion pipeline writes. It SHALL NOT read clause
records or any other relational table as an alternative source of retrieval truth.

#### Scenario: Retrieval returns records the pipeline wrote
- **WHEN** a document is ingested and then queried
- **THEN** the results SHALL be drawn from the chunk records that ingestion wrote for it

#### Scenario: No alternative retrieval table is queried
- **WHEN** any retrieval query executes
- **THEN** it SHALL NOT query a clause table or any second chunk-like table as retrieval truth

### Requirement: Lexical search declares its database extension as a per-environment precondition
The lexical relevance branch SHALL declare the database extension and index access method it requires. Its
availability SHALL be verified in each environment before retrieval is served there, and the verification SHALL
inspect the database the application actually connects to rather than a container image.

#### Scenario: Availability is verified against the live database
- **WHEN** an environment is prepared to serve retrieval
- **THEN** the required extension and index access method SHALL be confirmed present on the database the application connects to

#### Scenario: Absent extension degrades rather than errors
- **WHEN** the required extension is unavailable in an environment
- **THEN** the lexical branch SHALL be omitted, fusion SHALL continue with the remaining branches, and the omission SHALL be reported

#### Scenario: A missing extension does not abort unrelated work
- **WHEN** the required extension is unavailable
- **THEN** unrelated features SHALL continue to function and no unrelated schema operation SHALL fail because of it

### Requirement: Index identities required inside queries are defined once
Where a query must name a database index in order to read that index's corpus statistics, the index name SHALL be
defined in exactly one place and referenced from the query. No query SHALL embed an index name as a literal.

#### Scenario: Renaming the index updates the query
- **WHEN** the single index-name definition is changed
- **THEN** every query naming that index SHALL follow, with no literal left behind

#### Scenario: No literal index name remains in a query
- **WHEN** the retrieval queries are inspected
- **THEN** none SHALL contain an index name as a string literal

### Requirement: The fuzzy branch is brought up with its extension and index, or dropped on the record
The fuzzy text-similarity branch SHALL be served by a declared database extension and a declared index. If either
is unavailable in an environment, the branch SHALL be omitted from fusion and the omission SHALL be reported, and
if the branch is not carried forward at all that decision SHALL be recorded rather than left as an absence.

#### Scenario: The fuzzy branch contributes when its extension and index exist
- **WHEN** the fuzzy extension and its index are present
- **THEN** the fuzzy branch SHALL contribute candidates to fusion

#### Scenario: A missing fuzzy index omits the branch observably
- **WHEN** the fuzzy extension or its index is absent
- **THEN** the branch SHALL be omitted, the omission SHALL be reported, and fusion SHALL proceed with the remaining branches

### Requirement: Structure extraction runs upstream of persistence and graph writes
Structured extraction of document content SHALL run before chunk persistence and before knowledge-graph writes, so
that both consume already-extracted structure rather than re-deriving it.

#### Scenario: Extraction precedes persistence
- **WHEN** a document is ingested
- **THEN** structured extraction SHALL have completed before chunk records are written

#### Scenario: Extraction precedes graph writes
- **WHEN** knowledge-graph writes occur for a document
- **THEN** they SHALL consume the already-extracted structure rather than performing extraction themselves

#### Scenario: Extraction is not performed twice
- **WHEN** a document completes ingestion
- **THEN** structured extraction SHALL have been performed once for that document
