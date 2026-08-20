## Purpose

Define the re-ranking half of retrieval quality, and only that half: every path that returns ranked results
re-ranks the fused candidate set through one shared cross-encoding model that is loaded once per process, index
identities that a query must name are defined once instead of being embedded as literals, and structured extraction
runs upstream of persistence and graph writes.

**What this capability deliberately does not define.** The fused-retrieval contract itself — which modes run, that
their results are combined by rank position, that the chunk store is the single retrieval source of truth, and that a
retrieval mode whose required database extension or index is absent is a **loud provisioning failure** rather than a
silently narrowed result — belongs to `document-retrieval-schema` in the unified-schema change. This capability
reads that contract and adds re-ranking on top of it. It contains no fusion requirement, no single-source
requirement, and no degrade-and-continue behaviour for a missing database capability, because a missing extension
means the migration that creates it did not run.

## ADDED Requirements

### Requirement: Fused results are re-ranked before they are returned, by one shared re-ranker
Every retrieval path that returns ranked results to a caller SHALL re-rank the fused candidate set with a
cross-encoding relevance model before returning. There SHALL be exactly one re-ranking implementation, and its model
SHALL be loaded once per process and reused, not constructed per call. A retrieval path that fuses and returns
without re-ranking SHALL NOT remain.

#### Scenario: The direct fused retrieval path re-ranks
- **WHEN** a caller invokes the retrieval path that fuses branch results and returns them directly, rather than through the retrieval graph
- **THEN** that path SHALL re-rank the fused candidate set before returning, and the returned order SHALL reflect the re-ranker's relevance judgement rather than the fusion order alone

#### Scenario: Every ranked retrieval path re-ranks
- **WHEN** any retrieval path returns ranked results to a caller
- **THEN** that path SHALL have applied re-ranking

#### Scenario: One re-ranking implementation serves every path
- **WHEN** re-ranking is performed anywhere in the system
- **THEN** it SHALL be performed by the same re-ranking behaviour, and no second re-ranking implementation SHALL exist

#### Scenario: The re-ranking model is loaded once per process
- **WHEN** two retrieval requests in one process are re-ranked
- **THEN** the cross-encoding model SHALL have been loaded once and reused, and SHALL NOT be loaded again per request or per call site

### Requirement: Re-ranker unavailability degrades to the fused order and is observable
When the re-ranking model is unavailable or fails, the system SHALL return the fused order rather than raising, and
SHALL report the degradation as an observable signal, not only as a log line.

This is a runtime condition and is deliberately treated differently from an absent database extension. A model that
cannot be downloaded or loaded is a recoverable environmental failure whose degraded result is still a correct
ranking, and the degradation is visible on the health surface. An absent extension is a deployment error — the
migration that creates it did not run — and is a loud provisioning failure owned by `document-retrieval-schema`.
The two SHALL NOT be conflated.

#### Scenario: Model load failure returns fused order
- **WHEN** the re-ranking model cannot be loaded
- **THEN** the system SHALL return results in fused order and SHALL NOT raise to the caller

#### Scenario: Degradation is reported by the health surface
- **WHEN** the re-ranking model is unavailable
- **THEN** the system's health surface SHALL report the re-ranker as unavailable

### Requirement: Index identities required inside queries are defined once
Where a query must name a database index in order to read that index's corpus statistics, the index name SHALL be
defined in exactly one place and referenced from the query. No query SHALL embed an index name as a literal.

For the lexical branch this is stronger than a naming convention. The lexical extension's query-construction
function takes the index name as a **literal argument**, so the name is part of the query contract: an index of the
correct shape under a different name does not satisfy the query. The single definition SHALL therefore be the same
name the migration that creates the index uses, and the named index SHALL be one that a migration in the project's
history creates.

#### Scenario: Renaming the index updates the query
- **WHEN** the single index-name definition is changed
- **THEN** every query naming that index SHALL follow, with no literal left behind

#### Scenario: No literal index name remains in a query
- **WHEN** the retrieval queries and the index-maintenance calls are inspected
- **THEN** none SHALL contain an index name as a string literal

#### Scenario: The named index is one a migration creates under that exact name
- **WHEN** the single index-name definition is compared against the project's migration history
- **THEN** a migration SHALL create an index of that exact name using the lexical access method

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
