## Purpose

Define the ingest-time context-summary contract that fills the long-empty `preamble`
column and integrates document context into the searchable text — adopted only on
measured evidence.

## ADDED Requirements

### Requirement: The preamble column is filled at ingest time by amortized context summaries

Chunk ingestion SHALL write a document-context summary into the existing `preamble`
column, generated with the document-level context cached and reused across the
document's chunks rather than re-sent per chunk. The generated search text SHALL
include the preamble so context participates in retrieval.

#### Scenario: Context reaches the index without per-chunk re-sending

- **WHEN** a document is ingested under the feature
- **THEN** every chunk row carries a non-empty preamble derived from shared
  document-level context
- **AND** the provider call count for context generation is proportional to
  documents, not to chunks

#### Scenario: The empty-column state is distinguishable

- **WHEN** the feature is disabled for a document
- **THEN** its chunks are explicitly marked as context-free rather than silently
  carrying empty preambles that look like the feature ran and produced nothing

### Requirement: Adoption is measured before default-on

The feature SHALL be evaluated against the retrieval measurement baseline before it
serves by default, and a regression on that baseline SHALL revert the default. The
feature's effect SHALL be measurable per document class.

#### Scenario: A measured win turns it on; a measured loss turns it off

- **WHEN** the baseline comparison completes for a document class
- **THEN** the default-on decision, either direction, cites the measured delta
- **AND** no default flips without a recorded baseline comparison
