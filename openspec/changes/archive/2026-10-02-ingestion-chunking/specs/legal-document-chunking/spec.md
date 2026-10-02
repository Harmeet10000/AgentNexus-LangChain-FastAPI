## Purpose

Define the contract for chunking a legal corpus: chunk identity that survives re-ingestion at a new
document version, clause locus recovered when layout misses it, a strict separation between the text
that is embedded and the text a citation quotes, honest declaration of impure splits, and a
document-processing toolchain carrying no deep-learning runtime.

## ADDED Requirements

### Requirement: Chunk identity is scoped to a document version

Chunk identity SHALL include the document version, so that re-ingesting a document at a new version
neither overwrites nor orphans the chunks of prior versions.

#### Scenario: Two versions of one document coexist

- **WHEN** a document is re-ingested at a new version
- **THEN** chunks from the prior version SHALL remain addressable, and chunks from both versions SHALL
  be distinguishable by their version

#### Scenario: Retrieval scopes to a single version

- **WHEN** chunks are retrieved for a document
- **THEN** the result SHALL be scoped to one document version, so a superseded clause is not returned
  alongside the clause that superseded it

#### Scenario: Chunk uniqueness includes the version

- **WHEN** the uniqueness constraint on chunk identity is inspected
- **THEN** it SHALL be over the document, its version, and the chunk index together

#### Scenario: Every chunk carries a locus

- **WHEN** a chunk is written
- **THEN** it SHALL carry a locus field, which is either the structural position recovered for it or
  an explicit absence — never an empty string standing in for an unknown

### Requirement: Clause numbering survives a layout miss

Where clause numbers appear as inline emphasis rather than as headings, a post-pass SHALL recover them
into chunk locus rather than losing them.

#### Scenario: Inline-numbered clauses are recovered

- **WHEN** a document's clause numbers appear as inline bold runs rather than as headings
- **THEN** the post-pass SHALL recover them into the locus of the chunks they begin

#### Scenario: Recovery is pure

- **WHEN** clause numbering is recovered
- **THEN** the recovery SHALL be a function from chunks to chunks performing no input or output, so it
  is testable without a parser or a database

#### Scenario: Recovery does not invent a locus

- **WHEN** no clause number can be recovered for a chunk
- **THEN** its locus SHALL remain absent rather than being assigned a synthesised value

### Requirement: Embedded text and cited text are distinct

The text used to compute an embedding SHALL be distinguishable from the text a citation quotes, and
the two SHALL be stored in separate fields.

#### Scenario: The citation field holds bare chunk text

- **WHEN** a chunk is stored
- **THEN** its content field SHALL hold the chunk's own text without any prepended heading or section
  context

#### Scenario: The context field holds the contextualization

- **WHEN** a chunk is stored
- **THEN** its preamble field SHALL hold the contextualization derived from the document structure

#### Scenario: The embedding is computed over both

- **WHEN** a chunk is embedded
- **THEN** the embedded text SHALL be the concatenation of the contextualization and the chunk text,
  so that separating the fields does not weaken the embedding

#### Scenario: Lexical retrieval input is unchanged by the separation

- **WHEN** the separation is applied to a document that was previously chunked
- **THEN** the text available to lexical retrieval SHALL be unchanged, because it is derived from both
  fields together

#### Scenario: The token bound is enforced against the text actually embedded

- **WHEN** the token bound is checked
- **THEN** it SHALL be measured over the concatenated text that is embedded, not over the chunk text
  alone

### Requirement: Impure splits declare themselves

A chunk produced by a split that could not respect structure SHALL say so, and SHALL be the only kind
of chunk permitted to overlap its neighbour.

#### Scenario: A mid-clause split is flagged

- **WHEN** a split cuts through a clause because no structural boundary was available
- **THEN** the resulting chunks SHALL carry an impurity flag

#### Scenario: Only the fallback path overlaps

- **WHEN** chunks are produced by a structure-aware policy
- **THEN** their overlap SHALL be zero, and non-zero overlap SHALL occur only on the flagged fallback
  path

#### Scenario: Impurity is visible to a consumer

- **WHEN** a consumer reads a chunk
- **THEN** it SHALL be able to tell whether that chunk was produced by a structure-aware policy or by
  the fallback, without re-running chunking

### Requirement: The document-processing toolchain carries no deep-learning runtime

Document parsing, optical character recognition, and chunking SHALL be performed without a
deep-learning tensor runtime in the deployed image.

#### Scenario: The tensor runtime is absent

- **WHEN** the resolved dependency set of the runtime image is inspected
- **THEN** no tensor runtime and no sentence-transformer package SHALL be present

#### Scenario: Optical character recognition uses a lightweight engine

- **WHEN** a scanned document is parsed
- **THEN** the optical character recognition engine SHALL be one that requires no tensor runtime

#### Scenario: No parser is constructed with the default engine implicitly

- **WHEN** a document converter is constructed anywhere in the ingestion path
- **THEN** its pipeline options SHALL be supplied explicitly rather than inherited from the library
  default

#### Scenario: The token counter cache survives the change

- **WHEN** the toolchain change is complete
- **THEN** the process-scoped token counter caching behaviour SHALL be unchanged

### Requirement: Ingestion modules construct chat models through a public factory

A module in the ingestion path that needs a chat model SHALL obtain it from a public factory rather
than by importing a private symbol from another module.

#### Scenario: No private cross-module import remains in the ingestion path

- **WHEN** the ingestion path's modules are linted
- **THEN** no import of a private symbol from another module SHALL be reported

#### Scenario: One construction contract

- **WHEN** two ingestion modules each need a chat model
- **THEN** both SHALL obtain it from the same public factory

### Requirement: One chunking implementation serves the ingestion path

The ingestion path SHALL have a single chunking implementation, and pattern-matching text splitters
SHALL NOT remain reachable from it.

#### Scenario: The whitespace splitter is unreachable

- **WHEN** any ingestion call site chunks a document
- **THEN** it SHALL reach the structure-aware chunking path, and no blank-line or fixed-size splitter
  SHALL be reachable from ingestion

#### Scenario: No fixed chunk-size configuration governs ingestion

- **WHEN** the ingestion configuration is inspected
- **THEN** no fixed chunk-size or overlap setting SHALL govern the structure-aware path, whose bounds
  come from the resolved policy
