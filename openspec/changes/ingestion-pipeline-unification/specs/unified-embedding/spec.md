## Purpose

Define the single embedding contract every retrieval path in the system builds on: one provider, one configured
dimension, a declared task type on both the query and document sides, one shared cache, one normalisation
convention per stored column, and failure that raises rather than substitutes.

## ADDED Requirements

### Requirement: One embedding path serves every live query and document embedding
The system SHALL provide exactly one embedding path used by every live query-time and ingestion-time embedding
call. Any additional embedding implementation retained for batch or offline use SHALL NOT be reachable from a
live request or ingestion path.

#### Scenario: Query and ingestion embeddings come from one path
- **WHEN** a retrieval query is embedded and a document chunk is embedded
- **THEN** both SHALL be produced by the same embedding path with the same configuration

#### Scenario: A retained batch path is not live
- **WHEN** a live request or an ingestion stage embeds text
- **THEN** it SHALL NOT reach the batch or offline embedding implementation

#### Scenario: The embedding client is reused across calls
- **WHEN** two embedding calls occur in the same process
- **THEN** the underlying provider client SHALL be constructed once and reused

### Requirement: Every embedding consumer resolves to the single path
Every module that embeds text SHALL resolve to the single embedding path. No code path SHALL reference an
embedding provider from a module that does not exist.

#### Scenario: No unresolvable embedding reference remains
- **WHEN** any embedding call site is exercised
- **THEN** it SHALL resolve without a module-resolution error

#### Scenario: A deferred import failure is not acceptable
- **WHEN** a function-local embedding import is executed for the first time at call time
- **THEN** it SHALL resolve to the single embedding path rather than fail at that moment

### Requirement: Embedding dimension derives from configuration and every vector column agrees with it
The reported embedding dimension SHALL derive from a single configured value. Every persisted vector column used
by the retrieval and memory paths SHALL declare the same width as that value. No embedding dimension SHALL be
stated as a literal outside historical schema revisions.

#### Scenario: Reported dimension equals the configured value
- **WHEN** the embedding path reports its dimension
- **THEN** the reported dimension SHALL equal the configured embedding dimension

#### Scenario: Persisted column width equals the configured value
- **WHEN** a vector column used for retrieval or memory is inspected
- **THEN** its declared width SHALL equal the configured embedding dimension

#### Scenario: A model and dimension mismatch is rejected at startup
- **WHEN** the configured embedding model and configured dimension are inconsistent
- **THEN** startup SHALL fail with a diagnostic naming both values rather than continue

### Requirement: The configured dimension is a boot-time contract, not a runtime toggle
The configured embedding dimension SHALL be read once when the process starts. Changing it SHALL be treated as a
re-embedding operation over all stored vectors, not as a configuration change applied to existing data.

#### Scenario: The value is fixed for the process lifetime
- **WHEN** the configured dimension is changed while a process is running
- **THEN** the running process SHALL continue to use the value read at start

#### Scenario: A changed dimension against stored vectors is refused
- **WHEN** the configured dimension differs from the width of vectors already stored
- **THEN** the system SHALL refuse to write new vectors and SHALL report that a re-embedding operation is required

### Requirement: Embedding requests declare their task type
Every embedding request SHALL declare whether it is embedding a search query or a stored document, and that
declaration SHALL reach the provider. Query-side and document-side embeddings SHALL NOT be produced without a
declared, distinct task type.

#### Scenario: Document embeddings declare the document task type
- **WHEN** chunk text is embedded for storage
- **THEN** the request SHALL declare the document task type

#### Scenario: Query embeddings declare the query task type
- **WHEN** a search query is embedded
- **THEN** the request SHALL declare the query task type

#### Scenario: No embedding request omits the task type
- **WHEN** any embedding request is issued
- **THEN** it SHALL carry an explicit task type

### Requirement: A failed embedding raises and never substitutes a placeholder vector
When embedding fails, the system SHALL raise a typed failure. It SHALL NOT return a zero vector or any other
placeholder, and it SHALL NOT persist a placeholder vector as a chunk embedding.

#### Scenario: Provider failure raises
- **WHEN** the embedding provider fails for a text
- **THEN** the embedding path SHALL raise a typed failure naming the operation

#### Scenario: No placeholder vector is persisted
- **WHEN** an embedding fails during ingestion
- **THEN** no chunk record SHALL be written with a placeholder vector, and the document SHALL record the failure

### Requirement: A dimension mismatch is reported as a diagnostic, not a secondary error
When a returned vector's width differs from the configured dimension, the system SHALL emit a diagnostic naming
both widths. Emitting that diagnostic SHALL NOT itself raise.

#### Scenario: Mismatch emits a warning
- **WHEN** a provider returns a vector whose width differs from the configured dimension
- **THEN** the system SHALL emit a warning naming the expected and actual widths

#### Scenario: The diagnostic path does not raise
- **WHEN** the mismatch diagnostic is emitted
- **THEN** no attribute or type error SHALL be raised by the diagnostic itself

### Requirement: Embeddings are cached in one cross-process cache keyed by content digest
The system SHALL cache embeddings in a single cache shared across processes, keyed by a digest of the text
together with the model and task type, with a documented expiry. There SHALL be exactly one embedding cache
implementation.

#### Scenario: A repeated text is not re-embedded
- **WHEN** the same text is embedded twice with the same model and task type
- **THEN** the provider SHALL be called once and the second call SHALL be served from the cache

#### Scenario: The cache is visible to a second process
- **WHEN** one process caches an embedding and another process embeds the same text
- **THEN** the second process SHALL be served from the cache

#### Scenario: Task type is part of the cache identity
- **WHEN** the same text is embedded once as a query and once as a document
- **THEN** the two results SHALL occupy distinct cache entries

### Requirement: Normalisation is uniform for every vector stored in one column
The system SHALL apply the same vector normalisation convention to every vector written to a given column, and
that convention SHALL be recorded. Vectors produced under different conventions SHALL NOT be mixed in one column.

#### Scenario: All chunk vectors share one convention
- **WHEN** chunk embeddings are written by any path
- **THEN** every vector in that column SHALL have been produced under the same normalisation convention

#### Scenario: The convention is recorded
- **WHEN** the normalisation convention for a column is queried by a reader of the design record
- **THEN** it SHALL be stated explicitly rather than inferred from behaviour

### Requirement: Batched embedding is supported at a documented batch size
The embedding path SHALL support embedding a batch of texts in one provider call, and ingestion SHALL use the
batched form at a documented batch size rather than one text per call.

#### Scenario: Ingestion embeds in batches
- **WHEN** a document produces more chunks than the batch size
- **THEN** the provider SHALL be called once per batch rather than once per chunk

#### Scenario: Single-text embedding remains available
- **WHEN** a single query is embedded
- **THEN** the embedding path SHALL support the single-text form without constructing a batch of one per call site
