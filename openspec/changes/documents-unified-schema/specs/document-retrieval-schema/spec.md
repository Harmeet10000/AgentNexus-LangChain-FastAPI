## Purpose

Define one document store and one chunk store as the sole source of retrieval truth for the product: how a
document is identified per tenant, what provenance every stored row must carry, which retrieval modes the chunk
store supports and how their results are combined, how reads stay scoped to their owner, and the guarantee that
every database identifier the system names in a query actually exists.

## ADDED Requirements

### Requirement: Single retrieval source of truth

All document retrieval SHALL read from one unified document store and one unified chunk store. No other table
SHALL serve document or chunk retrieval, and no retrieval query SHALL reference a table that the project's
migrations do not create.

#### Scenario: A retrieval query targets the unified stores

- **WHEN** any hybrid, keyword, vector, fuzzy or graph-backed retrieval request executes
- **THEN** it SHALL read document and chunk content only from the unified document and chunk stores
- **AND** it SHALL NOT read from a superseded search-specific, clause-specific or vector-specific table

#### Scenario: No query names a table that no migration creates

- **WHEN** the source tree is scanned for retrieval query text
- **THEN** every table named in that query text SHALL be created by a migration in the project's migration
  history

#### Scenario: A second retrieval path is not introduced

- **WHEN** a new retrieval capability is added
- **THEN** it SHALL be served by the unified chunk store
- **AND** it SHALL NOT introduce an additional document or chunk store alongside it

### Requirement: Per-tenant document identity

Document identity SHALL be the pair of owning tenant and content digest. Byte-identical content ingested by two
different tenants SHALL be stored as two independent documents, each with its own chunk set. Identity SHALL NOT
be the content digest alone.

#### Scenario: Two tenants ingest byte-identical content

- **WHEN** tenant A and tenant B each ingest content with the same digest
- **THEN** two independent documents SHALL exist, one owned by each tenant
- **AND** each SHALL have its own complete chunk set
- **AND** neither tenant's ingest response SHALL disclose the other tenant's document identifier

#### Scenario: One tenant deletes a shared-content document

- **WHEN** tenant A deletes its document
- **THEN** tenant B's document with the same content digest SHALL remain retrievable
- **AND** tenant B's chunks SHALL remain retrievable

#### Scenario: The same tenant re-ingests identical content

- **WHEN** tenant A ingests content whose digest matches a document tenant A already owns
- **THEN** the existing document SHALL be reused rather than duplicated
- **AND** its chunks SHALL be replaced in place rather than appended

### Requirement: Mandatory ownership and object provenance

Every stored document SHALL carry an owning tenant and an immutable reference to the stored source object.
Every stored chunk SHALL carry an owning tenant. The system SHALL reject an ingestion request that cannot supply
both, and SHALL NOT substitute a shared owner, a sentinel owner, an empty reference, or a database-supplied
default.

#### Scenario: Ingestion without a resolvable owner is rejected

- **WHEN** an ingestion request arrives whose owning tenant cannot be resolved
- **THEN** the system SHALL reject the request with a client error before any document or chunk row is written
- **AND** it SHALL NOT assign a shared or sentinel owner in order to proceed

#### Scenario: Ingestion without a stored source object is rejected

- **WHEN** an ingestion request arrives for content that has not been persisted as an immutable source object
- **THEN** the system SHALL reject the request rather than store a document with an empty object reference

#### Scenario: The store provides no default for either value

- **WHEN** the document and chunk store definitions are inspected
- **THEN** the document owner, the document object reference and the chunk owner SHALL each be non-nullable
- **AND** none of them SHALL carry a database-supplied default value

### Requirement: Three rank-fused retrieval modes

Hybrid retrieval over the chunk store SHALL execute keyword-relevance matching, vector-similarity matching and
fuzzy character-similarity matching over the same searchable chunk text, and SHALL combine their results into a
single ordered result set by fusing rank positions rather than raw scores.

#### Scenario: A hybrid retrieval request runs all three modes

- **WHEN** a hybrid retrieval request executes
- **THEN** all three retrieval modes SHALL run against the chunk store's indexed searchable text
- **AND** their results SHALL be merged into one ordered result set by rank position
- **AND** a chunk returned by more than one mode SHALL appear once in the fused result

#### Scenario: One mode returns no rows

- **WHEN** one of the three retrieval modes returns no matching chunks
- **THEN** the fused result SHALL still contain the chunks returned by the remaining modes

#### Scenario: A mode's required database capability is absent

- **WHEN** the database does not provide an extension or index a retrieval mode requires
- **THEN** provisioning SHALL fail loudly with that missing capability named
- **AND** the system SHALL NOT silently serve a fused result from fewer modes than it declares

#### Scenario: One mode fails to execute

- **WHEN** one of the three retrieval modes fails to execute for any reason
- **THEN** the retrieval request SHALL fail, naming the mode that failed
- **AND** the system SHALL NOT return a result fused from only the modes that succeeded
- **AND** a mode that executes successfully and matches nothing SHALL NOT be treated as a failure

### Requirement: Tenant-scoped retrieval

Every retrieval path over the chunk store SHALL constrain its results to chunks owned by the requesting tenant.
A retrieval request SHALL NOT execute without an owning tenant.

#### Scenario: Results never cross a tenant boundary

- **WHEN** tenant A issues a retrieval request
- **THEN** every returned chunk SHALL be owned by tenant A
- **AND** no chunk owned by another tenant SHALL appear in the result or in the fused ranking that produced it

#### Scenario: An unscoped retrieval request is refused

- **WHEN** a retrieval request supplies no owning tenant
- **THEN** the system SHALL refuse the request rather than execute an unscoped scan of the chunk store

### Requirement: Every named database identifier exists

Every database index and every uniqueness constraint that the system names inside query text SHALL be created
by a migration, on a table that a migration creates. The project SHALL enforce this by an automated check that
does not require a database connection.

#### Scenario: A query names an index

- **WHEN** query text names a database index
- **THEN** a migration SHALL create an index of that name
- **AND** a migration SHALL create the table that index is defined on

#### Scenario: A conflict-resolution clause names a uniqueness constraint

- **WHEN** an insert-or-update operation names a uniqueness constraint to resolve conflicts against
- **THEN** that constraint SHALL be declared with the same name on the target table's definition

#### Scenario: A named identifier with no creating migration fails the build

- **WHEN** query text names an index or constraint that no migration creates
- **THEN** the project's automated checks SHALL fail and name the offending identifier
- **AND** the failure SHALL NOT depend on a reachable database

### Requirement: Chunk modification time

Every chunk SHALL record when its content or its embedding was last written, and that recorded time SHALL
advance whenever the chunk is rewritten. The value SHALL be written by the ingestion path itself, not left to
an update-only mechanism that a conflict-resolving insert bypasses.

#### Scenario: A newly stored chunk records a modification time

- **WHEN** a chunk is stored for the first time
- **THEN** it SHALL carry a non-null modification time

#### Scenario: A rewritten chunk advances its modification time

- **WHEN** an existing chunk is re-stored with different content or a different embedding
- **THEN** its recorded modification time SHALL be later than the value it held before the rewrite

#### Scenario: A conflict-resolving insert still advances the value

- **WHEN** a chunk is rewritten by an insert that resolves a uniqueness conflict into an update
- **THEN** the modification time SHALL still advance
- **AND** the value SHALL be supplied by the ingestion path rather than assumed from an update trigger

### Requirement: Exactly one derived searchable text per chunk

Each chunk SHALL expose exactly one searchable text representation, derived automatically by the store from that
chunk's own fields so it cannot drift from them. The chunk store SHALL NOT carry a second derived search
representation or an index over one that no retrieval path reads.

#### Scenario: Searchable text follows the chunk's content

- **WHEN** a chunk's content, preamble or classification changes
- **THEN** its searchable text SHALL reflect the new values without a separate write by the ingestion path

#### Scenario: No reader-less derived search column survives

- **WHEN** the chunk store definition is inspected
- **THEN** it SHALL contain no derived search column, and no index over one, that no retrieval path reads

### Requirement: Mounted document surface requires an authenticated owner

Every mounted document ingest or retrieval endpoint SHALL require an authenticated owner. Consolidating the
retrieval schema SHALL NOT make a previously unreachable ingest or retrieval endpoint reachable.

#### Scenario: The mounted route set gains nothing

- **WHEN** the mounted route set is enumerated after consolidation
- **THEN** it SHALL contain no document ingest or retrieval path that was unreachable before consolidation

#### Scenario: No mounted document endpoint resolves without an owner

- **WHEN** a mounted document ingest or retrieval endpoint is called without an authenticated owner
- **THEN** the request SHALL be refused
- **AND** the refusal SHALL be an explicit authorization failure, not an unhandled internal error

### Requirement: The authoritative schema migration creates only the unified stores

The project's authoritative schema migration SHALL create exactly one document store and one chunk store, and
SHALL NOT create a superseded search-specific document or chunk table. Outside migration history, no model,
query, task or test SHALL reference a superseded search-specific document or chunk table.

#### Scenario: The authoritative schema migration is rendered

- **WHEN** the authoritative schema migration is rendered to SQL
- **THEN** it SHALL contain exactly one statement creating the document store and one creating the chunk store
- **AND** it SHALL contain no statement creating a superseded search-specific document or chunk table

#### Scenario: The source tree names no superseded table

- **WHEN** the source tree and test suite are scanned outside migration history
- **THEN** no model, query, background task or test fixture SHALL name a superseded search-specific document or
  chunk table
