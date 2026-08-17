## Purpose

Define how extracted entities acquire one stable identity before anything is written to the knowledge graph. This
is the only irreversible contract in the ingestion change: once variant surface forms become separate nodes, no
later pass can merge them, because the disambiguating extraction context has already been discarded.

## ADDED Requirements

### Requirement: Extracted entities are canonicalised to a stable identity before any graph write
The system SHALL resolve every extracted entity to a stable canonical identity before it is written to the
knowledge graph. Canonicalisation SHALL be deterministic for a given surface form, and the raw surface form SHALL
be retained as an attribute of the entity rather than discarded.

#### Scenario: Variant surface forms resolve to one identity
- **WHEN** the same party appears in a document as differing surface forms that differ only in punctuation, casing, or a corporate suffix
- **THEN** all of them SHALL resolve to the same canonical identity

#### Scenario: Distinct parties do not collide
- **WHEN** two genuinely different parties are extracted from the same document
- **THEN** they SHALL resolve to different canonical identities

#### Scenario: The raw surface form is retained
- **WHEN** an entity is written to the knowledge graph under its canonical identity
- **THEN** the surface form as it appeared in the source document SHALL be retained as an attribute for audit

#### Scenario: Canonicalisation is deterministic
- **WHEN** the same surface form is canonicalised in two separate processes
- **THEN** both SHALL produce the same canonical identity

### Requirement: Every knowledge-graph write path keys on the canonical identity
Every entity write, relationship write, and episode write SHALL be keyed on the canonical identity. No graph write
path SHALL key on raw extracted text.

#### Scenario: Entity writes are keyed canonically
- **WHEN** an extracted entity is written to the knowledge graph
- **THEN** its node identity SHALL be the canonical identity

#### Scenario: Relationship writes are keyed canonically
- **WHEN** a relationship between two extracted entities is written
- **THEN** both endpoints SHALL be referenced by canonical identity

#### Scenario: Episode writes are keyed canonically
- **WHEN** a chunk-level episode carrying entity references is written
- **THEN** its entity references SHALL be canonical identities

#### Scenario: Repeated variants produce one node
- **WHEN** two variant surface forms of the same party are written from the same document
- **THEN** the knowledge graph SHALL hold exactly one node for that party

### Requirement: The canonical identity is the idempotency key for replayed writes
The canonical identity SHALL serve as the idempotency key for knowledge-graph writes, so that a replayed stage
writing the same entity or relationship produces no duplicate. The system SHALL NOT maintain a second, separate
idempotency mechanism for the same writes.

#### Scenario: A replayed stage produces no duplicate
- **WHEN** a stage that has already written entities is replayed after a failure
- **THEN** the knowledge graph SHALL hold the same node and edge set as after the first execution

#### Scenario: One idempotency mechanism governs graph writes
- **WHEN** a graph write's idempotency is determined
- **THEN** it SHALL be determined by the canonical identity rather than by a separately maintained key

### Requirement: Graph writes are refused when canonicalisation is unavailable
When canonicalisation cannot be performed for an extracted entity, the system SHALL refuse the graph write for
that document and record a failure. It SHALL NOT fall back to writing raw extracted text as an identity.

#### Scenario: Unavailable canonicalisation refuses the write
- **WHEN** canonicalisation fails for an extracted entity
- **THEN** no entity, relationship, or episode SHALL be written for that document and the document SHALL record a terminal failure

#### Scenario: No raw-text fallback identity is used
- **WHEN** canonicalisation is unavailable
- **THEN** the system SHALL NOT write any node keyed on raw extracted text
