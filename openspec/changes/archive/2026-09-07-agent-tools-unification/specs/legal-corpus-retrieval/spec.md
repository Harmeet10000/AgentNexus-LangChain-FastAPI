## Purpose

Defines where statute and precedent retrieval get their evidence: the unified document corpus rather than relations
that were never created, using the application's one ranked full-text and fusion path, with statute identity
attributes addressable and a truthful answer whenever the corpus cannot be reached.

## ADDED Requirements

### Requirement: Retrieval targets only relations the migration history creates

Every retrieval query issued by an agent tool SHALL target relations that the application's migration history creates.
No agent tool SHALL query a relation that no migration defines.

#### Scenario: Statute retrieval targets the unified corpus

- **WHEN** a statute section is retrieved
- **THEN** the query SHALL resolve against the unified document corpus

#### Scenario: Precedent search targets the unified corpus

- **WHEN** precedent text is searched
- **THEN** the query SHALL resolve against the unified document corpus

#### Scenario: No tool queries a relation that no migration defines

- **WHEN** the agent tool layer issues any retrieval query
- **THEN** every relation it names SHALL be one the migration history creates

### Requirement: Statute identity attributes are addressable and efficiently retrievable

The corpus SHALL carry the attributes that identify a statutory provision — the instrument name, the section
reference, and the year — under a documented contract, and a point lookup on the instrument name and section
reference SHALL be served by an index rather than a full scan of the corpus.

**Coordination point.** This change ships no DDL. The carrier for these three attributes and the index that serves
the point lookup are provided by the change that owns the retrieval schema, under its ADR *"`documents` / `chunks`
is the sole retrieval schema"* (`openspec/changes/documents-unified-schema/adrs.md`, status **Accepted**), and are
normatively specified by its `document-retrieval-schema` capability. That ADR is the authority on the attribute
names and on whether they are columns or a typed metadata sub-object; this capability deliberately states the
requirement at the attribute level and names no column. The retrieval work in this change therefore *consumes* the
contract and is gated on it — no task in this change creates, alters, or indexes a relation.

**The dependency is directional, and it constrains proofs as well as order.** This requirement is **unsatisfiable until
the retrieval-schema migration lands**: the carrier relation and its index do not exist in the deployed database today.
Every scenario below is therefore verified **after** that migration, or by an import- or type-level check that does not
touch the database. In particular, a proof asserting an index-served lookup **as of today** would be unexecutable — the
relation it queries is absent — and must not be written. This is a repeated failure mode in this refactor, not a
hypothetical one, so the constraint is stated in the capability rather than left to the task list.

#### Scenario: A statute section is addressable by instrument and section reference

- **WHEN** a statute section is requested by instrument name and section reference
- **THEN** the corpus SHALL be able to identify the matching record from those attributes

#### Scenario: The newest applicable version is selected

- **WHEN** more than one record matches the requested instrument name and section reference
- **THEN** the most recent applicable year SHALL be returned

#### Scenario: The point lookup is index-served

- **WHEN** the statute point lookup executes
- **THEN** it SHALL be served by an index on the identifying attributes

#### Scenario: The attribute contract is resolved before the retarget is written

- **WHEN** the statute point lookup is implemented against the unified corpus
- **THEN** the attribute names it reads SHALL be the ones the retrieval-schema contract documents
- **AND** the implementation SHALL NOT introduce attribute names of its own

### Requirement: Ranked retrieval and fusion have a single implementation

Full-text ranked retrieval and the fusion of ranked result lists SHALL be performed by one implementation shared across
the application. An agent tool SHALL NOT introduce a second ranking or fusion implementation.

#### Scenario: Precedent search uses the shared ranked retrieval path

- **WHEN** precedent search ranks textual matches
- **THEN** it SHALL use the application's shared ranked full-text retrieval path

#### Scenario: Combining ranked lists uses the shared fusion

- **WHEN** precedent search combines more than one ranked result list
- **THEN** it SHALL use the application's shared fusion of ranked lists

### Requirement: An unreachable corpus is reported as unavailable

When statute or precedent retrieval cannot reach the corpus, the tool SHALL report unavailability rather than an empty
or negative retrieval outcome, and SHALL NOT characterise the requested material as nonexistent.

#### Scenario: Corpus unreachable during statute retrieval

- **WHEN** the corpus cannot be reached while retrieving a statute section
- **THEN** the result SHALL report unavailability
- **AND** SHALL NOT report that the section does not exist

#### Scenario: Corpus unreachable during precedent search

- **WHEN** the corpus cannot be reached while searching precedents
- **THEN** the result SHALL report unavailability
- **AND** the evidence completeness SHALL be reported as unknown
