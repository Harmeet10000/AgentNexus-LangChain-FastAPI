## Purpose

Guarantee that the schema migration chain has exactly one head, that the target relational schema is defined in
full by exactly one authoritative revision, and that every relation the running application reads or writes
actually exists after the deployed database is upgraded.

## ADDED Requirements

### Requirement: The migration chain SHALL expose exactly one head

The revision graph SHALL present a single head. Where two lineages branched, they SHALL be joined by a revision
that has both as parents, so that upgrading, rendering and schema-comparison commands have an unambiguous target.

#### Scenario: Heads are enumerated

- **WHEN** the migration chain's heads are enumerated
- **THEN** exactly one head SHALL be reported

#### Scenario: Rendering an upgrade to head

- **WHEN** an upgrade to head is rendered without applying it
- **THEN** the render SHALL succeed
- **AND** it SHALL NOT fail because more than one head is present
- **AND** the render SHALL be understood to start from base and to emit every revision in the chain, since it has no
  database from which to read a recorded version — so its success proves single-head resolution and nothing about
  which revisions the deployed database still needs

#### Scenario: The historical branch point is preserved

- **WHEN** the revision graph is inspected after the join
- **THEN** the original branch point SHALL still be reachable in the history
- **AND** no pre-existing revision's identity or parentage SHALL have been rewritten

### Requirement: Every committed migration invocation SHALL name an unambiguous target

A committed command, build target, workflow step or documented instruction that upgrades the chain SHALL name a target
that resolves to exactly one revision. While more than one head exists, the singular head keyword does not resolve and
the invocation fails before doing any work; the plural form is the only well-defined target.

Single-head resolution SHALL be treated as a checked invariant rather than a one-time repair, because a chain that
forks again silently breaks every committed invocation that names the singular form.

#### Scenario: A singular target while the chain forks

- **WHEN** an upgrade names the singular head keyword and more than one head exists
- **THEN** the command SHALL fail reporting that multiple head revisions are present
- **AND** it SHALL be treated as a broken invocation rather than as a chain that merely needs a target chosen at
  run time

#### Scenario: Committed invocations are enumerated

- **WHEN** every committed invocation that upgrades the chain is enumerated — build targets, continuous-integration
  steps and documented instructions alike
- **THEN** each SHALL resolve to exactly one revision once the heads are joined
- **AND** any that still cannot resolve SHALL be corrected in the same change that joins the heads

#### Scenario: A future fork

- **WHEN** a later change introduces a second head
- **THEN** the invariant SHALL be treated as violated at that point, not at the point where an invocation next fails
- **AND** the check that detects it SHALL be the head enumeration, not the success of an upgrade

### Requirement: Exactly one revision SHALL be authoritative for the target schema

A single revision SHALL define the target relational schema in full: the unified document and chunk relations
with their uniqueness constraints, their vector, keyword and fuzzy retrieval indexes, the extensions those
indexes require, and the event-outbox relations. Anyone reading that one revision SHALL be able to see the whole
target shape without reconstructing it from earlier revisions.

That revision SHALL be idempotent and non-destructive: applying it to a database where an earlier revision
already created part of the target schema SHALL converge on the same final shape, SHALL NOT fail, and SHALL NOT
drop or truncate an existing relation.

#### Scenario: Applied to a database that has none of the target relations

- **WHEN** the authoritative revision is applied to a database that holds none of the target relations
- **THEN** the document, chunk and event-outbox relations SHALL exist afterwards with the target shape,
  constraints and indexes

#### Scenario: Applied to a database where an earlier revision already created the document relations

- **WHEN** the authoritative revision is applied to a database in which an earlier revision already created the
  document and chunk relations
- **THEN** the revision SHALL complete without error
- **AND** every row already stored in those relations SHALL still be present afterwards
- **AND** the final shape SHALL match the target, including any column the earlier revision omitted

#### Scenario: Extensions the target indexes depend on

- **WHEN** the authoritative revision is applied
- **THEN** the extensions required by the vector, keyword-ranking and fuzzy-matching indexes SHALL be created if
  they are not already installed
- **AND** applying the revision to a database where they are already installed SHALL NOT fail

#### Scenario: Ordering within the revision

- **WHEN** the authoritative revision is applied
- **THEN** the event-outbox relations SHALL be created before the document schema, so that a failure in the
  larger half cannot prevent the repair of the public endpoints that depend on the outbox

### Requirement: A relation whose creating revision is recorded as applied SHALL be created by a forward revision

Where the recorded version claims a revision that never executed, the relations that revision would have created are
unreachable by any upgrade, permanently, because upgrading skips revisions the recorded version already claims.
Absence combined with "already applied" does not resolve itself over time.

Such relations SHALL be created by a **forward** revision written idempotently. The recorded version SHALL NOT be
rewound below the falsely-claimed revisions in order to re-run them. Where a falsely-claimed revision cannot execute at
all — because it operates on a relation no revision creates — rewinding is not merely slower but non-terminating, and
the change record SHALL name that revision and the relation it presupposes.

Only relations on a live read or write path, as *live* is defined by this specification, SHALL be repaired. A relation
whose only readers are code a later change retargets SHALL NOT be created, and its exclusion SHALL name the change
that owns it.

#### Scenario: A relation absent while its creating revision is recorded as applied

- **WHEN** a relation a live path names is absent while the revision that would create it is recorded as applied
- **THEN** it SHALL be created by a forward revision
- **AND** the forward revision SHALL be idempotent, so it converges both on an instance where another revision already
  created part of the target schema and on an instance where nothing exists

#### Scenario: Rewinding the recorded version to re-run the skipped revisions

- **WHEN** rewinding the recorded version below the falsely-claimed revisions is considered as the repair
- **THEN** it SHALL be rejected
- **AND** the reason SHALL name the revision that cannot execute and the relation it presupposes but no revision
  creates, so the rejection is not read as a stylistic preference

#### Scenario: A relation the recorded version claims whose readers a later change retargets

- **WHEN** a relation is absent, its creating revision is recorded as applied, and its only readers are code a later
  sequenced change retargets or deletes
- **THEN** it SHALL NOT be created by the forward repair
- **AND** the change that owns its readers SHALL be named, so the omission is distinguishable from an oversight

#### Scenario: The revision that will actually execute

- **WHEN** the effect of upgrading the deployed database is described
- **THEN** the description SHALL be derived from the recorded version's ancestry, not from the revision graph alone
- **AND** the revisions the upgrade will actually execute SHALL be identified by name, because the graph and the
  recorded version disagree and only the disagreement explains what happens

### Requirement: Every relation on a live read or write path SHALL exist after an upgrade

Upgrading the deployed database to head SHALL leave no live read or write path pointing at a relation that does
not exist. In particular the event-outbox relations SHALL exist, because mounted public endpoints write to them
inside the same transaction that mutates user state.

**"Live" is defined for this requirement as: named by a code path reachable from a route mounted on a published API
version, through code that is not itself scheduled for deletion or for retargeting by the sequenced changes this
change is the first of.** The definition is stated because without it the requirement is unsatisfiable: raw SQL
elsewhere in the repository names relations this change deliberately does not create, and an implementer reading
"live" loosely would either have to create them or invent their own definition.

Resolved against the repository at authoring time, "live" names exactly these relations, and the requirement SHALL be
read as applying to this set:

- `outbox_events` and `dead_letter_events` — written by two mounted public endpoints and by the document upload path,
  and read by the relay.
- `documents` and `chunks` — the unified relations, reached from the mounted document router.
- The billing and audit relations, which already exist and are unaffected.

Deliberately **outside** the set, each with its owner named:

- `search_documents` and `search_chunks` — named only by the search repository's raw SQL, whose relation literals the
  search-consolidation change retargets onto the unified relations. Creating them here would create relations that
  change is about to strand.
- `clauses` and `parent_documents` — named by the knowledge-base ingestion nodes and the search repository. Their fate
  is decided by the same later change, and this change creates no DDL for them, because DDL with no surviving reader
  is the mirror image of the defect this requirement exists to close.

#### Scenario: A public endpoint that records a domain event

- **WHEN** a mounted public endpoint that records a domain event is called after the deployed database has been
  upgraded to head
- **THEN** the event row SHALL be recorded and the endpoint SHALL return its normal response
- **AND** it SHALL NOT fail with an undefined-relation error after having already persisted user state

#### Scenario: The event relay's startup scan

- **WHEN** the application starts after the upgrade and its event relay scans for unpublished events
- **THEN** the relation it scans SHALL exist
- **AND** the scan SHALL complete rather than being abandoned by its degradation handler

#### Scenario: No live path is left pointing at a missing relation

- **WHEN** the relations named by live read and write paths, as "live" is defined by this requirement, are compared
  against the upgraded database
- **THEN** every such relation SHALL be present

#### Scenario: A relation named only by a path a later change retargets

- **WHEN** a relation is named only by code a later sequenced change retargets or deletes
- **THEN** it SHALL NOT be created by this change
- **AND** the exclusion SHALL be recorded with the name of the change that owns it, so the omission is not read as
  an oversight


### Requirement: Stored chunks SHALL record when they were last written

The chunk relation SHALL carry a last-written timestamp that is never null and is supplied by the database when
a writer does not set it. Without it, a re-embedding campaign cannot distinguish a current-generation embedding
from one carried over from an earlier generation.

#### Scenario: The chunk relation exposes a last-written timestamp

- **WHEN** the deployed database has been upgraded to head
- **THEN** the chunk relation SHALL have a non-null `updated_at` timestamp column carrying a time zone
- **AND** inserting a chunk without supplying that column SHALL succeed, with the database supplying the value

### Requirement: The authoritative revision SHALL NOT claim relations an earlier revision already claims

Reversing the authoritative revision SHALL NOT remove relations whose creation an earlier revision also claims,
because that earlier revision's own reversal already removes them. A reversal that dropped them twice would fail
on a relation that no longer exists and would corrupt the earlier revision's contract.

#### Scenario: Reversing the authoritative revision

- **WHEN** the authoritative revision is reversed
- **THEN** the event-outbox relations SHALL be left in place
- **AND** the reversal SHALL state why, so the next reader does not read the omission as an oversight

### Requirement: The authoritative revision's own rendering SHALL NOT create a relation an earlier revision creates

Distinct from reversal: what the authoritative revision **creates** SHALL not duplicate what an earlier revision in
the chain creates under a different name. The relations the superseded search branch declares are that branch's to
create; the authoritative revision creates the unified relations and the event-outbox relations and nothing that
duplicates them.

This property SHALL be asserted over the SQL that the authoritative revision **alone** emits, rendered as a range
whose start is its own parent. It SHALL NOT be asserted over a render of the whole chain from base: an offline render
has no database from which to read a recorded version, so it always starts at base and always emits every revision in
the chain, including the superseded ones. Any assertion phrased against a from-base render measures the chain's
history rather than this revision's content, and cannot pass.

#### Scenario: Rendering the authoritative revision alone

- **WHEN** the SQL of the authoritative revision alone is rendered, as a range starting at its parent
- **THEN** it SHALL contain no statement creating a relation belonging to the superseded search branch
- **AND** it SHALL contain the statements creating the event-outbox relations and the unified document and chunk
  relations

#### Scenario: The same assertion made over a from-base render

- **WHEN** the same property is asserted over a render of the entire chain from base
- **THEN** the assertion SHALL be rejected as unmeasurable, because a from-base render emits every revision in the
  chain regardless of what any database currently holds

### Requirement: Retrieval indexes SHALL be created under the exact names the query text names

The keyword-ranking operator this repository uses takes **the index name as a literal SQL argument**. An index of the
correct shape created under a different name therefore matches nothing, raises no error, and returns zero rows. Index
naming for the retrieval indexes is consequently a cross-layer contract between the migration and the query text, not
a migration-local convention, and it SHALL be treated as one.

The authoritative revision SHALL create the retrieval indexes on the relations it creates under exact, recorded
names, and those names SHALL NOT be changed without changing every query literal and constant that names them in the
same commit. A conforming implementation SHALL NOT choose its own index names.

Where a query literal names an index on a relation this change deliberately does not create, that index SHALL NOT be
created here, and the change that retargets the literal SHALL be named. Retargeting those literals is **not** this
change's work, and this change SHALL NOT edit them.

#### Scenario: The keyword-ranking index on the relation this change creates

- **WHEN** the authoritative revision is applied
- **THEN** a keyword-ranking index SHALL exist on the chunk relation's generated search column, under the exact name
  already used by the revision that declares that relation, so that the two converge instead of producing two
  differently-named indexes of the same shape
- **AND** the vector index and the fuzzy trigram index on that relation SHALL likewise be created under those exact
  recorded names

#### Scenario: An index of the correct shape under a different name

- **WHEN** a retrieval index is created with the correct access method, column and operator class but a different name
  from the one the query text names
- **THEN** the implementation SHALL be treated as non-conforming
- **AND** the reason SHALL be recorded as silent: keyword ranking returns no rows and reports no error, so the defect
  is invisible to any check that only inspects index shape

#### Scenario: Index names a query literal pins on a relation this change does not create

- **WHEN** a query literal names a keyword-ranking index on a relation this change deliberately does not create
- **THEN** that index SHALL NOT be created by this change
- **AND** the change that retargets the literal onto the unified relation SHALL be named as its owner
- **AND** this change SHALL NOT edit the literal itself, so that ownership of that edit is not split between two
  changes

#### Scenario: Renaming a retrieval index later

- **WHEN** a rename of a retrieval index is proposed
- **THEN** it SHALL be treated as a change to the query contract, requiring every naming query literal and constant
  to change in the same commit
- **AND** it SHALL NOT be treated as an internal migration detail


### Requirement: A fresh environment SHALL have a documented, repeatable route to the target schema

Some revisions are recorded as applied in the deployed database while having created nothing, and they SHALL NOT
be rewritten. At least one of them cannot execute at all: it operates throughout on a relation no revision creates and
no model declares, so an upgrade that reaches it aborts with an undefined-relation error. Because of that, a plain
upgrade from an empty database is not a supported route to the target schema — and it is not made supported by
patience, retries, or a different starting point. The change record SHALL therefore identify those revisions by name,
SHALL state which of them is unrunnable and which relation it presupposes, and SHALL document one repeatable procedure
that brings a fresh environment to the target schema without editing any existing revision, naming the revisions the
procedure deliberately skips and what each of them would have created.

#### Scenario: Preparing a new environment

- **WHEN** an operator prepares a new environment from an empty database
- **THEN** a documented procedure SHALL bring that database to the target schema
- **AND** the procedure SHALL name the revisions it deliberately skips

#### Scenario: A revision that cannot execute against any database

- **WHEN** a revision in the chain operates on a relation that no revision creates and no model declares
- **THEN** the change record SHALL name that revision and the relation it presupposes
- **AND** the procedure SHALL skip it rather than attempt it, because the failure is unconditional rather than
  environment-dependent
- **AND** creating the presupposed relation SHALL NOT be adopted as the remedy where its only readers are code a later
  change retargets

#### Scenario: Extension preconditions of the procedure

- **WHEN** the documented procedure is followed on an instance whose base image installs none of the required
  extensions
- **THEN** the procedure SHALL state which extensions must exist before the first revision it does **not** skip is
  applied, because a revision the procedure runs builds an index whose access method that revision does not install
- **AND** the procedure SHALL reach the target schema without editing any existing revision


#### Scenario: Reversing below the joined head

- **WHEN** reversing the chain below the joined head is considered
- **THEN** the change record SHALL identify it as unsupported, because the reversals of the revisions recorded
  as applied would drop relations that were never created

### Requirement: Every extension the target schema depends on SHALL be created by the revision that depends on it

No index, operator class or search function in the target schema may depend on an extension that the chain never
creates. Ambient availability on a particular managed instance SHALL NOT be treated as a guarantee.

#### Scenario: All four required extensions are created explicitly

- **WHEN** the authoritative revision is applied
- **THEN** it SHALL create the `vector`, `vectorscale`, `pg_trgm` and `pg_textsearch` extensions explicitly, each
  conditionally so that an existing installation is left in place
- **AND** it SHALL NOT depend on any of the four already being present in the target instance's base image
- **AND** each extension SHALL be created before the first index, operator class or function that requires it

#### Scenario: An extension present only by accident of the hosting image

- **WHEN** an index built by a revision in the chain depends on an extension that no revision preceding it creates
- **THEN** that condition SHALL be treated as a defect in the chain rather than as a satisfied dependency
- **AND** the authoritative revision SHALL create that extension, so that every index the authoritative revision
  itself builds reproduces on an instance whose base image differs

#### Scenario: The defect sits in a revision ordered ahead of the authoritative one

- **WHEN** the revision that is unstamped and therefore executes on the next upgrade builds a vector index whose
  access method is supplied by an extension that revision does not create, and no revision before it creates it either
- **THEN** the change record SHALL identify that revision by name and SHALL state that the authoritative revision
  cannot repair it, because the authoritative revision is ordered **after** it and a failure there aborts the upgrade
  before the outbox repair runs
- **AND** the deployed upgrade SHALL be treated as depending on that extension being present beforehand, whether by
  ambient installation on the target instance or by creating it as an explicit precondition of the upgrade
- **AND** the condition SHALL NOT be recorded as "no revision in the chain ever creates this extension", which is
  false: an earlier, superseded revision does create it — the defect is that the one which will actually execute does
  not


#### Scenario: The deployment role can create the required extensions

- **WHEN** the upgrade is applied by the deployment role rather than by a superuser
- **THEN** creation of all four extensions SHALL succeed
- **AND** the fuzzy-matching retrieval branch SHALL be built rather than omitted, so the fused ranking combines
  vector, keyword-ranking and fuzzy-matching results
