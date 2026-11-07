# legal-corpus-retrieval Specification

## Purpose

Defines where statute and precedent retrieval get their evidence: the unified document corpus rather than relations
that were never created, using the application's one ranked full-text and fusion path, with statute identity
attributes addressable and a truthful answer whenever the corpus cannot be reached.

## Requirements

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
the application. An agent tool SHALL NOT introduce a second ranking or fusion implementation. No caller SHALL introduce
a second ranking or fusion implementation, whether in an agent tool, a repository method, a service, or a graph node.
Two callers issuing the same query with the same filters SHALL receive the same ranked identifiers in the same order.
The constants governing fusion SHALL have exactly one definition, and SHALL NOT be restated as literals at a call site.

#### Scenario: Precedent search uses the shared ranked retrieval path

- **WHEN** precedent search ranks textual matches
- **THEN** it SHALL use the application's shared ranked full-text retrieval path

#### Scenario: Combining ranked lists uses the shared fusion

- **WHEN** precedent search combines more than one ranked result list
- **THEN** it SHALL use the application's shared fusion of ranked lists

#### Scenario: Two doors return one answer

- **WHEN** the retrieval graph and the search endpoint issue the same query with the same filters
- **THEN** both SHALL return the same ranked chunk identifiers in the same order

#### Scenario: A repository method is not an exemption

- **WHEN** ranking or fusion is performed inside a repository method
- **THEN** it SHALL be the shared implementation, and a second ranking implementation reached through the
  data-access layer SHALL be treated as the violation an agent tool's would be

#### Scenario: Fusion constants have one home

- **WHEN** a fusion constant is used
- **THEN** its value SHALL be read from its single named definition, and SHALL NOT appear as a numeric
  literal in a query or a call site

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

### Requirement: The ask endpoints are served by the compiled retrieval graph

`/search/ask` and `/legal/ask` SHALL be served by the compiled retrieval graph, and no inline
duplicate of the plan→retrieve→rerank→grade→generate sequence SHALL remain on the served path once
the parity gate is green. Each endpoint SHALL preserve its request-scoped evidence policy —
`mechanically_verified` on `/search/ask`, `graph_indexed_only` on `/legal/ask` — and the served path
SHALL record document reads through the access-audit recording exactly as the implementation it
replaces did.

#### Scenario: Both endpoints answer through one implementation

- **WHEN** either ask endpoint answers a query
- **THEN** the answer SHALL have been produced by the compiled retrieval graph, and no second inline
  implementation of the ask sequence SHALL remain in the serving code

#### Scenario: The legal evidence policy survives the cutover

- **WHEN** `/legal/ask` answers
- **THEN** retrieval SHALL be filtered by the graph-indexed evidence requirement and the response
  SHALL carry `evidence_policy` of `graph_indexed_only`

#### Scenario: Access recording is preserved

- **WHEN** an ask completes through the compiled graph
- **THEN** the document reads it performed SHALL be recorded by the same access-audit recording the
  replaced path used, with the requested scope and outcome intact

#### Scenario: An empty authorized scope returns the fallback envelope

- **WHEN** a request arrives with an explicit empty authorized document scope
- **THEN** the response SHALL be the grounded fallback answer with no citations and the
  request's evidence policy, and no unscoped retrieval SHALL run

#### Scenario: The graph is not provisioned in the serving process

- **WHEN** the compiled retrieval graph was not provisioned at startup
- **THEN** the ask endpoints SHALL fail closed with a typed unavailability rather than falling back
  to any other answering path

### Requirement: Answer cache keys are version-aware

The ask answer cache SHALL derive its key from the query, the tenant, the authorized document
scope, the resolved document version of every scoped document, and the canonical pipeline version.
A cached answer SHALL NOT be served after any scoped document's version or the pipeline version has
changed, and the key derivation SHALL match the derivation the replaced inline path defined.

#### Scenario: A re-ingested document does not serve stale answers

- **WHEN** a scoped document's version changes and the same query is asked again
- **THEN** the derived cache key SHALL differ and a fresh answer SHALL be produced rather than the
  cached one

#### Scenario: A pipeline bump invalidates cached answers

- **WHEN** the canonical pipeline version changes
- **THEN** every previously cached answer key SHALL be unreachable under the new derivation

#### Scenario: Cache-key derivation is pinned by parity

- **WHEN** the parity suite derives answer cache keys through the graph path and the recorded
  inline contract for the same query and scope
- **THEN** the derivations SHALL be identical

### Requirement: Ask responses carry truthful coverage and warnings

The graph-served ask response SHALL report the same coverage and warning envelope as the contract
it replaces: unavailable authorized documents counted in coverage, a scope-unavailable warning when
an authorized document's version or contract scope could not be resolved, quality warnings merged
and deduplicated, and graph-indexed counts derived from the hydrated rows rather than asserted.

#### Scenario: Unavailable scope documents are disclosed

- **WHEN** an authorized document's version or contract scope cannot be resolved
- **THEN** the response SHALL count it in coverage as graph-unavailable and SHALL carry the
  scope-unavailable quality warning

#### Scenario: Warnings merge without duplication

- **WHEN** retrieval produces quality warnings across stages
- **THEN** the response SHALL carry them merged and deduplicated as the replaced path did
