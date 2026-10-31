## Purpose

Define the standing contract that gates re-ingestion exposure: the capability of re-running an
existing document through a newer pipeline version SHALL NOT exist until the semantics it depends
on are decided and written down. This capability is the decision record; the change that exposes
re-ingestion must satisfy every requirement here first.

## ADDED Requirements

### Requirement: Re-ingestion stays unexposed until the readiness contract is satisfied

Re-ingestion of an existing document under a new pipeline version SHALL NOT be exposed — not as a
product feature, not as an admin endpoint, and never as a model-callable agent tool — until every
requirement in this capability is satisfied. When exposure arrives, it SHALL be an approval-gated
administrative action (operator role, audited), not a general endpoint and not an agent
capability.

#### Scenario: No re-ingestion surface exists today

- **WHEN** the routes, toolsets, and agent tool registry are inspected
- **THEN** no re-ingestion endpoint, admin route, or model-callable tool SHALL be found

#### Scenario: A bulk re-index does not become an endpoint by accident

- **WHEN** the quarantined bulk-processing surface is ever revived for a corpus re-index
- **THEN** it SHALL go through this capability's contract as an approval-gated administrative
  action, and SHALL NOT be wired as a general or model-facing endpoint in passing

#### Scenario: Exposure cites its readiness

- **WHEN** a change proposes any re-ingestion surface
- **THEN** it SHALL demonstrate that every requirement in this capability is already satisfied, or
  land the missing ones in the same change

### Requirement: Graph edge semantics under re-ingestion are decided and enforced

The idempotent relationship writer SHALL define, in behaviour and not only in prose, what
re-ingesting a document means for its existing knowledge-graph edges: each edge is either
replaced, versioned, or superseded — exactly one of the three — and a re-ingested document SHALL
NOT produce duplicate edges, orphaned edges pointing at chunks that no longer resolve, or edges
silently left stale while the chunks they cite were replaced.

#### Scenario: A re-ingested document does not duplicate edges

- **WHEN** a document with existing relationship edges is re-ingested under a new pipeline version
  and the chosen semantics are exercised
- **THEN** the graph SHALL contain exactly the edges the chosen semantics prescribe — no second
  copy of an equivalent edge under a fresh identity

#### Scenario: No edge outlives its citation target silently

- **WHEN** re-ingestion replaces the chunks an existing edge cites
- **THEN** the edge SHALL be superseded or re-pointed per the chosen semantics, and never remain
  silently stale — a lookup through the edge SHALL resolve to chunk content that exists

#### Scenario: The semantics are one decision, stated where the writer implements it

- **WHEN** the writer's edge handling is reviewed
- **THEN** the replace/version/supersede decision SHALL be readable in the writer (a named policy,
  not an emergent side effect of idempotency keys)

### Requirement: Superseded chunk versions are retained while citations can still reach them

A re-ingestion that replaces chunks SHALL NOT delete a superseded chunk version while any open
evidence handle or any recorded citation can still resolve against it. Retention SHALL follow a
declared policy (how long, released by what event), and deletion SHALL be a policy decision,
never a side effect of the re-ingestion write.

#### Scenario: An open evidence handle survives a re-ingestion

- **WHEN** a document is re-ingested while an open evidence handle references one of its
  superseded chunks
- **THEN** the superseded chunk SHALL remain resolvable through that handle until the handle is
  closed or the retention policy releases it

#### Scenario: A recorded citation does not dangle

- **WHEN** a past answer's recorded citation names a chunk that a re-ingestion superseded
- **THEN** the cited chunk SHALL still resolve under the retention policy, and the citation SHALL
  NOT silently re-point at the replacement chunk's content

### Requirement: Chunk identity under a re-run is deterministic or explicitly re-mapped

Chunk identity across a re-ingestion SHALL follow one declared rule: either the chunk identifier
is a deterministic function of the document content and the pipeline version (identical inputs
reproduce identical `chunk_id`s), or the re-ingestion re-maps evidence handles from superseded to
replacement identifiers by an explicit, testable rule. Undeclared identifier churn that dangles
historical `EvidenceRef`s SHALL NOT ship.

#### Scenario: Identical inputs reproduce identical identifiers

- **WHEN** the same document content is processed twice under the same pipeline version
- **THEN** the produced chunk identifiers SHALL be identical between the runs

#### Scenario: A version change is the only legitimate churn

- **WHEN** chunk identifiers change between two runs
- **THEN** the pipeline version SHALL have changed, and the re-mapping rule SHALL account for
  every superseded identifier that an open evidence handle can name
