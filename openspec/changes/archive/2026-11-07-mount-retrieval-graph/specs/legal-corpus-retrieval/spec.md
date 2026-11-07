## ADDED Requirements

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
