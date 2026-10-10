## Purpose

Defines where statute and precedent retrieval get their evidence: the unified document corpus rather than relations
that were never created, using the application's one ranked full-text and fusion path, with statute identity
attributes addressable and a truthful answer whenever the corpus cannot be reached.

## MODIFIED Requirements

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
