## Purpose

Define the relationship-graph retrieval branch as an optional fourth fused branch in the documents
ask path: it enters the retrieval machine by registration in the existing branch registry, its
fusion weight is a measurement rather than a constant-named opinion, and it ships disabled until
that measurement exists. The three mandatory fused modes defined by `document-retrieval-schema`
are untouched — this capability is additive and opt-in.

## ADDED Requirements

### Requirement: The graph branch enters fusion by registration, not a second fusion implementation

The relationship-graph branch SHALL be registered in the ask path's branch registry (name paired
with its runnable), and the existing gather, pairing, and reciprocal-rank-fusion code SHALL be
reused unchanged. A second fusion or branch-orchestration implementation for the graph branch
SHALL NOT exist.

#### Scenario: Adding the branch leaves the fusion untouched

- **WHEN** the graph branch is registered
- **THEN** the gather/zip/fusion code path SHALL require no modification beyond the registration,
  and the branch's name SHALL travel with its runnable so a failure attributes to the branch

#### Scenario: The branch returns fusable output

- **WHEN** the graph branch executes
- **THEN** it SHALL return ranked chunk identifiers over the unified chunk store, tenant-scoped,
  in the same shape the other branches return, so the fusion combines them without special-casing

### Requirement: The graph branch ships disabled until its weight is measured

The graph branch SHALL be controlled by a setting that defaults to off. Its fusion weight SHALL be
derived from a harness report on the golden corpus, and the branch SHALL NOT flip to default-on
except by a change that cites that report, per the `retrieval-evaluation` baseline requirement.

#### Scenario: Default deployments do not run the branch

- **WHEN** the setting is at its default
- **THEN** the ask path SHALL fuse the declared mandatory modes only, and no relationship-graph
  read SHALL execute for the fusion

#### Scenario: Enabling is a measured decision

- **WHEN** a change proposes flipping the branch on by default
- **THEN** it SHALL cite a harness report on the golden corpus measuring the branch's contribution
  (recall/nDCG delta with the per-branch attribution rows), and the weight in the setting SHALL
  match the measured weight

#### Scenario: Disabled is not a provisioning failure

- **WHEN** the setting disables the graph branch
- **THEN** the ask path SHALL serve normally, and this SHALL NOT be reported through the missing-
  capability provisioning-failure surface that the mandatory modes use

### Requirement: An enabled graph branch fails loudly and is observable

When enabled, a failure of the graph branch SHALL fail the retrieval request naming the branch, and
the branch's per-request contribution (branches executed, identifiers contributed, weight applied)
SHALL be observable in the response path's diagnostics rather than silently absorbed.

#### Scenario: The enabled branch fails

- **WHEN** the graph branch raises while enabled
- **THEN** the retrieval request SHALL fail naming the graph branch, and the system SHALL NOT
  return a result fused from only the branches that succeeded

#### Scenario: The branch's contribution is visible

- **WHEN** an ask executes with the branch enabled
- **THEN** the execution diagnostics SHALL record that the graph branch ran and how many ranked
  identifiers it contributed to the fusion
