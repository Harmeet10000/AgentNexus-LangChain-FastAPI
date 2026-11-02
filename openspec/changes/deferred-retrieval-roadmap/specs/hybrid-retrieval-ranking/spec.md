## Purpose

Admit graph-traversal results as a fourth fused retrieval branch — only at the
reserved registration point, only when its contribution is measurable.

## ADDED Requirements

### Requirement: A graph-traversal branch fuses with a measured weight when admitted

The fusion step SHALL accept a fourth branch carrying graph-traversal results, fused
through the same weighted reciprocal-rank mechanism as the existing branches, at the
registration point reserved by the retrieval-engine cutover. The branch SHALL be
admitted only when relationship edges exist for the in-scope corpus and its weight
SHALL be set from measured contribution on the retrieval baseline, not asserted.

#### Scenario: The branch is admitted with evidence

- **WHEN** the fourth branch is enabled for a corpus
- **THEN** the corpus has relationship edges and a recorded baseline comparison
  supporting the branch's weight
- **AND** the fusion uses the same weighting mechanism and attribution as the
  existing three branches

#### Scenario: A graphless corpus is unaffected

- **WHEN** the corpus has no relationship edges
- **THEN** fusion proceeds exactly as the three-branch behaviour with no fourth
  branch, no empty-branch success, and no changed ranking

### Requirement: Branch absence is silent in mechanism, honest in telemetry

When the fourth branch is enabled but contributes nothing for a query, fusion SHALL
proceed over the remaining branches unchanged, and the decision event SHALL record
the branch's participation distinctly from the branches that matched.

#### Scenario: Empty traversal does not dilute ranking

- **WHEN** the graph branch returns no results for a query
- **THEN** the fused ranking equals the three-branch ranking for that query
- **AND** the decision event names the graph branch as participating-but-empty
