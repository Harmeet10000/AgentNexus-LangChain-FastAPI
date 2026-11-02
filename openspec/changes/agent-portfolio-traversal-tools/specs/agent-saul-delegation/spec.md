## Purpose

Admit portfolio traversal tools into the delegation contract under the cap discipline:
they join projections by replacement, ride the evidence-handle read path, and stay
gated until the graph can answer them.

## ADDED Requirements

### Requirement: Portfolio tools join projections by replacement under the cap

When the traversal gate opens, a role MAY gain a portfolio tool only by replacing a
same-role tool, keeping that role's model-visible toolset within the ≤7 projection
cap. The resulting toolset SHALL satisfy the projection-consistency invariant of the
tool-capability-catalog capability.

#### Scenario: A swap keeps the cap

- **WHEN** a portfolio tool is added to a role at the cap
- **THEN** exactly one same-role tool is removed in the same change
- **AND** the role's toolset identity test passes with the new composition
