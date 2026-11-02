## Purpose

Two gated additions to the retrieval loop: per-stage budget enforcement with a
standing degradation alert, and section-level context hydration for generation
(small-to-big), adopted on measured evidence.

## ADDED Requirements

### Requirement: The retrieval loop enforces per-stage cost and latency budgets

The retrieval loop SHALL enforce cost and latency budgets attributed per stage
(plan, embed, branches, fusion, rerank, grade, generate), built on the per-stage
spans the retrieval-engine cutover introduces. Budget exhaustion SHALL produce the
typed degraded outcome rather than an unbounded run, consistent with the validation
pipeline's budget-honesty contract.

#### Scenario: An over-budget stage degrades honestly

- **WHEN** a stage exceeds its attributed budget mid-ask
- **THEN** the ask returns the typed degraded outcome rather than continuing
  unbounded
- **AND** the decision event names the stage and the budget that fired

### Requirement: Budget degradation raises a standing alert before the pilot-kill criteria

Sustained budget degradations SHALL raise a standing alert distinct from the harden
change's pilot-kill criteria — the alert is the early warning; the kill criterion is
the last resort. The alert SHALL be attributable per stage and per document class.

#### Scenario: The alert fires before the kill criterion

- **WHEN** budget degradations persist across the alert threshold
- **THEN** a degradation alert is raised naming the stages involved
- **AND** the pilot-kill evaluation remains a separate, later decision

### Requirement: Section-level context hydration is adopted on measured evidence

When the measurement baseline attributes recall loss to context starvation, the
generation stage SHALL hydrate the parent section of matched clause-level chunks
using the stored structural tree, with the retrieved citation still resolving to the
clause. The hydration SHALL be measurable on the baseline and revertible by the same
evidence standard as every other gated feature.

#### Scenario: The citation stays clause-precise while generation sees the section

- **WHEN** generation runs with section hydration enabled
- **THEN** citations resolve to the matched clauses exactly as without hydration
- **AND** a baseline comparison records the recall/precision effect per document
  class
