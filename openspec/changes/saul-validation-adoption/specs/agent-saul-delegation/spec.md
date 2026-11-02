## Purpose

Bring Saul's verdict/entailment path onto the shared validation vocabulary without
changing its behaviour, and route its escalations through the one sink contract into
the existing human-review node.

## ADDED Requirements

### Requirement: The verdict path emits shared-vocabulary outcomes with unchanged behaviour

The verdict/entailment path's mechanical gates SHALL map onto the pipeline's
mechanical-predicate and semantic-report shapes, and its exits SHALL be expressible as
shared-vocabulary decisions. Every finding code the path emits SHALL be part of the
shared vocabulary. The path's observable behaviour SHALL be unchanged: the existing
verdict-path tests SHALL pass unmodified, and ACCEPT and ESCALATE SHALL both be
reachable in type-level tests.

#### Scenario: Vocabulary changes, behaviour does not

- **WHEN** the full existing verdict-path suite runs against the adopted path
- **THEN** every test passes without modification
- **AND** type-level tests reach both ACCEPT and ESCALATE outcomes

#### Scenario: Findings carry locators in the shared shape

- **WHEN** a mechanical gate produces a finding
- **THEN** the finding is a semantic-report finding with its locator, in the shared
  vocabulary rather than a private shape

### Requirement: ESCALATE routes to the human-review node through the sink

An ESCALATE outcome from any Saul-side validation SHALL reach the human-review node
through the `HumanReviewSink` contract — the interrupt driven *through* the sink, not
around it — and no Saul seam SHALL introduce an additional escalation path.

#### Scenario: The interrupt goes through the sink

- **WHEN** a Saul-side pipeline yields an ESCALATE outcome with a review-class reason
- **THEN** the human-review node receives it via the sink and the HITL interrupt fires
  exactly as the existing review flow does

### Requirement: Validation retries charge the budget ledger

Where a Saul-side seam runs the pipeline with retries, every producer re-invocation
SHALL charge Saul's budget ledger, exhaustion SHALL yield ESCALATE with reason
`budget_exhausted`, and no fresh allowance SHALL be minted.

#### Scenario: Exhaustion is honest at the ledger

- **WHEN** retries exhaust the ledger mid-validation
- **THEN** the outcome is ESCALATE with reason `budget_exhausted`
- **AND** the ledger's recorded charges match the actual re-invocations

### Requirement: Tool-calling seams keep their middleware retry boundary

The tool-calling seams' transport retry SHALL remain in the middleware stack
(`ModelRetryMiddleware` / `ToolRetryMiddleware`), unchanged by this adoption. A test
SHALL assert the stack so the semantic layer cannot silently absorb transport retry.

#### Scenario: The middleware stack is asserted unchanged

- **WHEN** the adoption lands
- **THEN** the tool-calling seams' middleware configuration is identical to its
  pre-adoption state, proven by an assertion test
