# agent-saul-delegation Specification

## Purpose
Define how role handoffs in the Saul agent appear in the shared transcript: every delegation is a
linked assistant/tool message pair, constructed exactly one way, so delegations are traceable,
trimmable, and identical across runs.

## Requirements

### Requirement: Role handoffs are recorded as linked message pairs

Every delegation from one role to another SHALL appear in the shared transcript as an assistant
message carrying exactly one tool call immediately followed by the tool message linked to that
call. Delegations SHALL NOT travel off-transcript.

#### Scenario: Orchestrator hands off to a role

- **WHEN** the orchestrator delegates to a role
- **THEN** the transcript SHALL contain an assistant message with one `transfer_to_<role>` tool
  call followed by the tool message bearing the same call id

#### Scenario: No invisible delegations

- **WHEN** any role transfer completes
- **THEN** a transcript reader SHALL be able to reconstruct who handed off to whom and why from
  messages alone, without consulting out-of-band payloads

### Requirement: The pair constructor is defined once and shared

The delegation pair SHALL be constructed by exactly one shared constructor: the assistant message
carries the handoff reason as content and one tool call with that reason as arguments, and the
tool message carries the same reason with the matching call id. All graphs SHALL use this
constructor; no graph SHALL define its own pair shape.

#### Scenario: Identical shape across graphs

- **WHEN** the same handoff reason is constructed for different graphs
- **THEN** the resulting pairs SHALL be field-identical apart from the role name

#### Scenario: Reason stated once

- **WHEN** a delegation pair is constructed
- **THEN** the reason text SHALL occur in both messages from a single source, never typed twice

### Requirement: Pairs merge through the transcript reducer

Delegation pairs SHALL enter shared state exclusively through the message reducer, never by raw
list concatenation, so re-runs and retries upsert by message id instead of duplicating pairs.

#### Scenario: Retry does not duplicate a delegation

- **WHEN** a node re-runs after a delegation pair was recorded
- **THEN** the transcript SHALL contain the pair exactly once

### Requirement: Trimming preserves pair integrity

Context management SHALL NOT leave half a pair: trimming or summarization that removes a
delegation assistant message SHALL also remove its linked tool message, and vice versa. A pair
that must be dropped for budget SHALL be replaced by a single summary entry naming the
delegation.

#### Scenario: Budget trimming keeps pairs whole

- **WHEN** context trimming reaches a delegation pair
- **THEN** both messages SHALL be retained together, dropped together, or replaced together by
  one summary entry

### Requirement: The dict handoff contract remains valid during migration

Until its follow-up removal, a handoff tool invoked with a reason SHALL continue to yield the
structured transfer payload at the tool boundary, in addition to the transcript pair.

#### Scenario: Existing routing keeps working

- **WHEN** a role invokes a handoff tool during migration
- **THEN** routing SHALL observe the same transfer target and reason as before this change

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

### Requirement: Risk discloses ingestion completeness

The risk role's toolset SHALL include the scoped `get_ingestion_state` capability, and
a risk conclusion produced while any in-scope document is not ingestion-complete SHALL
disclose that state in its output rather than reading as a complete analysis.

#### Scenario: Risk parity with compliance on readiness

- **WHEN** a risk analysis runs over a document set containing an incomplete document
- **THEN** the role can retrieve the document's ingestion state through the same
  scoped capability compliance uses
- **AND** the analysis output reflects the incompleteness instead of silently
  proceeding

#### Scenario: No capability without scope

- **WHEN** `get_ingestion_state` is invoked without a server-owned runtime context
- **THEN** it is refused per the tool-registry execution-authority contract

### Requirement: Portfolio tools join projections by replacement under the cap

When the traversal gate opens, a role MAY gain a portfolio tool only by replacing a
same-role tool, keeping that role's model-visible toolset within the ≤7 projection
cap. The resulting toolset SHALL satisfy the projection-consistency invariant of the
tool-capability-catalog capability.

#### Scenario: A swap keeps the cap

- **WHEN** a portfolio tool is added to a role at the cap
- **THEN** exactly one same-role tool is removed in the same change
- **AND** the role's toolset identity test passes with the new composition
