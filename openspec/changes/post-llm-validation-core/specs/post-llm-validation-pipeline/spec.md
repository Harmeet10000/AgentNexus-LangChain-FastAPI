## Purpose

Define the one typed pipeline every LLM seam can run its output through: what decisions
exist, what each decision requires, what happens when the evaluator cannot run, and how
escalation is routed.

## ADDED Requirements

### Requirement: Every outcome carries the shared decision vocabulary

The pipeline SHALL return a typed outcome carrying exactly one decision (ACCEPT,
RETRY, or ESCALATE), the validated value when and only when the decision is ACCEPT,
the attempt count, both gate reports, and a stable typed reason. A value SHALL never
accompany a non-ACCEPT decision, and a decision SHALL never be derivable from
anything but the outcome.

#### Scenario: The value travels only with ACCEPT

- **WHEN** the pipeline returns any outcome
- **THEN** the validated value is present if and only if the decision is ACCEPT

### Requirement: The schema gate runs first and fully

A value that fails schema validation SHALL never reach the semantic evaluator. The
schema gate's report SHALL carry typed errors, and coercion SHALL occur only where
the schema itself declares it.

#### Scenario: A schema failure short-circuits evaluation

- **WHEN** the produced value does not parse against the schema
- **THEN** the semantic evaluator is not invoked
- **AND** the outcome reports the schema errors, with the decision derived only from
  the retry rules

### Requirement: Evaluator unavailability is fail-closed

When the semantic evaluator is unconfigured, raises, or cannot run, its report SHALL
carry the `unavailable` verdict and the pipeline's decision SHALL be ESCALATE — never
ACCEPT and never a silent pass. Absence of evaluation SHALL be a typed state
indistinguishable-in-kind from rejection at the outcome level.

#### Scenario: A down evaluator stops the seam

- **WHEN** the semantic evaluator raises or is unconfigured during validation
- **THEN** the outcome's semantic report is `unavailable`
- **AND** the decision is ESCALATE with a reason naming the unavailability

### Requirement: Retry carries structured feedback

When the decision is RETRY, the pipeline SHALL invoke the producer again with
structured feedback comprising the schema errors, the semantic findings, and the
attempt number. Retrying without feedback SHALL be a contract violation of the
pipeline, not a caller choice.

#### Scenario: The second attempt knows why the first failed

- **WHEN** the first attempt is rejected and attempts remain for a retryable finding
- **THEN** the producer's next invocation receives the prior schema errors, semantic
  findings, and attempt number

### Requirement: Only retryable findings retry

A rejection whose findings are not in the pipeline's declared retryable set SHALL
ESCALATE immediately, regardless of attempts remaining. A retry SHALL occur only when
a retryable finding is present and attempts remain.

#### Scenario: A non-retryable rejection does not consume retries

- **WHEN** the evaluator rejects with only non-retryable findings and attempts remain
- **THEN** the decision is ESCALATE, not RETRY

### Requirement: Budget honesty

When a budget ledger is supplied, every re-invocation of the producer SHALL charge it.
Budget exhaustion mid-pipeline SHALL yield ESCALATE with reason `budget_exhausted`,
and the pipeline SHALL NOT mint a fresh allowance to continue.

#### Scenario: Exhaustion stops the loop with a typed reason

- **WHEN** the budget ledger reports exhaustion before attempts are exhausted
- **THEN** the decision is ESCALATE with reason `budget_exhausted`

### Requirement: Mechanical checks precede model consultation

An evaluator composition SHALL run its mechanical predicates first and SHALL consult
the model evaluator only when the mechanical checks pass. Mechanical findings SHALL
short-circuit the model call.

#### Scenario: A mechanical rejection never costs a model call

- **WHEN** a mechanical predicate fails during validation
- **THEN** the model evaluator is not invoked for that attempt

### Requirement: Every outcome emits one decision event

The pipeline SHALL emit exactly one structured decision event per outcome — decision,
reason, attempts, and the seam's name — and the caller MAY persist the outcome
alongside its result where a record exists.

#### Scenario: Observability is one-to-one

- **WHEN** any validation completes
- **THEN** exactly one decision event is emitted carrying decision, reason, attempts,
  and seam name

### Requirement: Escalation routing belongs to the caller

The pipeline SHALL decide ESCALATE and its reason; the routing of an ESCALATE outcome
to a terminal failure, a human-review queue, or a dead letter SHALL be decided and
performed by the caller through the sink contract. No sink SHALL be hard-wired into
the pipeline.

#### Scenario: The same ESCALATE routes differently per seam

- **WHEN** two seams with different configured sinks receive equivalent ESCALATE
  outcomes
- **THEN** each outcome reaches its own caller-configured sink and no shared
  routing decision is made inside the pipeline
