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

