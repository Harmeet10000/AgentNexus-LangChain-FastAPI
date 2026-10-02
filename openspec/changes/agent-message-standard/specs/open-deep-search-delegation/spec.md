# open-deep-search-delegation Specification

## Purpose
Define how supervisor, researcher, and compression handoffs in the deep-research graph appear in
the shared transcript: the same linked message pairs as every other graph, in the message
channels each subgraph already owns.

## ADDED Requirements
### Requirement: Supervisor delegations are linked message pairs

Every supervisor-to-researcher delegation SHALL appear as an assistant message with one tool call
immediately followed by the tool message bearing the same call id, in the supervisor message
channel.

#### Scenario: Research task is dispatched visibly

- **WHEN** the supervisor dispatches a research topic
- **THEN** the supervisor channel SHALL contain the delegation pair naming the topic

### Requirement: Researcher results return as linked pairs

Each researcher result SHALL be recorded as a tool message linked to the delegating call, so the
result is attributable to its dispatch without consulting out-of-band state.

#### Scenario: Result traces back to its dispatch

- **WHEN** a researcher completes a topic
- **THEN** the transcript SHALL link the result to the exact delegation call that requested it

### Requirement: Pairs use the shared constructor

All pairs in this graph SHALL be constructed with the constructor defined in
`agent-saul-delegation`. This graph SHALL NOT define its own pair shape.

#### Scenario: Cross-graph pair uniformity

- **WHEN** a deep-research delegation pair is compared with a Saul delegation pair for the same
  reason shape
- **THEN** both SHALL satisfy the same linkage and single-source-reason rules

### Requirement: Compression preserves pair integrity

Research compression and summarization SHALL NOT leave half a pair: a delegation whose detail is
compressed away SHALL be represented by one summary entry, never by an orphaned assistant or
tool message.

#### Scenario: Compressed research stays attributable

- **WHEN** raw researcher notes are compressed into a summary
- **THEN** the summary SHALL still identify which delegation produced it
