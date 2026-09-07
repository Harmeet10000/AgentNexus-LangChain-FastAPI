## Purpose

Defines how one agent hands work to another and how persisted agent state is admitted back into a run: a single
handoff envelope with an explicit router, a bounded step budget, and a state schema version that is either resolved
deterministically or refused, never ignored.

## ADDED Requirements

### Requirement: Agent-to-agent handoff uses one envelope

A handoff from one agent to another SHALL be expressed as a message in the run's message channel that names the
intended recipient, and routing SHALL be decided by reading that message. There SHALL be one construction path for
handoff messages and one routing rule that interprets them. Handoff SHALL NOT be expressed through a separate
side-channel or a bespoke message bus.

#### Scenario: A handoff names its recipient in the message channel

- **WHEN** an agent hands work to another agent
- **THEN** a message naming the intended recipient SHALL be appended to the run's message channel

#### Scenario: Routing reads the handoff message

- **WHEN** the router decides the next step after a handoff
- **THEN** it SHALL decide from the handoff message in the message channel

#### Scenario: An unrecognised recipient is refused

- **WHEN** a handoff message names a recipient that is not a known agent
- **THEN** routing SHALL fail with an explicit error rather than silently continuing

#### Scenario: Message history accumulates rather than being replaced

- **WHEN** successive agents append messages during one run
- **THEN** the message channel SHALL retain the accumulated history

### Requirement: A run has a bounded step budget

Every agent run SHALL execute under an explicit maximum number of steps. Exceeding it SHALL terminate the run with an
explicit error rather than looping indefinitely.

#### Scenario: A run exceeding its step budget terminates explicitly

- **WHEN** a run exceeds its configured maximum number of steps
- **THEN** it SHALL terminate with an explicit error identifying the exhausted budget

#### Scenario: The budget is set for every run

- **WHEN** an agent run is started
- **THEN** a maximum step count SHALL be in effect

### Requirement: State loaded from persistence is version-checked before use

Before persisted agent state is used, its recorded schema version SHALL be compared against the version the running
code expects. A matching version SHALL proceed. An older recognised version SHALL be upgraded deterministically. An
unrecognised or newer version SHALL be refused with a typed error. The run SHALL NEVER proceed on state of an unknown
version.

#### Scenario: Matching version proceeds

- **WHEN** persisted state is loaded whose recorded schema version equals the expected version
- **THEN** the run SHALL proceed with that state

#### Scenario: Older recognised version is upgraded

- **WHEN** persisted state is loaded whose recorded schema version is an older recognised version
- **THEN** the state SHALL be upgraded to the expected version before any other step reads it

#### Scenario: Unknown version is refused

- **WHEN** persisted state is loaded whose recorded schema version is unrecognised
- **THEN** the run SHALL be refused with a typed error
- **AND** the state SHALL NOT be used

#### Scenario: The version check runs before any reasoning step

- **WHEN** a run resumes from persistence
- **THEN** the version check SHALL occur before any reasoning step reads the state

### Requirement: One schema version value governs writing and reading

The schema version recorded when agent state is written SHALL be the same value that the version check expects when it
is read. There SHALL NOT be two independently maintained sources for that value.

#### Scenario: Written and expected versions agree

- **WHEN** agent state is written and then loaded
- **THEN** the recorded version SHALL be the version the check expects

#### Scenario: Bumping the version changes both sides together

- **WHEN** the state schema version is raised
- **THEN** both the value written and the value expected SHALL change together
