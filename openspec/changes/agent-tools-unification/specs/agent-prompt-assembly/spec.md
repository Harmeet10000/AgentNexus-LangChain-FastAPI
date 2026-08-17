## Purpose

Defines how agent prompts are assembled from sections: one seam for all callers, a deterministic order chosen for
model attention rather than authoring convenience, and serialized data payloads that reach the model exactly as
serialized instead of being consumed by template substitution.

## ADDED Requirements

### Requirement: One prompt-assembly seam

All agent prompt construction SHALL go through one section-assembly seam. A caller needing a plain assembled prompt and
a caller needing a chat-ready prompt SHALL be served by the same underlying assembly, so section ordering and section
formatting are implemented once.

#### Scenario: Both prompt entry points produce the same section order

- **WHEN** the same set of prompt sections is assembled through the plain entry point and through the chat-ready entry
  point
- **THEN** the resulting section order SHALL be identical

#### Scenario: No caller assembles prompt sections independently

- **WHEN** a component needs an assembled agent prompt
- **THEN** it SHALL obtain it from the shared assembly seam rather than concatenating sections itself

### Requirement: Deterministic attention-aware section ordering

Assembled prompts SHALL place sections in a fixed order: standing instructions and the required output contract first;
retrieved evidence in the middle, with the highest-salience evidence at the head and the tail of the evidence block;
and the restatement of the immediate task last. The order SHALL be a property of the assembly, not of the call site.

#### Scenario: Instructions and output contract lead the prompt

- **WHEN** a prompt containing instructions, evidence, and a task restatement is assembled
- **THEN** the standing instructions and the output contract SHALL appear before the evidence

#### Scenario: The task restatement is last

- **WHEN** a prompt containing a task restatement is assembled
- **THEN** the task restatement SHALL appear after all evidence sections

#### Scenario: Highest-salience evidence occupies the block edges

- **WHEN** ranked evidence items are assembled into the evidence block
- **THEN** the highest-ranked items SHALL occupy the first and last positions of that block

#### Scenario: Call order does not change section order

- **WHEN** the same sections are supplied in a different call order
- **THEN** the assembled section order SHALL be unchanged

### Requirement: Serialized data payloads survive prompt rendering byte-exact

A serialized data payload injected into a prompt SHALL reach the model exactly as serialized. The prompt layer SHALL
NOT interpret any part of a data payload as template syntax, and rendering a prompt containing such a payload SHALL
NOT fail.

#### Scenario: A payload containing template-like syntax renders without error

- **WHEN** a prompt is rendered with a serialized payload whose text contains brace-delimited field lists
- **THEN** rendering SHALL succeed

#### Scenario: The payload text is unmodified in the rendered prompt

- **WHEN** a prompt is rendered with a serialized payload containing a brace-delimited header
- **THEN** the rendered prompt SHALL contain that header verbatim, with its original delimiters intact and not doubled

#### Scenario: Template variables still substitute

- **WHEN** a prompt containing both a declared template variable and a serialized data payload is rendered
- **THEN** the declared variable SHALL be substituted
- **AND** the data payload SHALL remain unsubstituted and unaltered
