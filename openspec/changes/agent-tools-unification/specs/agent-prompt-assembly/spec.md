## Purpose

Defines how agent prompts are assembled from sections: one seam for all callers, a deterministic order chosen for
model attention rather than authoring convenience, and serialized data payloads that reach the model exactly as
serialized instead of being consumed by template substitution.

## ADDED Requirements

### Requirement: One ordering-aware assembly seam for agent prompts

Agent prompt construction that is subject to the ordering rule below SHALL go through one ordering-aware assembly
seam. A caller needing a plain assembled prompt and a caller needing a chat-ready prompt SHALL be served by the same
underlying assembly, so section ordering and section formatting are implemented once.

The seam SHALL distinguish sections by kind — standing instruction, output contract, retrieved evidence, task
restatement — rather than by label text, because ordering cannot be derived from a label-agnostic sequence of
positional arguments.

The label-agnostic section-rendering primitive SHALL remain available to callers that assemble their own prompts and
are not subject to the ordering rule. Such a caller SHALL NOT be required to migrate. Consequently the ordering rule
governs prompts assembled through the seam and not every prompt in the application; that boundary is a recorded gap,
not an omission.

#### Scenario: Both prompt entry points produce the same section order

- **WHEN** the same set of kinded prompt sections is assembled through the plain entry point and through the
  chat-ready entry point of the seam
- **THEN** the resulting section order SHALL be identical

#### Scenario: A prompt subject to the ordering rule is assembled through the seam

- **WHEN** a component needs an agent prompt whose section order is subject to the ordering rule
- **THEN** it SHALL obtain it from the seam rather than concatenating sections itself

#### Scenario: An existing caller of the label-agnostic primitive keeps working

- **WHEN** a caller assembles labelled sections through the label-agnostic rendering primitive
- **THEN** the primitive SHALL render them in the order supplied
- **AND** the caller SHALL NOT be required to declare section kinds

### Requirement: Deterministic attention-aware section ordering

Prompts assembled through the seam SHALL place sections in a fixed order by kind: standing instructions and the
required output contract first; retrieved evidence in the middle, with the highest-salience evidence at the head and
the tail of the evidence block; and the restatement of the immediate task last. The order SHALL be a property of the
assembly, not of the call site.

The seam SHALL accept retrieved evidence as a ranked sequence of items and a task restatement as its own section,
because neither can be ordered if supplied as opaque prose.

The standing instructions and output contract SHALL be assembled as the stable, reusable preamble, and the evidence
block and task restatement SHALL be assembled as per-turn content, so that ordering evidence for attention does not
invalidate reuse of the preamble on every turn.

#### Scenario: Instructions and output contract lead the prompt

- **WHEN** a prompt containing instructions, evidence, and a task restatement is assembled through the seam
- **THEN** the standing instructions and the output contract SHALL appear before the evidence

#### Scenario: The task restatement is last

- **WHEN** a prompt containing a task restatement is assembled through the seam
- **THEN** the task restatement SHALL appear after all evidence sections

#### Scenario: Highest-salience evidence occupies the block edges

- **WHEN** a ranked sequence of evidence items is assembled into the evidence block
- **THEN** the highest-ranked items SHALL occupy the first and last positions of that block

#### Scenario: Call order does not change section order

- **WHEN** the same kinded sections are supplied to the seam in a different call order
- **THEN** the assembled section order SHALL be unchanged

#### Scenario: The reusable preamble does not carry per-turn evidence

- **WHEN** a prompt is assembled through the seam for two successive turns with different evidence
- **THEN** the standing instructions and output contract portion SHALL be byte-identical across both turns

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
