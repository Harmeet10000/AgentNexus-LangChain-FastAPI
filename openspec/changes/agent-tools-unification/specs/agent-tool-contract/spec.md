## Purpose

Defines the single result envelope every agent tool returns, the absolute separation between "the corpus says no" and
"the corpus could not be reached", and the identity rules for idempotent tool replay — the contract every agent, node,
and downstream verdict in the product depends on.

## ADDED Requirements

### Requirement: One result envelope for every agent tool

Every agent tool SHALL return results through a single common envelope carrying an outcome flag, a payload, an error
description, an availability signal, and structured metadata. No agent tool SHALL return a differently-shaped result,
and no second envelope definition SHALL be reachable from application code.

"Envelope definition" SHALL be determined by shape and role, not by class name. Four definitions of this envelope
exist today under two different names — three named `ToolResult` and one named `ToolOutput` — and the requirement is
satisfied only when exactly one remains reachable. A definition SHALL NOT be treated as out of scope because its
class name differs from the survivor's.

#### Scenario: A successful tool call returns the common envelope

- **WHEN** an agent tool completes successfully
- **THEN** it SHALL return the common envelope with the outcome flag set to success and the payload populated

#### Scenario: A failing tool call returns the common envelope

- **WHEN** an agent tool fails
- **THEN** it SHALL return the common envelope with the outcome flag set to failure and an error description populated

#### Scenario: The envelope rejects unrecognised fields

- **WHEN** a result envelope is constructed or deserialized with a field that is not part of the contract
- **THEN** construction SHALL fail rather than silently retaining the unknown field

#### Scenario: No differently-named envelope of the same shape survives

- **WHEN** the application's source is searched for definitions of a tool-result envelope under any name
- **THEN** exactly one definition SHALL be found

#### Scenario: The shell tool group returns the common envelope

- **WHEN** any of the filesystem, shell, or search tools registered from the agent tools package returns a result
- **THEN** it SHALL return the common envelope
- **AND** it SHALL NOT return a differently-shaped envelope defined in the same package

### Requirement: Tool failures are never reported as free-text output

A tool SHALL report failure through the envelope's error and availability signals. A tool SHALL NOT return a
human-readable error sentence in the success payload, and SHALL NOT return an error string in place of a result.

No envelope SHALL provide a method that renders itself to a bare error sentence for consumption by the model, and no
tool SHALL call such a method on its return path.

#### Scenario: A backend exception does not become the tool's answer

- **WHEN** a tool's underlying call raises
- **THEN** the tool SHALL return a failure envelope
- **AND** the success payload SHALL NOT contain a rendered error message presented as content

#### Scenario: The envelope is returned as a value, not as a rendered sentence

- **WHEN** a tool returns a failure result
- **THEN** it SHALL return the envelope itself
- **AND** it SHALL NOT return a string formed by prefixing the error description with an error marker

### Requirement: Unavailability SHALL never be reported as absence

When a tool cannot reach its backing corpus, index, or graph, it SHALL report the result as unavailable. It SHALL NOT
report, imply, or phrase the outcome as the requested item not existing. Absence and unavailability SHALL be
distinguishable by the caller without parsing the error text.

#### Scenario: A datastore failure during statute retrieval reports unavailability

- **WHEN** statute retrieval cannot query its corpus because the datastore call fails
- **THEN** the result SHALL carry the unavailable signal
- **AND** the error description SHALL NOT state that the requested section was not found

#### Scenario: A genuinely absent record reports absence

- **WHEN** statute retrieval successfully queries the corpus and no record matches
- **THEN** the result SHALL report the section as not found
- **AND** the unavailable signal SHALL NOT be set

#### Scenario: The caller can tell the two apart without reading prose

- **WHEN** a caller receives a non-successful tool result
- **THEN** it SHALL be able to distinguish unavailability from absence from the envelope's fields alone

### Requirement: A sufficiency verdict SHALL NOT be computed from a partial source set

When a tool aggregates evidence from more than one source and any source was unavailable, the tool SHALL NOT emit a
sufficiency or basis verdict as though the source set were complete. It SHALL expose an explicit signal that the
completeness of the evidence is unknown.

#### Scenario: One evidence leg fails and sufficiency becomes unknown

- **WHEN** precedent search retrieves results from one source and its other source is unavailable
- **THEN** the result SHALL set the basis-unknown signal
- **AND** SHALL NOT report sufficient basis on the strength of the surviving source alone

#### Scenario: All legs succeed and the verdict is computed normally

- **WHEN** every evidence source is reachable
- **THEN** the basis-unknown signal SHALL NOT be set
- **AND** the sufficiency verdict SHALL be computed from the full source count

#### Scenario: An unimplemented evidence leg is reported and not counted

- **WHEN** an evidence source is not implemented and returns nothing
- **THEN** the tool SHALL record a warning naming it as not implemented
- **AND** SHALL NOT count that source toward the total number of sources consulted

### Requirement: Idempotency identity is structural for writes and content-canonical for reads

Replay protection SHALL derive a call's identity deterministically. For a tool with side effects, identity SHALL be
composed only of the step identifier, the requesting user, and structural identifiers of the target. For a read or
search tool, identity SHALL additionally include a canonicalised form of the query. Callers SHALL be required to state
which parts of their input are structural and which are content.

#### Scenario: A side-effecting tool replayed with reworded content does not repeat the effect

- **WHEN** a side-effecting tool is replayed for the same step, user, and target identifiers with differently-worded
  content
- **THEN** the derived identity SHALL be unchanged
- **AND** the effect SHALL NOT be applied a second time

#### Scenario: Two different search queries do not share a cached answer

- **WHEN** a search tool is called twice for the same step, user, and scope with semantically different queries
- **THEN** the derived identities SHALL differ
- **AND** neither call SHALL return the other's cached result

#### Scenario: Trivially reworded search queries share a cached answer

- **WHEN** a search tool is called twice with queries differing only in letter case or surrounding whitespace
- **THEN** the derived identities SHALL be equal

#### Scenario: Identity is stable across processes

- **WHEN** the same call identity is derived in two separate processes
- **THEN** the derived values SHALL be identical

### Requirement: Changing the persisted result shape invalidates prior entries

When the persisted shape of a tool result changes, previously persisted entries SHALL become unreachable rather than
being read back under the new shape. The system SHALL NOT attempt to interpret an entry written under an older shape.

#### Scenario: Entries written under the previous shape are not read under the new one

- **WHEN** the result envelope's persisted shape changes
- **THEN** lookups SHALL NOT return entries written under the previous shape
- **AND** the affected calls SHALL be treated as cache misses and re-executed
