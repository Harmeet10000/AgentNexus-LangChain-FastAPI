## Purpose

Guarantee that a dependency the startup sequence degrades past silently is visible in the health report, and that
an optional dependency being absent is reported as such without turning the whole service into a failing one.

## ADDED Requirements

### Requirement: A dependency the startup sequence degrades past SHALL be visible in the health report

When startup initialises a dependency and continues after that initialisation fails, the health report SHALL
carry an entry for that dependency. The report is the only observable signal that the degradation happened, so a
dependency that startup can leave absent SHALL NOT be omitted from it.

#### Scenario: The graph-memory dependency failed to initialise

- **WHEN** the graph-memory dependency fails to initialise at startup and the application continues serving
- **THEN** the health report SHALL include an entry for that dependency
- **AND** that entry SHALL show it is not available

#### Scenario: The graph-memory dependency is available

- **WHEN** the graph-memory dependency initialised successfully
- **THEN** the health report SHALL include an entry for it showing it is available

### Requirement: An absent optional dependency SHALL NOT be reported as a failure

An optional dependency that is not configured SHALL be reported as not configured. Its absence SHALL NOT change
the overall status of the report or the HTTP status code the health endpoint returns, so that an environment which
deliberately runs without an optional dependency does not report itself as down.

#### Scenario: An optional dependency is not configured

- **WHEN** the health report is produced while an optional dependency is not configured
- **THEN** that dependency's entry SHALL report that it is not configured
- **AND** the overall status and the HTTP status code SHALL be the same as they would be if that dependency were
  not part of the report at all

#### Scenario: A configured dependency that is failing

- **WHEN** the health report is produced while a dependency that is configured cannot be reached
- **THEN** that dependency's entry SHALL report the failure
- **AND** the overall status SHALL reflect it

### Requirement: The versioned health endpoint SHALL report every dependency any health surface reports

Where the application exposes more than one health surface, the versioned endpoint clients call SHALL report a
dependency set that is not smaller than any other surface's. A degradation that one surface can show SHALL NOT be
invisible on the surface consumers are versioned against.

#### Scenario: Comparing the reported dependency sets

- **WHEN** the dependency sets reported by each health surface are compared
- **THEN** every dependency named by any surface SHALL also be named by the versioned endpoint

### Requirement: Extending the health report SHALL be additive

Dependencies SHALL only be added to the report. No existing entry SHALL be renamed or removed, because the report
shape is published on more than one API version at once and the model rejects unknown fields, so a rename breaks
both versions simultaneously.

#### Scenario: The report gains a dependency

- **WHEN** a dependency is added to the health report
- **THEN** every entry that was previously reported SHALL still be reported under the same name
- **AND** every API version exposing the report SHALL expose the same set of entries
