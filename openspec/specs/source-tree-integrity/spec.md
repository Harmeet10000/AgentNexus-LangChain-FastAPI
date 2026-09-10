# source-tree-integrity Specification

## Purpose
The application source tree SHALL import, be fully tracked, carry no lint exemption for a path that
does not exist, execute its migration tooling, and record a measured gate baseline. The
specification baseline SHALL contain every capability that a change amends, before that change
amends it.
## Requirements
### Requirement: Importable application tree

The application package SHALL import without error, no module under the source or test trees SHALL
reference the retired document-processing package path, and every module reachable from the
application entrypoint SHALL be tracked by version control.

#### Scenario: The entrypoint imports

- **WHEN** the application entrypoint module is imported
- **THEN** import SHALL complete without a module-resolution error

#### Scenario: The retired path has no references

- **WHEN** the source and test trees are searched for the retired package path
- **THEN** there SHALL be zero matches

#### Scenario: No reachable module is untracked

- **WHEN** version control is queried for untracked files under the source tree
- **THEN** no module reachable from the application entrypoint SHALL be reported

#### Scenario: A reference held only as a string is still found

- **WHEN** a test refers to the retired path as a patch target string rather than an import
- **THEN** test collection SHALL be used as the check, because a symbol-import probe cannot observe
  that reference

### Requirement: Lint configuration tracks the package layout

Per-file lint exemptions SHALL name only paths that exist on disk.

#### Scenario: An exemption names a module the rename did not carry over

- **WHEN** an exemption names a module that has no counterpart in the replacement package
- **THEN** that exemption SHALL be removed rather than repointed

#### Scenario: No rename-attributable diagnostic remains

- **WHEN** the linter runs
- **THEN** no import-outside-top-level or unsorted-import diagnostic SHALL be attributable to the
  rename

### Requirement: The features directory is a regular package with no import side effects

The features directory SHALL contain an `__init__.py`, and that file SHALL execute no imports.

#### Scenario: Implicit-namespace-package diagnostics are cleared

- **WHEN** the linter runs over the features tree
- **THEN** no implicit-namespace-package diagnostic SHALL be reported

#### Scenario: Importing the package loads no feature module

- **WHEN** the features package is imported
- **THEN** no feature router or ORM model module SHALL become loaded as a side effect

### Requirement: Migration tooling executes against the live database

The migration environment module SHALL permit the standard migration commands to run without being
edited at invocation time.

#### Scenario: The current revision is reported

- **WHEN** the current-revision command is issued
- **THEN** it SHALL report a revision identifier that exists on disk, rather than failing during
  environment import

#### Scenario: A later migration derives its parent from the measurement

- **WHEN** a later change authors a migration
- **THEN** it SHALL derive its parent revision from the recorded identifier rather than from a value
  written in prose

#### Scenario: Conflicting claims about the head are reconciled in writing

- **WHEN** the recorded identifier contradicts a previously claimed head
- **THEN** the contradiction SHALL be recorded with the measured value winning, not silently resolved

### Requirement: A recorded gate baseline for the cluster

The repair SHALL record measured lint, type, test and migration-revision outputs to durable files.

#### Scenario: A later claim of improvement cites the recorded files

- **WHEN** a later change in this cluster claims a gate improvement
- **THEN** it SHALL compare against these recorded files rather than a number written in prose

#### Scenario: An unrunnable command records its failure verbatim

- **WHEN** a baseline command cannot execute
- **THEN** the recorded file SHALL carry the verbatim failure text instead of an assumed value

#### Scenario: Test outcomes are compared by summary counts

- **WHEN** a later change compares its test result against this baseline
- **THEN** the comparison SHALL use the summary pass and failure counts, because a configured
  coverage floor makes the process exit status a false signal about test outcomes

### Requirement: Every capability a change amends is present in the specification baseline

Capabilities whose requirements this cluster amends SHALL exist under the live specification tree
before any change declares a modification against them.

#### Scenario: A modification has a base to modify

- **WHEN** a change proposes a modification to a requirement
- **THEN** that requirement SHALL already be present in the live specification baseline

#### Scenario: A restored capability is reproduced without alteration

- **WHEN** a capability is restored from an archived change
- **THEN** its requirement and scenario text SHALL be reproduced without alteration, and its
  delta operation header SHALL be replaced by a plain requirements header

#### Scenario: The restored baseline validates

- **WHEN** the specification tree is validated
- **THEN** every restored capability SHALL pass

#### Scenario: A capability predating the requirement grammar is not restored

- **WHEN** an archived capability contains no requirement blocks because it predates the requirement
  grammar
- **THEN** it SHALL NOT be restored, and the reason SHALL be recorded rather than left implicit

