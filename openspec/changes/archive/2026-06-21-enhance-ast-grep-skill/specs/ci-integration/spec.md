## ADDED Requirements

### Requirement: GitHub Action setup documented

The skill SHALL document how to run ast-grep linting in GitHub Actions, including the workflow YAML and prerequisite setup.

#### Scenario: Basic CI workflow
- **WHEN** user wants to lint on every push
- **THEN** skill shows workflow with `actions/checkout@v4` and `ast-grep/action@v1.4`

### Requirement: Exit code behavior documented

The skill SHALL document that `ast-grep scan` exits with non-zero if any `error`-severity rule is triggered, enabling pipeline gating.

#### Scenario: Pipeline failure on errors
- **WHEN** user configures CI to fail on lint errors
- **THEN** skill explains error severity → non-zero exit → workflow failure
