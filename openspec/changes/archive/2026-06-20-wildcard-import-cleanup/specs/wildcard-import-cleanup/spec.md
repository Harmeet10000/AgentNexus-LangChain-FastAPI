## ADDED Requirements

### Requirement: No wildcard imports in src/
All `from X import *` statements within `src/` SHALL be replaced with explicit imports of only the names actually referenced in the file.

#### Scenario: Wildcard import is replaced with explicit imports
- **WHEN** a file contains `from module import *`
- **THEN** it SHALL be replaced with `from module import name1, name2, ...` listing only names used in that file

#### Scenario: No runtime import errors after cleanup
- **WHEN** wildcard imports are replaced with explicit imports
- **THEN** `python -c "import <module>"` SHALL succeed for each affected module

### Requirement: F403 lint rule enforced
The project SHALL enforce `ruff` rule `F403` (ambiguous `from module import *`) as a hard error, preventing regression.

#### Scenario: Wildcard import is caught by linter
- **WHEN** a developer introduces `from X import *` in `src/`
- **THEN** `ruff check` SHALL report an `F403` error

#### Scenario: F403 in pyproject.toml
- **WHEN** the project's `pyproject.toml` is inspected
- **THEN** `F403` SHALL be present in `[tool.ruff.lint] select` or equivalent rule selection

### Requirement: Full test suite passes after each file
Each file with wildcard imports SHALL be changed and verified independently. The test suite SHALL pass after each file change.

#### Scenario: Tests pass after wildcard import cleanup
- **WHEN** a file's wildcard import is replaced with explicit imports
- **THEN** `uv run pytest` SHALL pass with no new failures
