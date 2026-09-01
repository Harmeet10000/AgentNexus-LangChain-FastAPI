## Purpose

Defines measurable completion gates for the repository-wide error migration.

## ADDED Requirements

### Requirement: The error migration SHALL finish only when every legacy and coverage count reaches its target

Completion SHALL be established by repository-wide measurements, not by the
absence of planned feature work. The measurements SHALL be independently
reconciled and SHALL include gate configuration as well as source matches.

#### Scenario: Every converting feature owns a closed contract
- **WHEN** feature error contracts are enumerated
- **THEN** `errors.py` exists in 15 of 18 features, with `chat`, `search`, and `health` recorded as the only exceptions

#### Scenario: The legacy hierarchy is absent
- **WHEN** source code is scanned for `AppError`, its subclasses, constructions, aliases, and `app_error_to_exception`
- **THEN** every count is zero

#### Scenario: Feature contracts do not cross ownership boundaries
- **WHEN** feature imports are scanned
- **THEN** no feature imports another feature's error types, union, or code enum

#### Scenario: Every union is closed and type-checked
- **WHEN** the feature contracts and their dispatch sites are checked
- **THEN** every concrete feature error is named in its union, exhaustive dispatch closes with `assert_never`, and `uv run ty check src/ tests/` passes

#### Scenario: Enforcement coverage is demonstrated
- **WHEN** every migration gate is verified
- **THEN** its forbid fixture reports, its permit fixture is clean, and its configured exclusions have been read before a zero is cited

#### Scenario: Completion evidence contains no admitted debt
- **WHEN** completed task evidence is audited for `partial`, `deferred`, and `TODO`
- **THEN** no completed task relies on admitted unfinished work, and final totals agree across two structurally different measurements
