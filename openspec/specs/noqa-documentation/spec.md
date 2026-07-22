## ADDED Requirements

### Requirement: Every remaining noqa SHALL have an explanatory comment

Every `# noqa` comment that remains in the codebase after the BLE001 migration SHALL have an inline comment explaining WHY the suppression is correct. The comment SHALL be specific to the suppression reason, not generic.

#### Scenario: PLC0415 lazy import has explanatory comment
- **WHEN** a `# noqa: PLC0415` remains for a lazy import
- **THEN** the comment explains the specific reason (circular import, optional dependency, or deferred load for performance)

#### Scenario: TC003/TC002/TC001 runtime resolution has explanatory comment
- **WHEN** a `# noqa: TC003`, `TC002`, or `TC001` remains for a type-checking import
- **THEN** the comment explains which framework resolves it at runtime (Pydantic model build, SQLAlchemy mapper, etc.)

#### Scenario: TRY300 return-in-try has explanatory comment
- **WHEN** a `# noqa: TRY300` remains for a return inside a try block
- **THEN** the comment explains why the return must be inside the try (idempotency completion, trace layer, etc.)

#### Scenario: F401 side-effect import has explanatory comment
- **WHEN** a `# noqa: F401` remains for an unused import
- **THEN** the comment explains the side effect (model registration, type re-export, etc.)

#### Scenario: S104/S105 false positive has explanatory comment
- **WHEN** a `# noqa: S104` or `S105` remains for a hardcoded value
- **THEN** the comment explains why it's not a secret (error code string, bind address, etc.)

#### Scenario: ARG002 protocol argument has explanatory comment
- **WHEN** a `# noqa: ARG002` remains for an unused argument
- **THEN** the comment explains the protocol or interface constraint that requires the argument

#### Scenario: PLW0603 module global has explanatory comment
- **WHEN** a `# noqa: PLW0603` remains for a global statement
- **THEN** the comment explains why module-level state is necessary (OTEL providers, etc.)

#### Scenario: PLR0915/PLR0912 complexity has explanatory comment
- **WHEN** a `# noqa: PLR0915` or `PLR0912` remains for a complex function
- **THEN** the comment explains the inherent complexity (lifespan initialization, etc.)

#### Scenario: RET503 implicit return has explanatory comment
- **WHEN** a `# noqa: RET503` remains for an implicit return
- **THEN** the comment explains the pattern (try/except/else, etc.)

#### Scenario: B039 ContextVar default has explanatory comment
- **WHEN** a `# noqa: B039` remains for a mutable ContextVar default
- **THEN** the comment explains why it's safe (evaluated once at module load, etc.)

#### Scenario: A002 builtin shadowing is eliminated by rename
- **WHEN** a `# noqa: A002` exists for a parameter named `filter`
- **THEN** the parameter is renamed to `filter_query` and the noqa is removed entirely

### Requirement: noqa comments SHALL follow consistent format

All `# noqa` comments SHALL follow the format: `# noqa: RULE_CODE — <reason>`. The reason SHALL be a short phrase (under 80 chars) explaining why the suppression is correct.

#### Scenario: BLE001 degradation boundary has consistent format
- **WHEN** a `# noqa: BLE001` remains for a genuine degradation boundary
- **THEN** the comment follows the format `# noqa: BLE001 — <specific reason>` (e.g., `# noqa: BLE001 — optional dep, must not crash app startup`)

#### Scenario: PLC0415 lazy import has consistent format
- **WHEN** a `# noqa: PLC0415` remains for a lazy import
- **THEN** the comment follows the format `# noqa: PLC0415 — <specific reason>` (e.g., `# noqa: PLC0415 — lazy import to avoid circular dependency`)

### Requirement: noqa count SHALL decrease

The total number of `# noqa` comments in the codebase SHALL decrease from ~160 to approximately 55 after this migration. The ~105 eliminated suppressions SHALL be replaced by proper exception handling or the A002 rename.

#### Scenario: BLE001 count decreases
- **WHEN** the migration is complete
- **THEN** the number of `# noqa: BLE001` comments decreases from 102 to ~15 (only genuine degradation boundaries remain)

#### Scenario: Total noqa count decreases
- **WHEN** the migration is complete
- **THEN** the total number of `# noqa` comments is at most 55 (down from ~160)

### Requirement: noqa codes SHALL be categorized as eliminated or retained

After migration, every noqa code SHALL be in one of two categories:

**Eliminated (replaced with proper handling):**
- `BLE001` — all non-boundary sites replaced with specific exception catches (102 → ~15)
- `A002` — parameter renamed from `filter` to `filter_query`

**Retained (with explanatory comments):**
- `PLC0415` — 19 lazy imports for circular/optional deps
- `TC003/TC002/TC001` — 9 runtime-resolved imports
- `TRY300` — 4 return-in-try for idempotency/trace
- `F401` — 2 side-effect imports (model registration, type re-export)
- `S104/S105` — 6 false positive secrets
- `ARG002` — 4 protocol-mandated arguments
- `PLW0603` — 3 module-level state for OTEL
- `PLR0915` — 1 lifespan complexity
- `RET503` — 2 implicit returns in try/except/else
- `B039` — 2 ContextVar defaults
- `ANN001/E402` — 3 test file fixtures

#### Scenario: Each retained noqa has explanatory comment
- **WHEN** the migration is complete
- **THEN** every remaining `# noqa` comment includes a `— <reason>` explanation

#### Scenario: Each eliminated noqa is gone
- **WHEN** the migration is complete
- **THEN** no `# noqa: BLE001` exists outside genuine degradation boundaries, and no `# noqa: A002` exists anywhere

### Requirement: Migration SHALL be verifiable by counting noqa comments

The migration success SHALL be verified by counting `# noqa` comments before and after.

#### Scenario: Pre-migration baseline
- **WHEN** the migration starts
- **THEN** a baseline count is recorded: `rg "# noqa" --count-matches` across `src/` and `scripts/`

#### Scenario: Post-migration verification
- **WHEN** the migration is complete
- **THEN** the same `rg "# noqa" --count-matches` command shows total decreased from ~160 to ~55

#### Scenario: No new noqa suppressions introduced
- **WHEN** the migration is complete
- **THEN** `rg "# noqa" src/` returns no lines that were not present in the baseline (no new suppressions were added as a side effect)
