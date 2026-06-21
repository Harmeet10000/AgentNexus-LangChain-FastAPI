## ADDED Requirements

### Requirement: Per-language catalog of real-world patterns

The skill SHALL include a catalog of real-world search/rewrite patterns organized by language, with at least 20 patterns total across these languages: JavaScript/TypeScript (≥6), Python (≥4), Rust (≥3), Go (≥2), HTML (≥2), and others.

#### Scenario: TS import extraction pattern
- **WHEN** user needs to find and classify import statements
- **THEN** skill provides the comprehensive import extraction YAML that handles named, default, namespace, side-effect, dynamic, and aliased imports

#### Scenario: TS logical assignment operator pattern
- **WHEN** user searches for `$A = $A || $B` to replace with `$A ||= $B`
- **THEN** skill provides both the CLI one-liner and the full YAML rule

#### Scenario: TS useState type removal pattern
- **WHEN** user wants to remove redundant `useState<type>()` type annotations
- **THEN** skill provides the pattern `useState<type>($A)` → `useState($A)`

#### Scenario: Promise.all no-await pattern
- **WHEN** user finds `await` inside a `Promise.all()` array
- **THEN** skill provides the rule with `stopBy` usage and fix

#### Scenario: Python remove-async-await pattern
- **WHEN** user wants to remove unnecessary async/await from sync functions
- **THEN** skill provides the pattern that detects and removes redundant `async`/`await`

#### Scenario: Python SQLAlchemy pattern
- **WHEN** user migrates SQLAlchemy `Column` definitions to mapped_column
- **THEN** skill provides the rewrite rule

#### Scenario: Rust avoid-duplicated-exports pattern
- **WHEN** user wants to find duplicate `pub use` exports
- **THEN** skill provides the rule with relational `follows` or `precedes`

#### Scenario: Go function call matching
- **WHEN** user searches for specific function call patterns in Go
- **THEN** skill provides the pattern

#### Scenario: HTML i18n extraction pattern
- **WHEN** user extracts static text in HTML templates to i18n keys
- **THEN** skill provides the rule with `kind: text`, pattern, and `not` regex exclusion

### Requirement: Catalog patterns include CLI one-liner and full YAML

Each catalog pattern SHALL include the CLI one-liner (where applicable) and the full YAML rule, plus at least one example input and output.

#### Scenario: One-liner for simple patterns
- **WHEN** a pattern can be expressed as a single CLI command
- **THEN** the skill shows `ast-grep -p 'pattern' -r 'rewrite' -l lang`

#### Scenario: Full YAML for complex rules
- **WHEN** a pattern requires relational, composite, or utility rules
- **THEN** the skill shows the complete YAML with `id`, `rule`, and optionally `fix`

### Requirement: Catalog patterns in reference files

Full catalog pattern YAML SHALL be stored in `references/catalog/` to keep SKILL.md scannable. SKILL.md SHALL contain a compact summary table linking to reference entries.

#### Scenario: Table of contents
- **WHEN** user scans SKILL.md catalog section
- **THEN** they see a language-vs-pattern table with brief descriptions and links to `references/catalog/<lang>.md`

### Requirement: Catalog patterns marked with last-verified date

Framework-specific patterns SHALL include a `# Last verified: YYYY-MM` comment.
