## ADDED Requirements

### Requirement: Severity override flags documented

The skill SHALL document CLI severity override flags: `--error <rule-id>`, `--warning <rule-id>`, `--info <rule-id>`, `--hint <rule-id>`, `--off <rule-id>`.

#### Scenario: Override severity per scan
- **WHEN** user wants to change severity for a single scan without editing rule files
- **THEN** skill shows `ast-grep scan --error rule-id --warning other-rule-id`

### Requirement: Inline suppression documented

The skill SHALL document all forms of inline suppression: same-line, next-line, specific-rule, multi-rule, file-level.

#### Scenario: Suppress next line
- **WHEN** user places `// ast-grep-ignore` before a line
- **THEN** the next line's diagnostics are suppressed

#### Scenario: Suppress specific rules only
- **WHEN** user writes `// ast-grep-ignore: rule-a, rule-b`
- **THEN** only those rules are suppressed for that line

#### Scenario: File-level suppression
- **WHEN** user places `// ast-grep-ignore` on line 1 with an empty line 2
- **THEN** all diagnostics in the file are suppressed

### Requirement: Built-in suppression management rules documented

The skill SHALL document the `unused-suppression` and `no-suppress-all` built-in rules, including their enablement conditions.

#### Scenario: Unused suppression detection
- **WHEN** user runs scan with all rules enabled
- **THEN** `unused-suppression` hint-level diagnostics flag unused `ast-grep-ignore` comments

#### Scenario: Disallow suppress-all
- **WHEN** user runs `ast-grep scan --warning=no-suppress-all`
- **THEN** suppress-all comments without specific rule IDs are flagged
