## ADDED Requirements

### Requirement: ESQuery selector compatibility documented

The skill SHALL document how ESQuery selectors work in ast-grep, with a reference table mapping common eslint selectors to their ast-grep equivalents.

#### Scenario: ast-grep ESQuery usage
- **WHEN** user is migrating from eslint-style selectors
- **THEN** skill shows the `esquery` field: `rule: { esquery: "CallExpression[callee.name='eval']" }`

#### Scenario: Selector-to-rule mapping table
- **WHEN** user knows an ESQuery selector but not the ast-grep equivalent
- **THEN** skill provides a small reference table mapping common selectors (e.g., `CallExpression > Identifier` → ast-grep relational rules)
