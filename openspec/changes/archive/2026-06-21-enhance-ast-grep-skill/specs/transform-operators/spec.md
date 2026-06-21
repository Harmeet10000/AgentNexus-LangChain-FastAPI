## ADDED Requirements

### Requirement: All transform operators documented with examples

The skill SHALL document all transform operators: `replace`, `substring`, `convert` (with `toCase: lowerCase/upperCase/capitalize/camelCase/snakeCase/kebabCase/pascalCase`), `rewrite`, `capitalize`, `lowercase`, `uppercase`, `strip`, `convert_string`.

#### Scenario: Replace operator
- **WHEN** user needs regex-based text replacement on a matched variable
- **THEN** skill shows the `replace` operator with `source`, `replace`, `by` fields

#### Scenario: Convert case operator
- **WHEN** user needs to change case style (e.g., snake_case to camelCase)
- **THEN** skill shows the `convert` operator with the full case-type conversion table, separating case-sensitive vs non-sensitive conversions

#### Scenario: Rewrite operator with rewriters
- **WHEN** user needs to apply sub-rewrites to matched nodes
- **THEN** skill shows the `rewrite` operator with `source`, `rewriters`, `joinBy` fields

#### Scenario: Substring operator
- **WHEN** user needs to extract part of a matched variable
- **THEN** skill shows `substring` with `startChar`, `endChar`, `source` (with negative index support)

### Requirement: Both YAML dict style and string shorthand style documented

Each operator SHALL show both the YAML dict form and the compact string shorthand form for ast-grep ≥0.38.3.

#### Scenario: String shorthand
- **WHEN** user prefers inline syntax
- **THEN** skill shows `transform: NEW_VAR: replace($VAR, replace="^old", by="new")`

### Requirement: Operator examples in both SKILL.md and references

SKILL.md SHALL contain one brief example per operator plus a link to full details. `references/transform-operators.md` SHALL contain the full reference with all fields.
