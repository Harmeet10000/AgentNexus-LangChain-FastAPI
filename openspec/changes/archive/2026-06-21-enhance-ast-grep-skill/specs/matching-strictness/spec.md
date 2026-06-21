## ADDED Requirements

### Requirement: All matching strictness levels documented

The skill SHALL document all 5 strictness levels: `cst`, `ast`, `smart` (default), `relaxed`, `signature` — with their behavior, use cases, and tradeoffs.

#### Scenario: When to use each strictness
- **WHEN** user's pattern is not matching expected code
- **THEN** the skill explains how adjusting strictness changes matching behavior (whitespace, extra parentheses, comments, semicolons)

#### Scenario: Decision table
- **WHEN** user needs to pick the right strictness
- **THEN** the skill provides a decision table with:
  - `cst`: exact CST match (whitespace-sensitive, every detail counts)
  - `ast`: AST match ignoring comments/whitespace
  - `smart` (default): balances precision and flexibility
  - `relaxed`: lenient matching, ignores extra details like semicolons
  - `signature`: matches only the signature/skeleton, ignores implementations

### Requirement: Strictness usage in pattern and rule

The skill SHALL show how to set strictness in CLI (`--strictness`) and in YAML (`strictness: smart`), including the object-style pattern's `strictness` field.
