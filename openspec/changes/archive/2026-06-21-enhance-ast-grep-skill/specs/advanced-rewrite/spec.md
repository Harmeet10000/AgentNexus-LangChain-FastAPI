## ADDED Requirements

### Requirement: FixConfig with expandStart/expandEnd documented

The skill SHALL document the `FixConfig` object for advanced fix operations, including `template`, `expandStart`, and `expandEnd` for handling comma-separated lists (array items, object pairs).

#### Scenario: Delete array element with comma
- **WHEN** user needs to remove an array element and its trailing comma
- **THEN** skill shows `fix: { template: '', expandEnd: { regex: ',' } }`

### Requirement: Rewriters with rewrite transform documented

The skill SHALL document the rewriter system: defining sub-rewriters in the `rewriters:` section, applying them via the `rewrite` transform, and joining results with `joinBy`.

#### Scenario: Barrel import expansion
- **WHEN** user wants to expand `import { a, b } from './barrel'` to individual imports
- **THEN** skill shows the full barrel import example with `rewriters`, `rewrite` transform, and `joinBy`
