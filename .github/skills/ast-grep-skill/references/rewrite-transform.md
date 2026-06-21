## Rewrite & Transform

### `fix` — Textual Replacement

```yaml
rule:
  pattern: console.log($GREET)
fix: logger.info($GREET)
```

`fix` is a template string — meta variables are replaced textually (not AST-validated).

### `transform` — String Manipulation

```yaml
rule:
  pattern: const $NAME = require('$PATH')
transform:
  NAME:
    capitalize: {}    # uppercase first letter
  PATH:
    replace:
      source: $PATH
      replace: "^(\\./)?"    # strip leading ./
      by: ""
fix: import $NAME from '$PATH'
```

#### Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `capitalize` | Uppercase first char | `hello` → `Hello` |
| `lowercase` | All lowercase | `Hello` → `hello` |
| `uppercase` | All uppercase | `hello` → `HELLO` |
| `substring` | Extract by position | `substring: { startChar: 1, endChar: -1 }` strips first/last char |
| `replace` | Regex replace (Rust syntax) | `replace: { source: $X, replace: "^old", by: "new" }` |
| `convert` | Case conversion (7 types) | `convert: { toCase: snakeCase, source: $VAR }` |
| `strip` | Remove surrounding quotes | `'hello'` → `hello` |
| `convert_string` | Toggle quotes | `'x'` → `"x"` |
| `rewrite` | Apply sub-rewriters to nodes | `rewrite: { source: $$$IDENTS, rewriters: [r1], joinBy: "\n" }` |

#### Case Conversion (`convert`)

| `toCase` | Input | Output | Separator-sensitive? |
|----------|-------|--------|:---:|
| `lowerCase` | astGrep | astgrep | No |
| `upperCase` | astGrep | ASTGREP | No |
| `capitalize` | astGrep | AstGrep | No |
| `camelCase` | ast_grep | astGrep | Yes |
| `snakeCase` | astGrep | ast_grep | Yes |
| `kebabCase` | astGrep | ast-grep | Yes |
| `pascalCase` | astGrep | AstGrep | Yes |

Optional `separatedBy` field controls which separators to split on: `dash`, `dot`, `space`, `slash`, `underscore`, `caseChange`.

#### Compact Shorthand (ast-grep ≥0.38.3)

All operators support a string-style form:

```yaml
transform:
  NAME: replace($VAR, replace="^old", by="new")
  CASE: convert($VAR, toCase=kebabCase, separatedBy=[underscore])
  SUB: substring($VAR, startChar=1, endChar=-1)
```

Full operator reference: `references/transform-operators.md`

### `FixConfig` — Advanced Fix with Range Expansion

For list items (array elements, object pairs) that need comma handling:

```yaml
rule:
  kind: pair
  has:
    field: key
    regex: Remove
fix:
  template: ''              # empty string = delete
  expandEnd: { regex: ',' }  # include trailing comma in deletion range
```

`expandStart` / `expandEnd` accept a rule object (with optional `stopBy`) to expand the fix range until the rule no longer matches.

### `rewriters` — Multi-Node Transformation

For complex rewrites where one match expands into multiple replacements:

```yaml
rule:
  pattern: import {$$$IDENTS} from './barrel'
rewriters:
  - id: rewrite-identifier
    rule:
      pattern: $IDENT
      kind: identifier
    fix: import $IDENT from './barrel/$IDENT'
transform:
  IMPORTS:
    rewrite:
      source: $$$IDENTS
      rewriters: [rewrite-identifier]
      joinBy: "\n"
fix: $IMPORTS
```

This expands `import { a, b } from './barrel'` into individual imports per identifier.

