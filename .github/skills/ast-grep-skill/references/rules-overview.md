## Rule Essentials

Rules are YAML files with a `rule:` key containing the matching logic. The simplest rule:

```yaml
rule:
  pattern: eval($CODE)
```

### Rule Object Fields

| Field | Description |
|-------|-------------|
| `rule` | Core matching logic (atomic, relational, composite) |
| `constraints` | Filter meta-variable matches by `kind`, `regex`, `inside`, `has` |
| `transform` | Convert captured nodes to strings before `fix` |
| `fix` | Replacement template string using `$VAR` |

### `constraints` — Meta-variable Filters

Refine matches by adding rules on captured variables:

```yaml
rule:
  pattern: const $NAME = require('$PATH')
constraints:
  NAME:
    regex: "^[a-z]"              # $NAME must start with lowercase
  PATH:
    not:
      regex: "^\\."              # $PATH must not start with .
```

Constraints apply per meta-variable key (without `$` prefix) and accept the full rule object syntax (`kind`, `regex`, `inside`, `has`, `not`, etc.).

## Atomic Rules

### `pattern`

Match by code pattern with meta variables:

```yaml
rule:
  pattern: console.log($GREETING)
```

### `kind`

Match by AST node type (tree-sitter node kind):

```yaml
rule:
  kind: arrow_function
```

Use ast-grep playground or `ast-grep --debug-query` to discover node kinds.

### `regex`

Match node text against a Rust regex:

```yaml
rule:
  regex: "^get[A-Z]"
```

Regex is in Rust syntax (no lookahead/backreferences). Always combine with `kind` or `pattern` for performance — regex alone is slow.

Inline flags: `(?i)`, `(?m)`, `(?s)`

### `nthChild`

Match by position among siblings (1-based, named nodes only):

```yaml
rule:
  kind: number
  nthChild: 2        # second number child

# An+B formulas
nthChild: 2n+1        # odd positions
nthChild:
  position: 2n+1
  reverse: true       # count from end
  ofRule:
    kind: function_declaration  # filter siblings
```

### `range`

Match by source position (0-based, character column):

```yaml
rule:
  range:
    start: { line: 0, column: 0 }
    end:   { line: 1, column: 5 }
```

