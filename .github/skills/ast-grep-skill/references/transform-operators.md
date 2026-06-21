# Transform Operators Reference

## Overview

Transform operators manipulate meta variables captured by rules before applying `fix`. They can be written in two styles:

- **Dict style**: `OP_NAME: { field: value, ... }`
- **String shorthand** (≥0.38.3): `OP_NAME($VAR, field=value, ...)`

## `replace`

Regex-based text replacement on a captured meta variable.

```yaml
transform:
  NEW_VAR:
    replace:
      source: $VAR          # variable to operate on (required, must start with $)
      replace: "^old"       # Rust regex pattern (required)
      by: "new"             # replacement string (required)
```

**Shorthand:** `replace($VAR, replace="^old", by="new")`

Supports capture groups (`$1`, `$2`) in `by`.

## `substring`

Extract a substring by character position (Unicode-aware, like Python slicing).

```yaml
transform:
  NEW_VAR:
    substring:
      source: $VAR          # required
      startChar: 1          # optional, inclusive, default 0, supports negative
      endChar: -1           # optional, exclusive, default end, supports negative
```

**Shorthand:** `substring($VAR, startChar=1, endChar=-1)`

## `convert`

Convert string between case styles.

```yaml
transform:
  NEW_VAR:
    convert:
      source: $VAR          # required
      toCase: snakeCase     # required (see table below)
      separatedBy:          # optional — which separators to split on
        - underscore
        - caseChange
```

### Case Types

| `toCase` | Input | Output | Separator-sensitive? |
|----------|-------|--------|:---:|
| `lowerCase` | astGrep | astgrep | No |
| `upperCase` | astGrep | ASTGREP | No |
| `capitalize` | astGrep | AstGrep | No |
| `camelCase` | ast_grep | astGrep | Yes |
| `snakeCase` | astGrep | ast_grep | Yes |
| `kebabCase` | astGrep | ast-grep | Yes |
| `pascalCase` | astGrep | AstGrep | Yes |

### Separators

| Name | Character | Example split |
|------|:---------:|---------------|
| `Dash` | `-` | `ast-grep` → `[ast, grep]` |
| `Dot` | `.` | `ast.grep` → `[ast, grep]` |
| `Space` | ` ` | `ast grep` → `[ast, grep]` |
| `Slash` | `/` | `ast/grep` → `[ast, grep]` |
| `Underscore` | `_` | `ast_grep` → `[ast, grep]` |
| `CaseChange` | (case boundary) | `astGrep` → `[ast, Grep]`, `XMLHttpRequest` → `[XML, Http, Request]` |

**Shorthand:** `convert($VAR, toCase=kebabCase, separatedBy=[underscore])`

## `capitalize`

Uppercase the first character of the string.

```yaml
transform:
  NEW_VAR:
    capitalize:
      source: $VAR
```

**Shorthand:** Not available as shorthand; use `convert($VAR, toCase=capitalize)`.

## `lowercase`

Convert entire string to lowercase.

```yaml
transform:
  NEW_VAR:
    lowercase:
      source: $VAR
```

**Shorthand:** Not available as shorthand.

## `uppercase`

Convert entire string to uppercase.

```yaml
transform:
  NEW_VAR:
    uppercase:
      source: $VAR
```

**Shorthand:** Not available as shorthand.

## `strip`

Remove surrounding quotes from a string literal.

```yaml
transform:
  NEW_VAR:
    strip: {}
```

`'hello'` → `hello`, `"world"` → `world`

## `convert_string`

Toggle between single and double quotes.

```yaml
transform:
  NEW_VAR:
    convert_string: {}
```

`'x'` → `"x"`, `"y"` → `'y'`

## `rewrite`

Apply sub-rewriter rules to captured nodes. The most powerful transform — enables multi-node transformations.

```yaml
transform:
  NEW_VAR:
    rewrite:
      source: $$$IDENTS      # single ($VAR) or multi ($$$VAR)
      rewriters:             # list of rewriter IDs (from rewriters: section)
        - rewrite-identifier
      joinBy: "\n"           # optional, join rewritten results with separator
```

**Shorthand:** `rewrite($$$IDENTS, rewriters=[rewrite-identifier], joinBy='\n')`

See `rewriters` in SKILL.md for the barrel import example.

## Full Rule with Multiple Transforms

```yaml
rule:
  pattern: const $NAME = require('$PATH')
transform:
  NAME:
    capitalize: {}
  PATH:
    replace:
      source: $PATH
      replace: "^(\\./)?"
      by: ""
  MODULE:
    convert:
      source: $NAME
      toCase: kebabCase
fix: import $NAME from '$PATH'
```
