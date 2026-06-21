## Relational Rules

| Rule | Meaning | Target is... |
|------|---------|-------------|
| `inside` | Target is inside surrounding | **inside** the surrounding node |
| `has` | Target has a child | the parent of the child node |
| `follows` | Target follows surrounding | **after** the surrounding node |
| `precedes` | Target precedes surrounding | **before** the surrounding node |

```yaml
# Find await inside a loop
rule:
  pattern: await $PROMISE
  inside:
    any:
      - kind: for_in_statement
      - kind: while_statement
    stopBy: end

# Find a property with key "prototype"
rule:
  kind: pair
  has:
    field: key
    regex: prototype
```

### `stopBy`

Controls how far relational rules search:

- `stopBy: end` — search until root/leaf/first/last sibling
- `stopBy: neighbor` — default, one level
- `stopBy: { kind: function }` — custom rule, stops when matched

### `field`

Filter by named child field (tree-sitter field name):

```yaml
has:
  field: key
  pattern: $NAME
```

## Composite Rules

Combine rules with logical operators:

```yaml
# all: node must satisfy ALL rules
rule:
  all:
    - pattern: console.log($GREETING)
    - not:
        pattern: console.log('Hello World')

# any: node must satisfy ANY rule
rule:
  any:
    - pattern: var a = $A
    - pattern: const a = $A

# not: node must NOT satisfy the rule
rule:
  pattern: await $_
  not:
    inside:
      kind: for_in_statement
```

> `all` guarantees rule order (important for meta-variable matching). Rule object fields (`pattern` + `kind` + `inside` on the same object) do NOT guarantee order — use `all` when order matters.

### `matches` (Utility Rule)

Reference a reusable sub-rule defined in `utils:`:

```yaml
utils:
  literal:
    any:
      - kind: number
      - kind: string
      - kind: 'true'
      - kind: 'false'
      - kind: 'null'
rule:
  any:
    - matches: literal
    - kind: array
      has:
        matches: literal
```

`utils` can be in the same file or in separate files under a `utilsDir` in `sgconfig.yml`.

### ESQuery Selectors

ast-grep supports ESQuery-compatible selector syntax for users familiar with eslint:

```yaml
rule:
  esquery: "CallExpression[callee.name='eval']"
```

Common ESQuery selectors and their ast-grep rule equivalents:

| ESQuery | ast-grep rule |
|---------|---------------|
| `CallExpression` | `kind: call_expression` |
| `CallExpression > Identifier` | `kind: call_expression` + `has: { field: function, kind: identifier }` |
| `CallExpression[callee.name='eval']` | `pattern: eval($$$)` |
| `FunctionDeclaration[name=/get.*/]` | `kind: function_declaration` + `regex: "^get"` |
| `MethodDefinition` | `kind: method_definition` |
| `:matches(ArrowFunctionExpression, FunctionExpression)` | `any: [{kind: arrow_function}, {kind: function_expression}]` |

