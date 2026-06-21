# Rule Grammar Reference

## Rule Object Structure

```yaml
rule:
  # Atomic (pick one or more)
  pattern: <string> | { context: <string>, selector: <string> }
  kind: <string>
  regex: <string>
  nthChild: <number> | <string> | { position, reverse?, ofRule? }
  range: { start: { line, column }, end: { line, column } }

  # Relational
  inside:   <sub-rule>   # target is inside surrounding
  has:      <sub-rule>   # target has a child
  follows:  <sub-rule>   # target follows surrounding
  precedes: <sub-rule>   # target precedes surrounding

  # Composite
  all: [<sub-rule>, ...]  # match all rules
  any: [<sub-rule>, ...]  # match any rule
  not: <sub-rule>          # negate rule
  matches: <string>       # reference utility rule by id
```

Multiple fields on one rule object are AND-ed implicitly.

## Atomic Rules

### `pattern`

```
pattern: console.log($GREETING)
pattern:
  context: 'class H { $METHOD() { $$$ } }'
  selector: method_definition
```

| Form | Use |
|------|-----|
| `pattern: <string>` | Simple code pattern with meta variables |
| `pattern: { context, selector }` | Object-style: `context` provides parseable code, `selector` is the AST kind to match within context |

### `kind`

```
kind: arrow_function
kind: call_expression
kind: identifier
```

Matches by tree-sitter node type name. Discover kinds via:
- Playground (playground.ast-grep.net)
- `ast-grep run --debug-query -p '$X' -l ts`

### `regex`

```
regex: "^get[A-Z]"
regex: "(?i)error"          # case-insensitive
```

- Rust regex syntax (docs.rs/regex)
- No lookahead/backreferences
- Slow alone — pair with `kind` or `pattern` for performance

### `nthChild`

```
nthChild: 3                  # exact position (1-based)
nthChild: 2n+1               # An+B formula (odd positions)
nthChild:
  position: 2n+1
  reverse: true              # count from end
  ofRule:
    kind: function_declaration
```

- Named nodes only (unnamed/whitespace excluded)
- 1-based indexing (CSS-style)

### `range`

```
range:
  start: { line: 0, column: 0 }
  end:   { line: 1, column: 5 }
```

- 0-based line and column
- Character-based (not byte)
- Start inclusive, end exclusive

## Relational Rules

### `inside`

```
inside: <sub-rule>
```

Target node must be **inside** a surrounding node matching `sub-rule`.

```yaml
pattern: await $_
inside:
  kind: for_in_statement
  stopBy: end
```

### `has`

```
has: <sub-rule>
```

Target node must **have** a descendant matching `sub-rule`.

```yaml
kind: pair
has:
  field: key
  regex: prototype
```

### `follows`

```
follows: <sub-rule>
```

Target node must come **after** a sibling matching `sub-rule`.

```yaml
pattern: console.log('hello');
follows:
  pattern: console.log('world');
```

### `precedes`

```
precedes: <sub-rule>
```

Target node must come **before** a sibling matching `sub-rule`.

### Relational Sub-rule Options

| Option | Values | Description |
|--------|--------|-------------|
| `stopBy` | `end` / `neighbor` / `{ rule }` | How far to search. `end` = until root/leaf/boundary. `neighbor` = one level (default). Custom rule = stop when rule matches |
| `field` | `<string>` | Match only the named child field of the target node |

## Composite Rules

### `all`

```yaml
all:
  - pattern: console.log($GREETING)
  - not:
      pattern: console.log('Hello World')
```

- Node must match ALL sub-rules
- **Order is guaranteed** — use `all` when meta-variable matching depends on prior matches

### `any`

```yaml
any:
  - kind: number
  - kind: string
  - kind: 'true'
```

- Node must match ANY sub-rule

### `not`

```yaml
not:
  pattern: console.log('Hello World')
```

- Node must NOT match the sub-rule
- `constraints` do NOT work inside `not`

### `matches`

```yaml
matches: rule-id
```

- References a utility rule by `id` from `utils:` or `utilsDir`

## Lint Rule Schema

```yaml
id: <string>                    # required, unique rule ID
message: <string>               # required, diagnostic message
severity: error|warning|hint|info|off  # default: error
language: <string>              # required, target language
rule: <rule-object>             # required, matching logic
note: <string>                  # optional, explanation with fix guidance
fix: <string> | FixConfig       # optional, replacement template or advanced config
rewriters:                      # optional, rewriter sub-rules for multi-node transforms
  - id: <string>
    rule: <rule-object>
    fix: <string>
constraints:                    # optional, filter meta variables
  VAR_NAME:
    kind: <string>
    regex: <string>
    inside: <sub-rule>
    has: <sub-rule>
transform:                      # optional, string transformations (dict or shorthand)
  VAR_NAME:
    capitalize: {}
    lowercase: {}
    uppercase: {}
    convert: { toCase: snakeCase, source: $VAR }
    substring: { start, end }
    replace: { source, replace, by }
    convert_string: {}
    strip: {}
    rewrite: { source, rewriters, joinBy? }
  # shorthand (≥0.38.3):
  # VAR_NAME: replace($X, replace="^old", by="new")
files: [<glob>, ...]            # optional, restrict to matched files
ignores: [<glob>, ...]          # optional, exclude matched files
labels:                         # optional, customize highlighting
  VAR_NAME:
    style: primary|secondary
    message: <string>
utils:                          # optional, inline utility rules
  id: <string>
    rule: <rule-object>
```

### FixConfig

Advanced fix for range expansion (e.g., including trailing comma on deletion):

```yaml
fix:
  template: ''                  # replacement text
  expandEnd:                    # optional, expand end of replacement range
    regex: ','                  # expand until this regex stops matching
    stopBy: neighbor            # optional, default: neighbor
  expandStart:                  # optional, expand start of replacement range
    kind: comment
```

## sgconfig.yml Schema

```yaml
ruleDirs:
  - rules                        # required, directories with rule files
testConfigs:
  - testDir: rule-tests          # optional, test directories
utilsDir: utils                  # optional, global utility rule directory
languageGlobs:                   # optional, override language detection
  - glob: "*.tsx"
    language: tsx
customLanguages:                 # optional, register custom languages
  - name: prisma
    language: rust
    extensions: [prisma]
projectDir: .                    # optional, project root (default cwd)
```

## Test File Schema

```yaml
id: <string>                     # must match a rule id
valid:                           # code that should NOT trigger rule
  - <string>
  - <string>
invalid:                         # code that SHOULD trigger rule
  - <string>
  - <string>
```

## Transform Operators

| Operator | Input | Output |
|----------|-------|--------|
| `capitalize` | `hello` | `Hello` |
| `lowercase` | `Hello` | `hello` |
| `uppercase` | `hello` | `HELLO` |
| `substring: { start: 1, end: 4 }` | `hello` | `ell` |
| `convert: { toCase: camelCase }` | `ast_grep` | `astGrep` |
| `replace: { source: $X, replace: "^old", by: "new" }` | — | regex replace |
| `convert_string: {}` | `'x'` | `"x"` |
| `strip` | `'hello'` | `hello` |
| `rewrite: { source: $$$, rewriters: [id], joinBy: "\n" }` | — | sub-rewriter result |

### Shorthand Form ( ≥0.38.3)

All operators support a compact string-style syntax:

```yaml
transform:
  NAME: replace($VAR, replace="^old", by="new")
  CASE: convert($VAR, toCase=kebabCase)
  VAR: substring($VAR, startChar=1, endChar=-1)
```

Full operator reference: `references/transform-operators.md`

## Matching Strictness

Controls how closely source code must match pattern code:

| Strictness | Behavior |
|------------|----------|
| `cst` | Exact CST match — whitespace-sensitive, every detail |
| `ast` | AST match — ignores comments and whitespace |
| `smart` (default) | Balances precision and flexibility |
| `relaxed` | Lenient — ignores optional nodes, extra semicolons |
| `signature` | Skips function bodies, matches only skeleton |

Set via CLI `--strictness <mode>` or in YAML `pattern: { strictness: relaxed }`.

## Rule Categories

Rules can be organized into categories for filtering:

```bash
ast-grep scan --filter-category security
ast-grep scan --filter-category migration
```

Add a `category` field to rule YAML:

```yaml
id: no-eval
category: security
severity: error
rule:
  pattern: eval($$$)
```

## CLI Reference

| Command | Syntax |
|---------|--------|
| Search | `ast-grep -p <pattern> [-l <lang>] [paths]` |
| Replace | `ast-grep run -p <pattern> -r <rewrite> [-l <lang>] [paths]` |
| Scan | `ast-grep scan [--config <path>]` |
| Test | `ast-grep test [--skip-snapshot-tests]` |
| New project | `ast-grep new` |
| New rule | `ast-grep new rule` |
| LSP | `ast-grep lsp` |
| Completions | `ast-grep completions <zsh\|bash\|fish\|elvish\|powershell>` |
| Version | `ast-grep --version` |

### Output Format Options

- `--json` / `--json=pretty` (indented array)
- `--json=stream` (one JSON object per line, NDJSON)
- `--json=compact` (single-line array)

### Inspect Flags

| Flag | Purpose |
|------|---------|
| `--inspect summary` | Show project directory, config path, and rule counts |
| `--inspect entity` | List all loaded rules with their final computed severity |

### Stdin Mode

```bash
ast-grep -p '<div> $$$ </div>' -l html --json --stdin
```

Must specify `--lang` explicitly. No `--interactive` mode in stdin.
