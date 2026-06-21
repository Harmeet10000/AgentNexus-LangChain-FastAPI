## Lint Rules

Full lint rule structure:

```yaml
id: no-await-in-loop
message: Don't use await inside of loops
severity: error          # error | warning | hint | info
language: TypeScript
rule:
  pattern: await $_
  inside:
    any:
      - kind: for_in_statement
      - kind: while_statement
note: |
  Performing await in each loop iteration serializes async operations.
fix: |                    # optional auto-fix template
  const result = await $PROMISE
```

`message` supports `$MATCH` interpolation: `message: "Avoid $MATCH in production"`.

Multiple rules can be defined in one YAML file separated by `---` (useful for migration scripts):

```yaml
id: import-openai
language: python
rule:
  pattern: import openai
fix: from openai import Client
---
id: rewrite-client
language: python
rule:
  pattern: openai.api_key = $KEY
fix: client = Client($KEY)
```

### Severity Levels

| Level | Description |
|-------|-------------|
| `error` | Fails scan (exit code 1) |
| `warning` | Reported, non-fatal |
| `hint` | Suggestion |
| `info` | Informational |
| `off` | Disables the rule entirely |

Override severity per scan from CLI without editing rule files:

```bash
ast-grep scan --error rule-id --warning other-rule-id --off disable-this-rule
```

Severity `error` triggers non-zero exit — useful for CI/CD pipeline gating.

### `files` / `ignores`

Restrict rule application by glob patterns:

```yaml
files:
  - "src/**/*.ts"
ignores:
  - "**/*.test.ts"
  - "**/node_modules/**"
```

> Paths relative to `sgconfig.yml` directory. No `./` prefix.
> Globs can also use object form with `caseInsensitive: true`: `- glob: 'README.md'; caseInsensitive: true`.

### `labels`

Customize error highlighting in editor and terminal:

```yaml
labels:
  METHOD:
    style: primary      # primary | secondary
    message: the method name
  CLASS:
    style: secondary
    message: The class name
```

### Inline Suppression

Suppress diagnostics on specific lines using `ast-grep-ignore` comments:

```javascript
// ast-grep-ignore              ← suppresses ALL rules for next line
// ast-grep-ignore: rule-a, rule-b  ← suppresses specific rules
console.log('suppressed')    // ast-grep-ignore  ← same-line suppression
```

Rules:
- `ast-grep-ignore` alone suppresses all diagnostics for the next line
- Comment on the same line suppresses that line (must follow the code)
- Comment at line 1 with an empty line 2 suppresses the **entire file**
- To re-enable scanning for a single line, use the comment on that specific line

Report unused suppression comments with `unused-suppression` (hint-level, auto-fix):

```bash
ast-grep scan --error unused-suppression     # treat as error in CI
```

Disallow suppress-all (no rule ID) comments with the built-in `no-suppress-all` rule:

```bash
ast-grep scan --warning=no-suppress-all
```

`unused-suppression` is enabled by default only when all rules are active (no `--off` flags, no `--rule`, no `--filter`, no `--inline-rules`).

