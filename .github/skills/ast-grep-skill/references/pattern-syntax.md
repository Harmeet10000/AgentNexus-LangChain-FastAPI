## Pattern Syntax

### Meta Variables

Match any single AST node (like regex `.` but for syntax):

```
$VAR          — matches one AST node (uppercase, digits, underscore)
$$$           — matches zero or more AST nodes
$META_VAR_1   — named single-node capture
$$$ARGS       — named multi-node capture (arguments, parameters, statements)
$_            — anonymous wildcard (discarded match)
```

**Valid:** `$META`, `$META_VAR`, `$META_VAR1`, `$_`, `$_123`
**Invalid:** `$invalid`, `$Svalue` (starts with lowercase letter), `$123` (starts with digit), `$KEBAB-CASE`

### Meta Variable Capturing (Back-referencing)

Reusing the same meta variable name enforces structural equality:

```
$A == $A     → matches `a == a`, `1 + 1 == 1 + 1`
             → does NOT match `a == b`
```

Pattern `$PROP && $PROP()` matches property-check patterns by requiring the same expression before and after `&&`.

### Anonymous (Unnamed) Patterns

When tree-sitter can't parse a meta-variable pattern directly (e.g., `$M String $F;` in Java), use object-style pattern with `context` + `selector`:

```yaml
rule:
  pattern:
    context: 'class H { $METHOD() { $$$ } }'
    selector: method_definition
```

`context` provides parseable code; `selector` narrows to the specific AST node kind.

### Matching Strictness

Controls how precisely pattern code must match the AST:

| Mode | Behavior | When to use |
|------|----------|-------------|
| `cst` | Exact CST match — whitespace-sensitive, every detail counts | Debugging precise AST structure |
| `ast` | AST match ignoring comments/whitespace | General-purpose, ignores formatting |
| `smart` (default) | Balances precision and flexibility | Most cases — default |
| `relaxed` | Lenient matching; ignores extra semicolons, optional nodes | Capturing patterns despite minor structure differences |
| `signature` | Matches only the function/node skeleton, ignores implementations | Checking for presence of patterns regardless of body content |

Set strictness via CLI `--strictness ast` or in YAML rules:

```yaml
rule:
  pattern:
    context: 'function $NAME($$$) { $$$ }'
    selector: function_declaration
    strictness: signature
```
