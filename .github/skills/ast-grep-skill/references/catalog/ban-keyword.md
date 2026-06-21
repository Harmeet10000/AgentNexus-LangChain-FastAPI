# Catalog: Ban Keyword

## Ban `debugger` Statements

```yaml
# rule: no-debugger.yml
id: no-debugger
message: Unexpected `debugger` statement.
severity: error
rule:
  pattern: debugger
fix: ""
```

## Ban `TODO` / `FIXME` Comments

```yaml
# rule: no-todo.yml
id: no-todo
message: Remove TODO/FIXME before merging.
severity: warning
rule:
  pattern: $COMMENT
  kind: comment
  filters:
    - regex:
        source: $COMMENT
        regex: "(TODO|FIXME|HACK|XXX)"
```

## Ban `var` Declarations

```yaml
# rule: no-var.yml
id: no-var
message: Use `const` or `let` instead of `var`.
severity: error
rule:
  kind: variable_declaration
  pattern: var $NAME = $VALUE
fix: const $NAME = $VALUE
```

## Ban Arbitrary Custom Keyword

```yaml
# rule: ban-custom-keyword.yml
id: ban-custom-keyword
message: Avoid using "$KEYWORD" — prefer the sanctioned alternative.
severity: error
rule:
  pattern: $KEYWORD
  filters:
    - regex:
        source: $KEYWORD
        regex: "^forbidden_thing$"
```
