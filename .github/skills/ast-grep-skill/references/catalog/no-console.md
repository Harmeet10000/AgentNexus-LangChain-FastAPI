# Catalog: No `console.*`

## Disallow Console Logging in Production

Ban `console.log`, `console.warn`, `console.error`, `console.info`, etc.

```yaml
# rule: no-console.yml
id: no-console
message: Unexpected console statement.
severity: error
rule:
  pattern: console.$METHOD($$$)
  filters:
    - not:
        inside:
          kind: comment
utils:
  order: |
    function getAllMethods(x) {
      return x.getMatch('METHOD').text();
    }
```

**Variant — allow specific methods:**

```yaml
filters:
  - not:
      regex:
        source: console.$METHOD($$$)
        regex: "^(warn|error)$"
        on: $METHOD
```

**Auto-fix (strip):** Replace the match with empty string via `fix: ""`. For comment-and-keep style, wrap in a comment instead.
