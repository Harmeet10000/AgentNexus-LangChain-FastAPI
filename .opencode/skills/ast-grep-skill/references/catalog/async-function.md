# Catalog: Async Function Patterns

## No `await` Inside Loop

Detect potential `await` calls inside loops where serial execution may be unintended and suggest `Promise.all`.

```yaml
# file: no-await-in-loop.yml
id: no-await-in-loop
message: Avoid `await` inside loops — use `Promise.all` or move outside
note: |
  Consider collecting promises and using `Promise.all(...)` for parallel execution.
severity: warning
rule:
  pattern: await $FUNC($$$)
  inside:
    stopBy: end
    any:
      - pattern: for ($VAR of $ITER) { $$$ }
      - pattern: for ($$$) { $$$ }
      - kind: for_in_statement
      - kind: while_statement
```

**Detection:** Finds `await` calls nested inside any loop (for, for-of, for-in, while).

**Fix strategy:** Collect promises into an array, use `Promise.all` outside the loop.
