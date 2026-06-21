# Catalog: React Hooks

## No Hook Calls Inside Conditionals

React hooks must not be called inside conditions, loops, or early returns.

```yaml
# rule: react-hooks-rules-of-hooks.yml
id: react-hooks-rules-of-hooks
message: React Hook "$HOOK" is called conditionally.
severity: error
rule:
  pattern: $HOOK($$$)
  filters:
    - regex:
        source: $HOOK
        regex: "^use[A-Z]"
    - inside:
        stopBy: end
        any:
          - kind: if_statement
          - kind: for_statement
          - kind: for_in_statement
          - kind: while_statement
          - kind: switch_case
```

**Variant — detect hooks in callbacks:**

```yaml
inside:
  stopBy: end
  kind: arrow_function
```

**Variant — exhaustive deps (pattern sketch):**

Use `has` on `useEffect` calls to check that the deps array argument includes all reactive variables found inside the effect callback.
