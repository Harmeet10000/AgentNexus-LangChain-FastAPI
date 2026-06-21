# Catalog: Type Patterns

## Detect Legacy `React.FC`

```yaml
# rule: no-react-fc.yml
id: no-react-fc
message: Prefer explicit `Props` type over `React.FC<Props>`.
severity: hint
rule:
  pattern: React.FC<$TYPE>
```

## Detect Protected Route Typing Mismatch

```yaml
# rule: typed-route.yml
id: typed-route
message: Route handler missing typed `Request<Params>`.
severity: hint
rule:
  pattern: app.get($PATH, $HANDLER)
  has:
    field: arguments
    nthChild: 1
    kind: arrow_function
```

## Detect `any` Type Leaks

```yaml
# rule: no-explicit-any.yml
id: no-explicit-any
message: Avoid explicit `any` type.
severity: warning
rule:
  any:
    - kind: type_annotation
      has:
        pattern: any
    - pattern: as any
    - pattern: : any
```
