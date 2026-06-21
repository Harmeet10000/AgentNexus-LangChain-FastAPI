## Testing Rules

### Test File Structure

# rule-tests/no-await-in-loop-test.yml
id: no-await-in-loop
valid:
  - for (let a of b) { console.log(a) }
  - Promise.all(items.map(async item => await item))
invalid:
  - async function foo() { for (var bar of baz) await bar; }
```

### Running Tests

```bash
ast-grep test
ast-grep test --skip-snapshot-tests   # skip file-level snapshot
```

Output shows: `PASS`, `FAIL`, and per-case status (validated/noisy/missing/reported).

```bash
# Find all type assertions (TypeScript)
ast-grep -p '$x as $T' -l ts
ast-grep -p 'const $NAME = require($PATH)' -r 'import $NAME from $PATH' -l js

# Find empty catch blocks
ast-grep -p 'catch ($_) { }'

# Find console.log/console.debug
ast-grep -p 'console.log($$$)' --interactive

# Nullish coalescing conversion
ast-grep -p '$A = $A ?? $B' -r '$A ??= $B'
```

