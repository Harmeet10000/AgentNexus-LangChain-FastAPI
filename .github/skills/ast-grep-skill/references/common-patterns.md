## Common Patterns


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
