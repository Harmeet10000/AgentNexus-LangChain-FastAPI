## API Usage

For complex transformations beyond what rule language can express:

### JavaScript (`@ast-grep/napi`)

```javascript
import { parse, Lang } from '@ast-grep/napi';

const ast = parse(Lang.TypeScript, source);
const root = ast.root();

// Find first match
const node = root.find('console.log($A)');
if (node) console.log(node.getMatch('A').text());

// Find all matches
root.findAll('console.log($$$)').forEach(n => console.log(n.text()));

// Complex rule via NapiConfig
root.findAll({
  rule: {
    pattern: 'console.log($$$)',
    inside: { kind: 'arrow_function', stopBy: 'end' },
  },
  constraints: {},
});
```

### Python (`ast_grep_py`)

```python
from ast_grep_py import parse

ast = parse("typescript", source)
root = ast.root()

# Find all matches
for node in root.find_all("console.log($$$)"):
    print(node.text())

# Access captured variables
node = root.find("console.log($A)")
if node:
    print(node.get_match("A").text())

# Use with complex config
matches = root.find_all({
    "rule": {"pattern": "await $_", "inside": {"kind": "for_in_statement"}},
})
```

### Rust (`ast_grep_core`)

```rust
use ast_grep_config::from_str;
```
