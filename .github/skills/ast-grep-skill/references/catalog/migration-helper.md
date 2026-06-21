# Catalog: Migration Helpers

## CJS → ESM: `require` to `import`

Transform `const X = require('pkg')` → `import X from 'pkg'`.

```yaml
# rule: require-to-import.yml
id: require-to-import
message: Prefer ESM `import` syntax over CJS `require`.
severity: info
rule:
  pattern: const $NAME = require('$PATH')
transform:
  NAME:
    capitalize: {}
  PATH:
    replace:
      source: $PATH
      replace: "^(\\./)?(.+)$"
      by: "$2"
fix: import $NAME from '$PATH'
```

## Destructure Require → Named Imports

`const { a, b } = require('pkg')` → `import { a, b } from 'pkg'`.

```yaml
id: destructure-require-to-import
severity: info
rule:
  pattern: const { $$$NAMES } = require('$PATH')
fix: import { $$$NAMES } from '$PATH'
```

## Strip Lodash Full Import

`import _ from 'lodash'` → warn to use per-method imports.

```yaml
id: no-full-lodash
message: Prefer per-method lodash imports (`lodash/map`) over full import.
severity: warning
rule:
  pattern: import $_ from 'lodash'
```
