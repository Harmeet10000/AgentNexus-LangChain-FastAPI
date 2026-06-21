## Project Setup

### `sgconfig.yml`

```yaml
ruleDirs:
  - rules
testConfigs:
  - testDir: rule-tests
```

- ast-grep discovers `sgconfig.yml` by walking up from cwd
- Override with `--config path/to/sgconfig.yml`
- `ast-grep scan` requires `sgconfig.yml`; `ast-grep run` does not

### CI/CD Integration

Run ast-grep in GitHub Actions:

```yaml
on: [push]
jobs:
  sg-lint:
    runs-on: ubuntu-latest
    name: Run ast-grep lint
    steps:
      - uses: actions/checkout@v4
      - uses: ast-grep/action@v1.4
```

### Advanced Config

```yaml
ruleDirs:
  - rules
testConfigs:
  - testDir: rule-tests
utilsDir: utils           # shared utility rule directory
languageGlobs:
  - glob: "*.tsx"
    language: tsx         # treat .tsx files as TSX
customLanguages:          # for non-standard languages
  - name: prisma
    language: rust         # reuse tree-sitter parser
    extensions: [prisma]
```

