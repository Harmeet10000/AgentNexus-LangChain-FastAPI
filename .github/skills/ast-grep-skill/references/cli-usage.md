## CLI Commands

| Command | Description |
|---------|-------------|
| `ast-grep -p <pattern>` | Search with a pattern string |
| `ast-grep run -p <p> -r <r>` | Search & replace (textual rewrite) |
| `ast-grep scan` | Run lint rules from `sgconfig.yml` project |
| `ast-grep test` | Run rule tests |
| `ast-grep new` | Create a new project (`sgconfig.yml` + rule dir) |
| `ast-grep new rule` | Scaffold a single lint rule |
| `ast-grep --json` | Output matches as JSON |
| `ast-grep --json=stream` | NDJSON output (one match per line) |
| `ast-grep --color` | Enable/disable colored output |
| `ast-grep -l <lang>` | Specify language (auto-detected from extension) |
| `ast-grep --debug-query` | Debug pattern parsing |
| `ast-grep scan --inspect summary` | Show project directory & config path |
| `ast-grep scan --inspect entity` | List all rules with their final severity |

### Flags

- `-p, --pattern <pattern>` — search pattern
- `-r, --rewrite <string>` — replacement string
- `-l, --lang <lang>` — target language
- `--interactive` — confirm each replacement interactively
- `-i, --ignore-case` — case-insensitive pattern matching
- `--json[=pretty|stream|compact]` — output format
- `--color` — force color output
- `--strictness <mode>` — matching strictness (cst, ast, smart, relaxed, signature)
- `--stdin` — read code from standard input (pipe mode)
- `--no-ignore` — don't respect .gitignore/hidden files; accepts `hidden`, `dot`, `global`, `vcs`

### JSON Output

ast-grep outputs match results as a JSON object. Three styles:

| Flag | Format | Use case |
|------|--------|----------|
| `--json` / `--json=pretty` | Indented array | Human-readable, small results |
| `--json=stream` | NDJSON (one object per line) | Large result sets, streaming |
| `--json=compact` | Single-line array | Programmatic consumption |

> **Gotcha:** `--json=stream` requires `=`. `--json stream` parses as `--json=pretty stream` where `stream` is a file path.

Match object schema (TypeScript):

```typescript
interface Match {
  text: string
  range: { byteOffset: { start, end }, start: { line, column }, end: { line, column } }
  file: string          // relative path
  lines: string
  replacement?: string
  metaVariables?: { single: Record<string, MetaVar>, multi: Record<string, MetaVar[]>, transformed: Record<string, string> }
}
```

For lint rules, `Match` extends with `ruleId`, `severity`, `note`, `message`. All positions are 0-based.

Pipe to jq for filtering:

```bash
ast-grep run -p 'Some($A)' -r 'None' --json | jq '.[].replacement'
```

### Stdin Mode

Read code from stdin (pipe mode) for use in shell pipelines:

```bash
curl -s https://example.com/page.html |
  ast-grep -p '<div> $$$ </div>' -l html --json --stdin |
  jq '.[] | .text'
```

**Caveats:**
- Must specify `--lang` (cannot infer from file extension)
- No `--interactive` mode
- For `scan`: must pass a single rule via `--rule` or `-r`
- Must pass `--stdin` flag and must not be running in a TTY

### Shell Completions

```bash
ast-grep completions zsh     # or: bash, fish, elvish, powershell
eval "$(ast-grep completions)"  # add to ~/.zshrc
```
