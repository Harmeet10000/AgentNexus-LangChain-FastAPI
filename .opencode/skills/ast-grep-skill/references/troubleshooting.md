## Troubleshooting

**Rule matches nothing.**
Check `ast-grep scan --rule path/to/rule.yml` on a single file. Use `ast-grep ls` to confirm the rule loads. Verify tree-sitter has a grammar for the target language.

**`$META` captured no text.**
Verify the pattern position — meta variables inside nested patterns (`has`, `inside`, etc.) may not be in scope for `fix`.

**ast-grep doesn't find patterns I know exist.**
1. Confirm the file type — run with `--lang` explicit.
2. Try `--strictness relaxed` to loosen matching.
3. The file may be excluded by `ignores` or `.gitignore`. Run with `--no-ignore hidden` to check.
4. Tree-sitter grammar may not support that syntax version (e.g., TypeScript 5.0 decorators).

**Rule works in isolation but not in `scan`.**
Check `sgconfig.yml` rule dirs, file patterns, and `ignores` rules.

**`--json=stream` not working.**
Use `=` syntax: `--json=stream`. Without `=`, the next arg is treated as a positional file parameter.

**Not all rules fire in a scan run.**
Rules may be disabled by severity overrides or suppressed via `ast-grep-ignore` comments. Run `ast-grep scan --inspect entity` to see final severity per rule. Rules with no matching files are hidden.

**Regex is slow.**
Avoid backtracking patterns in `regex:` fields. Use `kind:` + `pattern:` for structural matching where possible — AST matching is faster than regex on source text.

**How to write a rule that fires on both `import` and `require`?**
Use `any:` with two pattern branches:

```yaml
rule:
  any:
    - pattern: import { $$$ } from '$PKG'
    - pattern: const { $$$ } = require('$PKG')
```

**How to detect a pattern in JSX?**
Use `lang: tsx` (or `jsx`) in `sgconfig.yml` or `--lang tsx` on CLI. JSX node kinds include `jsx_element`, `jsx_self_closing_element`, `jsx_expression`, `jsx_fragment`, etc.

**Can ast-grep rewrite across multiple files?**
Yes — the `rewrite` command handles multi-file rewrites. Pass `--update-all` to apply `fix` automatically to all matching files (with `.bak` backup).

**How to match only top-level (not nested) statements?**
Use the negation of `inside`:

```yaml
filters:
  - not:
      inside:
        kind: function_declaration
        stopBy: end
  - not:
      inside:
        kind: class_declaration
        stopBy: end
```

**How to match string literals with specific content?**
Use `kind: string` + `regex`:

```yaml
rule:
  kind: string
  regex: "your-content-pattern"
```

**How to test rules without a config file?**
Use inline rules:
```bash
ast-grep scan --inline-rules='{"rules":[{"id":"test","rule":{"pattern":"..."},"severity":"error"}]}'
```

**`$` interpreted by shell.** Use single quotes `'$PATTERN'`.

**Pattern fails to parse.** Use object-style `pattern: {context: ..., selector: ...}`.

**Slow search despite correct matching.** Add `kind` to narrow node types before regex.

**Rule not found in scan.** Check `ruleDirs` in `sgconfig.yml` points to correct path.

**Test cases not found.** Verify `testConfigs.testDir` in `sgconfig.yml`.

**Language not supported.** Use `customLanguages` in config to alias existing parser.

**Want to see AST node kinds.** Use playground, or `sg run --debug-query` with a simple pattern.

