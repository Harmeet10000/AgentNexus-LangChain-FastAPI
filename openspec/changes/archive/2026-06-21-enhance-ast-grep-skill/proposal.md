## Why

The current ast-grep skill (820 lines across 2 files) covers core concepts but omits many high-value sections from the original 17,315-line flat file. Users frequently need real-world catalog patterns, detailed transform operators, matching strictness control, CLI JSON output patterns, and API usage recipes — all present in the source file but absent from the skill. Adding these closes the gap between "understanding concepts" and "actually using ast-grep on real code."

## What Changes

- Add **catalog patterns** section: per-language real-world recipes (JS/TS, Python, Rust, Go, HTML, Ruby, C, C++, Kotlin, YAML)
- Add **matching strictness** section: `cst`/`ast`/`smart`/`relaxed`/`signature` modes with when to use each
- Add **transform operators deep-dive**: `replace`, `substring`, `convert` (with case conversion table), `rewrite` for multi-node transformations
- Add **JSON output patterns**: streaming, piping to jq, match object schema, `--json=stream` gotcha
- Add **severity & suppression details**: severity override flags, inline suppression rules, file-level suppression, `unused-suppression`, `no-suppress-all` rule
- Add **`sg run` vs `sg scan` rewrite details**: interactive mode, stdin mode, `--rule`, `--inline-rules`, shell completions
- Add **CI/CD integration**: GitHub Action setup, exit codes for pipeline use, `--error`/`--warning` flag usage
- Add **API usage recipes**: common JS `@ast-grep/napi` patterns, common Python `ast_grep_py` patterns
- Add **FixConfig & rewriter rules**: `expandStart`/`expandEnd`, `rewriters` with `rewrite` transform for multi-node transforms
- Add **ESQuery selector compatibility** for eslint users
- Add **FAQs and troubleshooting** entries for common user problems

## Capabilities

### New Capabilities

- **catalog-patterns**: Real-world search/rewrite recipes grouped by language (JS/TS, Python, Rust, Go, HTML, Ruby, C/C++, Java, Kotlin, YAML)
- **transform-operators**: All string manipulation operators with examples (replace, substring, convert case types, rewrite with rewriters)
- **matching-strictness**: `cst`/`ast`/`smart`/`relaxed`/`signature` modes, when to use each, how strictness affects pattern matching
- **cli-recipes**: JSON output streaming & piping, stdin mode, shell completions, `--json` match object schema
- **severity-and-suppression`: Severity level override flags, inline `ast-grep-ignore` comments, file-level suppression, `unused-suppression` and `no-suppress-all` built-in rules
- **ci-integration**: GitHub Action workflow setup, exit code behavior, severity-based pipeline gating
- **api-recipes**: Common JavaScript (`@ast-grep/napi`) and Python (`ast_grep_py`) patterns for complex transformations
- **advanced-rewrite**: `FixConfig` with `expandStart`/`expandEnd`, `rewriters` with `rewrite` transform, multi-node replacements (barrel import example)
- **esquery-selectors**: ESQuery-compatible selector syntax for users migrating from eslint
- **faq**: Expanded FAQ covering pattern parsing, rules not firing, slow search, inline suppression, language detection, AST debugging

### Modified Capabilities

(none — first version of this skill)

## Impact

- Two files modified: `SKILL.md` (~529 → ~850 lines) and `references/rule_reference.md` (~291 → ~350 lines)
- Both locations updated: `.github/skills/ast-grep-skill/` and `.opencode/skills/ast-grep/`
- No code, dependencies, or configuration changes outside the skill
