## 1. Matching Strictness & Severity (SKILL.md)

- [ ] 1.1 Add matching strictness section with decision table (`cst`/`ast`/`smart`/`relaxed`/`signature`, when to use each, CLI `--strictness` flag, YAML `strictness` field)
- [ ] 1.2 Add severity override flags (`--error`, `--warning`, `--info`, `--hint`, `--off`) to existing Lint Rules section
- [ ] 1.3 Add inline suppression subsection (same-line, next-line, specific-rule, file-level conditions)
- [ ] 1.4 Add built-in suppression rules (`unused-suppression`, `no-suppress-all`) with enablement conditions
- [ ] 1.5 Add `inspect` flag references (`--inspect summary`, `--inspect entity`)

## 2. Transform Operators Deep-Dive (SKILL.md + references)

- [ ] 2.1 Expand "Rewrite & Transform" section with brief examples of all operators: `replace`, `substring`, `convert` (7 case types), `capitalize`, `lowercase`, `uppercase`, `strip`, `convert_string`, `rewrite`
- [ ] 2.2 Document both YAML dict form and string shorthand form for each operator
- [ ] 2.3 Create `references/transform-operators.md` with full operator reference including all `convert` case-type table with examples

## 3. CLI JSON & Recipes (SKILL.md)

- [ ] 3.1 Add JSON output subsection with match object schema (both `Match` and `RuleMatch` interfaces)
- [ ] 3.2 Document all three output styles (`pretty`, `stream`, `compact`) with the `=` sign gotcha
- [ ] 3.3 Add stdin mode documentation with pipe examples and caveats
- [ ] 3.4 Add shell completions section (`ast-grep completions zsh`)
- [ ] 3.5 Add `sg config`/project inspection mentions

## 4. Advanced Rewrite (SKILL.md)

- [ ] 4.1 Add FixConfig subsection with `template`, `expandStart`, `expandEnd` for comma-separated list handling
- [ ] 4.2 Add `rewriters` and `rewrite` transform subsection with barrel import expansion example

## 5. ESQuery Selectors (SKILL.md)

- [ ] 5.1 Add ESQuery selector section with usage example (`esquery:` field) and reference table mapping common selectors to ast-grep equivalents

## 6. CI/CD Integration (SKILL.md)

- [ ] 6.1 Add CI/CD section with GitHub Action workflow YAML and exit code behavior explanation

## 7. API Usage Recipes (SKILL.md)

- [ ] 7.1 Expand API Usage with 3 JS patterns: `find`, `findAll`, complex `NapiConfig` rule
- [ ] 7.2 Expand API Usage with 3 Python patterns: `parse`, `find_all`, `get_match("A").text()`

## 8. Catalog Patterns (references/catalog/)

- [ ] 8.1 Create `references/catalog/typescript.md` with import extraction, logical assignment, useState type removal, Promise.all await, console.log→logger, barrel import patterns
- [ ] 8.2 Create `references/catalog/python.md` with remove-async-await, SQLAlchemy mapped_column, prefer-generator-expressions, migrate-openai-sdk patterns
- [ ] 8.3 Create `references/catalog/rust.md` with avoid-duplicated-exports, redundant-unsafe-function, get-digit-count patterns
- [ ] 8.4 Create `references/catalog/go.md` with match-function-call, find-func-declaration-with-prefix, defer-func-call-antipattern patterns
- [ ] 8.5 Create `references/catalog/html.md` with i18n-extract, upgrade-ant-design-vue patterns
- [ ] 8.6 Create `references/catalog/other.md` with No-unused-vars (Java), detect-path-traversal (Ruby), find-key-value (YAML) patterns
- [ ] 8.7 Add catalog summary table to SKILL.md with language column, pattern column, link column

## 9. FAQ Expansion (SKILL.md)

- [ ] 9.1 Expand Troubleshooting/FAQ section to 15+ entries covering: pattern not matching, shell escaping, language detection, meta variable capture, slow search, rules not firing, test discovery, strictness tuning, inline suppression, custom languages, JSON format, API usage, editor integration, multi-file rules, performance

## 10. Rule Reference Expansion (references/rule_reference.md)

- [ ] 10.1 Add strictness field to rule grammar reference
- [ ] 10.2 Add `utils` field to rule schema
- [ ] 10.3 Add `transform` operators table to reference (linking to full transforms doc)
- [ ] 10.4 Add `fix` with FixConfig schema
- [ ] 10.5 Add `rewriters` field schema
- [ ] 10.6 Add ESQuery field to rule schema
- [ ] 10.7 Add `url`, `metadata` fields to lint rule schema

## 11. Sync to .opencode/skills/ast-grep/

- [ ] 11.1 Copy all changes to `.opencode/skills/ast-grep/` (both files + new reference files)
