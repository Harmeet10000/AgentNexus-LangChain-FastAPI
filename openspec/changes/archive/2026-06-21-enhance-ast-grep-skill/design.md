## Context

This change extends the existing ast-grep skill at `.github/skills/ast-grep-skill/` and `.opencode/skills/ast-grep/` with content extracted from the original 17,315-line flat file at `.github/skills/ast-grep-skill.md`. The current skill covers core concepts (pattern syntax, atomic/relational/composite rules, lint rules, project setup, testing) but lacks real-world catalog patterns, detailed operator references, CLI output patterns, API usage recipes, and other intermediate-to-advanced content that users encounter during actual usage.

### Current State

- `SKILL.md` (529 lines): 17 sections covering overview through API usage
- `references/rule_reference.md` (291 lines): Grammar tables, schemas, operator reference
- The original flat file contains ~371 sections across guide, reference, catalog, and advanced domains

### Key Constraint

The skill must remain a quick-reference tool, not a full documentation site. Every added section must justify its inclusion by being a) commonly needed during daily ast-grep use, and b) not obvious from the CLI's `--help` output.

## Goals / Non-Goals

**Goals:**
- Add catalog patterns for the 5 most-used languages (JS/TS, Python, Rust, Go, HTML)
- Add matching strictness modes with decision table
- Add transform operators with examples for all 7+ operators
- Add CLI JSON output patterns (stream, jq, match object schema)
- Add severity/suppression details (inline, file-level, built-in rules)
- Add CI/CD integration (GitHub Action)
- Add API usage recipes (JS + Python)
- Add FixConfig and rewriter rules for multi-node transformations
- Add ESQuery selector reference
- Expand FAQ to 15+ entries

**Non-Goals:**
- Converting the entire flat file (that would make the skill larger than the reference)
- Adding every catalog pattern (only the highest-value ones per language)
- Adding contributing docs or playground documentation
- Adding custom language or language injection guides (too niche)
- Breaking existing SKILL.md section structure (we extend, not restructure)

## Decisions

| Decision | Option A | Option B | Choice & Rationale |
|----------|----------|----------|--------------------|
| Where to put catalog patterns | New `catalog/` dir alongside `references/` | Inline in SKILL.md | **Inline with tabular summary.** SKILL.md gets a compact language-vs-pattern table; full rule YAML goes in a `references/catalog/` dir. Keeps SKILL.md scannable. |
| How to present strictness | Single paragraph | Decision table | **Decision table** with example, use-case, and tradeoff per mode. This is a config choice users face — table is fastest to scan. |
| Transform operator detail | Inline table | Separate reference file | **Both.** Brief examples in SKILL.md under "Rewrite & Transform"; full operator reference (with all `convert` case types, `rewrite` rewriter config) in `references/transforms.md`. |
| API recipes depth | One example each | 5+ per language | **3 per language** — find, findAll, complex rule with NapiConfig. Enough to cover the pattern; users link to full API docs for more. |
| CI/CD section size | Full GitHub Action workflow | One-block reference | **One-block** — the GitHub Action YAML plus exit code explanation. CI setup is well-documented; the skill just needs the hook. |
| Catalog examples selection | All ~40 catalog entries | Top 20 highest-value | **Top 20** — picked by cross-reference with StackOverflow/discord frequency: import patterns, console.log, async/await, null checks, type assertions, useState type, barrel imports, etc. |

## Risks / Trade-offs

- **Size growth**: SKILL.md grows from ~529 to ~850 lines. Mitigation: catalog rules go into reference files, not the main skill.
- **Staleness**: Catalog patterns for specific frameworks (React, Vue, SQLAlchemy) may drift as those frameworks evolve. Mitigation: mark framework-specific patterns with "Last verified: 2026-06" in comments.
- **Duplicate knowledge**: Transform operators appear in both SKILL.md and references/transforms.md. Mitigation: SKILL.md has brief examples, reference has full detail — explicit cross-reference link in both places.
- **Openspec complexity**: 10 spec files for 10 capabilities may feel heavy for a documentation change. Mitigation: specs are lightweight (1-3 requirements each) — they exist to define scope, not to over-document.
