# Extraction contract — `spec.md` → `.feature`

**This file is the definition.** `javascript/extract-gherkin.cjs` and `python/extract_gherkin.py` are bindings of it; neither is the definition, and a port in a third language implements what is written here rather than reading either source. When behavior changes, this file changes first.

See [COMPOSITION.md](COMPOSITION.md) for what happens to the extracted files afterwards, and the skill's "Port parity" note for the cross-stack obligation.

## What extraction does

A spec is `openspec/specs/<capability>/spec.md` (source of truth) or `openspec/changes/<id>/specs/<capability>/spec.md` (delta). It is a standard OpenSpec Markdown spec: the **structure** lives in Markdown headings and only the **Given/When/Then steps** live inside column-0 ` ```gherkin ` fences.

Extraction **synthesizes** the Gherkin structure from the headings and copies fenced step lines verbatim, writing each `spec.md` to `acceptance-tests/.extracted/<same-relative-path>/spec.feature`.

Discovery covers `openspec/specs/` and each active `openspec/changes/<id>/specs/`, anchored to the literal basename `spec.md` — so `proposal.md`, `design.md` and `tasks.md` are excluded structurally. `openspec/changes/archive/` is excluded twice over: structurally, because the archive nests one level deeper than `changes/<id>/`, and again by a defensive filter on the collected paths.

`.extracted/` is **gitignored, wiped and rebuilt on every run, and never edited by hand**. The wipe is an invariant, not an optimization — a stale extraction would keep deleted or renamed capabilities executing.

## Line fidelity — the core invariant

Every input line maps to **exactly one** output line, so the extracted file has the IDENTICAL line count and **line N of the `.feature` is line N of the `.md`**.

gherkin-lint messages, runner failure locations and effective-spec line targeting (`spec.feature:27:33`) therefore all point at valid `spec.md` lines with zero translation. Read `.extracted/X/spec.feature:N` as `openspec/X/spec.md:N`, always.

**Never "improve" the extractor to collapse blank lines.** The whole toolchain leans on this.

## The mapping

Every line outside a fence is classified by exactly one row:

| Markdown line (outside any fence) | Emitted Gherkin line |
|---|---|
| `# <title>` (the single H1) | `Feature: <title>` |
| `## ADDED\|MODIFIED\|REMOVED\|RENAMED Requirements` | `  # @openspec: <OP>` (uppercased) |
| `### Requirement: <name>` | `  Rule: <name>` |
| `#### Scenario: <name>` | `    Scenario: <name>` |
| `#### Scenario Outline: <name>` | `    Scenario Outline: <name>` |
| any line inside a ` ```gherkin ` fence | copied **verbatim**, column unchanged |
| everything else — prose, requirement descriptions, `## Purpose`, `## Requirements`, other headings, fence markers, non-gherkin fence bodies, RENAMED `FROM:`/`TO:` bullets | blank line |

Heading matching is case-insensitive on the keywords (`Requirement:`, `Scenario:`, `Scenario Outline:`, the four operations); the emitted Gherkin keyword is always normalized to canonical casing.

Three consequences worth stating explicitly, because each is a decision rather than an accident:

- **Steps are not re-indented.** They keep the column the author wrote. Gherkin ignores indentation, verbatim copying keeps runner error columns pointing at real `spec.md` text, and this is why `indentation` is off in the pinned lint config.
- **Requirement description prose is blanked, not emitted as a `Rule:` description.** Free text that could accidentally parse as a Gherkin keyword is exactly the failure this format exists to avoid. The SHALL/MUST sentence stays visible in the rendered Markdown.
- **The delta operation is emitted once, at the section heading.** It is not repeated per requirement, so consumers must treat it as applying to every `Rule:` until the next marker. See [COMPOSITION.md](COMPOSITION.md).

## Fence mechanics

Fences follow CommonMark:

- An opener is 3+ backticks at **column 0** with info string **exactly** `gherkin`.
- The closer is at least as many backticks at column 0.
- Non-gherkin fences are tracked too, so a ` ```gherkin ` quoted inside a longer documentation fence cannot false-trigger.
- Gherkin docstrings delimited by ` ``` ` are safe: they are always indented, and the closer requires column 0.

A fence holds **only** steps (Given/When/Then/And/But), plus `Examples:` tables and docstrings.

## Edge cases and hard errors

All deliberate — silent drops are the failure mode to fear. **Both stacks classify every row identically**, with the same message and the same `file:line`:

| Case | Behavior |
|---|---|
| Unclosed fence | Error with `file:line` of the opener |
| No H1 title | **Hard error** — there would be no `Feature:` |
| More than one H1 | **Hard error** — a spec.md is exactly one capability |
| `#### Scenario:` with no fence before the next heading (or before EOF) | **Hard error** — a scenario with no steps would silently pass |
| `Feature:` / `Rule:` / `Scenario:` / `Scenario Outline:` / `Example:` inside a gherkin fence | **Hard error** — structure comes from headings. This is what an old-format spec hits, so the migration failure is loud |
| Zero gherkin fences in a `spec.md` | Fine — a REMOVED-only delta legitimately has no steps |
| `Examples:` and `Background:` inside a fence | Allowed — both legitimately belong there |
| Indented ` ```gherkin ` opener | **Hard error** — silently ignoring it would silently drop scenarios |
| ` ```gherkin extra-text ` | Not a gherkin opener (info string must be exactly `gherkin`) — treated as an ordinary fence, contents blanked |
| Non-gherkin fences (` ```js `, plain ` ``` `, 4+ backticks) | Tracked, contents blanked |
| Gherkin docstrings delimited by ` ``` ` | Safe — docstrings are indented; fence closers require column 0 |
| Files other than `spec.md` | Ignored — discovery is anchored to `spec.md` |
| Legacy `.feature` files under `openspec/` | Never run — extraction prints a WARNING naming them |
