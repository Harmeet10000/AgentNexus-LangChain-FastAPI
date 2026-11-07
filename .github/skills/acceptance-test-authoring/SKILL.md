---
name: acceptance-test-authoring
description: Use when creating or modifying acceptance tests, configuring cucumber-js or behave runners, writing or refactoring step definitions, linting executable Gherkin specs, choosing an acceptance stack, or implementing OpenSpec tasks that involve acceptance tests.
---

# Acceptance Test Authoring

When `spec-as-source` is active, the acceptance suite executes the Gherkin specs that live under `openspec/` against the running application. Specs are Markdown files named `spec.md`: Markdown headings carry the capability, requirement, and scenario structure, while column-0 `gherkin` fences contain only Given/When/Then steps. The runner extracts them into real `.feature` files on every run, synthesizing `Feature:`/`Rule:`/`Scenario:` from the headings.

This format is opt-in. Loading this skill without `spec-as-source` does not override the configured OpenSpec schema templates.

Everything in this file is stack-agnostic. Tool-specific filenames, dependencies, commands, and examples live in the stack packs.

## Choosing The Stack

The project's acceptance stack is declared as `stack:` in `openspec/config.yaml`:

```yaml
schema: intent-driven
stack: javascript      # javascript | python
```

Resolve it in this order:

1. Use `stack:` in `openspec/config.yaml`.
2. If absent and `acceptance-tests/` already exists, infer it from contents: `cucumber.cjs` means `javascript`, `behave.ini` means `python`; offer to record it.
3. Otherwise ask. Never guess silently, and never scaffold a runner without a recorded value.

When `spec-as-source` is active, adding `stack:` is a specs-zone edit under `openspec/`; follow that skill's mandatory BDD zone rules before editing it.

## Reference Files

| Stack | Pack | Runner |
| --- | --- | --- |
| `javascript` | [references/javascript/SETUP.md](references/javascript/SETUP.md) | cucumber-js |
| `python` | [references/python/SETUP.md](references/python/SETUP.md) | behave 1.2.7+ |

Each pack has a **Files to copy** table naming every destination filename and why it is load-bearing. Copy those files verbatim; they are the canonical runner.

Three files sit at the `references/` root because they are shared by both stacks:

| File | Role |
| --- | --- |
| [EXTRACTION.md](references/EXTRACTION.md) | Normative contract for `spec.md` to `.feature` extraction |
| [COMPOSITION.md](references/COMPOSITION.md) | Normative contract for which scenarios run |
| [gherkin-lintrc.json](references/gherkin-lintrc.json) | Shared lint configuration copied to `acceptance-tests/.gherkin-lintrc` |

The Markdown contracts are the definitions; the JavaScript and Python files are bindings. Change the relevant contract first, then both implementations.

## Spec Format And Extraction

When `spec-as-source` is active, draft specs from that skill's `references/spec.md`, which overrides the default schema template. A spec is `openspec/specs/<capability>/spec.md` (source of truth) or `openspec/changes/<id>/specs/<capability>/spec.md` (delta). Structure comes from Markdown headings; fences hold steps only.

- `# <capability>` is the single H1 title and becomes `Feature:`.
- `## ADDED|MODIFIED|REMOVED|RENAMED Requirements` is a delta section. The extractor emits one `# @openspec: <OP>` marker for the section.
- `### Requirement: <name>` becomes `Rule:`. Its SHALL/MUST description remains plain prose.
- `#### Scenario: <name>` or `#### Scenario Outline: <name>` becomes the corresponding Gherkin scenario. Each scenario must have a following gherkin fence before the next heading.
- Fences open with ` ```gherkin ` at column 0 and close with at least as many backticks at column 0.
- Fences contain only steps, `Examples:` tables, and docstrings. Gherkin structure keywords inside a fence are a hard error.

Extraction writes each `spec.md` to `acceptance-tests/.extracted/<same-relative-path>/spec.feature`, preserving exactly one output line per input line. `.extracted/` is gitignored, wiped and rebuilt on every run, and never edited by hand.

[references/EXTRACTION.md](references/EXTRACTION.md) is the normative definition of the complete mapping, fence mechanics, edge cases, and hard errors. Read it before modifying or porting an extractor.

## Runner Invariants

1. `acceptance-tests/` is an independent test project at the repo root. Its hooks boot the application before the suite and shut it down after, so the suite must run with a single command.
2. The default run executes the effective spec: every source-of-truth spec with every active change delta applied.
3. Superseded source-of-truth rules marked `MODIFIED` or `REMOVED` by active deltas must not reach the runner and must not be reported as skipped.
4. A green effective suite is the gate for sync/archive, and sync/archive must never change suite results.
5. Specs under `openspec/changes/archive/` must never execute.
6. Provide a source-of-truth-only regression run that executes `openspec/specs/` as-is.
7. Every test run generates an HTML report under `acceptance-tests/reports/`.
8. Verify composition whenever the runner config, extractor, or `openspec/` tree changes.

## Effective-Spec Composition

[references/COMPOSITION.md](references/COMPOSITION.md) is the normative definition. The important coupling with extraction is that a delta operation marker comes from a section heading and applies to every `Rule:` until the next marker, not only the first rule.

The JavaScript binding excludes superseded scenarios through cucumber-js line-targeted discovery. The Python binding blanks superseded rule blocks in generated `.extracted/` files because behave line selection would report them as skipped. Neither binding edits source specs.

## Port Parity

[references/EXTRACTION.md](references/EXTRACTION.md) and [references/COMPOSITION.md](references/COMPOSITION.md) are the contracts between stacks. A change to one implementation must be mirrored in the other and reflected in the relevant contract first. The strongest check is a cross-stack dry run on the same `openspec/` tree: the same scenario count and names.

## Linting Specs

Spec linting is shared across stacks: gherkin-lint over the extracted output with the pinned `.gherkin-lintrc`.

- Extract first, then lint; pass `.extracted` as a directory argument.
- gherkin-lint has no default rules; it requires `.gherkin-lintrc`.
- Reported line numbers are valid in source `spec.md` files.

Before an `acceptance-tests/` project exists, run the extractor from this skill:

```sh
node .agents/skills/acceptance-test-authoring/references/javascript/extract-gherkin.cjs openspec acceptance-tests/.extracted \
  && npx gherkin-lint --config .agents/skills/acceptance-test-authoring/references/gherkin-lintrc.json acceptance-tests/.extracted
```

The Python extractor is a drop-in substitute:

```sh
python .agents/skills/acceptance-test-authoring/references/python/extract_gherkin.py openspec acceptance-tests/.extracted
```

## Page Object Model

Step definitions must read as intent; all UI knowledge lives in page objects.

- Page objects live under `acceptance-tests/`, one per screen or flow.
- Page objects encapsulate routes, form field names, selectors, and ids.
- Parse responses with the stack's HTML parser, never with regexes over raw HTML.
- Page objects expose intent-level methods such as `open()`, `submit_signup(...)`, `error_message()`, and `confirmation_link()`.
- Step definitions contain no selectors, regexes, or URLs; only page-object calls and assertions.
- The World stays a thin HTTP client and state holder.

## Workflow Cadence

Implement one pending step definition at a time: run the suite so the step fails for the right reason, implement until it passes, then commit. The effective suite's red scenarios at propose time are the change's work list. Finish only when every scenario passes with zero pending or undefined steps and the HTML report is generated.
