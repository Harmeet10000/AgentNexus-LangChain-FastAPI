---
name: spec-as-source
description: Use when a user or project explicitly adopts OpenSpec spec-as-source, fenced Gherkin specifications as executable acceptance tests, or acceptance-test-first OpenSpec tasks.
---

# Spec As Source

## Overview

Use OpenSpec `spec.md` files as the executable source of truth. When this skill is active, fenced-Gherkin authoring, acceptance-first task ordering, spec-first implementation, and BDD zone isolation are one workflow.

**REQUIRED SUB-SKILL:** Use `gherkin-authoring` when drafting or modifying fenced Gherkin.

**REQUIRED SUB-SKILL:** Use `acceptance-test-authoring` when configuring or modifying the acceptance runner, extractor, linting, step definitions, or page objects.

## Activation

This workflow is opt-in. Apply it only when the user or project instructions explicitly require `spec-as-source`.

Once active, the BDD zone rules below are mandatory. Do not activate or disable zone isolation separately.

## Artifact References

- When this skill is active, its artifact references override the corresponding
  templates from the configured OpenSpec schema.
- Draft each `spec.md` from [references/spec.md](references/spec.md), not from
  `openspec/schemas/intent-driven/templates/spec.md`.
- Draft `tasks.md` from [references/tasks.md](references/tasks.md), not from the
  schema task template, preserving acceptance-test-first ordering.

## BDD Zone Rules

1. Code is written only for an active change under `openspec/changes/` whose scenarios describe the behavior.
2. A unit of work edits either the specs zone (`openspec/`) or the code zone (everything else), never both while changes are uncommitted.
3. Before crossing zones, finish and commit or stash the current zone.
4. Never commit `openspec/` files and code-zone files together.
5. Any file named `tasks.md` is exempt and may be edited from either zone.
6. Never revert or discard user-owned changes without explicit permission.

Use `git status --porcelain` before editing. Non-`tasks.md` paths under `openspec/` select the specs zone; other changed paths select the code zone.

## Workflow

1. Draft and validate the proposal and fenced-Gherkin delta specs.
2. Commit or stash specs-zone work before touching code.
3. Establish the failing acceptance scenario or pending step before implementation.
4. Implement the smallest change that makes the scenario pass.
5. Keep the effective acceptance suite green with no pending or undefined steps.
6. Commit or stash code-zone work before returning to specifications for sync or archive.

## Red Flags

- Implementing behavior with no active spec change.
- Editing `openspec/` and code-zone files in the same uncommitted unit of work.
- Writing implementation tasks before acceptance-test tasks.
- Treating generated `.feature` files as the source instead of `spec.md`.
- Running source-of-truth specs without active deltas when validating a change.

Stop and restore the workflow boundary when any red flag occurs.
