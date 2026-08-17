# Openspec conventions — authoring brief

Derived directly from `openspec/config.yaml` (68 lines) and `openspec/schemas/spec-gated/schema.yaml` (460 lines),
plus the templates under `openspec/schemas/spec-gated/templates/` and the live
`cognee-saul-memory-migration` change. Authored 2026-08-17 by the orchestrator (three subagent attempts at this
task died to transient API errors).

---

## FINDING 1 — the workflow has SIX artifacts, not four, and `tasks` is gated behind `review`

Earlier in this relay I described the cognee change as having "all four artifacts complete". That was wrong about
the schema. `schema.yaml` defines six artifact types with a hard dependency graph:

| id | generates | requires |
|---|---|---|
| `proposal` | `proposal.md` | — |
| `specs` | `specs/**/*.md` | `proposal` |
| `design` | `design.md` | `proposal` |
| `review` | `review.md` | `design` |
| `adr` | `adrs.md` (optional) | `design` |
| `tasks` | `tasks.md` | `specs`, `design`, **`review`** |
| `apply` | — | `tasks`; `tracks: tasks.md` |

`schema.yaml:394-396` is explicit: *"Read review.md first. If its VERDICT is CHANGES-REQUESTED, do not write
tasks until the listed items are fixed... Never write tasks against an unreviewed or failing plan."*
The tasks template repeats it at `templates/tasks.md:2`.

**Consequence for us:** each of the five new changes needs a `review.md` with a `VERDICT:` line before its
`tasks.md` is legitimate. The review is written *as a reviewer, not the author* (`schema.yaml:321`) — which maps
cleanly onto the relay: the review artifact is a natural fit for a **fresh subagent**, not for the author of the
proposal. That is an argument for having a subagent write each `review.md` rather than the orchestrator.

## FINDING 2 — the in-flight cognee change was authored under a DIFFERENT schema

`openspec/changes/cognee-saul-memory-migration/.openspec.yaml`:

```yaml
schema: spec-driven
created: 2026-06-15
```

But `openspec/config.yaml:1` says `schema: spec-gated`. The change declares `spec-driven`; the project is now
`spec-gated`. Consistent with that, the cognee change has **no `review.md`** — the gate did not exist when it was
written.

**Consequence for D8:** "extend the existing cognee change" is not a free option. It would have to be migrated to
`spec-gated` (add `review.md`, re-verify delta formatting against the current rules). Since that change is also
0/13 tasks done and 22 days old, **superseding it with a fresh change 4 is the cheaper and more honest path** —
with its two spec deltas harvested rather than edited in place.

## FINDING 3 — zero-delta changes are rejected unless explicitly opted out

`schema.yaml:49-59`: *"`openspec validate` rejects a change with zero deltas unless the change's `.openspec.yaml`
sets `skip_specs: true`. Use `skip_specs: true` only when no spec-level behavior changes (pure refactor, tooling,
docs)... Do not invent a requirement just to satisfy validation."*

**Consequence for change 0 (cleanup):** it is *mostly* a pure refactor — but it also merges alembic heads (making
a clean database deployable at all) and fixes AttributeError-on-first-request breaks. Both are observable
behaviour. So change 0 should carry real deltas, not `skip_specs: true`. Decide this per change, and never pad.

## FINDING 4 — every artifact must declare a change class S / M / L

A blockquote on line 1 of `proposal.md`, `design.md`, `tasks.md`, and `review.md`. The schema instructs that
section depth *scale to the class*, and explicitly forbids padding:

- `S` — single-file fix, config, dependency bump, docs.
- `M` — a feature confined to one module or subsystem.
- `L` — cross-cutting: multiple modules, data migration, security boundary, public API, or a new external dependency.

All five of our changes are **L** (multi-module + data migration). `design.md` is therefore mandatory for each
(`schema.yaml:264-273` lists the triggers: cross-cutting, new external dependency, significant data model change,
migration complexity — we hit all four).

---

## Artifact skeletons (verbatim structure)

### `proposal.md`

```markdown
> Change class (pick one): **S** single-file fix / config / bump / docs · **M** feature in one module · **L** cross-cutting (multi-module, migration, security, public API)

## Why

## What Changes

## Scope / Non-Goals

## Capabilities

### New Capabilities
- `<capability-path>`: <brief description of what this capability covers>

### Modified Capabilities
- `<existing-capability-path>`: <what requirement is changing>

## Impact

## Risks
```

Rules (`config.yaml:39-43`): Why in 1-2 sentences; What Changes as a specific bullet list; Capabilities names
kebab-case specs **after checking `openspec/specs/` for existing names first**; mark BREAKING explicitly; keep to
1-2 pages; no implementation detail.

### `design.md`

```markdown
> Change class: **S** / **M** / **L**. S with nothing left to decide → write `S change - no design required.` and stop.
> The proposal covers *why* and *what*; this covers *how*. Reference the proposal - do not restate it.

## Context

## Goals / Non-Goals

**Goals:**

**Non-Goals:**

## Decisions

## Risks / Trade-offs

## Migration Plan

## Open Questions
```

`Risks / Trade-offs` uses the literal format `[Risk] → Mitigation`. `Decisions` must include *alternatives
considered* per decision. `Open Questions` are for **genuinely deferrable** unknowns only — `schema.yaml:297-301`:
if a question would change the specs, the approach, or the task breakdown, **resolve it now, ask the user rather
than guess**.

### `specs/<capability-path>/spec.md`

```markdown
## Purpose

<50+ characters. NEW capabilities only. Delete for an existing capability.>

## ADDED Requirements

### Requirement: <name>
The system SHALL <normative statement>.

#### Scenario: <name>
- **WHEN** <condition>
- **THEN** <expected outcome>

## REMOVED Requirements

### Requirement: <name>
**Reason**: <why>
**Migration**: <what consumers do instead>
```

Hard rules:
- Delta operation headers are `## ADDED Requirements`, `## MODIFIED Requirements`, `## REMOVED Requirements`,
  `## RENAMED Requirements` (the last uses `FROM:`/`TO:`).
- **Scenarios MUST use exactly four hashtags.** `schema.yaml:164-165`: *"Using 3 hashtags or bullets will fail
  silently."* This is the single most dangerous formatting trap here — it does not error, it just drops.
- Every requirement MUST have ≥1 scenario. Use SHALL/MUST; avoid should/may.
- `MODIFIED` must copy the **entire** existing requirement block, from `### Requirement:` through all scenarios;
  header text must match the original whitespace-insensitively. Partial content silently loses detail at archive.
  If you are adding a concern rather than changing behaviour, use `ADDED` instead.
- `REMOVED` must include both **Reason** and **Migration**.
- `## Purpose` on a delta for an *existing* capability is ignored — to change an existing Purpose, edit
  `openspec/specs/<path>/spec.md` directly.
- Specs are behaviour contracts: no internal class/function names, no library choices, no step-by-step
  implementation. Quick test (`schema.yaml:109-111`): if the implementation can change without changing
  externally visible behaviour, it does not belong in the spec.

Real example from this repo (`changes/cognee-saul-memory-migration/specs/saul-cognee-final-report-write/spec.md`):

```markdown
## ADDED Requirements

### Requirement: Saul persist_memory writes approved final reports to Cognee
The Saul `persist_memory` node SHALL write the approved final report directly to Cognee after human approval.

#### Scenario: Approved final report is persisted
- **WHEN** the Saul graph reaches `persist_memory` after human approval
- **THEN** the node SHALL persist the final report to Cognee
```

Note: it references a node name (`persist_memory`) in the requirement text. That is a mild violation of the
"no internal names" rule that the house has tolerated — treat the house style as permitting *graph node names*
(which are part of the observable workflow) while still barring class and function names.

### `review.md`

```markdown
> Change class: **S** (short review) · **M** (normal checklist) · **L** (full checklist + verification matrix).
> Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness
## Correctness
## Standards
## Risk

## Verdict

**VERDICT:** `APPROVED` | `CHANGES-REQUESTED` | `INFO`
```

The four axes are prescribed (`schema.yaml:337-351`). **Standards** checks against
`.opencode/instructions/` — RESULT-PATTERN, EXCEPTION-RULES, PYTHON-TYPING-RULES, ARCHITECTURE-RULES — plus
`SecretStr.get_secret_value()`, `APIResponse` + `http_error()`, and **no `match/case` on Success/Failure**.

### `tasks.md`

```markdown
> Change class: **S** (1-3 tasks) · **M** (grouped sections) · **L** (grouped sections, each task verifiable).
> Never start with tasks until review.md has a `VERDICT:` that is not `CHANGES-REQUESTED`.

## 1. <Task Group Name>

- [ ] 1.1 <Task description - state verification: what command or check proves it's done?>
- [ ] 1.2 <Task description>

## 2. <Task Group Name>

- [ ] 2.1 <Task description>
```

`config.yaml:49-52`: checkboxes `- [ ] N.M Description` under `## N` headings; each small enough for one
session; **ordered by dependency**; each verifiable. `schema.yaml:406-409`: the apply phase *parses* the checkbox
format — tasks not using `- [ ]` are not tracked.

The live cognee `tasks.md` is the house style at 23 lines / 4 groups / 15 tasks — terse one-liners, no citations.
Our changes are much larger, and the template asks each task to **state its verification**, which the cognee file
does not do. Follow the template, not that file: the relay's "every step carries a Proof" rule and openspec's
"each task must be verifiable" rule are the same requirement, so the Proof goes in the task text.

### `adrs.md` (optional)

Sections: **Status** (Proposed | Accepted | Superseded) · **Context** · **Decision** · **Rationale /
Alternatives** · **Consequences**. Write one ONLY for a decision that outlives the change — a new dependency, a
data-model or interface contract others build on, a hard-won trade-off. Otherwise write the single line
`No durable architectural decision in this change.` (`schema.yaml:369-371` — "Missing non-decisions is not a defect".)

**Likely ADR candidates in this refactor:** the Graphiti/Cognee role boundary (D2) and the
`UnifiedDocument`/`UnifiedChunk` schema contract. Both outlive their change.

---

## `.openspec.yaml`

Two required keys, plus one conditional:

```yaml
schema: spec-gated      # must match openspec/config.yaml:1 — the cognee change says spec-driven and is stale
created: 2026-08-17
skip_specs: true        # ONLY when there is genuinely no spec-level behaviour change
```

---

## Project context injected into every artifact

`config.yaml:6-35` injects the stack and conventions into every artifact instruction. Two entries bear directly
on this refactor and will be checked by `review.md`'s Standards axis:

- *"Dependencies use `typing.Annotated` (Depends/Query/Header); shared clients live in lifespan and are read from
  `connection.app.state`."* — this is the rule that the `app.state.saul_graph` / `storage` / `mongodb` breaks
  violate, and the rule todo (f) (graph in `app.state`) is trying to satisfy.
- *"Async-first: all I/O through async clients; bounded fan-out via `asyncio.gather`; **no blocking calls in async
  functions**."* — this is the rule `parser.py:25`'s synchronous `converter.convert()` inside an `async def`
  violates.

So two of our findings are not merely bugs; they are documented convention violations, which strengthens their
place in change 0 / change 1.

`config.yaml:54-61` (apply guidance) prescribes the per-change verification commands:
`uv sync` → `uv run ruff format src/` → `uv run ruff check src/` → `uv run ty check src/` → `ast-grep scan`,
checking tasks off in `tasks.md` as they complete.

---

## Fog

- **No `openspec` CLI was verified** by me. `schema.yaml` repeatedly references `openspec validate` and
  `openspec validate --strict`, so a CLI is presumed to exist, but the exact invocation and whether the repo
  currently passes is unconfirmed. A sibling agent (`conventions-openspec-namespace.md`) was tasked with this.
- **Change-ID naming** is unresolved here: live changes are bare slugs, archived ones carry `YYYY-MM-DD-`
  prefixes. Whether archiving renames the directory determines what we name five new changes today. Also assigned
  to the namespace agent.
- **The capability namespace** (~22 dirs under `openspec/specs/`) is not enumerated in this brief. Deltas must
  target existing capability names where one fits; the proposal rules say to check first. Namespace agent owns this.
- Whether `review.md` is *enforced* by the CLI or merely instructed is unknown. If unenforced, our changes could
  ship without it — but `schema.yaml` is unambiguous that they should not.
