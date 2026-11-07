# Discovery: <product / PRD name>

> Status: in-progress — next phase: <ingest | personas | journey-map | moscow | stories>   *(or: complete)*
> Created: <YYYY-MM-DD> · Last revised: <YYYY-MM-DD>

> Release plan produced by the discovery skill. Resume or revise by re-running the skill.
> To build: run `/opsx:propose` and ask it to use the next unchecked story below.
> One story = one OpenSpec change (proposal ≈ 200 words). Create one at a time.

## Sources

- <YYYY-MM-DD> — <PRD link/path, or a one-paragraph summary of the input>

## Personas

### <Persona name>

- **Who**: <who they are>
- **Goal**: <what they're trying to reach>
- **Pain today**: <what's broken or missing for them now>
- **Success looks like**: <the observable signal they got value>

## Journey Map

<Persona name> journey:

```
  <Stage 1> ──► <Stage 2> ──► <Stage 3> ──► <Stage 4>
      │             │             │             │
     gap         partial      supported        gap
```

1. **<Stage>** — <what happens> — [supported | partial | gap]
2. **<Stage>** — <what happens> — [supported | partial | gap]

## MoSCoW

### Must

- <capability> — <why, tied to a persona's journey>

### Should

- <capability> — <why>

### Could

- <capability> — <why>

### Won't (this release)

- <capability> — <why it's deferred>

## Stories

Ordered release checklist. One story = one OpenSpec change (proposal ≈ 200 words).
Every story is a thin vertical slice — end-to-end and demoable, never a horizontal layer.

- [ ] 1. `<kebab-story-name>` — <one-line outcome>
  - **Persona served**: <persona>
  - **Journey segment**: <stage(s) covered>
  - **MoSCoW**: Must
  - **Why this story / why now**: <rationale; story 1 is the walking skeleton>
  - **Depends on**: nothing
  - **Scope**: <in: 2-5 bullets> / <out: 1-2 bullets>
  - **Relevant code**: <paths / modules found during exploration>
  - **Added**: <YYYY-MM-DD>
  - **Change**: _not yet proposed_

- [ ] 2. `<kebab-story-name>` — <one-line outcome>
  - **Persona served**: <persona>
  - **Journey segment**: <stage(s) covered>
  - **MoSCoW**: Must
  - **Why this story / why now**: <rationale>
  - **Depends on**: story 1
  - **Scope**: <in> / <out>
  - **Relevant code**: <paths>
  - **Added**: <YYYY-MM-DD>
  - **Change**: _not yet proposed_

## Open Questions

- <question surfaced during discovery, not yet resolved>

## Change Log

- <YYYY-MM-DD> — <what changed and why; e.g. "Initial plan from <PRD>", "Added story 7 from new billing requirement", "Story 3 superseded by story 8">
