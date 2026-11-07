---
name: discovery
description: Break a large PRD or product idea into a release plan for an OpenSpec project — personas, journey map, MoSCoW priorities, and thin vertical stories captured as a checklist in openspec/discovery.md that feeds /opsx:propose one story at a time. Use when a feature or PRD is too big for one OpenSpec change, when the user asks to split or prioritize work, or to resume or revise an existing discovery.md. Requires the openspec CLI.
---

Enter discovery mode. Break a large PRD or product idea into a prioritized release plan using discovery techniques: personas, a journey map, MoSCoW prioritization, and thin vertical stories.

**Discovery is for planning, not implementing.** You may read files, search code, and investigate the codebase, but you must NEVER write application code and NEVER create change proposals. Discovery produces one file: `openspec/discovery.md` (plus, only at hand-off and with the user's confirmation, one backlog line in `openspec/config.yaml`). Changes are created later, one at a time, by `/opsx:propose`.

**This is a stance with a light structure.** The phases below give the conversation a shape, but they are conversational, not a script. Move fluidly, revisit earlier phases when new information emerges, and confirm with the user before advancing to the next phase.

This skill is designed for and tested with OpenSpec projects. It requires the openspec CLI and an initialized OpenSpec root.

---

## The Stance

- **Curious, not prescriptive** - Ask questions that emerge naturally from the PRD
- **Visual** - Use ASCII diagrams and tables liberally (journey flows, priority matrices)
- **Grounded** - Explore the actual codebase; annotate the plan with what exists today
- **Adaptive** - Follow interesting threads, pivot when new information emerges
- **Outcome-oriented** - Unlike explore mode, discovery converges on a concrete file: the release plan
- **One phase at a time** - Don't rush all phases in one message; confirm before advancing
- **Checkpointed** - Every confirmed phase is written to disk immediately; the session can stop and resume at any phase boundary

---

## Locate the Discovery File

At the start:

1. Run `openspec list --json` and use `root.path` from the output to resolve the OpenSpec root. Do not assume `./openspec/` — always use the resolved root.
2. Read `<root.path>/openspec/config.yaml` (or `config.yml`) if it exists. Its `context` and `rules` are constraints for you to follow, not content to reproduce.
3. Check for `<root.path>/openspec/discovery.md`:
   - **Doesn't exist** → run the phases to create it.
   - **Exists with `Status: in-progress`** → enter Resume mode (below).
   - **Exists with `Status: complete`** → enter Revision mode (below).

If the openspec CLI is unavailable or no OpenSpec root is found, tell the user this skill requires an initialized OpenSpec project and stop. Do not improvise another location.

---

## The Phases

**Checkpoint every phase.** At the end of each phase, once the user confirms it, write that phase's section into `discovery.md` and set the status line to the next phase (`ingest → personas → journey-map → moscow → stories → complete`). The file on disk is the resume point — a future session with none of this conversation's context must be able to pick up exactly where this one stopped. Use the `date` command for the Created / Last revised / Change Log timestamps — never guess dates.

### Phase 1: Ingest

Take in the PRD or idea — a file path, pasted text, or just conversation. Then:

- Restate the scope in your own words; have the user confirm
- Surface goals and non-goals
- List open questions as they emerge (they land in the file)

**Checkpoint:** create `discovery.md` from [templates/discovery-template.md](templates/discovery-template.md) with the header metadata (Status, Created, Last revised), a dated Sources entry, and any open questions so far.

### Phase 2: Personas

Identify 1-4 personas conversationally. For each:

- **Who** they are
- **Goal** they're trying to reach
- **Pain today** without this product/feature
- **Success looks like** — the observable signal they got value

Push back if personas are roles without needs ("admin") rather than people with goals.

### Phase 3: Journey Map

For each primary persona, map their journey as ordered stages. Draw it:

```
  Discover ──► Sign up ──► First run ──► Daily use ──► Share
     │            │            │             │           │
    gap          gap        partial      supported      gap
```

Annotate every stage against the **real codebase** — read the code, don't guess:

- `supported` — works today
- `partial` — exists but incomplete for this journey
- `gap` — doesn't exist

### Phase 4: MoSCoW

Bucket the capabilities the journey surfaced into **Must / Should / Could / Won't (this release)**.

- Every Must needs a reason tied to a persona's journey
- Push back on Must inflation: if everything is a Must, nothing is
- Won't is a decision, not a dumping ground — record why

### Phase 5: Stories

Cut the Must (then Should) work into stories:

- Every story must be a **thin vertical slice** — end-to-end and demoable, never a horizontal layer. A persona can do something new when it ships.
- Each story is sized for **one OpenSpec change**: right-sized means its future proposal's what-and-why fits in **roughly 200 words**. If drafting the story packet already strains that budget, the story is too big — split it.
- The first story is the **walking skeleton**: the thinnest path through the whole journey
- Order by dependency first, then value

Each story gets a **story packet** — enough for a future `/opsx:propose` run that has NONE of this conversation's context:

- Persona served and journey segment covered
- MoSCoW bucket and why this story, why now
- Dependencies on earlier stories
- Scope: what's in (2-5 bullets), what's explicitly out (1-2 bullets)
- Relevant code: paths and modules you found while exploring

**Size check-out:** after drafting the stories, review each against the ~200-word proposal budget. List every story that looks too large, each with 1-2 candidate split lines (or a one-line reason it genuinely resists splitting), and resolve with the user before finalizing.

### Phase 6: Finalize

Flip the status line to `complete`, stamp Last revised, and show the user the story checklist.

Then offer to make propose discovery-aware: with the user's confirmation, append a backlog line to the `context:` block of `<root.path>/openspec/config.yaml` (create the block if absent; preserve everything already there):

> Backlog: openspec/discovery.md is the release plan. When asked to propose the next change without a specific request, pick the next unchecked story in discovery.md and use its story packet as the input. One story per change.

OpenSpec injects `context` into every artifact's instructions, so every future propose run will know to read discovery.md without being told. If the user declines, hand off manually:

> The release plan is in `openspec/discovery.md`. To start building, run `/opsx:propose` and ask it to use the next unchecked story in discovery.md. Create one story at a time. Re-run this skill anytime to revise the plan.

---

## Resume Mode

When `discovery.md` exists with `Status: in-progress`:

1. **Read it.** The status line records the next phase; the sections already present are confirmed work.
2. **Restate where discovery stopped** — a one-paragraph summary of what the file already captures — and confirm with the user.
3. **Continue from the recorded phase.** Don't re-litigate confirmed sections unless the user brings new information; then revisit and update them.

---

## Revision Mode

When `discovery.md` exists with `Status: complete`:

1. **Read it**, then reconcile the story checklist against reality:
   - `openspec list --json` → active changes
   - `ls <changesDir>/archive/` → completed changes
2. **Update checkboxes**:
   - Story's change is archived → flip `- [ ]` to `- [x]` and set `**Change**: <name> (archived)`
   - Story's change is active → keep `- [ ]`, set `**Change**: <name> (in progress)`
3. **Take in what the user brings** — a new PRD, new requirements, changed priorities, learnings from shipped stories. For each:
   - Add a dated entry to **Sources** (when there's a new input document) and to the **Change Log** (always: what changed and why)
   - Run new capabilities through MoSCoW — priorities of existing items are revisable too
   - Reflect the outcome in stories: new stories get an **Added** date; every new or changed story still passes the thin-vertical-slice shape rule and the ~200-word size check
4. **Never silently delete a story.** Mark superseded stories as such, with a one-line reason, so the plan's history stays legible.
5. Stamp **Last revised** with today's date.

---

## Handing Off to Propose

- Discovery **never** creates changes. The user creates them one at a time with `/opsx:propose`.
- The recommended wiring is the config.yaml backlog line (Phase 6) — with it, propose picks up the next unchecked story from discovery.md on its own. Without it, the user must tell propose explicitly: `/opsx:propose` then "use the next unchecked story in openspec/discovery.md".
- The story packet is the input — it must stand alone.
- One story per propose run. If the user asks you to batch-create proposals for all stories, decline and explain: proposals written ahead of implementation go stale and overlap on shared specs.

---

## Guardrails

- **Don't implement** - Never write application code
- **Don't create changes** - That's propose's job, one story at a time
- **Write only the plan** - `discovery.md` at the resolved root; the only other write allowed is the single backlog line in `config.yaml`, at hand-off, with the user's confirmation
- **Don't rush** - One phase per message beat; confirm before advancing
- **Do checkpoint every phase** - The file on disk is the resume point; never hold a confirmed phase only in conversation
- **Do keep story packets self-contained** - The propose run that consumes them starts cold
- **Do flag oversized stories** - Never finalize a story whose proposal can't be stated in ~200 words without calling it out and offering a split
- **Do explore the codebase** - "Relevant code" pointers come from reading, not guessing
- **Do visualize** - Journey maps and priority tables beat prose
- **Don't silently delete stories in revision** - Mark them superseded instead
