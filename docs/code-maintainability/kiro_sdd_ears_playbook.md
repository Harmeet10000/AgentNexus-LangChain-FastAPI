# Kiro SDD and EARS Playbook

## Purpose

This document explains three things:

- what EARS-compliant acceptance criteria are
- what Kiro appears to do for specification-driven development (SDD)
- how to replicate the useful parts of that workflow in this repository without depending on Kiro itself

It is a research-and-operations document, not a product endorsement.

## Short Answer

Kiro's observable SDD model is a gated three-artifact loop:

1. `requirements.md`
2. `design.md`
3. `tasks.md`

Its main explicit requirements-writing standard is EARS: `Easy Approach to Requirements Syntax`.

The rest of the workflow is a set of disciplined practices rather than one formal standard:

- requirements before design in behavior-led cases
- iterative refinement between artifacts
- traceability from requirements to design to tasks
- explicit edge-case and unchanged-behavior capture
- persistent workspace guidance via steering files
- event-driven automation via hooks

This repo can replicate that model with:

- `docs/superpowers/specs/` for approved requirement/design specs
- `docs/superpowers/plans/` for implementation task plans
- `AGENTS.md` and `.opencode/` as the steering layer
- optional opencode commands/plugins/hooks as the automation layer

## What EARS Is

EARS stands for `Easy Approach to Requirements Syntax`.

Primary reference:

- Alistair Mavin: `https://alistairmavin.com/ears/`

EARS is a lightweight constrained natural-language style for writing requirements. Its value is not that it is rigidly formal in the mathematical sense. Its value is that it removes a lot of the ambiguity that sneaks into ordinary prose requirements.

The canonical generic form is:

> While `<optional pre-condition>`, when `<optional trigger>`, the `<system name>` shall `<system response>`

The cited ruleset is:

- zero or many preconditions
- zero or one trigger
- one system name
- one or many system responses

### Core EARS Patterns

#### 1. Ubiquitous

Always true, always active.

```text
The <system name> shall <system response>
```

Example:

```text
The API shall return responses in the APIResponse envelope.
```

#### 2. State-driven

Applies while some condition remains true.

```text
While <precondition>, the <system name> shall <system response>
```

Example:

```text
While the user is unauthenticated, the auth router shall reject protected endpoints.
```

#### 3. Event-driven

Applies when something happens.

```text
When <trigger>, the <system name> shall <system response>
```

Example:

```text
When a client submits a valid refresh token, the auth service shall issue a new access token.
```

#### 4. Optional-feature

Applies only if a capability is present.

```text
Where <feature is included>, the <system name> shall <system response>
```

Example:

```text
Where OAuth login is enabled, the auth feature shall support provider callback validation.
```

#### 5. Unwanted-behavior

Specifies the response to an error or undesired condition.

```text
If <trigger>, then the <system name> shall <system response>
```

Example:

```text
If a JWT is expired, then the auth dependency layer shall reject the request with an unauthorized error.
```

#### 6. Complex

Combines state and event.

```text
While <precondition>, when <trigger>, the <system name> shall <system response>
```

Example:

```text
While a websocket connection is active, when the idle timeout is exceeded, the websocket security layer shall close the connection.
```

## What Makes Acceptance Criteria EARS-Compliant

An acceptance criterion is EARS-compliant when it:

- names the system under discussion
- states the trigger and/or precondition explicitly
- uses `shall` for required behavior
- describes externally verifiable behavior rather than implementation vibes
- stays atomic enough to test

Good:

```text
When a user submits an email address that already exists, the registration endpoint shall return a conflict response.
```

Bad:

```text
The registration flow should handle duplicate emails nicely.
```

The bad version fails because:

- `should` is weak
- the actor and observable system boundary are fuzzy
- `nicely` is non-testable

## Practical EARS Rules for This Repo

For this codebase, good acceptance criteria should usually satisfy all of the following:

- identify the boundary: router, service, repository, dependency, websocket layer, task, or graph node
- describe observable behavior, not hidden implementation details
- be narrow enough to map to one test or a small test cluster
- include failure behavior when failure matters
- preserve existing behavior explicitly during bugfix work

Use this shape:

```text
When <trigger>, the <system/boundary> shall <observable outcome>.
```

Use this for guardrails:

```text
If <invalid or risky condition>, then the <system/boundary> shall <safe response>.
```

Use this for regression protection:

```text
When <bug condition>, the <system> shall <fixed behavior>.
When <adjacent unchanged condition>, the <system> shall continue to <preserved behavior>.
```

## What Kiro Does for SDD

Based on Kiro docs, the core workflow is a specification pipeline rather than freeform prompting.

Primary references used for this summary:

- `https://kiro.dev/docs/specs/feature-specs/requirements-first/`
- `https://kiro.dev/docs/specs/best-practices/`
- `https://kiro.dev/docs/steering/`
- `https://kiro.dev/docs/hooks/`

### 1. Requirements-First workflow

Kiro's requirements-first workflow starts with `what the system should do` before `how to build it`.

Its generated artifacts are:

- `requirements.md`
- `design.md`
- `tasks.md`

Kiro explicitly says `requirements.md` should include:

- user stories with clear acceptance criteria
- system behaviors in EARS format
- functional requirements
- edge cases and error handling

### 2. Design as a downstream artifact

Kiro then derives `design.md` from approved requirements.

The documented design content includes:

- architecture and components
- sequence diagrams
- data models and interfaces
- tech stack recommendations
- error handling
- testing strategy

This is a key SDD discipline: design is constrained by approved behavior, not generated in a vacuum.

### 3. Tasks as a downstream execution plan

Kiro then derives `tasks.md` from the design.

Documented task characteristics:

- discrete and trackable
- expected outcomes
- dependency relationships
- optional vs required tasks

This turns specification into execution slices.

### 4. Iteration between artifacts

Kiro does not treat specs as immutable.

For requirements-first specs, the documented loop is:

1. update `requirements.md`
2. refine `design.md`
3. sync `tasks.md`

This is one of the most valuable parts of the model: downstream artifacts are subordinate to upstream requirements.

### 5. Workflow selection

Kiro supports both:

- Requirements-First
- Design-First

The documented guidance is roughly:

- Requirements-First when behavior is known and architecture is flexible
- Design-First when architecture or non-functional constraints dominate early decisions

That is not an EARS standard. It is a planning heuristic.

### 6. Bugfix-spec discipline

Kiro's bugfix guidance is stronger than ordinary bug tickets. It captures:

- reproduction steps
- current behavior
- expected behavior
- constraints on unchanged behavior

One especially useful pattern from Kiro's docs is explicit preservation language:

```text
WHEN [condition] THEN the system SHALL CONTINUE TO [existing behavior]
```

That is operationally important because most regressions happen when a fix lacks an explicit invariant about what must remain unchanged.

### 7. Requirement analysis before coding

Kiro documents an `Analyze Requirements` capability that checks for:

- ambiguity
- contradictions
- inconsistencies
- gaps

This is not a formal external standard either. It is a quality gate.

### 8. Steering

Kiro steering is persistent project guidance in markdown files. The documented foundational steering files are:

- `product.md`
- `tech.md`
- `structure.md`

These are included in future interactions so the agent does not need the same context repeated every time.

Kiro also recognizes `AGENTS.md` as a steering-style source.

### 9. Hooks

Kiro hooks are event-driven automations tied to development events such as:

- file save/create/delete
- prompt submission
- tool invocation
- spec task execution

The real idea is simple: attach quality checks and workflow enforcement to the moments where drift normally appears.

## Which Standards Kiro Uses

### Explicitly visible standard

The clearest directly cited standard-like practice is EARS for requirements wording.

### Strong conventions, not strict external standards

The rest is better understood as disciplined conventions:

- staged artifact flow: requirements -> design -> tasks
- iterative refinement
- traceability across artifacts
- edge-case capture
- unchanged-behavior capture for bugfixes
- steering files for persistent context
- hooks for enforcement/automation

So the accurate answer is:

- Kiro visibly uses EARS for requirement syntax in requirements-first specs.
- Kiro also uses an opinionated SDD operating model, but that model is mostly workflow convention rather than a single named industry standard.

## How to Replicate Kiro in This Repo

Do not try to clone the product surface. Clone the operating model.

### Repo mapping

Use these existing repo structures as the equivalent layers:

- Steering layer:
  - `AGENTS.md`
  - `.opencode/AGENTS.md`
  - `.opencode/agents/`
  - `.opencode/skills/`
- Requirements and design layer:
  - `docs/superpowers/specs/`
- Execution-plan layer:
  - `docs/superpowers/plans/`
- Implementation layer:
  - `src/app/...`

### Recommended operating model

For every non-trivial feature, use this flow.

#### Step 1. Write a requirements spec first

Create a file under `docs/superpowers/specs/`.

Suggested naming:

```text
YYYY-MM-DD-<feature>-requirements-design.md
```

Inside it, include:

- problem statement
- user stories
- EARS acceptance criteria
- edge cases
- non-goals
- invariants / unchanged behavior
- design approach
- test strategy

This compresses Kiro's `requirements.md` and `design.md` into one durable approved spec if you want fewer files.

If you want a more Kiro-like split, use two files:

- `docs/superpowers/specs/YYYY-MM-DD-<feature>-requirements.md`
- `docs/superpowers/specs/YYYY-MM-DD-<feature>-design.md`

#### Step 2. Derive a plan

Create the execution plan under `docs/superpowers/plans/`.

That plan should map directly back to the accepted requirements/design and contain:

- vertical slices
- task ordering
- dependencies
- verification checkpoints
- risk notes

This is the local equivalent of Kiro's `tasks.md`.

#### Step 3. Execute against the spec, not against memory

When implementation starts, the active agent should treat the approved spec and plan as the current source of truth.

That means:

- if code suggests a better design, update the spec first
- if requirements change, update the spec before continuing implementation
- if the plan drifts, rewrite the plan from the new approved spec

#### Step 4. Verify traceability

Before considering work done, confirm:

- every must-have requirement has an implementation home
- every risky behavior has a test or explicit verification step
- every bugfix records preserved behavior, not just changed behavior

## Proposed Repo Standards

If you want a stable house style, adopt these standards.

### Standard 1. EARS for acceptance criteria

All feature and bugfix specs should write must-have acceptance criteria in EARS form.

### Standard 2. Requirements before code for behavior-led work

For new features and significant changes, do not start in `src/` first. Start in `docs/superpowers/specs/` first.

### Standard 3. Plans must reference requirements

Each plan item should map back to one or more requirement IDs or headings.

### Standard 4. Bugfixes must capture unchanged behavior

Every significant bugfix spec should include at least one `shall continue to` invariant when adjacent behavior could regress.

### Standard 5. Steering stays in markdown, close to the repo

Persistent project behavior belongs in:

- `AGENTS.md`
- `.opencode/AGENTS.md`
- local skills and agent files

That is the closest analogue to Kiro steering already present in this repo.

### Standard 6. Verification is part of the spec

Each spec or plan should state the intended verification path, usually some combination of:

- `uv run ruff check src/`
- `uv run ty check src/`
- targeted tests
- targeted runtime/manual verification

## Templates

### Feature spec template

```md
# <Feature Name>

## Goal

<What user or business outcome this feature exists to produce>

## User Stories

- As a <user>, I want <capability>, so that <outcome>.

## Acceptance Criteria

- When <trigger>, the <system/boundary> shall <observable behavior>.
- If <invalid or risky condition>, then the <system/boundary> shall <safe response>.
- While <state>, the <system/boundary> shall <ongoing behavior>.

## Edge Cases

- <case>
- <case>

## Non-Goals

- <explicitly out of scope>

## Invariants

- When <adjacent condition>, the <system> shall continue to <existing behavior>.

## Design

### Boundaries

- Router:
- Service:
- Repository:
- Shared/runtime layer:

### Data Flow

1. <step>
2. <step>
3. <step>

### Risks

- <risk>

## Verification

- `uv run ruff check src/`
- `uv run ty check src/`
- <targeted tests>
- <manual checks>
```

### Plan template

```md
# <Feature Name> Implementation Plan

## Inputs

- Spec: `<path-to-spec>`

## Slices

1. <slice name>
   - Requirements covered: <ids/headings>
   - Files: <paths>
   - Verification: <checks>

2. <slice name>
   - Requirements covered: <ids/headings>
   - Files: <paths>
   - Verification: <checks>

## Risks

- <risk>

## Done Criteria

- All required slices complete
- Verification passed
- Spec and plan still match shipped behavior
```

## Example: Translating a Loose Request into EARS

Loose request:

> Add session revocation for admin users and make it safe.

Bad acceptance criteria:

- Admins can revoke sessions.
- It should be secure.

Better EARS acceptance criteria:

- When an admin revokes an active session, the auth service shall mark the session as revoked in the server-side session store.
- When a revoked refresh token is presented, the auth refresh endpoint shall reject the request.
- If a non-admin user attempts to revoke another user's session, then the auth service shall reject the action with an authorization error.
- When one session is revoked, the auth service shall continue to preserve unrelated active sessions for the same user unless revoke-all was explicitly requested.

That last criterion is the subtle one. It is where regression prevention lives.

## Minimal Automation to Add Later

If you want to push this repo closer to Kiro's workflow, the next useful additions would be:

1. A lightweight spec template generator under `.opencode/commands/` or equivalent local workflow.
2. A plan generator that turns approved specs into `docs/superpowers/plans/` skeletons.
3. A verification wrapper that reads spec verification sections and runs the declared checks.
4. A review checklist that rejects implementation when no acceptance criteria exist.

## Recommended Policy for This Repo

Use this default decision rule:

- tiny change: go straight to code
- medium change with behavior impact: write a spec first
- large feature, refactor, integration, or bugfix with regression risk: write spec and plan first

That gets most of Kiro's benefit without creating process drag for trivial edits.

## Sources

- Alistair Mavin, `EARS`: `https://alistairmavin.com/ears/`
- Kiro docs, `Requirements-First Workflow`: `https://kiro.dev/docs/specs/feature-specs/requirements-first/`
- Kiro docs, `Best practices`: `https://kiro.dev/docs/specs/best-practices/`
- Kiro docs, `Steering`: `https://kiro.dev/docs/steering/`
- Kiro docs, `Hooks`: `https://kiro.dev/docs/hooks/`

## Bottom Line

If you want to replicate Kiro well, do not obsess over matching UI features.

Replicate these five behaviors:

1. write requirements before implementation when behavior matters
2. make acceptance criteria EARS-shaped and testable
3. derive design and task plans from approved requirements
4. preserve traceability as requirements evolve
5. keep project guidance persistent and close to the repo

That is the part that actually changes engineering outcomes.
