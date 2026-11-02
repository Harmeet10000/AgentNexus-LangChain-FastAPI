# ADRs — deep-research-bounded-subgraph

## ADR-1: One tool, not several

**Decision:** the plan→search→read→synthesize machine is exposed as exactly one tool.
**Alternatives:** exposing the stages as separate tools (rejected: grows the deep_research projection, invites the model to improvise orchestration the subgraph already bounds).
**Consequence:** toolset identity is stable no matter how the internals evolve.

## ADR-2: Compiled subgraph over a prompt-side loop

**Decision:** the loop is a compiled graph with hard step bounds, not an unbounded agentic prompt.
**Alternatives:** a "research harder" prompt pattern (rejected: unbounded, unobservable, untestable budget).
**Consequence:** every run's step count is inspectable and every terminal state is typed.

## ADR-3: No interrupts across the tool boundary

**Decision:** approval stays on Saul's orchestration graph; the subgraph never pauses.
**Alternatives:** HITL inside deep research (rejected: interrupts do not cross a tool boundary — reference 09; a paused tool call is unresumable state).
**Consequence:** the tool call returns exactly one terminal envelope, always.

## ADR-4: Typed statuses over partial-success rendering

**Decision:** `completed | budget_exhausted | no_evidence | ingestion_incomplete` are envelope-level statuses; exhaustion carries partial handles and no confident answer.
**Alternatives:** returning the partial synthesis as if complete (rejected: the vacuous-success defect class this repo exists to kill).
**Consequence:** downstream (transcript, findings) can never mistake a truncated run for a finished one.

## ADR-5: Refine harden 6.9 in place

**Decision:** this change delivers 6.9's intent in its refined (subgraph) form and records the supersession, leaving the harden change as the umbrella.
**Alternatives:** a competing parallel change (rejected: two owners for one seam).
**Consequence:** one delivery path; the harden change's checklist stays authoritative.
