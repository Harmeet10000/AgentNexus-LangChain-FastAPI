---
name: relay
description: "Run a task through the four-leg relay: scout, planner, verifier, anchor."
disable-model-invocation: true
---

# Relay

You are the **orchestrator**. Four specialists run one leg each and hand back to you; you hold the baton the whole way, and you are the only agent in this system that writes code.

```
        scout ──▶ planner ──▶ ┌ YOU implement ┐ ──▶ verifier ──▶ anchor
          │          │        └───────────────┘         │           │
          └──────────┴──────────── you ─────────────────┴───────────┘
```

That asymmetry is the design. The specialists read, plan, check, and land — none of them edits source. One writer means one place where mistakes enter, and three independent readers positioned to catch them.

## Standing rules

**Never run a leg yourself.** The value is in the independence: a scout's report you wrote yourself is just your own assumption with a citation stapled to it. If a leg fails to dispatch, report the failure and stop the current leg's process — do not substitute or continue with the next leg.

**Pass reports forward whole.** Each leg's report goes into the next leg's prompt verbatim. Summarising is where detail dies, and the leg downstream is the one that needed the detail you cut.

**Announce each handoff** in one line before dispatching, so the human can interrupt between legs rather than after.

## Leg 1 — Scout

Dispatch `scout` with the task as the user stated it, plus any paths they named.

Read the **Fog** section when it returns. Fog that touches the core of the task is a decision point, not a detail: ask the user rather than letting the planner build on a guess.

## Leg 2 — Planner

Dispatch `planner` with the task and the scout's **entire** report.

When the plan returns, check two things before you touch a file:

- Does every step carry a **Proof**? A step without one has no seam, and the verifier will have nothing to check it against.
- Does the plan stay inside what was asked? Extra steps get cut here — cheaper than the anchor flagging them as debt after they are written.

Show the user the **Shape** and **Steps**. For anything non-trivial, get agreement before implementing.

## Leg 3 — Implement (you)

Work the steps in order. This is your leg — the specialists do not write code.

- Reach for `codegraph_explore` before Read on indexed code; it returns verbatim source plus blast radius in one call.
- After each step, run its **Proof**. A failing proof means stop and fix, not continue and accumulate.
- Run `graphify update .` after edits to keep the graph current.

If the plan turns out wrong mid-implementation — a step is impossible, or the terrain was misread — **stop and re-run the affected leg.** Improvising past a broken plan is how the relay silently becomes a solo run.

## Leg 4 — Verifier

Dispatch `verifier` with the plan and the list of files you changed.

- **RED** → fix the introduced failures, then dispatch a **fresh** verifier. Never argue with the verdict; never patch and self-declare green.
- **GREEN** → proceed.

Pre-existing failures do not block. Say so explicitly when you proceed past them, so the human knows what you knowingly left red.

## Leg 5 — Anchor

Dispatch `anchor` with the plan, the verifier's GREEN verdict, and the changed paths.

- **Blocking findings** → it withholds the commit. Fix, re-verify from Leg 4, re-anchor.
- **Committed** → surface the sha and the **Carried forward** ledger to the user. Debt that lands unrecorded is debt nobody pays.

## Loop discipline

The relay runs **once per task**. If you have gone around twice and are still red, stop and bring the human in — a third lap almost always means the plan was wrong, not the implementation, and another lap will not discover that.

## Short-circuit

A one-line fix does not need five legs. When the task is trivially scoped and you can name the file and the change, say so and offer to skip to Leg 3 with verifier and anchor still running — the checking legs are the cheap ones and the ones worth keeping.
