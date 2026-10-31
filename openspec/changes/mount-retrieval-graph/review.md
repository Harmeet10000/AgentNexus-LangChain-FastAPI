> Change class: **L** (full checklist + verification matrix).
> Role: reviewer, not author. Reviewed against proposal.md, specs/, design.md, and the current main specs.

## Completeness

- [x] Each spec requirement has at least one scenario — 3 requirements in `legal-corpus-retrieval`, 3 in `agentic-retrieval-loop`, 6 in `qna-retrieval-engine`; every one carries scenarios.
- [x] Edge and error cases are named, not just the happy path — empty scope, unprovisioned graph, exhausted budget, unresolvable edge spans, graphiti unavailability, cap-exhausted fallback, and provisioning failure all have scenarios.
- [x] Delta operations are correct against the current main spec — all three deltas are `## ADDED Requirements`; no requirement of the existing `legal-corpus-retrieval` or `agentic-retrieval-loop` specs is modified or removed, so no full-block copies are required. The existing "Ranked retrieval and fusion have a single implementation" requirement already covers the one-implementation rule; the deltas add the ask-serving, cache-key, coverage, edge-grounding, parity, and observability behaviour on top of it rather than restating it.
- [x] The topology-assertion requirement in `agentic-retrieval-loop` ("The graph's node and edge shape is asserted") is honoured: adding the `edge_grounding` node updates the asserted shape in the same change (task 1.1, design decision 2).

## Correctness

- [x] Requirements are observably testable: SHALL/MUST throughout, no `should`/`may` in normative text (one earlier `MAY` in a scenario THEN was rewritten).
- [x] Scenarios are concrete WHEN/THEN, not requirement restatements — e.g. "a re-ingested document does not serve stale answers", "drift fails the suite", "model-supplied identity is forbidden".
- [x] No implementation names baked into requirements — requirement text speaks of "the compiled retrieval graph", "the budget ledger", "the evidence registry" (existing system concepts), while file/function specifics (`_build_answer_cache_key`, `edge_grounding` node) live in design.md and tasks.md where they belong.
- [x] `skip_specs` is not used; three real capability deltas exist.
- [x] The new capability name `qna-retrieval-engine` is kebab-case and does not collide with any existing spec under `openspec/specs/`; the modified capabilities use their exact existing paths.

## Standards

- [x] No code is implemented in this change; artifacts only.
- [x] Design decisions respect the project conventions: async-first graph invocation, collaborators via invocation context rather than captured closures (`compiled-graph-lifecycle`), typed failures over bare exceptions (fail-closed scenarios), no new secret surface.
- [x] Result-pattern boundary preserved in the design: the access-audit recording and typed unavailability stay at the service boundary; the graph never records audit or renders HTTP.
- [x] Layering preserved: the graph surfaces stage usage instead of importing `agent_saul`'s budget domain (design decision 5); the ledger stays in Saul.

## Risk

- [x] The single largest risk — a behaviour change on a live endpoint — is gated by the parity suite (spec: "The compiled loop is parity-gated against the inline contract"), and the ordering in tasks.md makes the gate precede both the qna wiring and the router retarget.
- [x] Tenant-bypass risk on the new Saul exposure is closed by the V6 construction requirement with a dedicated propagation-test scenario.
- [x] Compile-once refactor risk is bounded by the lifecycle scenarios and the mechanical nature of the conversion; rollback is a single wiring revert until task 5.1, after which it is a revert commit — recorded in the migration plan.
- [x] Residue is explicit: out-of-scope items (G4/G5/G7, re-ingestion, Phase E ledger) are named in the proposal so the retired plan's register stays reconciled.

## Verification matrix

| Requirement (delta) | Verified by |
|---|---|
| Ask endpoints served by the compiled graph | Router retarget + parity suite + unprovisioned fail-closed fixture (tasks 4.1, 4.2, 2.1) |
| Version-aware answer cache keys | Parity suite cache-key derivation + stale-version tests (tasks 2.1, 1.2) |
| Truthful coverage and warnings | Parity suite warning/coverage assertions (tasks 2.1, 1.4) |
| Edge-aware grounding supplements the loop | Edge-free byte-parity check + span-resolution tests (tasks 1.1, 2.1) |
| Parity gate against the inline contract | The suite itself, plus the deliberate-drill failure check (tasks 2.1, 2.2) |
| Per-stage observability | Span/decision-log assertions on a two-iteration run (tasks 1.7) |
| qna node invokes the graph, never a tool | Toolset byte-identity + wiring review (tasks 3.2, 3.3) |
| Server-owned context construction | V6 construction/propagation test (task 3.2) |
| Explicit state mapping | Mapping/no-shared-keys checkpoint test (task 3.4) |
| Budget ledger draw | Pre-check/per-stage charge/BUDGET_EXHAUSTED tests (task 3.5) |
| Process-scoped engine graph | Compile-once + per-invocation collaborator tests (task 3.1) |
| Transcript delivery via existing events | WS frame + evidence-handle resolution test (task 3.6) |

## Concerns recorded (non-blocking)

1. The parity suite pins the inline contract as-is, including its quirks; any deliberate future divergence requires changing the recorded fakes consciously. This is stated in design risks but is worth remembering at review time when the suite first fails on a real inline defect.
2. `ask_via_retrieval_graph`'s current docstring prose ("Deliberately not exposed by any router") becomes false at cutover; the deletion task (5.1) removes the body, but the docstring of the surviving service method should be rewritten in the same task rather than left describing the quarantined state.

VERDICT: APPROVED
