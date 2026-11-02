> Change class: **M** (normal checklist). Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness

- [x] Each spec requirement has at least one scenario (5 requirements, 11 scenarios)
- [x] Edge and error cases are named, not just the happy path (no runtime, wrong grant, unset jail, symlink escape, missing cwd)
- [x] Delta operations are correct against the current main spec — all five requirements are ADDED to the existing `agent-tool-registry` capability; none of its existing requirements (single registry, explicit population, resolvability) are touched

## Correctness

- [x] Requirements are observably testable — each scenario names an observable (result envelope, execution/non-execution, message content), not an internal call
- [x] No implementation details in specs (executor/seam names live in design.md only)
- [x] "Plain schema model" requirement is phrased as observable schema surface, not as a class name

## Verification matrix

| Requirement | Proof |
|---|---|
| Deny without grant | `tests/unit/ai/test_system_tool_capabilities.py` — deny-without-runtime, wrong-grant, per-tool grant tests |
| Jail confinement | unset-jail refusal, inside-jail serve, symlink-escape denial, shell missing-cwd tests |
| Authority metadata | registration assertions (to add at apply: registry tag lookup test — see gaps) |
| Refusals name their fix | message-content assertions in the deny tests (`TOOL_WORKSPACE_ROOTS`, capability id) |
| Plain schema models | web-search schema assertion (to add at apply — see gaps) |

## Known gaps to close at apply time

1. The authority-metadata and plain-schema requirements are implemented but their *named* tests are not yet in the suite — tasks 1.6b/1.5b cover them so archive doesn't sync an untested requirement.
2. The full-suite run at implementation time (2026-10-27 tree) carried 39 failures, all attributed to live-service integration tests, suite-order pollution from the concurrent refactor, or files this change never touched (`review` of that attribution is in the session record). Re-run `pytest tests/unit/ai/` at apply time as the gate for this change specifically.

## Decisions to defend in review

1. **Executor-side enforcement over binding-time** (design Decision 1) — challenge with "what re-enables execution?" The answer should always be "a grant in server-owned context", never "a different call path".
2. **Tags now, structured fields in D4** (Decision 5) — confirm the interim tags don't become load-bearing selection criteria before the catalog lands.
3. **Shell's explicit-cwd refusal** (Decision 3) — convenience argument against it should lose to the unconfined-process argument; if it doesn't, the jail is decoration.

## Cross-change interactions

- `tool-capability-catalog` (D4) owns: structured authority fields, role → grant policy, web-search egress grant. This change's tags are its seed data.
- `deep-research-bounded-subgraph` (D5) consumes the same `ToolRuntimeContext` seam; its deny-without-context acceptance reuses this change's helper semantics.
- No interaction with `mount-retrieval-graph` (qna engine is a node, binds no tools from this band).
