# Review — graph-lifecycle

Sections marked **accepted cost** are known limitations recorded deliberately. Sections marked
**pending** are filled during implementation by the task that names them.

---

## Measured before planning — Agent Saul is dark today

`app.state.saul_graph` is **never assigned anywhere in the tree**. The reader at
`src/app/features/agent_saul/dependencies.py:49` therefore always takes its unavailable branch, and
every request to the Agent Saul surface reports the capability unavailable.

This is worth stating plainly because it changes how task 7.2 should be read. That task is not a
regression check — it is the **first** assertion in the repository that Saul works at all. If it passes,
the feature turned on in this change. If it is skipped, nothing else in this change proves the feature
is reachable, because every other Proof here is about lifetimes and can pass on a process where Saul
is still dark.

The same reader also reveals a documentation defect: `src/app/shared/rag/graphiti/registry.py:11-34`
is a **stale module docstring**, not a recipe. It names `build_tool_registry` where the real symbol is
`build_tool_bundle` (`:92`), and `app.state.saul_checkpointer` where the real reader uses
`app.state.langgraph_checkpointer`. Anyone wiring from that docstring wires against two names that do
not exist. Task 2.2 corrects it, and its Proof greps for both dead names.

---

## Accepted cost — a lifetime change with no type-level protection

Task 4.2 moves a database engine and a Graphiti client from per-task to per-process scope. Nothing in
the type system distinguishes a correctly-scoped resource from an incorrectly-scoped one; both are the
same object with the same interface. The three failure modes:

| Mistake | Symptom | Detected by |
|---|---|---|
| Teardown left on the task | second job in the process uses a closed client | 4.2's two-invocation spy test |
| Init on the task hook, not the process hook | every task still rebuilds; the change reads as done but is a no-op | 4.2's construction count |
| Teardown runs without init | crash during shutdown, the worst place to raise | 4.3's no-preceding-init test |

Accepted because all three are provable with spies and no infrastructure. The residual risk is that the
spies test the hooks in isolation and not a real forked worker — a genuine multi-process run remains
the only way to observe fork-time inheritance problems, and this change does not attempt one.

---

## Accepted cost — the fail-closed tests constrain the shape of the fix

Seven tests under `tests/unit/features/ingestion/` assert that synchronous surfaces fail closed when
the shared pipeline is not provisioned. They must **all still pass unchanged** after this change, which
is a real constraint: it rules out any implementation where provisioning success is assumed, and it
rules out making the ingestion graph eagerly required.

This is recorded as a cost rather than a benefit because it means the degrade path stays live code
forever, including on healthy processes where it never executes. That is the correct trade — the
alternative is a startup that aborts on a graph build failure — but it is a permanent piece of
machinery, and task 5.1's Proof requires the count of passing fail-closed tests to be **unchanged**,
not merely non-decreasing.

---

## Note — why this change cites three capabilities and amends none

`langgraph-checkpointing`, `celery-worker-deployment`, and `document-ingestion-pipeline` already
require, respectively: that the constructing process owns the checkpointer pool and that teardown
distinguishes nothing-to-close from a close; that task registration is explicit and worker readiness
independently verifiable; and that ingestion executes outside the request path with synchronous
surfaces failing closed when the pipeline is unprovisioned.

Every one of those is a requirement this change **satisfies**, not one it tightens. Restating them as
`## MODIFIED` blocks would replace them wholesale on archive — the delta form has no partial-edit
semantics — and would risk silently dropping a scenario that was not copied forward. Citing them costs
nothing and carries that risk not at all.

---

## Pending — the checkpointer teardown observation (task 1.2)

- Command run: _pending_
- Startup log line confirming provisioning: _pending_
- Shutdown log line confirming pool close: _pending_
- Confirmation that no credential substring appears in either: _pending_

---

## Pending — the worker lifetime evidence (tasks 4.2 and 4.3)

- Construction count across two simulated invocations: _pending_ (expected: 1)
- `close_graphiti` call count across one init/shutdown cycle: _pending_ (expected: 1)
- Engine disposal count across one init/shutdown cycle: _pending_ (expected: 1)
- Shutdown-without-init outcome: _pending_ (expected: completes, reports nothing to close)

---

## Pending — Saul is on (task 7.2)

- Request-level result against a started application: _pending_
- Confirmation the dependency bundle resolved without reporting the capability unavailable: _pending_

This is the single line in this change that distinguishes "the lifetimes are correct" from "the feature
works".

---

## Pending — the prose supersession (task 5.2)

- Rewritten text at `src/app/shared/langgraph_layer/checkpointer.py:11-17`: _pending_
- Rewritten text at `tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py:3-8`:
  _pending_
- Confirmation that neither rewrite introduces the two literals `test_checkpointer_lifecycle.py`
  requires to be absent: _pending_
