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

- Command run: `DEBUG=false uv run pytest tests/unit/shared/langgraph_layer/test_checkpointer_lifecycle.py -q` (zero failures, included in the 110-test focused run).
- Startup log line confirmed by the recording logger assertion: `LangGraph async checkpointer initialised`.
- Shutdown log line confirmed by the recording logger assertion: `LangGraph checkpointer connection pool closed`.
- The credentialed-failure tests assert both raw and percent-encoded password forms are absent from every captured log record.

---

## Pending — the worker lifetime evidence (tasks 4.2 and 4.3)

- Construction count across two simulated service invocations: **1** (`test_worker_compiles_once_and_two_service_invocations_reuse_the_graph`).
- Resource-release call count across one init/shutdown cycle: **1**; `_release_document_worker` is the single owner of both `close_graphiti` and engine disposal.
- Engine disposal count across one init/shutdown cycle: **1**, through that same single release call.
- Shutdown-without-init outcome: completes and makes **0** release calls (`test_shutdown_without_successful_initialization_releases_nothing`).

---

## Pending — Saul is on (task 7.2)

- Request-level result against a lifespan-started FastAPI probe: **HTTP 200** (`test_started_application_exposes_agent_saul_dependency_bundle`).
- The resolved bundle contains the exact graph, checkpointer, and Redis objects placed in application state; no unavailable branch executes.

This is the single line in this change that distinguishes "the lifetimes are correct" from "the feature
works".

---

## Pending — the prose supersession (task 5.2)

- `src/app/shared/langgraph_layer/checkpointer.py` now records that `graph-lifecycle` supersedes D17 and assigns ownership per process.
- `tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py` now explains the worker/API process split while preserving the fail-closed contract.
- `rg -n 'from_conn_string|hasattr\(.*"pool"' src/app/shared/langgraph_layer/checkpointer.py` returns no matches.
- The original `build_tool_registry|saul_checkpointer` proof was overbroad because `get_saul_checkpointer` is a valid dependency. The narrowed stale-form proof, `build_tool_registry|app.state.saul_checkpointer`, returns no matches.

---

## Implementation evidence by task

| Tasks | Evidence |
|---|---|
| 1.1 | `DEBUG=false uv run python -c "import app.lifecycle.graphs"` exits 0 with no output after making `app.lifecycle` lazy. |
| 1.2–1.4 | Checkpointer lifecycle suite passes, including pool ownership, outcome reporting, and credential-redaction failures. |
| 2.1 | `test_memory_provider_uses_the_configured_partition_prefix`; constructor grep reports exactly two sites. |
| 2.2, 7.2 | Started-app dependency test returns 200 and validates all three state collaborators. |
| 2.3 | The delta explicitly declines to govern recall, writes, or partition behavior. |
| 3.1 | Document and checkpoint-plumbing suites pass; the accessor raises a typed capability-naming exception. |
| 3.2 | `test_compiled_graph_captures_no_repository_or_session` checks closure values and all state keys. |
| 4.1–4.4 | Worker lifecycle suite proves one compile, two service invocations, one release, failed-init cleanup, and no-init shutdown. |
| 5.1–5.2 | All ingestion fail-closed tests pass; startup uses `STARTUP_POLICIES`; stale prohibitions and forbidden checkpointer literals are absent. |
| 5.3 | The live database was upgraded through `0020`; `tests/integration/test_document_ingestion_lifecycle.py` passes and proves a fixture reaches persisted chunk rows through the compiled graph. The test cleans its unique fixture row in `finally`. |
| 6.1 | `test_open_deep_search_import_compiles_no_graph` observes zero compile calls and three empty process caches after import. |
| 6.2 | Lifespan contains only the preserved commented `pageindex_client` line. |
| 7.1 | Strict OpenSpec, lifecycle-scoped ruff, lifecycle-scoped ty, and the 110-test focused suite pass. Repository-wide ruff currently has four concurrent-change findings, so the repository-wide gate remains pending. |
