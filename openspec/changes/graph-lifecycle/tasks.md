# Tasks — graph-lifecycle

## How to read the Proofs

1. **Never use a test-process exit code as a Proof of test outcome.** The coverage floor makes a green
   suite exit non-zero. Compare **summary pass and failure counts** against
   `docs/relay/baseline-pytest.txt`, re-measured immediately before the task.
2. **Never prove a schema fact by rendering migrations offline.**
3. **Never make a Proof depend on a durable outbound event firing.** The outbox tables do not exist.
4. **The live database may be used** — zero data, zero users, ruled available. This supersedes the
   archived rule requiring a local scratch instance.
5. **The upload-to-chunks acceptance check becomes writable in this change.** The archived rule
   forbade it because the shared graph wiring was to stay commented by decision. This change reverses
   that decision, so the check is now legitimate — and it belongs here, in task 5.3.

**Blocked by `rag-tree-repair`.** Task groups are ordered by a hard constraint, not by file: the
checkpointer must exist before Agent Saul can be constructed, because `build_saul_graph`'s
`checkpointer` parameter admits no absent value; the `configurable` seam must accept a repository
before the Celery wiring can pass one; and the import-time globals are last because nothing waits on
them.

## 1 Checkpointer provisioning

- [ ] 1.1 `src/app/lifecycle/graphs.py` — provider functions only, with no I/O performed at build time.
  The database engine is a handle assigned at `lifespan.py:464`, before the commented block at `:522`.
  **Proof:** `uv run python -c "import app.lifecycle.graphs"; echo $?` → `0`; `uv run ty check src/`
  diagnostic count ≤ `docs/relay/baseline-ty.txt`.
- [ ] 1.2 Checkpointer startup policy calling
  `setup_langgraph_checkpointer(get_database_url(flavour="plain"))`; delete the commented block at
  `lifespan.py:549-565`. Teardown already exists behind the guard at `:365`.
  **Proof:** `uv run pytest tests/unit/shared/langgraph_layer/test_checkpointer_lifecycle.py -q`
  summary shows zero failures.
- [ ] 1.3 **Landmine check.** That test file includes proofs that grep `checkpointer.py` for *absent*
  literals — the saver's connection-string classmethod name, and the old pool `hasattr` guard. Any
  edit to that module, including a docstring edit, must not reintroduce those strings.
  **Proof:** `rg -n 'from_conn_string|hasattr\(.*"pool"' src/app/shared/langgraph_layer/checkpointer.py; test $? -eq 1`
  → exit `0`.
- [ ] 1.4 Assert no credential reaches a log line on a setup failure.
  **Proof:** a unit test forces a setup failure with a credentialed connection string and asserts the
  password substring appears in no captured log record.

## 2 Agent Saul provisioning

- [ ] 2.1 Construct the memory service inside the same provisioning path. Its constructor
  (`agent_memory_service.py:136`) takes only a partition prefix plus optional callables that default to
  cognee's own functions — no client, no engine, no session. Do not add a third construction site; the
  only existing one is `src/tasks/agent_memory_tasks.py:119`.
  **Proof:** a unit test constructs the provider with settings alone and asserts the returned service's
  partition prefix; `rg -c 'AgentMemoryService\(' src/` returns exactly `2`.
- [ ] 2.2 Saul graph policy assigning the graph to process state.
  **Correction to earlier planning:** `src/app/shared/rag/graphiti/registry.py:11-34` is **not a recipe
  to follow**. It is a module docstring and it is stale — it names `build_tool_registry` when the real
  symbol is `build_tool_bundle` (`:92`), and `app.state.saul_checkpointer` when the actual reader,
  `agent_saul/dependencies.py:49`, uses `app.state.langgraph_checkpointer`. Wire against the reader,
  and correct the docstring.
  **Proof:** a unit test asserts the Saul dependency returns the graph when state is populated and
  still raises the typed service-unavailable when it is not; and
  `rg -n 'build_tool_registry|saul_checkpointer' src/; test $? -eq 1` → exit `0`.
- [ ] 2.3 Confirm memory semantics are not respecified here. Cognee owns them, and
  `openspec/specs/saul-memory-prefetch-and-retrieval/` already governs them.
  **Proof:** `rg -in 'recall|remember|memory partition' openspec/changes/graph-lifecycle/specs/; test $? -eq 1`
  → exit `0`, except for the one requirement that explicitly declines to specify them.

## 3 The invocation-configuration seam

- [ ] 3.1 Resolve the repository from the invocation configuration inside the node, instead of from a
  closure parameter. Add a typed accessor that raises a project exception when the key is absent.
  **Proof:** `uv run pytest tests/unit/documents tests/unit/shared/langgraph_layer/test_ingestion_checkpoint_plumbing.py -q`
  summary shows zero failures, plus a new test asserting the typed exception — not `KeyError` — on a
  missing key.
- [ ] 3.2 Assert no compiled graph captures a session or repository.
  **Proof:** a unit test compiles the graph, then asserts the compiled object's closure cells contain
  no session or repository instance; and graph state carries neither field.

## 4 Celery worker process wiring

- [ ] 4.1 Point `run_document_ingestion_task` (`src/app/features/documents/service.py:904-947`) at a
  process-cached compiled graph, passing the repository through the invocation configuration.
  **Proof:** a unit test patches the builder with a counting spy and asserts one compilation across two
  invocations.
- [ ] 4.2 **Highest-risk task in this cluster.** Hoist the per-task construction at
  `service.py:912-935` into `worker_process_init`, and move its `close_graphiti()` and engine disposal
  into `worker_process_shutdown`. This changes the lifetime of a database engine and a Graphiti client
  from per-task to per-process. A wrong shutdown hook leaks connections for the worker's whole life; a
  wrong init hook lets every task in the process share a half-built client.
  **Proof:** a unit test drives `worker_process_init` then `worker_process_shutdown` against spies and
  asserts exactly one construction, one `close_graphiti`, and one engine disposal; a second test
  asserts two simulated task invocations between the two hooks construct nothing.
- [ ] 4.3 Assert release is not attempted for a resource that was never constructed.
  **Proof:** a unit test drives `worker_process_shutdown` with no preceding init and asserts it
  completes without raising and reports nothing-to-close.
- [ ] 4.4 Register both hooks, following the existing prior art at `src/tasks/crawler_tasks.py:72`.
  **Proof:** `rg -n "worker_process_init|worker_process_shutdown" src/tasks/` shows both ingestion
  hooks; a unit test asserts the init hook populates the process cache.

## 5 The ingestion graph as a startup policy

- [ ] 5.1 Register the ingestion graph as a startup policy; delete the commented block at
  `lifespan.py:522-537` while **preserving** its embedding-function prohibition as a comment on the
  provider.
  **Proof:** `rg -n "embedding_fn" src/app/` still returns the note; and
  `uv run pytest tests/unit/features/ingestion -q` summary shows zero failures across all seven
  fail-closed tests, unchanged.
- [ ] 5.2 Rewrite the two prose sites that forbid this wiring —
  `src/app/shared/langgraph_layer/checkpointer.py:11-17` and
  `tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py:3-8` — to cite this change as
  superseding, without reintroducing the forbidden literals from 1.3. The reconciliation is per
  *process*: the ingestion graph's consumer is the worker, which never runs the lifespan, which is why
  it gets a worker hook; the Saul graph and checkpointer serve the API process, which does.
  **Proof:** `rg -n "stays commented" src/ tests/; test $? -eq 1` → exit `0`; and the full-suite
  failure count ≤ `docs/relay/baseline-pytest.txt`.
- [ ] 5.3 **Write the upload-to-chunks acceptance check** that the archived rule forbade. It was
  forbidden because the shared wiring stayed commented; that is no longer true.
  **Proof:** an integration test marked `requires_db` uploads a fixture document and asserts chunk rows
  exist for it; `uv run pytest -m requires_db -q` summary shows zero failures.

## 6 Remove import-time compilation

- [ ] 6.1 `src/app/shared/langgraph_layer/open_deep_search/graph.py:278,478,555` — convert the three
  module-global compilations into lazily built, process-cached providers.
  **Proof:** a unit test imports the module with a spy on the compile call and asserts zero invocations
  at import time.
- [ ] 6.2 `app.state.pageindex_client` (`lifespan.py:538`) stays unwired. `knowledge-stack` owns its
  disposition and removes the dependency.
  **Proof:** `rg -n "pageindex_client" src/app/lifecycle/lifespan.py` returns the single commented
  line, unchanged.

## 7 Close out

- [ ] 7.1 **Proof:** `openspec validate graph-lifecycle --strict` exits `0`;
  `uv run ruff check --no-cache src/` line count ≤ `docs/relay/baseline-ruff-after.txt`;
  `uv run ty check src/` count ≤ `docs/relay/baseline-ty.txt`; `uv run pytest -q 2>&1 | tail -1`
  failure count ≤ `docs/relay/baseline-pytest.txt`.
- [ ] 7.2 Confirm Agent Saul is actually on.
  **Proof:** a request-level test against a started application asserts the Saul dependency bundle
  resolves without reporting the capability unavailable — the behaviour that is broken today.
