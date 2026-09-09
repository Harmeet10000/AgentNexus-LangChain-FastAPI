# ADRs — graph-lifecycle

## ADR-001 — The process is the compilation scope

**Status:** accepted.

**Context.** `build_document_ingestion_graph` is called and compiled inside every Celery job. Three
graphs in `open_deep_search/graph.py` are compiled at module import. Neither is per-request, so neither
is correct-but-wasteful; both are simply the wrong scope.

**Decision.** A graph is compiled at most once per process and never as a side effect of an import.
Consumers read the compiled graph from process state — `app.state` in the API server, a module global
populated by a worker hook in Celery.

**Consequences.** Ingestion jobs drop one graph compilation each. Importing a graph module becomes
free, which makes the modules importable from tests and from tooling without paying for construction.
The cost is that process state is now load-bearing: a consumer that reads an unprovisioned graph must
get a typed failure rather than `None`, which is what ADR-003 specifies.

## ADR-002 — Job-scoped collaborators travel in the invocation configuration, not in graph state

**Status:** accepted (user ruling: "inject per-invoke via `configurable`").

**Context.** Once a graph is compiled once per process, it can no longer close over a per-job
repository or session. The two candidates for delivering them are the graph's state schema and
`config["configurable"]`.

**Decision.** Job-scoped collaborators are supplied through `config["configurable"]` at invocation.
Graph state carries neither a session nor a repository, and compilation captures neither. A typed
accessor reads the key and raises a project exception naming the missing collaborator.

**Consequences.** State stays serialisable, which keeps checkpoint resumption meaningful — a resumed
run is not carrying a repository bound to a session that closed days ago. The trade is that
`configurable` is an untyped dictionary; the typed accessor is the price of using it, and without that
accessor a wiring mistake surfaces as a `KeyError` from inside a node rather than as a named failure.

## ADR-003 — Graph provisioning is a startup policy that degrades

**Status:** accepted.

**Context.** `lifespan.py:171-342` already implements provisioning with failure tolerance. The
commented-out graph blocks used hand-rolled `try/except` instead. The ingestion fail-closed tests
already assert the degrade behaviour.

**Decision.** Register graph provisioning through the existing startup-policy mechanism. A graph that
fails to build leaves startup running with its state attribute absent; a dependency reading an
unprovisioned graph raises a typed service-unavailable naming the capability.

**Consequences.** One degrade path instead of two, so the behaviour the fail-closed tests assert is the
behaviour that actually runs. A failure to provision is visible at startup rather than at first use.
The cost is that a genuinely required graph no longer stops the process from starting — deliberate, and
the reason the typed service-unavailable must name the capability rather than producing a generic
server error.

## ADR-004 — The Celery worker is provisioned by process hooks, not by the lifespan

**Status:** accepted.

**Context.** The worker never runs the FastAPI lifespan and has no `app.state`. The per-task
construction at `service.py:912-935` exists because of that gap.

**Decision.** The worker provisions through `worker_process_init` and releases through
`worker_process_shutdown`, storing per-process state in module globals. The provider functions
themselves are shared with the lifespan; only the registration site differs. Prior art exists at
`src/tasks/crawler_tasks.py:72`.

**Consequences.** The two process types converge on one set of providers with two registration sites,
so a change to how a graph is built lands in both. A database engine and a Graphiti client change
lifetime from per-task to per-process — the highest-risk element of this change, mitigated by
spy-counted Proofs at tasks 4.2 and 4.3 rather than by inspection.

## ADR-005 — The checkpointer parameter stays non-optional

**Status:** accepted.

**Context.** `build_saul_graph` requires a checkpointer with no absent value permitted. Making it
optional would let Saul be provisioned even when checkpointer setup fails.

**Decision.** Keep it non-optional and order the provisioning around it: checkpointer first, Saul
second.

**Consequences.** The ordering constraint becomes a compile-time fact rather than a documentation
claim, which is the strongest form it can take. If the checkpointer fails to provision, Saul does not
provision either, and the capability reports itself unavailable — correct, because a Saul without
conversation persistence is a silently degraded feature rather than an absent one, and silent
degradation is the harder failure to detect.

## ADR-006 — The two source prose prohibitions are superseded and rewritten

**Status:** accepted (user ruling: "supersede — rewrite both prose sites").

**Context.** `src/app/shared/langgraph_layer/checkpointer.py:11-17` and
`tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py:3-8` state that this wiring
must stay commented out, citing an earlier change's decision.

**Decision.** Rewrite both to record that this change supersedes that decision, and to state the
per-process reconciliation: the ingestion graph's consumer is the worker, which never runs the
lifespan, so it is provisioned by a worker hook; the Saul graph and checkpointer serve the API process
and are provisioned by the lifespan.

**Consequences.** A reviewer reading either site no longer concludes that this change is prohibited.
The rewrite carries a specific hazard: `test_checkpointer_lifecycle.py` greps `checkpointer.py` for two
literals it requires to be absent, so a docstring that quotes either one turns that test red for a
reason unrelated to its content. Task 1.3 guards it with its own Proof.

## ADR-007 — Memory is constructed here and specified elsewhere

**Status:** accepted (user ruling: Saul's memory is exclusively cognee's, already specified in the
archived agent-memory work).

**Context.** Provisioning Agent Saul requires an `AgentMemoryService`. It would be easy, while wiring
it, to restate how memory behaves.

**Decision.** Construct the memory service in the provisioning path and supply it to the graph. State
nothing in this capability about how memory is recalled, written, or partitioned. Do not add a third
construction site — the only existing one is `src/tasks/agent_memory_tasks.py:119`.

**Consequences.** No duplicated or drifting memory semantics between this capability and the memory
capability that already owns them. Construction is cheap enough to make this trivially safe: the
constructor takes a partition prefix plus optional callables defaulting to cognee's own functions, so
it opens no client, engine, or session. The Proof is a count — exactly two construction sites in
`src/` — which detects the drift this ADR is meant to prevent.
