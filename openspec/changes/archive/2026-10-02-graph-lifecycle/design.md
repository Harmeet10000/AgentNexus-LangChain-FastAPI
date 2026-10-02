# Design — graph-lifecycle

## The unit of scope is the process, and there are two kinds of process

Everything in this change follows from one observation: this application runs in **two process types
with two different startup mechanisms**, and the graph wiring was written as though only one existed.

| | API server | Celery worker |
|---|---|---|
| Startup mechanism | the FastAPI lifespan | `worker_process_init` |
| Shutdown mechanism | the lifespan's teardown | `worker_process_shutdown` |
| Shared mutable home | `app.state` | module globals |
| Runs the lifespan? | yes | **never** |
| Needs the Saul graph | yes | no |
| Needs the ingestion graph | no | yes |

The worker never executes the lifespan. That is not a bug to fix; it is how Celery works. So a graph
whose only consumer is the worker cannot be provisioned by adding a line to the lifespan — and this is
exactly why the ingestion graph's lifespan block was commented out and then re-implemented, badly, as
a per-task construction inside `service.py:912-935`.

The fix is not to pick one home. It is to write the **providers once** and register them from both
places: the lifespan for the graphs the API serves, the worker hooks for the graphs the worker runs.

## Forced ordering — this is a constraint, not a preference

The task groups are ordered by hard dependency:

**checkpointer → Saul → the `configurable` seam → Celery → the KB graph → import-time globals**

- `build_saul_graph`'s `checkpointer` parameter **is not optional**. There is no value meaning
  "absent". Saul therefore cannot be provisioned before a checkpointer policy exists — not as a matter
  of tidiness, but because the call does not typecheck.
- The Celery wiring cannot pass a repository through the invocation configuration until the node reads
  it from there. Task group 3 must precede task group 4.
- The import-time module globals are last precisely because **nothing waits on them**. They are the
  only step in this change with no downstream consumer, which makes them the safest thing to do while
  the rest is still settling.

Every other ordering in this change is discretionary. These three are not.

## Rejected shape — passing the repository in graph state

The obvious alternative to `config["configurable"]` is to add the repository to the graph's state
schema and let it flow through the nodes like any other field.

Rejected, for three reasons that compound:

1. **State is checkpointed.** LangGraph serialises state to the checkpointer between steps. A database
   session or repository is not serialisable, and forcing it through would mean either a custom
   serialiser that stores a useless handle or an exclusion rule that has to be maintained forever.
2. **State is the resumption contract.** A checkpoint is meant to be resumable in a *different*
   process, potentially days later. A repository bound to a session that closed when the job ended is
   the definition of a value that must not survive into a resume.
3. **State is per-run data; `configurable` is per-run wiring.** The distinction already exists in
   LangGraph's own design, and using it costs nothing.

The third reason is the real one. `config["configurable"]` is dependency injection with a
per-invocation lifetime, which is precisely the lifetime a repository has.

The cost of using it is that the key lookup is untyped — a plain dictionary access that fails with
`KeyError` and a bare string in the message. That is why the spec requires a **typed accessor raising a
project exception naming the missing collaborator**: it converts LangGraph's weakest typing surface
back into the project's error contract.

## Rejected shape — a module-level lazy global

```python
_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build().compile()
    return _graph
```

This is genuinely simpler and it would satisfy "compiled once per process". It is rejected for the
graphs that have a provisioning story, and *accepted* for the three in
`open_deep_search/graph.py:278,478,555`.

The difference is whether anything needs to observe the failure. A lazily-built global fails at first
use, deep inside a request or a job, as whatever exception the builder happened to raise. A graph
provisioned at startup fails once, at startup, where the existing degrade machinery can record it and
mark the capability absent — which is the behaviour the fail-closed tests already assert. For
`open_deep_search`, nothing observes provisioning today and the only defect is that importing the
module compiles three graphs; the lazy global fixes exactly that defect and nothing more.

## Rejected shape — making the checkpointer optional

It would be locally convenient to give `build_saul_graph` an optional checkpointer so Saul could be
provisioned even when the checkpointer fails. Rejected: a Saul graph without a checkpointer is a Saul
graph without conversation persistence, which is a silently degraded feature rather than an absent
one. The non-optional parameter is doing useful work — it makes the ordering a compile-time fact
rather than a documentation claim. This change keeps it and orders around it.

## Provisioning is a startup policy, not a try/except

`lifespan.py:171-342` already contains the machinery: a registration mechanism that runs a provisioning
step, records failure, and leaves the process running with the capability's state attribute absent.
The commented-out graph blocks predate it, or ignored it.

Re-adding hand-rolled `try/except` blocks around graph construction would produce a second, subtly
different degrade path — one that logs differently, sets state differently, and drifts. Routing graph
provisioning through the existing policy mechanism means the fail-closed behaviour the ingestion tests
already assert is the *same* behaviour, not a parallel implementation of it.

## The highest-risk element: a lifetime change, not a code move

Task 4.2 moves the construction at `service.py:912-935` from per-task to per-process. Read as a diff it
is a hoist. Read as behaviour it changes the lifetime of **a database engine and a Graphiti client**.

Three failure modes, none of which a type checker sees:

- **Teardown in the wrong place.** If `close_graphiti()` and engine disposal stay attached to the task
  instead of moving to `worker_process_shutdown`, the second job in a process operates on a closed
  client. If they are dropped entirely, connections leak for the life of the worker.
- **Init in the wrong place.** Constructing in the task-level hook rather than the process-level hook
  means every task rebuilds anyway — the change becomes a no-op that reads as done.
- **Teardown without init.** A process that failed to initialise still runs its shutdown hook. Closing
  what was never opened is a crash during shutdown, which is the worst place to raise. Hence the
  explicit requirement that no release is attempted for a resource that was never constructed, with
  its own Proof at task 4.3.

The mitigation is that all three are provable with spies and no infrastructure: count constructions,
count closes, and drive shutdown with no preceding init.

## The two prose prohibitions, and why superseding them is not a reversal

Two places in the tree currently state that this wiring must stay commented out:
`src/app/shared/langgraph_layer/checkpointer.py:11-17` and
`tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py:3-8`.

They are not wrong about their own moment. They record a decision from an earlier change that the
shared graph wiring would not be enabled then. Reading them today, though, a reviewer concludes this
change is prohibited — so leaving them in place while doing the opposite is worse than either
following them or removing them.

The reconciliation is the per-process split at the top of this document. The earlier prohibition
assumed one provisioning site. The ingestion graph's consumer is the worker, which never runs the
lifespan, which is why it gets a worker hook rather than a lifespan line. The Saul graph and the
checkpointer serve the API process, which does run the lifespan, and get one. Both prose sites are
rewritten to say that.

There is a landmine in doing so. `test_checkpointer_lifecycle.py` greps `checkpointer.py` for two
literals it requires to be **absent** — the saver's connection-string classmethod name, and the old
pool `hasattr` guard. A docstring rewrite that quotes either string turns that test red for a reason
that has nothing to do with the docstring. Task 1.3 exists solely to keep that from happening.

## What this change deliberately does not touch

- **Memory semantics.** Cognee owns them and they are already specified. This change constructs
  `AgentMemoryService` and passes it; it says nothing about recall, writes, or partitioning. The
  constructor takes a partition prefix and optional callables defaulting to cognee's own functions —
  no client, no engine, no session — so constructing it in the provisioning path is genuinely cheap.
- **`app.state.pageindex_client`.** It stays unwired. `knowledge-stack` owns its disposition, and
  wiring a client here that a later change removes is churn.
- **Chunking, the embedder, retrieval SQL, fusion, the reranker.** Other changes in this cluster own
  them.
