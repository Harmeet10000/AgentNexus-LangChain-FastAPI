# Compile graphs once per process and provision Agent Saul

**Class: L.** Cross-cutting: it changes object lifetimes in two different process types (the API
server and the Celery worker) and reverses a prior decision recorded in source prose.

## Why

Three LangGraph graphs are in the wrong place. `build_document_ingestion_graph` is recompiled inside
every Celery job. The ingestion graph, the PageIndex client, and the checkpointer are commented out of
the lifespan. And `app.state.saul_graph` is **never assigned**, so every request to Agent Saul reports
the capability unavailable — the feature is dark in production today.

## What Changes

- **Compilation moves to process scope.** Each graph is compiled at most once per process, never at
  import time, and consumers read the compiled graph from process state.
- **Job-scoped collaborators arrive per invocation** through `config["configurable"]`, not by closure
  capture. A graph never captures a database session or repository, and graph state carries neither.
  A missing collaborator raises a typed exception naming it, not a `KeyError`.
- **Graph provisioning becomes a startup policy**, reusing the existing degrade machinery
  (`lifespan.py:171-342`) rather than restoring hand-rolled try/except blocks. A graph that fails to
  build leaves startup running and its state attribute absent; a dependency reading an unprovisioned
  graph raises a typed service-unavailable naming the capability.
- **Agent Saul is provisioned** in the serving process, with its checkpointer, turning the feature back
  on. Memory semantics are untouched — cognee owns them and they are already specified.
- **The Celery worker gets the same providers** through `worker_process_init`, and releases them
  through `worker_process_shutdown`. The per-task construction currently at `service.py:912-935` is
  hoisted, and its `close_graphiti()` and engine disposal move to the shutdown hook.
- **BREAKING for reviewers, not for callers:** two prose sites in source
  (`src/app/shared/langgraph_layer/checkpointer.py:11-17` and
  `tests/unit/features/ingestion/test_unprovisioned_graph_fails_closed.py:3-8`) currently record that
  this wiring must stay commented, citing an earlier change's decision. Both are superseded and
  rewritten. No public API or response shape changes.

## Capabilities

**New Capabilities**

- `compiled-graph-lifecycle` — process-scoped compilation, per-invocation collaborator supply,
  degrading provisioning, and Agent Saul availability.

**Modified Capabilities**

None. This change **cites** three restored capabilities without amending them:

- `langgraph-checkpointing` already requires that "the constructing process owns the checkpointer pool,
  and teardown distinguishes nothing-to-close from a close". This change satisfies that requirement
  rather than restating it.
- `celery-worker-deployment` already requires explicit task registration and independently verifiable
  worker readiness.
- `document-ingestion-pipeline` already requires that ingestion executes outside the request path and
  that synchronous surfaces fail closed when the shared pipeline is not provisioned.

## Impact

- **Code:** new `src/app/lifecycle/graphs.py`; `src/app/lifecycle/lifespan.py` (the commented blocks at
  `:522-537` and `:549-565`); `src/app/features/documents/service.py:904-947`; `src/tasks/`;
  `src/app/shared/langgraph_layer/open_deep_search/graph.py:278,478,555`;
  `src/app/shared/rag/graphiti/registry.py` (a stale docstring).
- **Behaviour:** Agent Saul stops returning service-unavailable on a healthy process. Ingestion job
  latency drops by one graph compilation per job.
- **Lifetimes:** a database engine and a Graphiti client change from per-task to per-process. This is
  the highest-risk element of the change and is called out in `design.md`.
- **Not touched:** chunking, the embedder, retrieval SQL, fusion, the reranker, the private
  chat-model import sites. `app.state.pageindex_client` stays unwired — `knowledge-stack` owns its
  disposition.
