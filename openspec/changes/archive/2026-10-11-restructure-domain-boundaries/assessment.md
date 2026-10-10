# Assessment of the supplied architecture review

Reviewed against the working tree on 2026-10-10. The current worker migration is in progress; this plan starts from its final layout. File counts and cycle counts in the attachment were not independently reproduced and are not acceptance criteria.

| Recommendation | Finding and decision |
|---|---|
| Keep billing together | Adopt. Existing billing subfeatures collaborate within one context; flattening them into peer features would weaken ownership. |
| Put all tasks under worker/tasks | Valid alternative, but unnecessary after the current migration. Billing orchestration now lives in billing/jobs.py, and feature task files are thin adapters. Keep them unless measured dependency problems justify another move. |
| Shared/connection modules import features | Confirmed examples: connections/cognee.py imports the document embedding dimension; utils/rate_limit/dependencies.py imports auth token decoding; Docling chunker imports document policy/classification; retrieval graph imports document services. Extract pure contracts or relocate domain implementations. |
| Any infrastructure-to-feature import is wrong | Correct the rule. Lifecycle, API composition, worker composition, Alembic registry and seeders necessarily wire features. MCP domain handlers can invoke feature services; protocol/auth infrastructure should not import domain internals. |
| Merge connections/outbox with shared/outbox | Do not call this duplication without evidence. The former constructs the relay and checks startup; the latter implements relay, model and writer. A common events namespace is optional; ownership and lifecycle remain separate. |
| Merge object storage/Tavily modules | Check factory-versus-adapter responsibilities first. Pick the canonical behavior by its contract and tests, not how many importers it has. |
| One rate limiter/chunker | Consolidate shared policy where identical. Preserve HTTP quota versus provider throttling semantics, and source-specific chunking engines. Redis client/TTL/key compatibility must be tested. |
| Replace 22 error modules with one hierarchy | Reject. Per-feature closed unions and shared ErrorKind implement ADR-001/002. A pure kernel can own the existing common spine; feature errors remain local, with no concrete subclassing. HTTP rendering stays outside the pure kernel. |
| Auth and users both own User | Confirmed shared data ownership: users/repository.py imports auth.User and performs role, activation and deletion writes. Define identity persistence ownership with explicit admin use cases; do not delete either API or assume repositories are duplicates. |
| Merge ingestion and documents | Not proven equivalent. IngestionService runs a graph inline and returns entity/clause/relationship counts. Durable document ingestion has separate task/storage semantics. Preserve both public surfaces and use adapters where reuse is demonstrated. |
| Move Agent Saul graph and legal tools into features | Adopt. Domain-specific graph nodes and tools belong with their feature; pure runtime, embeddings and generic tools remain reusable. Preserve node names, checkpoint state and tool names. |
| Make feature initializers lightweight | Adopt. auth/__init__.py eagerly imports router, repositories and websocket implementation despite its docstring. Audit actual re-export users and migrate imports before removal. |
| Move lifecycle/signals.py into worker | Reject. It explicitly delegates SIGINT/SIGTERM to Uvicorn; it is web lifecycle guidance, not Celery wiring. |
| Delete search tombstone | Conditional cleanup after checking source, tests, documentation and dynamic import references. The tombstone is deliberate and not a second implementation. |
| Move chat models into Agent Saul | Evaluate ownership, not file count. Preserve SQLAlchemy identity, table names, foreign keys and historical migration imports; chat may support more than one agent. |
| Flatten credits unconditionally | Avoid forcing multiple models/repositories/routers into giant files. Prefer consistent naming and a small public contract; flatten only if the resulting modules remain cohesive. |
| Move Lynk outside src | Adopt tools/lynk as the planned home; the Go module and test fixtures exist. No Python caller was found in the inspected references. Verify deployment/build/CLI callers before moving; do not create a new repository or adapter speculatively. |
| Delete pyc/egg-info | Treat these as generated artifacts: fix packaging/ignore rules and remove only confirmed disposable output. Editable-install metadata may be regenerated immediately; deleting it is not an architectural migration. |
| Remove RAG type exclusion | Adopt after an explicit diagnostic baseline and fixes. The excluded tree contains first-party integration code; no blanket new ignores should conceal the diagnostics. |
| Central database registry | Adopt. Preserve every current model and the legacy document_vectors registration; registration has a documented rollout dependency. |
| Missing registry model always makes autogenerate drop a table | Correct for this project: env.py currently excludes reflected tables with no metadata counterpart. That prevents the described drop but can hide a missing model. Expected-table-set verification is still required. |
| Drain queues before moving tasks | Not intrinsically required here: wire names were already explicit and remain stable. Drain only when changing payload compatibility or execution semantics; do not clear queued work. |
| Enforce all boundaries in warning mode | Start with a recorded violation inventory, then make each migrated boundary blocking. Keep already passing worker contracts blocking throughout. No indefinite blanket warning mode. |

Verified sources include src/app/features/auth/__init__.py, src/app/features/users/repository.py, src/app/features/ingestion/service.py, src/app/lifecycle/signals.py, src/app/connections/cognee.py, src/app/utils/rate_limit/dependencies.py, src/app/shared/rag/docling/chunker.py, src/app/shared/langgraph_layer/retrieval_kb/, src/alembic/env.py and pyproject.toml.
