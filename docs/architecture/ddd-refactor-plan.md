# DDD / Hexagonal / CQRS Refactor — Implementation Plan

Status: **v3 — Sol round-2 blockers applied**
Date: 2026-11-20
Branch: `refactor/ddd-slices` (from clean `main` @ `558d968a`)
Scope: `src/app/` feature slices + shared runtime; no product behaviour change.

---

## 0. Baseline (measured after parking WIP)

The in-flight `documents` work was moved to `wip/documents-deletion` (commit
preserved). `main` is the reference baseline.

| Gate | Result on clean `main` |
|---|---|
| `uv run ruff check src/` | **3 errors (pre-existing)** |
| `uv run ty check src/` | **2 diagnostics (pre-existing)** |
| `uv run pytest tests/unit --collect-only` | **2750 collected, 0 errors** |
| `uv run lint-imports` | **5 contracts KEPT** |

Pre-existing failures Phase 0 must clear:

| File:line | Rule |
|---|---|
| `src/app/ai/evaluation/assertions.py:31` | `typing-only-standard-library-import` |
| `src/app/features/documents/ingestion/legacy_service.py:245` | `no-self-use` |
| `src/app/worker/celery/app.py:10` | `typing-only-standard-library-import` |
| `src/app/features/documents/application/queries/documents.py:920` | `ty` — `Pregel.ainvoke` arg |
| `src/app/features/documents/retrieval/graph/nodes.py:476,485` | `ty` — `_identify_sources` arg |

---

## 1. Locked decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Ports split**: aggregate repositories in `domain/repositories.py`; external capabilities in `application/ports/` | Domain must not know what email is; it must know what a repository is |
| D2 | **Clock + IdGenerator are domain ports**; `SystemClock` is an adapter | 14 domain call sites read ambient time |
| D3 | **No `internal/` wrapper.** Private zone = `domain/`, `application/`, `adapters/`, enforced by `lint-imports` | Python has no compiler-enforced package privacy |
| D4 | **No `cmd/` tree.** Entry points = `[project.scripts]` + `entrypoints/` | Already exist: `dev`, `worker`, `beat`, `replay-outbox` |
| D5 | **Canonical `Module` descriptor lives at `app.shared.module`** until the D11 rename moves it to `app.platform.module`. `Module(routes, consumers, schedules)` holds **immutable descriptors or zero-argument factories only**. A slice's `module.py` may import descriptor types and application DTOs at module-import time; router, Celery, scheduler, driver, and persistence imports occur **only inside the relevant factory**. Importing every `module.py` in an API-only, worker-only, and scheduler-only process must succeed without composing another surface | A routes-only variant is a half-abstraction; a "lazy" descriptor that imports Celery during API discovery is a regression |
| D6 | **`public.py`** per slice, not `api.py` | `api.py` collides with every `features/*/api/` package |
| D7 | **`contracts/`** per slice, append-only, `_v1` suffix | Only legal cross-slice import |
| D8 | **Consumers live in the slice**; only generic lifecycle/execution stays in `platform/worker`. Every consumer move atomically updates `JOB_CATALOG[*].declaring_module`, derived `TASK_MODULES`, worker profiles, module consumer/schedule descriptors, beat composition, completeness tests, and test stubs carrying old task-module strings | Moving the file without the metadata creates a silent discovery bug |
| D9 | **`billing/` is not converted** | Subdomain layout already encodes the seams |
| D10 | **`users/` is folded into `auth/`** (user decision) | `users` is an administrative projection over auth accounts |
| D11 | **`platform/` physical rename is deferred** to one atomic post-phase migration | Avoids two homes while 5 agents work |
| D12 | **`documents/ingestion` + `documents/retrieval` are promoted to slices last** | Only after `legacy_*` is deleted |

---

## 2. Final target tree

```
src/
├── alembic/                        # unchanged (env.py owned by L5)
├── database/                       # registry + seeders
└── app/
    ├── main.py                     # app factory — wiring only
    ├── server.py                   # = cmd/api
    │
    ├── entrypoints/                # = cmd/*
    │   ├── api.py  worker.py  relay.py  scheduler.py
    │
    ├── kernel/                     # PURE algebra + vocabulary, no I/O, no frameworks
    │   ├── clock.py  ids.py  errors.py  enums.py  models.py  tenancy.py
    │
    ├── shared/                     # transition home for Module descriptor
    │   ├── module.py               #   → platform/module.py at D11
    │   └── jobs/                   #   wire schemas (exempt from rule of three)
    │
    ├── platform/                   # technical plumbing — zero domain rules
    │   ├── config/  logging/  http/  security/  resilience/
    │   ├── db/{postgres,mongodb,redis,neo4j}/
    │   ├── blob/  mailer/  broker/  messaging/  outbox/  cache/
    │   └── worker/                 #   GENERIC only: celery_app, signals, execution/,
    │                               #   generic CapabilityOwner
    │
    ├── capabilities/               # shared DOMAIN capability
    │   ├── ai/  retrieval/  extraction/
    │
    └── features/
        ├── audit/
        │   ├── module.py           # MODULE = Module(routes=..., consumers=..., schedules=...); recorder binder is separate
        │   ├── public.py           # AuditAction, RecordAuditEntry DTO, AuditRecorder Protocol
        │   ├── contracts/audit_entry_recorded_v1.py
        │   ├── domain/{entities,value_objects}/ enums.py events.py services.py errors.py repositories.py
        │   ├── application/{commands,queries,jobs,ports,dto,subscribers}/
        │   ├── adapters/inbound/http/{router,dto,presenters,dependencies}.py
        │   └── adapters/outbound/postgres/{models,repository,mappers}.py
        │       └── porttest/
        ├── health/     … probes in adapters/outbound/*/probe.py
        ├── profile/    … delegates credential/session mutation to auth
        ├── auth/       … hexagonal + public.py + api/http/admin_users_* + application/queries|commands/admin_*
        ├── documents/  … hexagonal; ingestion/retrieval promoted last
        ├── agent_saul/ … hexagonal; graph/ stays at slice root
        ├── crawler/    … hexagonal
        ├── search/     … hexagonal (ports first)
        └── billing/    … unchanged (D9)
```

### Slice invariants

> **Why `inbound`/`outbound`, not `in`/`out`:** `in` is a Python reserved
> keyword, so a package directory named `in` cannot appear in an import
> statement — `from app.features.x.adapters.in.http.router import router` is a
> `SyntaxError`, and every consumer would need `importlib.import_module`. The
> directories are therefore `adapters/inbound/` and `adapters/outbound/`.

1. `domain/` imports: stdlib, `pydantic`, `app.kernel`. **Nothing else.**
2. `application/` imports: `domain/`, `application/ports/`, `kernel/`, other slices' `contracts/` + `public.py`. **Never** `adapters/`, `api/`, `app.worker`.
3. `adapters/` imports: `domain/`, `application/`, `platform/`.
4. `api/` imports: `application/`, `application/dto/`, `kernel/errors`.
5. `module.py` **may** import: descriptor types (`app.shared.module`), application DTOs, and `public.py`. It **must not** import routers, Celery, scheduler, drivers, or persistence at module scope — those appear only inside factories.
6. `contracts/` imports: stdlib + `pydantic`.
7. `public.py` imports: `application/` + `domain/`.
8. **Every package `__init__.py` stays side-effect-free** — no re-exports, no composition.

### DTO taxonomy

| Kind | Lives in | May be imported by |
|---|---|---|
| HTTP request/response | `adapters/inbound/http/dto.py` | `module.py` only |
| Application command/query | `application/dto/` | `public.py`, `module.py` |
| Public facade DTO | `public.py` | other slices |
| Row ↔ entity mapping | `adapters/outbound/<db>/mappers.py` | that adapter only |
| Integration event | `contracts/*_v1.py` | publishers + subscribers |

---

## 3. DRY charter (binding)

DRY is **one definition per concept**, not one function per shape.

### 3.1 Mandatory single homes

| Concept | Single home | Must never appear elsewhere |
|---|---|---|
| Result algebra | `returns.result` (external, unmodified) | no project Result type |
| Failure classification + base errors | `kernel/errors.py` | no per-feature base error class |
| HTTP envelope + rendering | target `platform/http/` (today `api/envelope.py`, `shared/result/render.py`) | no second re-export path (`app.utils` re-exports removed) |
| SQL diagnostic enrichment | target `platform/db/postgres/` (transition: existing postgres namespace) | not in `kernel/` |
| Clock protocol | `kernel/clock.py` | no `datetime.now()` inside `domain/` |
| IDs | `kernel/ids.py` | no ad-hoc `uuid4()` in domain |
| Broker wire names | `platform/messaging/names.py` | no string literal task names |
| Broker payload schemas | `platform/messaging/payloads.py` | no slice re-declares a task payload |
| Job routing (queue, capability) | `platform/messaging/catalog.py` | no slice defines a queue name |
| Outbox table + relay | `platform/outbox/` | no slice runs its own relay |
| Celery app / signals / retry policy | `platform/worker/` | no slice constructs a `Celery()` |
| Integration event schema | `features/<pub>/contracts/<event>_v1.py` | no slice defines an event it does not publish |
| `ErrorKind` vocabulary | `kernel/errors.py` | feature `Code` enums stay feature-owned |
| `Module` descriptor | `app.shared.module` → `platform/module.py` (D11) | no slice-local descriptor shape |

> **Phase 5 deletes `kernel/result.py`.** `returns.result` remains the sole Result
> algebra; failure vocabulary moves to `kernel/errors.py`; HTTP envelope/rendering
> moves to the current HTTP platform namespace; SQL diagnostics move to the current
> PostgreSQL platform namespace. Compatibility re-exports are deleted after callers
> migrate.

### 3.2 Anti-duplication rules

1. **No `BaseRepository`, no `CrudService[T]`, no shared per-feature ORM base.** Shared base is `database/base.py`.
2. **Mappers are per-adapter.** Never shared across slices.
3. **No shared DTOs across slices.** Consume `contracts/` or call `public.py`.
4. **Rule of three** applies **only to incidental implementation helpers** — extraction on the third occurrence in a different slice, with no domain vocabulary.
   **Exempt (extract on first use, because ownership/compatibility demand one canonical definition):** wire schemas, integration contracts, boundary ports, clocks, ID providers, base error/failure taxonomy, generic process/loop capability-lifecycle ownership primitives.
5. **Ports are interfaces, not implementations.** The only shared code is the `porttest/` suite.
6. **`porttest/` is the one place duplication is mandatory** — every adapter subclasses it.
7. **No result-unwrapping helper that raises.** Explicit `isinstance(r, Failure)` per ADR-002.
8. **Never copy a route prefix, cron expression, or queue name.**
9. **`ConfigDict` is not shared if the semantics differ.** `frozen`, `extra`, `alias_generator` change behaviour.

### 3.3 Rejected abstractions

- `BaseEntity` with `id/created_at/updated_at` across aggregates.
- `UnitOfWork` before a second slice needs it.
- Shared `Probe` base class — share the Protocol, not the base.
- A `common/` feature package.

---

## 4. Phases

One branch (`refactor/ddd-slices`), **one commit per phase**, green gates, no HTTP
behaviour change. Landing is serial; only slice-local work runs in parallel (§6).

### Phase 0 — Baseline repair

Fix the 5 pre-existing failures in §0. No structural change.
**Gate:** ruff 0, ty 0, `pytest --collect-only` 2750, `lint-imports` 5 kept.

### Phase 1 — `audit` (259 LOC)

Least coupled; `billing` imports it 29×.

```
domain/enums.py                 ← AuditAction out of models.py
domain/entities/audit_entry.py  ← pure; no SQLAlchemy
domain/repositories.py          ← Protocol AuditLogRepository
domain/errors.py                ← from exceptions.py
application/commands/record.py  ← RecordAuditEntry use case
application/queries/list.py
application/ports/recorder.py   ← Protocol AuditRecorder
application/dto/
adapters/outbound/postgres/{models,repository,mappers}.py
adapters/outbound/postgres/porttest/
adapters/inbound/http/{router,dto,presenters}.py
module.py  public.py
```

**Transaction contract (blocker fix):**

- `audit.public` exports **only** `AuditAction`, immutable `RecordAuditEntry`
  input/output DTOs, and the `AuditRecorder` application protocol. It exports
  **no SQLAlchemy model and no concrete repository**.
- L5 composes a **session-bound recorder** from `audit.module` using the billing
  transaction's existing `AsyncSession`; billing receives that recorder by injection.
- `audit.module.build_audit_recorder(session)` is a separate L5 composition
  function, **not a field of `Module`**. It imports the PostgreSQL adapter only
  inside its function body, binds the caller-owned `AsyncSession`, and never
  commits or rolls back that session.
- Characterization tests must prove: (a) audit-flush failure rolls back the primary
  billing write, and (b) no billing module imports audit persistence.
- **Do not** introduce an independently committing audit service.

**Gate:** `rg "features\.audit" src/app/features/billing` → only `audit.public` / `audit.contracts`; rollback characterization tests green.

### Phase 2 — `health` (1080 LOC)

```
application/service.py        ← orchestrates probes; no driver imports
application/ports/probes.py   ← Protocol Probe
adapters/outbound/{postgres,mongodb,redis,neo4j,celery,system}/probe.py
adapters/inbound/http/{router,dto}.py
module.py
```

- Delete `gateway.py` (6-line stub) and `events.py` (5-line stub) — reserved-stub rule.
- **Gate:** `health/service.py` imports no driver (`sqlalchemy`, `motor`, `neo4j`, `redis`, `celery`, `psutil`).

### Phase 3A — Auth public seam

- **L3 owns** `src/app/features/auth/{public.py,application/**,domain/**}` plus all affected auth tests.
- Define auth-owned public operations + DTOs for: account read, profile mutation,
  credential change, session revocation, role/status administration, impersonation.
- Extract the named operations **without changing transport or persistence**.
- `public.py` never exposes Beanie documents, repositories, adapters, or HTTP dependencies.
- **Do not** add this behaviour to the 1,771-line `AuthService`.
- **Gate:** `auth/public.py` exposes no adapter/persistence symbol; existing auth tests unchanged and green.

### Phase 3B — Fold `users` into `auth`

- Move `users/router.py` + DTOs → `auth/api/http/admin_users_{router,dto}.py`.
- Reads → `auth/application/queries/admin_users.py`.
- Mutations → `auth/application/commands/admin_accounts.py`.
- Merge repository behaviour behind auth-owned account ports/adapters.
- Merge error codes into auth-owned errors.
- Update **both** API versions; then delete `users/`.
- **Gate:** `rg "features\.users" src/` → 0; route-count/OpenAPI diff limited to the intended moves.

### Phase 3C — Auth worker ownership

- Move `worker/jobs/auth.py` and `worker/operations/reconcile_pending_revocations.py`
  beside auth.
- Move its periodic schedule into `auth.module`.
- Atomically update catalog/discovery (D8 checklist).
- **Gate:** worker completeness + beat composition tests green for every role.

### Phase 4 — `profile` (494 LOC)

Structurally slice-local; repoints to `auth.public` (depends on Phase 3A).

- Profile does **not** own credential/session mutation — delegates to auth use cases.
- Delete stubs `gateway.py`, `events.py`.
- **Gate:** no `features.auth.*` import other than `auth.public`.

### Phase 5 — Shared primitives + enforcement

- Delete `kernel/result.py` per §3.1; remove `app.utils` re-export path.
- Add contracts (§5) and ast-grep rules.
- **Gate:** ruff, ty, `lint-imports`, `ast-grep scan src/`, full pytest, `alembic check`.

### Phase 6 — `search` ports

- `application/ports/`: `SearchRepository`, `Embeddings`, `JobQueue`.
- Replace direct adapter + `api.http.dto` imports in `application/service.py`.
- Move `worker/jobs/search.py` → `features/search/adapters/inbound/worker/tasks.py`, with
  the full D8 metadata update in the same commit.

### Phase 7+ — big slices

`auth` domain cleanup (Beanie out of `domain/user.py`);
`crawler` (`src/app/worker/capabilities/crawler.py` →
`src/app/features/crawler/adapters/inbound/worker/capabilities.py`);
`agent_saul`; `documents`
(`src/app/worker/capabilities/documents.py` →
`src/app/features/documents/adapters/inbound/worker/capabilities.py`, after legacy
ingestion/retrieval deletion). Each worker move includes the complete atomic D8
metadata checklist. Then perform the deferred D11 `platform/` rename.

---

## 5. Enforcement (add per phase, never ahead of the code)

```toml
# one contract per converted slice — enumerate, never a global app.features source
[[tool.importlinter.contracts]]
name = "audit domain is framework-free"
type = "forbidden"
source_modules = ["app.features.audit.domain"]
forbidden_modules = [
  "fastapi", "starlette", "sqlalchemy", "beanie", "motor", "pymongo",
  "redis", "celery", "httpx", "aiohttp", "socket",
  # legacy technical namespaces AND the target one — both forbidden from slice one
  "app.connections", "app.shared", "app.config", "app.utils", "app.events",
  "app.observability", "app.resilience", "app.platform", "app.worker",
]

[[tool.importlinter.contracts]]
name = "audit application does not import its adapters or transport"
type = "forbidden"
source_modules = ["app.features.audit.application"]
forbidden_modules = ["app.features.audit.adapters", "app.features.audit.api"]

[[tool.importlinter.contracts]]
name = "billing reaches audit only through public/contracts"
type = "forbidden"
source_modules = ["app.features.billing"]
forbidden_modules = [
  "app.features.audit.domain", "app.features.audit.application",
  "app.features.audit.adapters", "app.features.audit.api",
]

[[tool.importlinter.contracts]]
name = "contracts are dependency-free of implementations"
type = "forbidden"
source_modules = ["app.features.*.contracts"]
forbidden_modules = ["app.features", "app.worker", "app.connections"]
```

> **Transition rule:** from the first converted slice, domain contracts forbid both
> the legacy technical namespaces and `app.platform`. The D11 commit removes legacy
> entries only after their imports disappear.
> Do **not** add `source_modules = ["app.features"]` against all feature internals —
> it would forbid a slice importing its own domain. Cross-slice rules are per consumer.

---

## 6. Delegation plan (5 × GPT-5.6 Luna)

**One integration owner (L5); slice agents never touch shared composition files.**

| Agent | Owns (exclusive) | Wave |
|---|---|---|
| **L5 — integration owner** | `src/app/api/**`, `src/app/main.py`, `src/app/server.py`, `src/app/entrypoints/**`, `src/app/lifecycle/**`, `src/app/lifespan/**`, `src/database/**` (registry + seeders), `src/alembic/env.py`, `tests/conftest.py`, `tests/integration/conftest.py`, root/integration/deployment/property tests, `pyproject.toml`, `.ast-grep/**`, `src/app/shared/module.py`, `src/app/shared/jobs/catalog.py`, `src/app/worker/celery/{app,cli,signals,task,observability,registration,schedules,profiles,completeness}.py`, `src/app/worker/{jobs,operations,capabilities}/**` **except auth-owned files**, cross-slice rewiring, Phase 0 | 1 + serial landing |
| **L1 — audit** | `src/app/features/audit/**`, `src/app/features/billing/**` (audit imports + recorder injection only), `tests/unit/features/audit/**` | 1 |
| **L2 — health** | `src/app/features/health/**`, `tests/unit/features/health/**` | 1 |
| **L3 — auth seam** | `src/app/features/auth/{public.py,application/**,domain/**}`, `tests/unit/features/auth/**` | 1 |
| **L4 — profile structure** | `src/app/features/profile/**`, `tests/unit/features/profile/**` | 1 |
| **L3b — users fold** | `src/app/features/users/**` (deletion), `src/app/features/auth/api/http/admin_users_*`, `src/app/features/auth/application/queries/admin_users.py`, `src/app/features/auth/application/commands/admin_accounts.py` | 2 (after L3) |
| **L3c — auth worker** | `src/app/worker/jobs/auth.py`, `src/app/worker/operations/reconcile_pending_revocations.py`, `src/app/features/auth/module.py` schedules | 2 (after L3) |
| **L4b — profile repoint** | `src/app/features/profile/**` imports only | 2 (after L3) |

Rules:
- Slice agents produce slice-local commits; L5 lands shared-file changes serially in
  order `audit → health → auth → profile`.
- No agent edits `pyproject.toml` except L5.
- Edge rewiring edits only the consumer file and belongs to L5.
- `documents` is untouchable in this wave (parked on `wip/documents-deletion`).
- Every ownership path above is repository-relative and begins with `src/` or `tests/`.

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Architecture cycle survives after import cycles vanish | High | Review *state authority*, not just imports; auth owns accounts, users is a projection |
| Facade preserves imports but changes commit ownership | High | Audit recorder is session-bound and injected; rollback characterization test is a gate |
| "Lazy" descriptor still imports Celery during API discovery | High | D5 factory-only rule + three-process import gate (API-only, worker-only, scheduler-only) |
| ORM move breaks Alembic autogenerate | Medium | Identical table names/columns/constraints; assert `Base.metadata` keys; `alembic check` |
| Beanie model registration breaks at startup | Medium | Update `document_models` in lifespan in the same commit |
| `database.registry.MODEL_MODULE_NAMES` stale | High | Atomic update + import smoke test |
| Import-time side effects via `__init__.py` | Medium | Invariant 8 + package-import test with no I/O clients configured |
| Router registration drift | Medium | L5 alone edits `api/{v1,v2}/__init__.py`; OpenAPI route-count test |
| `public.py` becomes a god facade | Medium | Only exports actually used by another slice; prune per phase |
| Big-bang `platform/` rename | High churn | D11: one atomic migration, dual-namespace contracts |
| Test mirroring drift | High | Move tests in the same commit; node-ID inventory gate across legacy + `tests/unit/features/**` |

---

## 8. Definition of done (global)

- `ruff check src/ tests/` → 0 · `ty check src/` → 0 · `lint-imports` → all KEPT
- `pytest tests/unit -q` → green, count ≥ 2750 · `ast-grep scan src/` → 0
- `alembic check` → no autogenerate diff; `Base.metadata` table keys unchanged
- Node-ID inventory: no test lost between legacy and `tests/unit/features/**` trees
- For every worker role: catalog `declaring_module` → `TASK_MODULES` → profile →
  beat composition → completeness test all resolve
- Importing every `module.py` succeeds in API-only, worker-only, scheduler-only processes
- No slice imports another slice outside `public.py` / `contracts/`
- No `domain/` module imports a framework, `platform/`, or another slice
- No `datetime.now()` / `uuid4()` in any converted `domain/`
- No `__init__.py` side effects
- `docs/architecture/` reflects the landed tree
