# Architecture

The committed record of the implemented standard. The richer working rules in
`.opencode/instructions/ARCHITECTURE-RULES.md` are not version-controlled;
this file is. If the two disagree, this file describes what is enforced by
gates (`ruff`, `ty`, `lint-imports`, `pytest`).

## Model

Modular monolith. `src/app/` is one deployable composed of feature slices
under `features/` plus shared runtime (`shared/`, `kernel/`, `resilience/`,
`retrieval/`, `connections/`, `security/`, `observability/`, `utils/`) and
composition roots (`api/`, `lifecycle/`, `worker/`, `middleware/`, `mcp/`).

Each restructured feature is hexagonal: a domain core, an application layer
around it, and adapters plus transport at the rim.

```
features/<slice>/
├── api/          # transport: HTTP routers/dependencies/DTOs, WebSocket endpoints
├── application/  # use cases, orchestration, application ports; CQRS-lite commands/ queries/ where present
├── domain/       # entities, state machines, domain errors — no framework imports
└── adapters/     # outward bindings: persistence, external services
```

Known deviations from the plain four folders:

- `features/agent_saul/graph/` — the LangGraph orchestration sits at feature
  root, beside the four layers. Graph nodes are application-side
  orchestration; they import `domain/` and are wired by `application/`.
- `features/documents/ingestion/` — the folded legacy ingest pipeline kept as
  a shim subpackage (chunking, enrichment, parsing, graphiti verifier). New
  ingestion work extends it in place; it is not a fifth layer.
- CQRS-lite: `documents` splits use cases into `application/commands/` and
  `application/queries/`. Other slices use plain application services until
  the command/query split pays for itself.

## The dependency rule

Dependencies point inward only: `api → application → domain`; `adapters →
application/domain`. Domain imports nothing outward. Shared runtime
(`shared/`, `utils/`, `security/`, `retrieval/`, `connections/`) must not
import feature code — features import shared, never the reverse.

This is enforced, not aspirational: `lint-imports` reads the five contracts
in `pyproject.toml` (`[tool.importlinter]`):

1. **Reusable runtime does not import domain implementations** — `app.shared`,
   `app.utils`, `app.connections`, `app.retrieval`, `app.security` cannot
   import `app.features`.
2. **Pure contracts stay free of features and workers** — `app.retrieval`,
   `app.security` cannot import `app.features` or `app.worker`.
3. **Web and producers do not load worker execution** — `app.api`, `app.main`,
   `app.lifecycle`, `app.connections`, `app.shared`, `app.config`,
   `app.middleware` cannot import `app.worker`.
4. **Wire schemas are independent of feature implementations** —
   `app.shared.jobs.names`, `app.shared.jobs.payloads` cannot import
   `app.features`, `app.worker`, or `app.connections`.
5. **Domain use cases do not import task entry points** — feature services and
   application layers cannot import `app.worker`.

Composition roots — `app.api`, `app.lifecycle`, `app.worker`, the database
registry, seeders — wire features by design and are deliberately absent from
every `source_modules` list. Their absence from a contract IS the exception.

### The single-segment-`*` wildcard trap

In contract 5, `app.features.*.service` matches **exactly one segment**. When
the hexagonal restructure moved use cases from `features/auth/service.py` to
`features/auth/application/service.py`, the wildcard silently stopped
matching — `lint-imports` still passed, but the new nested modules were
unprotected. Each hexagonal feature therefore carries an explicit
`app.features.<slice>.application.**` entry. When you move a module that a
contract's wildcard used to catch, add the explicit pattern in the same
commit and verify with `lint-imports` (break the rule once in a scratch
branch if you want proof the contract still bites).

## Per-feature disposition

| Feature | Disposition | Notes |
|---|---|---|
| `auth` | hexagonal | `api/{http,websocket}`, `application/` (incl. `revocation/`), `domain/`, `adapters/persistence/mongodb` |
| `documents` | hexagonal + legacy shim folded | `application/commands|queries|tools|evidence`, `adapters/persistence/postgres`; legacy ingest lives under `ingestion/` behind `api/http/legacy_ingest_router.py` (deprecation path) |
| `agent_saul` | hexagonal, graph at feature root | `graph/` beside the four layers (see above); transcript persistence under `application/transcript/` + `adapters/persistence/` |
| `billing` | subdomain-layered **by design** | not converted — see below |

### Why billing stays subdomain-layered

`billing/` is organized by subdomain — `payments/`, `providers/`,
`subscriptions/`, `invoices/`, `credits/`, `dunning/`, `plans/`, `webhooks/` —
each owning its `service`/`repository`/`gateway`/`dto`/`events`/`router`
cluster. The seams a hexagonal conversion would produce already exist as
subdomain boundaries with explicit gateways and repositories; re-slicing the
same seams into `api/application/domain/adapters` folders would move code and
change nothing enforceable. Hexagonal layout is for slices whose internals
leaked (transport tangled with persistence), which billing's are not. Do not
convert it without a concrete violation to fix.

## The kernel rule

`app.kernel` holds pure primitives — `clock`, `ids`, `result`, `errors`,
`enums`, `tenancy` — importable from anywhere, including layers that run
below the features (the `connections → events → relay` path). Kernel modules
take no settings, no I/O clients, no feature imports. Precedent:
`kernel/tenancy.py` owns `personal_tenant_id` because the events relay cannot
import `features.auth`; `features.auth.domain.tenancy` re-exports it for
feature-internal callers instead of the relay importing a feature.

When a "shared" concept is needed by something beneath the feature layer, the
canonical definition goes to `kernel/` and features re-export — never the
other way around.

## Reserved-stub rule

A seam is created when a second implementation or a layering contract requires
it — not in anticipation. Reserved-stub packages ("Reserved: X will live
here") must never carry imports, code, or behavior: they rot, get
type-checked, and document an architecture that does not exist. The only
stub sites in the tree (`auth` and `agent_saul` `application/events.py` +
`application/ports/gateway.py`, `documents/application/ports/gateway.py`) are
deliberate docstring-only wayfinding markers, restored by explicit decision
(5f8ffc0) — adding more requires the same explicitness, and the moment a stub
grows an import it is either implemented or deleted.

## Shared subsystems

Shared subsystems follow the same inward-facing split as features when they
grow beyond one module. Example: `shared/rate_limit/` —

```
shared/rate_limit/
├── application/       # policy: scopes, window configuration
└── adapters/redis/    # fixed-window quota, GCRA throttle, FastAPI dependency wiring
```

`shared/jobs/` is the wire-schema exception: `jobs/names.py` and
`jobs/payloads.py` are contract 4's protected wire formats and must stay
dependency-free. Single-module shared utilities (e.g. `shared/upload_validation.py`)
stay flat until they need the split.

## Adding a new feature

1. Create `features/<slice>/{api,application,domain,adapters}/` (plain
   service-based layout is acceptable only if the slice is small and stays
   small; convert when transport and persistence start sharing a module).
2. Domain first: entities, state, domain errors. No FastAPI, no Redis, no
   ORM imports in `domain/`.
3. Application use cases next; outgoing calls behind ports defined in
   `application/`, implemented in `adapters/`.
4. Transport last: routers/dependencies/DTOs under `api/`. Handlers stay thin;
   they call application code and map results.
5. If the slice enqueues Celery tasks, application code imports task names
   from `shared/jobs/names.py` and never `app.worker` (contract 5).
6. Register the slice's routers in `app/api`, lifespan hooks in
   `app/lifecycle`, worker tasks in `app/worker` — composition roots only.
7. Add an explicit `app.features.<slice>.application.**` entry to contract 5
   if the slice has an application layer.
8. Mirror the layout in `tests/unit/features/<slice>/` (api → `api/`,
   use cases → `application/`, pure domain → `domain/`); a test spanning
   layers lives at the highest layer it exercises.
9. Gates: `uv run ruff check src/ tests/`, `uv run ty check src/`,
   `uv run lint-imports`, `uv run pytest tests/unit -q`.
