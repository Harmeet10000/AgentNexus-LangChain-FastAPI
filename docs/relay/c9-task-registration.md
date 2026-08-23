# C9 — Explicit task registration, typed dispatch, single task-name definition

Change: `ingestion-pipeline-unification`. Working tree left dirty, nothing committed.

Task application module settled on: **`app.connections.celery`**, exposing **`celery_app`**.
Worker command: `uv run celery -A app.connections.celery:celery_app worker` (bare
`-A app.connections.celery` also resolves, since Celery probes for a `celery_app`
attribute). See "For C8" below — the currently documented command names a module that
does not exist.

## What changed

**New — `src/app/connections/celery_task_names.py`.** Single definition site for all 16
dispatchable task names, plus `TASK_DECLARING_MODULES` (name → declaring module). The
mapping has two jobs: the dispatch helper imports the one module that declares the name
it was handed, and a unit test asserts the mapping and the application's `include` list
agree in both directions.

**`src/app/connections/celery.py`.** `include=` went from 5 entries (2 of them the only
listed ingestion path) to 8 explicit literals — every module that declares a task, sorted.
`beat_schedule` now references the name constants. The docstring records why
`tasks.auth_email_tasks_typed` is deliberately absent: it declares the same two task names
as the live email module, so listing both would let import order pick the winner.

**`src/app/connections/celery_registry.py`.** Harvested the archived typed-registry
contract with one deliberate tightening and one repair:

- tightening: `validate()` now raises `UnregisteredTaskError` for a name with no registered
  model. The archived text substituted a permissive model, logged a warning, and sent
  anyway — so a typo or a half-finished rename produced a well-formed message addressed to
  nobody, which Celery discards in silence.
- repair: `ensure_declared_module_imported()`. Registration is a side effect of importing
  the declaring module, and **nothing under `src/` imports the task package at all** — so
  the API process that runs the outbox relay held an *empty* registry and every payload was
  validated against the permissive fallback. The harvested contract validated nothing in
  the only process that used it. Verified, not assumed (`rg` over `src/` finds one hit, in
  a Markdown example).
- `TaskDispatchError(CeleryError)` is load-bearing: `OutboxRelay._publish` catches
  `(CeleryError, PostgresError)` to mark an event failed and retry it toward the
  dead-letter table. A bare Pydantic `ValidationError` escapes that catch into the relay's
  outer blanket handler, which logs a warning and drops the event — putting the
  invisibility back one layer up. The original `ValidationError` is preserved as
  `__cause__` and on `.validation_error`.
- deleted `LegacyTaskPayload` (the permissive fallback); added `NoKwargsPayload` for
  scheduler-dispatched jobs, which states "nothing, and nothing extra" as a contract rather
  than leaving those names unregistered.

`OutboxRelay._publish` already calls `CeleryTaskRegistry.typed_send`, so the helper proven
here is the live dispatch path — no wiring change was needed on the relay side.

**`src/tasks/*`.** Every `@celery_app.task(name=...)` now takes its name from the
definition module; every declared name registers a payload model. Leaf imports throughout
(`from app.connections.celery import ...`), per commit `319c698`. Payload models added for
document, search, pageindex, legal-batch extraction, both example tasks, both auth emails,
and all eight scheduled billing/credit jobs. `src/tasks/__init__.py` was deliberately not
modified — it is the file Proof 2 mutates and restores.

**New tests** — `tests/unit/celery/test_task_registration.py` (7 test functions, 9 items
after parametrisation), `tests/unit/celery/test_typed_dispatch.py` (8 functions, 9 items),
`tests/unit/celery/conftest.py` (the `real_celery` fixture). 18 items total, all passing.

### Two live defects found and fixed on the way

1. `src/tasks/document_extraction_tasks.py` declared `bind=True` while its body takes no
   `self`. Celery would have passed the `Task` instance as the first positional argument
   (`urls`). Removed, with a comment recording why.
2. `tests/unit/celery/conftest.py` documents the third: another unit test replaces
   `sys.modules["app.utils"]` at import time with a two-attribute proxy whose logger is an
   `AsyncMock`, and never restores it. Any module imported into that state binds a logger
   whose `.bind()` returns a coroutine, so the first diagnostic a refusal writes raises
   `AttributeError` instead. See "Test isolation" below.

## Test isolation (why there is a new conftest)

`tests/conftest.py` puts `MagicMock()` into `sys.modules` for `app.connections.celery`,
`tasks`, `tasks.auth_email_tasks`, and `tasks.search_tasks`. A `MagicMock` has no
`__path__`, so under those entries the declaring modules are not merely mocked — they are
unimportable, and the machinery reports `'tasks' is not a package`. Every C9 proof is
unwritable while they stand. This is the same situation that conftest's own comment
records for `app.shared.langgraph_layer`.

Rather than edit a file three concurrent tasks share, `tests/unit/celery/conftest.py`
provides an opt-in, module-scoped `real_celery` fixture that lifts those entries (plus
`app.connections.celery_registry`, which closes over the application, and the top level of
`app.utils`, for the reason above), imports the real modules, and restores the originals at
teardown. It is not autouse: the sibling module in that directory that wants the mocks
keeps them. Verified non-leaky — the pre-existing red set is byte-identical before and
after, and 7 mutation runs each restored it exactly.

**Recommendation for whoever owns `tests/conftest.py`:** the three `tasks*` stubs are dead.
`rg -o 'from tasks(\.[\w.]+)? import' src/` — the command that conftest's own comment says
regenerates the list — now returns nothing. Removing them would let the `real_celery`
fixture shrink to nothing. Not done here: not my file.

## Proofs

### Proof 1 — every dispatched task module listed explicitly

`uv run rg -n "include=|imports\s*=" src/app/connections/celery*.py src/tasks/`

```
src/app/connections/celery.py:222:        include=[
```

**The Proof as written does not show what it claims to check** — `rg` prints only the
matching line, so the list itself is invisible, and `imports` (Celery's other module
setting) is unused anywhere, so that half of the alternation never matches. Amended form
(`rg -n -A 10 "include="`) returns the evidence:

```
src/app/connections/celery.py:222:        include=[
src/app/connections/celery.py-223-            "tasks.auth_email_tasks",
src/app/connections/celery.py-224-            "tasks.billing_tasks",
src/app/connections/celery.py-225-            "tasks.credit_tasks",
src/app/connections/celery.py-226-            "tasks.document_extraction_tasks",
src/app/connections/celery.py-227-            "tasks.document_tasks",
src/app/connections/celery.py-228-            "tasks.example",
src/app/connections/celery.py-229-            "tasks.pageindex_tasks",
src/app/connections/celery.py-230-            "tasks.search_tasks",
src/app/connections/celery.py-231-        ],
```

Eight modules; every module under `src/tasks/` that declares a task except
`auth_email_tasks_typed`, which duplicates two live names and is excluded on purpose.

### Proof 2 — registration survives a tidy of the package initialiser

Destructive-then-restore, subprocess interrogation, no worker and no docker. Also satisfies
the "Dependency (change 0)" note: the proof was run *after* the initialiser was reduced to
what change 0's tidy would leave.

`src/tasks/__init__.py` was replaced with its docstring alone (all four re-export imports
and `__all__` removed), then:

```
initialiser imported anything?: False
include list: ["tasks.auth_email_tasks", "tasks.billing_tasks", "tasks.credit_tasks",
 "tasks.document_extraction_tasks", "tasks.document_tasks", "tasks.example",
 "tasks.pageindex_tasks", "tasks.search_tasks"]
  dispatched name registered: tasks.documents_ingest -> True
  dispatched name registered: tasks.search_ingest -> True
  dispatched name registered: tasks.pageindex_ingest -> True
declared names missing from the application: []
RESULT: ALL DECLARED NAMES REGISTERED
```

Restore verified:

```
before sha256: 044e71e4fd760f86443ac464555b5d98dfbe9c2f4b352d655ad98a66a522b98e
after  sha256: 044e71e4fd760f86443ac464555b5d98dfbe9c2f4b352d655ad98a66a522b98e
RESTORE: byte-for-byte identical
git diff for the file: []
git status --porcelain for the file: []
```

### Proof 3 — dispatch validation, helper invoked directly

No outbox, no relay, no broker, no durable event: the helper is called in-process and the
send is replaced with a spy, so a leaked dispatch is visible rather than attempted.
`tests/unit/celery/test_typed_dispatch.py`, 9 items, all passing. Unregistered name →
`UnregisteredTaskError` naming the task, spy empty. Registered name with a missing field,
with an extra field, and with an empty payload → `TaskPayloadValidationError` naming the
task, spy empty. A matching payload reaches the send unchanged.

### Proof 4 — no task-name literal outside the definition module

`uv run rg -n '"tasks\.' src/app/ src/tasks/` returns 31 hits. **The Proof's stated
expectation cannot be met by that pattern, and the pattern is also blind to most of the
names it is meant to police**: only 5 of the 16 declared task names begin with `tasks.`.
The other 11 (`auth.*`, `billing.*`, `credits.*`, `document_extraction.*`) are invisible to
it — including two live dispatch-side literals. Full classification of the 31:

| Hits | Where | Classification |
|---|---|---|
| 8 | `celery.py:223-230` | module paths in the `include` list — required by Proof 1 |
| 1 | `celery.py:289` | `"tasks.*"` — a routing glob, not a task name |
| 5 | `celery_task_names.py:46-48,72-73` | **the single definition site** (the other 11 names do not start with `tasks.`) |
| 8 | `celery_task_names.py:80-87` | private module-path constants, same file |
| 6 | `src/app/examples/CELERY.md` | documentation examples, not executable code |
| 1 | `settings.py:177` | `"tasks.dlx"` — a dead-letter *exchange* name, not a task name |
| 2 | `documents/service.py:184`, `search/service.py:109` | **genuine residual dispatch-side literals** |

**Replacement Proof, name-agnostic.** Build the alternation from the definition module
itself, so no name can hide from it:

```python
alt = "|".join(re.escape(n) for n in sorted(TASK_DECLARING_MODULES))
rg -n f'"({alt})"' src/ -g '!*.md' -g '!celery_task_names.py'
```

15 hits, in two groups:

```
src/app/features/documents/service.py:184:            event_type="tasks.documents_ingest",
src/app/features/search/service.py:109:            event_type="tasks.search_ingest",
src/app/features/auth/service.py:271:            event_type="auth.send_verification_email",
src/app/features/auth/service.py:298:            event_type="auth.send_password_reset_email",
```

...four genuine dispatch-side literals, and eleven `logger.bind(operation="...")` labels in
`src/tasks/billing_tasks.py` and `src/tasks/credit_tasks.py`.

The eleven log labels were **deliberately left as literals.** They are an observability
dimension, not a wire name: the same files also bind `operation="billing.invoice_backfill"`
and `"billing.receipt_backfill"`, which are not task names at all, so the taxonomy is
independent of the task registry and only coincides with it in places. Coupling it to the
task-name constants would make a task rename silently re-label existing dashboards and
alerts — the opposite of what stability means for a log field.

The four dispatch sites are **not** C9's files, so they were left untouched. Each needs one
import and one substitution:

- `src/app/features/documents/service.py:184` — `event_type=DOCUMENTS_INGEST`
- `src/app/features/search/service.py:109` — `event_type=SEARCH_INGEST`
- `src/app/features/auth/service.py:271` — `event_type=SEND_VERIFICATION_EMAIL`
- `src/app/features/auth/service.py:298` — `event_type=SEND_PASSWORD_RESET_EMAIL`

all from `app.connections.celery_task_names`. The documents and search payloads were
checked against the models registered for those names and match, so those two are name-only
and behaviour-neutral. The two auth payloads do **not** match — see finding 2.

### Proof 5 — declared but unimplemented task

`tasks.pageindex_ingest` is bound on the application (Proof 2's subprocess lists it) and
`.run()` raises `NotImplementedError`, not an unknown-task error:
`test_declared_but_unimplemented_task_is_registered_and_fails_explicitly`. Mutations M6
and M7 both kill it (see below).

## Mutation results

Each guard reverted in turn, full suite run, file restored and its sha256 re-checked.
Baseline red set = the 12 pre-existing websocket fixture-drift items. Every mutation killed
exactly the intended tests; **no mutation turned a pre-existing red item green**, and the
final re-check confirmed the red set was byte-identical to baseline.

| Mutation | Newly red | Tests killed |
|---|---|---|
| M1 `include` list drops `tasks.document_tasks` | 3 | `test_every_declaring_module_is_named_in_the_include_list`, `test_dispatched_task_modules_are_listed_explicitly[tasks.documents_ingest]`, `test_the_ingestion_module_is_listed_rather_than_reached_through_the_initialiser` |
| M2 unregistered name falls through permissively | 2 | `test_unregistered_name_is_reported_as_a_failure_naming_the_task`, `test_a_name_with_no_declaring_module_is_refused_rather_than_searched_for` |
| M3 payload mismatch warns and passes through | 4 | `test_registered_name_with_a_missing_field_is_refused_at_dispatch`, `test_registered_name_with_an_unexpected_field_is_refused_at_dispatch`, `test_the_original_validation_detail_is_preserved`, `test_the_helper_imports_the_declaring_module_before_deciding` |
| M4 helper stops importing the declaring module | 5 | the four M3 tests plus `test_a_matching_payload_still_reaches_the_send` — with no import the registry is empty, so every name looks unregistered: this is the production defect reproduced |
| M5 refusals stop deriving from `CeleryError` | 2 | `test_both_refusals_are_celery_errors[UnregisteredTaskError]`, `[TaskPayloadValidationError]` |
| M6 declaration drifts from the definition (literal name in the decorator) | 2 | `test_every_declared_task_name_is_bound_on_the_task_application`, `test_declared_but_unimplemented_task_is_registered_and_fails_explicitly` |
| M7 unimplemented task returns instead of raising | 1 | `test_declared_but_unimplemented_task_is_registered_and_fails_explicitly` |

M2 and M3 are the two the task asked for by name: both refusals fail loudly. Under M2 the
dispatch reaches the spy (proving the pre-tightening behaviour was a silent send); under M3
it reaches the spy with a payload the consumer cannot accept.

## Gates

| Gate | Result |
|---|---|
| `uv run ruff format src/` | 360 files left unchanged |
| `uv run ruff check --fix src/` | All checks passed! |
| `uv run ty check src/` | All checks passed! |
| `uv run ruff format` (the 3 new test files only) | 3 files left unchanged |
| `uv run ruff check` (the 3 new test files only) | All checks passed! |
| `uv run pytest -q` | `3 failed, 256 passed, 48 deselected, 9 errors` |

Baseline at dispatch was `194 passed, 3 failed, 9 errors`. The 3 failures and 9 errors are
the same pre-existing websocket fixture-drift items, unchanged. The pass count rose by 62;
18 of those are C9's, the rest arrived from concurrent tasks during the window.

Two `# noqa` in the new code, both following existing repo precedent rather than inventing
it: `S105` on the password-reset *task name* constant (same as `src/app/utils/codes.py:20`
and `src/app/features/auth/dependencies.py:19`), and nothing else. No `# type: ignore`.

## Findings the task did not describe

1. **The documented worker command cannot start a worker.** `Makefile:52` and
   `README.md:279` both say `uv run celery -A celery_config worker --loglevel=info`, and
   there is no `celery_config` module anywhere in the repo (`find` over the tree excluding
   `.venv`/`.git` returns nothing; `src/` holds `alembic app database lynk mcp_core tasks`).
   The correct target is `app.connections.celery:celery_app`. Both files belong to C8.
2. **Producer/consumer gap on the auth emails.** `auth/service.py:271` and `:298` emit
   `{user_id, email, token}`; both task bodies require `idempotency_key` as well. The
   payload models were registered faithful to the *declarations*, so this gap is now a loud
   dispatch refusal instead of a `None` lock key in the worker. No live behaviour change
   today — the outbox tables do not exist — but this will fire the moment change 0 creates
   them, and it is a real bug either way. Fix belongs to whoever owns `auth/service.py`.
3. **`tasks.auth_email_tasks_typed` declares the same two task names as the live email
   module.** Kept out of `include` and documented in place, but it is a deletion candidate:
   two modules declaring one name means the winner is whichever imports last.
4. **`document_extraction.legal_batch` and `tasks.pageindex_ingest` have zero dispatchers.**
   Both are registered and bound, so a future dispatch gets a real diagnostic, but nothing
   dispatches either today.
5. **The `tasks*` stubs in `tests/conftest.py` are dead** — see "Test isolation".
6. **Dispatch-message path inaccuracy.** `openspec/changes/decisions.md` and
   `openspec/changes/critical-path-210.md` do not exist; both live under `docs/relay/`.
7. **"Decision 16" is ambiguous.** `docs/relay/decisions.md`'s Decision 16 is about
   `UnifiedChunk` gaining `updated_at`. The Decision 16 that C9's text invokes — the
   harvest-vs-delta ruling — is inside `design.md` (~line 592, "the same situation as the
   node-failure-pattern capability in Decision 16"). Two different numbering schemes share
   one label.

## Proofs whose stated expectation needs amending in `tasks.md`

- **Proof 1** — as written it prints only the `include=` line, so it does not display the
  list it is meant to verify, and its `imports\s*=` alternation matches nothing (that Celery
  setting is unused here). Suggest `rg -n -A 10 "include=" src/app/connections/celery*.py`.
- **Proof 4** — "no task-name string literal anywhere else" cannot come back clean from
  `rg -n '"tasks\.'`: the pattern also matches the eight `include` module paths that Proof 1
  requires, the `"tasks.*"` routing glob, the `"tasks.dlx"` exchange name, and six
  Markdown examples. Worse, it is blind to 11 of the 16 task names — the `auth.*`,
  `billing.*`, `credits.*` and `document_extraction.*` families — so it misses two of the
  four live dispatch-side literals entirely. Replace it with the name-agnostic form above,
  which builds its alternation from the definition module and therefore cannot go stale.
