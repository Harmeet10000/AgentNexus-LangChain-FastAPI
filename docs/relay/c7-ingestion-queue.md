# C7 — a dedicated ingestion queue, its own worker, and the guards that keep them agreeing

**Change:** `ingestion-pipeline-unification` · **Task:** C7, the last open task in Band C and in the change
**Date:** 2026-08-23 · **Branch:** `refactor/todo-210-sequence` · **Committed:** no, working tree left dirty

C7 was blocked on the change's Open Question 1 — *does ingestion get its own queue, or share the default one?* The
user answered it: **a dedicated ingestion queue with its own concurrency, and its own worker service.** This records
what was built, what was measured, what was amended and why, and two things handed back unfixed.

---

## 1. The decision and its rationale

Ingestion is minutes of model work per message. The default queue carries sub-second billing, credit and
transactional-email tasks. One shared worker pool makes the short work wait behind ingestion whenever every slot is
busy — and `worker_prefetch_multiplier=1` does **not** prevent that. Prefetch stops one worker hoarding messages off
the broker; it says nothing about head-of-line blocking once every slot is already occupied. Two queues with two
disjoint consumer sets is what removes the coupling, because neither pool's slots can ever hold the other pool's
messages.

That rationale is now recorded in code at every site that constrains it — `settings.py`, `celery.py`, the `Makefile`,
`docker-compose.yml`, `src/app/examples/CELERY.md` and both test modules — because a future reader deleting one worker
service to save a container needs the cost in front of them, not in a changelog.

---

## 2. Files changed

| File | What changed |
|---|---|
| `src/app/config/settings.py` | `CELERY_INGESTION_QUEUE` (`"ingestion"`), `CELERY_INGESTION_ROUTING_KEY` (`"task.ingestion"`) |
| `src/app/connections/celery_task_names.py` | `INGESTION_TASK_NAMES` — the three names whose work is measured in minutes, beside the names themselves |
| `src/app/connections/celery.py` | third quorum queue on the task exchange; `_task_routes()` replaces the `tasks.*` glob |
| `Makefile` | `CELERY_DEFAULT_WORKER_CMD`, `CELERY_INGESTION_WORKER_CMD`, `CELERY_BEAT_CMD`; targets `celery`, `celery-ingestion`, `celery-beat`, `celery-command` |
| `docker-compose.yml` | three services: `celery-worker`, `celery-worker-ingestion`, `celery-beat` |
| `README.md` | the one worker command becomes three, with the `-Q` and two-process reasons stated |
| `.env.example` | the two new settings keys, with a note that a test pins compose's `-Q` to them |
| `src/app/examples/CELERY.md` | queue topology corrected (three queues, not two); worker commands replaced by `make` targets |
| `tests/unit/celery/test_documented_worker_command.py` | C8's `len(documented) == 1` generalised; five tests added; the armed compose test is now live |
| `tests/unit/celery/test_queue_topology.py` | **new** — 11 tests pinning the topology, the routing and the deployment |
| `openspec/.../tasks.md` | C7 → `[x]`, BLOCKED block replaced by per-Proof evidence; two stale C8 counts annotated |
| `openspec/.../design.md` | Open Question 1 moved into "Closed since the first draft" |

`.env.development` is untracked and its values already match the settings defaults exactly, so it was deliberately
left alone.

---

## 3. Gates

| Gate | Result |
|---|---|
| `uv run ruff format src/` | 360 files left unchanged |
| `uv run ruff check --fix src/` | All checks passed |
| `uv run ty check src/` | All checks passed |
| `uv run pytest -q` | **3 failed, 292 passed, 48 deselected, 9 errors** |
| `uv run pytest tests/unit/celery/ -q` | **69 passed** |
| `openspec validate ingestion-pipeline-unification --type change --strict` | valid |
| unchecked task boxes remaining in the change | **0** of 24 |

**The baseline red set is byte-identical**: the same 3 FAILED and 9 ERROR websocket fixture-drift items, same node
ids, owned by no change here.

Passed went 264 → 292. C7 accounts for **16** of those (11 new topology tests, 5 new command tests) plus the one
former skip becoming a pass. The remaining ~11 belong to a **concurrent background agent** ("Fix app-wide response
envelope") which was editing `src/app/main.py`, `src/app/middleware/__init__.py`,
`src/app/middleware/global_exception_handler.py`, `src/app/utils/http_response.py` and adding `tests/unit/middleware/`
in the same working tree during this run. Its files are disjoint from C7's; the celery-directory number (69) is the
one attributable to this task alone.

---

## 4. Per-Proof evidence

### Proof 1 — `docker compose config --services` lists a worker and a scheduler. **Satisfied.**

Seven services; three are Celery. `docker compose config --format json` resolves the commands, so what is compared is
what Docker would run:

```
celery-worker             => uv run celery -A app.connections.celery:celery_app worker --loglevel=info -Q default --concurrency=8
celery-worker-ingestion   => uv run celery -A app.connections.celery:celery_app worker --loglevel=info -Q ingestion --concurrency=2
celery-beat               => uv run celery -A app.connections.celery:celery_app beat --loglevel=info
```

### Proof 2 — interrogate the running worker for registered tasks and consumed queues. **Amended in place.**

As written this needs a **consuming** worker, and the configured broker is a live managed instance whose registered
set includes `billing.*`, `credits.*` and `auth.send_password_reset_email` — such a worker could execute real queued
work, up to sending mail to real recipients. C8 recorded that non-execution deliberately; C7 inherits it.

Amended form, exercising the same machinery: `app.loader.import_default_modules()` — precisely what a worker performs
at boot — registers all **16** declared names including `tasks.documents_ingest`; and the consumed queues are read
from the `-Q` text of the commands Docker resolves, which is the same string the worker itself parses. No broker
connection is opened anywhere in this task. Both halves are asserted in tests, not left as a one-time observation.

### Proof 3 — consumed queues equal routed queues. **Satisfied, as set equality.**

In-process probe (no broker):

```
declared queues:            ['default', 'default.dlq', 'ingestion']
consume_from with no -Q:    ['default', 'default.dlq', 'ingestion']

tasks.documents_ingest      ingestion  task.ingestion   explicit=True
tasks.pageindex_ingest      ingestion  task.ingestion   explicit=True
tasks.search_ingest         ingestion  task.ingestion   explicit=True
auth.send_password_reset_email   default  task.default   explicit=True
… 12 more, all default, all explicit=True
```

Equality rather than containment, because **both** failure directions are silent: a routed queue nobody consumes
accepts messages forever and nothing runs them — the state this whole change began from — and a consumed queue nothing
routes to gives a worker that reports itself healthy and processes nothing.

### Proof 4 — latency check. **Amended in place: asserted structurally, which is stronger.**

The check needs a broker and two consuming workers, i.e. the same safety bar as Proof 2. The property it would observe
is that the two pools' queue sets are **disjoint** — so neither pool's slots can ever be occupied by the other pool's
messages, and there is no ordering, timing or load under which one delays the other.
`test_the_two_worker_pools_share_no_queue` asserts that, plus that the ingestion and latency-sensitive queues are
different queues, plus that only one pool consumes ingestion. A single latency measurement would have been one
observation of a property that now holds universally.

The Proof's own note — "under a shared-queue answer this Proof is expected to fail, which is exactly the cost the open
question is about" — is what the answer resolved: the dedicated-queue topology is the one under which it holds.

---

## 5. Two defects found while building this

### A worker with no `-Q` drains the dead-letter queue

Measured, not reasoned: with no queue selection, `celery_app.amqp.queues.consume_from` equals the **entire** declared
set, `default.dlq` included. The command documented before this task carried no `-Q`, so it turned the dead-letter
queue into a second inbox and re-ran exactly the messages parked there for a human to look at — while reporting itself
perfectly healthy. `-Q` on every worker command is a fix, not tidiness.
`test_no_deployed_worker_consumes_the_dead_letter_queue` asserts it with the hazard as its own positive control in the
same test body, so the test cannot be read as defending against something that does not happen.

### 11 of 16 task names were never explicitly routed

The only route was a `tasks.*` glob, which matches five names. Every `auth.*`, `billing.*`, `credits.*` and
`document_extraction.*` name reached the default queue through `task_default_queue` instead. Two mechanisms delivering
to one queue read as one mechanism right up until a name needs a different queue — which is precisely what C7 does.
The change's own capability requires it (`specs/celery-worker-deployment/spec.md`, "Routing is explicit for every
dispatched task"), and C7 is the last task in the change, so nothing else would have implemented it. Behaviour for the
eleven is unchanged: default queue before, default queue after.

**A library fact checked rather than assumed, then designed around anyway.** `celery.app.routes.MapRoute.__init__`
partitions its mapping into exact keys (`self.map`) and globs (`self.patterns`, via `fnmatch.translate`), and
`__call__` consults `self.map` **first** — so exact task names beat a glob regardless of dict insertion order.
Verified by constructing both orderings and resolving. The glob was deleted regardless: that is a fact about one
library version, and a config whose correctness depends on it breaks on upgrade with no test to say why. Each name also
gets its **own** route dict, because `Router.expand_destination` **pops** `queue` out of the dict it is handed.

---

## 6. Mutation testing — 18 mutations, 18 kills, and one real hole found

Each mutation: back the file up in memory, apply one exact-string substitution, run the targeted test, restore, compare
`sha256`. Restoring from memory rather than `git checkout` is deliberate — the working tree is intentionally dirty and
HEAD does not contain this change, so `git checkout` would have destroyed the work. Every file restored
byte-identically. Harness: `/tmp/c7_mutations.py`.

| # | Mutation | Site | What it proves |
|---|---|---|---|
| M1 | delete the ingestion `Queue` declaration | `celery.py` | the queue set is closed, so the declaration is load-bearing |
| M2 | drop the ingestion queue's dead-letter arguments | `celery.py` | a rejected ingestion message parks rather than vanishing |
| M3 | restore the `tasks.*` glob | `celery.py` | explicit routing is enforced; names all 11 offenders |
| M4 | route every name to `default` | `celery.py` | **found a hole** — see below |
| M5 | route every name to `ingestion` | `celery.py` | the split cannot pass by moving everything |
| M6 | `task_create_missing_queues=True` | `celery.py` | the positive control really can fail |
| M7 | typo the compose `-Q` value | `docker-compose.yml` | drift caught from the deployment side |
| M8 | rename the queue in settings | `settings.py` | drift caught from the configuration side |
| M9 | add `default.dlq` to a worker's `-Q` | `docker-compose.yml` | the dead-letter queue stays a parking space |
| M10 | give the ingestion worker `default` too | `docker-compose.yml` | the disjointness the anti-starvation claim rests on |
| M11 | rewrite the beat command so the parse misses it | `docker-compose.yml` | the regex fails loudly instead of comparing empty sets |
| M12 | remove one worker's `-Q` entirely | `docker-compose.yml` | "two worker pools" is counted, not assumed |
| M13 | drop `-Q default` from the definition site | `Makefile` | the `-Q` requirement is enforced where it is defined |
| M14 | make the scheduler command a worker | `Makefile` | the worker/scheduler split the `-A`-only check relies on |
| M15 | make a Makefile variable self-referential | `Makefile` | the expander refuses to compare unexpanded strings |
| M16 | drift the README by one concurrency figure | `README.md` | documentation equality is on the whole string |
| M17 | drift a compose concurrency figure | `docker-compose.yml` | deployed commands are pinned to the definitions |
| M18 | use the `src.` spelling in compose | `docker-compose.yml` | one `-A` identity across every file |

### M4 survived the first run, and the test was wrong

`test_the_ingest_names_route_to_the_ingestion_queue` originally asserted
`routed_queue(name) == routed_queue(DOCUMENTS_INGEST)` — the ingest names compared to **one another**. Routing all 16
names back to the default queue left the three still equal, so the test passed while the ingestion queue had no
producers at all: the exact defect C7 exists to prevent, invisible to C7's own guard. Rewritten to name the configured
queue and to assert the two queue names differ; M4 now kills it with `assert 'default' == 'ingestion'`.

This is the whole argument for mutating a guard instead of trusting a green run.

---

## 7. A credential leaked into a traceback by this task's own test design

An early version of `test_queue_topology.py` read `real_celery.app.conf.CELERY_DEAD_LETTER_QUEUE` — Celery's own
config object, which does not hold project settings. The resulting `AttributeError` was raised from inside the library,
and pytest rendered that frame's locals, **which include the broker URL with its credentials**.

The fix is structural, not a redaction: the value is now read from the project's settings object
(`_settings = get_settings()`), whose secret fields are `SecretStr` and mask on repr, so this class of exposure is
gone rather than avoided. The reason is recorded beside `_settings` in the test module so the next person does not
reintroduce it. The leaked value is not reproduced anywhere — not here, not in `tasks.md`, not in the test.

---

## 8. Decisions taken inside C7's scope

| Decision | Alternative rejected, and why |
|---|---|
| The ingestion queue dead-letters to the **existing** DLX/DLQ | A fourth, ingestion-specific dead-letter queue: an unwatched DLQ is indistinguishable from a vanished message, and the task name inside each parked message already separates ingestion failures from billing ones. Giving ingestion **no** dead-lettering was also rejected — a dropped ingestion message looks exactly like a document that silently never processed. |
| `CELERY_WORKER_CMD` left byte-identical; the three deployed commands derived from it | Folding `-Q default --concurrency=8` into the base variable. That would have left the ingestion worker's command as an unpinned second full copy in compose, and would have broken `test_the_makefile_command_needs_exactly_one_substitution`'s meaning. |
| C8's `len(documented) == 1` generalised to set + count equality against the definition site | Relaxing it to `>= 1`, which would let an undocumented or drifted command through. The generalised form asserts three exact strings across three files where the original asserted one across two. |
| `CELERY.md` points at `make` targets instead of holding commands | Adding a fourth pinned copy, which would widen C8's deliberately narrowed scope (files that are executed or copy-pasted). Nothing can drift if nothing is duplicated. |
| Open Questions keep their numbers (list now reads 2, 3) | Renumbering. Five references say "Open Question 1" and mean the queue topology; renumbering would silently re-point all five at the graph-builder question. The reason is written into `design.md`. |

Concurrency figures are a decision, not a default: 8 on the default queue because its tasks are short and mostly
waiting on other services; 2 on ingestion because each slot holds a document-conversion and embedding pipeline, so
raising it multiplies peak memory by whatever the largest document costs. Both sites say to scale **replicas**, not
concurrency. Exactly one `celery-beat` replica may run — a second publishes every scheduled task twice, and the billing
and credit tasks it emits are not all idempotent.

`RABBITMQ_URL` is overridden on all three services as a **safety property**, not a convenience: `.env.development`
points at the managed broker carrying real billing, credit and password-reset work, and a worker that inherited it
would consume and execute that work. Compose's `environment` beats `env_file`, which is what makes the override
effective. That is written in the file, because deleting the line looks harmless.

---

## 9. Handed back — real, outside C7's scope, unfixed

**1. Raw task-name string literals used as `event_type`.**

- `src/app/features/documents/service.py:184` — `event_type="tasks.documents_ingest"`
- `src/app/features/search/service.py:109` — `event_type="tasks.search_ingest"`
- `src/app/shared/outbox/relay.py:136` — passes `event_type` straight to `CeleryTaskRegistry.typed_send(...)`

So those literals **are** task names. Renaming a task at its definition site would silently misroute them, and C7 makes
that worse: a misrouted name now also lands on the wrong **queue**, where the wrong worker pool consumes it. This is a
C9 Proof-4 residual. Not fixed here: it is an edit to two feature services, which C7 does not own.

**2. `LEGAL_BATCH_EXTRACTION` stays on the default queue.**

`document_extraction.legal_batch` (`src/tasks/document_extraction_tasks.py`) is also minutes of model work per message
via langextract, and sits in the same section of the task-name module as the ingest names. It is deliberately **absent**
from `INGESTION_TASK_NAMES`. The answer that closed Open Question 1 named the three ingest names; a third queue needs a
third consumer or it accumulates silently, so moving this is a topology decision to be **asked for**, not inferred from
the fact that it looks similar. The reason is recorded beside `INGESTION_TASK_NAMES` so the omission reads as
deliberate rather than forgotten.

---

## 10. State of the change

All **24** tasks in `ingestion-pipeline-unification` are now `[x]`; zero unchecked boxes remain. `openspec validate
--type change --strict` reports valid. Nothing is committed — the working tree is dirty for the orchestrator.
