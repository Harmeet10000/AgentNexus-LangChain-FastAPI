> Change class: **L**

## Context

See `proposal.md` — Why for motivation and the measured defect counts. This
section records only what constrains the approach.

**Four generations of the same rule are live simultaneously.** Every one of them
is written down somewhere authoritative:

| Artifact | What it says | Generation |
|---|---|---|
| `openspec/specs/pattern-matching-standard/spec.md` | `match`/`case` on `Success`/`Failure` SHALL be used; `isinstance` SHALL NOT | oldest, never implemented |
| `.kiro/steering/RESULT-PATTERN.md` | `isinstance` + `raise app_error_to_exception(error)` | superseded; the raise is gated as retired |
| `.opencode/instructions/RESULT-PATTERN.md`, `docs-site/…/error-and-result-pattern.mdx`, `.ast-grep/rules/` | `isinstance` + `http_error()` | current, and what the code does |
| this change | `isinstance` to open, `match` on the closed error union, render with a real status | target |

The code is unanimous — 122 `isinstance(result, Failure)` sites, zero
match-on-`Result` — so no generation of the *code* is in dispute. Only the
governance is.

**Constraints the design must respect:**

- **`ty` does not narrow through `case Success(value)`.** Measured on this
  repository: the bound name takes the union of the success and error types. Any
  design built on matching the container is unchecked in precisely the place the
  pattern was adopted for safety.
- **`ty` does verify exhaustiveness over a closed union** closed by
  `assert_never`, reporting `type-assertion-failure` and naming the uncovered
  member. This is the only static guarantee available, and the design is built to
  earn it.
- **`ty` cannot see a shadowed `match` arm.** A broader class pattern before a
  narrower one makes the narrower unreachable and the match still type-checks as
  exhaustive. The repo already manages this by hand: the public docs describe
  `mappers.py` as ordering arms "most-specific first" with a "catch-all
  `AppError()` at the end". That is a maintenance obligation carried in prose.
- **`http_error()` does not set a transport status.** It writes the status into
  the body and returns a plain object, so returning it from a route yields HTTP
  200 with `"success": false`.
- **The request-scoped session commits on clean exit and rolls back only on an
  escaping exception.** A service that swallows a `Failure` therefore reaches
  `commit()` on a poisoned session. No repository rolls back; `session.rollback`
  has never appeared under `src/app/features/`.
- **Baseline is not green.** Recorded below so a reviewer can distinguish a
  regression from an inheritance.

| Gate | Baseline | Owned by this change? |
|---|---|---|
| `uv run ruff check src/` | clean | — |
| `uv run ty check src/` | 2 errors (`…agents.memory.setup_types` unresolved) | no |
| `ast-grep scan src/` | 4 errors, 34 warnings | yes — the 34 are `no-raise-app-error-mapper`; the 4 are `no-raw-httpexception` in `examples/redis_examples.py` |
| `uv run pytest` | 400 of 439 collected, 2 collection errors, 12 known websocket fixture failures | no |

## Goals / Non-Goals

**Goals (design-level):**

- Make the `"DB_ERROR"` class of defect *unrepresentable*, not merely forbidden.
  A rule a reviewer must remember has already failed 56 times.
- Buy one static guarantee — exhaustiveness over each feature's failure modes —
  and keep the mechanism that buys it small enough to hold in mind.
- Keep boundary code a fixed-width dispatch. Adding a feature error type must not
  require touching any boundary.
- Leave every exception-native layer alone, but leave none of them undefined.

**Non-Goals (design-level):**

- A universal error base shared across features. The shared vocabulary is
  deliberately seven enum members and nothing else.
- Eliminating exceptions. Raising is correct where control flow must be abandoned
  rather than answered.
- A generic classification helper. Which library exception maps to which typed
  error is feature knowledge; centralising it would recreate the open hierarchy
  in helper form.

## Decisions

### D1 — `isinstance` opens the container; `match` dispatches the union

**Chosen:** two different constructs for two different jobs.

```python
result = await self._repo.find(sub_id)
if isinstance(result, Failure):
    error = result.failure()          # narrowed: SubscriptionError
    match error:
        case SubscriptionNotFoundError():   ...
        case SubscriptionDuplicateError():  ...
        case SubscriptionDbError():         ...
        case _ as unreachable: assert_never(unreachable)
subscription = result.unwrap()        # narrowed: Subscription
```

**Alternatives considered:**

| Option | Why not |
|---|---|
| `match` on `Success`/`Failure` (the deployed spec's rule) | `ty` binds the union of both sides; the following code is unchecked. Also contradicts three other artifacts and 122 call sites. |
| `.map()` / `.bind()` combinator chains | Reads well for pipelines, but the failure classification this change exists to enable needs the error *in hand*, and combinator chains hide it. |
| `isinstance` chains on the error too | Works, but nothing forces completeness — which is the entire point. |

The apparent conflict between the user's design (built on `match`) and the
`no-match-on-result` gate dissolves here: the gate forbids matching the
*container*, and the design needs to match the *union*. Both are satisfied
without compromise.

### D2 — classification is a `ClassVar`, not a field

**Chosen:** `kind`, `code`, `retryable` as `typing.ClassVar` on the error type,
with `model_config = ConfigDict(extra="forbid", frozen=True)`.

Verified at runtime before committing to it: the constants are absent from
`model_fields`, absent from `model_dump()`, passing `code=` as a keyword raises
`ValidationError`, and mutation raises `ValidationError`.

**Alternatives considered:**

| Option | Why not |
|---|---|
| Field with a default (**today**) | Every construction site may override or invent a value. This produced 68 distinct codes against an 18-member enum. The type is shaped to permit exactly the bug. |
| `kind: Literal["not_found"]` per subclass (**also today**) | Present on `AppError` subclasses and read by nothing. Because the base declares no `kind`, it is not a discriminated union and buys no narrowing. |
| `@property` returning a constant | Works, but is overridable by a subclass and invisible to `extra="forbid"`. `ClassVar` makes the override site the only place it can change. |

`code` is typed as the feature's StrEnum rather than `str`, so a typo is an
assignment error at the declaration, not a string in a client's response body.

### D3 — one `<Feature>Code` StrEnum per feature, no global enum

**Chosen:** each `errors.py` declares its own enum. No feature imports another's.

The global 18-member `ErrorCode` is the artifact that failed: it was too small to
cover 18 features, so call sites invented literals instead of extending it.
Growing it to 68+ members would make it unreviewable and every feature a
stakeholder in every other feature's additions.

**Trade-off accepted:** the same logical condition may get two names in two
features (`SubscriptionCode.NOT_FOUND`, `InvoiceCode.NOT_FOUND`). This is the
cost of independence, and `ErrorKind` supplies the cross-feature vocabulary that
boundary code actually needs.

### D4 — `ErrorKind` is seven members, and the only shared error vocabulary

`VALIDATION | NOT_FOUND | CONFLICT | AUTHENTICATION | AUTHORIZATION |
INFRASTRUCTURE | EXTERNAL_SERVICE`.

**This is a deliberate deviation from the five-member set agreed at scoping, forced
by measurement.** `src/app/features/auth/service.py` raises
`UnauthorizedException` at **16 sites** — invalid credentials, invalid token,
expired token, session-owner mismatch. The locked scope converts every feature
service to Result-typed, so all 16 become typed failures rendered by
`render_result`. With five members the only reachable statuses are 422, 404, 409,
502 and 500/503: a failed login would be answered **422**, and a permission denial
would be indistinguishable from it. The superseded system handles this correctly —
`ErrorCode` carries `UNAUTHORIZED`, `FORBIDDEN`, `INVALID_TOKEN`, `TOKEN_EXPIRED`,
`REFRESH_TOKEN_INVALID` — so shipping five members would be a security-boundary
regression, discovered at the *last* scheduled feature after 16 changes of false
confidence.

**Alternatives considered:**

| Option | Why not |
|---|---|
| Keep five members; carve `auth` out of the Result conversion | Contradicts the locked scope, and makes the one security-critical feature the one feature with no exhaustiveness guarantee. |
| Keep five; let an auth error carry an explicit status | Contradicts D6 and reinstates per-error status drift — the thing being removed. |
| Keep five; map `AUTHENTICATION` onto `VALIDATION` | Answers a failed login with 422. Wrong for clients, wrong for monitoring, and wrong for anything counting auth failures. |

Two extra members cost nothing structurally: boundary dispatch stays fixed-width
at seven arms, independent of how many feature error types exist. They are kept
separate from each other because 401 and 403 are different client contracts —
"authenticate" versus "do not retry, you will never be allowed".

Boundary adapters dispatch on `kind`; feature logic dispatches on the concrete
type. This split is what keeps boundary code fixed-width: `STATUS_BY_KIND` has
seven entries whether the system has 40 error types or 400. Two error types
sharing a `kind` can still need opposite handling inside a feature, which is why
feature logic must not dispatch on `kind`.

### D5 — status derives from `kind`, refined by `retryable`

`VALIDATION` → 422, `NOT_FOUND` → 404, `CONFLICT` → 409, `AUTHENTICATION` → 401,
`AUTHORIZATION` → 403, `EXTERNAL_SERVICE` → 502, `INFRASTRUCTURE` → 503 if
`retryable` else 500.

This is the fix for the sharpest live defect. `InfrastructureAppError` defaults
`retryable=True`, so all 56 `"DB_ERROR"` sites currently answer **503** — telling
the client to retry a transaction that has already failed and, after this change,
been rolled back. A rolled-back write declares `retryable = False`.

### D6 — the renderer sets the transport status; endpoints cannot override it

`render_result(...)` accepts any feature's `Result`, sets `response.status_code`
from `STATUS_BY_KIND`, and emits the existing envelope. It takes a **success**
status and a success message; it takes **no** failure-status parameter.

The success-status parameter is named `success_status`, not `status_code`. The
sketch agreed at scoping used `status_code=`, which at a call site reads as
though it governs the failure — the opposite of what it does. At 67 endpoints
that ambiguity would be misread, and a misread would look like it worked.

**Alternative considered:** let endpoints pass a status per failure. Rejected —
that is how 67 endpoints acquire 67 opinions about what a conflict means, and the
per-endpoint drift is the thing being removed.

Endpoints keep returning the same envelope the global handler produces, so a
client cannot tell which produced a given response.

### D7 — no inheritance among concrete error types

Every concrete type inherits `FeatureError` directly and nothing else. Shared
fields are repeated, or promoted to `FeatureError` when genuinely universal.

This is the one rule with no static backstop, which is why it is gated rather
than documented. `ty` reports a shadowed match exhaustive; there is no runtime
symptom either. The failure mode is the wrong branch's side effects — a refund
issued for a validation error, a retry scheduled for a permanent fault.

**Cost accepted:** 28 concrete-inherits-concrete chains exist today (19 under
`APIException`, 9 outside). They are flattened per feature, in the change that
migrates that feature.

### D8 — rollback lives in the repository

**Alternatives considered:**

| Option | Why not |
|---|---|
| Roll back in the service | The service does not know whether a statement was issued, and a service that swallows the `Failure` is exactly the case that breaks today. |
| Roll back in `get_postgres_db` | Its `except` only fires when an exception escapes. A returned `Failure` is not an exception, so this path is unreachable for the defect. |
| Make services always raise so the dependency's rollback fires | Reinstates the retired raise pattern and makes correctness depend on never swallowing a failure. |

The repository catches the exception, so the repository is the only layer that
knows a rollback is owed. Order: classify → rollback → log → return, so a logging
failure cannot leave the session poisoned.

**The rule is per-store, not per-repository.** Of 11 repository modules, 9 are
relational and carry 74 SQLAlchemy handlers; `users/repository.py` catches nothing;
and `auth/repository.py` is a **document-store repository** — `UserRepository` and
`RefreshTokenRepository` over MongoDB and Redis, with 13 `PyMongoError`/
`DuplicateKeyError` handlers and 6 `RedisError` handlers, and no statement on the
SQLAlchemy session. It has no transaction to roll back, so the rollback obligation
does not reach it; the classification obligation does.

That split matters beyond bookkeeping, because it splits the `"DB_ERROR"`
correction too. 49 of the 56 literals are relational, where the transaction is dead
and 503 → 500 is the fix. The other 7 are in `auth/repository.py` and describe Mongo
or Redis being unreachable — genuinely retryable, correctly 503. Applying one
correction to all 56 would have relabelled a retryable outage as a permanent failure
on the login path. One string was carrying two failure modes, which is the same
defect as `redis_func.py` raising `DatabaseException` for Redis, arrived at from the
opposite direction.

### D9 — log at construction only when an exception was involved

Inside an `except` block, log — something threw, and the stack context is here and
nowhere else. For a plain `if` / `is None` check, do not log: a not-found is often
ordinary control flow (a first upload rather than a duplicate), and logging every
one converts normal branching into incident noise.

**Alternative considered:** log in `FeatureError.__init__` uniformly. Rejected —
it cannot distinguish the two cases, and it puts I/O in a constructor.

### D10 — `app_error_to_exception` survives, narrowed

It is not deleted. Raising boundaries still need it — auth dependencies, WebSocket
sessions, Celery tasks, MCP handlers. What is retired is calling it from a
repository or service, which is what `no-raise-app-error-mapper` already flags at
34 sites.

### D11 — no feature carries two error systems

A feature's old exception classes are deleted in the same change that replaces
their last call site. A half-migrated feature with both systems live is worse than
either system alone: two vocabularies for one failure, and no exhaustiveness,
since the old classes remain constructible.

Repo-wide, both systems coexist while the 16 remaining features migrate. That is
acceptable — the boundary between them is a feature directory, which is a boundary
a reader can see.

### D12 — enforcement is gates, not review

A closed union is closed only if nothing subclasses the base outside the feature's
`errors.py`. That property cannot be maintained by attention. Rules added or
rewritten:

| Rule | Checks |
|---|---|
| `no-match-on-result` (rewrite) | currently matches only the empty `case Success()` / `case Failure()` form, so `case Success(value):` passes unflagged — it does not enforce what its message claims |
| `no-raise-app-error-mapper` (keep) | 34 outstanding violations retire as features migrate |
| `error-inherits-feature-error-directly` (new) | D7 — concrete types are flat siblings |
| `error-classification-is-classvar` (new) | D2 — `kind`/`code`/`retryable` are never fields |
| `feature-error-in-union` (new) | a concrete type declared but absent from the union |
| `repository-rollback-on-db-except` (new) | D8 |
| `no-cross-feature-error-import` (new) | D3 |
| `no-new-app-error-subclass` (new) | D13 — the retired hierarchy may only shrink |
| one rule per classified boundary | the layer table in `result-layer-boundaries` |

### D13 — the `AppError` hierarchy is frozen, not merely deprecated

No new `AppError` subclass and no new field on an existing one, from this change
until the last feature migrates. New error types are declared as `FeatureError`
subclasses in a feature's `errors.py`.

Deprecation-in-place fails here for a specific reason: the migration spans 17
changes, and during that window adding to the old hierarchy is locally reasonable
— the feature has not migrated yet, the old types are what it uses. Each such
addition is individually defensible and collectively makes the hierarchy grow
while it is supposed to be retiring. A gate converts "please don't" into "you
can't", and makes the subclass count a monotonically decreasing number that
reaching zero is the migration's completion signal.

### D14 — every gate is verified against a permit case and a forbid case

No gate is trusted, and no count derived from a gate is cited, until the rule has
been shown to flag the construct it forbids and to leave the nearest permitted
construct alone.

This is a direct response to what `no-match-on-result` turned out to be. Its
message says match/case does not narrow `Result`; its pattern is
`regex: ^(Success|Failure)\(\s*\)$` against a `case_pattern`, which matches only
the argument-less form. `case Success(value):` — the form the superseded spec
mandated, and the form this design forbids — sails through. Its zero-violation
count reads as "the codebase is clean" and means "the rule looked for something
nobody writes".

The risk this creates is specific and worth naming: the rewritten rule must reject
`case Success(value):` while accepting `case SubscriptionNotFoundError():`, because
this design forbids the first and *requires* the second. A pattern careless about
that distinction would flag the endorsed construct at every dispatch site in the
system.

### D15 — a union is owned by whoever classifies the exception, not by `features/`

The rule is: the module that catches a third-party exception owns the union it
classifies into. `features/` is the common case, not the definition.

This was forced by measurement. `src/app/shared/services/` — `storage.py` (boto3),
`tavily.py` (httpx), `mailer.py`, `rate_limiter.py` — has 31 raises of `APIException`
subclasses and 20 `except` clauses, **every one of them a third-party type**, and no
`Result` anywhere. Structurally it is indistinguishable from a feature repository: it
owns a boundary, it catches a library's taxonomy, it decides what that means. A
classification keyed on directory left it, `shared/rag/` (70 sites) and
`shared/crawler/` outside the contract while calling the contract complete.

| Alternative | Why not |
|---|---|
| Leave `shared/services/` raising `APIException` | A feature calling it must then `try`/`except` around an owned call — the exact thing `result-layer-boundaries` forbids — so the feature's own union would be incomplete for failures it must handle |
| Give them a single global `SharedError` union | Reintroduces the cross-cutting enum D3 removes; a storage failure and a Tavily failure share nothing but the word "shared" |
| Move them under `features/` | They have no router, no repository, no table; the directory would lie about what they are |
| **One union per classifying module** | Same rule as a feature, applied by role. `storage.py` owns storage codes, `tavily.py` owns search codes, and neither imports the other |

The consequence for `ErrorKind` is nil: these modules classify into the same seven
kinds, so the boundary dispatch stays fixed-width.

### D16 — the session dependency is pinned, not widened

`get_postgres_db` keeps exactly its current shape and is explicitly forbidden from
inspecting a `Result`.

`src/app/connections/postgres.py:241` yields the session, commits on clean exit, and
rolls back only inside `except Exception`. A returned `Failure` is not an escaping
exception, so when a service swallows one the `except` never fires and `commit()`
runs on a failed transaction. The obvious-looking fix is to make the dependency
smarter. It cannot be: at the point it commits, it has no reference to any `Result`
and has seen no exception. There is nothing for it to inspect.

| Alternative | Why not |
|---|---|
| Have the dependency inspect a returned `Result` | It has no access to one. Threading it there would couple a transport-agnostic dependency to the domain error type |
| Always roll back and require explicit commits | Inverts the contract at 63 call sites for a defect that belongs one layer down; every read path pays for it |
| Add a second rollback in the dependency's `finally` | Rolls back successful work; `finally` cannot tell the two apart |
| **Pin the dependency, roll back in the repository** | The repository is the only layer that knows a statement was issued, which is D8's reasoning arrived at from the other side |

Writing this down as a requirement matters because "make the session dependency
handle it" is the first idea every reader has, and it is unimplementable for a
reason that is not obvious from the dependency's own source.

### D17 — the global dispatcher is exempted, not conformed

`middleware/global_exception_handler.py` keeps its `isinstance` chain over framework
exception types, and no rule this change introduces may flag it.

The types it dispatches on are framework-owned and their inheritance is load-bearing:
`APIException` derives from `HTTPException`, and lines 166–200 of that file record
what happened when the registration was simpler. Starlette splits
`add_exception_handler` across two middlewares and routes the `Exception` key to a
different one than every other key; FastAPI pre-seeds `setdefault(HTTPException,
http_exception_handler)`, so the MRO walk for `ServiceUnavailableException` hit
FastAPI's entry three classes early and returned `{"detail": ...}` — the elaborate
`APIException` branch never executed for any member of the family. Starlette's and
FastAPI's `HTTPException` are also different classes. The comment ends by instructing
readers not to simplify it.

Two things follow. First, the flat-sibling and exhaustive-`match` rules must not be
pointed at this file; they describe closed unions the project owns, and this file
dispatches an open hierarchy it does not. Second — and this is the part that would
otherwise be missed — the handler contains **zero `except` blocks**, because it is
invoked by registration. Every gate that finds error handling by matching `except`
is blind to the single most important error-handling file in the repository, and per
D14 must say so rather than report a clean sweep.

### D18 — a family with no catch site is a defect; re-rooting beats widening

Of the six exception families under `connections/` and `shared/` rooted outside the
project base, **five are caught nowhere at all**:

| Family | Root | Raises | Catches |
|---|---|---|---|
| `TransientExternalError` | `Exception` | 2 | 4, by name |
| `CircuitBreakerOpenError` | `RuntimeError` | 2 | 0 |
| `IdempotencyLockError` | `RuntimeError` | 1 | 0 |
| `AgentMemoryError` +3 | `RuntimeError` | 2 | 0 |
| `CogneeSetupError` +1 | `RuntimeError` | 2 | 0 |
| `StateSchemaVersionError` | `ValueError` | 1 | 0 |

`TransientExternalError` is the shape to copy — declared at the retry boundary,
caught by name at each consuming node. The other five propagate to whichever generic
boundary happens to sit above them: the global handler's unhandled-500 branch on a
request path, `server_middleware.py`'s one bare `except Exception`, or nothing at all
in a Celery worker, whose outbox relay names only `CeleryError` and `PostgresError`.

`connections/celery_registry.py` shows the second correct resolution, and it is the
one that keeps the rule from over-firing. `TaskDispatchError` is rooted at
`CeleryError` and never raised directly; `UnregisteredTaskError` and
`TaskPayloadValidationError` are raised twice each and caught by name nowhere. A rule
that flags "family with no catch clause naming it" would report all three. It should
report none: the relay's `except (CeleryError, PostgresError)` reaches them through
their root, which is exactly the re-rooting outcome this decision prefers. Reachability
is therefore defined over ancestors, not over exact names — and an abstract base with
zero raise sites is a base, not dead code.

| Alternative | Why not |
|---|---|
| Widen a dispatcher to `except Exception` | Cannot distinguish an optional subsystem being absent from a required one being misconfigured; this is why `lifespan.py`'s 14 named handlers are the reference and its single catch-all is the exception |
| Root everything at `APIException` | A circuit-breaker trip inside a Celery worker has no HTTP response to render; the base would imply a transport that is not there |
| Leave them uncaught and rely on the generic 500 | The 500 is what happens today, and it is why a tripped breaker is indistinguishable from a null dereference in the logs |
| **Re-root to a base the path's dispatcher already catches; widen by name only where re-rooting would lie about the transport** | Keeps dispatch declarative, and `lifespan.py`'s `except CogneeDimensionMismatchError` shows the by-name form working where re-rooting a startup-only family would add nothing |

The audit obligation is stated as a requirement rather than a task because the same
gap reappears every time a new subsystem is added: declaring an exception is
satisfying, and wiring the catch is a separate act nobody is prompted to perform.

### D19 — the spine extends `shared/result/`; the old hierarchy is its predecessor, not its rival

Earlier drafts of this design said `app/shared/errors/` "is introduced". That was
written without opening `src/app/shared/result/`, which already contains `errors.py`
(`AppError` plus five subclasses), `types.py` (`AppResult`), `mappers.py` (the
exception bridge) and `logging.py` (the failure logger). Introducing a second
vocabulary package would have created two homes for one concept during the exact
window in which the first is being retired — and `result-layer-boundaries` already
classifies `shared/result/errors.py` as *the* vocabulary layer, so the two specs
would have contradicted each other on where `ErrorKind` lives.

`ErrorKind` and `FeatureError` therefore land in `app/shared/result/`. What the
existing hierarchy turns out to be is more interesting than the path:

```python
class ValidationAppError(AppError):
    kind: Literal["validation"] = "validation"
    code: str = ErrorCode.VALIDATION_ERROR
```

Five subclasses, each with a `kind` `Literal` field — `validation`, `not_found`,
`conflict`, `infrastructure`, `external_service`. **Exactly the five kinds scoping
proposed.** The five-member vocabulary was not an arbitrary starting point; it was
this hierarchy, read back. That also explains why `AUTHENTICATION` and
`AUTHORIZATION` were missing from it (D4/ADR-003): they are missing from the code
too, which is why `auth`'s 16 `UnauthorizedException` raises have no home in the
`AppError` world and route around it entirely.

So the change is narrower than "introduce a vocabulary" and sharper than "add two
members". It is three edits to a shape that already exists:

| Existing | Becomes | Why |
|---|---|---|
| `kind: Literal[...]` on 5 subclasses | `kind: ClassVar[ErrorKind]` on flat siblings | a field is settable per-instance and invites a sixth subclass; a `ClassVar` is not constructible at all |
| `code: str = ErrorCode.X` | `code: ClassVar[<Owner>Code]` | the annotation is `str`, which is precisely why 118 off-enum literals type-check today |
| 5 kinds | 7 kinds | 401 and 403 have no expression, so a failed login renders 422 |

Two consequences worth carrying into the tasks. First, the migration surface is
**123 construction sites**, not the 5 class definitions: `InfrastructureAppError` ×72,
`NotFoundAppError` ×21, `ConflictAppError` ×15, `ValidationAppError` ×10,
`ExternalServiceAppError` ×2, bare `AppError` ×3. They retire per feature, which is
what makes the 16 feature changes the bulk of this work rather than the foundation.

Second, `AppError` **has no `kind` field** — only its subclasses do. A bare instance
therefore raises `AttributeError` on `.kind`, and one exists:
`features/ingestion/service.py:86` constructs `AppError(code="UNKNOWN",
message=str(failure))`. Today that is harmless because nothing dispatches on `kind`
anywhere in the codebase — the field is declared and never read. The renderer will be
its first consumer, so the kindless instance has to go before `render_result` can meet
it, not after.

That bare instance is also a live misclassification, independent of this design.
`mappers.py`'s `case AppError():` arm sits last, correctly ordered narrowest-first, and
maps the base to `ValidationException` — **422**. So an internal failure labelled
`"UNKNOWN"` is currently reported to the client as a validation error, which tells the
caller to fix their input for a fault that is not theirs. It corrects to 500 under
`STATUS_BY_KIND`.

The same mapper is the codebase's own demonstration of D2's argument: because its
`match` ends in a concrete-base arm rather than a closed union, `assert_never` cannot
be used there, and no type checker can tell whether its six arms are complete.

### D20 — the exemption is the unit of work in the five later-added directories

`app/api/`, `app/config/`, `app/examples/`, `src/database/` and `src/tasks/` were added
to scope after the first three review passes. Together they are 22 `.py` files, 12
raises and 47 `except` clauses — under a tenth of the infrastructure surface D15–D18
cover. The interesting property is not their size but their **kind**: almost nothing in
them converts to `Result`. What they need is the opposite — a written statement of what
is *exempt*, and from what.

| Construct | Sites | Disposition | Why |
|---|---|---|---|
| Pydantic validator `ValueError` | `config/settings.py:473`, `api/strict_envelope.py:26` | exempt, never convert | `ValueError` *is* Pydantic's signalling protocol; a `Result` there validates successfully |
| PEP 562 module `__getattr__` `AttributeError` | `src/database/__init__.py:37` | exempt, never convert | `hasattr` and `from … import` depend on the raise |
| `NotImplementedError` stub | `tasks/pageindex_tasks.py:30` | exempt | an unwritten function, not error handling |
| Version-router aggregation | `api/v1.py`, `api/v2.py` | no rule | constructs, catches, propagates and renders nothing |
| ORM declaration | `database/base.py`, `database/schemas/*` | no rule | same |
| Broad catch **with** a written reason | 55 of 62 repo-wide `# noqa: BLE001` | endorsed form | already the convention; `lifespan.py:234` is the reference |
| Broad catch with a bare suppression | 7 — `subscriptions/service.py` ×4, `billing_tasks.py` ×3 | violation | worse than no suppression: silences the rule that asks *why* and answers nothing |
| Blind `except` ending in `raise` | `middleware/server_middleware.py:100` | no rule, no reason owed | nothing was survived; `BLE001` itself spares this shape |
| `except Exception` → relabel as unavailable | `api/generation_with_cb.py:33,36` | convert | a breaker that trips on the project's own `TypeError` makes its own metric unreadable |
| Seeder-loop catch | `database/seeders/run_seeders.py:81` | keep the catch, fix the exit status | a silently-failing seeder produces a database that looks seeded |

**The distinguishing test for the first three rows is who reads the raise, not what is
raised.** A `ValueError` inside a validator is read by Pydantic; a `ValueError` in a
service is read by project code and converts like any other. Typing the exemption to
the exception class would be wrong and would exempt the wrong sites.

**`app/examples/` is the one that changes how the gates are read.** Its violation is
not four bad lines; it is that `pyproject.toml`'s `per-file-ignores` disables
`BLE001`, `E722`, `B904`, `TRY201`, `TRY300`, `TRY301`, `TRY400` and `S112` for
`src/app/examples/*.py` — eight of the rules this change is about — with a second,
narrower block for `rag_agent_advanced.py` whose `BLE001` is already dead (that file
has no blind `except`). So `ruff check src/app/examples/` says *"All checks passed!"*
while `ast-grep scan` reports **4 `error`-level `no-raw-httpexception` violations** in
`redis_examples.py`. ast-grep is the only gate still reporting because it is the only
one with no per-path ignore configured.

This is **ADR-005 in a second form.** The first was a rule whose pattern matched a form
nobody writes; this is a working rule pointed away from the code. Both produce the same
artefact — a zero count that reads as coverage — from opposite causes. So ADR-005's
obligation extends: verifying a gate means checking what it was configured to skip, not
only that its pattern fires. The 8 entries are removed here, and the findings that
surface are fixed rather than re-suppressed.

Two consequences worth carrying forward. The 8 `except DatabaseException` catches in
`redis_examples.py` must move in the **same** change that reclassifies
`utils/cache/redis_func.py`, since this file is one of that module's only two importers
— split across two changes, the example would catch an exception nothing raises and
stop handling anything without failing. And 4 of the 7 reasonless `# noqa: BLE001`
sites are in `features/subscriptions/service.py`, the exemplar this change migrates, so
they close under section 5 whether or not the degradation rule exists.

## Risks / Trade-offs

- **Rewriting `no-match-on-result` could mask a real violation.** Its pattern is
  `regex: ^(Success|Failure)\(\s*\)$` on a `case_pattern` — it fires only on the
  argument-less form. The bound form the deployed spec mandates is invisible to
  it. → The rewrite is verified against a fixture containing both forms before it
  is trusted, and the "zero match-on-Result" count is re-measured with the new
  rule rather than carried over.
- **17 small enums invite copy-paste divergence.** → `ErrorKind` carries every
  cross-feature decision; the code enums are only for identity in the envelope.
- **`extra="forbid"` turns a stale keyword into a runtime `ValidationError`.**
  A call site left passing `code=` fails at construction, inside an `except`
  block, where it can mask the original error. → Migration is per feature and
  gated by `ty`, which sees the unknown keyword statically; the runtime failure
  is the second line of defence, not the first.
- **The rollback fix changes behaviour where a failure is currently swallowed.**
  → The 21 `webhooks` unwraps are the first place to look for tests that encoded
  the old behaviour.
- **`test_error_envelope_is_universal.py` asserts over 31 exception names, 12
  `ErrorCode`s and 8 envelopes.** → It is updated in this change, not deferred;
  it is the closest thing the repo has to a contract test for the envelope.
- **Two features cannot be verified end to end.** `crawler` and `ingestion`
  routers are mounted in neither `api/v1.py` nor `api/v2.py`. → Their changes
  note it; mounting is out of scope.
- **`utils/cache/redis_func.py` is reachable from no request path.** Its only
  importers are `utils/cache/__init__.py` and `examples/redis_examples.py`, so its 27
  `DatabaseException` raises for Redis failures are a latent misclassification rather
  than a live 500. → Scheduled with the foundation anyway, because it is the
  documented cache pattern and its 4 `no-raw-httpexception` violations live in its
  only caller; but the risk register should not carry it as a production fault.
- **The ten non-feature directories widen the foundation change.** Naming them
  by role rather than converting them keeps the added work bounded, but `shared/` is
  111 files and a later reader could read "in scope" as a mandate. → Scope is stated
  per subtree in the proposal's table, and D15's rule is "whoever classifies a
  third-party exception", which excludes the graph nodes and LLM plumbing by
  construction rather than by exception. The five added in the fourth pass are
  covered by D20, which states them as exemptions rather than conversions.
- **A passing gate may have been configured to skip the code.** Eight error-handling
  ruff rules are disabled for `src/app/examples/*.py`, so that directory's clean lint
  run has never been evidence. → D20 removes the entries; ADR-005's verification
  obligation is read to include auditing a gate's exclusions, not only its pattern.

## Migration Plan

**17 changes total: this foundation, then 16 features.** 18 features exist. The
arithmetic differs from the 18 changes estimated when scope was agreed for two
reasons: `subscriptions` migrates inside this change as the exemplar rather than
getting its own, and `chat` needs no change at all — it is `__init__.py` and
`model.py`, with zero raises and zero `except` clauses.

**Phase 1 — this change.** Shared spine (`app/shared/result/` extended in place, the
renderer, `mappers.py`, the global handler); rollback added to all 9 relational
repositories' 74 SQLAlchemy handlers; the doc surfaces reconciled; the gates written
and verified against fixtures; `subscriptions` migrated end to end.

Also in Phase 1, from the infrastructure scope: the six unrooted families re-rooted
or named and reachability defined over ancestors (D18); the session dependency's shape
pinned by requirement, with no code change (D16); the dispatcher's exemption written
into the gates so no new rule flags it (D17).

And from the five directories added in the fourth pass (D20): the three
framework-contract raises exempted by name so no gate fires on them; the eight
error-handling entries removed from `per-file-ignores` for `src/app/examples/*.py` and
the findings that surface fixed, including the 4 live `no-raw-httpexception`
violations; `generation_with_cb.py`'s broad-catch relabel replaced with named provider
classification; the seeder loop's exit status corrected; and the 7 reasonless
`# noqa: BLE001` sites given a reason — 4 of which are in `subscriptions/service.py`
and close under the exemplar work regardless.

**`utils/cache/` lands here, not deferred.** Task 7.5 reclassifies its 27
`DatabaseException` raises, and `examples/redis_examples.py`'s 8
`except DatabaseException` catches move in the same change because that file is one of
the module's only two importers — split apart, the example would catch an exception
nothing raises and stop handling anything without failing. An earlier draft of this
plan listed the pair as deferred while the risk register and `tasks.md` both scheduled
it; the schedule is correct and the deferral line was stale.

**Phase 1a — `shared/services/`, immediately after the foundation and before any
feature.** This is an ordering constraint, not a preference. A feature that migrates
while its shared dependency still raises `APIException` must `try`/`except` around a
call the project owns — precisely what `result-layer-boundaries` forbids — so the
wrapper has to be Result-typed before the first feature that consumes it.

The 31 raises are not spread evenly, and which module blocks which feature decides
the ordering:

| module | raises | catches | earliest consuming feature |
|---|---|---|---|
| `tavily.py` | 8 | 3 | **`crawler` (3rd)**, via `search` re-exported from the package `__init__` |
| `storage.py` | 21 | 15 | `profile` (7th), then `invoices` (9th), `documents` (15th); also `lifecycle/lifespan.py` |
| `mailer.py` | 2 | 2 | none — no importer outside `shared/services/` |
| `rate_limiter.py` | **0** | **0** | `crawler` (3rd), but nothing to convert |

`tavily.py` is what makes Phase 1a urgent, not `rate_limiter.py`. `rate_limiter.py`
raises nothing and catches nothing — `check_rate_limit` returns
`tuple[bool, dict[str, Any]]`, and the 429 is raised by `crawler/router.py` from that
boolean. There is no `APIException` for a consuming feature to wrap, so converting it
is a no-op for this contract and it is excluded from the conversion.
`result-layer-boundaries` classifies that guard so no gate fires on it.

Two shapes inside Phase 1a are worth naming before the work starts. Four of
`tavily.py`'s 8 raises (lines 58–64) are pre-flight argument guards — missing API key,
empty query, non-positive `max_results`, unknown `topic` — not classifications of a
third-party failure; a missing API key is a configuration fault and the other three
are caller-contract violations. Only its 4 `ExternalServiceException` raises are
genuine boundary classification. `storage.py` is the opposite and simpler: 17 of its
21 raises are `ServiceUnavailableException`, which maps to `INFRASTRUCTURE` with
`retryable=True` and renders 503 — the same status it produces today, so its
conversion has no observable break.

Three modules with 31 raises still makes this the smallest conversion in the
repository, which also makes it the second exemplar after `subscriptions`.

Deferred to their own changes: `shared/crawler/` alongside the `crawler` feature; and
`shared/rag/`'s provider boundary alongside `documents`, whose ingestion path consumes
it. `utils/cache/` is **not** among them — see Phase 1 above.

Rollback for all 9 relational repositories lands here rather than per feature, because it
is a correctness fix with no dependency on the error redesign, and leaving it
staged across 17 changes leaves poisoned-commit paths open for the duration.

**Phase 2 — 16 features, in ascending difficulty:**

`search` → `audit` → `crawler` → `users` → `ingestion` → `dunning` → `profile` →
`plans` → `invoices` → `payments` → `webhooks` → `agent_saul` → `health` →
`credits` → `documents` → `auth`

Difficulty here means error-site count, unwrap-site count, and how many
exception-native boundaries the feature touches. `search` is first because it is
small enough that a mistake in the exemplar surfaces cheaply. `auth` is last
because it is the security boundary and its dependencies are the one place
raising is structurally required. `plans` is early relative to its size because
it is the only repository already using the enum, so it is the least changed.

**Per-feature exit criteria** — all must hold before the change is archived:

1. `errors.py` exists; every repository and service method returns
   `<Feature>Result[T]`.
2. The feature's old exception classes are deleted; no call site remains.
3. Its chains are flattened; no concrete type inherits a concrete type.
4. Its endpoints render through `render_result`; no endpoint raises for an
   expected failure.
5. Its database handlers roll back.
6. `ruff` clean, `ty` no new errors, `ast-grep` no new violations, its tests pass.
7. No `# noqa` or `# ty: ignore` added to reach 6.

**Rollback strategy.** Per feature, a revert is a single-change revert: the
feature's `errors.py` and its call sites move together, and no other feature
imports them — that is what D3's no-cross-feature-import rule buys operationally.
The foundation change is the exception: reverting it after features have migrated
would strand them, so it is the one change that must be rolled forward, not back.

**Observable break.** `"DB_ERROR"` → the enum's `DATABASE_ERROR` at 56 sites,
with the status correcting 503 → 500. This lands in the foundation change, in one
step, rather than drifting across 17 releases.

## Open Questions

- Whether `docs-site/architecture/error-and-result-pattern.mdx` should document
  the per-feature union in full or link to `.opencode/instructions/`. Answerable
  when the rewritten instruction docs exist; it changes no spec, no interface,
  and no task beyond the wording of one doc task.
