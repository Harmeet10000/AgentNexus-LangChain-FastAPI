## Purpose

Extends the error contract to the non-feature directories that carry the machinery
every feature's errors travel through — `connections/`, `lifecycle/`, `middleware/`,
`shared/`, `utils/`, plus `app/api/`, `app/config/`, `app/examples/`, `src/database/`
and `src/tasks/` — so that the request-scoped session, the global dispatcher, the
shared third-party wrappers, the cache layer, the startup degradation boundary, the
task bodies, the framework-contract raises and the code the project offers as an
exemplar each have a stated rule rather than being covered only by implication.

## ADDED Requirements

### Requirement: A shared module that wraps a third-party service SHALL own an error union, not raise an APIException

A module under `src/app/shared/` that catches a third-party library's exception on
behalf of callers SHALL classify it into a typed error and return `Failure`, on the
same terms a feature repository does. It SHALL NOT convert a caught library
exception into a raised `APIException`.

Ownership of a union is a property of the module that classifies the error, not of
whether that module happens to live under `features/`. Such a module SHALL declare
its own `errors.py` with its own `<Module>Code` StrEnum, its own flat sibling error
types and its own closed union, exactly as `feature-error-contract` requires of a
feature. It SHALL NOT import a feature's error types, and no feature SHALL import
its codes.

This is measured, not hypothetical. `src/app/shared/services/` is four modules —
`storage.py` (boto3), `tavily.py` (httpx), `mailer.py`, `rate_limiter.py` — with
**31 raises** of `APIException` subclasses (`ServiceUnavailableException` ×16,
`ValidationException` ×9, `ExternalServiceException` ×6) and **20 catches, every
one of a third-party type** (`BotoCoreError`, `ClientError`, `httpx.HTTPStatusError`,
`httpx.TimeoutException`, `httpx.RequestError`). It uses no `Result` anywhere. It is
the exact shape this work converts — a service that owns a third-party boundary —
and it was outside the classification because it is not a feature. It is also not
peripheral: `storage` is imported by `profile`, `invoices` and `documents` as well as
the lifespan, and `rate_limiter` by `crawler`.

`shared/crawler/` is the same shape at smaller scale — 9 sites classifying
`RedisError`, `json.JSONDecodeError`, `httpx.HTTPError` and `PlaywrightError`.

`shared/rag/` is **mixed and SHALL NOT be converted wholesale.** Of its roughly 48
handlers, 17 catch bare `Exception` and 7 catch `ImportError` as optional-dependency
guards, while the genuine provider classification is concentrated in a
`_provider_failure` helper used at 4 sites alongside named catches of `DoclingError`,
`genai_errors.APIError`, `GraphitiError` and `yaml.YAMLError`. Only the provider
boundary owes a union. The pipeline around it stays exception-native, and an
`except ImportError` that sets an availability flag is capability detection, not error
handling, and owes no classification at all.

#### Scenario: A shared third-party wrapper returns a typed failure
- **WHEN** `shared/services/storage.py` catches `(BotoCoreError, ClientError)` from an S3 call
- **THEN** it returns `Failure` carrying a member of its own closed union, and raises no `APIException`

#### Scenario: An optional-dependency guard owes no union
- **WHEN** a module catches `ImportError` to record that an optional backend is unavailable
- **THEN** no error type is constructed and no union is required, because the handler reports a capability rather than a failure

#### Scenario: A mixed subtree is converted at its provider boundary only
- **WHEN** the rag pipeline's provider adapter catches a provider library's exception
- **THEN** that boundary returns a typed failure, while the surrounding pipeline's own handlers are classified as exception-native rather than rewritten

#### Scenario: A shared module owns its codes
- **WHEN** the shared services module needs a code for an upload failure
- **THEN** it declares that member in its own StrEnum, and no feature's `<Feature>Code` is imported to supply it

#### Scenario: A feature consuming a shared module narrows rather than catches
- **WHEN** a feature service calls a shared third-party wrapper
- **THEN** the call is not inside a `try`, and the returned `Result` is opened with `isinstance`

#### Scenario: A shared module is not exempted for being shared
- **WHEN** a reviewer argues a module needs no union because it is cross-feature infrastructure
- **THEN** the argument is rejected, because the module classifies third-party exceptions and therefore owns the classification

### Requirement: The cache layer SHALL classify a cache-backend failure as a cache failure

A failure of the cache backend SHALL be classified as a cache or infrastructure
failure distinguishable from a relational-database failure. It SHALL NOT be
reported to a caller as a database error, and its retryability SHALL reflect that a
cache backend is retryable.

A cache helper SHALL NOT catch bare `Exception` around its own logic and convert
whatever it catches into a backend failure. A `TypeError` or `AttributeError` raised
by the helper's own code is a defect in that helper, not a backend outage, and
SHALL NOT be reported as one.

This does not conflict with the degradation-boundary allowance in
`typed-exception-handling`. A degradation boundary catches `Exception` and then
*degrades* — returns a fallback, sets the dependency to `None`, re-enters its loop.
The cache helpers catch `Exception` and *escalate*, converting it into a raise that
becomes a 500. Catching broadly to keep serving is the endorsed pattern; catching
broadly to relabel is not.

Measured: `src/app/utils/cache/redis_func.py` raises `DatabaseException` at **27
sites** through a `_build_database_exception()` helper at line 36, each paired with
one of **27 `except Exception as exc`** handlers. `DatabaseException` carries
`ErrorCode.DATABASE_ERROR` and `HTTP_500_INTERNAL_SERVER_ERROR`, so a Redis outage
is currently reported as a non-retryable Postgres failure. The file also re-raises
its own family first (`except DatabaseException: raise`) before the catch-all, which
correctly avoids double-wrapping and SHALL be preserved in shape.

The module's only importers are `utils/cache/__init__.py` and
`examples/redis_examples.py`; it is reachable from no request path today. The
misclassification is therefore a latent defect and a bad exemplar rather than a live
production fault, and this requirement is scheduled accordingly — but the file is
the documented cache pattern, so leaving it wrong teaches the wrong classification.

#### Scenario: A backend outage is not a database error
- **WHEN** a cache read fails because the Redis backend is unreachable
- **THEN** the failure is classified as a retryable cache or infrastructure error, and the caller does not receive `DATABASE_ERROR`

#### Scenario: A defect in the helper is not laundered into an outage
- **WHEN** a cache helper's own code raises `TypeError` because a caller passed an unserialisable value
- **THEN** that failure is not reported as a cache-backend failure, and the caller can tell a misuse from an outage

#### Scenario: Broad catching to degrade is still permitted
- **WHEN** a caller decides a cache read failure should fall through to the source of truth
- **THEN** it may catch broadly and continue, because the endorsed degradation pattern is to keep serving — the rule forbids catching broadly in order to relabel a failure, not catching broadly in order to survive one

#### Scenario: The own-family re-raise is preserved
- **WHEN** a cache helper calls another helper in the same module that has already classified a failure
- **THEN** the already-classified failure passes through unchanged rather than being wrapped a second time

#### Scenario: The example is corrected with the pattern
- **WHEN** the cache module's classification changes
- **THEN** `examples/redis_examples.py`, its only caller, is updated in the same change, so the documented example never demonstrates the retired classification

### Requirement: The request-scoped session dependency SHALL remain the only committer and SHALL NOT inspect Results

`get_postgres_db` SHALL keep its present shape: yield, commit on clean exit, roll
back only when an exception escapes, close in `finally`. It SHALL NOT be modified to
inspect a returned `Result`, and it SHALL NOT be given a second rollback path
intended to compensate for a repository that did not roll back.

The dependency is the counterparty to `repository-transaction-safety` and explains
why that capability places rollback in the repository. Measured at
`src/app/connections/postgres.py:241`:

```python
async with session_local() as session:
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
```

A returned `Failure` is not an escaping exception. When a service swallows one, this
`except` never fires and `await session.commit()` runs against a session whose
transaction has already failed. The dependency cannot detect that condition — it
sees no exception and has no access to the `Result` — which is why the repository
must roll back and why widening the dependency is not an alternative.

#### Scenario: A swallowed failure still reaches this commit
- **WHEN** a service receives a `Failure` from a write and returns normally
- **THEN** the dependency's `except` branch does not run and `commit()` is reached, confirming that the repository is the only layer that can prevent the poisoned commit

#### Scenario: The dependency is not widened
- **WHEN** an implementer proposes threading a `Result` or a failure flag into the session dependency
- **THEN** the proposal is rejected, because it would couple a transport-agnostic dependency to the domain error type and duplicate a rollback the repository already owes

#### Scenario: The escaping-exception path is unchanged
- **WHEN** an unhandled exception escapes a route that used the session
- **THEN** the dependency still rolls back and re-raises, and this work does not alter that behaviour

### Requirement: The global exception handler's isinstance dispatch SHALL be exempt from the union rules

`middleware/global_exception_handler.py` dispatches on framework exception types
with an `isinstance` chain. That chain SHALL NOT be rewritten as a `match` over a
closed union, SHALL NOT be required to satisfy the flat-sibling rule, and SHALL NOT
be flagged by any rule this work introduces.

The types it dispatches on — `APIException`, `RequestValidationError`,
`StarletteHTTPException`, `Exception` — are framework-owned. Their inheritance is
real, load-bearing, and outside the project's control: `APIException` derives from
`HTTPException`, which is why the handler's registration is split the way it is. The
file carries a long comment at lines 166–200 recording that
`add_exception_handler(Exception, ...)` alone was insufficient, that Starlette routes
the `Exception` key to a different middleware than every other key, that FastAPI
pre-seeds `HTTPException` so the MRO walk resolved three classes early and the
`APIException` branch never ran, and that Starlette's and FastAPI's `HTTPException`
are different classes. That comment ends with an explicit instruction not to
simplify the registration. This requirement makes that instruction a rule.

The handler also contains **zero `except` blocks** — it is invoked by registration,
not by catching. Any enforcement rule that locates error handling by matching
`except` will not see the single most important error-handling file in the
repository, and SHALL NOT be described as covering the dispatcher.

#### Scenario: The dispatcher keeps its isinstance chain
- **WHEN** the container-match rule and the exhaustive-union rule are run over the global exception handler
- **THEN** neither reports a violation, because the handler dispatches framework types rather than a feature's closed union

#### Scenario: The registration is not simplified
- **WHEN** a change proposes collapsing `register_exception_handlers` to a single `Exception` registration
- **THEN** the proposal is rejected with reference to the recorded reason, and the split registration is preserved

#### Scenario: An except-based gate does not claim to cover the dispatcher
- **WHEN** a rule that matches `except` clauses reports its coverage
- **THEN** the global exception handler is named as outside that rule's reach, so a clean count is not read as the dispatcher having been checked

### Requirement: The startup degradation boundary SHALL name every family it survives

`lifecycle/lifespan.py` SHALL keep catching named exception types rather than being
widened to bare `Exception`, and each handler SHALL log before continuing so a
degraded start is visible.

It is the reference implementation of the "widen the dispatcher" resolution offered
by `result-layer-boundaries`' family-rooting requirement. Measured: **14 handlers**
naming roughly twenty distinct types across the startup sequence — including
`ExceptionGroup`, `redis.exceptions.RedisError`, `OperationalError`,
`PlaywrightError`, neo4j's `ServiceUnavailable`, `ServiceUnavailableException`, and
`CogneeDimensionMismatchError` — against exactly **one** `except Exception as exc`.
Catching `CogneeDimensionMismatchError` by name is the correct handling of a family
rooted outside the project's base: the dispatcher was widened to reach it rather
than the family being left unreachable.

A new startup step SHALL name the exceptions its own initialisation can raise. It
SHALL NOT rely on the single existing catch-all, because a catch-all cannot
distinguish an optional subsystem being absent from a required one being
misconfigured.

#### Scenario: A new startup step names its own failures
- **WHEN** a subsystem is added to the lifespan sequence
- **THEN** its initialisation is guarded by a handler naming the exception types it can raise, and it does not depend on the existing catch-all

#### Scenario: A degraded start is logged, not silent
- **WHEN** an optional subsystem fails to initialise and startup continues
- **THEN** a log line records which subsystem degraded and why, and the application does not report a clean start

#### Scenario: A required subsystem still fails the start
- **WHEN** a subsystem the application cannot serve requests without fails to initialise
- **THEN** startup fails rather than degrading, and the failure is distinguishable from an optional subsystem being unavailable

### Requirement: An exception family rooted outside the project base SHALL be caught by name or re-rooted

Every custom exception family declared under `connections/` or `shared/` and rooted
at `RuntimeError`, `Exception` or `ValueError` SHALL either be re-rooted so an
existing dispatcher catches it, or be named explicitly by every dispatcher on its
propagation path. A family with no catch site anywhere SHALL be treated as a defect,
not as a style choice.

Measured across these directories — six such families, of which **one** is ever
caught by name:

| Family | Root | Declared at | Raises | Catch sites |
|---|---|---|---|---|
| `TransientExternalError` | `Exception` | `shared/langgraph_layer/kb_retry.py:80` | 2 | **4**, by name in `ingestion_kb/nodes.py` |
| `CircuitBreakerOpenError` | `RuntimeError` | `connections/celery_reliability.py:69` | 2 | **none** |
| `IdempotencyLockError` | `RuntimeError` | `connections/celery_reliability.py:436` | 1 | **none** |
| `AgentMemoryError` (+3 subclasses) | `RuntimeError` | `shared/langchain_layer/agents/memory/agent_memory_service.py:32` | 2 | **none** |
| `CogneeSetupError` (+1 subclass) | `RuntimeError` | `shared/langchain_layer/agents/memory/cognee_client.py:54` | 2 | **none** |
| `StateSchemaVersionError` | `ValueError` | `shared/langgraph_layer/agent_saul/state.py:356` | 1 | **none** |

`TransientExternalError` is the pattern to copy: declared at the retry boundary,
caught by name at each consuming node. The five with no catch site propagate to
whatever generic boundary happens to sit above them — the global handler's unhandled
branch on a request path, `server_middleware.py`'s single bare `except Exception`, or
nothing at all in a Celery worker, where the outbox relay's publish handler names
only `CeleryError` and `PostgresError`.

`connections/celery_registry.py` shows the other correct resolution. Its
`TaskDispatchError` base is rooted at `CeleryError` and is never raised directly; its
two concrete subclasses `UnregisteredTaskError` and `TaskPayloadValidationError` are
each raised twice, and neither is caught by name anywhere. That is not a gap: the
relay's `except (CeleryError, PostgresError)` reaches them through their root. A
family SHALL be considered reachable when a dispatcher on its path catches any
ancestor of it, and an abstract base that is never raised directly SHALL NOT be
reported as an unused declaration.

#### Scenario: A family with no catch site is closed
- **WHEN** the audit finds a declared exception family with raise sites and no dispatcher on its path catching it or any of its ancestors
- **THEN** either a dispatcher is widened to name it or the family is re-rooted under a base an existing dispatcher already catches

#### Scenario: A family reachable through its root needs no dedicated catch
- **WHEN** a family's base is rooted under a type an existing dispatcher already catches, and only its concrete subclasses are raised
- **THEN** the family is treated as correctly wired, and the absence of a catch clause naming it is not a finding

#### Scenario: An unraised abstract base is not a defect
- **WHEN** a family's base class has zero raise sites because it exists only to root its concrete members
- **THEN** it is left in place, and only a family whose concrete members are all unraised is reported as an unused declaration

#### Scenario: A worker-path family reaches the worker's boundary
- **WHEN** a `RuntimeError`-rooted family is raised inside a Celery-dispatched operation
- **THEN** the task boundary names it, so the failure is retried or dead-lettered rather than escaping as an unclassified worker error

#### Scenario: The retry-boundary pattern is the reference
- **WHEN** a new family is declared at a boundary that wraps an external call
- **THEN** it follows the `TransientExternalError` shape — declared at the boundary, caught by name at each consumer — rather than being declared and left uncaught

### Requirement: Inheritance among the shared spine's exception families SHALL be ordered narrowest-first at every catch site

Where a family in these directories keeps concrete-to-concrete inheritance, every
handler that catches more than one of its members SHALL order them narrowest first.
The shadowing hazard that motivates the flat-sibling rule for `match` applies
identically to `except` clause ordering, which is also resolved by `isinstance` in
source order.

Measured chains inside these five directories, nine in total:
`InvalidTokenException`, `ExpiredTokenException` and `InvalidRefreshTokenException`
under `UnauthorizedException` (`utils/exceptions.py:245,255,262`);
`UnregisteredTaskError` and `TaskPayloadValidationError` under `TaskDispatchError`
(`connections/celery_registry.py:86,101`); `CogneeDimensionMismatchError` under
`CogneeSetupError`; and `ConversationIdentityRequiredError`,
`PartitionIdentityInvalidError` and `ConsolidationPreconditionError` under
`AgentMemoryError`.

These are exception families, not `FeatureError` unions, so `feature-error-contract`'s
flat-sibling rule does not reach them and they are not required to flatten while
they remain exception-based. A member that migrates into a feature's closed union
SHALL flatten at that point.

#### Scenario: A broader member does not shadow a narrower one
- **WHEN** a handler catches both a family's base and one of its subclasses
- **THEN** the subclass clause appears first, so the specific branch is reachable

#### Scenario: A migrating member flattens
- **WHEN** an authentication exception subclass is replaced by a member of the auth feature's closed error union
- **THEN** the replacement is a flat sibling of the other members and inherits from no other concrete error type

#### Scenario: An exception family is not forced to flatten prematurely
- **WHEN** a family in these directories remains exception-based because its layer is exception-native
- **THEN** its inheritance is permitted, and only its catch-site ordering is required

### Requirement: A raise that satisfies a framework contract SHALL be exempt from the union rules

Where a framework, the standard library or a protocol requires a specific builtin
exception to be raised, that raise SHALL remain a raise of that builtin and SHALL NOT
be converted into a `Result` or into a project error type. Such a site SHALL NOT be
reported as a violation by any gate this change introduces.

Three instances exist and they are the whole population:

| Site | Raises | Contract |
|---|---|---|
| `app/config/settings.py:473` | `ValueError` | a Pydantic v2 validator signals failure with `ValueError`; Pydantic wraps it into `ValidationError`. Returning a `Result` would make the field validate successfully |
| `src/database/__init__.py:37` | `AttributeError` | PEP 562 module `__getattr__` must raise `AttributeError` for an unknown name, or `hasattr` and `from … import` break |
| `app/api/strict_envelope.py:26` | `ValueError` | envelope shape assertion inside a validator, same Pydantic contract |

`src/tasks/pageindex_tasks.py:30` raises `NotImplementedError` for an unimplemented
task body. It is not a framework contract but it is not error handling either — it is
an unwritten function — and SHALL be treated the same way: left alone, and not
counted as an unclassified raise.

The distinguishing test is not the exception's type but whether the caller is the
project or the framework. A `ValueError` raised inside a validator is read by
Pydantic. A `ValueError` raised in a service is read by project code, and that one
converts.

#### Scenario: A settings validator keeps raising ValueError
- **WHEN** a Pydantic field validator rejects a configuration value
- **THEN** it raises `ValueError` and no gate reports it, because Pydantic converts that raise into the model's `ValidationError`

#### Scenario: A module __getattr__ keeps raising AttributeError
- **WHEN** `src/database/__init__.py` is asked for a name it does not export
- **THEN** it raises `AttributeError`, because returning a `Result` would make `hasattr` report the attribute as present

#### Scenario: An unimplemented task body is not an unclassified raise
- **WHEN** the gate encounters `raise NotImplementedError` in a task stub
- **THEN** no violation is reported, and the site is excluded from the count of raises awaiting classification

#### Scenario: The same builtin in project code is not exempt
- **WHEN** a service or repository raises `ValueError` for a domain condition
- **THEN** the exemption does not apply, because the caller is project code, and the site converts to a typed failure like any other

### Requirement: A deliberate degradation SHALL name the reason it degrades

A handler that catches broadly in order to keep processing SHALL record, at the catch
site, why the failure is survivable. A broad catch with no stated reason SHALL be a
violation; a broad catch with one SHALL NOT.

This requirement codifies a convention the repository already follows, rather than
introducing one. Measured repo-wide and reconciled by two structurally different
queries: **62 `# noqa: BLE001` sites, 55 of which carry a written reason** after the
code. The reference degradation boundary this change already designates follows it —
`lifecycle/lifespan.py:234` reads
`except Exception as exc:  # noqa: BLE001 — optional dependency; app degrades without it`.
So does `shared/crawler/processor.py:171,201` and
`shared/langchain_layer/agents/tools/shell.py:200`, the latter naming two codes and
then its reason. The rule is therefore already the practice at 55 of 62 sites; what is
missing is that it is written down and gated.

The **7 exceptions are the whole population** and they fall in exactly two places:

| Site | Count | Note |
|---|---|---|
| `features/subscriptions/service.py:324,390,429,482` | 4 | the exemplar feature — these are closed by this change's section 5 regardless |
| `src/tasks/billing_tasks.py:202,242,299` | 3 | the same file's `:134` carries *"one bad subscription must not kill the run"*, so the reason is known and simply not written at three of its four sites |

A bare `# noqa: BLE001` is not a reason. It silences `BLE001` — the rule whose entire
job is to ask why the catch is broad — and records nothing in its place, which makes
it **worse** than an unsuppressed `except Exception:`, because the unsuppressed form
still trips the gate and can be found. A suppression SHALL NOT be accepted in place of
a reason.

`src/tasks/` is exception-native — the framework owns retry and dead-lettering, and
`result-layer-boundaries` classifies task bodies as such. Its 17 `except` clauses
across 6 of its 10 modules divide into 2 documented degradations, 3 bare suppressions
(above), and 12 with no suppression and no reason: `credit_tasks.py:35,102`,
`document_tasks.py:56`, `billing_tasks.py:80`, and 8 in the `auth_email_tasks` pair.

`auth_email_tasks.py` and `auth_email_tasks_typed.py` hold the same four handlers
(2 `except ValueError`, 2 `except Exception`) at near-identical offsets — `:105,111,144,150`
against `:107,113,150,156`. Whichever survives, the duplicate's handlers SHALL NOT be
left as a second, divergent copy of the rule.

One mechanism must be preserved because the linter already encodes this
distinction: **`BLE001` does not fire on a blind `except` that re-raises.**
`middleware/server_middleware.py:100` catches `Exception`, logs with `.exception()`,
and ends in a bare `raise`, and needs no suppression for it. A handler that re-raises
is not degrading — it is a logging pass-through — so a gate written for this
requirement SHALL spare it on the same terms.

#### Scenario: A documented broad catch is preserved
- **WHEN** the gate encounters `except Exception as exc:  # noqa: BLE001 — optional dependency; app degrades without it`
- **THEN** no violation is reported, because the reason the failure is survivable is recorded at the catch site

#### Scenario: A bare suppression is not a reason
- **WHEN** the gate encounters `except Exception as exc:  # noqa: BLE001` with no text after the code
- **THEN** a violation is reported, because the suppression silenced the only rule that asks why the catch is broad without recording an answer

#### Scenario: An undocumented broad catch is a violation
- **WHEN** the gate encounters a bare `except Exception:` in a task body with no recorded reason and no re-raise
- **THEN** a violation is reported, and it is closed either by naming the families the task survives or by recording why surviving everything is correct here

#### Scenario: A logging pass-through is not a degradation
- **WHEN** a handler catches `Exception`, logs it, and ends in a bare `raise`
- **THEN** no violation is reported and no reason is owed, matching `BLE001`'s own behaviour — nothing was survived, so there is nothing to justify

#### Scenario: A task converts a Result it consumes at its own edge
- **WHEN** a task body calls a migrated service and receives a `Failure`
- **THEN** it converts that failure into a retry, a dead-letter or a logged skip at that call site, and does not thread the `Result` further

#### Scenario: The duplicated task module does not carry a divergent copy of the rule
- **WHEN** the two `auth_email_tasks` modules are reconciled
- **THEN** exactly one set of handlers remains, and no second copy of the same four handlers survives with different behaviour

### Requirement: Example code SHALL NOT demonstrate a pattern the project forbids

Code under `app/examples/` SHALL satisfy the same enforcement gates as production
code. An example demonstrating a retired or forbidden pattern SHALL be corrected or
deleted, not exempted, because an example's purpose is to be copied. The directory
SHALL NOT hold a lint exemption for an error-handling rule.

**The exemption is not implicit — it is written down.** `pyproject.toml`'s
`[tool.ruff.lint.per-file-ignores]` gives `src/app/examples/*.py` a **25-rule** block,
of which **8 are the error-handling rules this change is about**:

| Ignored rule | What it permits |
|---|---|
| `BLE001` | blind `except Exception` |
| `E722` | bare `except:` |
| `B904` | `raise` inside `except` without `from` — loses the cause chain |
| `TRY201` | `raise e` where a bare `raise` belongs |
| `TRY300` | a `return` in `try` that belongs in `else` |
| `TRY301` | `raise` inside `try`, caught by its own handler |
| `TRY400` | `logger.error` in an `except` where `.exception()` belongs |
| `S112` | `try`/`except`/`continue` — swallowing in a loop |

This is why `uv run ruff check src/app/examples/` reports **"All checks passed!"** while
`ast-grep scan` reports **4 `error`-level `no-raw-httpexception` violations** in the
same directory: ruff was told to look away, and ast-grep has no per-path ignore
configured, so it is the only gate still reporting. A green ruff run over this
directory is not evidence about the code — it is evidence about the ignore list. Those
8 entries SHALL be removed and the resulting findings fixed.

`app/examples/rag_agent_advanced.py` carries a **second**, additional block —
`ANN201, ARG001, ASYNC250, B007, BLE001, F821, PLC0415` — whose `BLE001` is already
dead: that file's 12 handlers are all named, so nothing in it is blind. It SHALL be
dropped as part of the same edit.

The concrete violations, enumerated:

| Site | Defect |
|---|---|
| `redis_examples.py:211,239,265,299` | `raise HTTPException(status_code=500, detail=…)` — the 4 live `no-raw-httpexception` errors; loses the `error_code`, the structured message and the `data` payload the global handler extracts |
| `redis_examples.py:97,108,136,148,164,179,263,297` | 8 `except DatabaseException as e` — downstream of the cache misclassification this change corrects |
| `redis_examples.py:209,237,323,353,386,417` | 6 `except Exception as e`, permitted only by the `BLE001` entry above |
| `logger_usage_example.py:60` | `raise e` inside its own `except Exception as e` handler — `TRY201`'s target; a bare `raise` re-raises in place without adding the current frame |

The 8 `except DatabaseException` catches SHALL be updated **in the same change** that
reclassifies `utils/cache/redis_func.py`, because this file is one of that module's
only two importers. Split across two changes, the example would catch an exception the
cache no longer raises and silently stop handling anything.

`app/examples/rag_agent_advanced.py` is the counter-example that keeps this rule from
over-firing: 9 handlers catch `(OpenAIError, GoogleAPIError)` by name and 3 catch
`EOFError`/`KeyboardInterrupt` for its CLI loop, which is the endorsed form. It needs
no behavioural change, and a gate that flags it is wrong.

#### Scenario: An example is held to the production gates
- **WHEN** `ast-grep scan src/` and `uv run ruff check src/` are run over a completed change
- **THEN** both report zero violations under `app/examples/`, and neither carries a path exclusion for an error-handling rule in that directory

#### Scenario: The error-handling ignores are removed from the ignore list
- **WHEN** the `per-file-ignores` entry for `src/app/examples/*.py` is edited
- **THEN** `BLE001`, `E722`, `B904`, `TRY201`, `TRY300`, `TRY301`, `TRY400` and `S112` are absent from it, and the findings that appear as a result are fixed rather than re-suppressed

#### Scenario: A green lint run over an exempted directory is not evidence
- **WHEN** a reviewer cites a passing `ruff check` over `app/examples/` as proof the directory is clean
- **THEN** the claim is rejected until the ignore entry is read, because 8 of the rules that would report were disabled for that path

#### Scenario: The raw HTTPException raises are corrected, not exempted
- **WHEN** the 4 `raise HTTPException(status_code=500, …)` sites in `redis_examples.py` are addressed
- **THEN** each raises a typed exception or renders a classified failure, and the `no-raw-httpexception` rule reports zero violations there

#### Scenario: The example's catches follow the cache reclassification
- **WHEN** `utils/cache/redis_func.py` stops raising `DatabaseException` for a Redis failure
- **THEN** the 8 `except DatabaseException` sites in `redis_examples.py` are updated in that same change, so the example does not catch an exception that is no longer raised

#### Scenario: A correct example is left alone
- **WHEN** the gate encounters handlers catching `(OpenAIError, GoogleAPIError)` by name
- **THEN** no violation is reported, because naming the third-party families is the pattern this change endorses

### Requirement: A generation adapter behind a circuit breaker SHALL classify by name, not relabel a broad catch

A module that wraps a model-provider call behind a circuit breaker SHALL name the
provider families it converts, and SHALL NOT catch `Exception` and relabel it as a
service-unavailable condition.

`app/api/generation_with_cb.py` is the single instance: `except Exception as e` at
line 33 followed by `raise ServiceUnavailableException(msg) from e` at line 36. Every
failure inside the guarded call — a provider timeout, a malformed response, a
`TypeError` in the project's own callback — is reported to the caller as the upstream
being unavailable. A breaker that trips on the project's own bug counts a local defect
as an upstream outage, and the metric that is supposed to protect the upstream becomes
unreadable.

This is the escalation half of the degradation distinction: catching broadly to keep
serving is endorsed, catching broadly to **relabel** is not. `shared/rag/`'s
`_provider_failure` boundary is the shape to follow.

The rest of `app/api/` — `v1.py`, `v2.py` and `__init__.py` — is version-router
aggregation with no error handling and needs no rule, per `result-layer-boundaries`'s
scenario for files that handle no errors.

#### Scenario: A provider failure is named
- **WHEN** the guarded generation call fails because the provider timed out or returned an error
- **THEN** the adapter catches that provider's exception by name and classifies it as an external-service failure

#### Scenario: A local defect is not reported as an upstream outage
- **WHEN** a `TypeError` or `AttributeError` originating in project code is raised inside the guarded call
- **THEN** it is not converted into a service-unavailable condition, and it does not count toward the circuit breaker's failure threshold

#### Scenario: The version routers need no rule
- **WHEN** a reviewer opens `app/api/v1.py` or `app/api/v2.py`
- **THEN** no row and no requirement applies, because they aggregate routers and neither construct, catch, propagate nor render an error

### Requirement: A seeder SHALL survive one failing seeder without reporting success

The database seeder runner SHALL record which seeder failed and SHALL NOT report a
successful seed when one of its steps raised. Its broad catch SHALL name the reason it
continues, on the same terms as a task body.

`src/database/seeders/run_seeders.py:81` holds the directory's only handler, an
`except Exception as exc:` in the seeder loop. Its role is operator-facing script, so
it owes no envelope — `result-layer-boundaries` classifies scripts that way — but it
does owe an accurate exit status, because it is run by hand and by the CI migration
step, and a seeder that fails silently produces a database that looks seeded.

`src/database/base.py` and `src/database/schemas/` are ORM models and declare no error
handling; they need no rule.

#### Scenario: A failing seeder is named and the run does not report success
- **WHEN** one seeder in the loop raises while the others succeed
- **THEN** the failure is logged with the seeder's identity, and the runner's exit status reports failure rather than success

#### Scenario: The ORM schema modules need no rule
- **WHEN** a reviewer opens `src/database/base.py` or a module under `src/database/schemas/`
- **THEN** no row and no requirement applies, because they declare tables and handle no errors
