> Change class: **L** — cross-cutting (18 features, shared spine, security boundary, public error envelope, enforcement gates)

## Why

`AppError` is an open hierarchy. Because anyone can subclass it anywhere, no type
checker can enumerate its subclasses, so no `match` over it can ever be proven
exhaustive. Every downstream weakness follows from that one property:

- **56 sites** across 9 repositories construct `InfrastructureAppError(code="DB_ERROR")`.
  `"DB_ERROR"` matches no `ErrorCode` member — the enum spells it `DATABASE_ERROR`.
  `plans/repository.py` is the only repository that uses the enum, in structurally
  identical `except SQLAlchemyError` blocks. The literal survives to the client:
  `mappers.py` forwards `error_code=error.code` and the handler emits it verbatim.
  Because `InfrastructureAppError` defaults `retryable=True`, these map to **503,
  not 500** — clients are told to retry a transaction that is already dead. For **49
  of the 56** that is the correction to make. The other **7 are in
  `auth/repository.py`, which is MongoDB and Redis, not SQLAlchemy** — those are
  genuinely retryable and must keep 503. One string was hiding two different failure
  modes.
- **118 off-enum code literals** emit **68 distinct codes** against an 18-member
  enum. `code` is a Pydantic field with a default, so every construction site can
  invent or mistype one. The drift is not an accident; it is the only thing the
  type is shaped to allow.
- **Zero repositories roll back.** Of 11 repository modules, **9 are relational** and
  carry 74 SQLAlchemy handlers between them; every one returns `Failure` and leaves
  the session in a failed transaction. (`auth` is a document store; `users` catches
  nothing.) Where a service swallows that `Failure` instead of raising —
  `webhooks/service.py` (21 unwraps, zero bridge calls), `dunning/service.py` — no
  exception escapes, so `get_postgres_db`'s `except` never fires and
  `await session.commit()` runs against a poisoned session. `session.rollback` has
  never existed under `src/app/features/` in the entire history of the repository.
- **28 concrete-inherits-concrete chains** already exist (19 under `APIException`,
  9 outside it). `match`'s class patterns are `isinstance`-based, so a broader arm
  placed before a narrower one silently makes the narrower arm dead — and a type
  checker still reports the match exhaustive, because from its view every case was
  covered. It has no concept of *reached but shadowed*. This is invisible
  statically and at runtime; it shows up only as the wrong branch's side effects.
- **Five of six unrooted exception families are caught nowhere.** Under
  `connections/` and `shared/`, six families are rooted at `RuntimeError`, bare
  `Exception` or `ValueError`. `CircuitBreakerOpenError`, `IdempotencyLockError`,
  `AgentMemoryError`, `CogneeSetupError` and `StateSchemaVersionError` all have raise
  sites and **zero catch sites** anywhere in the repository; only
  `TransientExternalError` is caught by name. The contrast that shows the fix:
  `celery_registry.py`'s `TaskDispatchError` family is rooted at `CeleryError`, so the
  outbox relay's existing catch reaches its two subclasses without any clause naming
  them.
- **The infrastructure directories were the plan's blind spot.**
  `src/app/shared/services/` is four third-party wrappers (boto3, httpx) with 31
  raises of `APIException` subclasses, 20 catches every one of a library type, and
  no `Result` at all — the exact shape this work converts, invisible to a rule
  scoped to `features/`. It is not peripheral: `storage` is imported by `profile`,
  `invoices` and `documents`, and `rate_limiter` by `crawler`.
  `utils/cache/redis_func.py` raises `DatabaseException` at 27 sites for **Redis**
  failures, so a cache outage would render as a non-retryable `DATABASE_ERROR` 500 —
  latent rather than live, since its only importers are its own `__init__` and
  `examples/redis_examples.py`.
- **One directory is exempted from the error-handling lint rules in writing, and it
  is the one that gets copied.** `pyproject.toml`'s `per-file-ignores` disables
  `BLE001`, `E722`, `B904`, `TRY201`, `TRY300`, `TRY301`, `TRY400` and `S112` for
  `src/app/examples/*.py`. That is why `ruff check src/app/examples/` reports "All
  checks passed!" while `ast-grep scan` — which has no per-path ignore — reports **4
  `error`-level `no-raw-httpexception` violations** in `redis_examples.py`. A green
  lint run over that path has never been evidence about the code.

A feature-local closed union fixes this at the type-system level rather than by
convention. Measured on this repository with the project's own `ty`:

| Construct | `ty` verdict |
|---|---|
| `match result: case Success(value)` | **no narrowing** — binds `int \| DupErr \| MissingErr` |
| `isinstance(result, Failure)` then `.failure()` | narrows to `DupErr \| MissingErr` |
| `match error:` over closed union + `assert_never` | passes |
| same, one arm removed | `error[type-assertion-failure]: Inferred type is MissingErr & ~DupErr` |

Exhaustiveness is real, but it belongs on the **error union**, not on the
`Result` container. That distinction is what makes this change compatible with
the existing `no-match-on-result` gate instead of in conflict with it.

## What Changes

- **BREAKING** — `app/shared/result/` gains `ErrorKind` (7 members),
  the `FeatureError` Pydantic base with `kind`/`code`/`retryable` as `ClassVar`,
  and `STATUS_BY_KIND`. `code` and `kind` stop being constructor parameters, so
  a mistyped code becomes unconstructible rather than merely discouraged.
  This extends the package that already owns the shared error vocabulary rather
  than adding a second one: `shared/result/errors.py` holds `AppError` and five
  subclasses that already carry `kind` as a `Literal` field, with the same five
  values, across 123 construction sites. The migration moves `kind` from a
  per-subclass field to a `ClassVar` on flat siblings and adds the two members the
  old hierarchy could not express — it does not introduce the concept.
  `ErrorKind` carries two members beyond the five agreed at scoping —
  `AUTHENTICATION` (401) and `AUTHORIZATION` (403) — because `auth/service.py`
  raises `UnauthorizedException` at 16 sites and the locked scope converts that
  service to Result-typed. Five members cannot express 401 or 403, so a failed
  login would render 422. See `design.md` D4.
- **BREAKING** — every feature gains `features/<name>/errors.py` declaring its own
  `<Feature>Code` StrEnum, its flat sibling error types, a closed
  `type <Feature>Error = A | B | C` union, and `type <Feature>Result[T]`.
  No feature imports another feature's error types or codes.
- **BREAKING** — repositories roll back inside the `except` block before returning
  `Failure`. This closes the poisoned-commit path and is the precondition for
  services no longer needing to raise.
- Routers render a `Result` through one shared `render_result(...)` that emits the
  `http_error` envelope **and** sets the real HTTP status from
  `STATUS_BY_KIND[error.kind]`. Today `http_error` writes the status into the body
  only, so returning it from a route yields HTTP 200 with `"success": false`.
- Every layer that is exception-native in its own right is **named** and given a
  written adapter contract plus an enforcement rule — WebSocket sessions, Celery
  tasks, both tenacity boundaries, FastAPI auth dependencies, LangGraph nodes,
  MCP handlers, the circuit breaker, SSE streaming, beat cron, scripts, alembic.
  These are not converted; they are classified, so no file is left undefined.
- The ten non-feature directories — `connections/`, `lifecycle/`,
  `middleware/`, `shared/`, `utils/`, `app/api/`, `app/config/`, `app/examples/`,
  `src/database/` and `src/tasks/` — are brought into the contract by role rather
  than by directory. Shared third-party wrappers own unions and return Results; the
  cache layer stops reporting Redis failures as database failures; the session
  dependency is pinned to its current shape and forbidden from inspecting Results;
  the global exception handler's `isinstance` dispatch over framework types is
  explicitly exempted from the union rules and its split registration protected; the
  lifespan's named-family degradation is made the reference for widening a
  dispatcher; every family caught nowhere is either re-rooted or named; a raise that
  satisfies a framework contract is exempted by name rather than by accident; a
  deliberate degradation in a task body must state why the failure is survivable; and
  the example directory loses the exemption it has been holding by habit — it
  currently carries **4 live `error`-level violations of the project's own
  `no-raw-httpexception` rule**, in code whose purpose is to be copied.
- `.opencode/instructions/EXCEPTION-RULES.md` and `RESULT-PATTERN.md` are
  rewritten, and the drifted `.kiro/steering/` copies plus the public
  `docs-site/` page are brought into lockstep. The four doc surfaces currently
  teach three different rules; after this change they teach one.
- The governance contradiction is resolved. `openspec/config.yaml`,
  the `spec-gated` review instruction, and the `no-match-on-result` gate say
  `isinstance`; the deployed `pattern-matching-standard` spec says `match`/`case`
  SHALL be used and `isinstance` SHALL NOT; `.kiro/` says `isinstance` then
  `raise`, which a third gate already calls retired. The code follows
  `isinstance` + `http_error` (122 unwrap sites; **zero** match-on-Result). The
  spec and the `.kiro/` copies are the stale artifacts and are corrected here.
- `subscriptions` is migrated end to end in this change as the executable
  exemplar every per-feature change is measured against.

## Scope / Non-Goals

**In scope:** the shared spine (`app/shared/result/`, `mappers.py`,
`utils/exceptions.py`, `utils/codes.py`, the global exception handler), the
repository rollback fix, the router renderer, the boundary classification and its
gates, the instruction docs and all their copies, and the `subscriptions`
feature as exemplar.

Also in scope, by role rather than by directory — the ten non-feature trees:

| Directory | What this change does to it |
|---|---|
| `connections/` | pins `get_postgres_db`'s shape as the rollback counterparty; re-roots or names `CircuitBreakerOpenError` and `IdempotencyLockError`; resolves the never-raised `TaskDispatchError` family |
| `lifecycle/` | codifies `lifespan.py`'s named-family degradation as the reference pattern; no rewrite |
| `middleware/` | exempts the global handler's `isinstance` dispatch from the union rules and protects its split registration; classifies the health probes and `server_middleware`'s catch-all |
| `shared/` | `shared/services/` (4 modules, 31 raises) becomes Result-typed with per-module unions, and must land **before `crawler`**, its first consuming feature; `shared/crawler/` follows the same shape at 9 sites; `shared/rag/` converts only its `_provider_failure` boundary — its 7 `ImportError` guards are capability detection, not error handling; the `RuntimeError`-rooted memory and cognee families are re-rooted or named |
| `utils/` | `exceptions.py` and `codes.py` are frozen as vocabulary; `cache/redis_func.py`'s 27 `DatabaseException` raises are reclassified as cache failures, with `examples/redis_examples.py` corrected alongside |
| `app/api/` | `generation_with_cb.py`'s `except Exception` → `ServiceUnavailableException` relabel is replaced by named provider classification, so the circuit breaker stops counting the project's own defects as upstream outages; `strict_envelope.py`'s validator `ValueError` is exempted as a framework contract; `v1.py`/`v2.py` are recorded as handling nothing |
| `app/config/` | `settings.py:473`'s validator `ValueError` is exempted as a framework contract and explicitly protected from conversion — a `Result` there would make an invalid field validate successfully |
| `app/examples/` | the **8 error-handling rules ruff is told to ignore for this path** (`BLE001`, `E722`, `B904`, `TRY201`, `TRY300`, `TRY301`, `TRY400`, `S112`) are removed from `per-file-ignores`, which is why a green `ruff check` here has never been evidence; the 4 `raise HTTPException(status_code=500, …)` violations in `redis_examples.py` are corrected; its 8 `except DatabaseException` catches follow the cache reclassification in the same change; and `rag_agent_advanced.py`'s 9 named `(OpenAIError, GoogleAPIError)` handlers are confirmed as the endorsed form and left alone |
| `src/database/` | `seeders/run_seeders.py:81`'s loop catch must name the failing seeder and report a failing exit status; `__init__.py:37`'s PEP 562 `AttributeError` is exempted as a framework contract; `base.py` and `schemas/` handle nothing |
| `src/tasks/` | the reason-carrying `# noqa: BLE001` form becomes the written rule, since **55 of the repo's 62 such sites already use it**; the 3 bare suppressions in `billing_tasks.py` and the 12 unsuppressed broad catches must name their families or their reason; `pageindex_tasks.py:30`'s `NotImplementedError` stub is excluded; the near-identical `auth_email_tasks.py` / `auth_email_tasks_typed.py` pair is reconciled to one |

**In-scope follow-on migration program:**

- Fourteen feature conversions, each delivered as its own OpenSpec change in this
  order: `audit` → `crawler` → `users` → `ingestion` → `dunning` → `profile` →
  `plans` → `invoices` → `payments` → `webhooks` → `agent_saul` → `credits` →
  `documents` → `auth`. Their old exception classes die in the same change that
  replaces their last call site; no feature carries a dual system.
- The complete feature arithmetic is **18 = 1 exemplar + 14 conversions + 2
  no-ops + 1 classify-only**. `subscriptions` is the exemplar. `chat` and `search`
  are no-ops: `chat` has no error-handling surface, while `search` is a tombstone
  whose implementation moved into `documents`. `health` is classify-only because
  its probes degrade to response data and own their 200/503 transport status;
  converting them to `Result` would change that contract.
- `shared/services/{storage,tavily,mailer}.py` converts before the feature program.
  `shared/crawler/` converts with `crawler`, and the `shared/rag/` provider boundary
  converts with `documents`, preserving the per-change ownership seams.

**Definition of complete:** the program is complete only when the section 17 gates
are measured and all hold: 15 of 18 features own `errors.py`; there are zero
`AppError` subclasses, constructions, and `app_error_to_exception` call sites; zero
cross-feature error imports; every feature union is closed and exhaustively checked
by `ty`; every enforcement fixture pair passes with its exclusions audited; and the
final totals are independently derived twice with no completed task admitting
"partial", "deferred", or "TODO" work.

**Out of scope — deliberately:**
- `src/mcp_core/` — 19 modules, 23 raises, 10 `except` clauses. Excluded by the
  owner's decision, not by oversight. `result-layer-boundaries` records the
  exclusion so a later reviewer does not report it as a coverage gap; the layer
  table's "MCP tool handlers and middleware" row classifies the *pattern* without
  scheduling the directory.
- Converting any exception-native layer to Result. Those layers get a contract
  and a gate, not a rewrite.
- Rewriting `lifecycle/lifespan.py` or the global exception handler. Both are
  measured to be doing the right thing; this change makes their behaviour a rule so
  it cannot be "simplified" away, and changes no code in them.
- The `shared/langchain_layer/` and `shared/langgraph_layer/` node bodies beyond
  re-rooting their families. Those are exception-native or state-carrying layers and
  are classified, not converted.
- The 2 pre-existing `ty` errors (`app.shared.langchain_layer.agents.memory.setup_types`
  unresolved) and the 2 pytest collection errors they cause. They predate this
  work and are owned by no requirement here.
- The 12 known websocket fixture-drift test failures.
- `src/lynk/` — a separate Go project: 24 `.go` files and **zero** `.py`, so it is
  outside this contract by nature rather than by decision. And `src/alembic/`
  migration bodies.

## Capabilities

### New Capabilities

- `feature-error-contract`: the `FeatureError` base, `ErrorKind`, the per-feature
  `<Feature>Code` StrEnum, the closed-union and flat-sibling rules, the freeze on
  the `AppError` hierarchy for the migration's duration, and the exhaustiveness
  obligation that makes them checkable.
- `result-layer-boundaries`: which layers are Result-typed, which are
  exception-native, and the adapter contract each exception-native layer owes at
  the point it hands off to domain code — including the `try`/`except` rule, the
  construction-site logging rule, and the obligation that every enforcement gate
  be verified against both a permitted and a forbidden construct before its counts
  are trusted.
- `repository-transaction-safety`: a repository that catches a database exception
  rolls back before returning `Failure`, so a swallowed failure can never reach
  `commit()`.
- `http-result-rendering`: routers render a `Result` into the `APIResponse`
  envelope with the correct HTTP status derived from `ErrorKind`.
- `shared-infrastructure-errors`: the rules for the ten directories that are not
  features but carry the machinery every feature's errors travel through — the
  shared third-party wrappers that own unions, the cache layer's classification, the
  session dependency's fixed shape, the global dispatcher's exemption, the startup
  degradation boundary, the obligation that a family rooted outside the project
  base be caught by name or re-rooted, the exemption for a raise a framework
  contract requires, the obligation that a deliberate degradation name its reason,
  the seeder's exit status, the circuit-breaker adapter's named classification, and
  the rule that example code holds no lint exemption of its own.

### Modified Capabilities

- `pattern-matching-standard`: the requirement mandating `match`/`case` on
  `Success`/`Failure` is inverted to match measured `ty` behaviour —
  `isinstance` opens the `Result`, `match` is reserved for the closed error
  union where it actually narrows and where `assert_never` has meaning.
- `typed-exception-handling`: the agent-tools requirement names
  `ToolOutput.fail()`, which was deleted in `5994dd8`. Its scenarios are
  restated against the surviving `ToolResult`, and the third-party catch
  taxonomy is reconciled with the new adapter contract.

## Impact

- **Code, this change:** `app/shared/result/` (`errors.py`, `types.py`,
  `mappers.py`, `logging.py` — extended, not replaced), `utils/exceptions.py`,
  `utils/codes.py`, `utils/http_response.py`,
  `middleware/global_exception_handler.py`,
  `features/subscriptions/*`, and the 9 relational repositories' 74 SQLAlchemy
  handlers for the rollback fix. The 123 existing `*AppError` construction sites
  (72 of them `InfrastructureAppError`) are the migration surface for the
  vocabulary change; they are retired per feature, not here.
- **Code, downstream changes:** the three `shared/services/` classifiers and 14
  feature conversions become Result-typed; `subscriptions` is already the exemplar,
  `chat` and `search` are recorded no-ops, and `health` remains exception-native and
  classify-only. 152 files fall under a named rule.
- **Infrastructure surface, newly in scope:** 148 files across the five original
  directories (`connections/` 12, `lifecycle/` 3, `middleware/` 6, `shared/` 111,
  `utils/` 16), carrying 115 raise sites and 228 `except` clauses. The dense spots are
  `utils/cache/redis_func.py` (69 sites in one file), `shared/rag/` (70 across 23
  files), `shared/services/` (51 across 4), and `lifecycle/lifespan.py` (14 handlers
  naming ~20 distinct types against exactly one bare `except Exception`).
- **The five later-added directories are small and dense in one place:**

  | Directory | `.py` | raises | `except` | Where the work is |
  |---|---|---|---|---|
  | `app/api/` | 5 | 2 | 1 | the single `except Exception` → `ServiceUnavailableException` relabel in `generation_with_cb.py` |
  | `app/config/` | 2 | 1 | 0 | one validator `ValueError`, exempted not converted |
  | `app/examples/` | 4 | 7 | 28 | 25 of the 28 catches and all 4 gate violations are in `redis_examples.py`; the ruff ignore list is the real edit |
  | `src/database/` | 7 | 1 | 1 | one seeder-loop catch; the rest is ORM declaration |
  | `src/tasks/` | 10 | 1 | 17 | 15 of 17 catches owe a reason; 2 already carry one |

  22 files, 12 raises, 47 `except` clauses in total — under a tenth of the
  infrastructure surface above, and no new capability is needed to hold them.
- **Error surface:** 431 error sites, 111 `AppError` constructions, 187
  `APIException` raises, 122 `isinstance(result, Failure)` sites, 5 copies of
  `_repo_failure`/`_repo_error` across 44 call sites.
- **Public API:** error `code` values change wherever a hardcoded literal is
  replaced by an enum member — most visibly `"DB_ERROR"` → `DATABASE_ERROR` at
  56 sites, whose status also corrects from 503 to 500. **BREAKING** for any
  client matching on those strings.
- **Docs:** four surfaces carry the rule, and they currently disagree by
  generation — `.opencode/instructions/{EXCEPTION-RULES,RESULT-PATTERN}.md`
  (isinstance + `http_error`), `.kiro/steering/{EXCEPTION-RULES,RESULT-PATTERN}.md`
  (isinstance + `raise app_error_to_exception` — one generation stale, and it
  teaches the exact pattern `no-raise-app-error-mapper` flags), the public
  `docs-site/architecture/error-and-result-pattern.mdx` (aligned with
  `.opencode/`), and the deployed `pattern-matching-standard` spec (mandates
  `match`, forbids `isinstance` — two generations off). Also
  `openspec/config.yaml`'s context block, the `spec-gated` review instruction,
  and `CLAUDE.md`'s "Key files" line, which points at
  `src/app/shared/response_type.py` — a path that does not exist (the real module
  is `src/app/utils/response_type.py`).
- **Gates:** `no-match-on-result` is rewritten — its pattern matches only the
  argument-less `case Success()` form, so the bound `case Success(value):` form
  passes unflagged and its reported zero violations are not evidence of anything.
  `no-raise-app-error-mapper` is kept and its 34 violations retire per feature.
  New rules cover the closed-union, flat-sibling, `ClassVar`-classification,
  repository-rollback, no-cross-feature-import and `AppError`-freeze rules, plus
  one per classified boundary. Every rule, new or rewritten, must be shown to flag
  what it forbids and spare what it permits before its counts are used.
- **Lint configuration:** `pyproject.toml`'s `per-file-ignores` loses 8 entries from
  `src/app/examples/*.py` (`BLE001`, `E722`, `B904`, `TRY201`, `TRY300`, `TRY301`,
  `TRY400`, `S112`) and 1 from `src/app/examples/rag_agent_advanced.py` (`BLE001`,
  already dead — that file has no blind `except`). Expect ruff to report findings in
  that directory for the first time; they are fixed, not re-suppressed.
- **Tests:** 33 of 68 test files assert on error types, codes, messages or
  envelopes. `tests/unit/middleware/test_error_envelope_is_universal.py` (31
  exception names, 12 `ErrorCode`s, 8 envelopes) is the hardest to keep green
  and must be updated in this change, not deferred.
- **Dependencies:** none new. `returns` and `pydantic` are already present.

## Risks

- **The status correction is observable.** `"DB_ERROR"`/503 → `DATABASE_ERROR`/500
  changes what clients see at 56 sites. → Land it here, in one change, with the
  before/after table in `design.md`, rather than letting it drift feature by
  feature.
- **The rollback fix changes behaviour where a failure is currently swallowed.**
  A request that silently committed a partial write will now roll back. → This is
  the bug being fixed, but it can surface as newly-failing tests that encoded the
  old behaviour; the 21 `webhooks` unwraps are the place to look first.
- **A closed union is only closed if nothing else subclasses the base.** A new
  subclass added outside a feature's `errors.py` and omitted from the union
  silently reopens the hierarchy and `assert_never` stops meaning anything. The
  same hazard applies to the retired hierarchy from the other direction: over 17
  changes, adding to `AppError` is locally reasonable in every unmigrated feature
  and collectively makes it grow while it is supposed to be retiring.
  → Two enforcement rules, not conventions: one closing each hierarchy.
- **The design rests on gates, and one of the existing gates does not work.**
  `no-match-on-result` reports zero violations because its pattern looks for a
  form nobody writes, not because the codebase is clean. Since a closed union is
  closed only because a rule says so, an unverified gate reads as coverage while
  providing none. → Every gate is verified against a permitted and a forbidden
  fixture before its counts are cited; this is a requirement, not a practice.
- **`auth` is scheduled last and is where the design was weakest.** The
  five-member `ErrorKind` could not express 401 or 403, which would have surfaced
  16 changes in. → Found by measurement during review and fixed before any code;
  `ErrorKind` ships with 7 members. Recorded here because the near-miss is the
  argument for auditing the plan against the ground rather than the reverse.
- **`ty` verifies exhaustiveness but not shadowing.** A broader arm before a
  narrower one is undetectable statically. → The flat-sibling rule is gated, and
  the 28 existing chains are flattened as their features migrate.
- **Baseline is not green** (ty 2, ast-grep 4 errors + 34 warnings, pytest
  400/439 with 2 collection errors). → `design.md` records the baseline so
  reviewers can tell a regression from an inheritance.
- **Two features are unreachable.** `crawler` and `ingestion` routers are mounted
  in neither `api/v1.py` nor `api/v2.py`, so their 4 endpoints cannot be verified
  end to end. → Their changes note it; mounting them is not in scope here.
- **The infrastructure scope was found by the author of the request, not by the
  plan** — twice. A classification keyed on `features/` looked complete while missing
  four third-party wrappers that are structurally identical to a repository, the
  dispatcher every error passes through, and the session dependency the rollback
  requirement exists to protect. Then a scope keyed on `src/app/{features,connections,
  lifecycle,middleware,shared,utils}` looked complete while missing five more trees,
  one of which holds live gate violations. → The classification is now keyed on
  *role*, and `result-layer-boundaries` carries a scenario stating that not living
  under `features/` is not grounds for exemption, plus one naming the two trees that
  *are* excluded so their absence is a recorded decision rather than an implied
  omission. The general lesson is recorded because a directory-shaped assumption hid
  live error-handling code twice in the same change.
- **A suppression list can make a gate report success.** `ruff check
  src/app/examples/` passes today because 8 error-handling rules are disabled for that
  path in `per-file-ignores`, while `ast-grep` — which has no per-path ignore — reports
  4 errors in the same files. This is ADR-005's hazard in a second form: the first was
  a rule whose pattern matched nothing, this is a working rule pointed away from the
  code. → The 8 entries are removed in this change, and ADR-005's obligation is read
  to include *checking what a passing gate was configured to skip*, not only whether
  its pattern works.
- **`shared/` is large and unevenly relevant.** 111 files, but the error density is
  concentrated in four places and much of the rest is graph nodes and LLM plumbing
  that is correctly exception-native. → Scope is stated per subtree in the table
  above rather than as "all of `shared/`", so a later reader cannot read the
  directory name as a mandate to convert the LangGraph layer.
