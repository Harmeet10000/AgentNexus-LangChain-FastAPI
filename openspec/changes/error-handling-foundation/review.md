> Change class: **L** — reviewed as a full checklist.
> Reviewed: `proposal.md`, 6 spec files (4 ADDED capabilities, 2 deltas), `design.md`.
> Findings below were verified against the codebase, not inferred from the artifacts.

## Completeness

**Requirement/scenario coverage:** every requirement carries at least one
scenario; `openspec validate error-handling-foundation --strict` passes. Counts:
`feature-error-contract` 5/17, `result-layer-boundaries` 6/19,
`repository-transaction-safety` 3/8, `http-result-rendering` 5/16,
`pattern-matching-standard` 3 MODIFIED + 1 REMOVED + 2 ADDED,
`typed-exception-handling` 3 MODIFIED + 2 ADDED.

**MODIFIED deltas verified against the archive-replacement semantic.** All 7
MODIFIED/REMOVED headers resolve exactly against their main specs. Scenario sets
were diffed: 3 scenarios do not carry forward, each deliberately replaced because
its behaviour changed — `Unique violation catches UniqueViolationError` and
`Client misuse catches InterfaceError` (both said "raises", which the design
forbids in a repository), and `Outbox relay dead-letters on any failure` (false —
see Correctness 3). No scenario is lost silently.

**Gaps found:**

1. **No requirement governs `AppError` during the migration.** The specs forbid
   annotating methods with `AppResult[T]`, but nothing stops a *new* `AppError`
   subclass being added while 16 features still use the old system. The change
   spans 17 openspec changes; over that window the hierarchy this change exists to
   retire can legitimately grow, and every addition makes the last feature harder.
   Needs a requirement freezing `AppError` to its existing subclasses.

2. **No scenario covers the enforcement rules' own correctness.** `design.md` D12
   flags that `no-match-on-result` does not enforce what its message claims, and
   commits to a rewrite — but no spec requirement makes rule correctness
   observable. Given that this change adds six new rules and the whole design
   leans on them (a closed union is closed only because a gate says so), the gates
   need a testable obligation.

## Correctness

3. **CONFIRMED — the deployed spec being modified contains a false scenario.**
   `typed-exception-handling` asserts the outbox relay "dead-letters on any
   failure … catches `Exception`". Measured: `src/app/shared/outbox/relay.py:139`,
   the line that actually dead-letters, catches `(CeleryError, PostgresError)`.
   The `except Exception` pair at `:76` and `:90` are the scan and listen loops,
   which dead-letter nothing. The delta corrects this and names the partiality.
   Correctly handled.

4. **CONFIRMED — the `ty` claims in the artifacts are measured, not asserted.**
   Re-verified during review: `ClassVar[SubscriptionCode] = "DUPLICATE_SUBSCRIPTION"`
   produces `error[invalid-assignment]` even though the string value is *correct*,
   and so do a typo and a member of the wrong enum — 3 diagnostics, with the
   enum-member form clean. The guarantee is therefore stronger than
   `feature-error-contract` claims: a code cannot be written as a string at all,
   right or wrong. Worth stating that way in the spec.

5. **MUST FIX — `ErrorKind` cannot express 401 or 403, and the design regresses
   the security boundary.** `STATUS_BY_KIND` maps its five members to 422, 404,
   409, 502, and 500/503. There is no route to 401 or 403.

   This is not theoretical. `src/app/features/auth/service.py` raises
   `UnauthorizedException` at **16 sites** ("Invalid credentials", token invalid,
   token expired, session owner mismatch). The locked scope converts every
   feature's service to Result-typed, so all 16 become typed failures rendered by
   `render_result` — which, as specified, can only give them 422, 404, or 500.
   The old system covers this correctly: `ErrorCode` has `UNAUTHORIZED`,
   `FORBIDDEN`, `INVALID_TOKEN`, `TOKEN_EXPIRED`, `REFRESH_TOKEN_INVALID`, and
   `UnauthorizedException`/`ForbiddenException` carry 401/403.

   Shipping the five-member enum would mean the auth feature either cannot
   migrate, or migrates by answering failed authentication with 422 — a
   correctness and security regression, on the one feature the plan schedules last
   and therefore discovers latest.

   The three ways out: add `AUTHENTICATION` → 401 and `AUTHORIZATION` → 403;
   or carve `auth` out of the Result conversion (contradicts the locked scope);
   or let an error carry an explicit status (contradicts D6, and reintroduces the
   per-endpoint drift being removed). Only the first is consistent with the rest
   of the design, and it keeps boundary dispatch fixed-width — seven arms instead
   of five, still independent of how many feature error types exist.

6. **MUST FIX — `result-layer-boundaries`' "Every source file is classified" is
   unsatisfiable as written.** The requirement says every file under `src/` falls
   under exactly one row, and the scenario makes a file matching no row a gap to
   close. But all 17 rows describe *error-handling* behaviour, and a large part of
   the tree — DTOs, schemas, ORM models, config, enums, pure helpers — handles no
   errors at all and matches no row. As written the requirement can never be
   satisfied, which makes it unusable as a gate and invites the reader to
   rubber-stamp it.

   The user's scope decision was that every file has a *rule*, not that every file
   returns a Result. Scope the requirement to files that construct, catch,
   propagate, or render an error, and say explicitly that a file doing none of
   those is out of scope.

7. **Testability.** `repository-transaction-safety`'s scenario "The rule is
   verifiable across the codebase" states that a count "is reportable" — that is a
   property of a tool, not observable system behaviour. Either tie it to the
   named gate or drop it; the requirement's other two scenarios already carry the
   obligation.

8. **Delta operations are correct.** REMOVED + ADDED for *"Service-layer Result
   unwrapping uses match/case"* rather than MODIFIED is the right call: a MODIFIED
   block must keep its header byte-identical, which would leave a header asserting
   `match` above a body mandating `isinstance`, permanently, in the archived spec.
   The REMOVED block carries both **Reason** and **Migration**, and the Migration
   accounts for the two scenarios it drops.

## Standards

Checked against `.opencode/instructions/`:

- **RESULT-PATTERN** — the artifacts adopt the *current* `.opencode/` rule
  (`isinstance` + envelope, no raise) and extend it. They correctly do **not**
  follow `.kiro/steering/RESULT-PATTERN.md`, which is one generation stale and
  teaches `raise app_error_to_exception(error)` — the pattern
  `no-raise-app-error-mapper` already flags at 34 sites. The change updates that
  copy rather than leaving it to contradict the new rule. Good.
- **EXCEPTION-RULES** — the response envelope stays `APIResponse` + `http_error()`;
  D6 adds the transport status without replacing the envelope. `e.add_note()`
  obligations in the `typed-exception-handling` delta are preserved verbatim.
- **ARCHITECTURE-RULES** — layering is respected: rollback is placed in the
  repository (D8) with the two alternatives rejected on stated grounds.
- **PYTHON-TYPING-RULES** — PEP 695 `type` aliases and `ClassVar` are used as the
  project does; no `# noqa` or `# type: ignore` is introduced, and per-feature exit
  criterion 7 forbids reaching green with a suppression.
- **No `match`/`case` on `Success`/`Failure`** — satisfied, and the review
  instruction's own check is what the `pattern-matching-standard` delta brings the
  deployed spec into line with.
- **Secrets** — no `SecretStr` surface is touched.

**INFO —** the agreed renderer sketch names its parameter `status_code`, which
reads as the failure status but means the success status. `success_status` removes
an ambiguity that will otherwise be misread at 67 call sites.

## Risk

- **Security boundary (see 5).** Blocking. The gap surfaces at the *last*
  scheduled feature, so shipping the enum as-is buys 16 changes of false
  confidence before the problem appears.
- **Breaking change is correctly localised.** `"DB_ERROR"` → `DATABASE_ERROR` with
  503 → 500 at 56 sites lands in one change with a before/after record, rather
  than drifting across 17 releases. The status correction is the more important
  half: clients are currently told to retry a dead transaction.
- **Data integrity.** The rollback fix is the right shape and the alternatives are
  rejected for the right reasons. Landing all 9 relational repositories here rather than per
  feature is correct — it is error-type-independent, and staging it would leave
  poisoned-commit paths open for the duration of the migration. `tasks.md` should
  say this explicitly so it is not re-opened later as churn.
- **Gate correctness is load-bearing (see 2).** The rewritten
  `no-match-on-result` must reject `case Success(value):` while accepting
  `case SubscriptionNotFoundError():` — the design endorses the second and
  forbids the first, and a careless pattern would flag both. Verify against a
  fixture holding both forms before trusting the "zero match-on-Result" count.
- **Baseline is honestly recorded** (ty 2, ast-grep 4 + 34, pytest 400/439 with 2
  collection errors) and scoped out of this change's ownership. Re-verified during
  review: ast-grep reports exactly 34 `no-raise-app-error-mapper` and 4
  `no-raw-httpexception`, matching `design.md`.
- **`extra="forbid"` moves a stale `code=` keyword to a runtime `ValidationError`
  raised inside an `except` block**, where it can mask the original error. Mitigated
  in design (ty catches it statically first), and acceptable.

## Must-fix list

1. Add `AUTHENTICATION` (401) and `AUTHORIZATION` (403) to `ErrorKind`; update
   `STATUS_BY_KIND`, the member list in `feature-error-contract`, and the mapping
   in `http-result-rendering`. Record the deviation from the five-member design
   and why the ground forced it.
2. Scope `result-layer-boundaries`' file-classification requirement to files that
   construct, catch, propagate, or render an error; state that files doing none of
   those are out of scope.
3. Add a requirement freezing `AppError` to its existing subclasses for the
   duration of the migration.
4. Add a requirement making the enforcement rules' own correctness testable, and
   name the `case Success(value):` vs `case <ConcreteError>():` discrimination as
   the case that must be verified.
5. Strengthen `feature-error-contract`'s code-typing scenario to what was measured:
   a bare string is rejected even when its value is correct.
6. Rename the renderer's success-status parameter; drop or re-anchor
   `repository-transaction-safety`'s "is reportable" scenario.

**VERDICT:** CHANGES-REQUESTED — items 1–4 are blocking (1 is a security-boundary
regression; 2 makes a requirement unsatisfiable; 3 leaves the retired hierarchy
free to grow across 17 changes; 4 leaves the design's only enforcement mechanism
unverified). Items 5–6 are non-blocking wording fixes. Do not write `tasks.md`
until 1–4 are fixed in the artifacts.

## Resolution

All six items were fixed in the artifacts before `tasks.md` was written. What
changed:

1. **`ErrorKind` → 7 members.** `AUTHENTICATION` (401) and `AUTHORIZATION` (403)
   added. Updated in `feature-error-contract` (member list plus two scenarios
   distinguishing 401 from 403), `http-result-rendering` (mapping, and a scenario
   asserting that converting `auth`'s service to Result does not change the status
   a client sees), `design.md` D4 — which now records the deviation from the agreed
   five-member set, the 16 measured raise sites that forced it, and the three
   rejected alternatives — plus D5, the proposal's What Changes, and its Risks.
2. **File classification scoped.** `result-layer-boundaries` now applies to files
   that construct, catch, propagate, or render an error, with an explicit scenario
   putting DTOs, schemas, models, settings and pure helpers outside it.
3. **`AppError` frozen.** New requirement in `feature-error-contract`: no new
   subclass and no new field for the migration's duration, with a scenario making
   the subclass count monotonically decreasing and zero the completion signal.
   `design.md` D13 records why deprecation-in-place fails here. Gate
   `no-new-app-error-subclass` added to D12.
4. **Gate correctness made a requirement.** New requirement in
   `result-layer-boundaries`: every new or changed rule is verified against a
   fixture holding both the forbidden construct and the nearest permitted one, and
   a count from a corrected rule is re-measured rather than carried forward. It
   names the `case Success(value):` versus `case <ConcreteError>():`
   discrimination explicitly. `design.md` D14 records the reasoning.
5. **Code-typing scenario strengthened** to what was measured: a bare string is
   rejected even when its value is correct, so a code cannot be spelled by hand at
   all. A second scenario covers a member of another feature's enum.
6. **`success_status`** replaces `status_code` in the renderer's signature, with a
   requirement clause and scenario forbidding the ambiguous name.
   `repository-transaction-safety`'s "is reportable" scenario is re-anchored to the
   gate's output on a migrated feature.

Re-verified after the fixes: `openspec validate error-handling-foundation
--strict` passes; 32 requirements and 120 scenarios across 6 spec files; all 7
MODIFIED/REMOVED headers still resolve; the scenario diff still shows exactly the
3 deliberate replacements and no silent loss.

**VERDICT:** APPROVED — items 1–6 resolved in the artifacts and re-verified.
`tasks.md` may be written. Two things carry forward into implementation as
must-verify rather than must-decide: the rewritten `no-match-on-result` has to
reject `case Success(value):` while accepting a concrete-type arm, and the 401/403
statuses for `auth` have to be asserted end to end when that feature migrates,
since that is the one place this change could silently regress a security
boundary.

---

## Second pass — infrastructure scope

Scope was widened after the verdict above, at the request's author's direction, to
name `connections/`, `lifecycle/`, `middleware/`, `shared/` and `utils/`. The
approval above does not cover that material, so it was reviewed on the same four
axes. Reviewed: the new `shared-infrastructure-errors` capability (7 requirements,
24 scenarios), the edits to `result-layer-boundaries` (classification table, two
requirement bodies, one scenario), `repository-transaction-safety`'s session-dependency
citation, `design.md` D15–D18, and the proposal's Why, What Changes, Scope, Impact
and Risks.

### Completeness

The gap was real and it was not a wording gap. Four third-party wrappers under
`shared/services/` — 31 raises of `APIException` subclasses, 20 catches every one of
a library type, no `Result` — are structurally identical to a feature repository and
were outside a classification that called itself total. So were the dispatcher every
error passes through and the session dependency `repository-transaction-safety`
exists to protect. A classification keyed on `features/` could not have found them.
The fix is to key it on role, and `result-layer-boundaries` now carries a scenario
stating that not living under `features/` is not grounds for exemption.

Totals after the widening: 39 requirements, 149 scenarios across 7 spec files.
`openspec validate error-handling-foundation --strict` passes. WHEN and THEN bullets
pair at 149 each. No scenario header carries the wrong hash count. The one
requirement with zero scenarios is the REMOVED block, which carries Reason and
Migration instead — correct for the delta format.

### Correctness

Four claims in the first draft of this material were asserted from partial
measurement and are corrected. All four were found by re-measuring rather than by
re-reading, which is the only method that has worked on this change.

9. **The `TaskDispatchError` family was described backwards.** The draft called it
   "correctly rooted at `CeleryError` with **zero raise sites** — a declaration
   implying handling that never happens." The count came from grepping
   `raise TaskDispatchError`, which the base never is. Its two concrete subclasses
   `UnregisteredTaskError` and `TaskPayloadValidationError` are each raised twice in
   `connections/celery_registry.py`. The family is not a defect — it is the second
   *correct* resolution, reachable through its root by the relay's existing
   `except (CeleryError, PostgresError)` without any clause naming it.

   This mattered beyond the fact. A rule written from the draft — "flag any family
   with no catch clause naming it" — would have reported all three classes as
   findings, and the endorsed pattern would have been gated against. Reachability is
   now defined over ancestors, not exact names, with a scenario for an abstract base
   that is deliberately never raised.

10. **`shared/rag/` was not a third-party classifier and should not be converted
    wholesale.** The draft's layer table put it alongside `shared/services/`. Of its
    ~48 handlers, 17 catch bare `Exception` and **7 catch `ImportError`** as
    optional-dependency guards; the genuine provider classification is confined to a
    `_provider_failure` helper at 4 sites. It also raises builtins — `ValueError` ×6,
    `TypeError`, `FileNotFoundError` — against 3 `APIException`-family raises. Only
    the provider boundary owes a union.

    This surfaced a category the classification was missing entirely: an
    `except ImportError` that sets an availability flag is **capability detection, not
    error handling**, and owes no classification. It now has its own row and its own
    scenario, which also keeps every future gate from firing on optional-backend
    probes.

11. **`shared/services/` has a hard ordering constraint the migration plan did not
    have.** `rate_limiter` is imported by `crawler` — the *third* feature in the
    order — and `storage` by `profile`, `invoices` and `documents`. A feature
    migrating while its shared dependency still raises `APIException` would have to
    `try`/`except` around a call the project owns, which
    `result-layer-boundaries` forbids. The plan had `shared/services/` merely
    "after the foundation". It is now Phase 1a, explicitly before any feature, with
    the reason recorded.

12. **The two `except Exception` rules read as contradictory.**
    `typed-exception-handling` preserves `except Exception` at degradation
    boundaries; the new cache requirement forbids it in the cache helpers. Both are
    right, and the distinction is now stated: a degradation boundary catches broadly
    and *degrades* — fallback, `None`, re-enter the loop — while
    `redis_func.py` catches broadly and *escalates*, converting a `TypeError` into a
    500 labelled `DATABASE_ERROR`. Catching broadly to keep serving is endorsed;
    catching broadly to relabel is not. A scenario now carries it, because a
    reviewer reading only one of the two files would otherwise have to guess.

13. **Confirmed, not corrected — the session dependency.**
    `src/app/connections/postgres.py:241` reads exactly as
    `repository-transaction-safety` assumed: yield, `await session.commit()`,
    `except Exception: rollback; raise`, `finally: close`. A returned `Failure` is
    not an escaping exception, so the swallowed-failure path reaches `commit()`. D16
    now records why the obvious fix — teach the dependency about Results — is
    unimplementable rather than merely undesirable: at the point it commits it holds
    no `Result` and has seen no exception. Worth writing down because it is the first
    idea every reader has.

14. **Confirmed — the dispatcher exemption is necessary, not defensive.**
    `middleware/global_exception_handler.py` contains **zero `except` blocks**; it is
    invoked by registration. Any gate that locates error handling by matching
    `except` is blind to the most important error-handling file in the repository, and
    D14 requires it to say so instead of reporting a clean sweep. Its lines 166–200
    also record, in detail, why the registration is split — Starlette routes the
    `Exception` key to a different middleware than every other key, and FastAPI's
    `setdefault(HTTPException, ...)` made the MRO walk resolve three classes early so
    the `APIException` branch never ran — and end with an instruction not to simplify
    it. D17 turns that comment into a rule.

15. **`auth/repository.py` is a document store, and two claims rested on it not
    being one.** Every artifact said "11 repositories catch `SQLAlchemyError`" and
    "the 56 `DB_ERROR` sites correct from 503 to 500". Measured: of 11 repository
    modules holding 12 classes, **9 are relational** with 74 SQLAlchemy handlers,
    `users/repository.py` catches nothing, and `auth/repository.py` is MongoDB and
    Redis — 13 `PyMongoError`/`DuplicateKeyError` handlers, 6 `RedisError`, zero
    SQLAlchemy, zero `session` writes.

    Two consequences, both material:

    - **The rollback fix touches 9 modules, not 11.** MongoDB has no request-scoped
      transaction under `get_postgres_db`, so there is nothing to roll back. The
      requirement claimed to apply "to every existing `except` block"; it is now
      scoped to the relational modules, with a companion requirement stating what a
      document-store repository owes instead — classification, and explicitly *not* a
      rollback, so it cannot be read as exempt.
    - **The 503 → 500 correction is right for 49 sites and wrong for 7.** `auth`'s 7
      `"DB_ERROR"` literals describe Mongo or Redis being unreachable, which is
      genuinely retryable. Sweeping all 56 into one correction would have relabelled
      a transient outage as a permanent failure **on the login path** — the same
      feature the first review pass found was one enum short of a 401. A scenario
      now pins the retryable classification for those 7.

    The root cause is the same as finding 9's: a count taken from one grep pattern
    (`repository.py`, `SQLAlchemyError`) and generalised to a claim about a
    population. `credits/` also holds two repositories under a `repositories/`
    subdirectory that the file-name pattern missed. D8 now records the per-store
    split rather than treating "repository" as one kind of thing.

### Standards

- The new capability follows `feature-error-contract`'s rules rather than inventing
  parallel ones: per-module StrEnum, flat siblings, closed union, no cross-module
  code imports. D15's rule — the module that classifies owns the union — subsumes
  the feature case rather than special-casing `shared/`.
- `ErrorKind` is unchanged. The shared modules classify into the same seven kinds,
  so boundary dispatch stays fixed-width and the review's item-1 fix is not
  disturbed.
- No requirement here asks for a `# noqa` or `# type: ignore`, and none proposes
  rewriting `lifespan.py` or the global handler — both are measured to be correct,
  and the change makes their behaviour a rule while touching no code in them.

### Risk

- **The widening is bounded by role, not by directory.** `shared/` is 111 files, and
  "in scope" could be misread as a mandate to convert the LangGraph layer. The
  proposal states scope per subtree and D15's rule excludes the graph nodes by
  construction. This is the residual risk most worth re-checking at implementation.
- **`utils/cache/redis_func.py` is reachable from no request path** — only
  `utils/cache/__init__.py` and `examples/redis_examples.py` import it. Its 27
  `DatabaseException` raises for Redis failures are a latent misclassification and a
  bad exemplar, not a live 500. Scheduled, but it should not be carried in the risk
  register as a production fault, and this is stated in both design and proposal.
- **Five families are caught nowhere.** `CircuitBreakerOpenError`,
  `IdempotencyLockError`, `AgentMemoryError`, `CogneeSetupError` and
  `StateSchemaVersionError` all have raise sites and zero catch sites. On a request
  path they reach the unhandled-500 branch; in a worker, nothing. This is the
  highest-severity finding in the infrastructure scope and it lands in Phase 1.
- **The blind spot was found by the request's author, not by the plan.** That is the
  second directory-shaped assumption in this change to have hidden live
  error-handling code. Recorded as a risk in the proposal so the pattern is
  available to whoever writes the per-feature changes, rather than being
  rediscovered a third time.

**VERDICT:** APPROVED — the infrastructure scope is coherent with the four
capabilities approved above, and the five errors found in its first draft are
corrected in the artifacts and re-validated. Carried forward into implementation:
the family-reachability gate must be checked against `celery_registry.py`'s
correctly-rooted family before its counts are trusted (D14 applied to D18's own
rule); `shared/services/` must land before `crawler` or the try/except rule is
violated by construction; and the `"DB_ERROR"` migration must be split 49/7 by store,
because the two halves correct in opposite directions.

**Method note.** Five of the five errors in this pass came from generalising a count
taken with one grep pattern. Findings 9 and 15 are the same mistake twice — `raise
TaskDispatchError` missed the subclasses, `-name 'repository.py'` missed
`credits/repositories/`, and `SQLAlchemyError` treated a Mongo repository as
relational. The rule that would have caught all five: before a count becomes a claim
about a population, enumerate the population by a second, structurally different
query and reconcile the two. That is D14's principle applied to measurement rather
than to gates, and it belongs in the per-feature changes' method as well.



---

## Third pass — the spine's own ground, before tasks

Written while enumerating concrete task targets. Four findings; two are corrections
to approved artifacts, one is a spec gap that would have fired a gate on correct
code, one is a live defect the design now inherits.

16. **Corrected — the shared spine is not new.** Both earlier passes let
    `app/shared/errors/` "is introduced" stand, in the proposal's What Changes, its
    Scope, its Impact, and design Phase 1. `src/app/shared/result/` already holds
    `errors.py`, `types.py`, `mappers.py` and `logging.py`, and
    `result-layer-boundaries` already classifies `shared/result/errors.py` as the
    vocabulary layer — so `feature-error-contract` naming a different package put two
    approved specs in direct conflict about where `ErrorKind` lives. Neither pass
    caught it because both reviewed the plan against itself. Fixed in
    `feature-error-contract` (with two new scenarios), the proposal at three sites,
    design Phase 1, and D19.

17. **The five-kind vocabulary was the existing hierarchy read back.**
    `ValidationAppError`, `NotFoundAppError`, `ConflictAppError`,
    `InfrastructureAppError` and `ExternalServiceAppError` each already declare
    `kind: Literal[...]` — the same five values scoping proposed. This does not change
    a requirement, and it strengthens ADR-003 rather than weakening it: the two
    missing members are missing from the *code*, which is why `auth`'s 16
    `UnauthorizedException` raises route around `AppError` entirely instead of
    classifying into it. It also resizes the work honestly — the migration surface is
    **123 construction sites** (72 `InfrastructureAppError`), not five class
    definitions, and they retire per feature.

18. **Gap closed — a pre-service policy guard was unclassified.**
    `features/crawler/router.py` raises `TooManyRequestsException` at three sites after
    `check_rate_limit` returns `(False, info)`. No `Result` is produced and no
    exception is caught, so it is not error handling — but a "routers render, do not
    raise" gate reads it as three violations in a file that is correct. It is the same
    shape as the `except ImportError` capability-flag row, and it is now a row in the
    layer table plus a scenario. Seven kinds still suffice: 429 never crosses the
    Result boundary, so no eighth kind is needed. Had the gate been written first, the
    likely repair would have been an eighth kind for a construct that owes no
    classification at all.

19. **Sixth instance of the Method-note error — this time in the ordering rationale.**
    Phase 1a justified itself with "`rate_limiter` is imported by `crawler`, the third
    feature in the order". `shared/services/rate_limiter.py` raises nothing and catches
    nothing; it returns `tuple[bool, dict]`. The 31/20 total for `shared/services/` was
    right and the per-module attribution was invented. Measured: `storage` 21/15,
    `tavily` 8/3, `mailer` 2/2, `rate_limiter` 0/0. The conclusion survives — Phase 1a
    must still precede `crawler` — but via **`tavily.py`**, whose `search` the package
    `__init__` re-exports and `crawler/service.py:18` imports. Also measured while
    fixing it: `mailer.py` has no importer outside `shared/services/`, so it blocks no
    feature; 4 of `tavily.py`'s 8 raises are pre-flight argument guards rather than
    boundary classification; and 17 of `storage.py`'s 21 are `ServiceUnavailableException`,
    which keeps its 503 under the new kinds, so that conversion has no observable
    break. All of this is now in the Phase 1a table.

20. **Live defect the design now inherits, not one it creates.**
    `features/ingestion/service.py:86` constructs `AppError(code="UNKNOWN",
    message=str(failure))`. The base declares no `kind` field, and `mappers.py`'s final
    `case AppError():` arm maps it to `ValidationException` — so an unknown internal
    failure is reported to the client as **422**, telling the caller to fix input for a
    fault that is not theirs. It corrects to 500 under `STATUS_BY_KIND`. It is also the
    only object in the codebase that would `AttributeError` on `error.kind`, which is
    harmless today only because **nothing dispatches on `kind` anywhere** — the field
    is declared and never read. The renderer is its first consumer, so this site must
    be fixed before `render_result` ships, not with `ingestion`'s own change.

**VERDICT: APPROVED.** The four original capabilities and the infrastructure scope are
unchanged in substance; this pass corrected where the spine lives, classified one
construct that would have produced three false violations, and repaired the Phase 1a
rationale without changing its conclusion. Added to the carry-forward list: the
`ingestion/service.py:86` kindless instance is a Phase 1 task, not a Phase 2 one.

**Method note, second entry.** Finding 19 is the sixth error of the form the first
Method note named, and finding 16 shows the review procedure's own blind spot: two
passes reviewed the plan for internal consistency and neither opened the directory the
plan proposed to create. Reviewing a plan against itself cannot find a claim that is
coherent and false. The check that found both was mechanical — `ls` the path the plan
says it will create, and read the module it says it will extend — and it belongs at the
top of every subsequent change's review, before any reasoning about the plan's content.

---

## Fourth pass — the five directories the owner added after approval

Scope changed after the third pass: `src/app/api`, `src/app/config`,
`src/app/examples`, `src/database` and `src/tasks` were declared out of scope, recorded
as such in `HANDOFF.md` §9, and then brought back in. `src/mcp_core` stays out by
decision; `src/lynk` is out by nature.

This pass enumerated all five before writing anything — 22 `.py` files, 12 raises, 47
`except` clauses — and found five things worth recording.

### Findings 21–25

**21. `HANDOFF.md` §9 asserted the opposite of the current scope.** It read
"Explicitly out of scope, by the owner's decision — do not touch, do not report as a
coverage gap" and listed all five. A handoff document is the one artifact an
implementing agent is told to trust over its own judgement, so a stale scope boundary
there is worse than a stale count anywhere else. Corrected, with D20 named as the place
that tabulates the dispositions.

**22. `design.md` contradicted itself and `tasks.md` on when `utils/cache/` lands.**
The Migration Plan listed "`utils/cache/` with `examples/redis_examples.py`" under
*Deferred to their own changes*, while the risk register two sections earlier said
"Scheduled with the foundation anyway" and `tasks.md` 7.5 schedules the
reclassification in this change. Two of three said foundation, and the schedule is the
one with a task attached, so the deferral line was the stale one. This was found only
because folding `examples/` in forced the question of *which* change fixes its 8
`except DatabaseException` catches — the contradiction had survived three passes because
nothing else needed the answer.

**23. The `src/tasks/` degradation count was wrong in exactly the way the pass-2 Method
note describes.** A summed grep gave "6 `# noqa: BLE001` with a written reason";
enumerating the sites gave **2** (`agent_memory_tasks.py:72`, `billing_tasks.py:134`),
with 3 more carrying a bare `# noqa: BLE001` and no reason. Seventh instance in this
change of a total being right while its attribution was invented. Reconciled by two
structurally different queries — one keyed on an em-dash after the code, one on
`(—|--) ` — which disagreed at first (54 vs 53) because `processor.py` uses an ASCII
hyphen and `shell.py` writes its reason after a second code. Both resolve to the same
population: **62 sites, 55 with a reason, 7 without** — 4 in
`features/subscriptions/service.py`, 3 in `src/tasks/billing_tasks.py`.

That reconciliation changed the requirement rather than just its numbers. At 55 of 62,
"a deliberate degradation names its reason" is not a rule this change invents — it is
the repository's existing convention, followed by the very file the change designates
as the reference degradation boundary (`lifespan.py:234`). The requirement went from
*proposing* a practice to *writing down* one, and the 7 exceptions became a finite,
named worklist instead of a category.

**24. `app/examples/` is exempted from the error-handling lint rules in writing, and
that is the finding.** `pyproject.toml`'s `per-file-ignores` disables `BLE001`, `E722`,
`B904`, `TRY201`, `TRY300`, `TRY301`, `TRY400` and `S112` for `src/app/examples/*.py`,
with a second block for `rag_agent_advanced.py` whose `BLE001` is already dead. So
`ruff check src/app/examples/` reports "All checks passed!" while `ast-grep scan`
reports 4 `error`-level `no-raw-httpexception` violations in `redis_examples.py` —
ast-grep is simply the only gate there with no per-path ignore.

The first three passes treated the 4 ast-grep violations as the defect. They are the
symptom. The defect is that the directory whose entire purpose is to be copied has been
excused from the rules the copies will be held to, and a reader checking "is
`examples/` clean?" with the project's primary linter gets a green answer.

**25. `BLE001` already encodes the degrade-versus-escalate distinction this change
argues for.** It does not fire on a blind `except` that ends in a bare `raise` —
`middleware/server_middleware.py:100` catches `Exception`, logs with `.exception()`,
re-raises, and needs no suppression. That is the same line D14/the cache requirement
draw between *catching broadly to keep serving* and *catching broadly to relabel*, and
it was already in the linter. Any gate written for the degradation rule has to spare
that shape or it will contradict a tool the project already runs.

### What did not change

The five directories added **no new capability**. Every one of their rules fits an
existing one — `shared-infrastructure-errors` for the five requirements, four table rows
and four scenarios in `result-layer-boundaries`. That is a positive signal about D15's
framing: keying the classification on *role* rather than directory meant a scope
expansion of five trees needed no structural change to the spec, only more rows.

Nothing in `feature-error-contract`, `http-result-rendering`,
`pattern-matching-standard` or `repository-transaction-safety` was touched, and no ADR
was reopened. D20 is additive.

**VERDICT: APPROVED.** The widened scope is coherent with the six ADRs and the seven
capabilities. Section 9's 16 tasks are almost entirely exemption-writing and
suppression-removal rather than conversion, which is why they add a fifth small PR
instead of enlarging any existing one.

**Method note, third entry.** Findings 21 and 22 are a class the first two notes do not
cover: **a scope change invalidates artifacts that were correct when written, and the
handoff document is the most dangerous place for that to happen.** The two rules so far
both concern measuring the *code*. This one concerns measuring the *plan against its own
decisions*: when scope moves, grep every artifact for the boundary it used to state —
`grep -rn "out of scope" .` found finding 21 in one call, and grep for the directory
names found finding 22.

Finding 24 adds a third measurement rule, and it is ADR-005 in a second form. ADR-005
says a gate is not trusted until it is shown to permit and to forbid. That catches a
rule whose *pattern* is wrong. It does not catch a working rule *pointed away from the
code* — `per-file-ignores`, a `ruleDirs` omission, a rule-level path filter. Both
produce the identical artefact: a clean run that reads as coverage. So: **before citing
a gate's zero, read its exclusion list.** Task 10.7 makes this a step in the change
rather than advice.

---

## Fifth pass — 2026-08-31, the plan versus the branch

The previous four passes reviewed the plan against the *code*. This one reviews it
against the *implementation of itself*, which had started while the fourth pass was
being written.

**26. The handoff described an unstarted change, and 26 of its tasks were already
done.** `HANDOFF.md` opened "Your job is the 95 tasks in `tasks.md`". Two things were
wrong. The total is **97** — a per-section sum reconciles (13+7+11+5+8+9+7+7+16+7+7),
and `grep -c '^- \[ \] '` returned 71 because 26 boxes are already `[x]`. And the work
was not prospective: sections 1 (13/13), 2 (7/7) and 4 (5/5) plus task 3.1 are complete
and committed, each with a `> **DONE:**` block naming the sites it touched. An agent
handed that file would have re-audited nine repositories that were already fixed.

The PR table carried the same arithmetic error independently: PR 4 was listed at 25
tasks against sections 6+7+8 = 9+7+7 = **23**.

**27. The five-PR split no longer maps onto the branch history, and could not be
retrofitted.** The table assigns section 1 to PR 1, sections 2+3 to PR 2, and sections
4+5 to PR 3. On the branch, `58422c1` contains sections **2 and 4 together, plus task
3.1** — one commit spanning three planned PRs. This is not a deviation to correct; the
seams the table cut on were dependency seams, and nothing about landing 2 and 4 together
violates a dependency. The table was simply written before the history existed. §3 now
records what landed and splits only the remainder (A–E), which preserves the reviewable-
unit goal without asking anyone to rewrite four commits.

**28. The planning artifacts were untracked, were committed mid-session, and the commit
captured a `tasks.md` that had lost section 9.** When this pass began, `proposal.md`,
`design.md`, `adrs.md`, `review.md`, `specs/` and `HANDOFF.md` were all `??` in
`git status` — only `tasks.md` was tracked, because `0ca39ea` had committed the section 1
ticks. `853fe25 chore(openspec): add planning artifacts and no-match fixture for PR A`
then committed the whole set, which resolves the original finding: the branch now carries
its proposal, ADRs and spec deltas, and `openspec archive` has something to fold.

What it did *not* resolve is worse, and is the reason this finding stays open. The
`tasks.md` that `853fe25` committed is the **79-task, 10-section** version — the
pre-fold-in state. Section 9 (the five later-added directories, 16 tasks), the
renumbering of Verification to 10 and Handoff to 11, and tasks 10.7 and 11.5 were all
absent from it. The specs kept every one of their 45 requirements and 180 scenarios, so
the change still *specifies* the five directories while its task file no longer schedules
any work against them. That is the exact failure mode the openspec archive hazard
describes, arriving through a different door: a delta that validates while the plan
behind it has been silently narrowed. Restored and re-committed; verify with
`grep -n '^## ' tasks.md` that section 9 reads "The five later-added directories" and
that the total is 97.

The mechanism matters for anyone working this branch. Two workers were editing the change
concurrently — one implementing, one planning — and `git status` is the only thing that
distinguished "my edit is safe" from "my edit is the uncommitted side of a file someone
else is about to commit from a stale tree." An untracked artifact survived; the tracked
one was rewound.

**29. Task 3.1's deferred re-measure is now done, and the answer is a real zero.** 3.1's
own `DONE` block said "DONE (partial…) — full violation re-measure deferred". Completed
here: the corrected `no-match-on-result` reports **0 violations in `src/`, 0 in
`tests/`**, and `rg 'case\s+(Success|Failure)\s*\('` over the same trees independently
returns 0. Both queries agree, so this is the honest zero, not ADR-005's "the rule looked
for something nobody writes" — nothing in the codebase matches on a `Result`, which is
what ADR-002's 122 `isinstance` unwrap sites predicted.

The rule's behaviour was also verified two-sided against a probe: it flags
`case Success(value)`, `case Failure(err)`, `case Success()` and `case Failure()`, and
spares `case SubscriptionNotFoundError():` and `case SubscriptionConflictError(code=c):`
— exactly ADR-002's stated obligation. But the probe was a throwaway in `/tmp`, so the
repo still holds nothing that would catch a future re-narrowing of the regex. That is
task 3.2, and it is the reason ADR-005 asks for a *committed* fixture rather than a
demonstration.

Applying ADR-005's second form to this pass's own measurement: `ast-grep scan` reads
**411 of the 427** `.py` files under `src/ tests/`. The 16 it skips are exactly the 16
zero-byte `__init__.py` files, and `sgconfig.yml` declares no path exclusion — so unlike
ruff, ast-grep's coverage of everything with content in it is total. Consequence for the
plan: `no-match-on-result`'s severity is `warning`, as is `no-raise-app-error-mapper`,
while the other three rules are `error`. The seven new rules should pick deliberately.

### What did not change

No spec, no ADR, no requirement and no scenario. The counts stand at **45 requirements /
180 scenarios** across seven capabilities, WHEN = THEN = 180, and
`openspec validate error-handling-foundation --strict` reports valid with 6/6 artifacts
complete. This pass edited only `HANDOFF.md` (§ intro, 3, 6) and `tasks.md`'s 3.1 `DONE`
block. Findings 26–28 are all bookkeeping in the one artifact an implementing agent is
told to trust over its own judgement — which is precisely where finding 21 lived too.

**VERDICT: APPROVED.** The plan is sound and the first 26 tasks implement it faithfully.
Two things must happen before PR A opens: commit the planning artifacts, and land task
3.2's fixture pair.

**Method note, fourth entry.** The three existing rules assume the plan is the only thing
moving. Once implementation starts, **the plan and the branch are two moving objects, and
the handoff is where they are reconciled.** Two concrete probes, both cheap:

1. **Never state a task total from memory or from another document — derive it, twice.**
   `grep -c '^- \[ \] '` counts only the *open* boxes; the total needs
   `grep -cE '^- \[( |x)\] '`, and a per-section sum is the reconciling second query. The
   disagreement between those two numbers (71 vs 97) is what revealed that implementation
   had started at all. It was not reported by anyone.
2. **A `DONE` block that says "partial" is a debt, and nothing collects it.** Task 3.1
   was ticked `[x]` while its own annotation deferred the re-measure. `--strict` cannot
   see that; `openspec status` counts it complete. Grep the `DONE` blocks for "partial",
   "deferred" and "TODO" before opening the PR that contains them.
