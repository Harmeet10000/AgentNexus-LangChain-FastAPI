# Tasks — `error-handling-foundation`

**Classification: L.** Grouped sections, dependency-ordered.

Section 1 is independent of the rest and closes live poisoned-commit paths — it can
land alone. Sections 2–4 build the spine. Section 5 is the exemplar and must come
after them. Sections 6–8 can run in parallel with 5. Section 9 gates the change.

Two tasks reach outside `features/subscriptions/` on purpose and are marked where
they appear: 2.6 fixes a kindless error in `ingestion` because the renderer is its
first consumer, and 1.x touches nine features' repositories because staging the
rollback across seventeen changes leaves the defect open for the duration.

## 1. Repository rollback (independent — no dependency on the error redesign)

- [x] 1.1 Add `await session.rollback()` to every SQLAlchemy handler in `features/audit/repository.py`, ordered classify → rollback → log → return
- [x] 1.2 Same for `features/credits/repositories/credit_repository.py`
- [x] 1.3 Same for `features/credits/repositories/consumption_repository.py`
- [x] 1.4 Same for `features/documents/repository.py`
- [x] 1.5 Same for `features/invoices/repository.py`
- [x] 1.6 Same for `features/payments/repository.py`
- [x] 1.7 Same for `features/plans/repository.py`
- [x] 1.8 Same for `features/subscriptions/repository.py`
- [x] 1.9 Same for `features/webhooks/repository.py`
- [x] 1.10 Confirm `features/users/repository.py` needs no change (catches nothing) and record that in the change's notes rather than editing the file
- [x] 1.11 Confirm the three non-repository SQLAlchemy catchers are read-only and need no rollback: `features/health/service.py`, `shared/langchain_layer/agents/tools/retrieve_statute_section.py`, `shared/langchain_layer/agents/tools/search_legal_precedents.py`
- [x] 1.12 Add a regression test that a caught `IntegrityError` leaves the session usable — a subsequent statement on the same session succeeds instead of raising `PendingRollbackError`
- [x] 1.13 Add a regression test that a service which swallows a repository `Failure` does not reach a successful commit carrying the failed write

## 2. Shared spine — extend `app/shared/result/`, do not add a package

- [ ] 2.1 Add `ErrorKind` StrEnum to `app/shared/result/` with exactly seven members: `VALIDATION`, `NOT_FOUND`, `CONFLICT`, `AUTHENTICATION`, `AUTHORIZATION`, `INFRASTRUCTURE`, `EXTERNAL_SERVICE`
- [ ] 2.2 Add the `FeatureError` Pydantic base with `kind`, `code` and `retryable` as `ClassVar`, `ConfigDict(extra="forbid", frozen=True)`, and no classification in the serialised payload
- [ ] 2.3 Verify with `uv run ty check` that a hand-written code string is rejected — `code: ClassVar[XCode] = "SOME_VALUE"` must fail as `invalid-assignment` even when the value is correct; if it does not, the ClassVar design is not enforceable and stop here
- [ ] 2.4 Add `STATUS_BY_KIND` mapping the seven kinds to statuses, with `INFRASTRUCTURE` refined by `retryable` (500 when dead, 503 when transient)
- [ ] 2.5 Add `AUTHENTICATION`/`AUTHORIZATION` coverage tests asserting 401 and 403, since no `AppError` subclass could express either
- [ ] 2.6 Fix the one kindless error: `features/ingestion/service.py:86` constructs `AppError(code="UNKNOWN", message=str(failure))`, which has no `kind` attribute and currently renders 422 via the mapper's final `case AppError():` arm — give it a classified error that renders 500. Reaches outside `subscriptions` because `render_result` is `kind`'s first consumer
- [ ] 2.7 Leave the five `*AppError` subclasses in place and unmodified — `feature-error-contract` freezes the hierarchy; they retire per feature across the 123 construction sites

## 3. Enforcement gates (ADR-005 — no rule is trusted before its fixture pair passes)

- [ ] 3.1 Fix `.ast-grep/rules/no-match-on-result.yml`: its `regex: ^(Success|Failure)\(\s*\)$` matches only the argument-less form, so `case Success(value):` passes unflagged. Make it reject that form and re-measure the violation count from scratch
- [ ] 3.2 Give `no-match-on-result` a fixture pair proving it flags `case Success(value):` and spares `case SubscriptionNotFoundError():`
- [ ] 3.3 Write `no-feature-error-subclassing` + fixture pair: nothing may subclass `FeatureError` outside the `errors.py` that owns it
- [ ] 3.4 Write `no-concrete-error-inheritance` + fixture pair: no concrete error type inherits another concrete error type
- [ ] 3.5 Write `no-cross-feature-error-import` + fixture pair: a feature may not import another feature's error types or code enum
- [ ] 3.6 Write `repository-rollback-required` + fixture pair: a database handler returning `Failure` without a preceding rollback is a violation; a read-only handler is not
- [ ] 3.7 Write `no-new-apperror-subclass` + fixture pair, enforcing the frozen hierarchy and its monotonic shrink
- [ ] 3.8 Write `router-renders-result` + fixture pair, and verify it spares all three exempt shapes: the dispatcher's `isinstance` chain, an `except ImportError` capability flag, and a pre-service policy guard
- [ ] 3.9 Verify `router-renders-result` reports zero violations for `features/crawler/router.py`'s three `raise TooManyRequestsException` sites — they follow a boolean `check_rate_limit`, produce no `Result` and catch nothing
- [ ] 3.10 Verify the dispatcher exemption holds: `middleware/global_exception_handler.py` contains zero `except` blocks and must not be flagged by any new rule
- [ ] 3.11 Register every new rule in `sgconfig.yml` and confirm `ast-grep scan src/` runs them

## 4. HTTP rendering

- [ ] 4.1 Add `render_result(result, response, message=..., success_status=...)` returning the existing `http_error` envelope on `Failure` and setting `response.status_code` from `STATUS_BY_KIND`
- [ ] 4.2 Name the success parameter `success_status`, not `status_code` — at a call site the latter reads as the status of the response being rendered, which is wrong on the failure path
- [ ] 4.3 Add a test that a `Failure` renders a real HTTP status, not 200-with-`success: false`, which is what returning `http_error()` directly produces today
- [ ] 4.4 Add a test that an endpoint cannot override the failure status
- [ ] 4.5 Leave `APIResponse` and `http_error()` shapes unchanged — only the transport status is added

## 5. `subscriptions` exemplar (depends on 2, 3, 4)

- [ ] 5.1 Create `features/subscriptions/errors.py`: `SubscriptionCode` StrEnum, concrete error types as flat siblings, closed `type SubscriptionError = ...` union, `type SubscriptionResult[T]`
- [ ] 5.2 Convert every `features/subscriptions/repository.py` method to `SubscriptionResult[T]`, keeping the rollback from 1.8
- [ ] 5.3 Convert every `features/subscriptions/service.py` method to `SubscriptionResult[T]`
- [ ] 5.4 Add an exhaustive `match` over `SubscriptionError` closed with `assert_never`, and verify by deleting one arm that `ty` reports `type-assertion-failure` naming the missing type
- [ ] 5.5 Flatten the feature's inheritance chains — no concrete type inherits a concrete type
- [ ] 5.6 Convert `features/subscriptions/router.py` to `render_result`; no endpoint raises for an expected failure
- [ ] 5.7 Delete `features/subscriptions/exceptions.py` and confirm no call site remains
- [ ] 5.8 Confirm the `*AppError` count in the codebase strictly decreased and the feature's own error types are absent from every other feature's imports

## 6. Exception-family reachability (design D18 / ADR-006)

- [ ] 6.1 Re-measure reachability over **ancestors**, not exact names — `raise TaskDispatchError` finds nothing because only its subclasses `UnregisteredTaskError` and `TaskPayloadValidationError` are raised
- [ ] 6.2 Validate the reachability measurement against `connections/celery_registry.py`'s correctly-rooted family first; if the method flags that family, the method is wrong and the counts are not usable
- [ ] 6.3 Re-root or catch by name: `CircuitBreakerOpenError`
- [ ] 6.4 Re-root or catch by name: `IdempotencyLockError`
- [ ] 6.5 Re-root or catch by name: `AgentMemoryError`
- [ ] 6.6 Re-root or catch by name: `CogneeSetupError`
- [ ] 6.7 Re-root or catch by name: `StateSchemaVersionError`
- [ ] 6.8 Order every catch site over the nine measured inheritance chains narrowest-first, so no broader handler shadows a narrower one
- [ ] 6.9 Add a test that a deliberately-unraised abstract base is not reported as unreachable

## 7. Classification corrections with observable effects

- [ ] 7.1 Replace the 49 `"DB_ERROR"` literals in the 9 relational repositories with the enum member; status corrects 503 → 500 because a failed relational transaction is dead
- [ ] 7.2 Replace the 7 `"DB_ERROR"` literals in `features/auth/repository.py` with the enum member, **keeping** them retryable at 503 — they are Mongo and Redis failures, which are genuinely retryable, and they sit on the login path
- [ ] 7.3 Add a test pinning the 49/7 split so a later sweep cannot collapse the two halves, which correct in opposite directions
- [ ] 7.4 Classify `auth/repository.py`'s `DuplicateKeyError` handlers as `CONFLICT`, not infrastructure, and confirm no rollback is added to a document-store repository
- [ ] 7.5 Reclassify `utils/cache/redis_func.py`'s 27 `DatabaseException` raises as cache failures; note in the change that this module is off any request path (importers are its own `__init__` and `examples/redis_examples.py`), so it is a bad exemplar rather than a production fault
- [ ] 7.6 Pin `connections/postgres.py`'s `get_postgres_db` shape by test, with no code change — it commits on clean exit, rolls back only on an escaping exception, and cannot see a `Result`
- [ ] 7.7 Name every exception family `lifecycle/lifespan.py` survives, so its 14 named handlers stay the reference and its single catch-all stays the exception

## 8. Documentation and configuration

- [ ] 8.1 Rewrite `.opencode/instructions/EXCEPTION-RULES.md` for the per-feature union, the flat-sibling rule and its shadowing footgun, and try/except as third-party adapter only
- [ ] 8.2 Rewrite `.opencode/instructions/RESULT-PATTERN.md` for `isinstance` on the `Result` and `match` + `assert_never` on the error union
- [ ] 8.3 Reconcile the drifted `.kiro/steering/` copies of both files, or replace them with a pointer to the `.opencode/instructions/` originals
- [ ] 8.4 Update `docs-site/architecture/error-and-result-pattern.mdx` and `docs-site/api-reference/errors.mdx` for the seven kinds and the rendered status
- [ ] 8.5 Reconcile `openspec/config.yaml`'s context block and the `spec-gated` review instruction with the new rule
- [ ] 8.6 Fix `CLAUDE.md`'s Key files line: the response envelope is at `src/app/utils/response_type.py`, not `src/app/shared/response_type.py`
- [ ] 8.7 Record in the docs that nothing dispatches on `kind` today — the field exists on five subclasses and is never read — so `render_result` is its first consumer

## 9. Verification (gates the change)

- [ ] 9.1 `uv run ruff format src/` and `uv run ruff check --fix src/` clean
- [ ] 9.2 `uv run ty check src/` introduces no new errors; measure the baseline first rather than trusting a recorded count, and check whether fixing a shadow import turns any `# ty: ignore` dead
- [ ] 9.3 `ast-grep scan src/` introduces no new violations, with every rule's fixture pair passing
- [ ] 9.4 `uv run pytest` — the 103 passing tests still pass; the 12 pre-existing websocket fixture-drift failures are owned by no change here and must not grow
- [ ] 9.5 Confirm no `# noqa` or `# ty: ignore` was added to reach 9.1–9.4
- [ ] 9.6 `openspec validate error-handling-foundation --strict` passes

## 10. Handoff to Phase 1a and Phase 2

- [ ] 10.1 Record the hard ordering constraint in the next change's proposal: `shared/services/` must land **before `crawler`**, because `crawler/service.py:18` imports `search` — re-exported from `tavily.py`, which raises 8 exceptions. Not because of `rate_limiter.py`, which raises nothing and catches nothing
- [ ] 10.2 Record that Phase 1a covers three modules, not four: `storage.py` (21 raises), `tavily.py` (8), `mailer.py` (2, no importer outside the package so it blocks nothing); `rate_limiter.py` is excluded
- [ ] 10.3 Record that 4 of `tavily.py`'s 8 raises are pre-flight argument guards rather than third-party classification, and that 17 of `storage.py`'s 21 are `ServiceUnavailableException` which keeps its 503 — so that conversion has no observable break
- [ ] 10.4 Confirm the feature order and its rationale: `search` → `audit` → `crawler` → `users` → `ingestion` → `dunning` → `profile` → `plans` → `invoices` → `payments` → `webhooks` → `agent_saul` → `health` → `credits` → `documents` → `auth`
- [ ] 10.5 Carry the per-feature exit criteria into each feature change's tasks as its own checklist
- [ ] 10.6 Carry both Method notes into each feature change's review step: enumerate a population by a second structurally different query before a count becomes a claim, and `ls` the paths a plan says it will create before reasoning about the plan's content
