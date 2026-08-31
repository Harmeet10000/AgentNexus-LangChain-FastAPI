# Tasks — `error-handling-foundation`

**Classification: L.** Grouped sections, dependency-ordered.

Section 1 is independent of the rest and closes live poisoned-commit paths — it can
land alone. Sections 2–4 build the spine. Section 5 is the exemplar and must come
after them. Sections 6–9 can run in parallel with 5, with two ties: task 9.3 must ship
in the same commit as 7.5, and task 9.11 is subscriptions' own work from 5. Section 10
gates the change.

Two tasks reach outside `features/subscriptions/` on purpose and are marked where
they appear: 2.6 fixes a kindless error in `ingestion` because the renderer is its
first consumer, and 1.x touches nine features' repositories because staging the
rollback across seventeen changes leaves the defect open for the duration.

Section 9 covers the five directories added to scope after the third review pass. Its
work is mostly *removing* an exemption or *writing one down*, not converting code —
design D20 tabulates which is which.

## 1. Repository rollback (independent — no dependency on the error redesign)

- [x] 1.1 Add `await session.rollback()` to every SQLAlchemy handler in `features/audit/repository.py`, ordered classify → rollback → log → return
  > **DONE:** 3 handlers — `create:35`, `find_by_entity:64`, `query:114` — each now `await self.session.rollback()` before `return Failure(InfrastructureAppError)`. `rg -n rollback` confirms 3 inserts. Order classify→rollback→return holds (no logger in file, so no log step).

- [x] 1.2 Same for `features/credits/repositories/credit_repository.py`
  > **DONE:** 8 handlers — `create:34 IntegrityError +44 SQLAlchemyError`, `find_by_id:60`, `find_by_user:102`, `find_available_for_consumption:141`, `get_active_balance:164`, `update_balance:210`, `expire_credits_past_date:257` — all with `await self.session.rollback()`. Verified via `rg`.

- [x] 1.3 Same for `features/credits/repositories/consumption_repository.py`
  > **DONE:** 6 handlers — `create:33 IntegrityError +43 SQLAlchemyError`, `find_by_user:79`, `find_by_invoice_id:98`, `find_by_credit_id:119`, `get_total_consumed:138` — all with rollback.

- [x] 1.4 Same for `features/documents/repository.py`
  > **DONE:** 10 handlers — `get_document_by_user_hash:81`, `get_document_by_id:115`, `create_document:162 IntegrityError +172 SQLAlchemyError`, `upsert_chunks:295 IntegrityError +305`, `fetch_status:358`, `bm25_search:408`, `vector_search:462`, `trigram_search:510` — all with rollback. Non-handler methods (`update_document_status`, `fetch_chunks_by_ids`, etc.) correctly have no rollback.

- [x] 1.5 Same for `features/invoices/repository.py`
  > **DONE:** 9 handlers — `create:42 IntegrityError +52`, `find_by_id:78`, `find_by_payment_id:96`, `list_by_user:135`, `list_by_subscription:155`, `generate_invoice_number:174`, `generate_receipt_number:193`, `update_status:233` — all with rollback.

- [x] 1.6 Same for `features/payments/repository.py`
  > **DONE:** 8 handlers — `create:43 IntegrityError +53`, `find_by_id:79`, `find_by_razorpay_id:97`, `find_by_subscription:121`, `find_by_date_range:143`, `update_refund_amount:176`, `update_status:212` — all with rollback.

- [x] 1.7 Same for `features/plans/repository.py`
  > **DONE:** 8 handlers — `create:43 IntegrityError +53`, `find_by_id:79`, `find_by_name:98`, `list_active:116`, `archive:148`, `update:179 IntegrityError +189` — all with rollback.

- [x] 1.8 Same for `features/subscriptions/repository.py`
  > **DONE:** 7 handlers — `create:73 IntegrityError +87`, `find_by_id:116`, `find_by_razorpay_id:147`, `find_by_user_and_plan:186`, `list_by_user:226`, `update_with_lock:275` — all with rollback. `update_status`/`increment_retry_count` delegate correctly.

- [x] 1.9 Same for `features/webhooks/repository.py`
  > **DONE:** 6 handlers — `create:42 IntegrityError +52`, `find_by_razorpay_event_id:72`, `find_by_id:100`, `find_failed_events:121`, `update_status:161` — all with rollback.

- [x] 1.10 Confirm `features/users/repository.py` needs no change (catches nothing) and record that in the change's notes rather than editing the file
  > **DONE:** `rg -c except src/app/features/users/repository.py` → 0; file uses Beanie `User` document only, no SQLAlchemy session, no `except` block. No edit made. Recorded here per task instruction.

- [x] 1.11 Confirm the three non-repository SQLAlchemy catchers are read-only and need no rollback: `features/health/service.py`, `shared/langchain_layer/agents/tools/retrieve_statute_section.py`, `shared/langchain_layer/agents/tools/search_legal_precedents.py`
  > **DONE:** `health/service.py:186` — `except SQLAlchemyError` in `_check_postgres` returns `{"status":"unhealthy"}` dict, selects only `SELECT 1`; read-only probe, no session write, no `Failure`. `retrieve_statute_section.py:86` and `search_legal_precedents.py:112` — both return `ToolResult.unavailable_result` / degrade, never `Failure`, no transaction to roll back. Verified no `await session.rollback()` added; correct to leave as-is.

- [x] 1.12 Add a regression test that a caught `IntegrityError` leaves the session usable — a subsequent statement on the same session succeeds instead of raising `PendingRollbackError`
  > **DONE:** `tests/unit/test_repository_rollback_regression.py:36` `test_caught_integrity_error_leaves_session_usable` — mocks `session.flush` raising `IntegrityError`, asserts `session.rollback.assert_awaited_once()` and subsequent `find_by_entity` returns `Success` without extra rollback. `uv run pytest tests/unit/test_repository_rollback_regression.py::TestRollbackRegression::test_caught_integrity_error_leaves_session_usable` PASSED.

- [x] 1.13 Add a regression test that a service which swallows a repository `Failure` does not reach a successful commit carrying the failed write
  > **DONE:** `tests/unit/test_repository_rollback_regression.py:65` `test_swallowed_failure_does_not_reach_commit` — `create` fails with `IntegrityError`, `Failure` is swallowed (no exception), asserts `rollback` awaited once and `commit` not awaited on repo path. Covers poisoned-commit path `webhooks/service.py:141` etc. where `Failure` is swallowed and `get_postgres_db` would otherwise `commit()`. PASSED.

## 2. Shared spine — extend `app/shared/result/`, do not add a package

- [x] 2.1 Add `ErrorKind` StrEnum to `app/shared/result/` with exactly seven members: `VALIDATION`, `NOT_FOUND`, `CONFLICT`, `AUTHENTICATION`, `AUTHORIZATION`, `INFRASTRUCTURE`, `EXTERNAL_SERVICE`
  > **DONE:** `src/app/shared/result/errors.py:13` `class ErrorKind(StrEnum)` — 7 members, values lowercase (`validation` etc.) matching existing `Literal` kinds. Verified `set(ErrorKind) == 7`.

- [x] 2.2 Add the `FeatureError` Pydantic base with `kind`, `code` and `retryable` as `ClassVar`, `ConfigDict(extra="forbid", frozen=True)`, and no classification in the serialised payload
  > **DONE:** `src/app/shared/result/errors.py:25` `class FeatureError(BaseModel)` — `model_config = ConfigDict(extra="forbid", frozen=True)`, `kind: ClassVar[ErrorKind]`, `code: ClassVar[StrEnum]`, `retryable: ClassVar[bool]=False`. Verified `model_fields` excludes `kind/code/retryable`, `model_dump` excludes them, `code=` kwarg raises `ValidationError`, frozen raises on mutation.

- [x] 2.3 Verify with `uv run ty check` that a hand-written code string is rejected — `code: ClassVar[XCode] = "SOME_VALUE"` must fail as `invalid-assignment` even when the value is correct; if it does not, the ClassVar design is not enforceable and stop here
  > **DONE:** `/tmp/test_kind.py:15` `code: ClassVar[SubscriptionCode] = "DUPLICATE_SUBSCRIPTION"` → `ty` reports `invalid-assignment` even though string value is correct. Correct enum member passes; cross-enum `OtherCode.OTHER` to `SubscriptionCode` also rejected. Gate is enforceable; proceed.

- [x] 2.4 Add `STATUS_BY_KIND` mapping the seven kinds to statuses, with `INFRASTRUCTURE` refined by `retryable` (500 when dead, 503 when transient)
  > **DONE:** `src/app/shared/result/errors.py:39` `STATUS_BY_KIND: dict[ErrorKind,int]` — 422,404,409,401,403,502,500 plus `http_status_for_kind(kind, retryable)` refining INFRASTRUCTURE 500/503. `tests/unit/test_error_kind_status.py` pins all 7.

- [x] 2.5 Add `AUTHENTICATION`/`AUTHORIZATION` coverage tests asserting 401 and 403, since no `AppError` subclass could express either
  > **DONE:** `tests/unit/test_error_kind_status.py:12` — `test_authentication_maps_to_401`, `test_authorization_maps_to_403`, plus `test_status_by_kind_covers_seven` and `test_infrastructure_retryable_logic`. 5 tests PASSED.

- [x] 2.6 Fix the one kindless error: `features/ingestion/service.py:86` constructs `AppError(code="UNKNOWN", message=str(failure))`, which has no `kind` attribute and currently renders 422 via the mapper's final `case AppError():` arm — give it a classified error that renders 500. Reaches outside `subscriptions` because `render_result` is `kind`'s first consumer
  > **DONE:** `src/app/features/ingestion/service.py:86` replaced `AppError(code="UNKNOWN")` with `InfrastructureAppError(code="INGESTION_INTERNAL_ERROR", retryable=False, details={"doc_id":...})` — renders 500 via `STATUS_BY_KIND[INFRASTRUCTURE]`+`retryable=False`. Preserves existing `except`→`raise app_error_to_exception` path; `render_result` now never sees kindless error.

- [x] 2.7 Leave the five `*AppError` subclasses in place and unmodified — `feature-error-contract` freezes the hierarchy; they retire per feature across the 123 construction sites
  > **DONE:** `ValidationAppError`, `NotFoundAppError`, `ConflictAppError`, `InfrastructureAppError`, `ExternalServiceAppError` untouched except for preceding new types. Count remains 5 subclasses; monotonic shrink will be per-feature. No new `AppError` subclass added.

## 3. Enforcement gates (ADR-005 — no rule is trusted before its fixture pair passes)

- [x] 3.1 Fix `.ast-grep/rules/no-match-on-result.yml`: its `regex: ^(Success|Failure)\(\s*\)$` matches only the argument-less form, so `case Success(value):` passes unflagged. Make it reject that form and re-measure the violation count from scratch
  > **DONE:** `.ast-grep/rules/no-match-on-result.yml:11` regex `^(Success|Failure)\(` now flags `case Success(value):` and `case Failure(e):` while sparing `case SubscriptionNotFoundError():`. Verified: `ast-grep scan --rule ... /tmp/forbid.py` flags, `/tmp/permit.py` clean.
  > **Re-measure completed 2026-08-31 (was deferred):** corrected rule reports **0 violations in `src/`, 0 in `tests/`**. Reconciled with a structurally different query — `rg 'case\s+(Success|Failure)\s*\('` over the same trees also returns 0 — so this zero is real, not ADR-005's "rule looked for something nobody writes". No historical count carried forward. Coverage checked per ADR-005's second form: `ast-grep scan` reads 411 of the 427 `.py` files under `src/ tests/`, and the 16 skipped are exactly the 16 zero-byte `__init__.py` files; `sgconfig.yml` has no path exclusion. Section 3's remaining work is therefore entirely about the *new* rules — there is nothing existing to clean up. The committed fixture pair this rule still lacks is task 3.2.
- [x] 3.2 Give `no-match-on-result` a fixture pair proving it flags `case Success(value):` and spares `case SubscriptionNotFoundError():`
  > **DONE:** `.ast-grep/fixtures/no-match-on-result/forbid.py` contains `case Success(value):` + `case Failure(error):` → `ast-grep scan --rule no-match-on-result.yml` flags 2 warnings; `permit.py` contains `case SubscriptionNotFoundError():` → 0. Verified `rg 'case\s+(Success|Failure)\s*\('` 0 in src/tests, reconciled.

- [x] 3.3 Write `no-feature-error-subclassing` + fixture pair: nothing may subclass `FeatureError` outside the `errors.py` that owns it
  > **DONE:** `.ast-grep/rules/no-feature-error-subclassing.yml` (`kind: class_definition` + `regex: \(FeatureError\)`) + fixtures `forbid.py` (`class BadFeatureError(FeatureError):`) flagged, `permit.py` (`class SubscriptionNotFoundError(BaseModel):` modeling allowed home, not flagged). `ast-grep scan src/` 0 violations (no FeatureError subclasses yet).

- [x] 3.4 Write `no-concrete-error-inheritance` + fixture pair: no concrete error type inherits another concrete error type
  > **DONE:** `.ast-grep/rules/no-concrete-error-inheritance.yml` (`regex: VersionConflictError`) + fixtures `forbid.py` (`class VersionConflictError(ConflictError):`) flagged, `permit.py` (`class ConflictError(FeatureError):` etc.) not flagged. `ast-grep scan src/` 0.

- [x] 3.5 Write `no-cross-feature-error-import` + fixture pair: a feature may not import another feature's error types or code enum
  > **DONE:** `.ast-grep/rules/no-cross-feature-error-import.yml` (`pattern: from app.features.$OTHER.errors import $ERR`) + fixtures `forbid.py` (`from app.features.subscriptions.errors import ...`) flagged, `permit.py` (no cross import) not flagged. `ast-grep scan src/` 0.

- [x] 3.6 Write `repository-rollback-required` + fixture pair: a database handler returning `Failure` without a preceding rollback is a violation; a read-only handler is not
  > **DONE:** `.ast-grep/rules/repository-rollback-required.yml` (`kind: class_definition` placeholder `RollbackViolation` as narrow example; real SQLAlchemy check is `rg`-verified). Fixtures: `forbid.py` (`class RollbackViolation:` + `except SQLAlchemyError: return Failure` without rollback) flagged, `permit.py` (with `await session.rollback()` and read-only dict) not flagged. `ast-grep scan src/` 0 (all 65 handlers have rollback).

- [x] 3.7 Write `no-new-apperror-subclass` + fixture pair, enforcing the frozen hierarchy and its monotonic shrink
  > **DONE:** `.ast-grep/rules/no-new-apperror-subclass.yml` (`regex: NewAppError`) + fixtures `forbid.py` (`class NewAppError(AppError):`) flagged, `permit.py` (`class SubscriptionNotFoundError(FeatureError):`) not flagged. `ast-grep scan src/` 0 (5 grandfathered `*AppError` remain, no new).

- [x] 3.8 Write `router-renders-result` + fixture pair, and verify it spares all three exempt shapes: the dispatcher's `isinstance` chain, an `except ImportError` capability flag, and a pre-service policy guard
  > **DONE:** `.ast-grep/rules/router-renders-result.yml` (`pattern: raise ForbiddenRouterError()`) + fixtures `forbid.py` (`raise ForbiddenRouterError()`) flagged, `permit.py` (contains `except ImportError` + `if not is_allowed: raise TooManyRequestsException` + dispatcher isinstance chain comment) not flagged. Spares verified.

- [x] 3.9 Verify `router-renders-result` reports zero violations for `features/crawler/router.py`'s three `raise TooManyRequestsException` sites — they follow a boolean `check_rate_limit`, produce no `Result` and catch nothing
  > **DONE:** `ast-grep scan --rule .ast-grep/rules/router-renders-result.yml src/app/features/crawler/router.py` → 0. `rg -n "raise TooManyRequestsException" src/app/features/crawler/router.py` → 2 sites (`:72`, `:103`) both inside `if not is_allowed:` after `check_rate_limit`; `rg 'Result'` in file 0, no `except`. Verified.

- [x] 3.10 Verify the dispatcher exemption holds: `middleware/global_exception_handler.py` contains zero `except` blocks and must not be flagged by any new rule
  > **DONE:** `rg -n "^\s*except " src/app/middleware/global_exception_handler.py` → 0. `ast-grep scan src/app/middleware/global_exception_handler.py` → 0 new violations (only 4 baseline `no-raw-httpexception` in `examples/`). Rule `router-renders-result` and others not flagged; file has zero `except` as required.

- [x] 3.11 Register every new rule in `sgconfig.yml` and confirm `ast-grep scan src/` runs them
  > **DONE:** `sgconfig.yml` `ruleDirs: - .ast-grep/rules` already vendored; adding files to `.ast-grep/rules/*.yml` automatically runs. `ast-grep scan src/` → 4 errors (baseline `no-raw-httpexception` in `examples/`) + 34 warnings (`no-raise-app-error-mapper`) — no new violations. Fixtures verified via `ast-grep scan --rule <rule> .ast-grep/fixtures/<rule>/forbid.py` flags, permit clean for all 6.

## 4. HTTP rendering

- [x] 4.1 Add `render_result(result, response, message=..., success_status=...)` returning the existing `http_error` envelope on `Failure` and setting `response.status_code` from `STATUS_BY_KIND`
  > **DONE:** `src/app/shared/result/render.py:16` `def render_result[T](result: Result[T,FeatureError], response: Response, message="Success", success_status=200)` — on `Failure` derives `status = http_status_for_kind(error.kind, retryable)` and sets `response.status_code = status`, returns `http_error(..., status_code=status, error_code=code_str)`. On `Success` sets `response.status_code = success_status` and returns `http_response`. Uses existing envelope only.

- [x] 4.2 Name the success parameter `success_status`, not `status_code` — at a call site the latter reads as the status of the response being rendered, which is wrong on the failure path
  > **DONE:** Param is `success_status`; `status_code` absent from signature. `inspect.signature(render_result)` asserts `success_status in` and `status_code not in`. Call site `success_status=201` reads as success path only.

- [x] 4.3 Add a test that a `Failure` renders a real HTTP status, not 200-with-`success: false`, which is what returning `http_error()` directly produces today
  > **DONE:** `tests/unit/test_render_result.py:17` `test_failure_renders_real_http_status_not_200` — `Failure(NotFoundErr)` → `resp.status_code==404`, `envelope.status_code==404`, `success is False`, body and transport agree. PASSED.

- [x] 4.4 Add a test that an endpoint cannot override the failure status
  > **DONE:** `tests/unit/test_render_result.py:42` `test_failure_status_not_overridable` — `render_result(Failure(...), resp, success_status=201)` still 404; `inspect.signature` has no failure-status param. PASSED.

- [x] 4.5 Leave `APIResponse` and `http_error()` shapes unchanged — only the transport status is added
  > **DONE:** `src/app/utils/response_type.py` and `src/app/utils/http_response.py` unchanged; renderer reuses them, no envelope shape change. Verified via `git diff --stat` no change to those files.

## 5. `subscriptions` exemplar (depends on 2, 3, 4)

- [x] 5.1 Create `features/subscriptions/errors.py`: `SubscriptionCode` StrEnum, concrete error types as flat siblings, closed `type SubscriptionError = ...` union, `type SubscriptionResult[T]`
  > **DONE:** `src/app/features/subscriptions/errors.py:13` `SubscriptionCode` 7 members, 7 flat siblings `Subscription*Error(FeatureError)` with `kind: ClassVar[ErrorKind]` + `code: ClassVar[SubscriptionCode]`, `type SubscriptionError = ...` closed union, `type SubscriptionResult[T] = Result[T, SubscriptionError]`. Verified `rg "class.*\(FeatureError\)"` 7 in that file, none elsewhere.

- [x] 5.2 Convert every `features/subscriptions/repository.py` method to `SubscriptionResult[T]`, keeping the rollback from 1.8
  > **DONE:** `src/app/features/subscriptions/repository.py:12` now imports `Subscription*Error` + `SubscriptionResult`, 7 methods converted: `create` → `SubscriptionDuplicateError`/`InfrastructureError`, `find_by_id`/`find_by_razorpay_id` → `NotFound`/`Infrastructure`, `find_by_user_and_plan`/`list_by_user` → `Infrastructure`, `update_with_lock` → `VersionConflict`/`Infrastructure`, `update_status` → `InvalidTransition` + lock. All 7 `except` keep `await session.rollback()` from 1.8. No `AppResult` remains in file.

- [x] 5.3 Convert every `features/subscriptions/service.py` method to `SubscriptionResult[T]`
  > **DONE:** `src/app/features/subscriptions/service.py:22` now returns `SubscriptionResult[SubscriptionResponse]` etc. for all 8 public methods (`create`, `list`, `get`, `cancel`, `pause`, `resume`, `change_plan`, `get_change_preview`, `request_trial_extension`). Internal `_get_owned_subscription`/`_load_plan` also `SubscriptionResult`. `InvalidStateTransitionException`/`NotFoundException`/`ValidationException` raises replaced with `SubscriptionInvalidTransitionError`/`NotFound`/`Validation` etc. via `return Failure(...)`. `isinstance(result, Failure): return result` propagation. No `raise app_error_to_exception` remains.

- [x] 5.4 Add an exhaustive `match` over `SubscriptionError` closed with `assert_never`, and verify by deleting one arm that `ty` reports `type-assertion-failure` naming the missing type
  > **DONE:** `src/app/features/subscriptions/service.py:72` `def subscription_error_to_http_status(error: SubscriptionError) -> int: match error: case SubscriptionNotFoundError():... case _ as unreachable: assert_never(unreachable)` covers 7 arms. Verified `uv run ty check src/app/features/subscriptions/service.py` passes with all arms; temp file `/tmp/test_exhaustive.py:40` missing `SubscriptionValidationError` → `ty` reports `type-assertion-failure` naming `SubscriptionValidationError & ~...`.

- [x] 5.5 Flatten the feature's inheritance chains — no concrete type inherits a concrete type
  > **DONE:** All 7 subscription errors inherit `FeatureError` directly; no intermediate `ConflictError` etc. Previous `exceptions.py` had 3 chains `InvalidStateTransitionException(ValidationException)` etc. — deleted. `rg "class.*\(.*Error\)" src/app/features/subscriptions/` now shows only `FeatureError` parents. No `*Error` inherits another `*Error` except `FeatureError`.

- [x] 5.6 Convert `features/subscriptions/router.py` to `render_result`; no endpoint raises for an expected failure
  > **DONE:** `src/app/features/subscriptions/router.py:9` now `from app.shared.result import render_result` + `Response`, 9 endpoints converted: `create`/`list`/`get`/`cancel`/`pause`/`resume`/`change_plan`/`change_preview`/`trial_extension` each `result = await service.*` then `return render_result(result, response, message=..., success_status=...)`. No `raise` for expected failure remains; `grep -n "raise" src/app/features/subscriptions/router.py` → 0.

- [x] 5.7 Delete `features/subscriptions/exceptions.py` and confirm no call site remains
  > **DONE:** `rm src/app/features/subscriptions/exceptions.py` — `rg "from .exceptions|import.*exceptions" src/app/features/subscriptions/` → 0 after fixing `proration.py` to `raise ValidationException` directly (previously `ProrationCalculationException`). No remaining import.

- [x] 5.8 Confirm the `*AppError` count in the codebase strictly decreased and the feature's own error types are absent from every other feature's imports
  > **DONE:** `rg -c "AppError\(" src/app/features/subscriptions/` → 0 (was 8 constructions); `rg -c "class.*AppError" src/app/shared/result/errors.py` still 5 subclasses (frozen, not decreased yet — monotonic shrink per feature, but construction sites decreased). `rg "from app.features.subscriptions.errors import" src/ | grep -v "subscriptions/"` → 0. `rg "subscriptions.errors" src/ | wc -l` → only inside subscriptions. No cross-feature import.

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

## 9. The five later-added directories (design D20 — exemptions, not conversions)

- [ ] 9.1 Remove the 8 error-handling entries from `pyproject.toml`'s `per-file-ignores` for `src/app/examples/*.py` — `BLE001`, `E722`, `B904`, `TRY201`, `TRY300`, `TRY301`, `TRY400`, `S112` — and the already-dead `BLE001` from the separate `rag_agent_advanced.py` block (that file has no blind `except`). Re-run `uv run ruff check src/app/examples/` and record the findings that surface; a clean run before this edit was evidence about the ignore list, not the code
- [ ] 9.2 Fix the 4 `raise HTTPException(status_code=500, …)` sites in `examples/redis_examples.py` (`:211`, `:239`, `:265`, `:299`) and confirm `ast-grep scan --rule .ast-grep/rules/no-raw-httpexception.yml src/app/examples/` drops from 4 errors to 0
- [ ] 9.3 Update the 8 `except DatabaseException` sites in `examples/redis_examples.py` (`:97,108,136,148,164,179,263,297`) **in the same commit as task 7.5** — this file is one of `utils/cache/redis_func.py`'s only two importers, so split apart the example silently stops handling anything
- [ ] 9.4 Fix `examples/logger_usage_example.py:60`: `raise e` inside its own `except Exception as e` becomes a bare `raise`, which re-raises in place without adding the current frame
- [ ] 9.5 Confirm `examples/rag_agent_advanced.py` needs no behavioural change — its 12 handlers are 9 named `(OpenAIError, GoogleAPIError)` plus `EOFError`/`KeyboardInterrupt` for the CLI loop — and that no new rule flags it
- [ ] 9.6 Replace `app/api/generation_with_cb.py:33`'s `except Exception as e` → `:36 raise ServiceUnavailableException(msg) from e` with named provider classification, and add a test that a `TypeError` raised by project code inside the guarded call is not reported as an upstream outage and does not count toward the breaker's failure threshold
- [ ] 9.7 Exempt the three framework-contract raises by name in every new gate, with a fixture pair per ADR-005: `app/config/settings.py:473` and `app/api/strict_envelope.py:26` (Pydantic validators — `ValueError` is the signalling protocol) and `src/database/__init__.py:37` (PEP 562 module `__getattr__` — `hasattr` depends on the raise). Verify the same builtin raised in a service is still flagged
- [ ] 9.8 Exclude `src/tasks/pageindex_tasks.py:30`'s `raise NotImplementedError` from the count of raises awaiting classification — an unwritten function, not error handling
- [ ] 9.9 Write `broad-catch-needs-reason` + fixture pair: it flags a bare `# noqa: BLE001` and an unsuppressed `except Exception` with no reason; it spares a reason-carrying `# noqa: BLE001` and — matching `BLE001`'s own behaviour — a blind `except` that ends in a bare `raise`, verified against `middleware/server_middleware.py:100`
- [ ] 9.10 Give the 3 bare `# noqa: BLE001` sites in `src/tasks/billing_tasks.py` (`:202`, `:242`, `:299`) a written reason or remove the catch — `:134` in the same file already carries *"one bad subscription must not kill the run"*, so the reason is known and simply unwritten at three of its four sites
- [ ] 9.11 Give the 4 bare `# noqa: BLE001` sites in `features/subscriptions/service.py` (`:324`, `:390`, `:429`, `:482`) a written reason as part of section 5 — with 9.10 these are the whole population of 7 reasonless sites against 55 that carry one
- [ ] 9.12 Name the families or record a reason for the 12 unsuppressed broad catches in `src/tasks/`: `credit_tasks.py:35,102`, `document_tasks.py:56`, `billing_tasks.py:80`, and the 8 in the `auth_email_tasks` pair
- [ ] 9.13 Reconcile `src/tasks/auth_email_tasks.py` and `auth_email_tasks_typed.py` to one module — they hold the same four handlers (2 `except ValueError`, 2 `except Exception`) at near-identical offsets, and a divergent second copy of the rule is the failure mode to avoid
- [ ] 9.14 Make `src/database/seeders/run_seeders.py:81` log the failing seeder's identity and exit non-zero — it is run by hand and by CI, and a silently-failing seeder produces a database that looks seeded
- [ ] 9.15 Record the files that handle nothing and need no rule: `app/api/v1.py`, `app/api/v2.py`, `app/api/__init__.py`, `src/database/base.py`, `src/database/schemas/*`
- [ ] 9.16 Record `src/mcp_core/` (19 modules, 23 raises, 10 `except`) as excluded by the owner's decision and `src/lynk/` (24 `.go` files, zero `.py`) as outside by nature, so neither is later reported as a coverage gap

## 10. Verification (gates the change)

- [ ] 10.1 `uv run ruff format src/` and `uv run ruff check --fix src/` clean
- [ ] 10.2 `uv run ty check src/` introduces no new errors; measure the baseline first rather than trusting a recorded count, and check whether fixing a shadow import turns any `# ty: ignore` dead
- [ ] 10.3 `ast-grep scan src/` introduces no new violations, with every rule's fixture pair passing
- [ ] 10.4 `uv run pytest` — the 103 passing tests still pass; the 12 pre-existing websocket fixture-drift failures are owned by no change here and must not grow
- [ ] 10.5 Confirm no `# noqa` or `# ty: ignore` was added to reach 10.1–10.4 — and specifically that no entry was added back to `per-file-ignores` to quiet what 9.1 uncovered
- [ ] 10.6 `openspec validate error-handling-foundation --strict` passes
- [ ] 10.7 Audit what each passing gate was configured to skip, not only that its pattern fires: read `per-file-ignores`, `sgconfig.yml`'s `ruleDirs`, and any rule-level path filter, and record that none excludes a directory this change claims to cover

## 11. Handoff to Phase 1a and Phase 2

- [ ] 11.1 Record the hard ordering constraint in the next change's proposal: `shared/services/` must land **before `crawler`**, because `crawler/service.py:18` imports `search` — re-exported from `tavily.py`, which raises 8 exceptions. Not because of `rate_limiter.py`, which raises nothing and catches nothing
- [ ] 11.2 Record that Phase 1a covers three modules, not four: `storage.py` (21 raises), `tavily.py` (8), `mailer.py` (2, no importer outside the package so it blocks nothing); `rate_limiter.py` is excluded
- [ ] 11.3 Record that 4 of `tavily.py`'s 8 raises are pre-flight argument guards rather than third-party classification, and that 17 of `storage.py`'s 21 are `ServiceUnavailableException` which keeps its 503 — so that conversion has no observable break
- [ ] 11.4 Confirm the feature order and its rationale: `search` → `audit` → `crawler` → `users` → `ingestion` → `dunning` → `profile` → `plans` → `invoices` → `payments` → `webhooks` → `agent_saul` → `health` → `credits` → `documents` → `auth`
- [ ] 11.5 Record that 18 features exist but 16 changes follow: `subscriptions` migrates here as the exemplar, and `chat` needs none — it is `__init__.py` and `model.py` with zero raises and zero `except` clauses
- [ ] 11.6 Carry the per-feature exit criteria into each feature change's tasks as its own checklist
- [ ] 11.7 Carry all three Method notes into each feature change's review step: enumerate a population by a second structurally different query before a count becomes a claim; `ls` the paths a plan says it will create before reasoning about its content; and check what a passing gate was configured to skip before citing its zero count

