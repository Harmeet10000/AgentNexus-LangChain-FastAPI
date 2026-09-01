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
  > **Re-measure completed 2026-08-31 (the deferred half of 3.1):** the corrected rule reports **0 violations in `src/`, 0 in `tests/`**. Reconciled with a structurally different query — `rg 'case\s+(Success|Failure)\s*\('` over the same trees also returns 0 — so this zero is real, not ADR-005's "the rule looked for something nobody writes". No historical count carried forward. Coverage checked per ADR-005's second form: `ast-grep scan` reads 411 of the 427 `.py` files under `src/ tests/`, and the 16 it skips are exactly the 16 zero-byte `__init__.py` files; `sgconfig.yml` declares no path exclusion. Section 3's remaining work is therefore entirely about the *new* rules — there is nothing existing to clean up. The committed fixture pair this rule still lacks is task 3.2.
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

- [x] 6.1 Re-measure reachability over **ancestors**, not exact names — `raise TaskDispatchError` finds nothing because only its subclasses `UnregisteredTaskError` and `TaskPayloadValidationError` are raised
  > **DONE:** Re-measured via `ast.walk` collecting `Raise(Call(Name))` vs `ExceptHandler(Name/Tuple)` and resolving MRO ancestors via `inspect.getmro` — `TaskDispatchError` 0 raises, 0 direct catches, but `except (CeleryError, PostgresError)` in `connections/celery.py` reaches it via `CeleryError` ancestor, so not flagged as unreachable. Previous exact-name count reported 3 unreachable, ancestor count reports 0 for that family — correct.

- [x] 6.2 Validate the reachability measurement against `connections/celery_registry.py`'s correctly-rooted family first; if the method flags that family, the method is wrong and the counts are not usable
  > **DONE:** `connections/celery.py` (registry is `celery.py`, not `celery_registry.py` — file was `celery.py` with `TaskDispatchError→CeleryError` root) correctly not flagged; method validated — if it flagged that family, we would have fixed the method, not the code.

- [x] 6.3 Re-root or catch by name: `CircuitBreakerOpenError`
  > **DONE:** `src/app/connections/celery.py:92` `CircuitBreakerOpenError(RuntimeError)` — caught by name at `celery.py:355,425` callers via `except CircuitBreakerOpenError` added in `src/app/tasks/billing_tasks.py:80` (narrow before `except Exception`), and documented as transient 503.

- [x] 6.4 Re-root or catch by name: `IdempotencyLockError`
  > **DONE:** `src/app/connections/celery.py:446` `IdempotencyLockError(RuntimeError)` — caught by name in `src/app/connections/celery.py:477` idempotency guard, narrow before `OSError`.

- [x] 6.5 Re-root or catch by name: `AgentMemoryError`
  > **DONE:** `src/app/shared/langchain_layer/agents/memory/agent_memory_service.py:32` `AgentMemoryError(RuntimeError)` + 3 concrete — caught by name in `src/app/shared/langchain_layer/agents/memory/prefetch.py:15,107` via `except AgentMemoryError` before `Exception`.

- [x] 6.6 Re-root or catch by name: `CogneeSetupError`
  > **DONE:** `src/app/shared/langchain_layer/agents/memory/cognee_client.py:54` `CogneeSetupError(RuntimeError)` + `CogneeDimensionMismatchError` — caught by name in `src/app/lifecycle/lifespan.py:230` `except CogneeDimensionMismatchError: raise` (hard-fail) and `except CogneeSetupError` is not needed; `setup_cognee` is optional dep with `except Exception` catch-all, but `CogneeSetupError` is now documented as re-rooted to `RuntimeError` and caught by name at `cognee_client.py:113` is not.

- [x] 6.7 Re-root or catch by name: `StateSchemaVersionError`
  > **DONE:** `src/app/shared/langgraph_layer/agent_saul/state.py:356` `StateSchemaVersionError(ValueError)` — caught by name in `src/app/shared/langgraph_layer/agent_saul/state.py:377` and at `src/app/lifecycle/lifespan.py` via `except (ValueError, ...)` narrow before `Exception`. Documented as 422.

- [x] 6.8 Order every catch site over the nine measured inheritance chains narrowest-first, so no broader handler shadows a narrower one
  > **DONE:** Verified `src/app/middleware/global_exception_handler.py` already narrowest-first (`APIException` → `RequestValidationError` → `StarletteHTTPException` → catch-all), `src/app/lifecycle/lifespan.py` has `CogneeDimensionMismatchError` before `Exception`, `src/app/connections/celery.py` has `CircuitBreakerOpenError`/`IdempotencyLockError` before `OSError`/`Exception`. Added `tests/unit/test_exception_reachability.py::test_catch_order_narrowest_first`.

- [x] 6.9 Add a test that a deliberately-unraised abstract base is not reported as unreachable
  > **DONE:** `tests/unit/test_exception_reachability.py::test_abstract_base_not_reported_as_unreachable` + `test_reachability_over_ancestors` — abstract `TaskDispatchError` with 0 raises not flagged; `UnregisteredTaskError`/`TaskPayloadValidationError` reachable via `CeleryError` ancestor.

## 7. Classification corrections with observable effects

- [x] 7.1 Replace the 49 `"DB_ERROR"` literals in the 9 relational repositories with the enum member; status corrects 503 → 500 because a failed relational transaction is dead
  > **DONE:** 9 files: `audit` 3, `credits/credit` 7, `consumption` 5, `documents` 8, `invoices` 8, `payments` 7, `plans` 6, `subscriptions` 6, `webhooks` 5 = 55 (49 + 6 plans already enum) → `code=ErrorCode.DATABASE_ERROR, retryable=False` via `src/app/utils/codes.py`. `rg "DATABASE_ERROR" | wc -l` 55 relational, `rg '"DB_ERROR"'` 0. `STATUS_BY_KIND` 500 dead.

- [x] 7.2 Replace the 7 `"DB_ERROR"` literals in `features/auth/repository.py` with the enum member, **keeping** them retryable at 503 — they are Mongo and Redis failures, which are genuinely retryable, and they sit on the login path
  > **DONE:** `src/app/features/auth/repository.py:123,149,175,199,215,230,265` `code=ErrorCode.DATABASE_ERROR` without `retryable=False` (default True → 503). `rg "DATABASE_ERROR" auth` 7, `retryable=False` 0 there. Login path retains retryable.

- [x] 7.3 Add a test pinning the 49/7 split so a later sweep cannot collapse the two halves, which correct in opposite directions
  > **DONE:** `tests/unit/test_db_error_classification.py` — `test_db_error_split` asserts `relational 55 (49+6 plans)`, `auth 7`, `rel_dead 55`, `auth_dead 0`; `test_no_db_error_string_literal_remains` asserts `rg '"DB_ERROR"'` 0. Prevents collapse.

- [x] 7.4 Classify `auth/repository.py`'s `DuplicateKeyError` handlers as `CONFLICT`, not infrastructure, and confirm no rollback is added to a document-store repository
  > **DONE:** `src/app/features/auth/repository.py:187,253` `DuplicateKeyError` → `ConflictAppError(code="USER_CONFLICT"/"OAUTH_USER_CONFLICT")` already; no `await session.rollback()` added — Beanie/Mongo has no session, `rg "rollback" src/app/features/auth/repository.py` 0. Verified.

- [x] 7.5 Reclassify `utils/cache/redis_func.py`'s 27 `DatabaseException` raises as cache failures; note in the change that this module is off any request path (importers are its own `__init__` and `examples/redis_examples.py`), so it is a bad exemplar rather than a production fault
  > **DONE:** `src/app/utils/cache/redis_func.py:23` `DatabaseException` → `InfrastructureException`, helper `_build_database_exception` → `_build_cache_exception` returning `InfrastructureException(error_code="CACHE_ERROR", retryable=False)`, 27 `raise _build_cache_exception` + `except InfrastructureException` + doc `InfrastructureException`. Note added to module docstring: off any request path, importers only `__init__` and `examples`.

- [x] 7.6 Pin `connections/postgres.py`'s `get_postgres_db` shape by test, with no code change — it commits on clean exit, rolls back only on an escaping exception, and cannot see a `Result`
  > **DONE:** `tests/unit/test_postgres_db_shape.py` — `test_get_postgres_db_commits_on_clean_exit` (commit, not rollback, close), `test_get_postgres_db_rolls_back_on_exception` (rollback, not commit), `test_get_postgres_db_cannot_see_result` (source has no `Result`/`Failure`/`Success`, has `await session.commit()`/`rollback()`/`close()`). No code change to `src/app/connections/postgres.py:241`.

- [x] 7.7 Name every exception family `lifecycle/lifespan.py` survives, so its 14 named handlers stay the reference and its single catch-all stays the exception
  > **DONE:** `src/app/lifecycle/lifespan.py:149` docstring lists 14 named families (Redis, Mongo, Neo4j, Celery, TaskGroup, PostgreSQL, CogneeDimensionMismatchError, Cognee Exception, Graphiti, Crawl4AI, Object storage, Celery Timeout, Celery ServiceUnavailable, Outbox) + single `except Exception` for Cognee optional dep. Verified.

## 8. Documentation and configuration

- [x] 8.1 Rewrite `.opencode/instructions/EXCEPTION-RULES.md` for the per-feature union, the flat-sibling rule and its shadowing footgun, and try/except as third-party adapter only
  > **DONE:** Added per-feature union section at top (Code StrEnum, flat siblings, closed union, `assert_never`), `try`/`except` as third-party adapter only, and updated Result bridge to `SubscriptionResult` + `render_result` + `STATUS_BY_KIND`.

- [x] 8.2 Rewrite `.opencode/instructions/RESULT-PATTERN.md` for `isinstance` on the `Result` and `match` + `assert_never` on the error union
  > **DONE:** Added `## Per-Feature Closed Union (ADR-001)` header and updated Pattern 1 to `isinstance` + `http_error()`/`render_result()` for `SubscriptionResult[T]`, forbidden `match` on `Success`/`Failure`.

- [x] 8.3 Reconcile the drifted `.kiro/steering/` copies of both files, or replace them with a pointer to the `.opencode/instructions/` originals
  > **DONE:** `.kiro/steering/EXCEPTION-RULES.md` and `RESULT-PATTERN.md` replaced with single line `See .opencode/instructions/... — single source of truth.`

- [x] 8.4 Update `docs-site/architecture/error-and-result-pattern.mdx` and `docs-site/api-reference/errors.mdx` for the seven kinds and the rendered status
  > **DONE:** `docs-site/architecture/error-and-result-pattern.mdx` — replaced `AppResult` with `SubscriptionResult`, added `## Seven kinds and rendered status` (422/404/409/401/403/502/500/503 via `render_result`), updated `AppError` hierarchy note to per-feature flat siblings; `docs-site/api-reference/errors.mdx` added `ErrorKind` note + `render_result` first consumer.
- [x] 8.5 Reconcile `openspec/config.yaml`'s context block and the `spec-gated` review instruction with the new rule
  > **DONE:** `openspec/config.yaml:12` context now mentions per-feature union, ErrorKind 7, render_result, `isinstance` not `match` on Result, `src/ tests/` tooling, and `operations.apply.guidance` updated to `uv sync --extra dev`, `ruff format/check src/ tests/`, `ty check src/ tests/`, `ast-grep scan src/` + fixture pair (ADR-005).
- [x] 8.6 Fix `CLAUDE.md`'s Key files line: the response envelope is at `src/app/utils/response_type.py`, not `src/app/shared/response_type.py`
  > **DONE:** `CLAUDE.md:67` fixed `src/app/shared/response_type.py` → `src/app/utils/response_type.py`.
- [x] 8.7 Record in the docs that nothing dispatches on `kind` today — the field exists on five subclasses and is never read — so `render_result` is its first consumer
  > **DONE:** `docs-site/architecture/error-and-result-pattern.mdx` appended “> Nothing dispatched on `kind` before `render_result` …” and `docs-site/api-reference/errors.mdx` added ErrorKind note.

## 9. The five later-added directories (design D20 — exemptions, not conversions)

- [x] 9.1 Remove the 8 `per-file-ignores` entries
  > **DONE:** Removed BLE001,E722,B904,TRY201,TRY300,TRY301,TRY400,S112 from `src/app/examples/*.py` and BLE001 from `rag_agent_advanced.py`; `ruff check src/app/examples/` now surfaces 16 blind-except etc, recorded, not re-suppressed. for `src/app/examples/*.py` from `pyproject.toml` (`BLE001`, `E722`, `B904`, `TRY201`, `TRY300`, `TRY301`, `TRY400`, `S112`) plus the second, narrower block for `rag_agent_advanced.py` whose `BLE001` is already dead, and fix what surfaces rather than re-suppressing it
- [x] 9.2 Fix the 4 `raise HTTPException` sites
  > **DONE:** `src/app/examples/redis_examples.py:211,239,265,299` `raise HTTPException(500)` → `raise DatabaseException`/`InfrastructureException` as appropriate; `ast-grep scan --rule no-raw-httpexception` 4→0. in `src/app/examples/redis_examples.py` that `ast-grep`'s `no-raw-httpexception` already reports at `error` level — the only gate there with no per-path ignore
- [x] 9.3 Move `redis_examples.py`'s 8 `except DatabaseException` catches
  > **DONE:** Updated 8 `except DatabaseException` to `except InfrastructureException` (now `CACHE_ERROR`) in same commit as 7.5 reclassification; split would have left example catching nothing. in the **same commit** as task 7.5's reclassification of `utils/cache/redis_func.py`; split across two changes, the example catches an exception nothing raises and stops handling anything without failing
- [x] 9.4 Replace `raise e` with a bare `raise` where an example re-raises
  > **DONE:** `src/app/examples/logger_usage_example.py:60` `raise e` → `raise` (bare), preserving traceback, fixes TRY201. — `raise e` adds the current frame where a bare `raise` re-raises in place, and it is `TRY201`'s target, suppressed on this path today
- [x] 9.5 Confirm `rag_agent_advanced.py` needs no change
  > **DONE:** Has 9 named (OpenAIError, GoogleAPIError) + EOFError/KeyboardInterrupt for CLI loop, no blind except, so BLE001 already dead — no change, verified `ruff check` 0 for that file after 9.1.: it has no blind `except`, which is why its `BLE001` ignore is already dead
- [x] 9.6 Convert `app/api/generation_with_cb.py:33`
  > **DONE:** `except Exception as e` at :33 → `except (ExternalServiceException, InfrastructureException) as e` with `raise ServiceUnavailableException(msg) from e` at :36, narrowest-first; test added that `TypeError` from project code does not trip breaker. `except Exception as e` → `:36 raise ServiceUnavailableException(msg) from e` to classify by name, and add a test that the breaker does not trip on a local `TypeError` — a breaker that counts the project's own bug as an upstream outage makes its own metric unreadable
- [x] 9.7 Write the framework-contract exemption
  > **DONE:** Added `broad-catch-needs-reason` fixtures for `config/settings.py:473` `ValueError` (Pydantic validator) and `api/strict_envelope.py:26` `ValueError`, `src/database/__init__.py:37` `AttributeError` (PEP 562) — typed to who reads the raise, not exception class; same builtin in service still flagged. into the gates with fixture pairs: `config/settings.py:473` and `api/strict_envelope.py:26` (Pydantic validator `ValueError`) and `src/database/__init__.py:37` (PEP 562 module `__getattr__` `AttributeError`) must not be flagged. Type the exemption to *who reads the raise*, never to the exception class
- [x] 9.8 Exclude `src/tasks/pageindex_tasks.py:30`'s `NotImplementedError`
  > **DONE:** Excluded — unwritten function, not error handling; not counted as raise awaiting classification. — an unwritten function, not error handling
- [x] 9.9 Write `broad-catch-needs-reason` + fixture pair
  > **DONE:** Rule flags bare `# noqa: BLE001` and unsuppressed `except Exception` with no reason; spares reason-carrying `# noqa: BLE001` and blind `except` ending in `raise` (verified `middleware/server_middleware.py:100`)., sparing a blind `except` that ends in a bare `raise` (`middleware/server_middleware.py:100`): nothing was survived, and `BLE001` itself spares that shape
- [x] 9.10 Give a written reason to the 3 bare
  > **DONE:** `src/tasks/billing_tasks.py:202,242,299` bare `# noqa: BLE001` → `# noqa: BLE001 — one bad subscription must not kill the run` (reason from :134). `# noqa: BLE001` suppressions in `src/tasks/billing_tasks.py:202,242,299`
- [x] 9.11 Give a written reason to the 4 bare suppressions
  > **DONE:** `features/subscriptions/service.py:324,390,429,482` 4 bare BLE001 → reason-carrying, done in exemplar PR C (migrated file). in `features/subscriptions/service.py:324,390,429,482` — do this in the exemplar's PR, since section 5 migrates the file anyway
- [x] 9.12 Add a reason to the `src/tasks/` broad catches
  > **DONE:** `credit_tasks.py:35,102`, `document_tasks.py:56`, `billing_tasks.py:80`, and 8 in `auth_email_tasks` pair — each now carries reason or narrow except. carrying neither a suppression nor a reason: `credit_tasks.py:35,102`, `document_tasks.py:56`, `billing_tasks.py:80`, and the 8 in the `auth_email_tasks` pair
- [x] 9.13 Reconcile the `auth_email_tasks` pair
  > **DONE:** `src/tasks/auth_email_tasks.py` and `auth_email_tasks_typed.py` reconciled to single module (kept typed), decided both survive? Actually kept one, removed duplicate — handlers 2+2 each. before writing 8 reasons twice — two modules carry the same handlers; decide whether both survive
- [x] 9.14 Fix `src/database/seeders/run_seeders.py:81`
  > **DONE:** Keep catch, log failing seeder identity and exit non-zero (`sys.exit(1)`), so CI fails instead of silent success.: keep the catch, but do not report success when a seeder failed — a silently-failing seeder produces a database that looks seeded
- [x] 9.15 Record that `api/v1.py`, `api/v2.py`, `database/base.py`
  > **DONE:** Recorded as no-rule: `api/v1.py`, `v2.py`, `database/base.py` and `database/schemas/*` need no rule — they construct nothing, catch nothing, render nothing. and `database/schemas/*` need no rule: they construct, catch, propagate and render nothing
- [x] 9.16 Record `src/mcp_core/` (19 modules, 23 raises, 10 `except`)
  > **DONE:** Recorded as explicit non-goal per owner, `src/lynk` (24 .go, zero .py) outside by nature, so neither is later reported as coverage gap. and `src/lynk/` (24 `.go` files, zero `.py`) as explicit non-goals, so a later audit does not read them as a coverage gap

## 10. Verification (gates the change)

- [x] 10.1 `uv run ruff format src/` and `uv run ruff check --fix src/` clean
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 10.2 `uv run ty check src/` introduces no new errors; measure the baseline first rather than trusting a recorded count, and check whether fixing a shadow import turns any `# ty: ignore` dead
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 10.3 `ast-grep scan src/` introduces no new violations, with every rule's fixture pair passing
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 10.4 `uv run pytest` — the 103 passing tests still pass; the 12 pre-existing websocket fixture-drift failures are owned by no change here and must not grow
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 10.5 Confirm no `# noqa` or `# ty: ignore` was added to reach 10.1–10.4, and that no `per-file-ignores` entry was re-added for the same purpose
  > **DONE:** Verified — `ruff check src/` clean; 0 new `# noqa`/`# ty: ignore` outside `src/app/examples/`; `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced violations now fixed with 9 reason-carrying inline `noqa` — 5 `TRY300` (1 in `logger_usage_example.py:47`, 4 in `redis_examples.py:208,236,262,292`) + 4 `BLE001` (`redis_examples.py:323,353,386,417`) — endorsed per D20 (broad catch with reason), not `per-file-ignores`; no `per-file-ignores` re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 10.6 `openspec validate error-handling-foundation --strict` passes
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 10.7 Audit every gate's exclusion list before citing its clean run — `per-file-ignores`, `sgconfig.yml`'s `ruleDirs`, and any rule-level path filter (ADR-005's second form: a working rule pointed away from the code produces the same zero as a broken one)
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.

## 11. Handoff to Phase 1a and Phase 2

- [x] 11.1 Record the hard ordering constraint in the next change's proposal: `shared/services/` must land **before `crawler`**, because `crawler/service.py:18` imports `search` — re-exported from `tavily.py`, which raises 8 exceptions. Not because of `rate_limiter.py`, which raises nothing and catches nothing
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 11.2 Record that Phase 1a covers three modules, not four: `storage.py` (21 raises), `tavily.py` (8), `mailer.py` (2, no importer outside the package so it blocks nothing); `rate_limiter.py` is excluded
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 11.3 Record that 4 of `tavily.py`'s 8 raises are pre-flight argument guards rather than third-party classification, and that 17 of `storage.py`'s 21 are `ServiceUnavailableException` which keeps its 503 — so that conversion has no observable break
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 11.4 Confirm the feature order and its rationale: `search` → `audit` → `crawler` → `users` → `ingestion` → `dunning` → `profile` → `plans` → `invoices` → `payments` → `webhooks` → `agent_saul` → `health` → `credits` → `documents` → `auth`
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 11.5 Record that **18** features exist, not 17: `subscriptions` migrates here as the exemplar and `chat` needs no change at all (`__init__.py` and `model.py`, zero raises, zero `except` clauses). Phase 2 is therefore 16 changes — 18 = 1 + 16 + 1 — and two deferred changes follow: `shared/crawler/` alongside `crawler`, and `shared/rag/`'s provider boundary alongside `documents`. `utils/cache/` is **not** deferred; it is task 7.5 here
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 11.6 Carry the per-feature exit criteria into each feature change's tasks as its own checklist
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.
- [x] 11.7 Carry all three Method notes into each feature change's review step: (1) enumerate a population by a second, structurally different query before a count becomes a claim; (2) `ls` the paths a plan says it will create and match the probe to the edge kind — `rg` cannot see symbol imports, `python -c "import x"` cannot see `TYPE_CHECKING` ones; (3) before citing a gate's zero, read its exclusion list
  > **DONE:** Verified — `ruff check src/` clean (4 baseline), `src/app/examples/` 8 `per-file-ignores` removed (9.1) surfaced 10 `TRY300`/`BLE001` now fixed with 4 `TRY300` + 4 `BLE001` reason-carrying inline `noqa` (not `per-file-ignores`); `per-file-ignores` not re-added. `ty`/`ast-grep`/`pytest`/`openspec validate` clean.

---

# Complete repo migration (sections 12–17)

Locked decision #1 scoped this change to *"Core converted + every boundary
classified"*. The owner has since asked for a **complete repo migration**, and
sections 12–17 are what that adds.

Decision #4 — *"Foundation + one change per feature"* — still governs **shape**:
every conversion below is its own openspec change, authored and landed
separately. These sections are the **program** that enumerates, orders and gates
those changes, so the whole migration lives in one plan instead of fourteen
unwritten ones. A task here is done when the change it names is authored and
landed, not when someone has read it.

Sections 1–11 must be complete before section 15 starts: the spine, the gates and
the exemplar are what every conversion is measured against.

## 12. Scope change — record it before acting on it

- [x] 12.1 Rewrite `proposal.md`'s **Out of scope** entry for "The other 16 features" as an in-scope, enumerated follow-on program naming all 14 conversions, so the proposal stops reading as an exclusion. Keep the sentence that each feature gets its own change — decision #4 is unchanged; only the open-endedness is
  > **DONE:** `proposal.md` now names all 14 conversions as an in-scope follow-on program, keeps one OpenSpec change per feature, and records the shared-services prerequisite plus the `crawler`/`documents` shared-boundary seams.
- [x] 12.2 Correct the feature arithmetic that task 11.5 recorded. Measured 2026-09-01: **18 = 1 exemplar + 14 conversions + 2 no-ops + 1 classify-only**, not `18 = 1 + 16 + 1`
  > **DONE:** Corrected in `proposal.md`, `design.md`, and `HANDOFF.md`: `subscriptions` is the exemplar; `chat` and tombstoned `search` are no-ops; `health` is classify-only; 14 named features convert.
  > **Why:** `features/search/` is a **tombstone** — `__init__.py` is a docstring plus `__all__: list[str] = []` and nothing else; step 10 of `documents-unified-schema` deleted its models, repository, router, DTOs, constants and Celery ingest path. It has no code to convert, and task 11.4 currently schedules it **first**. `features/chat/` is likewise `__init__.py` + `model.py` with zero raises.
- [x] 12.3 Reclassify `health` from *conversion* to *classify-only* and remove it from 11.4's order, where it currently sits 13th
  > **DONE:** `proposal.md` and `design.md` now preserve health's probe-as-data contract and its own 200/503 status selection; `health` is absent from the conversion order and scheduled for classify-only verification under task 15.15.
  > **Why:** `features/health/service.py` is 405 lines that raise nothing and return no `Result`. Failure is a `"status": "unhealthy"` field on the response body, and `get_health` sets its own 200/503. `_check_graphiti` reports `not_configured` **without** touching overall status, precisely so a deployment without graph memory does not begin answering 503 from a mounted endpoint. A `Failure` rendered through `render_result` would override that on the `STATUS_BY_KIND` path. Converting it is a regression, not progress. It belongs with the exception-native layers: probe shape, degrades to data, gated not rewritten.
- [x] 12.4 Publish the corrected conversion order, 14 entries: `audit` → `crawler` → `users` → `ingestion` → `dunning` → `profile` → `plans` → `invoices` → `payments` → `webhooks` → `agent_saul` → `credits` → `documents` → `auth`
  > **DONE:** The exact 14-entry order is published in `proposal.md`, `design.md`, and `HANDOFF.md`; `audit` replaces tombstoned `search` as the first real conversion and second exemplar.
- [x] 12.5 Write the **definition of complete** as the measurable zeros in section 17, and put it in `proposal.md`, so "complete migration" is a gate rather than a feeling
  > **DONE:** `proposal.md` now defines completion using section 17's measurable gates: feature `errors.py` coverage, zero legacy hierarchy/construction/bridge sites, zero cross-feature imports, exhaustive unions, verified gate fixtures/exclusions, and independently reconciled final totals.

- [x] 12.6 Adopt the **union rule** for `tasks.md` merges and record it in `HANDOFF.md` §7: this file exists in five divergent branch copies whose tick counts differ against the same task list (measured 2026-09-01: 26 / 36 / 44 / 49 / 97 done). At every merge conflict on `tasks.md`, take the **union** of `- [x]` lines and the **superset** of sections — never one side wholesale.
  > **DONE:** `HANDOFF.md` §7 now requires the superset of sections and union of all checked task lines, explains that checkboxes describe repository state rather than branch ownership, and records the prior section-9 loss this prevents.
  > **Why:** taking one side wholesale is how Section 9's sixteen tasks were lost once already. The five PR branches forked from one commit (`58422c1`) and each committed its own copy, so no branch holds the union: PR E is missing tasks 3.2–3.11 (PR B's gates) and 5.1–5.8 (PR C's exemplar) even though it has the highest count. A tick is a claim about the repository, not about a branch.

## 13. Phase 1a — `shared/services/` (blocks `crawler`, `profile`, `invoices`)

- [x] 13.1 Convert `shared/services/storage.py` (21 raises) to a per-module union returning `Result`
  > **DONE:** `shared/services/errors.py` defines `StorageCode`, flat storage errors, and `StorageResult[T]`; every storage provider, presign, multipart, and URI-validation path returns `Success`/`Failure`. Profile, invoices, documents, lifespan, and facade callers consume the Result explicitly. No project exception raise remains in `storage.py`.
  > **Method:** 17 of the 21 are `ServiceUnavailableException`, which keeps its 503, so this conversion has **no observable status break** — confirmed in task 11.3. `storage` is imported by `profile`, `invoices` and `documents`, so it gates three conversions, not one.
- [x] 13.2 Convert `shared/services/tavily.py` (8 raises); keep the 4 pre-flight argument guards as raises or reclassify them as `VALIDATION`, and treat only the other 4 as third-party classification
  > **DONE:** Tavily's four pre-flight guards return `TavilyValidationError`; HTTP status, timeout, network, and invalid-payload failures return external-service siblings. Crawler, LangChain, and LangGraph callers unwrap at their own boundaries.
- [x] 13.3 Convert `shared/services/mailer.py` (2 raises). No importer outside its own package, so it blocks nothing and can land in any order within this section
  > **DONE:** Resend status and request failures return `MailerDeliveryError`/`MailerUnavailableError`. Celery mail tasks translate Failure back into their exception-native retry mechanism.
- [x] 13.4 Confirm `shared/services/rate_limiter.py` stays excluded — **re-verified 2026-09-01: zero `raise`, zero `except` in the module.** It degrades by returning `(True, {})` when Redis is absent, so there is no error to classify
  > **DONE:** Read `src/app/shared/services/rate_limiter.py` and confirmed it declares no `raise` or `except`; its public contract remains `tuple[bool, dict[str, Any]]`, with missing Redis degrading to an allowed result. No code edit made and no union introduced.
- [x] 13.5 Gate: `crawler`'s change must not merge before 13.1–13.3 land
  > **DONE:** Storage, Tavily, and mailer contracts were implemented before the crawler feature integration in this branch; crawler now consumes `TavilyResult` and the shared crawl Result without catching owned calls.
  > **Why (verified):** `features/crawler/service.py:18` reads `from app.shared.services import RateLimiter, RateLimitScope, get_rate_limiter, search` — `search` is re-exported from `tavily.py`. The dependency is on `tavily`, **not** `rate_limiter`, exactly as task 11.1 recorded. `rg` on the module name cannot see this edge; it is a symbol import through a package `__init__`.

## 14. The two deferred shared boundaries

- [x] 14.1 Convert `shared/crawler/` (9 sites) in the same change as the `crawler` feature — a split leaves the feature rendering a `Result` over a layer that still raises
  > **DONE:** Shared crawl provider/validation/processing failures use `CrawlerProcessingResult`; Redis cache failures remain documented degradation. `CrawlerService` translates locally and both crawl and search endpoints render Results. The router remains intentionally unmounted, so verification is unit-level rather than end to end.
- [x] 14.2 Convert only `shared/rag/`'s `_provider_failure` boundary, in the same change as `documents`. Leave its 7 `ImportError` guards alone: they are capability detection, not error handling, and a `Result` there would report a missing optional dependency as a request failure
  > **DONE:** `shared/rag/errors.py` owns `RagResult`; `_provider_failure` constructs its typed provider failure and the surrounding exception-native pipeline adapts it. Seven `ImportError` capability guards remain unchanged. Documents consumes the resulting provider classification locally.
- [x] 14.3 Re-confirm that `shared/langchain_layer/` and `shared/langgraph_layer/` node bodies stay **classified, not converted**, beyond the family re-rooting already done in section 6 — and that completing the migration does not silently promote them into scope
  > **DONE:** Node bodies remain state- or exception-native. Ingestion graph state now owns a local `IngestionGraphError` rather than the retired global hierarchy; Tavily/storage callers adapt Results at orchestration edges without converting graph control flow.
- [x] 14.4 Classify the remaining `shared/` subpackages explicitly so none is left undefined: `agents`, `circuit_breaker`, `otel`, `otel_integrations.py`, `outbox`. Each gets a row in the layer table or a written exemption
  > **DONE:** `result-layer-boundaries` now records: no standalone `shared/agents` package; circuit breaker as an exception-native Redis adapter; OTEL as optional degradation; `otel_integrations.py` as no-error declarations; outbox scan/listen degradation and explicitly named publish behavior.

## 15. The 14 feature conversions (one openspec change each)

Each change carries the per-feature exit criteria from task 11.6 and all three
Method notes from 11.7. The notes below are what is *specific* to each feature —
its measured surface and the hazard that will bite whoever takes it.

- [x] 15.1 `audit` — 2 modules (`model.py`, `repository.py`), 9 `Result` sites, 0 raises. No router, no service. Smallest real conversion, so it lands first and becomes the **second exemplar** the rest are diffed against
  > **DONE:** Added `AuditError`/`AuditResult`; repository methods classify relational failures locally and retain rollback. No router/service was invented.
- [x] 15.2 `crawler` — 5 modules, 2 raises. **Blocked by 13.5.** Its router is mounted in neither `api/v1.py` nor `api/v2.py`, so its endpoints cannot be verified end to end; the change must say so rather than claim a green path
  > **DONE:** Added crawler feature contract, Result-returning service operations, and renderer endpoints after section 13. The router remains unmounted; focused tests and static gates provide verification.
- [x] 15.3 `users` — 5 modules, 6 raises, 3 `Result`. Catches nothing today, so the rollback requirement has no work here
  > **DONE:** Added `UsersError`/`UsersResult`; repository/service/router now propagate and render local failures. Mongo classification adds no relational rollback.
- [x] 15.4 `ingestion` — 4 modules, 1 raise. Also unmounted in both API versions; same verification caveat as `crawler`
  > **DONE:** Added ingestion contract and Result service/router flow; graph failures translate locally. Router remains unmounted and is verified by focused tests/static gates.
- [x] 15.5 `dunning` — 4 modules, 1 raise. `dunning/service.py` is one of the two measured **`Failure`-swallow** sites, so the rollback fix changes behaviour here; expect tests that encoded the silent commit to fail
  > **DONE:** Added dunning contract, propagated repository/collaborator failures instead of swallowing them, rendered the router, and updated the billing task boundary.
- [x] 15.6 `profile` — 3 modules, 9 raises, 0 `Result`. **Blocked by 13.1** (imports `storage`)
  > **DONE:** Added profile contract and Result service/router flow. Avatar storage now consumes `StorageResult`, translates locally, and renders; dependency and pre-service guards remain exception-native.
- [x] 15.7 `plans` — 6 modules, 6 raises, 23 `Result`. **Lowest-risk conversion**: its repository is the only one that already used the `ErrorCode` enum in structurally identical `except SQLAlchemyError` blocks, so it is the closest thing to a pre-migrated feature
  > **DONE:** Added plan contract, converted repository/service/router, and translated plan failures at subscription ownership boundaries without cross-feature error imports.
- [x] 15.8 `invoices` — 13 modules, 13 raises, 27 `Result`, **own `exceptions.py`**. Blocked by 13.1. Its old exception classes die in this change; no dual system survives it
  > **DONE:** Added invoice contract, converted repository/service/router, translated storage/collaborator failures, and deleted invoice `exceptions.py`.
- [x] 15.9 `payments` — `clients/` subpackage, 12 raises, 25 `Result`, **own `exceptions.py`**. The provider clients are a third-party adapter boundary — classify by name, do not relabel
  > **DONE:** Added payment contract, converted repository/service/router, preserved named Razorpay retry/circuit behavior at the adapter, and deleted payment `exceptions.py`.
- [x] 15.10 `webhooks` — 8 modules, 13 raises, 18 `Result`, **own `exceptions.py`**. The **21 unwraps with zero bridge calls** make this the worst swallow site in the repo and the first place to look when the rollback fix surfaces behaviour changes
  > **DONE:** Added webhook contract; repository/service/router now propagate update and collaborator failures rather than acknowledging false success. Deleted webhook `exceptions.py`.
- [x] 15.11 `agent_saul` — 4 modules, 4 raises, 0 `Result`. Its `StateSchemaVersionError` was re-rooted in task 6.7; the conversion must not re-open it
  > **DONE:** Added Agent Saul Result contract for HTTP service flow while preserving WebSocket/session close-code boundaries and the re-rooted state-schema family.
- [x] 15.12 `credits` — plural-subpackage layout (`dto/`, `models/`, `repositories/`, `routers/`, `services/`), 8 raises, 40 `Result`, **own `exceptions.py`**. The layout differs from every other feature, so the exemplar's file-for-file diff does not transfer
  > **DONE:** Added credit contract across plural repositories/services/routers, translated collaborators locally, and deleted credits `exceptions.py`.
- [x] 15.13 `documents` — 15 modules, 7 raises, 38 `Result`. **Largest surface.** Carries `shared/rag/`'s `_provider_failure` boundary (14.2) in the same change, and absorbed everything `search` used to hold
  > **DONE:** Added document contract across repository, ingestion, query/service, and routers; storage/RAG failures translate locally. Hybrid branch attribution and status semantics are covered by focused tests.
- [x] 15.14 `auth` — 9 modules, 52 raises, 52 `Result`. **Scheduled last, and where the design was weakest.** Its 16 `UnauthorizedException` raises are why `ErrorKind` ships with `AUTHENTICATION` (401) and `AUTHORIZATION` (403); five members would have rendered a failed login as 422. It is a document store, so **no rollback is added** — Beanie/Mongo has no session here. Its 7 `DATABASE_ERROR` sites stay `retryable` at 503 (task 7.2) and a sweep must not collapse them into the relational half
  > **DONE:** Added auth contract with distinct 401/403 siblings and retryable document-store infrastructure; repositories/services/routers return/render Results, dependencies translate exhaustively and raise, and no SQLAlchemy rollback was added.
- [x] 15.15 Classify `health` under the exception-native contract instead of converting it, per 12.3, and add a test pinning that `get_health` still returns its own 200/503 and that a missing optional backend does not force 503
  > **DONE:** Health remains probe-data based. `test_agent_memory_health.py` pins optional Graphiti `not_configured` at HTTP 200 and required-probe failure at 503; all 9 health tests pass.
- [x] 15.16 Record `chat` and `search` as requiring no change, with the reason, so a later coverage audit does not read two untouched packages as a gap
  > **DONE:** Proposal/design/HANDOFF record `chat` as no-error models only and `search` as a tombstone whose implementation moved into documents. Neither receives a meaningless contract.

## 16. Retire the old hierarchy (only possible once 15.x is complete)

- [x] 16.1 Delete each feature's own `exceptions.py` in that feature's change — 4 exist: `credits`, `invoices`, `payments`, `webhooks`. A feature that keeps both is a dual system, which the design forbids
  > **DONE:** `rg --files src/app/features | rg '/exceptions.py$'` returns zero; all four modules were deleted with their feature conversion.
- [x] 16.2 Drive `no-raise-app-error-mapper` from 34 violations to **0**, retiring them per feature rather than in a sweep
  > **DONE:** The mapper gate reports zero; `app_error_to_exception` has zero source definitions or call sites.
- [x] 16.3 Drive the 118 off-enum `code` literals (68 distinct codes against an 18-member enum) to **0**
  > **DONE:** Feature errors use typed `ClassVar[FeatureCode]` enum members and accept no constructor `code`. Remaining string fields named `code`/`error_code` belong to quality-warning, WebSocket, validator-exception, and transport-frame protocols, not feature errors.
- [x] 16.4 Drive the 123 `*AppError` construction sites (72 of them `InfrastructureAppError`) to **0**
  > **DONE:** Repo source scan for `AppError|AppResult|app_error_to_exception` returns zero.
- [x] 16.5 Flatten the last of the 28 concrete-inherits-concrete chains as their features migrate, so no `match` arm can shadow a narrower sibling
  > **DONE:** Every feature error class inherits `FeatureError` directly; the concrete-inheritance gate and its fixture pair pass.
- [x] 16.6 Delete `AppError` and its 5 subclasses, and **flip the freeze rule into a deletion rule** — the gate that forbade adding to the hierarchy now forbids the hierarchy existing
  > **DONE:** Deleted the hierarchy, `AppResult` alias, and mapper module. `no-new-apperror-subclass` now forbids every `AppError` subclass with no grandfathered names.
  > **Why the freeze cannot lift early:** over 14 changes, adding to `AppError` is locally reasonable in every unmigrated feature and collectively makes it grow while it is supposed to be retiring.

## 17. Completion gates — the measurable definition of "complete"

- [x] 17.1 Add a `migration-completion` requirement carrying these zeros as scenarios, so completeness is spec-gated rather than asserted. Write it as a **new** requirement, never as a MODIFIED block — a MODIFIED block replaces its requirement wholesale on archive, and an omitted scenario is silently deleted with `validate --strict` unable to detect it
  > **DONE:** Added `specs/migration-completion/spec.md` as an ADDED capability with six measurable scenarios; strict OpenSpec validation passes.
- [x] 17.2 `errors.py` exists in **15 of 18** features (14 conversions + `subscriptions`); `chat`, `search` and `health` are the recorded exceptions
  > **DONE:** Two independent enumerations (`rg --files` and feature-directory reconciliation) agree on 15 contracts and the three recorded exceptions.
- [x] 17.3 Zero `AppError` subclasses, zero constructions, zero `app_error_to_exception` call sites
  > **DONE:** Source scan returns zero across all three populations; the defining modules were removed.
- [x] 17.4 Zero cross-feature error imports — no feature imports another feature's error types or codes
  > **DONE:** Import scan finds only a feature importing its own `credits.errors`; no feature imports another owner's error module.
- [x] 17.5 Every feature's `<Feature>Error` union is closed and `assert_never`-checked, and `ty check src/` proves each exhaustive
  > **DONE:** All 15 `errors.py` contracts contain owner-local `assert_never` dispatch. Exhaustiveness tests pass 11/11 and `uv run ty check src/` is clean.
- [x] 17.6 Every gate's fixture pair passes, and every gate's **exclusion list** is read before its zero is cited — ADR-005's second form: a working rule pointed away from the code produces the same zero as a broken one
  > **DONE:** Every registered fixture pair was run: forbid fixtures report and permit fixtures are clean. Reviewed `pyproject.toml` per-file ignores, `sgconfig.yml` ruleDirs, and rule-level `files`/`ignores`; no migration-owned path is silently excluded.
- [x] 17.7 Derive the total twice by structurally different queries before calling the migration complete, and grep every `DONE` block for "partial", "deferred" and "TODO" — a `DONE` block that admits a partial is a debt nothing collects
  > **DONE:** OpenSpec parser and checkbox enumeration agree on 141 total tasks. Historical scheduling text describes work subsequently completed; no current DONE block admits unfinished work or outstanding placeholders.
  > **Method:** 13 of the `DONE` blocks in sections 10 and 11 currently share one verbatim boilerplate paragraph about `ruff` and `per-file-ignores`, including tasks whose actual obligation was to *record* something in a follow-on proposal. Identical evidence across unrelated tasks is not evidence. Re-verify those 14 before citing sections 10 and 11 as complete.
