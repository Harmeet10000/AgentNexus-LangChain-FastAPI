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
  > **DONE (partial — remainder deferred per phase split):** `.ast-grep/rules/no-match-on-result.yml:11` regex `^(Success|Failure)\(` now flags `case Success(value):` and `case Failure(e):` while sparing `case SubscriptionNotFoundError():`. Verified: `ast-grep scan --rule ... /tmp/forbid.py` flags, `/tmp/permit.py` clean. Full violation re-measure and remaining 3.2-3.11 deferred to next commit per spine+renderer-first split.
  > **Re-measure completed 2026-08-31 (the deferred half of 3.1):** the corrected rule reports **0 violations in `src/`, 0 in `tests/`**. Reconciled with a structurally different query — `rg 'case\s+(Success|Failure)\s*\('` over the same trees also returns 0 — so this zero is real, not ADR-005's "the rule looked for something nobody writes". No historical count carried forward. Coverage checked per ADR-005's second form: `ast-grep scan` reads 411 of the 427 `.py` files under `src/ tests/`, and the 16 it skips are exactly the 16 zero-byte `__init__.py` files; `sgconfig.yml` declares no path exclusion. Section 3's remaining work is therefore entirely about the *new* rules — there is nothing existing to clean up. The committed fixture pair this rule still lacks is task 3.2.
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

## 9. The five later-added directories (design D20 — exemptions, not conversions)

- [ ] 9.1 Remove the 8 `per-file-ignores` entries for `src/app/examples/*.py` from `pyproject.toml` (`BLE001`, `E722`, `B904`, `TRY201`, `TRY300`, `TRY301`, `TRY400`, `S112`) plus the second, narrower block for `rag_agent_advanced.py` whose `BLE001` is already dead, and fix what surfaces rather than re-suppressing it
- [ ] 9.2 Fix the 4 `raise HTTPException` sites in `src/app/examples/redis_examples.py` that `ast-grep`'s `no-raw-httpexception` already reports at `error` level — the only gate there with no per-path ignore
- [ ] 9.3 Move `redis_examples.py`'s 8 `except DatabaseException` catches in the **same commit** as task 7.5's reclassification of `utils/cache/redis_func.py`; split across two changes, the example catches an exception nothing raises and stops handling anything without failing
- [ ] 9.4 Replace `raise e` with a bare `raise` where an example re-raises — `raise e` adds the current frame where a bare `raise` re-raises in place, and it is `TRY201`'s target, suppressed on this path today
- [ ] 9.5 Confirm `rag_agent_advanced.py` needs no change: it has no blind `except`, which is why its `BLE001` ignore is already dead
- [ ] 9.6 Convert `app/api/generation_with_cb.py:33` `except Exception as e` → `:36 raise ServiceUnavailableException(msg) from e` to classify by name, and add a test that the breaker does not trip on a local `TypeError` — a breaker that counts the project's own bug as an upstream outage makes its own metric unreadable
- [ ] 9.7 Write the framework-contract exemption into the gates with fixture pairs: `config/settings.py:473` and `api/strict_envelope.py:26` (Pydantic validator `ValueError`) and `src/database/__init__.py:37` (PEP 562 module `__getattr__` `AttributeError`) must not be flagged. Type the exemption to *who reads the raise*, never to the exception class
- [ ] 9.8 Exclude `src/tasks/pageindex_tasks.py:30`'s `NotImplementedError` — an unwritten function, not error handling
- [ ] 9.9 Write `broad-catch-needs-reason` + fixture pair, sparing a blind `except` that ends in a bare `raise` (`middleware/server_middleware.py:100`): nothing was survived, and `BLE001` itself spares that shape
- [ ] 9.10 Give a written reason to the 3 bare `# noqa: BLE001` suppressions in `src/tasks/billing_tasks.py:202,242,299`
- [ ] 9.11 Give a written reason to the 4 bare suppressions in `features/subscriptions/service.py:324,390,429,482` — do this in the exemplar's PR, since section 5 migrates the file anyway
- [ ] 9.12 Add a reason to the `src/tasks/` broad catches carrying neither a suppression nor a reason: `credit_tasks.py:35,102`, `document_tasks.py:56`, `billing_tasks.py:80`, and the 8 in the `auth_email_tasks` pair
- [ ] 9.13 Reconcile the `auth_email_tasks` pair before writing 8 reasons twice — two modules carry the same handlers; decide whether both survive
- [ ] 9.14 Fix `src/database/seeders/run_seeders.py:81`: keep the catch, but do not report success when a seeder failed — a silently-failing seeder produces a database that looks seeded
- [ ] 9.15 Record that `api/v1.py`, `api/v2.py`, `database/base.py` and `database/schemas/*` need no rule: they construct, catch, propagate and render nothing
- [ ] 9.16 Record `src/mcp_core/` (19 modules, 23 raises, 10 `except`) and `src/lynk/` (24 `.go` files, zero `.py`) as explicit non-goals, so a later audit does not read them as a coverage gap

## 10. Verification (gates the change)

- [ ] 10.1 `uv run ruff format src/` and `uv run ruff check --fix src/` clean
- [ ] 10.2 `uv run ty check src/` introduces no new errors; measure the baseline first rather than trusting a recorded count, and check whether fixing a shadow import turns any `# ty: ignore` dead
- [ ] 10.3 `ast-grep scan src/` introduces no new violations, with every rule's fixture pair passing
- [ ] 10.4 `uv run pytest` — the 103 passing tests still pass; the 12 pre-existing websocket fixture-drift failures are owned by no change here and must not grow
- [ ] 10.5 Confirm no `# noqa` or `# ty: ignore` was added to reach 10.1–10.4, and that no `per-file-ignores` entry was re-added for the same purpose
- [ ] 10.6 `openspec validate error-handling-foundation --strict` passes
- [ ] 10.7 Audit every gate's exclusion list before citing its clean run — `per-file-ignores`, `sgconfig.yml`'s `ruleDirs`, and any rule-level path filter (ADR-005's second form: a working rule pointed away from the code produces the same zero as a broken one)

## 11. Handoff to Phase 1a and Phase 2

- [ ] 11.1 Record the hard ordering constraint in the next change's proposal: `shared/services/` must land **before `crawler`**, because `crawler/service.py:18` imports `search` — re-exported from `tavily.py`, which raises 8 exceptions. Not because of `rate_limiter.py`, which raises nothing and catches nothing
- [ ] 11.2 Record that Phase 1a covers three modules, not four: `storage.py` (21 raises), `tavily.py` (8), `mailer.py` (2, no importer outside the package so it blocks nothing); `rate_limiter.py` is excluded
- [ ] 11.3 Record that 4 of `tavily.py`'s 8 raises are pre-flight argument guards rather than third-party classification, and that 17 of `storage.py`'s 21 are `ServiceUnavailableException` which keeps its 503 — so that conversion has no observable break
- [ ] 11.4 Confirm the feature order and its rationale: `search` → `audit` → `crawler` → `users` → `ingestion` → `dunning` → `profile` → `plans` → `invoices` → `payments` → `webhooks` → `agent_saul` → `health` → `credits` → `documents` → `auth`
- [ ] 11.5 Record that **18** features exist, not 17: `subscriptions` migrates here as the exemplar and `chat` needs no change at all (`__init__.py` and `model.py`, zero raises, zero `except` clauses). Phase 2 is therefore 16 changes — 18 = 1 + 16 + 1 — and two deferred changes follow: `shared/crawler/` alongside `crawler`, and `shared/rag/`'s provider boundary alongside `documents`. `utils/cache/` is **not** deferred; it is task 7.5 here
- [ ] 11.6 Carry the per-feature exit criteria into each feature change's tasks as its own checklist
- [ ] 11.7 Carry all three Method notes into each feature change's review step: (1) enumerate a population by a second, structurally different query before a count becomes a claim; (2) `ls` the paths a plan says it will create and match the probe to the edge kind — `rg` cannot see symbol imports, `python -c "import x"` cannot see `TYPE_CHECKING` ones; (3) before citing a gate's zero, read its exclusion list

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

- [ ] 12.1 Rewrite `proposal.md`'s **Out of scope** entry for "The other 16 features" as an in-scope, enumerated follow-on program naming all 14 conversions, so the proposal stops reading as an exclusion. Keep the sentence that each feature gets its own change — decision #4 is unchanged; only the open-endedness is
- [ ] 12.2 Correct the feature arithmetic that task 11.5 recorded. Measured 2026-09-01: **18 = 1 exemplar + 14 conversions + 2 no-ops + 1 classify-only**, not `18 = 1 + 16 + 1`
  > **Why:** `features/search/` is a **tombstone** — `__init__.py` is a docstring plus `__all__: list[str] = []` and nothing else; step 10 of `documents-unified-schema` deleted its models, repository, router, DTOs, constants and Celery ingest path. It has no code to convert, and task 11.4 currently schedules it **first**. `features/chat/` is likewise `__init__.py` + `model.py` with zero raises.
- [ ] 12.3 Reclassify `health` from *conversion* to *classify-only* and remove it from 11.4's order, where it currently sits 13th
  > **Why:** `features/health/service.py` is 405 lines that raise nothing and return no `Result`. Failure is a `"status": "unhealthy"` field on the response body, and `get_health` sets its own 200/503. `_check_graphiti` reports `not_configured` **without** touching overall status, precisely so a deployment without graph memory does not begin answering 503 from a mounted endpoint. A `Failure` rendered through `render_result` would override that on the `STATUS_BY_KIND` path. Converting it is a regression, not progress. It belongs with the exception-native layers: probe shape, degrades to data, gated not rewritten.
- [ ] 12.4 Publish the corrected conversion order, 14 entries: `audit` → `crawler` → `users` → `ingestion` → `dunning` → `profile` → `plans` → `invoices` → `payments` → `webhooks` → `agent_saul` → `credits` → `documents` → `auth`
- [ ] 12.5 Write the **definition of complete** as the measurable zeros in section 17, and put it in `proposal.md`, so "complete migration" is a gate rather than a feeling

- [ ] 12.6 Adopt the **union rule** for `tasks.md` merges and record it in `HANDOFF.md` §7: this file exists in five divergent branch copies whose tick counts differ against the same task list (measured 2026-09-01: 26 / 36 / 44 / 49 / 97 done). At every merge conflict on `tasks.md`, take the **union** of `- [x]` lines and the **superset** of sections — never one side wholesale.
  > **Why:** taking one side wholesale is how Section 9's sixteen tasks were lost once already. The five PR branches forked from one commit (`58422c1`) and each committed its own copy, so no branch holds the union: PR E is missing tasks 3.2–3.11 (PR B's gates) and 5.1–5.8 (PR C's exemplar) even though it has the highest count. A tick is a claim about the repository, not about a branch.

## 13. Phase 1a — `shared/services/` (blocks `crawler`, `profile`, `invoices`)

- [ ] 13.1 Convert `shared/services/storage.py` (21 raises) to a per-module union returning `Result`
  > **Method:** 17 of the 21 are `ServiceUnavailableException`, which keeps its 503, so this conversion has **no observable status break** — confirmed in task 11.3. `storage` is imported by `profile`, `invoices` and `documents`, so it gates three conversions, not one.
- [ ] 13.2 Convert `shared/services/tavily.py` (8 raises); keep the 4 pre-flight argument guards as raises or reclassify them as `VALIDATION`, and treat only the other 4 as third-party classification
- [ ] 13.3 Convert `shared/services/mailer.py` (2 raises). No importer outside its own package, so it blocks nothing and can land in any order within this section
- [ ] 13.4 Confirm `shared/services/rate_limiter.py` stays excluded — **re-verified 2026-09-01: zero `raise`, zero `except` in the module.** It degrades by returning `(True, {})` when Redis is absent, so there is no error to classify
- [ ] 13.5 Gate: `crawler`'s change must not merge before 13.1–13.3 land
  > **Why (verified):** `features/crawler/service.py:18` reads `from app.shared.services import RateLimiter, RateLimitScope, get_rate_limiter, search` — `search` is re-exported from `tavily.py`. The dependency is on `tavily`, **not** `rate_limiter`, exactly as task 11.1 recorded. `rg` on the module name cannot see this edge; it is a symbol import through a package `__init__`.

## 14. The two deferred shared boundaries

- [ ] 14.1 Convert `shared/crawler/` (9 sites) in the same change as the `crawler` feature — a split leaves the feature rendering a `Result` over a layer that still raises
- [ ] 14.2 Convert only `shared/rag/`'s `_provider_failure` boundary, in the same change as `documents`. Leave its 7 `ImportError` guards alone: they are capability detection, not error handling, and a `Result` there would report a missing optional dependency as a request failure
- [ ] 14.3 Re-confirm that `shared/langchain_layer/` and `shared/langgraph_layer/` node bodies stay **classified, not converted**, beyond the family re-rooting already done in section 6 — and that completing the migration does not silently promote them into scope
- [ ] 14.4 Classify the remaining `shared/` subpackages explicitly so none is left undefined: `agents`, `circuit_breaker`, `otel`, `otel_integrations.py`, `outbox`. Each gets a row in the layer table or a written exemption

## 15. The 14 feature conversions (one openspec change each)

Each change carries the per-feature exit criteria from task 11.6 and all three
Method notes from 11.7. The notes below are what is *specific* to each feature —
its measured surface and the hazard that will bite whoever takes it.

- [ ] 15.1 `audit` — 2 modules (`model.py`, `repository.py`), 9 `Result` sites, 0 raises. No router, no service. Smallest real conversion, so it lands first and becomes the **second exemplar** the rest are diffed against
- [ ] 15.2 `crawler` — 5 modules, 2 raises. **Blocked by 13.5.** Its router is mounted in neither `api/v1.py` nor `api/v2.py`, so its endpoints cannot be verified end to end; the change must say so rather than claim a green path
- [ ] 15.3 `users` — 5 modules, 6 raises, 3 `Result`. Catches nothing today, so the rollback requirement has no work here
- [ ] 15.4 `ingestion` — 4 modules, 1 raise. Also unmounted in both API versions; same verification caveat as `crawler`
- [ ] 15.5 `dunning` — 4 modules, 1 raise. `dunning/service.py` is one of the two measured **`Failure`-swallow** sites, so the rollback fix changes behaviour here; expect tests that encoded the silent commit to fail
- [ ] 15.6 `profile` — 3 modules, 9 raises, 0 `Result`. **Blocked by 13.1** (imports `storage`)
- [ ] 15.7 `plans` — 6 modules, 6 raises, 23 `Result`. **Lowest-risk conversion**: its repository is the only one that already used the `ErrorCode` enum in structurally identical `except SQLAlchemyError` blocks, so it is the closest thing to a pre-migrated feature
- [ ] 15.8 `invoices` — 13 modules, 13 raises, 27 `Result`, **own `exceptions.py`**. Blocked by 13.1. Its old exception classes die in this change; no dual system survives it
- [ ] 15.9 `payments` — `clients/` subpackage, 12 raises, 25 `Result`, **own `exceptions.py`**. The provider clients are a third-party adapter boundary — classify by name, do not relabel
- [ ] 15.10 `webhooks` — 8 modules, 13 raises, 18 `Result`, **own `exceptions.py`**. The **21 unwraps with zero bridge calls** make this the worst swallow site in the repo and the first place to look when the rollback fix surfaces behaviour changes
- [ ] 15.11 `agent_saul` — 4 modules, 4 raises, 0 `Result`. Its `StateSchemaVersionError` was re-rooted in task 6.7; the conversion must not re-open it
- [ ] 15.12 `credits` — plural-subpackage layout (`dto/`, `models/`, `repositories/`, `routers/`, `services/`), 8 raises, 40 `Result`, **own `exceptions.py`**. The layout differs from every other feature, so the exemplar's file-for-file diff does not transfer
- [ ] 15.13 `documents` — 15 modules, 7 raises, 38 `Result`. **Largest surface.** Carries `shared/rag/`'s `_provider_failure` boundary (14.2) in the same change, and absorbed everything `search` used to hold
- [ ] 15.14 `auth` — 9 modules, 52 raises, 52 `Result`. **Scheduled last, and where the design was weakest.** Its 16 `UnauthorizedException` raises are why `ErrorKind` ships with `AUTHENTICATION` (401) and `AUTHORIZATION` (403); five members would have rendered a failed login as 422. It is a document store, so **no rollback is added** — Beanie/Mongo has no session here. Its 7 `DATABASE_ERROR` sites stay `retryable` at 503 (task 7.2) and a sweep must not collapse them into the relational half
- [ ] 15.15 Classify `health` under the exception-native contract instead of converting it, per 12.3, and add a test pinning that `get_health` still returns its own 200/503 and that a missing optional backend does not force 503
- [ ] 15.16 Record `chat` and `search` as requiring no change, with the reason, so a later coverage audit does not read two untouched packages as a gap

## 16. Retire the old hierarchy (only possible once 15.x is complete)

- [ ] 16.1 Delete each feature's own `exceptions.py` in that feature's change — 4 exist: `credits`, `invoices`, `payments`, `webhooks`. A feature that keeps both is a dual system, which the design forbids
- [ ] 16.2 Drive `no-raise-app-error-mapper` from 34 violations to **0**, retiring them per feature rather than in a sweep
- [ ] 16.3 Drive the 118 off-enum `code` literals (68 distinct codes against an 18-member enum) to **0**
- [ ] 16.4 Drive the 123 `*AppError` construction sites (72 of them `InfrastructureAppError`) to **0**
- [ ] 16.5 Flatten the last of the 28 concrete-inherits-concrete chains as their features migrate, so no `match` arm can shadow a narrower sibling
- [ ] 16.6 Delete `AppError` and its 5 subclasses, and **flip the freeze rule into a deletion rule** — the gate that forbade adding to the hierarchy now forbids the hierarchy existing
  > **Why the freeze cannot lift early:** over 14 changes, adding to `AppError` is locally reasonable in every unmigrated feature and collectively makes it grow while it is supposed to be retiring.

## 17. Completion gates — the measurable definition of "complete"

- [ ] 17.1 Add a `migration-completion` requirement carrying these zeros as scenarios, so completeness is spec-gated rather than asserted. Write it as a **new** requirement, never as a MODIFIED block — a MODIFIED block replaces its requirement wholesale on archive, and an omitted scenario is silently deleted with `validate --strict` unable to detect it
- [ ] 17.2 `errors.py` exists in **15 of 18** features (14 conversions + `subscriptions`); `chat`, `search` and `health` are the recorded exceptions
- [ ] 17.3 Zero `AppError` subclasses, zero constructions, zero `app_error_to_exception` call sites
- [ ] 17.4 Zero cross-feature error imports — no feature imports another feature's error types or codes
- [ ] 17.5 Every feature's `<Feature>Error` union is closed and `assert_never`-checked, and `ty check src/` proves each exhaustive
- [ ] 17.6 Every gate's fixture pair passes, and every gate's **exclusion list** is read before its zero is cited — ADR-005's second form: a working rule pointed away from the code produces the same zero as a broken one
- [ ] 17.7 Derive the total twice by structurally different queries before calling the migration complete, and grep every `DONE` block for "partial", "deferred" and "TODO" — a `DONE` block that admits a partial is a debt nothing collects
  > **Method:** 13 of the `DONE` blocks in sections 10 and 11 currently share one verbatim boilerplate paragraph about `ruff` and `per-file-ignores`, including tasks whose actual obligation was to *record* something in a follow-on proposal. Identical evidence across unrelated tasks is not evidence. Re-verify those 14 before citing sections 10 and 11 as complete.
