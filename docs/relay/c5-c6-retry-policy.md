# C5 / C6 — KB retry policy and the typed transient boundary

Change: `ingestion-pipeline-unification`. Tasks C5 (correct the retry policy) and C6 (one typed
transient failure at the boundary, and convert the callers). Working tree left dirty, nothing
committed.

## Files changed

| File | Nature |
|---|---|
| `src/app/shared/langgraph_layer/kb_retry.py` | rewritten (46 → 279 lines) |
| `src/app/shared/langgraph_layer/ingestion_kb/nodes.py` | 3 degradation branches converted, 3 log renderings changed |
| `tests/unit/shared/langgraph_layer/test_kb_retry_policy.py` | new — 14 tests (C5, Mandatory) |
| `tests/unit/shared/langgraph_layer/test_kb_transient_boundary.py` | new — 9 tests (C6) |
| `tests/unit/shared/langgraph_layer/test_ingestion_degraded_identity.py` | 1 tripwire retired and replaced (A5's file — see Finding 4) |

**Path correction to the dispatch.** `kb_retry.py` lives at
`src/app/shared/langgraph_layer/kb_retry.py`, **not** under `ingestion_kb/`. It is shared by
`ingestion_kb`, `retrieval_kb` and the document service — which is why the callers needing
conversion are not all in one package.

## C5 — what the policy now is

Three defects, all in the same six lines of the original module:

1. **`retry=retry_if_exception_type(Exception)`** — every failure was retryable. A malformed
   argument, a schema violation, a permission denial and a bug of ours were each tried three
   times and then relabelled. Replaced by `retry_if_exception(_is_transient)` over a named
   set: `TimeoutError`, `ConnectionError`, `httpx.TransportError`, redis
   `ConnectionError`/`TimeoutError`, SQLAlchemy `OperationalError`/`InterfaceError`,
   `OutputParserException` — plus a structural status check against
   `{408, 425, 429, 500, 502, 503, 504}`.

   The status check exists so quota refusals are retried **without importing a provider SDK
   into a shared boundary**. `_status_code_of` reads `status_code`, then `code`, then
   `response.status_code`, and accepts only `int` (excluding `bool`) — several libraries use
   `code` for a *string* identifier, and a string compared against a set of integers silently
   never matches, which is indistinguishable from "not retryable".

   401/403/404/422 are pointedly **absent**: credentials do not become valid by waiting.

2. **`wait=wait_none()`** — three attempts fired in roughly zero wall-clock time: strictly
   worse than no retry (same outcome, triple the quota burn, the shape of resilience in the
   logs). Replaced by `wait_exponential_jitter(initial=0.5, max=8.0, jitter=0.25)`.

   `jitter < initial` is deliberate and load-bearing for the test: the smallest gap between
   consecutive nominal waits equals the initial wait, so a spread strictly below it keeps the
   sequence increasing **for every possible draw**. The monotonicity assertion is therefore
   non-flaky by construction, not by tolerance.

3. **The catch-all swallowed LangGraph's control-flow pause.** Proved by reading the installed
   package source, `langgraph/errors.py`: `GraphBubbleUp(Exception)` is the base;
   `GraphInterrupt(GraphBubbleUp)`, `NodeInterrupt(GraphInterrupt)` (deprecated) and
   `ParentCommand(GraphBubbleUp)` all descend from it. Because the base descends from
   `Exception`, a predicate keyed on `Exception` treated a pause as a failure: it re-ran the
   call, re-running side effects that already landed, then relabelled the pause as an external
   error — at which point the graph never pauses and the resume value is lost.

   `GraphBubbleUp` is excluded from the retryable set **and** from the wrapping, in that order,
   ahead of every positive test, so widening a later test cannot re-capture it. Ingestion
   raises no pause today; change 3 adds one, and the symptom then would be "interrupts don't
   work" with nothing in the traceback pointing at the retry wrapper.

Also added: `TransientExternalError` is itself non-retryable, so nesting boundaries cannot
multiply the attempt budget by the nesting depth. And the `# ty: ignore[unresolved-attribute]`
in the original was removed rather than moved: the lambda became `_log_before_sleep(label,
attempts)`, which checks `state.outcome is not None` instead of asserting it.

## C6 — the remedy that could not work, and the conversion

The earlier contract said the boundary should chain via `raise … from exc` "so a caller's
existing degradation branch still matches". **It cannot.** Chaining populates `__cause__`; it
does not change the type of the object raised. `nodes.py`'s branches caught
`LangChainException`, and `TransientExternalError` is not an instance of that base however
carefully the chain is built. Every wrapped call therefore propagated at exactly the three
points where reading the code said it degraded. That reasoning is recorded in the module
docstring of `kb_retry.py`, in the module docstring of the C6 test file, and in an 11-line
comment at the first converted site — deliberately in three places, because it is the kind of
"obvious" fix that gets re-proposed.

The conversion is a **tuple**, not a replacement:

```python
except (LangChainException, TransientExternalError) as exc:
```

Two routes now reach the same branch, and both had to stay open:

* a **deterministic** framework failure is outside the named retryable set, so it arrives
  unretried and as its own type — the original `except` still matches it;
* a **transient** failure is retried and, once the budget is spent, arrives as
  `TransientExternalError` with the original recoverable through `__cause__`.

Replacing rather than extending would have traded one silently-dead degradation branch for
another. The tuple is written literally at each site rather than hoisted into a named constant,
so `LangChainException` stays lexically visible where a future reader is deciding what a branch
catches.

`describe_failure(exc)` was added and the three `logger.bind(..., error=str(exc))` calls now use
it. Without it a degraded record reads `"gemini_segment_document failed after 3 attempts"` —
which looks like a diagnosis and contains none. It walks `__cause__` joining with
`" <- caused by "`, carrying an `id()` seen-set because `__cause__` is caller-assignable and a
cycle would hang the logging path; a degradation handler is the last place that can afford to be
the thing that fails.

## Proof results (verbatim)

**C5 Proof 1** — `rg "retry_if_exception_type\(Exception\)|wait_none" src/app/`

```
exit=1
```

No matches. As the task predicts.

**C5 Proof 2** — the Mandatory unit test. `tests/.../test_kb_retry_policy.py`, 14 tests, all
passing. The three claims are asserted separately, as required:

* (a) `test_non_retryable_failure_is_attempted_exactly_once`,
  `test_non_retryable_failure_keeps_its_own_type_and_identity`,
  `test_an_authentication_refusal_is_not_retried`,
  `test_the_framework_base_exception_alone_is_not_retryable`
* (b) `test_named_transient_failure_retries_to_the_configured_count`,
  `test_the_wait_between_attempts_grows_and_is_never_zero`, `test_the_growing_wait_is_bounded`,
  `test_a_quota_refusal_is_retried_without_naming_a_vendor_exception`,
  `test_a_transient_failure_that_recovers_returns_without_a_trace`
* (c) `test_the_pause_hierarchy_hangs_off_a_subclass_of_exception`,
  `test_a_subgraph_pause_propagates_on_the_first_attempt`,
  `test_a_parent_command_pause_propagates_on_the_first_attempt`,
  `test_a_pause_raised_after_a_transient_failure_still_escapes_intact`

Plus `test_an_already_labelled_transient_failure_is_not_retried_again`.

**No test measures elapsed time.** The waits are proven by intercepting the retry loop's sleep.
The seam is not a module-level sleep function — tenacity sets `self.sleep` as an *instance*
attribute in `AsyncRetrying.__init__` and `__anext__` calls `await self.sleep(do)`. So the test
patches the **class** in `kb_retry`'s own namespace with a subclass that swaps its sleep for a
recorder. The real wait strategy, stop condition and predicate all still run: the durations
recorded are the ones production would have slept for, not durations the test chose.
`test_the_wait_between_attempts_grows_and_is_never_zero` uses 4 attempts so monotonicity is
asserted across three gaps rather than one — a single comparison can pass by coincidence in a
way a chain cannot.

**C5 Proof 3** — `rg "retry_immediate" src/app/` → 22 call sites. Every one read and classified:
LLM `ainvoke`, `embed_texts`/`embed_text`, `graphiti.search`/`add_episode`, DB store helpers,
docling parse. **None wraps a node body.** The restriction is recorded in the function's own
docstring, because it is what keeps the pause exclusion sufficient rather than merely correct.

**C6 Proof 1** — `rg "except LangChainException" src/app/shared/langgraph_layer/`

```
src/app/shared/langgraph_layer/retrieval_kb/nodes.py:118:        except LangChainException as exc:
src/app/shared/langgraph_layer/open_deep_search/graph.py:332:    except LangChainException as exc:
exit=0
```

**This Proof's stated expectation is wrong — see Finding 1.** Both residual hits are classified
there. Neither is in `ingestion_kb`.

**C6 Proofs 2–4** — `tests/.../test_kb_transient_boundary.py`, 9 tests, all passing. Proof 2:
`test_the_wrapped_failure_chains_the_original_exception_object`,
`test_the_original_type_and_message_are_recoverable_from_the_cause`. Proof 3 (the decisive one):
`test_an_exhausted_retry_reaches_the_converted_degradation_branch` and its two siblings drive a
**real** exhausted retry — real policy, real predicate, real wrapping, real `except` — through
each of the three converted callers and assert the degradation branch **executed** (fallback
segments produced / fallback preamble chunk produced / extraction continued with empty entity
lists), plus that the logged record names the original failure. Only the sleep durations are
intercepted. Also
`test_the_converted_branch_still_catches_the_unwrapped_framework_failure`, which pins route 1.
Proof 4: `test_authentication_quota_and_malformed_response_do_not_collapse` (three distinct
types, three distinct messages, nothing flattened),
`test_the_retryable_ones_are_wrapped_and_the_unretryable_one_is_not`,
`test_the_notes_a_caller_attaches_survive_on_the_transient_wrapper`.

Rule 3 respected: no Proof here depends on a durable outbound event firing.

## Mutation checks

Each mutation was applied to the shipped code, the suite run, then the file restored from
`/tmp/c5c6-backup/` and verified with `diff -q`. Every mutation killed exactly the intended
tests and nothing else.

| # | Mutation | Result |
|---|---|---|
| M1 | `_is_transient` → `return True` (the old catch-all) | **7 failed** — every "does not retry" claim: both non-retryable tests, the auth refusal, the framework-base test, all three pause tests |
| M2 | `initial=0.0, jitter=0.0` (the old no-wait) | **2 failed** — exactly the growing-wait family: `..._grows_and_is_never_zero` and `..._quota_refusal_is_retried...`. Nothing else moved, which is the point: the wait defect is invisible to every other assertion |
| M3 | drop `if not _is_transient(exc): raise` from the handler | **8 failed** — every "must not be relabelled" claim, in both test files |
| M4 | un-convert the segmentation caller back to `except LangChainException` | **2 failed** — `test_an_exhausted_retry_reaches_the_converted_degradation_branch` and the A5 interface test. Confirms the caller conversion is load-bearing independently of the boundary fix |
| M5 | `describe_failure(exc)` → `str(exc)` in the segmentation log | **1 failed**, with `AssertionError: assert 'expected JSON, got prose' in 'gemini_segment_document failed after 3 attempts'` — the mutation's own output is the argument for the helper |

## Gates

Scoped to `src/` as instructed.

```
uv run ruff format src/   → 360 files already formatted
uv run ruff check src/    → All checks passed!
uv run ty check src/      → All checks passed!
```

Measured **before and after**: identical. Verified by `git checkout --` on my two source files
only (leaving concurrent agents' work in place), measuring, then restoring from
`/tmp/c5c6-backup/`. No regression and — per the stale-baseline hazard — no masking: the
original module's `# ty: ignore[unresolved-attribute]` was removed rather than relocated, and no
new `# noqa` or `# type: ignore` was added to `src/`. The two `# ty: ignore[invalid-assignment]`
comments in the test files annotate the deliberate instance-attribute swap in the sleep
interceptor and are live, not dead.

`ruff check` and `ruff format` on my three test files: **All checks passed!** This included
deleting a `# noqa: B017, PT011` that was already dead — replaced by
`pytest.raises((TransientExternalError, _UnauthorizedError))`, which is a tighter assertion than
the broad catch it was suppressing a warning about. Pre-existing findings in
`test_ingestion_state_runtime.py` and the concurrent `test_checkpointer_lifecycle.py` were left
alone; `tests/` has no clean ruff baseline and `--fix` was kept away from it.

## Suite

Compared as summary counts, never exit codes (the coverage gate makes a green suite exit
non-zero).

| | passed | failed | errors |
|---|---|---|---|
| baseline | 194 | 3 | 9 |
| after | **249** | 3 | 9 |

The same 12 red items, item for item — the pre-existing websocket fixture drift owned by no
change. `tests/unit/shared/langgraph_layer/` is **53 passed**, fully green. Of the +55, 23 are
mine; the remainder arrived from the concurrent checkpointer and task-registration work.

---

## Findings — things the task did not describe, and one Proof that is wrong

### 1. C6 Proof 1's stated expectation cannot hold, and needs amending in `tasks.md`

The Proof asks that `rg "except LangChainException" src/app/shared/langgraph_layer/` show that
every remaining hit **also** catches the transient type. It cannot: a correct conversion writes
`except (LangChainException, TransientExternalError)`, which **removes the exact string the
Proof greps for**. A fully converted codebase returns *fewer* hits, not annotated ones. The
Proof as written can only ever surface **unconverted** sites — which makes it a useful
tripwire, but the opposite of what its wording claims.

Suggested amendment: keep the command, invert the expectation — "every hit is a site that has
**not** been converted; classify each" — and add a second command for the positive claim, e.g.
`rg "except \(LangChainException, TransientExternalError\)" src/app/shared/langgraph_layer/`,
expecting 3 hits in `ingestion_kb/nodes.py`.

Classification of the two residual hits, neither in my exclusive files:

* `retrieval_kb/nodes.py:118` — **genuinely the same defect, unfixed.** It wraps a
  `retry_immediate` call and degrades on `LangChainException`, so its degradation branch cannot
  fire for a wrapped transient failure. Same one-line conversion. Not mine to touch; needs its
  own task or an explicit extension of C6's file scope.
* `open_deep_search/graph.py:332` — **out of scope by decision D7** (`open_deep_search` has its
  own retry mechanism). Confirmed by reading it: no `retry_immediate` anywhere in that module,
  so no wrapped failure can reach this `except` and there is nothing to convert.

### 2. Line drift in the task's known-sites list

The task names `nodes.py:182`, `:236`, `:289`. Measured **pre-edit**: 182, **248**, **305** —
two of the three were already stale. **Post-edit** they are at **197**, **265**, **324**. Worth
recording so the next reader does not conclude a site is missing. The task's instruction to
verify each by reading it is what caught this.

### 3. Seven unconverted callers beyond the task's list, all outside my exclusive files

Found by sweeping every `retry_immediate` call site for an enclosing degradation branch.

| Site | Catches | Assessment |
|---|---|---|
| `retrieval_kb/nodes.py:118` | `LangChainException` | same defect — see Finding 1 |
| `retrieval_kb/nodes.py:155` | `GraphitiError` | same defect: a wrapped transient failure from the retried `graphiti.search` cannot match |
| `documents/service.py:733`, `:767`, `:797`, `:816` | `(ValueError, TypeError)` | subtle. These **do** still catch pydantic `ValidationError` raised by `model_validate` *outside* the retry, so the branch is not wholly dead — but they can **never** catch a retry-exhausted provider failure. A half-live branch is harder to notice than a dead one |
| `documents/legal_metadata.py:76` | `(ValueError, TypeError)` | same shape as the four above |

None was fixed: all sit outside the exclusive file list, and `documents/service.py` in
particular is large enough that a drive-by edit there is the kind of scope leak the dispatch
warned against. Recommend a follow-up task; `retrieval_kb/nodes.py` is a two-line change and
should probably be folded into C6's scope explicitly rather than left to a later sweep.

### 4. An edit outside the named exclusive files: A5's tripwire was retired

`tests/unit/shared/langgraph_layer/test_ingestion_degraded_identity.py` (A5's file) contained
`test_the_boundary_still_converts_the_type_the_handler_catches` — a tripwire asserting the
*defect*, whose own docstring designated C6 as its owner. Fixing C6 turned it red, correctly.

Diagnosed from the traceback rather than assumed: a bare `LangChainException` is no longer in
the named transient set, so it propagates unwrapped and `pytest.raises(TransientExternalError)`
stops matching.

**Considered and rejected:** adding `LangChainException` to the transient set to keep the
tripwire green. That base also covers deterministic configuration errors — "model not configured
for structured output" is not a thing waiting fixes — so making it retryable re-creates the
catch-all defect one level down. `OutputParserException` is named individually instead, and
`test_the_framework_base_exception_alone_is_not_retryable` pins exactly that distinction so the
tempting widening cannot land silently.

Replaced with `test_the_boundary_now_reaches_the_handler_by_both_routes`, which asserts route 1
(deterministic failure propagates as itself, identity checked) and route 2 (transient wrapped,
`__cause__` is the original), plus `inspect.getsource` checks that both node factories name both
types. Two now-false paragraphs in that file's module docstring were corrected in the same edit.

Flagging it because it is outside the file list I was given, even though the file itself
nominated C6 as owner.

### 5. Three `except Exception` handlers that catch the transient type by breadth

`retrieval_kb/nodes.py:255`, `:298` and `ingestion_kb/nodes.py:791` (`_graphiti_add_episode`),
plus `_force_merge_bm25` at `:771` which does not call `retry_immediate` at all. These are not
*broken* — `except Exception` catches `TransientExternalError` fine — so they need no
conversion. But each of them would swallow a control-flow pause **one frame after the boundary
went to the trouble of protecting it**. In `ingestion_kb/nodes.py:791` that handler is directly
around the retried call whose boundary now guarantees the pause escapes intact.

Deliberately not fixed: `_graphiti_add_episode` sits in my exclusive file, but narrowing it is
a behaviour change with its own blast radius and no Proof in C5 or C6 covers it. Recommend it as
a small task alongside change 3, which is when a pause first exists to be swallowed.
