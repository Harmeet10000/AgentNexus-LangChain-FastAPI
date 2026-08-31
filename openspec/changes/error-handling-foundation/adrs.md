# Architectural Decision Records — `error-handling-foundation`

Six decisions from `design.md` outlive this change: other parts of the system build
on them, or a future reader will otherwise re-litigate a trade-off that was settled
by measurement. The remaining twelve decisions arrange existing pieces and stay in
`design.md`.

---

## ADR-001 — Error classification is a class constant on a closed per-feature union

- **Date:** 2026-08-29
- **Change:** `error-handling-foundation`
- **Status:** Accepted

**Context.** `AppError` is an open Pydantic hierarchy whose `code` is a field with a
default. Any construction site can invent or mistype a code, and 118 off-enum
literals emitting 68 distinct codes against an 18-member enum is what that shape
produced. Because anyone can subclass the base anywhere, no type checker can
enumerate the subclasses, so no dispatch over the hierarchy can ever be proven
total.

**Decision.** `kind`, `code` and `retryable` are `ClassVar` on a frozen
`extra="forbid"` Pydantic base, never constructor parameters. Each feature — and each
shared module that classifies a third-party exception — declares its own `<Owner>Code`
StrEnum, its concrete error types as flat siblings with no inheritance between them,
and a closed `type <Owner>Error = A | B | C` union in its own `errors.py`.

**Rationale / Alternatives.** Measured with the project's `ty`:
`code: ClassVar[SubscriptionCode] = "DUPLICATE_SUBSCRIPTION"` is rejected as
`error[invalid-assignment]` *even though the string value is correct*. A code cannot
be spelled by hand at all — right or wrong — which is a stronger guarantee than a
validator or a lint rule can give. Keeping `code` a field with an enum annotation was
rejected because a default makes the field optional at every call site; a global enum
was rejected because it is the cross-cutting coupling this design removes.

The flat-sibling constraint is the non-obvious half. `match`'s class patterns are
`isinstance`-based, so a broader arm placed before a narrower one makes the narrower
arm dead — and the type checker still reports the match exhaustive, because from its
view every case was covered. It has no concept of *reached but shadowed*. 28 such
chains already exist in the codebase.

**Consequences.**
- A mistyped or wrong-enum code is a type error, not a runtime surprise or a
  client-visible string.
- Adding a failure mode means editing the union, which makes every `match` over it
  fail to compile until the new arm is handled. That is the intended cost.
- No type-checker backstop exists for the flat-sibling rule; it is enforced by an
  ast-grep gate, which is why ADR-005's verification obligation is not optional.
- `extra="forbid"` turns a stale `code=` keyword into a runtime `ValidationError`
  raised inside an `except` block, where it can mask the original error. `ty` catches
  it statically first; the runtime failure is the second line of defence.

---

## ADR-002 — `isinstance` opens the `Result`; `match` dispatches the error union

- **Date:** 2026-08-29
- **Change:** `error-handling-foundation`
- **Status:** Accepted

**Context.** The deployed `pattern-matching-standard` spec mandated `match`/`case` on
`Success`/`Failure` and forbade `isinstance`. The project's own ast-grep gate, its
instruction docs and 122 unwrap call sites do the opposite. Something had to give,
and the question was which.

**Decision.** Code unwrapping a `Result` uses `isinstance(result, Failure)`. `match`
is reserved for the error union obtained *after* that narrowing, closed with
`assert_never`.

**Rationale / Alternatives.** This is settled by measurement, not taste. On this
repository's `ty`:

| Construct | Verdict |
|---|---|
| `match result: case Success(value)` | **no narrowing** — binds the union of success *and* error types |
| `isinstance(result, Failure)` then `.failure()` | narrows to the error union |
| `match error:` over a closed union + `assert_never` | passes |
| same, one arm removed | `error[type-assertion-failure]` naming the missing type |

Exhaustiveness checking is real and worth having — it just belongs on the error
union, not on the `Result` container. Once that distinction is drawn the apparent
conflict dissolves: the gate and the spec were describing different constructs.
Mandating `match` on the container would have cost narrowing at 122 sites for no
benefit; banning `match` outright would have given up `assert_never`, which is the
mechanism the whole closed-union design leans on.

**Consequences.**
- The deployed spec's requirement is REMOVED rather than MODIFIED: a MODIFIED header
  must stay byte-identical, which would archive a header asserting `match` above a
  body mandating `isinstance`, permanently.
- The `no-match-on-result` gate must reject `case Success(value):` while accepting
  `case SubscriptionNotFoundError():`. A pattern careless about that distinction
  would flag the endorsed construct at every dispatch site in the system.

---

## ADR-003 — `ErrorKind` is a seven-member, fixed-width boundary vocabulary

- **Date:** 2026-08-29
- **Change:** `error-handling-foundation`
- **Status:** Accepted

**Context.** Per-feature unions solve drift inside a feature but leave every boundary
adapter — HTTP renderer, Celery task, MCP handler, auth dependency — needing to act
on an error whose concrete type it cannot know. Scoping agreed five kinds:
`VALIDATION`, `NOT_FOUND`, `CONFLICT`, `INFRASTRUCTURE`, `EXTERNAL_SERVICE`.

**Decision.** Seven: the agreed five plus `AUTHENTICATION` (401) and `AUTHORIZATION`
(403). It is the only vocabulary shared across features, and boundary dispatch is
over `kind`, never over a concrete error type.

**Rationale / Alternatives.** Five members cannot express 401 or 403.
`auth/service.py` raises `UnauthorizedException` at 16 sites, and the locked scope
converts every service to Result-typed — so with five kinds a failed login would have
rendered **422**. The old `ErrorCode` covers this correctly (`UNAUTHORIZED`,
`FORBIDDEN`, `INVALID_TOKEN`, `TOKEN_EXPIRED`, `REFRESH_TOKEN_INVALID`), so the
five-member design was a regression against shipped behaviour, on the feature
scheduled last and therefore discovered latest. Carving `auth` out of the conversion
contradicts the locked scope; letting an error carry an explicit status reintroduces
the per-endpoint drift being removed.

Seven keeps the property that mattered: dispatch width is independent of how many
concrete error types exist. Seventeen features can add error types indefinitely and
every adapter still has the same seven arms.

**Consequences.**
- Boundary adapters are stable under feature growth.
- Adding a kind is a breaking change for every adapter, which is the correct
  friction — it is a change to the system's shared vocabulary.
- Status is derived from `kind`, refined by `retryable` only for `INFRASTRUCTURE`
  (500 when dead, 503 when transient).
- The near-miss is the argument for auditing a plan against the ground rather than
  the reverse. It was found by measurement during review, before any code.

---

## ADR-004 — The router renders; the renderer owns the transport status

- **Date:** 2026-08-29
- **Change:** `error-handling-foundation`
- **Status:** Accepted

**Context.** `http_error()` writes a status *into the response body* and does not set
the HTTP status. Returning it from a route therefore yields **HTTP 200** with
`"success": false` — a body that says 404 delivered with a 200. The alternative in
use is `raise app_error_to_exception(error)`, which an existing gate flags at 34
sites as retired.

**Decision.** One shared `render_result(result, response, message=...,
success_status=...)`. On `Failure` it sets `response.status_code` from
`STATUS_BY_KIND[error.kind]` and returns the `http_error` envelope. Endpoints cannot
override the failure status.

**Rationale / Alternatives.** Raising from the router keeps the exception path the
whole change is retiring and makes the router's return type a lie. Letting each
endpoint pass a failure status is how the current drift happened — 67 call sites
each deciding independently. Deriving it from `kind` means the status is a
consequence of classification, and classification is already a type-checked
`ClassVar`.

The parameter is named `success_status`, not `status_code`: at a call site
`status_code=201` reads as the status of the response being rendered, which is
exactly wrong on the failure path.

**Consequences.**
- The envelope shape does not change; `APIResponse` + `http_error()` survive. Only
  the transport status is added.
- A route wanting a non-standard failure status must change the error's `kind`, which
  is a visible edit to the feature's contract rather than a local override.
- Endpoints no longer import a feature's error types to render them — the renderer
  needs only `kind`.

---

## ADR-005 — A gate is not trusted until it is shown to permit and to forbid

- **Date:** 2026-08-29
- **Change:** `error-handling-foundation`
- **Status:** Accepted

**Context.** Two of this design's central guarantees have no type-checker backstop: a
union is closed only because a rule forbids subclassing the base elsewhere, and a
sibling set is flat only because a rule says so. Both rest entirely on ast-grep.

Then `no-match-on-result` turned out not to work. Its pattern is
`regex: ^(Success|Failure)\(\s*\)$` against a `case_pattern`, matching only the
argument-less form — so `case Success(value):`, the exact construct the superseded
spec mandated and this design forbids, passes unflagged. Its message says match/case
does not narrow `Result`. Its zero-violation count meant "the rule looked for
something nobody writes".

**Decision.** Every rule this work introduces or changes ships with a fixture holding
the construct it forbids *and* the nearest construct it permits, and must be shown to
flag the first and spare the second before any count derived from it is cited. A
rule's message must describe what its pattern actually matches. A count from a
corrected rule is re-measured, never carried forward.

**Amended after the fourth review pass — a gate has two ways to report nothing.** The
above catches a rule whose *pattern* is wrong. It does not catch a working rule that was
*pointed away from the code*. `pyproject.toml`'s `per-file-ignores` disables `BLE001`,
`E722`, `B904`, `TRY201`, `TRY300`, `TRY301`, `TRY400` and `S112` for
`src/app/examples/*.py` — eight of the rules this change is about — so
`ruff check src/app/examples/` reports "All checks passed!" while `ast-grep scan`, the
only gate there with no per-path ignore, reports **4 `error`-level violations** in
`redis_examples.py`. A broken pattern and a configured exclusion produce the identical
artefact: a clean run that reads as coverage. So the decision extends: **a gate's clean
run may not be cited until its exclusion list has been read** — `per-file-ignores`,
`sgconfig.yml`'s `ruleDirs`, and any rule-level path filter.

**Rationale / Alternatives.** Reviewing rules by reading them is what produced the
broken one. An unverified gate is indistinguishable from no gate while reading as
coverage, which is worse than an absent gate because it stops anyone looking. The
same holds for an excluded path, with one aggravating difference: the exclusion is
usually deliberate and old, so nobody re-examines whether its reason still applies.

**Consequences.**
- Seven new rules cost seven fixture pairs. That is the price of the closed-union
  guarantee, not overhead on top of it.
- Historical violation counts from unverified rules are not evidence and are not
  cited as baselines.
- The same discipline extends to measurement, not just gates: before a count becomes
  a claim about a population, the population is enumerated by a second, structurally
  different query and the two reconciled. Four errors in this change's own drafting
  came from skipping that step.
- The eight `per-file-ignores` entries for `src/app/examples/*.py` are removed (task
  9.1) and the findings that surface are fixed, not re-suppressed. Task 10.5 forbids
  adding an entry back to reach a green check, and task 10.7 makes reading the
  exclusion lists an explicit verification step.
- One shape must be **spared** on this reasoning rather than flagged: `BLE001` does not
  fire on a blind `except` that ends in a bare `raise`, because nothing was survived.
  A new degradation gate that flags `middleware/server_middleware.py:100` would
  contradict a tool the project already runs.

---

## ADR-006 — A union is owned by whoever classifies the exception, not by `features/`

- **Date:** 2026-08-29
- **Change:** `error-handling-foundation`
- **Status:** Accepted

**Context.** The first draft of this design keyed the error contract on `features/`
and described its layer classification as total. It was not.
`src/app/shared/services/` is four third-party wrappers — boto3, httpx — with 31
raises of `APIException` subclasses and 20 catches, every one of a library type, and
no `Result` anywhere. Structurally it is indistinguishable from a feature repository:
it owns a boundary, catches a library's taxonomy, and decides what that means. A
directory-keyed rule could not see it. Nor could it see the dispatcher every error
passes through, or the session dependency the rollback requirement exists to protect.

**Decision.** The module that classifies a third-party exception owns the union it
classifies into, on the same terms as a feature: its own StrEnum, flat siblings,
closed union, no cross-module code imports. `features/` is the common case, not the
definition. Layers are classified by the role they play — domain classifier,
dispatcher, session dependency, degradation boundary, vocabulary — and not living
under `features/` is never grounds for exemption.

**Rationale / Alternatives.** Leaving `shared/services/` raising `APIException` would
force any feature that calls it to `try`/`except` around a call the project owns —
the one thing the try/except rule forbids. A single global `SharedError` union
reintroduces the cross-cutting coupling ADR-001 removes: a storage failure and a
Tavily failure share nothing but the word "shared". Moving the modules under
`features/` would make the directory lie about what they are.

**Consequences.**
- Ordering constraint, not a preference: `shared/services/` must be Result-typed
  before `crawler`, its first consuming feature, or that feature violates the
  try/except rule by construction.
- The rule cuts both ways and keeps scope bounded. A module that classifies nothing
  owes nothing — an `except ImportError` that sets an availability flag is capability
  detection, not error handling, and `shared/rag/`'s pipeline stays exception-native
  while only its provider boundary converts.
- `ErrorKind` is unaffected: shared modules classify into the same seven kinds, so
  ADR-003's fixed-width property holds.
- The blind spot was found by the request's author, not by the plan — the second
  directory-shaped assumption in this change to have hidden live error-handling code.
