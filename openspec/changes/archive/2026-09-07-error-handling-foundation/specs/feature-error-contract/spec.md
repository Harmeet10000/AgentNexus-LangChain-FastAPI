## Purpose

Defines the shape every domain error in this system must take: a frozen Pydantic
model whose classification is fixed at the class rather than supplied per call
site, gathered into a per-feature closed union that a type checker can verify a
`match` has covered exhaustively.

## ADDED Requirements

### Requirement: Error classification SHALL be a class constant, not a constructor parameter

`kind`, `code`, and `retryable` SHALL be declared as `typing.ClassVar` on the
error type. They SHALL NOT be Pydantic fields, SHALL NOT appear in
`model_fields`, and SHALL NOT be acceptable as constructor keyword arguments.

The base SHALL set `model_config = ConfigDict(extra="forbid", frozen=True)` so
that passing a classification value at a construction site raises
`ValidationError` rather than silently overriding the class constant.

`code` SHALL be typed as the feature's own StrEnum, never as a bare `str`, so a
value outside that enum is a type error before it is a runtime value.

#### Scenario: Classification is absent from the model fields
- **WHEN** a concrete error type declares `code`, `kind`, and `retryable` as `ClassVar`
- **THEN** `model_fields` contains only the instance data fields, and none of `code`, `kind`, or `retryable`

#### Scenario: Supplying a code at the construction site is rejected
- **WHEN** a caller constructs an error type passing `code="ANYTHING"` as a keyword argument
- **THEN** construction raises `ValidationError` because `extra="forbid"` refuses the unknown key, so a mistyped code cannot be represented

#### Scenario: A code cannot be written as a string at all
- **WHEN** a concrete error type assigns `code: ClassVar[SubscriptionCode] = "DUPLICATE_SUBSCRIPTION"` as a string literal, even though that string is the correct value of an existing enum member
- **THEN** `uv run ty check src/` reports `invalid-assignment`, because a `str` literal is not assignable to the enum type — so a code cannot be spelled by hand whether it is right or wrong, and the enum member must be referenced

#### Scenario: A code from another feature's enum is a type error
- **WHEN** a concrete error type assigns a member of a different feature's code enum
- **THEN** `uv run ty check src/` reports `invalid-assignment`, so codes cannot leak between features even by accident

#### Scenario: An error instance is immutable
- **WHEN** code assigns to any field of a constructed error instance
- **THEN** the assignment raises `ValidationError` because the model is frozen

#### Scenario: Classification does not leak into the serialised payload
- **WHEN** an error instance is serialised with `model_dump()`
- **THEN** the output contains only the instance data fields, and the `code`, `kind`, and `retryable` constants are read from the class when the boundary needs them

### Requirement: Each feature SHALL own a closed error union in its own errors.py

Every feature SHALL declare `src/app/features/<name>/errors.py` containing a
`<Feature>Code` StrEnum, its concrete error types, a closed union
`type <Feature>Error = A | B | C` naming every one of them, and
`type <Feature>Result[T] = Result[T, <Feature>Error]`.

Repository and service methods SHALL be annotated with `<Feature>Result[T]`.
They SHALL NOT be annotated with `AppResult[T]`, whose error side is pinned to
the open `AppError` base and therefore cannot be exhaustively matched.

No feature SHALL import another feature's error types, error union, or code enum.
Where two features genuinely exchange a failure, the calling feature SHALL
translate it into a member of its own union at the call boundary.

#### Scenario: A repository method carries the feature's own Result type
- **WHEN** a subscriptions repository method is annotated
- **THEN** its return type is `SubscriptionResult[Subscription]`, not `AppResult[Subscription]`

#### Scenario: Every concrete error type appears in the union
- **WHEN** a feature's `errors.py` declares a concrete error type
- **THEN** that type is named in the feature's `type <Feature>Error` union, and a type declared but omitted from the union is a rule violation

#### Scenario: Cross-feature failure is translated, not re-exported
- **WHEN** a service in one feature receives a `Failure` from a collaborator in another feature
- **THEN** it constructs an error from its own union to carry that failure onward, and does not import or propagate the collaborator's error type

### Requirement: Concrete error types SHALL be flat siblings with no inheritance between them

Every concrete error type SHALL inherit directly from `FeatureError` and from
nothing else. No concrete error type SHALL inherit from another concrete error
type, and no intermediate error base SHALL be introduced between `FeatureError`
and a concrete type — including for the sole purpose of sharing fields.

The reason is not style. `match` class patterns are `isinstance`-based, so a
broader arm placed before a narrower one silently makes the narrower arm
unreachable, and a type checker still reports that `match` exhaustive because
every case was covered. There is no static or runtime signal for a shadowed arm;
the only symptom is the wrong branch's side effects running.

Where several error types in a feature need the same field, the field SHALL be
repeated on each type, or added to `FeatureError` if it is genuinely universal.

#### Scenario: Introducing a shared intermediate base is rejected
- **WHEN** a feature adds `class ConflictError(FeatureError)` and has `VersionConflictError` inherit from it to share fields
- **THEN** the enforcement rule reports a violation, because a `case ConflictError():` arm would silently shadow `case VersionConflictError():`

#### Scenario: A shared field is duplicated rather than inherited
- **WHEN** three error types in one feature each need a `subscription_id` field
- **THEN** each declares the field on itself, and no intermediate base is created to hold it

#### Scenario: The existing deep chains are flattened as their feature migrates
- **WHEN** a feature migrates that owns one of the 28 existing concrete-inherits-concrete chains
- **THEN** that feature's chains are flattened to direct children of `FeatureError` in the same change, and none survives the migration

### Requirement: A match over a feature's error union SHALL be provably exhaustive

Code that dispatches on a feature's error union for business logic SHALL `match`
on the union and SHALL close the match with `case _ as unreachable:
assert_never(unreachable)`.

Adding a member to a feature's union without adding its arm to an existing
exhaustive match SHALL fail `uv run ty check src/`. This is the property the
whole design exists to buy, and it SHALL be verified rather than assumed.

#### Scenario: An exhaustive match type-checks
- **WHEN** a match covers every member of the feature's error union and closes with `assert_never`
- **THEN** `uv run ty check src/` reports no diagnostic for that match

#### Scenario: A missing arm is caught by the type checker
- **WHEN** a member of the feature's error union has no arm in a match closed by `assert_never`
- **THEN** `uv run ty check src/` reports `type-assertion-failure`, naming the uncovered member as the inferred type of the `assert_never` argument

#### Scenario: Adding a new error type breaks every stale match
- **WHEN** a new error type is added to a feature's union
- **THEN** every existing exhaustive match over that union fails type-checking until an arm is added, so no dispatch site can silently ignore the new failure mode

### Requirement: ErrorKind SHALL be the only shared classification vocabulary

`app/shared/result/` SHALL define `ErrorKind` as a StrEnum with exactly the
members `VALIDATION`, `NOT_FOUND`, `CONFLICT`, `AUTHENTICATION`, `AUTHORIZATION`,
`INFRASTRUCTURE`, and `EXTERNAL_SERVICE`. It SHALL be the only error vocabulary
shared across features.

`ErrorKind` SHALL live in the existing `app/shared/result/` package, not in a new
one. That package already owns the shared error vocabulary — `errors.py` holds
`AppError` and its five subclasses, `types.py` the `AppResult` alias, `mappers.py`
the exception bridge, `logging.py` the failure logger — and `result-layer-boundaries`
already classifies `shared/result/errors.py` as the vocabulary layer. A second
vocabulary package would give the codebase two competing homes for the same concept
during the exact window in which the old one is being retired.

The five existing subclasses SHALL be read as the predecessor of this enum, not as
an unrelated hierarchy. `ValidationAppError`, `NotFoundAppError`, `ConflictAppError`,
`InfrastructureAppError` and `ExternalServiceAppError` each declare a
`kind: Literal[...]` field, and their five values are exactly the five kinds this
requirement extends to seven. The migration therefore moves `kind` from a
per-subclass `Literal` field to a `ClassVar` on flat siblings, and adds the two
members the old hierarchy could not express — it does not introduce the concept.

`AppError` itself SHALL NOT be treated as carrying a kind. The base declares no
`kind` field, so `error.kind` on a bare `AppError` instance raises
`AttributeError`; `app/features/ingestion/service.py:86` constructs exactly that
(`AppError(code="UNKNOWN", message=str(failure))`). Any kind-keyed adapter SHALL
therefore be unreachable from a bare-base instance by the time it ships, either
because the instance is gone or because the base is no longer constructible.

`AUTHENTICATION` and `AUTHORIZATION` are separate members because the boundary
must distinguish "we do not know who you are" from "we know, and you may not do
this" — they are different HTTP statuses, different log severities, and different
client behaviours. No other kind can express either, and a system that cannot
express them answers a failed login with a validation error.

Boundary adapters — HTTP status selection, log severity, retry eligibility —
SHALL match on `kind`, never on a feature's concrete types, so that boundary code
stays a fixed-width dispatch no matter how many feature error types exist.
Feature business logic SHALL match on the concrete type, never on `kind`.

#### Scenario: The boundary dispatches on kind
- **WHEN** the HTTP boundary selects a status code for a failure
- **THEN** it reads `error.kind` and consults `STATUS_BY_KIND`, and does not reference any feature's concrete error type

#### Scenario: Feature logic dispatches on the concrete type
- **WHEN** a subscriptions service decides whether a failure is retryable business state or a hard stop
- **THEN** it matches on the concrete error types in `SubscriptionError`, not on `kind`, because two error types sharing a `kind` can need opposite handling

#### Scenario: A failed authentication is not a validation failure
- **WHEN** an authentication service returns a failure for invalid credentials, an invalid token, or an expired token
- **THEN** the error's kind is `AUTHENTICATION`, and the boundary renders 401 rather than 422

#### Scenario: A permission denial is distinguished from an identity failure
- **WHEN** a service returns a failure because an authenticated caller may not act on a resource
- **THEN** the error's kind is `AUTHORIZATION` and the boundary renders 403, distinct from the 401 an `AUTHENTICATION` failure produces

#### Scenario: Adding feature error types does not grow the boundary
- **WHEN** a feature adds three new error types to its union
- **THEN** `STATUS_BY_KIND` and every other kind-keyed adapter are unchanged, because each new type declares one of the seven existing kinds

#### Scenario: The shared vocabulary has exactly one home
- **WHEN** a reviewer looks for the definition of `ErrorKind` or `STATUS_BY_KIND`
- **THEN** it is under `app/shared/result/`, alongside the `AppError` hierarchy it replaces, and no second shared error-vocabulary package exists

#### Scenario: A kind-keyed adapter never receives a kindless error
- **WHEN** the HTTP renderer or any other kind-keyed adapter is reached by a failure produced before the migration completes
- **THEN** that failure carries a `kind`, and no code path can hand the adapter a bare `AppError` whose `kind` attribute does not exist

### Requirement: The AppError hierarchy SHALL be frozen for the duration of the migration

From this change until the last feature has migrated, no new subclass of
`AppError` SHALL be added, and no new field SHALL be added to an existing one.
New error types SHALL be declared as `FeatureError` subclasses in a feature's
`errors.py`.

The migration spans many changes. Over that window the open hierarchy being
retired can otherwise legitimately grow, and every addition enlarges the work
remaining for the features that have not yet migrated. Freezing it makes the
hierarchy strictly shrink.

An `AppError` subclass SHALL be deleted in the change that migrates its owning
feature, not deprecated in place.

#### Scenario: A new AppError subclass is rejected
- **WHEN** a change adds a class inheriting from `AppError` or one of its subclasses
- **THEN** the enforcement gate reports a violation and directs the author to the owning feature's `errors.py`

#### Scenario: An unmigrated feature may still use the old types
- **WHEN** a feature that has not yet migrated constructs one of the existing `AppError` subclasses
- **THEN** that is permitted, because the alternative is a partial migration carrying two error systems inside one feature

#### Scenario: The hierarchy shrinks monotonically
- **WHEN** any feature's migration change is complete
- **THEN** the total number of `AppError` subclasses in the codebase is strictly lower than before that change, and reaches zero when the last feature migrates
