# outbox-helper-extraction Specification Delta

## MODIFIED Requirements

### Requirement: `AuthService` has a private outbox-publish helper

The authentication service SHALL enqueue its outbox events through a single internal helper, so that no
event-publishing call site manages a connection resource of its own. Every publish site SHALL route through that one
helper, and the helper SHALL record the event in the same transaction as the session it uses.

Where the service has been given the application's shared session factory, the helper SHALL draw its session from
that factory and SHALL NOT create, configure or dispose of a connection resource of its own.

Where the service has **not** been given a session factory, the helper today falls back to constructing a private
engine from the connection URL and disposing of it in a `finally` block — one engine built and torn down per
published event. That fallback exists, it is reachable, and **this change does not remove it.** It is carried as an
outstanding defect against the `infrastructure-client-access` requirement *Connection pools SHALL be owned by the
startup sequence*, whose named owner is the change that does the connection-plumbing pass. A conforming
implementation of *this* change therefore still contains the fallback; the requirement is discharged when that later
change deletes it, and until then the site SHALL remain recorded rather than silently tolerated.

This requirement is restated for two reasons. First, to attach the pooled-session obligation and the named owner of
the remaining branch. Second, to **withdraw a false claim recorded earlier in this namespace**: an earlier reading
concluded that the engine-per-call implementation "no longer exists" because the service had been refactored onto the
shared session factory. That reading stopped one line short of the fallback branch. The engine-per-call
implementation does exist, on the branch taken when no session factory is supplied, and one mounted construction
site supplies none. Anyone treating this capability as documentation of the running system should read the two
branches as both live.

The scenario titles below are reproduced verbatim from the accepted spec because a MODIFIED block replaces the whole
requirement and must not drop scenarios. The first title consequently names an engine lifecycle that the system has
on only one of its two branches, and its body states which branch does what.

#### Scenario: `_publish_outbox_event` creates an engine, calls `with_outbox`, and disposes

- **WHEN** the helper is asked to enqueue an outbox event for an aggregate
- **THEN** it SHALL record the event with its aggregate type, aggregate identifier, event type and payload in the
  same transaction as the session it uses
- **AND** where the service holds the application's shared session factory, it SHALL acquire the session from that
  factory and SHALL NOT create, configure or dispose of a connection resource of its own
- **AND** where the service holds no session factory, the surviving fallback that builds and disposes a private
  engine per call SHALL be recorded as an outstanding defect with a named owner, and SHALL be removed by that owner's
  change rather than by this one
- **AND** it SHALL release the session or the connection resource when the enqueue completes, on both the success and
  the failure path

#### Scenario: `resend_verification` uses the helper

- **WHEN** the verification-resend path needs to publish an outbox event
- **THEN** it SHALL enqueue through the shared helper with the verification-email event type
- **AND** it SHALL NOT acquire or manage a connection resource directly

#### Scenario: `forgot_password` uses the helper

- **WHEN** the password-reset-request path needs to publish an outbox event
- **THEN** it SHALL enqueue through the shared helper with the password-reset-email event type
- **AND** it SHALL NOT acquire or manage a connection resource directly
