## Purpose

Guarantee that the authenticated caller's identity is derived from validated access-token claims, so that
endpoints needing an identity answer with a correct authentication failure instead of an internal error, and so
that no endpoint depends on request state nothing ever assigns.

## ADDED Requirements

### Requirement: The caller's identity SHALL be derived from validated access-token claims

An endpoint that needs the caller's identity SHALL obtain it from the claims of a validated access token. It
SHALL NOT read the identity from per-request state, because no component in the application assigns that state
and no authentication middleware exists to do so.

#### Scenario: A request carrying a valid access token

- **WHEN** a request carrying a valid access token reaches an endpoint that needs the caller's identity
- **THEN** the identity used SHALL be the subject of the validated token claims

#### Scenario: A request carrying no credentials — BREAKING

- **WHEN** a request with no access token reaches an endpoint that needs the caller's identity
- **THEN** the response SHALL be `401 Unauthorized`
- **AND** it SHALL NOT be a `500` produced by reading unset request state

#### Scenario: A request carrying a token of the wrong type

- **WHEN** a request carrying a token that is valid but is not an access token reaches such an endpoint
- **THEN** the response SHALL be `401 Unauthorized`

#### Scenario: Identity resolution costs no database round trip

- **WHEN** the caller's identity is resolved
- **THEN** it SHALL be resolved from the token claims alone, with no database query

#### Scenario: The public contract does not change

- **WHEN** the generated API description is compared against the one produced before this change
- **THEN** the set of paths SHALL be unchanged
- **AND** identity resolution SHALL NOT appear as a query, header, cookie or body parameter on any operation

### Requirement: No endpoint SHALL depend on identity state that nothing assigns

Across every feature, no code path SHALL read a caller-identity attribute from per-request state. A branch that
existed only to tolerate that attribute's absence SHALL be removed rather than kept alive by introducing a writer
for it.

#### Scenario: Repository-wide inspection

- **WHEN** identity resolution is inspected across every feature, mounted or not
- **THEN** no code path SHALL read a caller-identity attribute from per-request state

#### Scenario: A previously guarded read

- **WHEN** a handler previously tested for the presence of that state attribute before reading it
- **THEN** that branch SHALL be removed, since nothing assigns the attribute
- **AND** no writer for it SHALL be introduced in order to keep the branch reachable

### Requirement: Repairing identity SHALL leave the repaired endpoints working end to end

Repairing identity resolution SHALL NOT simply relocate a failure one layer deeper. An authenticated request to
an endpoint that both needs an identity and records a domain event SHALL complete, which requires the relation
that receives the event to exist by the time the identity repair is in effect.

#### Scenario: An authenticated request to an endpoint that records a domain event

- **WHEN** an authenticated request reaches a mounted endpoint that needs the caller's identity and records a
  domain event
- **THEN** the request SHALL complete and the event SHALL be recorded
- **AND** it SHALL NOT fail with an undefined-relation error once identity resolution has been repaired
