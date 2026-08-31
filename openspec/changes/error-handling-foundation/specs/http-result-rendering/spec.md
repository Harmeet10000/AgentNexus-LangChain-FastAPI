## Purpose

Defines how an HTTP endpoint turns a typed failure into a response: one shared
renderer that derives the real status code from the error's kind and emits the
project's standard error envelope, so a service never has to raise in order for a
client to receive the right status.

## ADDED Requirements

### Requirement: A router SHALL render a Result rather than raise on an expected failure

An endpoint that calls a Result-typed service SHALL open the `Result` and hand a
`Failure` to the shared renderer. It SHALL NOT convert an expected failure into an
exception for the global exception handler to catch.

Raising remains correct for the cases where control flow must be abandoned rather
than answered: a FastAPI dependency rejecting a request before the endpoint body
runs, and a genuinely unexpected exception reaching the global handler.

#### Scenario: An expected failure is rendered
- **WHEN** an endpoint receives `Failure(<Feature>NotFoundError(...))` from its service
- **THEN** it returns the rendered error response directly, and no exception is raised

#### Scenario: A dependency still raises
- **WHEN** an authentication dependency determines the caller is not authorised
- **THEN** it raises, because a dependency has no response to return and raising is the only way to stop the request

#### Scenario: An unexpected exception still reaches the handler
- **WHEN** an unanticipated exception escapes an endpoint
- **THEN** the global exception handler renders it, and this requirement does not apply

### Requirement: The rendered response SHALL carry the real HTTP status, not only a status in the body

The renderer SHALL set the response's HTTP status code. A response describing a
failure SHALL NOT be transmitted with a 2xx status.

This corrects a live defect: the existing `http_error` helper writes the status
into the JSON body and returns a plain object, so an endpoint that returns it
produces **HTTP 200** with `"success": false`. Clients, proxies, retry
middleware, and monitoring all key on the transport status, and every one of them
currently reads these failures as successes.

#### Scenario: A not-found failure returns 404
- **WHEN** an endpoint renders a failure whose kind is `NOT_FOUND`
- **THEN** the HTTP response status is 404, and the body's status field agrees with it

#### Scenario: The transport status and body status agree
- **WHEN** any failure is rendered
- **THEN** the status in the response body equals the HTTP status line, so no client can reach a different conclusion depending on which it reads

#### Scenario: A success is unaffected
- **WHEN** an endpoint renders a `Success`
- **THEN** the response carries the endpoint's declared success status and the standard success envelope

### Requirement: The status SHALL be derived from ErrorKind through a single shared mapping

The renderer SHALL select the status from a single shared mapping keyed by
`ErrorKind`. An endpoint SHALL NOT hand-pick a status per failure, and the mapping
SHALL NOT be keyed on any feature's concrete error type.

The mapping SHALL be: `VALIDATION` → 422, `NOT_FOUND` → 404, `CONFLICT` → 409,
`AUTHENTICATION` → 401, `AUTHORIZATION` → 403, `EXTERNAL_SERVICE` → 502,
`INFRASTRUCTURE` → 500 when the error is not retryable and 503 when it is.

#### Scenario: Each kind maps to its status
- **WHEN** failures of each of the seven kinds are rendered
- **THEN** they return 422, 404, 409, 401, 403, 502, and 500 or 503 respectively, from the shared mapping

#### Scenario: An authentication failure is not answered as a validation failure
- **WHEN** an endpoint renders a failure whose kind is `AUTHENTICATION` — invalid credentials, invalid token, expired token
- **THEN** the HTTP status is 401, matching what the equivalent raised exception produces today, so converting the service to Result does not change the status a client sees

#### Scenario: An authorization failure is distinguished from an authentication failure
- **WHEN** an endpoint renders a failure whose kind is `AUTHORIZATION`
- **THEN** the HTTP status is 403, and a client can distinguish it from the 401 an `AUTHENTICATION` failure produces

#### Scenario: A retryable infrastructure failure signals retry
- **WHEN** an infrastructure error declares itself retryable
- **THEN** the response status is 503, and a non-retryable infrastructure error returns 500

#### Scenario: A dead transaction is not advertised as retryable
- **WHEN** a repository classifies a failed database write that has been rolled back
- **THEN** the error it constructs is not retryable, so the client receives 500 and is not told to retry work that cannot succeed

#### Scenario: Adding a feature error type does not touch the mapping
- **WHEN** a feature adds new error types to its union
- **THEN** the shared mapping is unchanged, because each new type declares one of the seven existing kinds

### Requirement: Every endpoint SHALL emit the same error envelope shape

A rendered failure SHALL use the project's standard error envelope, with the same
fields in the same shape as the global exception handler produces. A client SHALL
NOT be able to tell from the envelope whether a failure was rendered by an
endpoint or by the handler.

The error `code` in the envelope SHALL be the feature error type's class constant.
It SHALL NOT be a string composed at the endpoint.

#### Scenario: Rendered and handled failures look alike
- **WHEN** the same logical failure is produced once by a rendered `Failure` and once by an escaping exception
- **THEN** the two response bodies have the same envelope structure and field names

#### Scenario: The code comes from the error type
- **WHEN** a failure is rendered
- **THEN** the envelope's error code is the value declared as a class constant on the error type, so it cannot differ between two endpoints rendering the same error

#### Scenario: No endpoint composes its own error body
- **WHEN** an endpoint needs to return a failure
- **THEN** it calls the shared renderer, and does not construct an error dictionary or response object itself

### Requirement: Rendering SHALL NOT require the endpoint to know the feature's error types

The renderer SHALL accept any feature's `Result` and require no
per-feature configuration, so adding a feature adds no renderer code.

An endpoint SHALL be able to override the success status and supply a
human-readable success message, but SHALL NOT be able to override the failure
status, because that would reintroduce the per-endpoint status drift this
requirement exists to remove.

The parameter carrying the success status SHALL be named for what it is — a
success status — and SHALL NOT be named `status_code`, which reads at the call
site as though it governs the failure.

#### Scenario: One renderer serves every feature
- **WHEN** endpoints from different features render their own Results
- **THEN** they call the same renderer with no feature-specific arguments

#### Scenario: A success status is configurable
- **WHEN** a creation endpoint renders a `Success`
- **THEN** it can declare 201 as the success status while the failure statuses remain governed by the shared mapping

#### Scenario: The success-status parameter is unambiguously named
- **WHEN** a developer reads a `render_result(...)` call at an endpoint
- **THEN** the parameter's name makes clear it applies only to the success path, so it is not misread as setting the failure status

#### Scenario: A failure status is not overridable at the call site
- **WHEN** an endpoint attempts to force a specific status for a failure
- **THEN** the renderer offers no parameter to do so, and the status remains the one derived from the error's kind
