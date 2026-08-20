## Purpose

Guarantee that request handlers resolve shared infrastructure clients under the names the startup sequence
publishes, and that every database consumer obtains a connection URL it can actually use, in the flavour its own
driver requires, from one place rather than by repairing a raw configuration value at the call site.

## ADDED Requirements

### Requirement: Handlers SHALL resolve shared clients under the names the startup sequence publishes

A handler that needs a client created during startup SHALL read it under the same name the startup sequence
assigns. A read under a name startup never assigns is an unconditional failure on the first request to that
endpoint, regardless of whether the underlying service is healthy.

#### Scenario: A mounted endpoint resolving a shared client

- **WHEN** a mounted endpoint resolves a client that startup creates
- **THEN** the name it reads SHALL be a name the startup sequence assigns

#### Scenario: Repository-wide inspection of shared-client reads

- **WHEN** every read of startup-published state is compared against every name startup assigns
- **THEN** no read SHALL name state that nothing assigns, except where a later change owns that wiring and the
  gap is recorded

### Requirement: An optional client that failed to initialise SHALL produce a service-unavailable response

Where startup records an optional client as absent because its initialisation failed, an endpoint that needs it
SHALL answer `503 Service Unavailable`. It SHALL NOT raise an attribute error, and it SHALL NOT pass the absent
client onward to fail later and further from the cause.

#### Scenario: The endpoint's optional client is absent

- **WHEN** a request reaches an endpoint whose optional client startup recorded as absent
- **THEN** the response SHALL be `503 Service Unavailable`
- **AND** the failure SHALL be reported at the point of resolution, not deferred into the operation

#### Scenario: The endpoint's optional client is present

- **WHEN** the client initialised successfully
- **THEN** the request SHALL proceed normally

### Requirement: Every database consumer SHALL obtain its connection URL from the shared accessor

No consumer SHALL read the raw configured database URL, which carries no credentials and names a scheme the
application's drivers cannot all use. The only permitted direct readers are the accessor itself and diagnostics
that display nothing but the host and database name.

#### Scenario: A consumer requests a database URL

- **WHEN** any component needs a database URL
- **THEN** it SHALL obtain it from the shared accessor
- **AND** the URL it receives SHALL carry the credentials required to authenticate

#### Scenario: Repository-wide inspection of raw reads

- **WHEN** every read of the raw configured database URL is enumerated
- **THEN** the only remaining readers SHALL be the accessor itself and diagnostics that display only the host and
  the database name

### Requirement: The accessor SHALL serve every driver flavour its consumers need

Consumers connect through different drivers, and a single URL string cannot satisfy all of them. The accessor
SHALL expose each required flavour explicitly. A consumer SHALL NOT derive one flavour from another by editing
the string at the call site, and no flavour SHALL be produced by editing a string that has already been edited.

#### Scenario: A consumer connecting through the asynchronous ORM layer

- **WHEN** the component that owns the application's connection pool requests a URL
- **THEN** it SHALL receive one whose scheme selects the asynchronous driver the pool uses
- **AND** connection parameters that driver rejects as query arguments SHALL be absent

#### Scenario: A consumer connecting directly with a low-level driver

- **WHEN** a component that connects directly, without the ORM layer, requests a URL
- **THEN** it SHALL receive a plain connection URL its driver accepts, carrying credentials and retaining the
  transport-security parameter that driver requires

#### Scenario: The set of flavours is closed at two, and a third is not invented

- **WHEN** the flavours the accessor exposes are enumerated
- **THEN** there SHALL be exactly two URL flavours — the asynchronous ORM form and the plain low-level-driver form
- **AND** no flavour SHALL be added for a consumer that does not accept a connection URL at all

#### Scenario: No flavour is derived at the call site

- **WHEN** database URL construction is inspected across the application
- **THEN** no call site SHALL produce a flavour by removing or rewriting part of a URL the accessor returned

### Requirement: Consumers that assemble their own connection SHALL be served discrete fields, not a URL

Some embedded components accept only discrete connection fields — host, port, user, credential, database name — and
expose no connection-string setting at all. For those consumers the accessor SHALL expose the same underlying values
as discrete fields drawn from the same configuration the URL flavours are built from. A URL flavour SHALL NOT be
invented for them, because there is nothing on the consumer that could receive it.

#### Scenario: A component configured from discrete fields

- **WHEN** an embedded component that accepts only discrete connection fields is configured
- **THEN** it SHALL receive host, port, user, credential and database name from the shared accessor
- **AND** those values SHALL be the same ones the accessor's URL flavours are built from, so the component cannot be
  pointed at a different instance than the application's own pool
- **AND** it SHALL NOT be handed a connection URL, and no URL flavour SHALL exist solely for it

#### Scenario: Discrete fields and URL flavours cannot disagree

- **WHEN** the discrete fields and any URL flavour are both requested
- **THEN** the host, port, user, credential and database name they carry SHALL be identical
- **AND** neither SHALL be assembled from configuration the other does not read

### Requirement: A credential SHALL survive being placed into a URL

The accessor SHALL encode the credential it inserts so that characters which are reserved in a URL do not change
the URL's meaning, and the credential parsed back out SHALL equal the credential that was configured.

#### Scenario: A credential containing reserved characters

- **WHEN** the configured credential contains characters that are reserved in a URL
- **THEN** the URL the accessor returns SHALL parse back to exactly that credential

#### Scenario: A missing credential

- **WHEN** no credential is configured at all
- **THEN** the failure SHALL be reported when the URL is requested
- **AND** it SHALL NOT be silently skipped by comparing the configured value against a placeholder literal

### Requirement: Connection pools SHALL be owned by the startup sequence

Components SHALL use the pool the startup sequence created. Building an additional pool per operation and
disposing of it afterwards SHALL be treated as a defect, since it duplicates the pool the application already
owns and hides its cost from the pool's configuration.

#### Scenario: An operation that needs a database connection

- **WHEN** a component outside the startup sequence needs a database connection
- **THEN** it SHALL take one from the pool startup created
- **AND** where it does not yet do so, that site SHALL be recorded as an outstanding defect with a named owner
