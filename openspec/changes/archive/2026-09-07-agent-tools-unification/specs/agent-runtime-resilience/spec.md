## Purpose

Defines how the agent surface behaves when its dependencies are absent or flaky: a deliberately unavailable graph
answers with a service-unavailable response instead of an internal error, retries are bounded and owned in one place,
and a human-in-the-loop pause is never mistaken for a transient fault.

## ADDED Requirements

### Requirement: An unavailable agent dependency yields a service-unavailable response

Every request-scoped dependency of the agent surface SHALL fail closed. When the agent graph, the persistence layer, or
another required client has not been provisioned, the request SHALL be answered with a service-unavailable response
carrying an explanatory message. It SHALL NOT surface as an internal server error, and it SHALL NOT raise an attribute
error from unprovisioned application state.

#### Scenario: The agent graph has not been provisioned

- **WHEN** a request reaches an agent route and the agent graph has not been provisioned
- **THEN** the response SHALL be service-unavailable with an explanatory message

#### Scenario: The persistence layer attribute was never assigned

- **WHEN** a request reaches an agent route and the persistence layer was never assigned to application state
- **THEN** the response SHALL be service-unavailable
- **AND** no attribute error SHALL escape the dependency

#### Scenario: The persistence layer was provisioned as absent

- **WHEN** the persistence layer is present in application state but holds no value
- **THEN** the response SHALL be service-unavailable

#### Scenario: A provisioned dependency is returned unchanged

- **WHEN** every required dependency has been provisioned
- **THEN** the dependency SHALL be returned and the request SHALL proceed

### Requirement: The unwired agent graph remains unwired

The agent graph SHALL NOT be constructed during application startup as part of this change, and no configuration
default SHALL cause it to be constructed. Startup SHALL complete without it, and the routes that depend on it SHALL
answer as unavailable. Nothing SHALL be introduced that prevents the graph from later being wired without redesign.

#### Scenario: Startup performs no agent graph construction

- **WHEN** the application starts
- **THEN** no agent graph construction SHALL be attempted
- **AND** startup SHALL complete successfully

#### Scenario: Routes depending on the graph answer as unavailable

- **WHEN** a request reaches a route depending on the agent graph while it remains unwired
- **THEN** the response SHALL be service-unavailable rather than an internal server error

#### Scenario: The construction path remains importable and type-correct

- **WHEN** the modules that would construct the agent graph are imported and type-checked
- **THEN** the import SHALL succeed and the construction entry point SHALL accept the arguments its callers would pass

### Requirement: Model and tool retries are bounded and owned in one place

Retries around model invocation and tool invocation SHALL be applied at a single designated seam with a bounded attempt
count and backoff. Retry behaviour SHALL NOT be re-implemented inside individual graph nodes or tool bodies.

#### Scenario: A transient model failure is retried and then surfaced

- **WHEN** a model invocation fails transiently on every attempt
- **THEN** it SHALL be retried up to the configured bound
- **AND** the failure SHALL then be surfaced to the caller

#### Scenario: A permanent failure is not retried

- **WHEN** a model invocation fails with an error that cannot succeed on retry
- **THEN** it SHALL NOT be retried

#### Scenario: Nodes do not carry their own retry loops

- **WHEN** a graph node invokes a model or a tool
- **THEN** the retry behaviour SHALL come from the designated seam and not from a retry loop inside the node

### Requirement: A pause request propagates through the retry seam

A human-in-the-loop pause raised during a wrapped model or tool invocation SHALL propagate unchanged. It SHALL NOT be
retried, suppressed, converted into an error result, or counted as a failed attempt.

#### Scenario: A pause raised inside a wrapped invocation is not retried

- **WHEN** a wrapped invocation raises a human-in-the-loop pause
- **THEN** the pause SHALL propagate to the graph runtime
- **AND** no retry attempt SHALL be made

#### Scenario: The graph actually pauses

- **WHEN** a node under the retry seam requests human input
- **THEN** the run SHALL pause awaiting that input rather than continuing or failing

### Requirement: Tool execution errors are returned to the model, not crashed

When a tool invocation raises, the error SHALL be returned to the model as a tool result it can act on, within the
bounds of the retry policy, rather than terminating the run. The returned error SHALL distinguish unavailability from a
failed request in the same terms as the tool result contract.

#### Scenario: A raising tool does not terminate the run

- **WHEN** a tool invocation raises during an agent step
- **THEN** the error SHALL be delivered to the model as a tool result
- **AND** the run SHALL continue or pause rather than aborting

#### Scenario: Backend unavailability reaches the model as unavailability

- **WHEN** a tool fails because its backend is unreachable
- **THEN** the tool result delivered to the model SHALL indicate unavailability
