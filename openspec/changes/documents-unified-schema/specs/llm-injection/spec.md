## MODIFIED Requirements

### Requirement: Dependency Layer Updates

The dependency layer that assembles the document retrieval and question-answering services SHALL construct the
chat model once per resolution and inject it into the service it builds. A service SHALL NOT construct a chat
model for itself. With the superseded search service dissolved, the unified document path is the only dependency
layer this requirement governs.

#### Scenario: Dependency layer creates LLM once and injects

- **WHEN** the document retrieval or question-answering service dependency is resolved
- **THEN** the chat model SHALL be constructed exactly once for that resolution
- **AND** the same model instance SHALL be injected into the constructed service
- **AND** the service SHALL NOT construct a chat model of its own

#### Scenario: No dependency layer survives for the dissolved search service

- **WHEN** the dependency layer is enumerated after consolidation
- **THEN** it SHALL contain no provider for the superseded search service
- **AND** no chat model SHALL be constructed for a service that no longer exists

## REMOVED Requirements

### Requirement: SearchService Constructor Injection

**Reason**: The superseded search service is dissolved by this change, leaving the requirement no subject — its ingest path is deleted rather than retargeted, its two retrieval methods duplicate the unified document service's mounted equivalents, its graph-backed legal-ask path moves to the document query service, and the dependency module named by the requirement is deleted with it.

**Migration**: No consumer action is required, because the removed requirement's surface was never reachable — the router exposing it was never mounted, so no request or response contract changes. The same contract for the surviving path (the chat model is injected, never constructed inside the service, and the provider import is absent from the service module) is already carried by this capability's `Document Service Injection` requirement and by the modified `Dependency Layer Updates` above.
