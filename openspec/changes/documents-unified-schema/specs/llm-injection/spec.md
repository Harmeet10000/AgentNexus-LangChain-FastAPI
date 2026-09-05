## MODIFIED Requirements

### Requirement: Dependency Layer Updates

The dependency layer that assembles the document retrieval and question-answering services SHALL inject a chat
model factory into the service it builds, and the service SHALL construct the chat model from that factory
lazily, at most once, on first use — never eagerly at dependency-resolution time. Eager construction lets an
environment failure (missing provider package, bad key) answer an unauthenticated request with a 500 that masks
the 401 the request had already earned, because FastAPI resolves a path operation's dependencies as a set and
answers with whichever raises first. With the superseded search service dissolved, the unified document path is
the only dependency layer this requirement governs.

#### Scenario: Dependency layer injects an LLM factory; the service constructs lazily at most once

- **WHEN** the document retrieval or question-answering service dependency is resolved
- **THEN** a chat model factory SHALL be injected into the constructed service without constructing a chat model
- **AND** the service SHALL construct the chat model from that factory at most once, on first use
- **AND** the service SHALL NOT construct a chat model eagerly at dependency-resolution time

#### Scenario: Dependency layer creates LLM once and injects

- **WHEN** the document retrieval or question-answering service dependency is resolved
- **THEN** the chat model SHALL be created at most once per service instance, from the injected factory, on first use
- **AND** the same constructed instance SHALL serve every subsequent use within that service instance

#### Scenario: No dependency layer survives for the dissolved search service

- **WHEN** the dependency layer is enumerated after consolidation
- **THEN** it SHALL contain no provider for the superseded search service
- **AND** no chat model SHALL be constructed for a service that no longer exists

## REMOVED Requirements

### Requirement: SearchService Constructor Injection

**Reason**: The superseded search service is dissolved by this change, leaving the requirement no subject — its ingest path is deleted rather than retargeted, its two retrieval methods duplicate the unified document service's mounted equivalents, its graph-backed legal-ask path moves to the document query service, and the dependency module named by the requirement is deleted with it.

**Migration**: No consumer action is required, because the removed requirement's surface was never reachable — the router exposing it was never mounted, so no request or response contract changes. The same contract for the surviving path (a chat model factory is injected, the service constructs from it lazily at most once and never eagerly at resolution time, and the provider import is absent from the service module) is already carried by this capability's `Document Service Injection` requirement and by the modified `Dependency Layer Updates` above.
