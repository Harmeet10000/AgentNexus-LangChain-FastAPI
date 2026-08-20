## Purpose

Defines how agent tools are registered, discovered by tag, resolved by name, and assigned to agent roles, so that
exactly one registry is authoritative for the whole application and an unknown tool name can never be mistaken for a
tool that exists but returned nothing.

## ADDED Requirements

### Requirement: One registry of record for agent tools

The system SHALL expose exactly one authoritative registry of agent tools. Every lookup of a tool by name, and every
selection of tools by tag, SHALL resolve through that single registry. No second registry of agent tools SHALL be
reachable from application code.

#### Scenario: Tool lookup by name resolves through the single registry

- **WHEN** any part of the application resolves an agent tool by name
- **THEN** the tool SHALL be returned by the single authoritative registry
- **AND** no alternative registry SHALL be consulted

#### Scenario: The published registry symbol names the authoritative registry

- **WHEN** application code obtains the tool registry from the agent tools package
- **THEN** it SHALL receive the authoritative registry
- **AND** it SHALL NOT receive a second, differently-behaving registry with the same name

### Requirement: The registry is populated by explicit registration before any consumer resolves a tool

Population SHALL be performed by an explicit registration entry point, and that entry point SHALL have been called
before any consumer resolves a tool by name or selects tools by tag. Population SHALL NOT depend on a module import
side effect, and SHALL NOT depend on the order in which modules are imported.

#### Scenario: The registration entry point populates the registry

- **WHEN** the explicit tool-registration entry point is called
- **THEN** the registry SHALL report a non-empty set of registered tool names

#### Scenario: Importing the package registers nothing on its own

- **WHEN** the agent tools package is imported and the registration entry point has not been called
- **THEN** no tool registration SHALL have occurred as a side effect of that import

#### Scenario: Registration is idempotent

- **WHEN** the registration entry point is called more than once in one process
- **THEN** the registry SHALL hold each registered tool exactly once

#### Scenario: Every tool advertised as registered is resolvable

- **WHEN** the registry reports a tool name as registered
- **THEN** resolving that name SHALL return a usable tool

#### Scenario: Resolving before registration fails loudly

- **WHEN** a consumer resolves a tool by name before the registration entry point has been called
- **THEN** resolution SHALL raise rather than return an empty or absent tool

### Requirement: Resolving an unknown tool name fails loudly

Resolving a tool name that is not registered SHALL raise an error. It SHALL NOT return an empty value, and it SHALL
NOT be silently skipped when building an agent.

#### Scenario: Unknown tool name raises

- **WHEN** a caller resolves a tool name that is not registered
- **THEN** the system SHALL raise an error naming the unresolved tool

#### Scenario: Agent construction with an unresolvable tool name does not silently degrade

- **WHEN** an agent is configured with a tool name that is not registered
- **THEN** construction SHALL fail
- **AND** the agent SHALL NOT be created with that tool silently omitted

### Requirement: Tools are selectable by tag

The registry SHALL support selecting the set of tools carrying a given tag, and SHALL be the only mechanism by which
grouped tool sets are obtained.

#### Scenario: Selecting a tag returns every tool carrying it

- **WHEN** a caller selects tools by a tag
- **THEN** the registry SHALL return every registered tool carrying that tag
- **AND** SHALL return no tool that does not carry it

#### Scenario: Web-capable tools are reachable by their tag

- **WHEN** a caller selects the web-capable tool group
- **THEN** the web search and page-retrieval tools SHALL be included

### Requirement: Every agent role receives the tools assigned to it

Each agent role SHALL be constructed with the tool set assigned to that role. No agent role that is documented as
tool-using SHALL be constructed with an empty tool set. Every one of the three agent roles is tool-using: the
orchestrating role delegates through handoff tools, and the two reasoning roles retrieve evidence.

#### Scenario: The orchestrator role receives its delegation tools

- **WHEN** the orchestrating agent is constructed
- **THEN** it SHALL receive the handoff tools by which it delegates work to the other agent roles

#### Scenario: The compliance role receives its statutory tools

- **WHEN** the compliance agent is constructed
- **THEN** it SHALL receive the precedent-search and statute-retrieval tools

#### Scenario: The risk role receives its graph tools

- **WHEN** the risk analysis agent is constructed
- **THEN** it SHALL receive the knowledge-graph query and obligation-chain tools

#### Scenario: No tool-using agent is constructed with an empty tool set

- **WHEN** any tool-using agent role is constructed
- **THEN** its tool set SHALL be non-empty
