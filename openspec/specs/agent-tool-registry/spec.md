# agent-tool-registry Specification

## Purpose

Defines how agent tools are registered, discovered by tag, resolved by name, and assigned to agent roles, so that
exactly one registry is authoritative for the whole application and an unknown tool name can never be mistaken for a
tool that exists but returned nothing.

## Requirements

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

### Requirement: Tool execution is denied without a server-side capability grant

Every tool in the system/filesystem band SHALL execute only when the server-owned tool
runtime context grants that tool's capability. A missing runtime, or a runtime whose
context does not carry the capability, SHALL produce a typed failure result and SHALL
NOT execute the operation. The capability SHALL NOT be acceptable as a model-visible
tool argument.

#### Scenario: No runtime context at all

- **WHEN** a system/filesystem tool is invoked with no tool runtime
- **THEN** the result is a failure naming the required capability
- **AND** the operation did not execute

#### Scenario: A grant for a different capability does not authorize

- **WHEN** the runtime context grants `filesystem.read` and `write_file` is invoked
- **THEN** the result is a failure naming `filesystem.write`
- **AND** no file is written

#### Scenario: The granted path executes

- **WHEN** the runtime context grants the tool's capability and all other guards pass
- **THEN** the tool executes and returns its normal result envelope

### Requirement: Path-taking tools are confined to the configured workspace jail

Every tool that accepts a filesystem path or working directory SHALL resolve the path
and require the resolved path to sit inside one of the configured workspace roots.
Symlinks SHALL be resolved before the containment check. When no workspace root is
configured, the tool SHALL refuse with a message naming the configuration to set. The
shell tool additionally SHALL refuse when invoked without an explicit working
directory inside the jail.

#### Scenario: Jail unset means everything refused

- **WHEN** no workspace roots are configured and a file tool is invoked with a granted capability
- **THEN** the result is a failure naming the workspace-root setting
- **AND** the operation did not execute

#### Scenario: A path inside the jail is served

- **WHEN** the jail is configured and the requested path resolves inside a configured root
- **THEN** the tool executes against the resolved path

#### Scenario: Symlink escape is denied

- **WHEN** a symlink inside the jail points outside every configured root and a file
  tool is pointed at the link
- **THEN** the result is a failure naming the workspace jail
- **AND** the operation did not execute

#### Scenario: Shell without a working directory is refused

- **WHEN** the shell tool is invoked with its capability granted but no explicit
  working directory
- **THEN** the result is a failure explaining the jailed-working-directory requirement

### Requirement: Registered tools carry authority metadata

The five system/filesystem tools SHALL be registered with authority metadata — the
capability they enforce and a side-effect class — alongside their existing category
tags. Selection by the legacy category tags SHALL behave exactly as before.

#### Scenario: Authority metadata is discoverable

- **WHEN** the default tool set is registered and the registry is queried for the
  shell or filesystem tools
- **THEN** each tool's metadata names its enforced capability and side-effect class
- **AND** selection by the legacy category tags returns the same tools as before

### Requirement: Refusals name their fix

A refusal caused by a missing grant, an unconfigured jail, or a jail violation SHALL
name the missing capability or the setting to configure, in the failure message the
caller (and the operator reading logs) sees.

#### Scenario: The refusal is greppable

- **WHEN** any refusal in this capability is read from a result or a log
- **THEN** the message names either the missing capability identifier or the
  configuration key whose absence caused it

### Requirement: Argument schemas are plain schema models

A class used as a tool's argument schema SHALL be a plain schema model, not a tool
type. Every model-facing argument schema SHALL expose exactly its declared fields and
SHALL NOT leak injected runtime context into the model-visible schema.

#### Scenario: The web-search input schema is schema-only

- **WHEN** the web-search tool's argument schema is inspected
- **THEN** it is a plain schema model exposing only its declared fields
- **AND** it is not itself registered or usable as a tool

### Requirement: Registration carries structured authority metadata

The registry's registration surface SHALL accept and store structured authority
metadata (capability identifier, side-effect class, egress classification) for every
tool, and SHALL expose it for policy evaluation and projection checks.

#### Scenario: The five system/filesystem tools migrate losslessly

- **WHEN** the default tool set registers with structured metadata
- **THEN** each tool's capability identifier and side-effect class match the values
  its interim authority tags recorded
- **AND** the enforcement suite for those tools passes unmodified

### Requirement: The web-search tool honours the executor grant contract

The web-search tool SHALL resolve its server-owned runtime context through the same
injection seam as the evidence and system/filesystem tools, and SHALL refuse without
its `web.search` grant.

#### Scenario: Schema stays model-clean

- **WHEN** the web-search tool's model-facing schema is inspected after the seam
  migration
- **THEN** it exposes exactly its declared query arguments
- **AND** the runtime context is not a model-visible argument
