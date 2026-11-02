## Purpose

Extend the tool-registry capability with the execution-authority contract for the
system/filesystem band: grants are enforced at the executor, paths are jailed, and
refusals are self-explanatory. (Capability exists — this delta only adds requirements.)

## ADDED Requirements

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
