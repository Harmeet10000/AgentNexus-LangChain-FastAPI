## Purpose

Move the tool registry's authority contract from interim tags to structured,
policy-backed metadata, and make web-search egress gated like any other capability.

## ADDED Requirements

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
