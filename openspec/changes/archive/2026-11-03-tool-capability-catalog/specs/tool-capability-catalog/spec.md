## Purpose

Define the declarative authority layer over the tool registry: what a tool declares
about itself, who may receive it, and the invariant that the model-visible projection
is always a subset of the policy-authorized set.

## ADDED Requirements

### Requirement: Every tool declares structured authority metadata

Registration SHALL carry structured authority metadata — the enforced capability
identifier, a side-effect class, and an egress classification — for every registered
tool. The interim authority tags of `agent-tool-capability-enforcement` SHALL be
derived from (and consistent with) the structured fields, and legacy category tags
SHALL continue to select exactly as before.

#### Scenario: Metadata is authoritative, tags are views

- **WHEN** a tool's structured authority metadata is read and compared with its
  authority tags
- **THEN** they agree for every registered tool
- **AND** a disagreement is a registration error, not a silent drift

### Requirement: One policy source maps roles to granted capabilities

There SHALL be exactly one declarative policy mapping each agent role (and the
unbound/system context) to its granted capabilities. Runtime capability tuples SHALL
be populated from that policy; no call site SHALL assemble a capability grant ad hoc.
A role that grants no policy SHALL receive an empty grant set.

#### Scenario: The policy is the only author of grants

- **WHEN** the application builds a tool runtime context for any role
- **THEN** the context's capability set equals the policy's entry for that role

#### Scenario: An unknown role grants nothing

- **WHEN** a runtime context is built for a role the policy does not name
- **THEN** the capability set is empty (deny-by-default at the policy layer)

### Requirement: The selector shrinks and never widens

The relevance-based tool selector SHALL choose only among tools the role's policy has
authorized for the current context. A tool excluded by policy SHALL NOT be selectable
at any relevance score, and the executor's grant check SHALL remain the final gate.

#### Scenario: High relevance cannot resurrect an unauthorized tool

- **WHEN** the selector ranks a tool the role's policy does not authorize above every
  authorized tool
- **THEN** the unauthorized tool is absent from the resulting projection

#### Scenario: The projection is a subset of the authorization

- **WHEN** any role's model-visible toolset is inspected at any turn
- **THEN** every visible tool is authorized by that role's policy entry

### Requirement: Egress-classified tools require their grant at execution

A tool classified as egress (model-chosen content leaves the trust boundary) SHALL
require its capability grant at execution under the same executor contract as every
other tool, and its refusal SHALL name the missing grant. The web-search tool SHALL
be classified egress and gated accordingly.

#### Scenario: Web search without a grant refuses

- **WHEN** `web_search` is invoked from a context whose policy did not grant
  `web.search`
- **THEN** the result is a typed refusal naming the capability
- **AND** no query text leaves the trust boundary

#### Scenario: The granted path works unchanged

- **WHEN** the policy grants `web.search` to the invoking context
- **THEN** web search executes exactly as before the gate
