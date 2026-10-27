## ADDED Requirements

### Requirement: Chunks expose provenance required for citation validity

The retrieval schema MUST expose document version, pipeline version, and graph-indexed timestamp or an explicit legacy
marker for every chunk. Evidence responses MUST include document version and pipeline version, and legacy rows MUST be
identifiable as such.

#### Scenario: Evidence reports pipeline provenance
- **WHEN** a chunk is returned by legal retrieval
- **THEN** its evidence envelope MUST include document version and pipeline version

#### Scenario: Legacy row is not presented as graph-verified
- **WHEN** a pre-migration chunk has no graph-indexed timestamp
- **THEN** the response MUST mark it legacy/unindexed rather than `graphiti_verified`

#### Scenario: Multi-contract chunk reports identity
- **WHEN** a chunk belongs to a document containing multiple contracts
- **THEN** its retrieval representation MUST carry `contract_index` for evidence and downstream findings

### Requirement: Tenant and version scope participate in derived keys

Search and answer caches and idempotency keys MUST include tenant, document versions, and pipeline version so a new
document version or another tenant cannot reuse a prior result.

#### Scenario: Version change invalidates reuse
- **WHEN** the same query is issued after a document version or pipeline version changes
- **THEN** the prior cached answer MUST NOT be returned as current

### Requirement: Document deduplication is tenant-aware

Document identity deduplication MUST use `(tenant_id, sha256)` as the unique key after tenant migration. The legacy
`(user_id, sha256)` constraint MUST be removed, while user identity remains available for audit.

#### Scenario: Teammates share one document identity
- **WHEN** two members of the same tenant upload the same content
- **THEN** the system MUST retain one document record and resolve both uploads to that tenant-scoped identity

#### Scenario: Different tenants may upload identical content
- **WHEN** users in different tenants upload identical bytes
- **THEN** each tenant MUST receive its own document identity and neither tenant may resolve the other's record
