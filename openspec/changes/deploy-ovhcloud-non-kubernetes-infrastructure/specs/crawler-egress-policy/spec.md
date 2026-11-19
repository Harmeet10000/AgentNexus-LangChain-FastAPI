# crawler-egress-policy

## ADDED Requirements

### Requirement: Provider deployment proves the non-bypass boundary
Production deployment acceptance MUST demonstrate from the crawler worker identity that direct public egress, cloud metadata, loopback-to-host services, private application subnets, database ports, and management endpoints are unreachable, while the exact broker, DNS, proxy, and mutually authenticated Redis-relay paths remain available.

#### Scenario: Direct public bypass is attempted in OVHcloud
- **WHEN** a crawler process bypasses proxy variables and dials a controlled public endpoint directly
- **THEN** the network denies the connection while the same HTTP(S) request through the policy boundary is evaluated normally

#### Scenario: Cloud metadata is requested
- **WHEN** the crawler or a crawled page attempts to connect to the link-local cloud metadata address
- **THEN** both direct routing and proxied connection policy deny the request before metadata is returned

#### Scenario: Application data plane is scanned
- **WHEN** the crawler attempts private connections to PostgreSQL, MongoDB, Neo4j, object storage administration, telemetry administration, or unrelated Redis/RabbitMQ endpoints
- **THEN** every connection is denied and the denial is observable without revealing credentials or topology to the crawl result

#### Scenario: Provider security group contains implicit egress
- **WHEN** the provider or OpenStack control plane retains an allow-egress default outside the declared Terraform rules
- **THEN** the crawler remains unable to bypass the proxy because host routing and nftables enforce the boundary independently

### Requirement: Test colocation does not masquerade as production isolation
A cost-reduced test deployment MAY colocate crawler boundary components, but production acceptance MUST identify and verify the stronger machine- or equivalent enforcement boundary that separates untrusted crawler execution from permitted egress mediation.

#### Scenario: Test uses one host
- **WHEN** the worker, proxy, and relay share a test host
- **THEN** the acceptance report records that container-network evidence alone does not prove production machine-level isolation
