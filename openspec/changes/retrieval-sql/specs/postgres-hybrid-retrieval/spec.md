## Purpose

Define how ranked retrieval executes against Postgres: one fused path reachable from every door, each
leg planned over its own index, the keyword extension's inverted score convention respected, ordering
deterministic, filters uniform across branches, tenancy enforced on the chunk relation, exact phrases
matched without a full-text vector column, and the extensions all of this depends on declared rather
than assumed.

## ADDED Requirements

### Requirement: Retrieval exposes exactly one fused search path

Every caller SHALL reach ranked retrieval through one fused implementation, and a second ranking or
fusion implementation SHALL NOT be reachable from the application.

#### Scenario: Both doors agree

- **WHEN** the retrieval graph and the search endpoint issue the same query with the same filters
- **THEN** both SHALL return the same ranked chunk identifiers in the same order

#### Scenario: A failed branch is not silently fused away

- **WHEN** a retrieval branch raises
- **THEN** the caller SHALL receive a failure naming the branch, and SHALL NOT receive results fused
  from only the surviving branches

#### Scenario: An empty branch is not a failure

- **WHEN** a retrieval branch returns no rows without raising
- **THEN** fusion SHALL proceed over the remaining branches, because returning nothing and failing are
  distinct outcomes

#### Scenario: No second fusion implementation remains

- **WHEN** the application is scanned for ranked-retrieval implementations
- **THEN** exactly one SHALL be found, and no repository method SHALL perform its own multi-branch
  ranking

### Requirement: Each retrieval leg is planned over its own index

A leg's ranking expression, ordering, and limit SHALL appear in the same statement, so the planner can
reach the index built for that leg.

#### Scenario: The keyword leg reaches the keyword index

- **WHEN** the query plan for the keyword leg is captured
- **THEN** it SHALL reference the keyword index rather than scanning a materialised intermediate
  relation

#### Scenario: The vector leg reaches the vector index

- **WHEN** the query plan for the vector leg is captured
- **THEN** it SHALL reference the vector index rather than scanning a materialised intermediate
  relation

#### Scenario: A leg carries its own ordering and limit

- **WHEN** a leg is evaluated
- **THEN** its ordering and limit SHALL appear in the same statement as its ranking expression, and
  SHALL NOT be applied to a set already reduced by an unordered limit

#### Scenario: Vector query-time tuning is applied on the path that runs

- **WHEN** the vector leg executes
- **THEN** the approximate-search query-time parameters SHALL be set in the same transaction as the
  scan they govern

### Requirement: Keyword scoring respects the inverted sign convention

The keyword extension returns negative relevance scores. Ordering and filtering SHALL be written for
that convention, and a change that inverts it SHALL fail the test suite.

#### Scenario: More relevant means more negative

- **WHEN** the keyword leg orders results
- **THEN** it SHALL sort the relevance expression ascending, so that a more-negative score ranks higher

#### Scenario: A non-negative score is rejected

- **WHEN** a keyword relevance score is not negative
- **THEN** it SHALL be excluded rather than treated as a weak match

#### Scenario: Inverting the convention breaks the build

- **WHEN** the ordering of the keyword relevance expression is inverted
- **THEN** the test suite SHALL fail

### Requirement: Fused ranking is deterministic and reproducible

The same query over unchanged data SHALL produce the same ordering every time, including among rows
that score equally.

#### Scenario: Two runs agree exactly

- **WHEN** the same query runs twice over unchanged data
- **THEN** the fused ordering SHALL be identical, including among equally-scored rows

#### Scenario: Every leg carries a tiebreaker

- **WHEN** a leg orders its results
- **THEN** its ordering SHALL include a unique tiebreaking column, in both its ranking expression and
  its statement ordering

### Requirement: Every retrieval branch honours the same filter surface

Filters SHALL be defined once and applied by every branch, so that adding a predicate does not require
editing each branch.

#### Scenario: No branch ignores a filter

- **WHEN** a filter is supplied
- **THEN** every branch SHALL apply it, and no branch SHALL silently ignore it

#### Scenario: A new predicate takes effect everywhere at once

- **WHEN** a predicate is added to the shared filter surface
- **THEN** it SHALL take effect on all branches without a per-branch edit

#### Scenario: Every supplied parameter is consumed

- **WHEN** filter parameters are built for a query
- **THEN** every parameter SHALL be consumed by every branch that receives it, so an unused parameter
  is detectable rather than inert

### Requirement: Retrieval is tenant-scoped on the chunk relation

The tenant predicate SHALL be applied to the chunk relation directly, so that tenant filtering happens
before rather than after approximate search.

#### Scenario: Tenancy does not depend on a join for correctness

- **WHEN** a search runs for a user
- **THEN** the tenant predicate SHALL be applied on the chunk relation, and SHALL NOT depend on a join
  to the document relation for correctness

#### Scenario: A small tenant is not starved by the candidate pool

- **WHEN** a user owns a small fraction of all chunks
- **THEN** the vector leg SHALL return fewer than the requested number of candidates only because
  fewer exist, and not because the approximate candidate pool was consumed by other tenants' rows
  before filtering

#### Scenario: The document join survives only where it projects

- **WHEN** a leg does not project a document column
- **THEN** it SHALL NOT join the document relation

### Requirement: Exact-phrase retrieval works without a full-text vector column

A quoted phrase SHALL match literally, achieved by over-fetching ranked candidates and post-filtering
them, because the keyword extension provides no phrase query.

#### Scenario: A quoted phrase matches literally

- **WHEN** a query carries a quoted phrase
- **THEN** every returned chunk SHALL contain that phrase literally

#### Scenario: Words apart are not a phrase match

- **WHEN** a chunk contains every word of a phrase but not the phrase itself
- **THEN** it SHALL be excluded from the phrase-filtered results

#### Scenario: Pattern metacharacters are literal

- **WHEN** a phrase contains a character that the pattern-matching operator treats as a wildcard
- **THEN** that character SHALL match literally and SHALL NOT act as a wildcard

#### Scenario: Both callers get phrase filtering

- **WHEN** either the retrieval graph or the search endpoint supplies a phrase
- **THEN** the phrase filter SHALL be applied, because it is part of the shared branch input rather
  than one path's local behaviour

### Requirement: Application code does not use a full-text vector column

Full-text vector constructs SHALL NOT appear in application code, so that the lexical signal is not
double-counted across two branches of the fusion.

#### Scenario: No full-text vector construct in application code

- **WHEN** application source is scanned
- **THEN** no full-text vector, full-text query, or vector-construction call SHALL appear

#### Scenario: Historical migrations are exempt and the exemption is recorded

- **WHEN** the prohibition is enforced
- **THEN** migration history SHALL be excluded as an immutable historical record, and that exclusion
  SHALL be stated rather than implied

### Requirement: Required extensions are declared, not assumed

The schema chain SHALL create every extension the retrieval indexes depend on, before any index
depending on one is built.

#### Scenario: A fresh environment provisions its own extensions

- **WHEN** a fresh environment is provisioned from the migration chain alone
- **THEN** every extension the retrieval indexes require SHALL be created by that chain before those
  indexes are built

#### Scenario: The keyword access method is verified to exist

- **WHEN** the keyword extension is present
- **THEN** an access method under the name the schema and queries use SHALL be registered, and its
  absence SHALL be reported as a provisioning failure rather than discovered at query time

### Requirement: Fusion weights are explicit per leg

Each branch's contribution to the fused score SHALL come from a named constant, and changing one SHALL
change the fused ordering.

#### Scenario: A weight change changes the order

- **WHEN** three branches are fused and one branch's weight is changed
- **THEN** the fused ordering SHALL change

#### Scenario: Weights come from named constants

- **WHEN** a branch's weight is applied
- **THEN** its value SHALL come from a named constant rather than a literal at the fusion site

#### Scenario: The default is unweighted

- **WHEN** a caller fuses without supplying weights
- **THEN** fusion SHALL be unweighted, so that an existing caller's behaviour is unchanged by the
  introduction of weights
