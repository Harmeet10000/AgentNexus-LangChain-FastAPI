# semantic-answer-cache

## Purpose

Serve ask answers for questions that are semantically similar — not byte-identical — to
questions answered before, without re-running the retrieval-and-generation pipeline, while
keeping every cached answer reachable only within the authorization partition of the request
that would run without the cache.

## ADDED Requirements

### Requirement: A near-duplicate question is answered from the cache before generation runs

The ask path SHALL attempt the exact-match cache first and, on an exact-match miss, SHALL
attempt a similarity-matched lookup against previously stored ask answers before generating a
fresh answer. A stored answer served by similarity SHALL be the complete answer envelope the
fresh run would have produced (answer text, confidence, citations, coverage, warnings,
evidence policy), and the pipeline that produces fresh answers SHALL NOT run for the served
request.

#### Scenario: Paraphrased repeat serves the stored envelope

- **WHEN** a request's query differs in wording from a previously answered query but the two
  carry the same meaning within the similarity threshold
- **THEN** the stored answer envelope for that earlier query SHALL be served without the
  retrieval-and-generation pipeline running for the new request

#### Scenario: A dissimilar query runs the pipeline unchanged

- **WHEN** no stored entry under the request's authorization partition is similar within the
  threshold
- **THEN** the ask path SHALL run exactly as it does without this capability, and a fresh
  answer SHALL be produced

### Requirement: Similarity never substitutes for authorization partition identity

A similarity lookup SHALL consider only entries stored under the same authorization partition
as the request: the requesting user, the tenant, the authorized document scope, the resolved
document version of every scoped document, and the evidence policy. Entries stored under a
different value of any partition member SHALL be unreachable by similarity lookup regardless
of their measured similarity, and similarity SHALL be evaluated only after the partition
constraint is applied.

#### Scenario: The same question from a different user never matches

- **WHEN** two different users within the same tenant submit semantically identical queries
- **THEN** the second request SHALL NOT be served from the entry stored for the first user

#### Scenario: The same question under a different authorized scope never matches

- **WHEN** a request's authorized document scope or evidence policy differs from the scope or
  policy an entry was stored under
- **THEN** that entry SHALL be unreachable by similarity lookup for the request, whatever its
  similarity

### Requirement: Degraded answers never enter the similarity layer

An answer envelope produced by a degraded generation — the system's fallback output, whether
from a generation failure, a budget refusal, or a grade-cap escape — SHALL remain eligible for
exact-match caching exactly as today, but SHALL be excluded from similarity-lookup storage. A
similarity match SHALL never serve a degraded envelope, and when the degraded status of an
envelope cannot be determined, the store SHALL treat it as degraded (fail closed).

#### Scenario: A transient generation failure does not spread by similarity

- **WHEN** a run's generation falls back to the degraded envelope and the run is finalised
- **THEN** the degraded envelope SHALL be stored for exact-match serving as today, and SHALL
  NOT be stored for similarity lookup, so paraphrases of that query produce fresh runs

#### Scenario: An unclassifiable envelope is treated as degraded

- **WHEN** an envelope's degraded status cannot be evaluated at store time
- **THEN** the store SHALL exclude it from the similarity layer

### Requirement: Similarity is only compared within one vector space

A similarity lookup SHALL match only entries embedded by the same embedding model, with the
same output dimension, as the current lookup embedding. Entries stored under a different
embedding model or output dimension SHALL be unreachable regardless of measured distance, and
the vector index SHALL be scoped so a dimension change never compares vectors of different
widths.

#### Scenario: An embedding model change stops cross-space matching

- **WHEN** the configured embedding model changes and a near-duplicate query is asked
- **THEN** entries stored under the previous embedding model SHALL NOT be served by
  similarity, and the request SHALL produce a fresh answer

#### Scenario: A dimension change never compares mismatched widths

- **WHEN** the configured embedding dimension changes
- **THEN** similarity lookup and store SHALL operate on an index scoped to the new dimension,
  and no lookup SHALL compare vectors of different widths

### Requirement: Similarity is threshold-gated and deployment-configurable

The system SHALL declare two entries similar only when their measured semantic distance is at
or below a configured threshold.

| Scale | Meter | MUST |
|---|---|---|
| Semantic distance between request query and stored question (normalized, lower = more similar) | Deployment setting, readable without code change, changeable without redeploy of the corpus | Serving a stored entry only at distance ≤ the configured threshold; default tight enough that distinct-meaning queries are not matched |

#### Scenario: An above-threshold entry is not served

- **WHEN** the nearest partition-eligible entry's distance exceeds the configured threshold
- **THEN** the ask path SHALL treat the lookup as a miss and produce a fresh answer

#### Scenario: The threshold is a deployment setting

- **WHEN** an operator changes the similarity threshold in configuration
- **THEN** subsequent lookups SHALL apply the new threshold without a code change

### Requirement: Every fresh answer is stored for future similarity lookup

Whenever the ask path generates a fresh answer that qualifies for exact-match caching, the
system SHALL also store that answer, its query, and its authorization partition for future
similarity lookup, under the TTL discipline configured for the semantic cache, independent of
the exact-match cache's TTL. An answer
that does not qualify for exact-match caching SHALL NOT become similarity-served either.

#### Scenario: A fresh answer becomes similarity-eligible

- **WHEN** a fresh answer is generated and cached under the exact-match discipline
- **THEN** a subsequent near-duplicate request within the same authorization partition SHALL
  be eligible to serve it by similarity until the entry expires

#### Scenario: A non-cacheable answer is never similarity-served

- **WHEN** a fresh answer was excluded from exact-match caching
- **THEN** that answer SHALL NOT be stored for similarity lookup

### Requirement: Cache embeddings resolve to the single embedding path

Embeddings produced for similarity lookup and for stored entries SHALL be produced by the
system's single embedding path with the same configuration as retrieval embeddings. No second
embedding implementation SHALL be reachable from the live ask path for cache purposes, and
the embedding client serving the cache SHALL be constructed once and reused.

#### Scenario: Lookup and retrieval embeddings come from one path

- **WHEN** a request is embedded for similarity lookup
- **THEN** the embedding SHALL come from the same embedding path and configuration used for
  retrieval query embeddings

### Requirement: Stored query text retention is explicit and bounded

The similarity layer SHALL store the query text of each entry no longer than the entry's TTL,
in the configured text form — verbatim or a restricted form (e.g., truncated) — applied
identically to stores and lookups. The searchability of stored query text SHALL be a declared
property of this capability, and the matching vector SHALL be computed from the full query
regardless of the stored text form.

#### Scenario: Restricted text form narrows the searchable surface without affecting matching

- **WHEN** the deployment configures a restricted query-text form
- **THEN** stored entries SHALL carry only the restricted form as searchable text, while
  similarity matching SHALL continue to operate on embeddings of the full query

#### Scenario: Entry expiry removes query text with the entry

- **WHEN** an entry reaches its TTL
- **THEN** its query text SHALL expire together with the envelope — no query text outlives
  its entry

### Requirement: Degradation reproduces the uncached ask path

When the cache backend is unreachable, the backend's search capability is absent, the lookup
fails or exceeds its deadline, or similarity evaluation fails, the ask path SHALL run exactly
as it does without this capability. A cache failure SHALL never fail a request, alter a fresh
answer, or exceed a bounded deadline, and every degradation SHALL be recorded in logs.

#### Scenario: An unavailable backend leaves the ask path unchanged

- **WHEN** the cache backend is unreachable at lookup or store time
- **THEN** the request SHALL complete with a fresh answer produced exactly as the uncached
  path produces it, and the degradation SHALL appear in logs

#### Scenario: An absent search capability disables the layer for the process

- **WHEN** the backend reports the search capability unavailable at process start
- **THEN** similarity lookup and store SHALL be disabled for the process lifetime, the
  exact-match cache SHALL be unaffected, and the disabling SHALL appear in logs

### Requirement: Cache hits carry truthful provenance

An ask response served from cache SHALL indicate its serving mode with a provenance value
distinguishing exact-match from similarity-match. A response carrying no provenance value
SHALL be interpreted as served fresh, so the field is additive and existing consumers stay
compatible. Every cache-served response SHALL also be recorded in logs with its provenance.

#### Scenario: Each serving mode is distinguishable

- **WHEN** responses are served fresh, by exact match, and by similarity match
- **THEN** the two cache-served responses SHALL carry distinct provenance values (exact-match
  vs similarity-match) and the fresh response SHALL carry none

### Requirement: Semantic hits serve through the exact-hit boundary

A similarity-served answer SHALL produce the same downstream result as an exact-match hit of
the same envelope: the same serving path, the same downstream state shape (answer, citations,
coverage, warnings, evidence policy), and zero stage spend. Any consumer that derives state
from a served envelope — including agent consumers deriving evidence handles and ledger
charges — SHALL derive identical values whether the hit was exact or semantic.

#### Scenario: An agent consumer cannot distinguish an exact hit from a semantic hit

- **WHEN** an agent consumer receives a served envelope via a semantic hit
- **THEN** the derived evidence references, coverage, and stage charges SHALL be identical to
  an exact hit of that envelope, with zero stage spend on both

#### Scenario: Semantic serving adds no second serving path

- **WHEN** the serving behavior of exact hits and semantic hits is compared
- **THEN** both SHALL flow through the same serving boundary, and no serving logic SHALL
  exist for semantic hits that does not also govern exact hits

### Requirement: Cache hits have explicit budget semantics

A cache-served answer SHALL charge zero retrieval-and-generation stage spend to the run's
budget — the ledger consumes only the calls that actually ran. The upstream intent-scoring
call SHALL remain charged for cache-served runs. A run whose budget is exhausted before the
engine invocation SHALL NOT reach the cache lookup; the exhaustion gate keeps its current
position ahead of any cache read, so cache-as-rescue for exhausted runs is out of scope.

#### Scenario: A cache-served run is charged only for what ran

- **WHEN** a run is served from cache by exact match or similarity
- **THEN** the run's ledger SHALL show zero stage spend for the served engine stages and
  SHALL show only the actually-executed upstream charges

#### Scenario: An exhausted run cannot read the cache

- **WHEN** a run's budget is exhausted before the engine invocation
- **THEN** the run SHALL be refused as today, and the cache lookup SHALL NOT be consulted

### Requirement: The similarity lookup is time-bounded

| Scale | Meter | MUST |
|---|---|---|
| Time from lookup start to lookup outcome | Deployment deadline setting, enforced by the cache adapter | 100% of lookups produce a hit or a degradation within the deadline; an overrun is treated as degradation |

#### Scenario: A slow backend degrades rather than stalls the ask

- **WHEN** the similarity lookup exceeds its configured deadline
- **THEN** the lookup SHALL be abandoned as a degradation and the ask path SHALL produce a
  fresh answer

### Requirement: Observe mode validates the threshold without serving

The system SHALL support an observe mode, selected by deployment configuration, in which
similarity lookups run and record their outcome — including the nearest candidate, its
measured distance, and the would-serve decision — but no cached answer is ever served and the
fresh pipeline always runs. Observe mode SHALL suspend only the serving behavior of this
capability; recording, degradation, and store behavior SHALL operate as configured. Moving
from observe mode to serving SHALL require configuration only, never a code change.

#### Scenario: Observe mode records without serving

- **WHEN** observe mode is enabled and a lookup finds an above-threshold candidate
- **THEN** the candidate, its distance, and the would-serve decision SHALL be recorded, the
  fresh pipeline SHALL run for the request, and no cached answer SHALL be served

#### Scenario: Observe mode is distinguishable from serving in records

- **WHEN** logs are reviewed across a period containing both observe-mode and serve-mode
  lookups
- **THEN** an outcome record SHALL identify whether its lookup would have served or did
  serve, so the two modes cannot be confused in analysis

### Requirement: Lookups record their measured distance for tuning

Every partition-filtered similarity lookup SHALL record in logs the measured semantic distance
of its nearest partition-eligible candidate together with the lookup outcome — served,
below-threshold miss, or degraded — even when no entry is served, so the distance distribution
per outcome can be reviewed for threshold tuning without reproducing queries.

#### Scenario: Threshold tuning works from recorded distances

- **WHEN** lookups run across traffic that includes hits, threshold misses, and degraded
  lookups
- **THEN** each lookup's nearest-candidate distance and outcome SHALL appear in logs, grouped
  by outcome
