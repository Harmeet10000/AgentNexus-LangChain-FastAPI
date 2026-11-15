# semantic-answer-cache

## Purpose

Serve ask answers for questions that are semantically similar — not byte-identical — to
questions answered before, without re-running the retrieval-and-generation pipeline, while
keeping every cached answer reachable only within the authorization partition of the request
that would run without the cache.

## ADDED Requirements

### Requirement: A near-duplicate question is answered from the cache before generation runs

After the existing planning step, the ask path SHALL attempt the exact-match cache first
and, on an exact-match miss with semantic mode enabled, SHALL
attempt a similarity-matched lookup against previously stored ask answers before generating a
fresh answer. A stored answer served by similarity SHALL be the complete answer envelope the
fresh run would have produced (answer text, confidence, citations, coverage, warnings,
evidence policy). In serve mode the downstream retrieval, grading, and generation steps
SHALL NOT run for a semantic hit; planning and upstream work retain their current ordering.

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
same output dimension, and under the same embedding-contract revision, as the current lookup
embedding. Entries stored under a different model, output dimension, or contract revision
SHALL be unreachable regardless of measured distance. The vector index SHALL be scoped so a
dimension change never compares vectors of different widths, and the embedding path SHALL
export the contract revision used by the cache rather than letting the cache define its own.

#### Scenario: An embedding model change stops cross-space matching

- **WHEN** the configured embedding model changes and a near-duplicate query is asked
- **THEN** entries stored under the previous embedding model SHALL NOT be served by
  similarity, and the request SHALL produce a fresh answer

#### Scenario: A dimension change never compares mismatched widths

- **WHEN** the configured embedding dimension changes
- **THEN** similarity lookup and store SHALL operate on an index scoped to the new dimension,
  and no lookup SHALL compare vectors of different widths

#### Scenario: A query-instruction change makes old entries unreachable

- **WHEN** the embedding path's query-side instruction contract changes while the model and
  output dimension stay the same
- **THEN** the exported contract revision SHALL change, and entries stored under the earlier
  revision SHALL NOT be served by similarity even though their model and width match

### Requirement: Similarity is threshold-gated and deployment-configurable

The system SHALL declare two entries similar only when their measured semantic distance is at
or below a configured threshold. The default threshold SHALL be documented as a conservative
policy choice, not as a proven safety bound, and serve mode SHALL additionally require the
labeled evaluation in the threshold-acceptance requirement.

| Scale | Meter | MUST |
|---|---|---|
| Semantic distance between request query and stored question (normalized, lower = more similar) | Deployment setting, readable without code change, changeable without redeploy of the corpus | Serving a stored entry only at distance ≤ the configured threshold; a documented default that is conservative for distinct-meaning queries |

#### Scenario: An above-threshold entry is not served

- **WHEN** the nearest partition-eligible entry's distance exceeds the configured threshold
- **THEN** the ask path SHALL treat the lookup as a miss and produce a fresh answer

#### Scenario: The threshold is a deployment setting

- **WHEN** an operator changes the similarity threshold in configuration
- **THEN** subsequent lookups SHALL apply the new threshold without a code change

### Requirement: Threshold acceptance is evidenced by labeled safe-substitution evaluation

Before serve mode may be enabled, the system SHALL produce a repeatable evaluation report over
labeled `(query, candidate answer, evidence)` cases. The false-hit rate SHALL mean unsafe
substitutions divided by all would-serve decisions at the fixed threshold, reported
separately from hit rate and false positives among known-negative pairs. The latter
denominator SHALL NOT substitute for the served-answer error rate. The evaluation SHALL
use calibration and held-out, query-family-disjoint splits, and SHALL include adversarial
negation, party swap, jurisdiction, and multilingual cases. Serve mode SHALL require a
one-sided 95% upper confidence bound on the held-out false-hit rate of at most 1%; with
zero observed errors, this requires at least 299 independent would-serve cases sampled
from the intended workload. This is an initial risk-policy gate, not a correctness proof.
The report SHALL state its threshold, embedding-contract identity, workload and partition
sampling procedure, sample sizes, confidence method, and observed errors. No would-serve
cases means insufficient evidence, not zero error. Repeated paraphrases from one query
family SHALL NOT count as independent samples. A threshold or embedding-contract change
SHALL require new held-out acceptance evidence; failed held-out cases SHALL NOT be reused
for tuning while still being reported as unseen acceptance data.

#### Scenario: A zero-error evaluation with too few would-serve cases is insufficient

- **WHEN** the held-out evaluation has zero false hits but fewer than 299 independent
  would-serve cases
- **THEN** the threshold SHALL NOT be accepted for serve mode, because the one-sided 95%
  false-hit upper bound exceeds 1%

#### Scenario: Adversarial must-not-serve pairs are never served

- **WHEN** the held-out evaluation includes negation, party-swap, jurisdiction, or
  multilingual pairs whose answers must not be substituted
- **THEN** none of those pairs SHALL be served, and any false hit SHALL be reported and block
  acceptance at the configured bound

#### Scenario: Calibration data does not double as acceptance data

- **WHEN** the threshold is selected using the calibration split
- **THEN** the acceptance metric SHALL be computed only on the held-out,
  query-family-disjoint split

### Requirement: A qualifying fresh answer is stored for future similarity lookup

Whenever the ask path generates a fresh answer that qualifies for exact-match caching, and
semantic mode is not `off`, and the answer is not degraded, the system SHALL also store that
answer, a stable query digest plus a bounded preview, its authorization partition, and its
embedding identity for future similarity lookup. The store SHALL use the semantic cache's
configured maximum lifetime, independent of the exact-match cache's TTL. An answer that does
not qualify for exact-match caching, a bypassed request, or an `off`-mode request SHALL NOT
be stored for similarity lookup.

#### Scenario: A fresh answer becomes similarity-eligible

- **WHEN** a fresh, non-degraded answer is generated and cached under the exact-match
  discipline while semantic mode is `observe` or `serve`
- **THEN** a subsequent near-duplicate request within the same authorization partition and
  embedding identity SHALL be eligible to serve it by similarity until the entry expires

#### Scenario: A non-cacheable answer is never similarity-served

- **WHEN** a fresh answer was excluded from exact-match caching, the request bypassed caching,
  or semantic mode is `off`
- **THEN** that answer SHALL NOT be stored for similarity lookup

### Requirement: A semantic entry has a non-sliding maximum lifetime

A fresh generation SHALL carry a stable generation identity and absolute expiry fixed
before cache publication and preserved across retries and checkpoint resume. The expiry
SHALL be no later than generation completion plus the configured lifetime. It SHALL
NOT be extended by similarity hits, exact-match hits, observe-mode checks, replays, or any
other read. A subsequent store for an existing entry SHALL NOT refresh its expiry, whether
or not it replaces the entry's value. An entry whose lifetime has elapsed, or whose expiry
metadata is missing, SHALL be treated as a miss. The stored query representation and the
full answer envelope SHALL share that lifetime. Entry publication and expiry assignment
SHALL be atomic. Replaying an expired generation SHALL NOT recreate it even if its Redis
key no longer exists; a genuinely new generation may create a new entry.

#### Scenario: Repeated hits do not extend entry lifetime

- **WHEN** an entry is served by similarity repeatedly and also checked in observe mode
- **THEN** its expiry SHALL remain the original absolute expiry,
  and the entry SHALL become a miss at that time regardless of hit frequency

#### Scenario: Replay does not refresh expiry

- **WHEN** a graph interrupt or replay re-executes a path that would store the same entry
- **THEN** the existing entry's expiry SHALL NOT be extended, and an expired generation
  SHALL NOT be recreated after its key has disappeared

#### Scenario: A store interruption cannot publish an immortal entry

- **WHEN** a cache write times out or the client disconnects during publication
- **THEN** the entry SHALL either be absent or have a finite expiry no later than the
  generation's original deadline; no follow-up lookup or cleanup timer SHALL be required
  to assign its expiry

#### Scenario: Expired query representation and envelope disappear together

- **WHEN** an entry reaches its maximum lifetime
- **THEN** the stored query preview and the answer envelope SHALL expire together, and a
  subsequent lookup SHALL be a miss

### Requirement: Cache embeddings resolve to the single embedding path

Embeddings produced for similarity lookup and for stored entries SHALL be produced by the
system's single embedding path with the same configuration as retrieval embeddings. No second
embedding implementation SHALL be reachable from the live ask path for cache purposes, and
the embedding client serving the cache SHALL be constructed once and reused. That path SHALL
export a public embedding-contract revision that changes whenever the query-side instruction
template, provider-input contract, normalisation, or output-space namespace changes; the
cache SHALL use that exported revision for identity.

#### Scenario: Lookup and retrieval embeddings come from one path

- **WHEN** a request is embedded for similarity lookup
- **THEN** the embedding SHALL come from the same embedding path and configuration used for
  retrieval query embeddings

#### Scenario: The cache does not own the revision

- **WHEN** the embedding path's query-side contract changes
- **THEN** the exported revision SHALL change at that source, and the cache SHALL reject
  entries carrying the older revision without defining a parallel revision constant

### Requirement: Stored query representation is a digest plus bounded preview

The similarity layer SHALL store at most a bounded preview of the query plaintext. Each entry
SHALL carry a stable digest of the full query as its identity value and a preview truncated
to a configured maximum as its only human-readable query text, both retained no longer than
the entry's non-sliding maximum lifetime. The digest SHALL replace plaintext in the identity
field, but SHALL NOT be described as anonymization: guessed queries can be hashed and compared.
A query at or below the preview bound may appear in full in the preview, which is a
declared privacy surface of this capability. Matching SHALL use the vector computed from the
full query, never the preview.

#### Scenario: A long query is bounded to the preview

- **WHEN** a query longer than the configured preview bound is stored
- **THEN** its identity value SHALL be a stable digest, its only human-readable query text
  SHALL be the truncated preview. Query metadata SHALL contain no additional query text;
  the answer envelope and embedding remain separately retained data and can themselves
  reveal information about the question

#### Scenario: Preview text does not affect matching

- **WHEN** two distinct full queries share the same truncated preview
- **THEN** their digest values SHALL differ and similarity matching SHALL still operate on
  the distinct full-query embeddings

#### Scenario: Entry expiry removes query representation with the entry

- **WHEN** an entry reaches its maximum lifetime
- **THEN** its digest and preview SHALL expire together with the envelope — no query
  representation outlives its entry

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

### Requirement: Index lifecycle is non-destructive and schema-validated

At process start the system SHALL inspect the semantic index and distinguish a missing index,
an absent search capability, an authorization denial, a connectivity failure, and a schema
mismatch. A missing index SHALL be created non-destructively, without dropping or overwriting
an existing index, and a concurrent already-exists result SHALL be treated as success after
re-inspection. The resulting index SHALL be validated for prefix/storage, vector dimensions,
dtype, distance metric, algorithm, and required and custom attributes before the adapter is
wired. An incompatible existing schema SHALL NOT be overwritten and SHALL disable semantic
lookup and store for the process with an explicit log.

#### Scenario: First boot creates and validates a missing index

- **WHEN** the semantic index does not exist at startup and the search capability is available
- **THEN** the system SHALL create it non-destructively, validate the full schema, and wire
  the semantic adapter

#### Scenario: Concurrent startup does not destroy a peer's index

- **WHEN** multiple workers start and one creates the missing index while another receives an
  already-exists result
- **THEN** the second worker SHALL re-inspect and validate the existing index rather than
  overwrite or drop it

#### Scenario: An incompatible index is never overwritten

- **WHEN** an existing index has a mismatched schema
- **THEN** the system SHALL leave it intact, disable semantic lookup and store for the
  process, and record the incompatibility in logs

#### Scenario: Missing capability is distinguished from a missing index

- **WHEN** the backend cannot serve search queries or the credential is denied
- **THEN** the system SHALL disable semantic lookup and store without attempting destructive
  index changes, and the exact-match cache SHALL be unaffected

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
coverage, warnings, evidence policy), and zero downstream engine-stage spend (retrieval,
grading, generation). Any consumer that derives state from a served envelope — including
agent consumers deriving evidence handles and ledger charges — SHALL derive identical values
whether the hit was exact or semantic.

#### Scenario: An agent consumer derives identical state from an exact or semantic hit

- **WHEN** an agent consumer receives a served envelope via a semantic hit
- **THEN** the derived evidence references, coverage, and downstream stage charges SHALL be
  identical to an exact hit of that envelope, with zero retrieval/grading/generation spend
  on both

#### Scenario: Semantic serving adds no second serving path

- **WHEN** the serving behavior of exact hits and semantic hits is compared
- **THEN** both SHALL flow through the same serving boundary, and no serving logic SHALL
  exist for semantic hits that does not also govern exact hits

### Requirement: Cache hits have explicit budget semantics

A cache-served answer SHALL charge zero retrieval, grading, and generation stage spend to the
run's budget — those stages do not run. Planning charges, including actual retries, and the
upstream intent-scoring charge SHALL remain in force because those calls do run. A cache hit
SHALL NOT be described as zero total spend. A run whose budget is exhausted before the engine
invocation SHALL NOT reach the cache lookup; the exhaustion gate keeps its current position
ahead of any cache read, so cache-as-rescue for exhausted runs is out of scope.

#### Scenario: A cache-served run is charged for planning and upstream work

- **WHEN** a run is served from cache by exact match or similarity
- **THEN** the run's ledger SHALL show zero retrieval/grading/generation stage spend, the
  planning charge including actual retries, and the upstream intent-scoring charge

#### Scenario: An exhausted run cannot read the cache

- **WHEN** a run's budget is exhausted before the engine invocation
- **THEN** the run SHALL be refused as today, and the cache lookup SHALL NOT be consulted

### Requirement: Lookup and store are time-bounded end to end

| Scale | Meter | MUST |
|---|---|---|
| Time from lookup start to lookup outcome | Deployment deadline setting, enforced by the cache adapter, covering query embedding plus backend I/O | 100% of lookups produce a hit, miss, or degradation within the deadline; an overrun is treated as degradation |
| Time from store start to store outcome | Deployment deadline setting, enforced by the cache adapter, covering any embedding plus backend I/O | 100% of stores complete or degrade within the deadline; a store timeout SHALL NOT fail or alter the fresh answer |

#### Scenario: A slow backend degrades rather than stalls the ask

- **WHEN** the similarity lookup, including its query embedding, exceeds its configured
  deadline
- **THEN** the lookup SHALL be abandoned as a degradation and the ask path SHALL produce a
  fresh answer

#### Scenario: A slow store does not fail the fresh answer

- **WHEN** a fresh answer has been produced but the semantic store, including any embedding,
  exceeds its configured deadline
- **THEN** the store SHALL be abandoned as a degradation, the fresh answer SHALL be returned
  unchanged, and the degradation SHALL appear in logs

### Requirement: Observe mode validates the threshold without serving

The system SHALL support an observe mode, selected by deployment configuration, in which
similarity lookups run and record their outcome — including the nearest candidate, its
measured distance, and the would-serve decision — but no semantically matched answer is ever
served and, after an exact-match miss, the fresh downstream pipeline runs. Observe mode SHALL suspend only the serving
behavior of this capability; exact-match serving, recording, degradation, and store behavior
SHALL operate as configured. Moving
from observe mode to serving SHALL require configuration only, never a code change.

#### Scenario: Observe mode records without serving

- **WHEN** observe mode is enabled and a lookup finds a candidate at or below the threshold
- **THEN** the candidate, its distance, and the would-serve decision SHALL be recorded, the
  fresh pipeline SHALL run for the request, and no semantically matched answer SHALL be
  served

#### Scenario: Observe mode is distinguishable from serving in records

- **WHEN** logs are reviewed across a period containing both observe-mode and serve-mode
  lookups
- **THEN** an outcome record SHALL identify whether its lookup would have served or did
  serve, so the two modes cannot be confused in analysis

#### Scenario: Observe mode does not disable the exact-match cache

- **WHEN** semantic mode is `observe` and a request has an exact-match hit
- **THEN** the exact-match cache SHALL serve it as today, and only semantic serving SHALL be
  suspended

### Requirement: Lookups record their measured distance for tuning

Every partition-filtered similarity lookup SHALL record in logs its lookup outcome —
hit-and-served, hit-would-serve (observe mode), miss-with-distance-above-threshold,
miss-with-no-candidate, or degraded — and, when a candidate exists, the measured semantic
distance of the nearest partition-eligible candidate, even when no entry is served, so the
distance distribution per outcome can be reviewed for threshold tuning without reproducing
queries. Records SHALL use bounded linkage (partition/embedding fingerprints and correlation/candidate
identifiers), never query previews, unbounded query text, or answer text. Evaluation content
SHALL be held in a separately access-controlled artifact with an explicit retention bound
and joined by those identifiers; Redis entry expiry does not expire application logs.

#### Scenario: Threshold tuning works from recorded distances

- **WHEN** lookups run across traffic that includes hits, threshold misses, and degraded
  lookups
- **THEN** each lookup's outcome and, only when a candidate exists, its nearest-candidate
  distance SHALL appear in logs, grouped by outcome
