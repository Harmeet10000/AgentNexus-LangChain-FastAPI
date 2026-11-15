# Design: add-semantic-answer-cache

## Context

The retrieval graph's query-analyzer node already carries the exact-match answer cache: it
derives a version-aware key, reads `CachedAskEnvelope` on the plan path, and the finalisation
path writes the envelope back. Both consumers of the compiled graph — the documents ask query
(`features/documents/application/queries/documents.py`, mount-retrieval-graph 4.x) and Agent
Saul's qna engine (`features/agent_saul/graph/factory.py`) — receive cache behavior without
knowing about it. Redis is Redis Cloud (Search & Query capability confirmed enabled), reached
through the process-lifetime client built in `connections/redis.py` and read from
`app.state`. Embeddings flow through one path (`src/app/ai/embeddings.py`, Gemini,
`EMBEDDING_DIMENSION` = 768), which already versions its own embedding cache namespace
(`_CACHE_NAMESPACE = "embedding:v2"`) because query-side instructions can change the vector
contract without changing model or width. Shared cache helpers with the
`CacheError`/`CacheResult` union live in `src/app/utils/cache/`. See proposal.md — Why for
motivation.

**Stated assumption (verified 2026-11-15):** every cache layer on the ask path — exact-match
and semantic — treats the graph-entry query as the *complete semantic input to the answer*.
Verified at both entry points: the direct ask API is single-turn by contract
(`UnifiedAskRequest.query: str`, dto.py:331), and Agent Saul's qna engine passes the
transcript's latest user turn (`_qna_engine_query`, nodes.py:692) into a boundary that is
query-only by construction (`_retrieval_state_from_agent` sets `messages: []`) — so the
engine's answer is a function of (query, partition) alone, and both caches preserve that
function rather than alter it. Elliptical follow-ups are answered antecedent-blindly by fresh
runs already; the caches make that consistent, not wrong. Task 4.3 pins this with a test so
the assumption cannot silently break.

## Goals / Non-Goals

**Goals:**
- A swappable semantic-cache backend port with the RedisVL adapter shipped now, LangCache
  anticipated behind the same port.
- Lookup on exact-match miss, store on fresh answer, both inside the graph's cache seam —
  consumers untouched.
- Authorization partition enforced structurally (equality digest), similarity applied only
  inside a partition.
- Embedding-identity (model + dimension + contract revision) enforced as part of the same
  lookup predicate, so model/instruction changes cannot silently compare incompatible spaces.
- Time-bounded lookup and store that include the embedding call, not just Redis I/O.
- Degradation identical in shape to the existing cache degradation discipline.

**Non-Goals:**
- Tenant-level or cross-user matching (explicitly excluded; config-flag expansion is a
  future change).
- Caching retrieval *results* (`SearchResponse` hybrid cache) or any LLM call other than the
  grounded answer path (planner/orchestrator/entity/finalization calls stay uncached).
- Sharing embeddings between the cache lookup and hybrid retrieval (different text: original
  vs plan-rewritten query; one extra embedding per exact-match miss is the accepted cost).
- LangCache adapter implementation in this change — port only. RedisVL is the only
  selectable backend; a future LangCache adapter must be added before its value is accepted
  by settings, never silently selected.

## Decisions

### D1: Backend port + RedisVL adapter, LangCache deferred
**Choose:** a thin async port (protocol with lookup/store/close) in `src/app/utils/cache/`,
implemented by a RedisVL `SemanticCache` adapter. `SEMANTIC_CACHE_BACKEND` accepts only
`redisvl` in this change; an unsupported value fails settings/startup validation loudly
instead of falling back. The LangCache value is added only with its adapter.
**Why:** RedisVL runs on the Redis database already in the stack, and its custom vectorizer
lets the cache embed through `src/app/ai/embeddings.py` — keeping the `unified-embedding`
one-path contract intact. LangCache embeds server-side with its own model, which would create
a second embedding path and couple the ask path to an additional SaaS auth surface. The port
is two methods; a LangCache adapter is a bounded future change, not a parallel
implementation now. Alternatives: LangCache now (rejected per answered backend decision and
unified-embedding); in-house vector search over Redis sets (rejected — reimplements what
redisvl provides for free).

### D2: The partition is a single equality digest reused from the key derivation
**Choose:** compute a `partition_digest` — SHA-256 over the same sorted-key bytes the
exact-match derivation uses, minus the query — and store it as the cache entry's tag
attribute; the lookup applies exactly one tag filter on it, over a per-user digest.
**Why:** the exact-match key derivation is already the reviewed, parity-pinned encoding of
user, tenant, scope, resolved versions, and policy. Reusing it minus the query means there is
exactly one definition of the authorization partition in the codebase, and the modified
`legal-corpus-retrieval` requirement is satisfied structurally rather than by a parallel
metadata schema drifting out of sync. Alternatives: individual tag filters per partition
member (more moving parts, same guarantee, drift risk between the two encodings); embedding
similarity with post-filters (unsafe — similarity ranking would be computed across
partitions before filtering, and metadata post-filtering after vector ranking is exactly the
defect class the performance doc documented for this repo).

### D3: Threshold, TTL, and deadlines are settings with conservative defaults
**Choose:** `SEMANTIC_CACHE_DISTANCE_THRESHOLD` default `0.05` (cosine distance; a **policy
default, not a proven performance fact** — task 4.4 measures it), TTL default `86400`s (24h,
decoupled from the exact-match cache's 900s), query-embedding character bound `8000`, lookup
deadline `0.5s`, store deadline `0.5s`, backend setting accepting only `redisvl`, and a mode
enum `off | observe | serve` defaulting
to `observe` — all in `config/settings.py`.
**Why:** a semantic hit serves an answer generated from a specific retrieved context; the
partition digest pins scope and versions, so the residual risk is wording drift, but legal
answers carry citations and are load-bearing — tight beats generous. The video's 0.1
illustration is a starting point for traffic tuning, not a default for this domain. Both
deadlines reuse the existing bounded-deadline discipline and cover the whole operation: lookup
includes the query embedding plus Redis I/O; store includes any embedding plus Redis I/O. The
TTL is decoupled deliberately: scope/version and embedding-identity checks exclude known
incompatible entries. These checks depend on correct version/revision maintenance and do
not prove answer freshness against every external change; the 24h TTL remains a policy bound. The lifetime is **non-sliding**: reads, replays, and observe checks
never extend it (D5/D11).

### D4: Index lifecycle at startup, degrade at runtime
**Choose:** at lifespan, inspect the module's search index and distinguish five outcomes:
missing index, unsupported Search capability, ACL denial, connectivity failure, and schema
mismatch. Missing index is created non-destructively (`overwrite=False`, never
`create-overwrite`), then re-inspected to tolerate a concurrent creator; the resulting schema
is validated (index prefix/storage, vector dimensions/dtype/distance/algorithm, required and
custom fields). Only then is the adapter wired. An incompatible existing schema is never
overwritten and disables semantic lookup/store with an explicit log (exact-match cache
unaffected). Runtime lookup/store failures or deadline overruns degrade per-call, logged.
**Why:** matches the standing degradation discipline in the graph ("a dead cache must never
fail a search") and the Redis Cloud plan reality. A missing index is not an absent capability
— it is the normal first boot — while silently dropping and recreating an incompatible index
would destroy data. Alternatives: hard dependency (rejected — cache is an accelerator); lazy
per-call probing (rejected — probe cost on every ask); blind overwrite at boot (rejected —
destructive on schema drift and racing workers).

### D5: Stock RedisVL schema, explicit vectors, and a non-sliding TTL
**Choose:** use RedisVL's stock `SemanticCacheIndexSchema` — FLAT cosine FLOAT32 vector over
`EMBEDDING_DIMENSION`, with its standard `prompt`, `response`, `prompt_vector`,
`inserted_at`, and `updated_at` fields — plus custom filterable fields: `partition` (tag),
`embedding_model` (tag), `embedding_dimension` (numeric), and `embedding_revision` (tag).
Generation ID and absolute expiry are metadata. A bounded `query_preview` also lives in
metadata, not in `filters`: preview edits must not change the stock prompt-plus-filters ID.
The index name is dimension-scoped (e.g.
`ask-answer-semantic-cache-768`). `prompt` is a stable digest of the **full** normalized query
using the same normalization as `_answer_cache_key` (not a truncated prefix), `query_preview`
is truncated text for debuggability only, and
`response` is `CachedAskEnvelope.model_dump_json()`. Vectors are always supplied explicitly
from the unified embedder (`vector=`); a thin vectorizer delegate guards the one embedding
path and must never choose a model or embed on its own at lookup time. Lookup filters only
partition and embedding identity, never full-query equality (that would collapse similarity
to exact matching). The stock cache is constructed with `ttl=None`, `refresh_ttl_on_hit=False`, and every
lookup uses `refresh_ttl=False`, including observe. Do not use stock `astore` followed by
an expiry command: cancellation between those calls can leave persistent content. Use
RedisVL's stock entry serialization and deterministic key derivation, but publish the hash
fields and absolute `PEXPIREAT` expiry in one bounded, single-key Lua operation through
the existing Redis client. Persist a generation ID, generation completion time, and
`expires_at` before the write attempt; resume/retry must reuse them. The script checks
Redis server time, rejects expired generations before mutation, rejects an older generation
replacing a newer one, and preserves the earlier of any existing finite expiry and the
incoming deadline. It validates input/type before mutation and handles command errors:
Redis script atomicity prevents interleaving but does not roll back earlier writes on an
error. A failed publication must leave no new persistent value. Exercise failure paths
against real Redis; no grace-period cleanup is part of the correctness argument.
A genuinely new generation can create an entry after an old key expires; a delayed replay
cannot obtain a new lifetime by recreating its absent key. Unknown/missing generation or
expiry metadata fails closed. Lookup validates both the absolute expiry and a positive
remaining TTL. It never performs a delayed unconditional delete that could remove a newer
writer's entry. A lookup calls the stock check with maximum cosine distance (`2.0`) and
applies the configured threshold in the adapter so nearest-miss telemetry remains available.
The adapter injects the process's async Redis client into the stock
cache (0.32's constructor exposes only a sync `redis_client`), so cache I/O uses the same
pool, timeout, retry, and keepalive settings instead of opening a second connection pool. The
adapter marks the cache non-owning and its `close` is a no-op: lifespan owns shutdown.
Because 0.32 exposes no public async-client seam, pin the dependency below 0.33 and keep an
adapter test on the private seam; revisit the pin when the library exposes a public async
injection point. Startup verifies the script and expiry permissions as well as Search;
unsupported or denied atomic publication disables the layer. Integration tests target the
deployed Redis version (production Terraform pins 7.2).
The persistence exception is adapter-local: RedisVL still owns schema, serialization and
vector queries; the script owns atomic publication and absolute expiry only. See
[Redis scripting semantics](https://redis.io/docs/latest/develop/programmability/eval-intro/)
and [PEXPIREAT](https://redis.io/docs/latest/commands/pexpireat/).
**Why:** the envelope — not the bare answer — is what a hit must serve, matching the existing
exact-match contract ("the cache stores the envelope so a hit serves the same coverage,
warnings, citations, and evidence policy the miss served"). RedisVL's stock schema is FLAT
by design for this cache API; assuming HNSW (`query_vector`) or non-standard field names
would not match the supported schema. A full-query digest avoids stock RedisVL entry IDs
colliding on shared prefixes, while the preview keeps diagnostics. A non-sliding lifetime
means a frequently hit entry cannot live forever and deleted source text ages out on a fixed
horizon.

### D6: Provenance is an additive optional field
**Choose:** the ask response gains an optional `cache_provenance` field (`"exact" |
"semantic"`), absent when the answer is fresh; the graph state carries it from the cache
read to the final response, and logs record provenance per served response.
**Why:** the spec requires three distinguishable modes; an optional additive field keeps
every existing consumer byte-compatible (the response shape change is strictly
additive). Log provenance also gives the hit-rate signal needed to tune the threshold.

### D7: Store path is the exact-match eligibility gate, plus a quality predicate
**Choose:** semantic store happens exactly where `_cache_ask_envelope` runs, gated by the
same conditions (no bypass, cache configured, mode not `off`), plus one additional predicate:
degraded envelopes — the fallback output the grade-cap escape and generation-budget refusal
write today (verified at nodes.py:1270–1289) — are excluded from similarity storage while
staying exact-match-eligible. Unclassifiable output is excluded (fail-closed).
**Why:** today a transient generator failure caches the fallback for 15 minutes to
byte-identical queries only; the semantic layer would spread that "uncertain" envelope to
every paraphrase for 24 hours. The quality predicate is fail-closed (unclassifiable =
degraded). One gate, two layers, no drift; bypassing operators get identical semantics on
both layers (bypass flag verified at documents.py:596/620/880).

### D8: Vector-space identity includes a public embedding-contract revision
**Choose:** the match predicate is partition-digest equality AND embedding-identity equality
over `(model id, output dimension, embedding-contract revision)`, each stored as its own
filterable attribute. The revision is exported by `src/app/ai/embeddings.py` — the module
already versions its internal embedding cache namespace (`embedding:v2`) for exactly this
reason. The semantic adapter reads that public revision; it does not define its own.
**Why:** the digest is the *authorization* boundary (user, tenant, scope, versions, policy);
embedding identity is a *match-validity* pre-condition — the two are conceptually distinct,
and merging them buries a model/instruction swap inside a hash where it cannot be inspected.
Model and dimension are insufficient: the module's query/document instruction prefixes mean
the same model and width can produce a different vector contract when the query-side template
changes. Without the revision, old and new entries would be compared as though they shared a
space. The dimension-suffixed index name makes a dimension change create a fresh index
naturally; the revision filter makes an instruction change produce clean misses even with an
identical model and width, and old-revision entries stay unreachable until TTL. Task type
needs no filter: both lookup and stored questions are query-side embeddings by construction.
Alternatives: model + dimension only (rejected — silent cross-contract comparisons); fold
into digest (rejected — conflates concerns, ops-opaque).

### D9: Distance telemetry is part of the lookup contract
**Choose:** every partition-filtered lookup logs its outcome (hit-and-served / hit-would-serve
in observe mode / miss-with-distance-above-threshold / miss-with-no-candidate / degraded) and,
when a candidate exists, the nearest partition-eligible candidate's measured distance.
Outcome is explicit, never inferred from the absence of a distance; observe-mode hits use
`hit-would-serve`, never `hit-and-served`. The lookup call carries serve/observe solely so
the adapter can label the outcome; an observe hit returns no servable match.
**Why:** the threshold is a tuning knob, and a knob without a signal is a guess — this is the
difference between tuning from a week of distance distributions and tuning from vibes. Costs
one log line per lookup, which the lookup path already pays for provenance.

### D10: Rollout runs through observe mode; serving is a data-informed flip
**Choose:** a three-value mode (`off | observe | serve`, default `observe`) over the binary
enable flag, where observe computes and records the would-serve decision after an exact miss and runs
the fresh downstream pipeline. Exact hits retain their current behavior.
**Why:** the threshold is a *probabilistic* match serving *legal answers* — the one feature
class where "enable and watch" transfers validation risk to users. Observe mode validates the
threshold against real distance distributions and real would-serve candidates with zero
user-facing risk, and the serve flip is then a config change, not a deploy. The default being
`observe` (not `off`) means the feature accumulates its own evidence from the first deploy.
Observe is not acceptance by itself: serving additionally requires D3's threshold to be
backed by the labeled safe-substitution evaluation (task 4.4), not just a plausible-looking
distance histogram.
Alternatives: binary flag (rejected — first serve happens before any evidence); canary
traffic splitting (rejected — heavier machinery for the same signal).

### D11: Stored query text is a bounded preview; the entry identity is a digest
**Choose:** the stock `prompt` field holds a stable full-query digest (hash), not plaintext;
a `query_preview` metadata value holds a prefix bounded by a configured maximum (default 64
characters, capped) for diagnostics. Matching always uses the vector computed from the full
query. The generation's absolute expiry is fixed before publication, and no read refreshes it. Declare
the preview surface in the capability spec.
**Why:** this is a *new* exposure, not an inherited one — the exact-match key is
`sha256(sorted bytes)` (verified at nodes.py:1644) and the envelope carries no query field,
so today no query plaintext exists in Redis in recoverable form. The semantic layer must not
reintroduce unbounded full text as a side effect of RedisVL's entry-ID derivation: a shared
truncated prefix could also collide across distinct questions. A digest avoids prefix collisions; it is not anonymization and permits dictionary guessing.
The short preview retains the debuggability that makes misses answerable. Queries at or below the preview bound may appear in full in that preview — an
explicitly declared, bounded surface, not an accident. A non-sliding lifetime ensures
deleted or sensitive source text does not persist merely because the entry is popular.
Alternatives: full plaintext prompt (rejected — unbounded surface); omit
all text (rejected — loses diagnostics); sliding TTL (rejected — hot entries never expire).

### D12: Budget semantics are status quo, documented precisely
**Choose:** no ordering change — the exhaustion gate stays ahead of the cache read. Planning
charges (including actual retries) and the upstream intent-scoring charge stay in force. A
cache hit reports zero **downstream** stage spend only: retrieval, grading, and generation do
not run. It does not report zero total spend, and it does not apply to budget-exhausted runs,
which never consult the cache.
**Why:** a budget-exhausted run reaching a free cached answer ("cache-as-rescue") is an
attractive but scope-widening UX change to the ledger flow. The planner runs before the
cache read today (verified at nodes.py:352–361/384–408), so claiming "zero stage spend" for
all stages is both wrong and would encourage moving the cache ahead of planning. Documenting
the exact ordering prevents a future reader from assuming free-ness extends to exhausted
runs or to the planning/upstream charges.

## Risks / Trade-offs

- [Threshold too loose serves a paraphrase whose retrieved context differed in emphasis] →
  partition digest pins scope/versions/policy; 0.05 is a conservative **policy default**, not
  a proven bound; provenance in logs makes semantic hits auditable; task 4.4 must show a
  one-sided 95% upper bound ≤ 1% on unsafe substitutions among held-out would-serve
  decisions before serve. This is an initial risk-policy choice, not a correctness proof. Distance histograms alone are not acceptance.
- [Embedding-model or query-instruction change shifts vector space so stored vectors stop
  meaning anything] → structurally prevented by D8: model id, dimension, and public contract
  revision are match-predicate components, and the index is dimension-scoped — a change
  produces clean misses, not noisy matching, for as long as old entries live.
- [Index memory grows unbounded on the 24h TTL and the Redis plan has no headroom] →
  arithmetic first: ~768-dim FLOAT32 (3KB) + envelope JSON + metadata ≈ 6–10KB/entry (FLAT
  adds no graph overhead); at an assumed 50k lookups/day × ~30% store rate the steady state
  is ~15–20k entries ≈ 100–200MB. Startup validation logs the Redis Cloud plan's cap and
  `maxmemory-policy` (`noeviction` means store failures surface as silent degradation, which
  the probe must call out); the observe window measures real accumulation before serve.
- [Serve mode goes live with an unvalidated threshold] → structurally prevented by D10:
  the default mode is observe; serving is gated by the migration plan's evaluation rule on
  labeled data, not by elapsed observation time.

**Named and accepted** (one-line dispositions, no artifacts):

- LangGraph interrupt/replay re-executes nodes: store is idempotent (same entry ID replaces
  its value); a replayed lookup may see entries stored between runs — bounded, benign, and
  lookup determinism across replays is not asserted. Replay reuses the persisted generation deadline, including after key expiry; it never
  creates a new retention horizon.
- Boot stampede: N workers concurrently create the same missing index — `overwrite=False`,
  tolerate the already-exists race, re-inspect, then validate the resulting schema. No
  destructive recreate.
- Bulk re-ingest orphans every entry (all version digests change at once) — bounded 24h dead
  weight; no pattern-flush machinery.
- TTL layering blur: exact 900s vs semantic 24h means byte-identical repeats after 15 minutes
  arrive via semantic provenance — hit-rate dashboards must count both layers as hits.
- Oversized queries past the embedding model's token limit truncate silently → garbage
  distance: a conservative character bound (`8000`) rejects before embed and degrades to
  exact-match-only; it is deliberately a pre-tokenizer guard, not an exact token count.
- Task-type trap: stored questions and lookups must both use the query-side task type — the
  vectorizer's silent default is the failure mode (pinned by task 2.2).
- [One extra embedding call per exact-match miss adds latency/cost] → bounded by the lookup
  deadline; degradation on overrun; explicitly a non-goal to share embeddings with hybrid
  retrieval (different query text).
- [redisvl typing may be incomplete for `ty check`] → adapter isolates redisvl behind the
  port; any needed ignores follow the `noqa-documentation` spec conventions.
- [Partition proliferation (user × scope × version) grows the index] → entries are small
  envelopes; TTL bounds growth; measure filtered FLAT latency at observed partition sizes; no asymptotic speedup is assumed.
- Default mode is `observe`; `off` is the production kill switch and performs no semantic
  lookup or store. Exact-match behavior remains independent.

## Migration Plan

No data migration. Deploy order, gated by measured data:

1. Build the labeled safe-substitution set and calibration harness (task 4.4). Deploy with
   mode `observe`; the default threshold stays `0.05` as a policy starting point, not a
   proven value.
2. Run observe for **≥ 14 days** across staging and production. Startup validation confirms
   memory headroom and `maxmemory-policy`; the observe window measures real accumulation and
   real distance distributions. Observe logs only correlation/candidate IDs and fingerprints,
   never previews or answer text. Consented evaluation samples live in an access-controlled
   artifact with an explicit retention deadline; IDs join the request, candidate envelope
   and evidence, and fresh answer without extending Redis query-text retention through logs.
3. **Decision rule (the success criterion):** enable `serve` only if —
   (a) projected hit rate at the selected threshold is ≥ 5% of exact-match misses;
   (b) on the held-out, query-family-disjoint labeled safe-substitution evaluation, the
   one-sided 95% upper confidence bound on unsafe substitutions among would-serve decisions
   is ≤ 1% — with zero errors this requires **≥ 299 independent would-serve cases**
   representative of the intended workload (20 only bounds error near 14%). Report known-
   negative false positives separately; they cannot supply this denominator;
   (c) adversarial cases (negation, party swap, jurisdiction, multilingual paraphrase) are
   represented, with zero unsafe substitutions in the must-not-serve cases; safe multilingual
   paraphrases remain eligible;
   (d) the gated real-Redis integration test has passed in a Search-capable environment;
   (e) memory headroom holds at observed accumulation.
   If the rule fails: revise on calibration data, collect new held-out evidence after any
   threshold/embedding-contract change, or set mode `off` and record why — the feature must earn its serving, never assume it.
4. Rollback: mode `off` — semantic serving and writes stop while exact-cache behavior remains; the index
   and entries are inert.

Steps 2–3 are operational gates *after* the implementation tasks complete — they belong to
the operator runbook, not to tasks.md, so archiving is not blocked on a 14-day wait. The
evaluation report is a deliverable, not a substitute for the operational window.

## Open Questions

None — backend choice, scoping, and Redis capability were resolved with the user before
specs; the remaining unknowns (threshold tuning, LangCache timing) are deferrable without
touching specs, approach, or task order.

## Blast Radius

| Symbol/module | Callers | Dependents |
|---|---|---|
| Query-analyzer node cache seam (`features/documents/retrieval/graph/nodes.py`) | compiled retrieval graph; graph built in `lifecycle/graphs.py` | documents ask query; agent_saul qna engine (`factory.py`, `graph.py`) |
| `_answer_cache_key` derivation | query-analyzer node (read), finalisation write | semantic `partition_digest` (reuses bytes, minus query) |
| `_cache_ask_envelope` write path | finalisation node | semantic store (same gate plus quality/mode checks) |
| `CachedAskEnvelope` model | cache read/write nodes, ask response building | serialized into semantic entries; `cache_provenance` is additive |
| `src/app/ai/embeddings.py` client | hybrid retrieval, ingestion, memory paths | gains a cache-vectorizer consumer and a public contract-revision export; no embedding behavior change |
| `src/app/utils/cache/` shared union | existing Redis ops, rate limits | gains semantic adapter; no change to existing members |
| `lifecycle/lifespan.py` / `connections/redis.py` | app startup | inspects, creates-if-missing, validates the semantic index; wires the disable flag |
| `config/settings.py` | everywhere | additive settings only; unsupported backend fails validation |

## Async Semantics

No async interfaces in this change — the lookup and store are in-request reads/writes over
the existing Redis client, and expiry is Redis key TTL with a non-sliding maximum lifetime.
Nothing feeds contracts/asyncapi.yaml.
