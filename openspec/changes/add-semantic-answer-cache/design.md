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
`EMBEDDING_DIMENSION` = 768). Shared cache helpers with the `CacheError`/`CacheResult` union
live in `src/app/utils/cache/`. See proposal.md — Why for motivation.

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
  adapter anticipated behind the same port.
- Lookup on exact-match miss, store on fresh answer, both inside the graph's cache seam —
  consumers untouched.
- Authorization partition enforced structurally (equality digest), similarity applied only
  inside a partition.
- Degradation identical in shape to the existing cache degradation discipline.

**Non-Goals:**
- Tenant-level or cross-user matching (explicitly excluded; config-flag expansion is a
  future change).
- Caching retrieval *results* (`SearchResponse` hybrid cache) or any LLM call other than the
  grounded answer path (planner/orchestrator/entity/finalization calls stay uncached).
- Sharing embeddings between the cache lookup and hybrid retrieval (different text: original
  vs plan-rewritten query; one extra embedding per exact-match miss is the accepted cost).
- LangCache adapter implementation in this change — port + config selection only.

## Decisions

### D1: Backend port + RedisVL adapter, LangCache deferred
**Choose:** a thin async port (protocol with lookup/store/close) in `src/app/utils/cache/`,
implemented by a RedisVL `SemanticCache` adapter, selected by a settings enum, over adopting
LangCache directly.
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

### D3: Threshold, TTL, and deadline are settings with conservative defaults
**Choose:** `SEMANTIC_CACHE_DISTANCE_THRESHOLD` default `0.05` (cosine distance; task 4.4
replaces this asserted default with a derived one before serve mode), TTL
default `86400`s (24h) — decoupled from the exact-match cache's 900s — lookup deadline default
`0.5s`, backend enum default `redisvl`, and a mode enum `off | observe | serve` defaulting to
`observe` — all in `config/settings.py`.
**Why:** a semantic hit serves an answer generated from a specific retrieved context; the
partition digest pins scope and versions, so the residual risk is wording drift, but legal
answers carry citations and are load-bearing — tight beats generous. The video's 0.1
illustration is a starting point for traffic tuning, not a default for this domain. The
deadline reuses the existing bounded-deadline discipline around Redis calls in the graph. The
TTL is decoupled deliberately: with D8 in place, every staleness vector (document versions,
pipeline version, embedding model, dimension) is structurally prevented from matching, so the
exact-match 900s — which is hygiene, not correctness — stops being the right horizon, and a
cached answer is exactly as valid as its exact-match counterpart for as long as the entry
lives.

### D4: Capability probe at startup, degrade at runtime
**Choose:** at lifespan, probe the search capability (`FT.INFO` against the cache index);
absent → a process-lifetime flag disables semantic lookup and store (exact-match cache
unaffected). Runtime lookup/store failures or deadline overruns degrade per-call, logged.
**Why:** matches the standing degradation discipline in the graph ("a dead cache must never
fail a search") and the Redis Cloud plan reality — the capability probe turns an
environment regression into a logged disable rather than a red ask path. Alternatives:
hard dependency (rejected — cache is an accelerator); lazy per-call probing (rejected —
probe cost on every ask).

### D5: Index schema and entry shape
**Choose:** one RedisVL schema-backed index for the whole process, named with the configured
dimension (e.g. `ask-answer-semantic-cache-768`): `query_text` (text), `query_vector` (HNSW,
cosine, FLOAT32, width = `EMBEDDING_DIMENSION`), `partition` (tag), `embedding_identity` (tag
— model id + dimension), `query_text` (text — truncated prefix per D11), `envelope` (text —
serialized `CachedAskEnvelope`). Entries keyed
under a dedicated prefix, TTL applied at store.
**Why:** the envelope — not the bare answer — is what a hit must serve, matching the
existing exact-match contract ("the cache stores the envelope so a hit serves the same
coverage, warnings, citations, and evidence policy the miss served"). HNSW+cosine is the
production default for text embeddings at any realistic corpus scale.

### D6: Provenance is an additive optional field
**Choose:** the ask response gains an optional `cache_provenance` field (`"exact" |
"semantic"`), absent when the answer is fresh; the graph state carries it from the cache
read to the final response, and logs record provenance per served response.
**Why:** the spec requires three distinguishable modes; an optional additive field keeps
every existing consumer byte-compatible (the response shape change is strictly
additive). Log provenance also gives the hit-rate signal needed to tune the threshold.

### D7: Store path is the exact-match eligibility gate, plus a quality predicate
**Choose:** semantic store happens exactly where `_cache_ask_envelope` runs, gated by the
same conditions (no bypass, cache configured), plus one additional predicate: degraded
envelopes — the fallback output the grade-cap escape and generation-budget refusal write
today (verified at nodes.py:1270–1289) — are excluded from similarity storage while staying
exact-match-eligible.
**Why:** today a transient generator failure caches the fallback for 15 minutes to
byte-identical queries only; the semantic layer would spread that "uncertain" envelope to
every paraphrase for 24 hours. The quality predicate is fail-closed (unclassifiable =
degraded). One gate, two layers, no drift; bypassing operators get identical semantics on
both layers (bypass flag verified at documents.py:596/620/880).

### D8: Vector-space identity is a separate match-predicate component
**Choose:** the match predicate is partition-digest equality AND embedding-identity equality
(model id + output dimension, both stored as their own tag attribute), with the index named
per dimension, over folding the embedding model into the partition digest.
**Why:** the digest is the *authorization* boundary (user, tenant, scope, versions, policy);
embedding identity is a *match-validity* pre-condition — the two are conceptually distinct,
and merging them buries a model swap inside a hash where it can't be inspected or flushed.
Cross-space distance is meaningless noise in both directions (false matches and false
misses), so the constraint must sit with the filter evaluation, before thresholding. This
matches the repo's own precedent — the embedding cache keys are already
task-type/model/dimension-aware — and the dimension-suffixed index name makes a dimension
change create a fresh index naturally instead of erroring on width mismatch. Task type needs
no filter: both lookup and stored questions are query-side embeddings by construction.
Alternatives: fold into digest (rejected — conflates concerns, ops-opaque); ignore (rejected
— a real noise window on every model swap, however TTL-bounded).

### D9: Distance telemetry is part of the lookup contract
**Choose:** every partition-filtered lookup logs its nearest partition-eligible candidate's
measured distance plus the outcome (served / below-threshold miss / degraded), hit or miss.
**Why:** the threshold is a tuning knob, and a knob without a signal is a guess — this is the
difference between tuning from a week of distance distributions and tuning from vibes. Costs
one log line per lookup, which the lookup path already pays for provenance.

### D10: Rollout runs through observe mode; serving is a data-informed flip
**Choose:** a three-value mode (`off | observe | serve`, default `observe`) over the binary
enable flag, where observe computes and records the would-serve decision but always runs the
fresh pipeline.
**Why:** the threshold is a *probabilistic* match serving *legal answers* — the one feature
class where "enable and watch" transfers validation risk to users. Observe mode validates the
threshold against real distance distributions and real would-serve candidates with zero
user-facing risk, and the serve flip is then a config change, not a deploy. The default being
`observe` (not `off`) means the feature accumulates its own evidence from the first deploy.
Alternatives: binary flag (rejected — first serve happens before any evidence); canary
traffic splitting (rejected — heavier machinery for the same signal).

### D11: Stored query text is a declared privacy surface, truncated by default
**Choose:** store the query text truncated (bounded prefix) as the searchable text field,
with the matching vector always computed from the full query; declare the surface in the
capability spec.
**Why:** this is a *new* exposure, not an inherited one — the exact-match key is
`sha256(sorted bytes)` (verified at nodes.py:1644) and the envelope carries no query field,
so today no query plaintext exists in Redis in recoverable form. The semantic layer makes
every legal question FT-searchable for the entry's lifetime. Truncation narrows the
searchable surface without touching match quality (the vector is the matching key; the text
is metadata). Alternatives: omit text entirely (rejected — loses the debuggability that
makes misses answerable); verbatim storage (rejected — strictly wider surface for zero
matching benefit).

### D12: Budget semantics are status quo, documented
**Choose:** no behavior change — the exhaustion gate stays ahead of the cache read, cache
hits report zero stage spend, and the upstream intent-scoring call stays charged.
**Why:** a budget-exhausted run reaching a free cached answer ("cache-as-rescue") is a
an attractive but scope-widening UX change to the ledger flow; documenting the current ordering
prevents a future reader from assuming the free-ness extends to exhausted runs.

## Risks / Trade-offs

- [Threshold too loose serves a paraphrase whose retrieved context differed in emphasis] →
  partition digest pins scope/versions/policy; default 0.05 is tight; provenance in logs
  makes semantic hits auditable; threshold is a runtime setting tuned from D9's recorded
  distance distributions, not vibes.
- [Embedding-model change shifts vector space so stored vectors stop meaning anything] →
  structurally prevented by D8: model id + dimension are match-predicate components and the
  index is dimension-scoped — a swap produces clean misses, not noisy matching, for as long
  as old entries live.
- [Index memory grows unbounded on the 24h TTL and the Redis plan has no headroom] →
  arithmetic first: ~768-dim FLOAT32 (3KB) + HNSW graph overhead (~1.5–2×) + envelope JSON
  ≈ 8–12KB/entry; at an assumed 50k lookups/day × ~30% store rate the steady state is
  ~15–20k entries ≈ 150–250MB — task 3.1 verifies the Redis Cloud plan's cap and logs
  `maxmemory-policy` (`noeviction` means store failures surface as silent degradation, which
  the probe must call out), and the observe window measures real accumulation before serve.
- [Serve mode goes live with an unvalidated threshold] → structurally prevented by D10:
  the default mode is observe; serving is gated by the migration plan's decision rule on
  measured data.

**Named and accepted** (one-line dispositions, no artifacts):

- LangGraph interrupt/replay re-executes nodes: store is idempotent (overwrite); a replayed
  lookup may see entries stored between runs — bounded, benign, and lookup determinism across
  replays is not asserted.
- Boot stampede: N workers concurrently create the same index — create-overwrite, tolerate
  the race error.
- Bulk re-ingest orphans every entry (all version digests change at once) — bounded 24h dead
  weight; no pattern-flush machinery.
- TTL layering blur: exact 900s vs semantic 24h means byte-identical repeats after 15 minutes
  arrive via semantic provenance — hit-rate dashboards must count both layers as hits.
- Oversized queries past the embedding model's token limit truncate silently → garbage
  distance: cap query length before embed and degrade to exact-match-only for oversized ones.
- Task-type trap: stored questions and lookups must both use the query-side task type — the
  vectorizer's silent default is the failure mode (pinned by task 2.2).
- [One extra embedding call per exact-match miss adds latency/cost] → bounded by the lookup
  deadline; degradation on overrun; explicitly a non-goal to share embeddings with hybrid
  retrieval (different query text).
- [redisvl typing may be incomplete for `ty check`] → adapter isolates redisvl behind the
  port; any needed ignores follow the `noqa-documentation` spec conventions.
- [Partition proliferation (user × scope × version) grows the index] → entries are small
  envelopes; TTL bounds growth; single-tag filter keeps lookup O(rank-within-partition).
- [Feature flag off by default hides regressions] → default on in dev/staging; the enable
  flag exists for production kill-switch, matching the exact-match cache's bypass discipline.

## Migration Plan

No data migration. Deploy order, gated by measured data:

1. Deploy with mode `observe` + task 4.4's calibration eval sets the derived threshold
   default (the asserted 0.05 is a placeholder until then).
2. Run observe for **≥ 14 days** across staging and production. Task 3.1's probe confirms
   memory headroom and `maxmemory-policy`; the observe window measures real accumulation and
   real distance distributions.
3. **Decision rule (the success criterion):** enable `serve` only if — (a) projected hit
   rate at the calibrated threshold is ≥ 5% of exact-match misses, (b) a manual sample of
   ≥ 20 would-serve candidates reads as answers the fresh run would plausibly have produced,
   and (c) memory headroom holds at observed accumulation. If the rule fails: keep observing
   with an adjusted threshold, or set mode `off` and record why — the feature must earn its
   serving, never assume it.
4. Rollback: mode `off` — the ask path returns byte-for-byte to current behavior; the index
   and entries are inert.

Steps 2–3 are operational gates *after* the implementation tasks complete — they belong to
the operator runbook, not to tasks.md, so archiving is not blocked on a 14-day wait.

## Open Questions

None — backend choice, scoping, and Redis capability were resolved with the user before
specs; the remaining unknowns (threshold tuning, LangCache timing) are deferrable without
touching specs, approach, or task order.

## Blast Radius

| Symbol/module | Callers | Dependents |
|---|---|---|
| Query-analyzer node cache seam (`features/documents/retrieval/graph/nodes.py`) | compiled retrieval graph; graph built in `lifecycle/graphs.py` | documents ask query; agent_saul qna engine (`factory.py`, `graph.py`) |
| `_answer_cache_key` derivation | query-analyzer node (read), finalisation write | semantic `partition_digest` (reuses bytes, minus query) |
| `_cache_ask_envelope` write path | finalisation node | semantic store (same gate) |
| `CachedAskEnvelope` model | cache read/write nodes, ask response building | unchanged — serialized verbatim into semantic entries |
| `src/app/ai/embeddings.py` client | hybrid retrieval, ingestion, memory paths | gains a cache-vectorizer consumer; no behavior change |
| `src/app/utils/cache/` shared union | existing Redis ops, rate limits | gains semantic adapter; no change to existing members |
| `config/settings.py` | everywhere | additive settings only |

## Async Semantics

No async interfaces in this change — the lookup and store are in-request reads/writes over
the existing Redis client, and expiry is the existing TTL discipline. Nothing feeds
contracts/asyncapi.yaml.
