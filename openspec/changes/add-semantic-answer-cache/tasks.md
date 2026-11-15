# Tasks: add-semantic-answer-cache

Implements the `semantic-answer-cache` capability spec and the modified
`legal-corpus-retrieval` "Answer cache keys are version-aware" requirement. Design decisions:
D1 port+adapter, D2 partition digest, D3 settings/TTL/deadlines, D4 index lifecycle, D5
stock schema, D6 provenance, D7 store gate, D8 vector-space identity, D9 distance telemetry,
D10 observe-mode rollout, D11 bounded query representation, D12 budget semantics. ADRs: 001
(backend port), 002 (partition digest), 003 (embedding-contract identity), 004 (non-sliding
TTL), 005 (non-destructive stock index). Rollout acceptance gates live in design.md —
Migration Plan, not here; the labeled evaluation artifact is task 4.4.

## 1. Dependency, settings, and embedding contract

- [ ] 1.1 Upgrade and pin the existing `redisvl` dependency: it is declared at
      `pyproject.toml:143` (`>=0.18.0`) and currently locked at `0.28.0`, which lacks the
      per-call `refresh_ttl` and `refresh_ttl_on_hit` controls required by the non-sliding
      lifetime. Raise the floor to `>=0.32.0,<0.33.0` and refresh `uv.lock` deliberately to the
      resolved version after confirming it provides the stock schema, explicit `vector=` on
      `acheck`, stock entry serialization/key derivation, custom `filterable_fields`, and hit-TTL controls used by design D5;
      do not install a second copy or silently float the lock.
      Implements: design D1, D5 · ADR-001, ADR-004, ADR-005
      verify: `uv lock --upgrade-package redisvl && uv sync --locked && uv run python -c "import redisvl; print(redisvl.__version__)"` — prints the resolved version inside `>=0.32.0,<0.33.0`, exit 0
- [ ] 1.2 Add semantic-cache settings to `config/settings.py`: mode
      (`off` | `observe` | `serve`, default `observe` — design D10), backend
      (`redisvl` only; any unsupported value, including `langcache` until its adapter
      exists, fails settings validation — design D1), distance threshold (default `0.05`
      documented as a conservative **policy** default on the cosine-distance [0, 2] scale,
      not a proven bound), TTL (default `86400`s, non-sliding — design D3/D11), query-preview
      bound (default `64` characters, configurable and capped — design D11), query-embedding
      character bound (default `8000`, a conservative pre-tokenizer cap — design D3), lookup
      deadline (default `0.5`s, covering embedding + Redis — design D3), and store deadline
      (default `0.5`s, covering embedding + Redis — design D3)
      Implements: semantic-answer-cache "Similarity is threshold-gated and deployment-configurable" (both scenarios) · design D3, D10
      verify: `uv run pytest tests/unit/test_settings.py -q` — new fields validated, `langcache`/unknown backend rejected, exit 0
- [ ] 1.3 Add the optional `cache_provenance` field to the ask response DTO chain (absent on
      fresh, `"exact"` | `"semantic"` on cache-served), strictly additive
      Implements: semantic-answer-cache "Cache hits carry truthful provenance" (scenario "Each serving mode is distinguishable")
      verify: `uv run pytest tests/unit/features/documents -q -k provenance` — DTO round-trip test passes
- [ ] 1.4 Export a public embedding-contract revision from `src/app/ai/embeddings.py` (the
      module already versions `_CACHE_NAMESPACE = "embedding:v2"`); the revision changes
      when the query-side instruction template, provider-input contract, normalisation, or
      output namespace changes. Do not add a second embedding implementation or a parallel
      revision constant in the cache layer.
      Implements: semantic-answer-cache "Similarity is only compared within one vector space" (scenario "A query-instruction change makes old entries unreachable") · "Cache embeddings resolve to the single embedding path" (both scenarios) · design D8 · ADR-003
      verify: `uv run pytest tests/unit/shared/langchain_layer/test_embeddings_unified.py -q -k contract_revision` — public revision exported, cache identity uses it, no second embedder, exit 0

## 2. Shared cache layer: port, digest, adapter

- [ ] 2.1 Split the exact-match answer-key derivation into query-free partition bytes and
      derive the partition digest from them; the exact-match key derivation and its parity
      contract stay byte-identical
      Implements: semantic-answer-cache "Similarity never substitutes for authorization partition identity" · legal-corpus-retrieval "Answer cache keys are version-aware" (scenarios "Cache-key derivation is pinned by parity", "A similarity match does not cross a version or scope boundary") · design D2 · ADR-002
      verify: `uv run pytest tests/unit/features/documents/retrieval -q -k partition_digest` — same-partition digests equal, differing user/tenant/scope/version/evidence-policy digests differ, exact-match key unchanged, exit 0
- [ ] 2.2 Implement the backend port (lookup/store/close) and the RedisVL adapter in the
      shared cache layer using the stock `SemanticCacheIndexSchema` per design D5: FLAT
      cosine FLOAT32 over `EMBEDDING_DIMENSION`, dimension-suffixed index name, stock
      `prompt`/`response`/`prompt_vector`/timestamps, custom filter fields for `partition`,
      `embedding_model`, `embedding_dimension`, and `embedding_revision`. Keep generation ID, absolute expiry, and bounded
      `query_preview` in metadata, outside the filters used by stock entry-ID derivation.
      `prompt` is a stable digest of the full normalized query, using the
      same normalization as `_answer_cache_key`; `response` is the serialized
      `CachedAskEnvelope`. Supply vectors explicitly from
      `src/app/ai/embeddings.py` (single path, client constructed once, query-side task type
      for BOTH store and lookup); a thin vectorizer delegate guards the path and never
      chooses a model or embeds independently. Inject the process's tuned async Redis client
      into the stock cache (0.32 has no async-client constructor arg; do not open a second
      pool; inject it as the cache's async client through the version-pinned private seam,
      mark the cache non-owning, and make the
      adapter's `close` a no-op because lifespan owns shutdown), construct with `ttl=None`
      (task 2.5 owns expiry) and RedisVL
      index management disabled because startup owns lifecycle (task 3.1), and filter only
      partition + identity + revision — never full-query equality. Query with maximum cosine
      distance (`2.0`) and apply the configured threshold in the adapter so a nearest
      above-threshold candidate is still available for telemetry (task 2.4). Lookup takes a
      serve/observe flag: a qualifying observe hit logs `hit-would-serve` and returns no
      servable match, so observe cannot accidentally serve.
      Implements: semantic-answer-cache "A near-duplicate question is answered from the cache before generation runs" (scenario "Paraphrased repeat serves the stored envelope") · "Similarity is only compared within one vector space" (all scenarios) · "A qualifying fresh answer is stored for future similarity lookup" (scenario "A fresh answer becomes similarity-eligible") · "Cache embeddings resolve to the single embedding path" (all scenarios) · "Stored query representation is a digest plus bounded preview" (all scenarios) · design D1, D5, D8, D11 · ADR-001, ADR-003, ADR-005
      verify: `uv run pytest tests/unit/utils/cache/test_semantic_cache_adapter.py -q` — store/lookup round-trip against fakes (no Redis); different model/dimension/revision unreachable; digest distinct from preview; query metadata beyond the preview bound absent; embedding path asserted single; process Redis client reused; exit 0
- [ ] 2.3 Enforce end-to-end lookup and store deadlines with failure-to-degradation
      translation, reusing the graph's existing bounded-deadline discipline (no parallel
      mechanism). Lookup includes the query embedding plus Redis I/O; store includes any
      embedding plus Redis I/O and reuses the lookup vector when available. Store timeout
      never fails or alters the fresh answer. A query above the configured embedding
      character bound is not embedded: semantic lookup and store degrade to exact-match-only
      and log the reason. Map outcomes to the shared `CacheResult` / `CacheError` discipline.
      Implements: semantic-answer-cache "Degradation reproduces the uncached ask path" (scenario "An unavailable backend leaves the ask path unchanged") · "Lookup and store are time-bounded end to end" (all scenarios) · design D3
      verify: `uv run pytest tests/unit/utils/cache/test_semantic_cache_adapter.py -q -k "degradation or deadline"` — unreachable backend, failed lookup, slow embedding, oversized query, lookup timeout, and store timeout all yield the intended outcome; fresh answer preserved; exit 0
- [ ] 2.4 Record the nearest partition-eligible candidate's measured distance plus the
      explicit lookup outcome (`hit-and-served` / `hit-would-serve` in observe /
      `miss-with-distance-above-threshold` / `miss-with-no-candidate` / `degraded`) in logs
      for every lookup, hit or miss, using
      bounded linkage (partition and embedding fingerprints, candidate/generation IDs,
      correlation id) — never query previews, query text, or answer text. Join a separately
      access-controlled, retention-bounded evaluation artifact by those IDs when collecting
      safe-substitution labels; do not turn application logs into a question/answer archive. This is why task 2.2 queries at max distance and
      applies the threshold itself.
      Implements: semantic-answer-cache "Lookups record their measured distance for tuning" (all scenarios) · design D9, D11
      verify: `uv run pytest tests/unit/utils/cache/test_semantic_cache_adapter.py -q -k telemetry` — distance+explicit outcome logged for served hit, observe-only hit, above-threshold miss, no-candidate miss, and degraded lookups; no previews or query/answer text in records; exit 0
- [ ] 2.5 Implement atomic publication with non-sliding absolute expiry. Use stock RedisVL
      entry serialization/key derivation, but replace the split `astore`/expiry calls with
      a bounded single-key Lua operation through the existing client. It validates arguments
      and hash type before mutation, checks server time, rejects expired generations and
      older generations overwriting newer ones, and publishes fields plus `PEXPIREAT` while
      preserving the earlier of existing finite expiry and incoming absolute expiry. Handle
      script command errors explicitly: atomic execution is not rollback. Persist generation
      ID/completion time/deadline before the first write (task 4.2) and preserve on retry.
      Keep `refresh_ttl_on_hit=False` and `refresh_ttl=False`; lookup requires valid absolute
      expiry and positive remaining TTL. No grace-period or unconditional delayed deletion.
      Implements: semantic-answer-cache "A semantic entry has a non-sliding maximum lifetime" (all scenarios) · "Stored query representation is a digest plus bounded preview" (scenario "Entry expiry removes query representation with the entry") · design D3, D5, D11 · ADR-004
      verify: `uv run pytest tests/unit/utils/cache/test_semantic_cache_adapter.py -q -k ttl` — hits/observe/retry preserve deadline; replay after key expiry is rejected; missing metadata fails closed; concurrent writes retain finite expiry; script failure and client cancellation leave no persistent entry; exit 0
- [ ] 2.6 Add a gated real-Redis integration test for the adapter against a Search-capable
      Redis: index field set, FLAT/cosine/dimensions, custom filters and revision scoping,
      digest/response round-trip, atomic publication and non-sliding expiry, rejected replay
      after expiry, concurrent writers, and client interruption during publication. Mark `integration` and `real_redis`;
      skip with a clear reason when no Search-capable Redis is provisioned. This test is an
      acceptance gate for serve mode (task 5.6), not optional evidence.
      Implements: semantic-answer-cache "Similarity is only compared within one vector space" (all scenarios) · "A semantic entry has a non-sliding maximum lifetime" (all scenarios) · design D4, D5 · ADR-004, ADR-005
      verify: `SEMANTIC_CACHE_TEST_REDIS_URL=redis://localhost:6379 uv run pytest tests/integration/utils/cache/test_semantic_cache_redis.py -m "integration and real_redis" -q` — passes against a Search-capable Redis (a skip is not a pass); exit 0

## 3. Lifespan wiring and index lifecycle

- [ ] 3.1 At startup, inspect the semantic index and distinguish missing index, absent Search
      capability, ACL denial, connectivity failure, and schema mismatch. Create a missing
      index non-destructively (`overwrite=False`, never drop); treat a concurrent
      already-exists result as success after re-inspection; verify required scripting and
      expiry permissions using a disposable expiring probe; validate prefix/storage, vector dimensions/dtype/distance/algorithm, and
      required/custom fields; then construct the selected backend onto `app.state` with index
      management disabled. An incompatible
      schema is left intact and disables semantic lookup/store for the process; exact-match
      cache is unaffected. Log `maxmemory-policy` and current memory against the capacity
      arithmetic in design Risks, explicitly calling out `noeviction`.
      Implements: semantic-answer-cache "Index lifecycle is non-destructive and schema-validated" (all scenarios) · "Degradation reproduces the uncached ask path" (scenario "An absent search capability disables the layer for the process") · design D4, D5 · ADR-005 · Risks (capacity arithmetic)
      verify: `uv run pytest tests/unit/lifecycle/test_semantic_cache_lifecycle.py -q` — clean boot creates+validates; concurrent start re-inspects; missing script/expiry permissions disable semantic caching; incompatible schema not overwritten and semantic disabled; unsupported/ACL/connectivity distinguished; policy warning on noeviction; exit 0

## 4. Retrieval graph integration

- [ ] 4.1 Query-analyzer cache seam: on exact-match miss and mode not `off`, attempt semantic
      lookup (digest filter + embedding identity/revision + threshold + deadline); in
      `observe` it logs but returns no servable match; a serve-mode hit returns the stored
      envelope with semantic provenance, sets `cache_hit` true, and skips downstream retrieval/grading/generation; the
      existing exact-hit path sets `"exact"`; every cache-served response logs its
      provenance; entries under a different partition or embedding identity are unreachable.
      Do not move the lookup ahead of the planner/exhaustion gate.
      Implements: semantic-answer-cache "A near-duplicate question is answered from the cache before generation runs" (both scenarios) · "Similarity never substitutes for authorization partition identity" (both scenarios) · "Similarity is only compared within one vector space" (all scenarios) · "Similarity is threshold-gated and deployment-configurable" (scenario "An above-threshold entry is not served") · "Observe mode validates the threshold without serving" (all scenarios) · "Semantic hits serve through the exact-hit boundary" (both scenarios) · "Cache hits have explicit budget semantics" (both scenarios) · "Cache hits carry truthful provenance" (scenario "Each serving mode is distinguishable")
      verify: `uv run pytest tests/unit/features/documents/retrieval -q -k semantic_lookup` — paraphrase hit serves envelope without downstream pipeline run; observe logs but does not serve while an exact hit still serves; off never looks up; exact and semantic hits carry distinct provenance while fresh has none; cross-user/cross-scope/cross-revision/above-threshold produce fresh runs; semantic-hit downstream delta and retrieval/grading/generation spend identical to exact-hit while planning/upstream charges remain; exhausted run never consults the lookup; exit 0
- [ ] 4.2 Finalisation path: store the fresh envelope for semantic lookup on exactly the
      exact-match eligibility gate (bypass, non-cacheable answers, and `off` mode skip the
      semantic layer) plus the quality predicate: degraded fallback envelopes (grade-cap
      escape, generation-budget refusal) stay exact-match-only, unclassifiable treated as
      degraded (fail closed). Store uses the original graph-entry query and the lookup vector
      when available — never the plan-rewritten retrieval query. A replayed or repeated store
      does not extend the entry's lifetime. Persist the generation identity and absolute
      expiry in checkpointed state before publishing; if this cannot be established, skip
      the cache write. Retry/resume reuses that identity, while actual regeneration creates
      a new one. Never derive a new expiry merely from replay time.
      Implements: semantic-answer-cache "A qualifying fresh answer is stored for future similarity lookup" (both scenarios) · "Degraded answers never enter the similarity layer" (both scenarios) · "A semantic entry has a non-sliding maximum lifetime" (scenario "Replay does not refresh expiry") · design D7 · ADR-004
      verify: `uv run pytest tests/unit/features/documents/retrieval -q -k semantic_store` — cacheable answer stored for both layers with the original-query vector; bypass/excluded/degraded/off answer stored by exact only or neither; replay does not extend expiry; exit 0
- [ ] 4.3 Pin the self-contained-query assumption with a test: the qna engine boundary
      receives query-only state (no transcript messages) and the ask API request carries a
      single standalone query field — the exact-match and semantic caches therefore serve
      answers whose full semantic input is (query, partition)
      Implements: design Context stated assumption · legal-corpus-retrieval "Answer cache keys are version-aware" (all scenarios presuppose query-complete inputs)
      verify: `uv run pytest tests/unit/features/agent_saul -q -k engine_query_boundary` — query-only boundary assertion passes, exit 0
      verify: `uv run pytest tests/unit/features/documents -q -k ask_request_shape` — single-query request shape assertion passes, exit 0
- [ ] 4.4 Build the labeled safe-substitution evaluation and calibration report under the
      evals inventory (for example `evals/golden/semantic_answer_cache_v1.jsonl` and
      `evals/reports/semantic_answer_cache_v1.calibration.json`). Label `(query, candidate
      answer, evidence)` cases as safe-to-substitute vs must-not-serve; keep calibration and
      held-out splits query-family-disjoint; include negations, party swaps, jurisdiction
      changes, and multilingual paraphrases. Use the single embedding path and the same
      threshold logic as serving. Select the threshold from calibration, then compute the
      false-hit rate (unsafe substitutions / all would-serve decisions) and its one-sided
      95% upper confidence bound on representative held-out data. Report known-negative
      false positives separately. The initial serve gate is an upper bound ≤1%; with zero
      errors this requires ≥299 independent would-serve cases. No hits or a smaller sample
      is insufficient evidence. Report exact one-sided Clopper–Pearson bounds, the frozen
      threshold, embedding contract, sampling procedure, effective independent sample count,
      and adversarial outcomes. A contract/threshold change invalidates prior acceptance;
      do not tune on a failed holdout and keep calling it unseen. Explicitly label the 1%
      bound as an initial risk policy rather than a proof of safety.
      Implements: semantic-answer-cache "Threshold acceptance is evidenced by labeled safe-substitution evaluation" (all scenarios) · "Similarity is threshold-gated and deployment-configurable" (scenario "The threshold is a deployment setting") · design D3, D10 · Migration Plan
      verify: `uv run python evals/experiments/semantic_answer_cache_calibration.py --golden evals/golden/semantic_answer_cache_v1.jsonl --report evals/reports/semantic_answer_cache_v1.calibration.json` — report includes calibration and held-out metrics, sample sizes, false-hit upper bound, and adversarial outcomes; exit 0
      verify: `uv run pytest tests/unit/ai/evaluation -q -k semantic_cache_calibration` — split isolation, correct served-case denominator, zero-hit rejection, and one-sided
      bound calculation (298 insufficient / 299 sufficient at zero errors) are unit-tested; exit 0

## 5. Acceptance gate

- [ ] 5.1 Run the format and lint gates
      verify: `uv run ruff format src/ tests/ && uv run ruff check src/ tests/` — clean, exit 0
- [ ] 5.2 Run the type gate
      verify: `uv run ty check src/` — clean, exit 0
- [ ] 5.3 Run the default/unit test suite (the repo's default addopts exclude `integration`
      and `requires_db`)
      verify: `uv run pytest -q` — all pass, exit 0
- [ ] 5.4 Run the structural rules scan
      verify: `ast-grep scan src/` — no new violations
- [ ] 5.5 Validate the change's spec deltas strictly
      verify: `openspec validate add-semantic-answer-cache --type change --strict` — all deltas pass, exit 0
- [ ] 5.6 Run the gated real-Redis semantic-cache integration tests in a Search-capable
      environment and record the run. A skip is not acceptance for serve mode.
      verify: `SEMANTIC_CACHE_TEST_REDIS_URL=redis://localhost:6379 uv run pytest tests/integration/utils/cache/test_semantic_cache_redis.py -m "integration and real_redis" -q` — passes, exit 0

## Task Dependency Graph

- **Wave 1** (independent): 1.1, 1.2, 1.3, 1.4
- **Wave 2**: 2.1 (needs 1.2 settings only for tests)
- **Wave 3**: 2.2 (needs 1.1 dependency, 1.4 public revision, 2.1 digest)
- **Wave 4**: 2.3, 2.4, 2.5 (extend 2.2); 3.1 (needs 2.2 adapter)
- **Wave 5**: 2.6 (needs 2.2–2.5 and 3.1 startup), 4.1, 4.2 (need 2.1–2.5 and 3.1 wiring),
  4.3 (assumption pin), 4.4 (evaluation — independent of graph integration)
- **Wave 6** (gates): 5.1–5.6, strictly last; 5.6 requires a Search-capable environment
