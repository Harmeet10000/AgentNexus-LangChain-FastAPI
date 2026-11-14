# Tasks: add-semantic-answer-cache

Implements the `semantic-answer-cache` capability spec and the modified
`legal-corpus-retrieval` "Answer cache keys are version-aware" requirement. Design decisions:
D1 port+adapter, D2 partition digest, D3 settings defaults, D4 probe+degradation, D5 index
schema, D6 provenance, D7 store gate, D8 vector-space identity, D9 distance telemetry,
D10 observe-mode rollout. ADRs: 001 (backend port), 002 (partition digest), 003
(vector-space identity). Rollout gates live in design.md — Migration Plan, not here.

## 1. Dependency and settings

- [ ] 1.1 Add the `redisvl` dependency with uv
      Implements: design D1 · ADR-001
      verify: `uv sync && uv run python -c "import redisvl; print(redisvl.__version__)"` — version prints, exit 0
- [ ] 1.2 Add semantic-cache settings to `config/settings.py`: mode enum
      (`off` | `observe` | `serve`, default `observe` — design D10), backend enum
      (`redisvl` | `langcache`, default `redisvl`), distance threshold (default 0.05,
      description documenting the cosine-distance [0, 2] scale), TTL (default 86400s —
      decoupled from the exact-match 900s per design D3, safe because staleness is
      structurally prevented), lookup deadline (default 0.5s)
      Implements: semantic-answer-cache "Similarity is threshold-gated and deployment-configurable" (both scenarios) · "Observe mode validates the threshold without serving" (both scenarios) · "The similarity lookup is time-bounded" · design D3, D10
      verify: `uv run pytest tests/unit/config/test_settings.py -q` — new fields validated, exit 0
- [ ] 1.3 Add the optional `cache_provenance` field to the ask response DTO chain (absent on
      fresh, `"exact"` | `"semantic"` on cache-served), strictly additive
      Implements: semantic-answer-cache "Cache hits carry truthful provenance" (scenario "Each serving mode is distinguishable")
      verify: `uv run pytest tests/unit/features/documents -q -k provenance` — DTO round-trip test passes

## 2. Shared cache layer: port, digest, adapter

- [ ] 2.1 Split the exact-match answer-key derivation into query-free partition bytes and
      derive the partition digest from them; the exact-match key derivation and its parity
      contract stay byte-identical
      Implements: semantic-answer-cache "Similarity never substitutes for authorization partition identity" · legal-corpus-retrieval "Answer cache keys are version-aware" (scenarios "Cache-key derivation is pinned by parity", "A similarity match does not cross a version or scope boundary") · design D2 · ADR-002
      verify: `uv run pytest tests/unit/features/documents/retrieval -q -k partition_digest` — same-partition digests equal, differing user/scope/version/policy digests differ, exact-match key unchanged, exit 0
- [ ] 2.2 Implement the backend port (lookup/store/close) and the RedisVL adapter in the
      shared cache layer: index schema per design D5 (dimension-suffixed index name), custom
      vectorizer bridging `src/app/ai/embeddings.py` (single path, client constructed once,
      query-side task type for BOTH store and lookup — never the vectorizer's silent
      default), reusing the process's existing tuned Redis client rather than a
      redisvl-constructed one (same timeout/retry/keepalive discipline),
      tag filters on the partition digest AND the embedding identity (model id + dimension,
      per design D8 · ADR-003), TTL at store
      Implements: semantic-answer-cache "A near-duplicate question is answered from the cache before generation runs" (scenario "Paraphrased repeat serves the stored envelope") · "Similarity is only compared within one vector space" (both scenarios) · "Every fresh answer is stored for future similarity lookup" (scenario "A fresh answer becomes similarity-eligible") · "Cache embeddings resolve to the single embedding path" (both scenarios) · design D1, D5, D8 · ADR-001, ADR-003
      verify: `uv run pytest tests/unit/cache/test_semantic_cache_adapter.py -q` — store/lookup round-trip against fakes (no Redis), entries under a different model id or dimension unreachable, embedding path asserted single, exit 0
- [ ] 2.3 Enforce the lookup deadline and failure-to-degradation translation in the adapter,
      reusing the graph's existing bounded-deadline discipline (no parallel mechanism)
      Implements: semantic-answer-cache "Degradation reproduces the uncached ask path" (scenario "An unavailable backend leaves the ask path unchanged") · "The similarity lookup is time-bounded" (scenario "A slow backend degrades rather than stalls the ask")
      verify: `uv run pytest tests/unit/cache/test_semantic_cache_adapter.py -q -k degradation` — unreachable backend, failed lookup, and deadline overrun all yield the degradation outcome, exit 0
- [ ] 2.4 Record the nearest partition-eligible candidate's measured distance plus the lookup
      outcome (served / below-threshold miss / degraded) in logs for every lookup, hit or miss
      Implements: semantic-answer-cache "Lookups record their measured distance for tuning" (scenario "Threshold tuning works from recorded distances") · design D9
      verify: `uv run pytest tests/unit/cache/test_semantic_cache_adapter.py -q -k telemetry` — distance+outcome logged for hit, miss, and degraded lookups, exit 0

## 3. Lifespan wiring and capability probe

- [ ] 3.1 Probe the search capability at startup (FT.INFO against the cache index — compare
      the returned field set against the expected schema, not mere existence, so a stale
      index from a schema change fails the probe), construct
      the selected backend onto `app.state`, and set the process-lifetime disabled flag when
      the capability is absent (exact-match cache unaffected). The probe also logs the
      server's `maxmemory-policy` and current memory against the design D8 capacity
      arithmetic, explicitly calling out `noeviction`
      Implements: semantic-answer-cache "Degradation reproduces the uncached ask path" (scenario "An absent search capability disables the layer for the process") · design D4, Risks (capacity arithmetic)
      verify: `uv run pytest tests/unit/test_lifespan_semantic_cache.py -q` — probe success wires the adapter, probe failure leaves semantic disabled and logs the disable, policy warning logged on noeviction, exit 0

## 4. Retrieval graph integration

- [ ] 4.1 Query-analyzer cache seam: on exact-match miss, attempt semantic lookup (digest
      filter + threshold + deadline); a hit returns the stored envelope with semantic
      provenance and skips the pipeline; entries under a different partition are unreachable
      Implements: semantic-answer-cache "A near-duplicate question is answered from the cache before generation runs" (both scenarios) · "Similarity never substitutes for authorization partition identity" (both scenarios) · "Similarity is threshold-gated" (scenario "An above-threshold entry is not served") · "Semantic hits serve through the exact-hit boundary" (both scenarios) · "Cache hits have explicit budget semantics" (both scenarios) · "Cache hits carry truthful provenance"
      verify: `uv run pytest tests/unit/features/documents/retrieval -q -k semantic_lookup` — paraphrase hit serves envelope without pipeline run; cross-user/cross-scope/above-threshold produce fresh runs; semantic-hit delta shape and zero stage spend identical to exact-hit; exhausted run never consults the lookup, exit 0
- [ ] 4.2 Finalisation path: store the fresh envelope for semantic lookup on exactly the
      exact-match eligibility gate (bypass and non-cacheable answers skip both layers) plus
      the quality predicate: degraded fallback envelopes (grade-cap escape, generation-budget
      refusal) stay exact-match-only, unclassifiable treated as degraded (fail closed)
      Implements: semantic-answer-cache "Every fresh answer is stored for future similarity lookup" (both scenarios) · "Degraded answers never enter the similarity layer" (both scenarios) · design D7
      verify: `uv run pytest tests/unit/features/documents/retrieval -q -k semantic_store` — cacheable answer stored for both layers; bypass/excluded/degraded answer stored by exact only or neither, exit 0
- [ ] 4.3 Pin the self-contained-query assumption with a test: the qna engine boundary
      receives query-only state (no transcript messages) and the ask API request carries a
      single standalone query field — the exact-match and semantic caches therefore serve
      answers whose full semantic input is (query, partition)
      Implements: design Context stated assumption · legal-corpus-retrieval "Answer cache keys are version-aware" (all scenarios presuppose query-complete inputs)
      verify: `uv run pytest tests/unit/features/agent_saul -q -k engine_query_boundary and tests/unit/features/documents -q -k ask_request_shape` — both assertions pass, exit 0
- [ ] 4.4 Threshold calibration eval: embed a sampled query set (the repo's evals inventory)
      through the single embedding path, measure the pairwise distance distribution within
      partitions, identify the knee, and replace the asserted 0.05 default with the derived
      value plus the distribution artifact
      Implements: semantic-answer-cache "Similarity is threshold-gated and deployment-configurable" (scenario "The threshold is a deployment setting") · design D3
      The sample set MUST include adversarial pairs: negations, party swaps
      ("can I terminate without notice" vs "can they terminate me without notice"),
      jurisdiction-adjacent phrasings, and cross-language pairs within one jurisdiction
      (multilingual embeddings collapse language barriers) — a knee from natural queries only
      is too loose for these.
      verify: eval artifact written under the evals inventory (natural + adversarial distance
      distributions) + settings default updated + `uv run pytest tests/unit/config/test_settings.py -q` passes, exit 0

## 5. Acceptance gate

- [ ] 5.1 Run the format and lint gates
      verify: `uv run ruff format src/ tests/ && uv run ruff check src/ tests/` — clean, exit 0
- [ ] 5.2 Run the type gate
      verify: `uv run ty check src/` — clean, exit 0
- [ ] 5.3 Run the full test suite
      verify: `uv run pytest -q` — all pass, exit 0
- [ ] 5.4 Run the structural rules scan
      verify: `ast-grep scan src/` — no new violations
- [ ] 5.5 Validate the change's spec deltas strictly
      verify: `openspec validate add-semantic-answer-cache --type change --strict` — all deltas pass, exit 0

## Task Dependency Graph

- **Wave 1** (independent): 1.1, 1.2, 1.3
- **Wave 2**: 2.1 (needs 1.2 settings only for tests)
- **Wave 3**: 2.2 (needs 1.1 dependency, 2.1 digest)
- **Wave 4**: 2.3, 2.4 (extend 2.2), 3.1 (needs 2.2 adapter)
- **Wave 5**: 4.1, 4.2 (need 2.1–2.3 and 3.1 wiring), 4.3 (assumption pin), 4.4 (calibration
  eval — independent of graph integration)
- **Wave 6** (gates): 5.1–5.5, strictly last
