# Review: add-semantic-answer-cache

**Class:** L

Reviewed as a reviewer, against proposal.md, both spec deltas, design.md, adr.md, and the
contracts skip. Cross-checked the MODIFIED delta against
`openspec/specs/legal-corpus-retrieval/spec.md` ("Answer cache keys are version-aware") and
the integration seam against `features/documents/retrieval/graph/nodes.py`.

## Completeness

- [x] Every requirement has ≥1 scenario (`####`, WHEN/THEN) — 15 added requirements (1–3
  scenarios each) + 1 modified requirement with 4 scenarios
- [x] Edge and error cases named: backend down, capability absent, deadline overrun,
  similarity-eval failure, cross-user/cross-scope attempts, above-threshold miss,
  non-cacheable answers — not just the hit path
- [x] Quality attributes quantified: threshold and lookup deadline use Scale | Meter | MUST
  Planguage tables; no bare adjectives ("tight" is anchored to a default value in design D3,
  not left in the spec)
- [x] No decision tables required — no state machines in this capability; provenance is a
  3-way classification covered by scenarios

## Correctness

- [x] Requirements observably testable; SHALL/MUST throughout; no internal class or function
  names in spec text (cache mechanics named generically: "partition", "serving mode")
- [x] Delta operations right: ADDED (new capability with `## Purpose`, 50+ chars) and
  MODIFIED (entire existing block copied verbatim, extension appended, all three original
  scenarios preserved plus one new) — verified against the live main spec
- [x] `openspec validate` — run at the end of planning (see validation task); must pass
  strict before archive

## Standards

- [x] Layering respected: behavior port + adapter in the shared cache layer
  (`src/app/utils/cache/`), integration confined to the documents retrieval graph, no
  consumer (documents ask query, agent_saul qna engine) edited; backend constructed in
  lifespan and read from `app.state`
- [x] Secrets: no new secrets; backend selection is an enum setting; existing
  `REDIS_PASSWORD` SecretStr handling untouched
- [x] Test plan respects the `test-mock-isolation` convention (minimal proxies, no blanket
  module replacement) and keeps Redis-out-of-the-loop unit tests; live-Redis tests gated
- [x] Gates appear as explicit tasks: `uv run ruff format src/ tests/`, `uv run ruff check
  src/ tests/`, `uv run ty check src/`, `uv run pytest`, `ast-grep scan src/`

## Risk

- [Resolved] **Embedding-model drift vs stored vectors** — was flagged in the first review
  pass; now structurally mitigated by ADR-003 (model id + dimension as match-predicate
  components, dimension-scoped index), spec requirement "Similarity is only compared within
  one vector space", design D8, task 2.2. Residual exposure: none — a model swap yields
  clean misses until old entries expire.
- [Flagged] **Lookup deadline (default 0.5s) vs the existing 5.0s Redis timeout discipline**
  — the two settings must stay coherent; the deadline is the tighter, ask-path-specific
  bound and the design says overruns degrade. Implementation must reuse the existing
  bounded-deadline helper rather than inventing a parallel one.
- [Flagged] **Threshold default assumes cosine distance** — RedisVL reports cosine distance
  in [0, 2]; the spec's Planguage meter says "normalized, lower = more similar" without
  pinning the scale. The adapter MUST document the scale in the setting's description so
  operators tuning it are not misled. No spec change required.
- [Note] Security boundary is the strongest part of the plan: equality-digest partitioning
  (ADR-002) structurally prevents the cross-partition ranking that post-filtering allows.

## Contracts

- [x] Discovery records no new async behavior; contracts holds the one-line
  "No async interfaces in this change." skip; no channels required; no payload migrations.

## Post-review amendment

2026-11-15, same day: the user asked whether a version should be added to the match
predicate; the review above was refreshed after folding in three improvements — vector-space
identity (ADR-003), distance telemetry (spec requirement + D9), and semantic TTL decoupling
(D3, default 86400s). The new spec requirements were re-checked against this checklist:
scenarios present, SHALL/MUST, no internal names, no new async surface (contracts skip still
holds). Tasks gained 2.4 (telemetry) and identity-filter citations; waves remain complete.

Second pass, same day: an adversarial "is this S-tier" review found four gaps, all now
folded in — (1) observe/shadow mode as the default rollout mode (spec requirement "Observe
mode validates the threshold without serving" + design D10; the serving risk no longer
transfers to users), (2) the self-contained-query assumption named in design Context with
verified evidence (single-turn ask API; query-only engine boundary) and pinned by task 4.3,
(3) success criterion as a decision rule + capacity arithmetic (design Migration Plan steps
2–3 as operational gates; probe logs maxmemory-policy per task 3.1; deliberately NOT a spec
requirement — process gates belong to the runbook, and the 14-day wait must not block
archive), (4) threshold calibration eval (task 4.4 replaces the asserted 0.05 with a
derived default). Requirements now 11; re-checked: scenarios present, no internal names, no
new async surface, wave list complete (17 tasks, 6 waves).

Third pass, same day — project-specific edge-case sweep, all code-verified:
(1) fallback-answer poisoning (the grade-cap/budget-refusal fallback is cached today; the
semantic layer would spread it to paraphrases for 24h) → new requirement "Degraded answers
never enter the similarity layer" + D7 quality predicate, fail-closed; (2) query plaintext
becomes FT-searchable (today the key is sha256 and the envelope carries no query — a *new*
surface) → "Stored query text retention is explicit and bounded" + D11 truncated-prefix
default, vector from full query; (3) serving-boundary parity for agent_saul (delta consumes
generated_answer/resolved_citations/stage_usage) → "Semantic hits serve through the
exact-hit boundary" + parity assertions in task 4.1; (4) budget semantics answered as
status-quo-documented → "Cache hits have explicit budget semantics" + D12. Task sharpenings:
2.2 (existing client reuse, query-side task type), 3.1 (schema comparison not existence),
4.4 (adversarial pairs: negations, party swaps, multilingual). Tier-3 dispositions recorded
as design "Named and accepted": LangGraph replay idempotency, boot stampede, bulk-invalidation
orphaned entries, TTL layering blur, oversized-query cap, task-type trap. Requirements now
15; re-checked: scenarios present, SHALL/MUST, no internal names, no new async surface.

VERDICT: APPROVED
