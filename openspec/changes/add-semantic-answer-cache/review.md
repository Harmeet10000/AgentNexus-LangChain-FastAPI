# Review: add-semantic-answer-cache

**Class:** L

Reviewed as a reviewer, against proposal.md, both spec deltas, design.md, adr.md, and the
contracts skip. Cross-checked the MODIFIED delta against
`openspec/specs/legal-corpus-retrieval/spec.md` ("Answer cache keys are version-aware") and
the integration seam against `features/documents/retrieval/graph/nodes.py`.

## Completeness

- [x] Every requirement has ≥1 scenario (`####`, WHEN/THEN) — 18 added requirements (1–4
  scenarios each) + 1 modified requirement with 4 scenarios
- [x] Edge and error cases named: backend down, missing/incompatible index, capability
  absent, ACL denial, lookup/store deadline overrun, similarity-eval failure,
  cross-user/cross-scope attempts, cross-revision attempts, above-threshold miss,
  non-cacheable/bypassed/degraded answers — not just the hit path
- [x] Quality attributes quantified: threshold (with labeled acceptance bound) and lookup/store
  deadlines use Scale | Meter | MUST Planguage tables; no bare adjectives ("tight" is anchored
  to a policy default in design D3, not claimed as proven)
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
  module replacement) and keeps Redis-out-of-the-loop unit tests; the real-Redis integration
  test is gated and is a serve-mode acceptance gate, not optional evidence
- [x] Gates appear as explicit tasks: `uv run ruff format src/ tests/`, `uv run ruff check
  src/ tests/`, `uv run ty check src/`, default `uv run pytest`, `ast-grep scan src/`, and
  the gated `-m "integration and real_redis"` run

## Risk

- [Resolved] **Embedding drift vs stored vectors** — flagged earlier; now structurally
  mitigated by ADR-003 (model id + dimension + public embedding-contract revision as
  match-predicate components, dimension-scoped index) and the matching spec requirement. A
  model or query-instruction swap yields clean misses until old entries expire, even at
  identical width.
- [Resolved] **Threshold default could be mistaken for a proven bound** — the 0.05 default is
  now explicitly a policy starting point, and serve mode requires the labeled
  safe-substitution evaluation with a one-sided 95% upper bound ≤1% on unsafe substitutions among held-out would-serve decisions
  (≥299 independent would-serve cases for zero errors). Known-negative false positives
  are reported separately and do not establish served-answer accuracy.
- [Resolved] **Index creation could race or destroy schema** — startup now inspects,
  creates-if-missing non-destructively, re-inspects on already-exists, validates the full
  schema, and never overwrites an incompatible index (ADR-005, D4).
- [Resolved] **Sliding TTL could make hot entries immortal** — expiry is non-sliding from
  a persisted generation deadline; atomic publication prevents a split-write crash window,
  and retries cannot resurrect an expired generation (ADR-004, D3/D11).
- [Flagged] **Lookup/store deadline (default 0.5s) vs the existing 5.0s Redis timeout
  discipline** — the settings must stay coherent; deadlines are the tighter ask-path bounds
  and cover embedding plus Redis I/O. Implementation must reuse the existing
  bounded-deadline helper rather than inventing a parallel one.
- [Flagged] **Cosine-distance scale** — RedisVL reports cosine distance in [0, 2]; the
  adapter MUST document the scale in the setting description so operators are not misled.
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
archive), (4) threshold calibration eval (task 4.4 evaluates the asserted 0.05 rather than
assuming it). Requirements then 11; re-checked: scenarios present, no internal names,
no new async surface, wave list then complete (17 tasks, 6 waves).

Third pass, same day — project-specific edge-case sweep, all code-verified:
(1) fallback-answer poisoning (the grade-cap/budget-refusal fallback is cached today; the
semantic layer would spread it to paraphrases for 24h) → new requirement "Degraded answers
never enter the similarity layer" + D7 quality predicate, fail-closed; (2) query plaintext
becomes FT-searchable (today the key is sha256 and the envelope carries no query — a *new*
surface) → "Stored query representation is a digest plus bounded preview" + D11 bounded
query representation, vector from full query; (3) serving-boundary parity for agent_saul (delta consumes
generated_answer/resolved_citations/stage_usage) → "Semantic hits serve through the
exact-hit boundary" + parity assertions in task 4.1; (4) budget semantics answered as
status-quo-documented → "Cache hits have explicit budget semantics" + D12. Task sharpenings:
2.2 (existing client reuse, query-side task type), 3.1 (schema comparison not existence),
4.4 (adversarial pairs: negations, party swaps, multilingual). Tier-3 dispositions recorded
as design "Named and accepted": LangGraph replay idempotency, boot stampede, bulk-invalidation
orphaned entries, TTL layering blur, oversized-query cap, task-type trap. Requirements then
15; re-checked: scenarios present, SHALL/MUST, no internal names, no new async surface.

Fourth pass, same day — consistency pass against the parent review, with code and
dependency facts re-verified:
(1) embedding identity now includes the public embedding-contract revision exported by
`src/app/ai/embeddings.py` (the module already versions `_CACHE_NAMESPACE`), because model +
dimension alone cannot detect query-instruction changes;
(2) threshold acceptance is a spec requirement backed by a labeled safe-substitution
evaluation, calibration/held-out family separation, adversarial cases, and a one-sided 95%
upper bound ≤1% on unsafe substitutions among would-serve decisions (≥299 independent
cases for zero observed errors; known-negative counts cannot supply this denominator);
(3) budget wording corrected: planning plus retries and upstream intent scoring remain
charged; only retrieval/grading/generation are zero; planner-before-cache ordering and the
exhaustion gate are preserved;
(4) TTL is explicitly non-sliding from a persisted generation deadline, with atomic
publication and hit/observe/replay refresh disabled; late replay cannot resurrect an entry;
(5) startup index lifecycle is inspect → create-if-missing non-destructively → re-inspect on
race → full schema validation; incompatible schemas are never overwritten and disable the
layer;
(6) adapter assumptions now match the stock RedisVL schema (FLAT cosine, `prompt`/
`response`/`prompt_vector`/timestamps, custom filter fields), with a full-query digest as the
stock prompt, a bounded preview, explicit vectors from the unified path, and identity/revision
filters;
(7) lookup and store deadlines both cover embedding + Redis I/O; store timeout cannot fail a
fresh answer; the real-Redis test is a serve-mode gate, not fake-only evidence;
(8) spec wording corrected: distance ≤ threshold is a hit, above-threshold is a miss; observe
suspends semantic serving only; non-cacheable, bypassed, degraded, and `off`-mode answers are
not stored; the two task 4.3 commands are separate; stale `query_text` duplicate removed.
Also corrected the dependency task: `redisvl` is already declared (`>=0.18.0`, locked
`0.28.0`), but the non-sliding lifetime needs `refresh_ttl` controls absent from `0.28.0`;
task 1.1 raises the floor to `>=0.32.0,<0.33.0` and refreshes the lock deliberately rather
than adding a second package; the upper bound pins the private async-client seam.
Requirements now 18 added (42 scenarios) + 1 modified (4 scenarios); tasks now 21 across
6 waves.

Parent final pass: corrected the statistical denominator (unsafe / would-serve), selected
an initial 1% upper-bound policy, and separated known-negative adversarial testing. Replaced
the split hash-write/expiry and grace-period-delete design with atomic publication and
checkpoint-stable generation deadlines. Added interruption and post-expiry replay checks.
Removed query previews from application logs and the claim that a plain query digest is
anonymous; evaluation artifacts need separate access control and retention. These are plan
changes only: Redis integration, runtime performance, and serving safety remain unverified
until the implementation and operational acceptance gates run.

VERDICT: APPROVED FOR IMPLEMENTATION; serve mode remains gated by the evaluation report and
the gated real-Redis run.
