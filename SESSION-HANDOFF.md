# SESSION HANDOFF — 2026-11-14

**Everything a new session needs to continue without re-deriving anything.**
Read this top to bottom; the paste-able handover prompt is at the end (§9).

---

## 1. What this session did (in order)

Eleven commits, `0703ed5` → `de3e100`. All gates green at every landing;
final state: **2499+ unit tests, 0 failures**; `openspec validate --all` green;
`lint-imports` 5/5; ty 2 pre-existing diagnostics only.

| Commit | What |
|---|---|
| `0703ed5` | **harden Phase 3 (7 tasks) + Phase 4 (4 tasks) closed** — operation policy (live-membership self-review fix), session scope verified, state-version upgrades verified, cache/idempotency scope tests, **status machinery implemented** (typed `DocumentStatus`, stuck-age detection, dead-letter→failed, and the live-run diagnosability gap closed: every `extraction_incomplete` write carries a typed reason), outbox terminal-event tests, checkpoint-retention tests. Phase 4: two-contract identity fixture, fidelity-visibility fixtures (**fixed real bug: `completed_with_warnings` was reported as `index_not_ready`**), ingestion gold set (`ingestion_fidelity_v1.jsonl`, 25 rows/17 artifacts), offline fidelity report (25/25 gate pass). |
| `f36bf5d` | **harden Phase 5 (8) + Phase 6 (10) closed** — one `retrieve()` service over the mounted graph's fused core; provider timeouts + typed degradation; ONE row-hydration implementation (**fixed citations pinned at page 0**); 8-status evidence envelope; all four knowledge tools behind runtime context + **assertion-tested registry (caught real drift: compliance had lost `retrieve_statute_section`)**; O1 degradation counters + alerts; graph shape frozen + offline multi-hop A/B. Saul: **LLM re-segmentation deleted** (deterministic persisted-chunk adaptation, `chunk_id` as clause ID), checkpoint text removal (256 KB limit), plan-conditional extraction, typed `ComplianceNotAssessableReason`, 10-case grounding-gate matrix, review rendering with override provenance, gated finalization. |
| `dd3abfe` | **harden Phase 7 (7.2–7.10) + Phase 8 (8.1–8.8) closed** — deletion machinery **built and gated** (typed `deletion_policy_not_approved` refusal first; per-kind removal paths; 12 tests); Phase 7 cleanup audit; port-or-discard table with E4 evidence; **Phase 8: 40-row answer gold set (double-labeled, SME pending), entailment report schema, harness wiring proof (24,863-chunk universe, 0 missing golden ids), E7/E8 reports, experiment set, CI isolation/near-miss tests, zero-tolerance assertion suite + CI steps.** Red-team found a **pre-existing production regression**: `establish_contract_boundaries`' length arm severed Governing-Law clauses (SRC-01 split into 840 slices) — root-caused, fixed, reconciled (red-team 88/0). |
| `a6a86d3` | **Trigram removed from the served RRF fusion** (owner directive + E4 evidence: weakest leg solo 0.5097; the fused pair traded −0.32 recall for +0.61 MRR) — branch registry, hybrid node weights (2 sites), `RRF_WEIGHT_TRIGRAM` deleted, parity re-recorded to the two-branch contract, and the removal **unmasked a real two-door divergence**: `_normalize_plan` forced weights by query_type while the inline path total-normalized only — aligned (parity drift item 9, closed). **STRIDE threat model written** (`docs/security/stride-threat-model.md`) — 6 boundaries walked, residuals with owners (D-1/D-2/D-3 among them). |
| `80b9006` | **5.11 + 9.1 closed (harden 77/86)** — the deliberate `graphiti_verified` difference documented in the spec + 3 side-by-side tests (Graphiti is never legal truth); 9.1 observability audit (10/11 event families EXIST; one gap owned by auth); **first live bm25 scoring over the partial corpus recorded** (evals/reports/live_baseline_legal_retrieval_v2.json). |
| `c9334e7` | **The 2 long-failing tests fixed** — they predated the key-set boundary contract; now route bytes through a stub object store to reach their graph-translation paths. **First fully green suite: 2486, 0 failures.** |
| `77febfe` | **Conditional parsing** — plain text (txt/md) skips Docling entirely (raw text as markdown, form-feed page counting, OCR `None` per the schema's documented non-PDF semantics, budgets enforced); 6 tests incl. a pdf-costume routing guard. |
| `641b60c` | **Reranker fixed live** — the committed default `rerank-v3.0` doesn't exist for the Cohere account (every rerank 404'd, degrading to fused order invisibly); model list re-queried from `/v1/models`; default → `rerank-v3.5`; **verified live: cross-encoder reordered against the fused order (relevant 0.8142 vs distractor 0.0517)**. |
| `7b7b2dd` | **D-2 fixed (STRIDE)** — parser singleton-converter (process-wide, convert serialised by lock; the per-call ~3.7 GB model load fed the kernel OOM killer); 5 tests. **Rerank stratum measured** on the golden corpus (see §5). STRIDE residuals D-1/D-2/D-3 updated. |
| `de3e100` | Two active plan documents tracked (they were untracked at risk); the retired `PLAN-bulletproof-ingestion.md` loose duplicate in changes/archive/ removed (the dated archive copy is authoritative); corpus-resume todo recorded in todo.md. |

Also delivered outside commits: the **live measurement run** (§4), the **testing principal** (§3), the **STRIDE threat model** (`docs/security/stride-threat-model.md`), and this handoff.

---

## 2. Current verified state

| Program | State |
|---|---|
| **harden** | **77/86** — Phases 0–8 closed. Remaining 9: 7.3 (external legal signoff), 9.2–9.9 (staged rollout/pilot — need staging). |
| **roadmap** | **6/11** — G4 closed (branch default-off, measured weight cited), G2/G5/G6 implemented **flag-off**, G7 console-gated. |
| **Tests** | 2499+ passing, 0 failures. lint-imports 5/5. ty: 2 pre-existing diagnostics. openspec validate green. |
| **Served retrieval** | Compiled graph serves `/search/ask` + `/legal/ask` (inline duplicate deleted); fusion = **bm25 + vector** (trigram removed, `remove-trigram-fusion-branch`); graphiti leg behind `DOCUMENTS_GRAPHITI_FUSION_ENABLED` (default off); reranker wired but **degrades until the key's model ID is used** (see §6). |
| **Saul** | Deterministic segmentation (no LLM), evidence-handle-based text reads (256 KB checkpoint limit), grounding gate + 10-case matrix, HumanReviewSink, ledger adapter, deep-research bounded subgraph. |
| **Deletion** | Machinery built and **gated**: `DELETION_POLICY_APPROVED=true` set in dev env per owner approval (2026-11-14); end-user surface not built; real client data still blocked behind 7.3/9.6. |
| **Security** | STRIDE model at `docs/security/stride-threat-model.md` — 8 residuals with owners (D-2 FIXED). Red-team 88/0. |

---

## 3. Infrastructure, credentials, and how to boot

**Full details: `scripts/live_measurement/RUNBOOK.md` — read it first.** Short form:

```bash
cd /home/harmeet/Desktop/Projects/langchain-fastapi-production
# API (boot ~6-7 min; NO make target — manual uvicorn; the API ships in docker):
setsid nohup uv run uvicorn app.main:app --host 127.0.0.1 --port 5000 > /tmp/uvicorn.log 2>&1 < /dev/null &
# Workers (TWO required — the ingestion queue is separate):
setsid nohup make celery > /tmp/celery_worker.log 2>&1 < /dev/null &
setsid nohup make celery-ingestion > /tmp/celery_ingestion.log 2>&1 < /dev/null &
# Readiness: curl http://127.0.0.1:5000/health → 200   (NOT /api/v1/health — that 404s)
```

- **Credentials**: `.testing-credentials.json` (gitignored) — testing principal
  `testing@live-measurement.dev` (user_id `6af208ec501517bb55b81c8b`), verified, cookies
  captured. Regenerate: API running, then `uv run python scripts/live_measurement/testing_user.py`.
- **Cloud infra (all in .env.development, no docker)**: Cloudflare R2 (object store),
  CloudAMQP (broker), Timescale Cloud (Postgres), Neo4j Aura, Redis Cloud, Gemini.
- **Rate limits**: register 3/60s, login 5/60s, HTTP upload 3/300s (the in-process driver bypasses).
- **`.env.development` changes made this session**: `MAX_UPLOAD_SIZE=52428800` (50 MiB —
  SRC-04 is 37 MB, the 2.4 feedback loop), `DOCUMENT_EXTRACTION_TIMEOUT_SECONDS=600`
  (swap-throttled Docling exceeds 120 s), `DELETION_POLICY_APPROVED=true`,
  `RERANKER_API_KEY=<user's Cohere key>`, `RERANKER_MODEL=rerank-v3.5`.

---

## 4. The live measurement run — state and resume

- **Corpus**: 15 artifacts; **5 sources fully ingested** (SRC-01/02/12/15/16 + SYN-01..05
  = 211 chunks, 202 embedded); **7 SRC PDFs quota-failed** (SRC-03/04/07/13/14 + SYN already
  done — the failed set is SRC-03/04/07/13/14 and SRC-16 partially — see the per-source table
  in the session transcript; the runbook driver re-runs idempotently).
- **Driver** (in-process, survives worker kills): `scripts/live_measurement/reingest_failed_inprocess.py`
  — assembles the worker runtime and calls `run_document_ingestion_task` per non-terminal
  document. Sequential; SRC-04 (37 MB) needs the raised 600 s extraction budget.
- **⚠️ This machine OOM-kills long workers** (7.5 GiB RAM; systemd-oomd SIGTERM'd the cgroup
  once, and worker children get SIGTERM'd within ~1 min under memory pressure). The driver
  runs **alone** (stop uvicorn + default worker first — the in-process path needs neither;
  freeing ~1 GB). STRIDE residuals D-1/D-2/D-3 record it.
- **Scoring**: `scripts/live_measurement/live_score.py` — `LIVE_BRANCH=bm25|trigram|vector`
  (bm25/trigram need no Gemini; vector needs the quota). Writes
  `evals/reports/live_baseline_legal_retrieval_v2.json`. The rerank comparison report
  (`live_rerank_comparison_legal_retrieval_v2.json`) already carries the measured rerank deltas.
- **Runbook gotchas**: golden chunk ids are a corpus convention (`SRC-XX:pNN:cNN`, mapped by
  text containment — NOT production ids); `DocumentUploadResponse.doc_id` is the JOB id (the
  real row id maps via content sha256 — done in `.live-ingestion-mapping.json`); the response
  doc_id ≠ row id; the object key is `<sha256>.<ext>`.

**Pending on the next quota window** (all scripted, one session):
1. Rerun `ingest_corpus.py` → the 12 failed docs complete (idempotent; ~1–2 h with SRC-04).
2. `LIVE_BRANCH=vector uv run python scripts/live_measurement/live_score.py` → the vector leg.
3. G2 contextual-preamble comparison → G5 hydration measurement → G4 live confirmation.
4. Tick roadmap 2.2/4.3 with the per-class numbers; the flag decisions cite the reports.

---

## 5. Evals inventory (what each report is)

| Report | Meaning |
|---|---|
| `baseline_legal_retrieval_v2_wave4.json` (+ `.run2`, `.thresholds`) | Offline deterministic baseline over the full recorded corpus: recall@10 **0.5774**, MRR 0.3889, nDCG@10 0.4305. **The CI quality gate cites this file** — a threshold change must confront it. Determinism proven (two runs byte-identical). |
| `live_baseline_legal_retrieval_v2.json` | LIVE index, bm25 branch, **partial corpus**: recall@10 0.1645 diluted / **0.2865 on the 89 coverable queries**. The headline mixes answerable + unanswerable queries — read the coverable subset. |
| `live_rerank_comparison_legal_retrieval_v2.json` | **The reranker's measured delta**: MRR +26.7% relative on coverable queries (12 improved/3 worsened), +1.1 s/query latency, counter-flip verified, 20/155 Cohere 429s degraded honestly. Partial corpus. |
| `graphiti_branch_legal_retrieval_v2_wave4.json` + `.measurements` | The G4 weight sweep (8 points, argmax w=0.1: recall 0.5774→0.6032). The branch is registered **default-off** citing this report. |
| `multihop_ab_legal_retrieval_v2.summary.json` | Offline multi-hop A/B: +2.58 recall for the graph leg. |
| `contextual_preamble_legal_retrieval_v2_wave4.comparison.json` | G2 fixture-context comparison: contracts **+0.143 recall**, class regressions on nDCG/MRR — default-on pending the live generated-sentence measurement. |
| `context_starvation_legal_retrieval_v2_wave4.json` | G5 attribution: 46/59 missed chunks below-cut-retrievable (consistent with starvation); hydration implemented flag-off. |
| `ingestion_fidelity_i4_offline.json` | I4 fidelity gate: 25/25 rows pass (boundary counts sane after the heading fix). |
| `e8_per_contract_vs_per_document_src04.json` | E8: boundary-local extraction CONDITIONAL (marker detects 1/5 on real SRC-04; page-range seeding is the follow-up). |
| `redteam_phase8.json` | Red-team: **88/0** after the boundary fix. |
| `entailment_answer_golden_v1_offline.json` | 47 claims, coverage 1.0, citation validity 1.0 (offline deterministic judge). |
| `answer_golden_v1.assertions.json` | The 8.8 zero-tolerance assertion thresholds. |
| `e4_bm25_only` / `e4_trigram_only` | The E4 leg-isolation evidence (bm25 0.5806 > fused 0.5774 recall; trigram weakest). |

Gold sets: `legal_retrieval_v2.jsonl` (155 rows, provenance-validated, CI-gated),
`ingestion_fidelity_v1.jsonl` (25 rows), `answer_golden_v1.jsonl` (40 rows, SME pending).
Annotations: `evals/annotations/` (wave-3 + answer-golden annotator a/b).

---

## 6. Remaining work, by owner

### Automated (quota window — one session, all scripted)
- Corpus completion (7 SRC PDFs) → vector-leg scoring → **G2/G5/G4 flag decisions** →
  roadmap 2.2/4.3 ticks. Everything per §4.

### User (decisions, not tasks)
- **Phase 7 activation**: policy APPROVED; say when the end-user delete command surface is wanted.
- **7.3 formal external legal review** before real client data (9.6 keeps synthetic-only).
- **Staging environment** (Phase 9's blocker) — second Timescale/Neo4j + deployment target.
- **ag-ui-copilotkit-console** — excluded by instruction; G7 waits on it.

### Next-session code work (in priority order)
1. **Rerank stratum → served-path decision** (the report exists; the harness stratum could
   formalize it — small).
2. **D-3**: quota-aware embedding backoff with a typed `quota_exhausted` state (the one DoS
   residual still owed code).
3. **D-1**: LISTEN-loop resilience was verified sound (the relay reconnects); the API shutdown
   was the oomd kill — resolved by D-2. Verify once on the next long run.
4. **Phase 7 activation** when the user says the delete command surface is wanted.
5. **Hygiene**: the `pyproject.toml` name for the reports dir (`live_*` files are gitignored?
   verify); `scripts/live_measurement/*` lint conformance (scripts are outside the src/ ruff
   gate by design).

---

## 7. Deletable files (disk cleanup — verified 2026-11-14)

**Safe to delete now (regenerable or stale):**
- `htmlcov/` (21 MB) — coverage HTML; regenerate with `uv run pytest --cov --cov-report=html`.
- `coverage.xml` (800 KB) — regenerable.
- `.pytest_cache/` (400 KB) — regenerates on next run.
- `/tmp/*.log`, `/tmp/*.json`, `/tmp/zcode-bash-*` — session logs (uvicorn, celery, polls,
  dispatch, reranker tests, queuedepth, mapfix, remap, dbstate, health) — all outside the repo.
- `.live-ingestion-mapping.json.note` — **already deleted** (empty accidental file).
- `openspec/changes/archive/PLAN-bulletproof-ingestion.md` — **already deleted** (loose
  duplicate; the dated archive copy `2026-10-31-plan-bulletproof-ingestion.md` is authoritative).

**Regenerable but keep (resume convenience):**
- `.live-ingestion-mapping.json` (gitignored) — the source→doc-id map; regenerable via the
  sha256 remap, but keeping it saves the step.

**NEVER delete:**
- `.testing-credentials.json` (the reusable principal), `.env.development` (all keys),
  `evals/reports/*` (the measurement record), `evals/golden/*` (the corpus + gold sets),
  `evals/corpus/**` (the acquired artifacts), `openspec/**`, `docs/security/stride-threat-model.md`,
  `scripts/live_measurement/**` (RUNBOOK + drivers), the two PLAN documents in
  `openspec/changes/archive/`.

**Not mine to delete (other sessions' work):** `.research/` (parallel deep-research
artifacts), `graphify-out/` (verify purpose first), anything under the harden change's notes.

---

## 8. Concurrent-session notes

The user runs multiple agent sessions against this tree. This session's commits interleave
with theirs (their websocket-revocation and evals commits landed between mine). Two collisions
were found and reconciled (the fetch_chunks_by_ids fail-open fix; the ingestion-fixture
boundary semantics). **Always re-read files before editing; never clobber.**

---

## 9. HANDOVER PROMPT (paste into the new session)

```
Continue the Agent Saul hardening program in this repo. Read these first, in order:
1. SESSION-HANDOFF.md (repo root) — the full state: what was delivered, what remains,
   how to boot the stack, where the credentials and eval reports are.
2. scripts/live_measurement/RUNBOOK.md — the live-run runbook (stack boot, gotchas,
   the quota-blocked resume procedure).
3. openspec/changes/harden-agent-saul-evidence-pipeline/tasks.md — 77/86 done; the
   remaining 9 tasks are 7.3 (external legal signoff) and 9.2–9.9 (staged rollout —
   needs a staging environment the user has not provisioned yet).
4. openspec/changes/deferred-retrieval-roadmap/tasks.md — 6/11 done; G2/G5/G6 are
   implemented flag-off and their measured-adoption tasks need the live embedding run.

State summary: harden Phases 0–8 closed; the compiled retrieval graph serves
/search/ask + /legal/ask (inline deleted, parity suite is the regression gate);
fusion is bm25+vector (trigram removed per E4 evidence, remove-trigram-fusion-branch);
the cross-encoder reranker is live on rerank-v3.5 (verified: reordered against the
fused order); the deletion machinery is built and gated behind
DELETION_POLICY_APPROVED=true (owner-approved 2026-11-14); Saul's segmentation is
deterministic, checkpoints are text-free (256 KB limit), the grounding gate has a
10-case matrix. Live corpus: 5 of 15 sources ingested (211 chunks, 202 embedded) —
the other 10 are quota-failed; the resume is two idempotent scripts per the RUNBOOK.
The first live scoring exists: bm25 branch, recall@10 0.2865 on the 89 coverable
queries (0.1645 diluted over 155 — 66 queries' sources are not ingested), rerank
MRR +26.7% relative (evals/reports/live_rerank_comparison_legal_retrieval_v2.json).

The immediate work, on the next Gemini quota window (free tier, 1000 embed
requests/day — check scripts/live_measurement/RUNBOOK.md §ROOT CAUSE):
1. Resume corpus ingestion (idempotent driver) until all 15 sources are terminal.
2. Run the vector-branch live scoring + the G2 contextual-preamble comparison and
   the G5 hydration measurement → make the three per-class flag decisions citing
   the reports → tick roadmap tasks 2.2 and 4.3 in
   openspec/changes/deferred-retrieval-roadmap/tasks.md.
Then: harden Phase 9 remains staging-gated (user decision); 7.3 remains
external-gated. Do not implement delete's end-user surface or ingest real client
data without the user's explicit go.

House rules: follow .opencode/skills/orient/SKILL.md search routing (codegraph
first); subagents use codegraph too; only the 2 known pre-existing test failures
are acceptable (tests/unit/features/test_feature_result_migrations.py); never
rewrite golden expectations (map instead); all flags default off; every claim
gets a measurement or a test.
```

---

*End of handoff.*
