# Review — rag-tree-repair

**Reviewer verdict: BLOCK** (impl/rag-tree-repair, 6 commits on main).
Report returned by impl-reviewer subagent; saved verbatim by orchestrator.

## 0. Verdict: **BLOCK**

Three independent blockers, none of them scope:

1. **`review.md` task-4.6 `_pending_` slots still unfilled** (2 slots) + tasks **4.5/4.6 unticked** — known live-Timescale TCP outage (`Errno 111`), not implementer negligence. DoD #1/#2 fail until the human restores DB access and the implementer re-runs `alembic current`.
2. **Live database credential committed on the branch** — `docs/relay/baseline-alembic.txt` contains the Timescale password in plaintext dict-repr on 9 lines (973, 979, 986, 1222, 1228, 1235, 1454, 1460, 1467). The implementer's "redacted ×3" claim covered only the `ConnectionParameters(...)` repr lines (1013, 1262, 1494). Secret is in commit `fe243af`'s blob → sanitize + rotate before merge; orchestrator must decide on history purge (branch is unmerged/private, so purge is cheap and safe).
3. **CodeRabbit critical/major dispositions open** — the credential finding (#20) is Correct-and-blocking; the 6 docling code findings are Correct-but-out-of-scope (pre-existing, follow-ups); all spec-markdown findings are non-actionable (frozen). No code fix is required inside this change's contract, but the dispositions must be recorded and the credential fixed, then re-review.

Scope is **CLEAN**. Proofs: **0 ticked-without-evidence** (21/23 ticked, all with recorded Proof output; 4.5/4.6 honestly unticked). Change-specific checks **PASS** except the DB-dependent half of 4.6 (blocked by #1 above).

Methodology note: first `coderabbit review` invocation timed out at 300 s with no stored findings; re-ran with stdout redirected to `/tmp/opencode/cr-rag-tree.log` — exit 0, **32 findings**. Triage below covers all 32. Full JSONL preserved at that path on this machine.

## 1. Scope diff — CLEAN (with one ledger-gap note)

`git -C <worktree> diff --name-only main...HEAD` = 37 files (implementer said 36; trivial rename-counting difference, verified by `--stat`). Every file categorised:

- Declared: 7× `docling/` (git-detected rename of `document_processing/`, content diffs: `__init__` 10 lines, `embedder` 2 lines, rest 0), `langextract_to_graph.py`, `classification.py`, `parser.py`, `ingestion_kb/nodes.py`, `utils/embedding.py`, `policy_examples.py`, `0013_*.py`, 3× `tests/unit/shared/rag/`, 10× `openspec/specs/`.
- W0 Amendment (in-scope per current ledger): `pyproject.toml`, `tests/unit/test_auth_documents_feature_errors.py`, `shared/langchain_layer/embeddings.py`, `features/__init__.py` (empty), implied `document_processing/` deletion (confirmed: present on `main`, absent on HEAD, rename-detected).
- Task-mandated meta (explicitly required by tasks): `openspec/changes/rag-tree-repair/tasks.md` (tick-only), 5× `docs/relay/baseline-{alembic,pytest,ruff-after,ruff-before,ty}.txt`.
- **Outside all three — NONE.** No `service.py`, no `model.py`, no new migration, no `retrieval_kb/`, no `evaluation/`, no `lifecycle/`, no `settings.py`.

- The `0013_*.py` diff is **comment-only** (one-word `document_processing`→`docling` in a comment at :101) — complies with 4.5's "no revision authored/edited/applied."
- `src/alembic/env.py` (declared) is untouched, consistent with 4.5 unticked.
- **Ledger-gap note (not a block):** recommend standing authorisation for `<change>/tasks.md` tick-edits and `docs/relay/baseline-*` artifacts.

## 2. Proof check — 0 ticked-without-evidence; 2 honestly-unticked (DB outage)

**21/23 ticked** (implementer wrote 20/23 — undercount, safe direction). Every ticked box has Proof output in its group commit body.

Spot re-executions, all PASS: 2.7 repo-wide grep → exit 1 (zero references); 2.5 collect-only → exit 0; 5.1 per-file counts exactly `8/19, 10/27, 4/12, 7/16, 4/12, 8/24, 10/24, 10/27, 4/11, 5/15`, `validate --strict` → exit 0; 4.1/6.1 ruff → "All checks passed!"; archive-vs-live body diff for `hierarchical-document-chunking` → identical; no `ADDED Requirements` in any restored file.

Recorded deviations (acceptable): 4.1 expected `PLC2701`-only, measured zero (cause recorded with A/B verification); 6.1c notes `baseline-alembic.txt` not reproducible (DB down); `validate --specs` → 36 passed / 2 failed, both pre-existing untouched specs.

`review.md` 4.6 slots still `_pending_` → **BLOCKS** per DoD #2. Cause: documented TCP refusal. 3.2 conditional slot correctly empty.

## 3. CodeRabbit findings — all 32 triaged

### 3a. Correct + BLOCKING (1)

- **#20 `docs/relay/baseline-alembic.txt` (critical, credential leak) — CORRECT.** 9 lines carry the live Timescale password in dict-repr; only 3 `ConnectionParameters(...)` lines were redacted. Strip/sanitize the artifact, **rotate the credential**, decide history purge of `fe243af` before merge. (Value redacted from this report deliberately.)

### 3b. Correct but out of scope → follow-ups (8)

Pure-rename passengers, byte-identical to `main`. Fixing here would breach the rename-only contract.

- **#14 `ingest_v2.py:111`** (critical): `_transcribe_audio` imported but defined nowhere → runtime ImportError on audio ingest. Pre-existing. Follow-up (likely `ingestion-chunking`).
- **#18 `ingest_v2.py:70`** (critical, UTC): substance correct, **but the suggested fix is itself broken** (bare `UTC` undefined; correct is `timezone.utc`). Follow-up with corrected fix.
- **#15 `docling_enhanced.py:290–293`** (major, sync calls in coroutine). Pre-existing. Follow-up.
- **#17 `chunker.py:325`** (major, no forward-progress clamp). Pre-existing. Follow-up.
- **#19 `docling_enhanced.py:346–351`** (major, unescaped table HTML). Pre-existing. Follow-up.
- **#16 `chunker.py:392`** (minor, unstripped `paragraph`). Trivial; follow-up, lowest priority.
- **#7 `document-ingestion-pipeline/spec.md:20–22`** + **#8 `graph-entity-canonicalisation/spec.md:52–55`**: in-diff but verified verbatim archive restores — clarification must come as a future owning-change delta, never an edit to the restore.

### 3c. Frozen-spec findings — NOT actionable (9)

#2, #3 (`agentic-retrieval`), #4, #5 (`rag-eval-harness`), #6 (`knowledge-stack`), #9/#11 + #10 (`graph-lifecycle`), #12 (`retrieval-sql`): files outside this branch's diff, frozen text. Merit (if any) belongs to those changes' loops.

### 3d. Wrong — dismissed (1)

- **#13 `rag-tree-repair/tasks.md`** (major, "align 1.1 with 7.1") — WRONG: this change has no §7; the finding confuses two changes' numbering. Proof lines frozen anyway.

### 3e. Out-of-scope files — dismissed, recorded (13)

#1 (`test_mcp_client_settings.py`), #21–#32 (protocol/plans/research critiques): planning docs not in diff, not owned by this change; several already satisfied/decided. Recorded, no action.

No findings against `pyproject.toml`, `policy_examples.py`, `classification.py`, `parser.py`, `nodes.py`, both `embeddings.py`, `langextract_to_graph.py`, the `0013` comment edit, or any touched test file — the edited surface is review-clean.

## 4. Change-specific checks

- No MODIFIED block (correct). N/A.
- §5.1 count table: PASS; bodies identical modulo mandated header swap; `Purpose` carried.
- `uv run alembic heads` (offline): exactly one head — `0017`. PASS.

## 5. DoD mapping (protocol §10)

1. Boxes + Proofs — **NO** (4.5/4.6 unticked, DB-blocked). 2. `_pending_` filled — **NO** (4.6 two slots). 3. `validate --strict` — **YES**. 4. Gates ≥ baseline — **YES**. 5. CodeRabbit — **PARTIAL** (triage complete; #20 fix pending). 6. Scope ⊆ declared — **YES**. 7. One head — **YES**. 8. MODIFIED diff — **N/A**.

## 6. Follow-ups (not blocking)

1. Six `docling/` defects (#14–#19) → owning changes; #18's patch must be corrected before use. 2. Two restored-spec ambiguities (#7, #8) → owning changes' deltas. 3. Ledger: standing authorisation for `tasks.md` ticks + `baseline-*`. 4. After DB recovery, 4.5/4.6 re-run settles the three-way head claim — `ingestion-chunking` 2.2 consumes it. 5. Trivia: 20/23→21/23, 36→37 files, both errors safe-direction.

**Bottom line:** land-blocking items are (i) the two unfilled 4.6 slots behind the dead database, and (ii) the committed credential in `baseline-alembic.txt` requiring sanitisation + rotation.

---

# Re-review v2 — CLEAN (remediation verified 2026-09-10)

**NEW VERDICT: CLEAN** — tip `6c2c4cc` on `main@50093df`. All three v1 blockers verified remediated by independent re-execution. One required pre-merge human step (credential rotation, outside the branch).

## 1. Scope — CLEAN, 38/38

27 declared (7× `docling/` renames R085–R100 + implied deletion) + 4 W0 amendments + 7 standing/meta (tasks.md, 5× baseline-*, own review.md slot-fill). Outside: NONE. Worktree clean. Ledger-gap note: standing text should also name `<change>/review.md` (slot-fill only).

## 2. Secret absence — CLEARED on branch (counts only)

Working tree + HEAD: `password` lines 0, credential-URL shapes 0, unplaceholdered kwarg/dict-key shapes 0. Old blob: 9/9/9 values placeholder-valued, 12/12 URL shapes placeholdered; placeholders 24× REDACTED, 24× REDACTED_HOST, 24× REDACTED_USER. Remaining keyword lines benign (mapper logs, PostgresqlImpl context, Secret-defaults warning, loguru bookkeeping). `fe243af` not a valid object; fsck --unreachable empty; main never contained the file (0 hits); branch never pushed.

## 3. Proofs — 23/23 ticked, zero `_pending_`

Tip `6c2c4cc` records 4.5 (exit 0, `0016` = on-disk literal) + 4.6 (hash matches nothing; `0004` contradicted; disk 0001–0017; single head; 0016→0017 chain). Divergences recorded outside tasks.md. Live re-execution: `current` → 0016 exit 0; `heads` → single 0017.

## 4. CodeRabbit re-review (`--light`, docs-only delta) — prior #20 GONE

2 findings, both dismissed: #1 (verify doc stale Errno-111 row — out-of-scope orchestrator doc, corroborates remediation); #2 (review doc stale blocker — self-resolving, superseded by this v2). Prior triage stands (#14–#19 follow-ups, frozen-spec non-actionable, #13 Wrong). Log: /tmp/opencode/cr-rag-tree-rereview.log.

## 5. Change-specific — PASS (no MODIFIED; single head 0017, twice-verified).

## DoD: 1 YES (23/23) · 2 YES (0 pending) · 3 YES · 4 YES · 5 YES · 6 YES (38/38) · 7 YES · 8 N/A.

## Follow-ups

Six docling defects → owning changes (#18 patch needs `timezone.utc` fix); two spec ambiguities → owning deltas; ledger +review.md standing auth; `ingestion-chunking` 2.2 unblocked (`down_revision` = 0016... see verify: DB at 0016, head 0017); **human pre-merge: rotate Timescale credential** (hygiene, not branch defect).

**Bottom line: CLEAN. Merge after rotation; no re-review needed for rotation itself.**
