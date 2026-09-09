# Implementation Protocol — the seven-change RAG cluster

**Audience:** the orchestrating agent that will implement these changes, and every subagent it spawns.
**Status:** the specs are written, validated, and frozen. This document governs how they get built.
**Written:** 2026-09-10. Every fact marked *measured* was verified against the tree on that date.

---

## 0. What you are building, and what you are not

Seven OpenSpec changes sit under `openspec/changes/`. Each has six artifacts:

| Artifact | What it is | Your relationship to it |
|---|---|---|
| `proposal.md` | why, what, capability ownership | **read-only** — never edit |
| `specs/<capability>/spec.md` | the requirement delta | **read-only** — never edit |
| `design.md` | the reasoning, the rejected shapes | **read-only** |
| `adrs.md` | the decisions and their consequences | **read-only** |
| `tasks.md` | the ordered work, each with a **Proof** | **tick boxes only** |
| `review.md` | measured findings; has `_pending_` slots | **fill the pending slots** |

Two of those are unusual and you must respect them:

- **`tasks.md` is not a suggestion.** Its ordering is load-bearing. Several groups are ordered by a
  hard constraint (a measurement that must be taken before a design is chosen, a guard that must land
  before a behaviour moves), and the group headers say so. Do not reorder to suit convenience.
- **Every task has a `**Proof:**` line.** A task is not done when the code is written. It is done when
  the Proof command has been executed and its output recorded. A ticked box with no executed Proof is
  a false report and the reviewer must reject it.

**You do not edit the specs.** If implementation reveals a spec is wrong, that is a finding: stop,
record it, and escalate to the human. Editing a spec to match the code you wrote inverts the entire
point of the exercise.

---

## 1. The dependency graph — measured, not assumed

Every edge below is stated in a `tasks.md` header. Line numbers are given so you can verify rather
than trust.

```
rag-tree-repair                     (no blockers — the root)
├── rag-eval-harness                tasks.md:24  "Blocked by rag-tree-repair"
├── graph-lifecycle                 tasks.md:16  "Blocked by rag-tree-repair"
├── ingestion-chunking              tasks.md:15  "Blocked by rag-tree-repair and rag-eval-harness"
│   └── §7 only                     tasks.md:17  "additionally blocked by agentic-retrieval"
├── retrieval-sql                   tasks.md:15  "Blocked by rag-tree-repair and rag-eval-harness"
│   └── agentic-retrieval           tasks.md:15  "Blocked by retrieval-sql"
└── knowledge-stack                 tasks.md:15  "Blocked by rag-tree-repair and ingestion-chunking"
```

### The apparent cycle, and why it is not one

`ingestion-chunking §7` waits on `agentic-retrieval`. `agentic-retrieval §4` counts a token budget with
a tokenizer whose *choice* belongs to `ingestion-chunking`. That looks circular. It is not, and the
reason is a deliberate design decision you must not undo:

> `agentic-retrieval` ADR-004: the tokenizer is received as an **injected callable**. It is not
> imported. Task 4.3's Proof asserts that no tokenizer library is imported under `features/documents/`.

The injection seam is what breaks the cycle. If an implementer "simplifies" that by importing a
tokenizer directly, the two changes become genuinely circular and neither can land. **Treat task 4.3's
Proof as a structural invariant, not a style preference.**

The same pattern governs the reranker: `agentic-retrieval` *creates* the seam and removes nothing
(its task 2.4 Proof asserts `sentence_transformers` is **still present**), and `ingestion-chunking §7`
removes the dependency afterward. An implementer who "helpfully" drops torch in `agentic-retrieval`
breaks the ordering contract recorded in both changes.

### A second, invisible ordering edge — on `archive`, not on merge

**Measured 2026-09-10.** Both MODIFIED targets — `hierarchical-document-chunking` and
`legal-corpus-retrieval` — **do not exist in `openspec/specs/`**. They live only inside archived
changes. `rag-tree-repair` task group 5 ("Restore the specification baseline") is what puts them back.

Consequence, and it is easy to miss:

> **`rag-tree-repair` must be *archived*, not merely merged, before `ingestion-chunking` or
> `retrieval-sql` can be archived.** A `## MODIFIED Requirements` block that targets a requirement
> absent from the live baseline has nothing to modify.

`openspec validate --strict` passes all seven changes today and **will never warn about this** — it
validates a change's internal grammar, not whether its delta has a target to apply to. The failure
surfaces at `openspec archive` time, after the code has already merged.

This edge constrains the *archive* step only. The code ordering in §2 is unaffected.

---

## 2. The wave schedule

```
W0  rag-tree-repair                                   SOLO — everything waits
W1  rag-eval-harness        ∥  graph-lifecycle        parallel, zero file overlap
W2  ingestion-chunking §1-6,§8  ∥  retrieval-sql      parallel, TWO shared resources (§4)
W3  agentic-retrieval                                 SOLO
W4  ingestion-chunking §7   (reopen the branch)       SOLO — the dependency drop
W5  knowledge-stack                                   SOLO
```

**W0 is genuinely serial and there is no way around it.** The tree does not import today — a rename
from `document_processing/` to `docling/` is half-finished in the working tree. Until `rag-tree-repair`
lands, `uv run pytest` cannot collect and every other change's baseline task is uncollectable. Do not
start W1 on a promise; start it on a green import.

**W1 is clean parallelism.** `rag-eval-harness` builds a new package under `src/app/shared/evaluation/`
and its tests. `graph-lifecycle` works in `src/app/lifecycle/`, `shared/rag/graphiti/registry.py`,
`open_deep_search/graph.py`, and `features/documents/service.py`. The owned sets are disjoint —
verified by inspection of both Impact sections.

**W2 is parallel with two named collisions.** See §4. Do not start it until you have read that section.

**W4 exists because a branch reopens.** `ingestion-chunking` merges at the end of W2 with group 7
unticked, and reopens after `agentic-retrieval` lands. This is unusual and it is correct — the
alternative puts a retrieval-quality decision inside a dependency change. Record the reopening
explicitly in the ledger so nobody reads the unticked group as abandoned.

### Where you may compress the schedule, and the exact cost

`knowledge-stack` (W5) and `agentic-retrieval` (W3) have *nearly* disjoint file sets — the overlap is
that `knowledge-stack` task 4.2 adds a retrieval branch through `repository.py`/`service.py` while
`agentic-retrieval` works in `retrieval_kb/` and `rag.py`. Textually disjoint; behaviourally coupled,
because both change what retrieval returns.

You may run them in parallel **only if** you accept that a retrieval regression appearing at the end of
W3 cannot be attributed to one of them without bisecting. Given that `rag-eval-harness` exists
specifically to make retrieval quality measurable, throwing away attribution to save one wave is a poor
trade. **Default: serial.** Compress only under explicit human instruction.

---

## 3. The three-layer non-overlap mechanism

One layer is not enough. Declarations get ignored, and physical isolation does not stop two agents
declaring the same file. Use all three.

### Layer 1 — physical: one worktree per change

```bash
git worktree add ../lcfp-<change-id> -b impl/<change-id> main
```

An implementer works **only** inside its own worktree and cannot physically write into another's
checkout. Verified: the repo currently has a single worktree at the primary path, so the tree is clean
for this.

The orchestrator creates worktrees; implementers never do. An implementer that finds itself outside its
assigned worktree path must stop and report, not `cd`.

```bash
git worktree list                       # orchestrator's ledger check
git worktree remove ../lcfp-<change-id> # after merge, by the integrator only
```

### Layer 2 — declarative: the file-ownership ledger

The orchestrator maintains `docs/relay/ledger.md`. Before a wave starts, every change in that wave has
a declared owned-path set. **The orchestrator refuses to schedule two changes in one wave whose owned
sets intersect**, except for the two managed resources in §4.

Declared sets, derived from each proposal's Impact section:

| Change | Owned paths |
|---|---|
| `rag-tree-repair` | `src/app/shared/rag/docling/**`, `src/app/shared/rag/langextract/langextract_to_graph.py`, `src/app/features/documents/**` (imports only), `src/app/shared/langgraph_layer/ingestion_kb/nodes.py`, `src/app/utils/embedding.py`, `src/app/examples/policy_examples.py`, `src/alembic/env.py`, `src/alembic/versions/0013_*`, `tests/unit/shared/rag/**`, `openspec/specs/**` |
| `rag-eval-harness` | `src/app/shared/evaluation/**` (new), `tests/unit/shared/evaluation/**`, `tests/property/**`, the golden-set data file |
| `graph-lifecycle` | `src/app/lifecycle/**`, `src/app/shared/langgraph_layer/checkpointer.py`, `src/app/shared/langgraph_layer/open_deep_search/graph.py`, `src/app/shared/rag/graphiti/registry.py`, `src/app/features/documents/service.py`, the Celery worker bootstrap module |
| `ingestion-chunking` | `src/app/features/documents/chunking.py`, `src/app/shared/rag/docling/**` chunker, **`model.py` (column definitions)**, **`src/alembic/versions/` (holds the token)**; §7 adds `pyproject.toml`, `uv.lock`, `retrieval_kb/reranker.py` |
| `retrieval-sql` | `src/app/features/documents/repository.py`, `fusion.py`, `constants.py`, `service.py`, `src/app/shared/langgraph_layer/retrieval_kb/nodes.py`, `src/app/examples/policy_examples.py`, **`model.py` (`__table_args__` only)**, **an index-only migration** |
| `agentic-retrieval` | `src/app/shared/langgraph_layer/retrieval_kb/**`, `src/app/features/documents/rag.py` |
| `knowledge-stack` | `src/app/shared/rag/langextract/**`, `src/app/shared/rag/pageindex/**` (deleted), `src/app/shared/rag/__init__.py`, `src/app/shared/rag/graphiti/write_clause_episodes.py`, `src/app/lifecycle/lifespan.py`, `src/app/config/settings.py`, `docs-site/configuration/environment-variables.mdx`, `src/app/features/documents/repository.py` + `service.py` (navigation branch only), conditionally `model.py` + a migration |

**A change's first task is to confirm its declared set is complete.** If implementation needs a file
outside the declaration, that is not a small thing to do quietly — it is a ledger amendment that goes
through the orchestrator, because the orchestrator may have scheduled another change against the old
declaration.

### Layer 3 — detective: the scope diff

Before any branch is reviewed, the implementer runs and reports:

```bash
git diff --name-only main...HEAD
```

The reviewer compares that list against the declared set. **Any file outside the declaration is a scope
violation and blocks the merge**, regardless of whether the edit was correct. This catches the case
Layers 1 and 2 cannot: an implementer editing a file it genuinely needed but nobody scheduled around.

---

## 4. The two managed resources in W2

`ingestion-chunking` and `retrieval-sql` run in parallel and share exactly two things. Both are handled
by a token, held by `ingestion-chunking`, because it is the change with the real schema work.

### Resource A — the Alembic migration head

Migrations live at `src/alembic/versions/`. **Measured 2026-09-10: head on disk is
`0017_scope_statute_identity_index.py`.**

A migration chain is linear: each revision names exactly one `down_revision`. Two agents both appending
to the head produce two revisions claiming the same parent — a branched chain that Alembic will refuse
to upgrade with a "multiple heads" error, discovered at deploy time rather than at write time.

**Protocol:**

1. `ingestion-chunking` holds the token. It writes its identity migration with `down_revision` taken
   from the value **`rag-tree-repair` recorded** — read from that change's `baseline.md`, never guessed
   and never read from a stale memory.
2. `retrieval-sql` writes **no migration file** while the token is held. Its DDL is index and extension
   only, and its proposal states so explicitly: *"No columns are added — which is what makes a rebase
   against `ingestion-chunking`'s identity migration mechanical rather than a merge conflict."*
3. After `ingestion-chunking` merges, `retrieval-sql` rebases and *then* writes its migration against
   the new head.
4. The integrator verifies with `uv run alembic heads` — exactly one head, always.

> **Landmine, from prior measurement:** `src/alembic/env.py` has a defect that breaks every migration
> command until `rag-tree-repair` fixes it (that change owns `alembic/env.py`). Do not conclude from a
> failing `alembic` command in W1 that a migration is wrong; conclude that W0 is incomplete.

### Resource B — `src/app/features/documents/model.py`

Both changes need it, for genuinely different regions:

- `ingestion-chunking` adds `document_version` and `locus` columns to `UnifiedChunk` and changes the
  uniqueness constraint.
- `retrieval-sql` needs its index DDL reflected in `__table_args__` — it cannot live only in the
  migration, because `alembic autogenerate` compares model metadata against the database and would
  propose dropping an index the model does not declare.

**Protocol:** `ingestion-chunking` holds the file. `retrieval-sql` sequences its `model.py` edit as the
**last** task on its branch, performed after `ingestion-chunking` has merged and its own branch has
rebased. Until then it works entirely in `repository.py`, `fusion.py`, `constants.py`, and
`retrieval_kb/nodes.py`, which are exclusively its own.

If this ordering proves impractical, the fallback is to **run W2 serially** — `ingestion-chunking`, then
`retrieval-sql`. That costs one wave and removes the collision entirely. It is the right call the moment
the token protocol starts generating conflicts, and choosing it is not a failure.

---

## 5. The per-change loop

This is the methodology. Every change runs through it identically.

```
  ORCHESTRATOR                IMPLEMENTER            REVIEWER            VERIFIER          INTEGRATOR
       │                           │                    │                   │                  │
  1. create worktree ──────────►   │                    │                   │                  │
  2. hand dossier ─────────────►   │                    │                   │                  │
       │                    3. baseline (§1 tasks)      │                   │                  │
       │                    4. measurement gates        │                   │                  │
       │  ◄── re-scope report if a gate flips           │                   │                  │
       │                    5. implement group by group │                   │                  │
       │                    6. Proof each task          │                   │                  │
       │                    7. fill review.md pendings  │                   │                  │
       │                    8. scope diff ──────────►   │                   │                  │
       │                           │             9. coderabbit --agent      │                  │
       │                           │  ◄── findings ─────┤                   │                  │
       │                   10. fix + re-review ────►    │                   │                  │
       │                           │             11. CLEAN ────────────►    │                  │
       │                           │                    │           12. gates + Proof re-exec  │
       │  ◄──────────────────── RED verdict ────────────┴───────────┤                          │
       │                           │                    │      GREEN ──────────────────────►   │
       │                           │                    │                   │        13. rebase, merge
       │                           │                    │                   │        14. openspec archive
       │  ◄────────────────────────────────────────────────────── merged sha, ledger update ───┤
```

**Steps 3 and 4 are where changes get re-scoped, and that is by design.** Several changes open with a
measurement whose answer determines what the change *is*:

| Change | Task | What it decides |
|---|---|---|
| `retrieval-sql` | 1.2 | If the `bm25` access method is absent, `model.py:114` and `repository.py:418` are both wrong and **the whole change re-scopes**. This is a stop-gate — halt and escalate. |
| `retrieval-sql` | 1.3 | `EXPLAIN (ANALYZE, BUFFERS)` before anything moves. If the CTE-materialisation diagnosis is wrong, groups 3–4 shrink to determinism and tuning fixes. The change survives being wrong; the plan says so. |
| `knowledge-stack` | 1.2 | Whether the parser's structural tree is persisted. *No* makes group 2 mandatory and converts the change from wiring into a storage build. |
| `ingestion-chunking` | 1.3 | The `chunks` row count. Zero makes the generated-column rewrite free. |
| `graph-lifecycle` | 1.3 | A landmine check on `checkpointer.py`. |

**An implementer that hits one of these reports the answer to the orchestrator before proceeding.** It
does not pick the convenient branch and carry on. The re-scope is the deliverable at that moment.

---

## 6. CodeRabbit CLI — the review leg

**Measured 2026-09-10:** `coderabbit` v0.7.6 at `/home/harmeet/.local/bin/coderabbit` (alias `cr`),
authenticated as `Harmeet10000`. `.coderabbit.yaml` validates against the current schema.

The repo config already does a lot of work for you: `profile: assertive`, `auto_incremental_review`,
ruff and ast-grep enabled with `rule_dirs: [.ast-grep/rules]`, and **per-path review instructions** for
`shared/langgraph_layer/**` (graph state must be `TypedDict` with `total=False`; state reads use
`.get()` with documented fallbacks; checkpointer resume paths hydrate before logic nodes) — which is
directly relevant to `graph-lifecycle` and `agentic-retrieval`.

### The command

```bash
coderabbit review --agent --base main --include-untracked -c CLAUDE.md
coderabbit review findings          # re-read without spending another review
```

Each flag earns its place:

- `--agent` — structured findings instead of prose. This is the machine-consumable mode.
- `--base main` — compares the whole branch against main, not just the last commit. On a worktree
  branch this is the correct comparison.
- `--include-untracked` — **essential here.** `rag-eval-harness` creates an entire new package under
  `src/app/shared/evaluation/`. Without this flag those files are invisible to the review.
- `-c CLAUDE.md` — feeds the project's own rules in as review instructions, so findings are measured
  against this repo's conventions rather than generic Python style.

Add `--light` only when re-reviewing a small fix; never for a change's first review.

### Note on scope

`.coderabbit.yaml`'s `path_filters` exclude `openspec/changes/**/archive/**` but **not**
`openspec/changes/**`. CodeRabbit will therefore review the spec markdown alongside the code. That is
usually useful — but findings against spec files are **not actionable**, because the specs are frozen
(§0). Record them, do not act on them. If the noise becomes a problem, scope the review with
`--dir src`, accepting that tests then go unreviewed.

### The fix loop

1. Review → findings.
2. **Triage before fixing.** Not every finding is right. A finding that contradicts a recorded ADR is
   answered by citing the ADR, not by changing the code. The reviewer records the citation.
3. Fix what survives triage.
4. Re-review.
5. Commit the fixes separately, following the repo's established convention — measured in git log:
   `fix(result): address CodeRabbit review findings`, `fix(context): ...`, `fix(policy): ...`

**The reviewer never edits source.** It produces findings and a verdict. The implementer fixes. A
reviewer that fixes its own findings has destroyed the evidence and can no longer tell you whether the
fix worked.

---

## 7. Git workflow

### Branches

```
main
└── impl/<change-id>        one per change, in its own worktree
```

Commit messages follow the repo's conventional-commit style, measured from git log:
`feat(scope):`, `fix(scope):`, `refactor:`, `docs(scope):`, `ci(scope):`, `chore(deps):`.

**One commit per task group**, not one per change and not one per file. The task groups are the review
units — they are ordered by constraint, so they are exactly the granularity at which someone bisecting
would want to land. Subject line names the group:

```
feat(retrieval): collapse hybrid search to one fused path

Implements retrieval-sql task group 3. legal_rrf_search deleted;
the three branch methods now share the fusion helper.

Proof 3.1: <command> → <result>
Proof 3.2: <command> → <result>
```

Put the executed Proofs in the commit body. That is where they survive; a Proof recorded only in a
transcript is lost at the next context boundary.

### Merging

The integrator, and only the integrator:

```bash
git -C <worktree> rebase main          # rebase, do not merge main in
uv run pytest -q                       # gates must survive the rebase
git checkout main
git merge --no-ff impl/<change-id>     # --no-ff keeps the change legible as a unit
git tag impl/<change-id>-done
openspec archive <change-id> -y        # moves the change into openspec/changes/archive/
git worktree remove ../lcfp-<change-id>
```

`--no-ff` matters: a fast-forward merge dissolves the change into main's history, and these changes are
the units the whole plan is organised around.

**`openspec archive` is the closing act and it has teeth** — it applies the change's spec deltas to
`openspec/specs/`, which is the live baseline. A `## MODIFIED Requirements` block **replaces its
requirement wholesale**: any original scenario the block did not reproduce is silently deleted, and
`validate --strict` cannot detect the loss. Two changes carry MODIFIED blocks —
`ingestion-chunking` (`hierarchical-document-chunking`) and `retrieval-sql` (`legal-corpus-retrieval`).
**Before archiving either, diff the MODIFIED block's scenarios against the archived original's.** If a
scenario is missing, stop; archiving destroys it.

### Never

- Never force-push a branch another agent has read.
- Never merge main *into* a change branch — rebase, so the change stays one legible unit.
- Never `git add -A` from a worktree root; stage the declared files.
- Never commit `graphify-out/` churn (hooks dirty it every turn; it is path-filtered from review but
  not from git).

---

## 8. Gates — and why "green" is the wrong target

```bash
uv lock --check                  # FIRST, always — see the landmine below
uv run ruff format --check src/
uv run ruff check --no-cache src/
uv run ty check src/
uv run pytest -q
ast-grep scan src/
```

Never bare `ruff` or `ty` — `uv run` always, or you are checking a different environment than the
project's.

**The gates are red at baseline and that is not your fault.** Measured 2026-09-10 — and note this
supersedes an older "~12 websocket fixture-drift failures" figure that is now **stale**; do not use it:

> `uv run pytest` **never completes**. It aborts with **14 collection errors**, every one of them the
> same `ModuleNotFoundError: No module named 'app.shared.rag.document_processing'`. The residual
> failures behind that are 4 celery dispatch, 1 graph resilience, and 2 celery registration errors —
> and 6 of those 7 are the *same* import error again.

That single missing module is the half-finished rename, and it is exactly what `rag-tree-repair`
exists to fix. Until W0 lands, **no other change can even collect a test baseline** — which is the
concrete reason W0 is serial rather than a formality.

The correct criterion for every change except `rag-tree-repair` is:

> **no worse than the baseline recorded in that change's own task 1.1.**

This is why every change opens with a baseline-capture task writing to its own `baseline.md`. A verifier
comparing against a remembered number rather than that file will produce a wrong verdict.

Two specific traps, both previously measured:

- **`uv sync` uninstalls the test toolchain.** The test dependencies sit outside
  `default-groups = ["dev"]`, so a bare `uv sync` removes `pytest-asyncio` and the suite collapses in a
  way that looks like a code failure. Run `uv lock --check` first; it usually shows no sync is needed.
- **Fixing a shadowed import can make the type-error count go UP.** It did before: 13 `# ty: ignore`
  comments turned dead and became findings. A rising count after a genuine fix is not automatically a
  regression — read the findings.

---

## 9. Proof mechanics — matching the probe to the edge

Many tasks prove a *negative* with `rg '<string>' <paths>; test $? -eq 1`. Two blind spots, both hit
before, both live in this cluster:

- **`rg` on a string cannot see a symbol import.** `from x import Y` then bare `Y(...)` does not contain
  the module name. Grep for the symbol as well as the module.
- **`python -c "import ..."` cannot see a `TYPE_CHECKING` import.** This is not hypothetical here:
  `knowledge-stack`'s entire premise is that `graphiti/write_clause_episodes.py:42` imports the
  extraction types under `if TYPE_CHECKING:` at line 40 — a type-level edge with **no runtime edge
  beneath it**. A liveness probe based on importing the module reports that edge as present when the
  interpreter never traverses it. That change's task 5.3 Proof asserts the guard no longer covers the
  import, which is the right shape.

**Never use a test-process exit code as a Proof of test outcome.** Compare summary pass/fail counts
against the recorded baseline. A suite that fails to collect exits non-zero exactly like a suite with a
failing assertion, and they mean opposite things.

---

## 10. Definition of done, per change

A change is done when **all** of these hold. The verifier checks every one; the integrator merges only
on a full set.

1. Every box in `tasks.md` is ticked, and **every Proof was executed** with its output recorded in the
   commit body or `review.md`.
2. Every `_pending_` slot in `review.md` is filled with a measured value — including the ones that
   record a *negative* or *absent* result. An absent improvement is a finding, not a failure.
3. `openspec validate <change-id> --strict` exits 0.
4. Gates are at or better than the change's own `baseline.md`.
5. `coderabbit review --agent --base main` returns no unresolved finding above the agreed threshold;
   every dismissed finding has a written reason, and any dismissal citing an ADR names it.
6. `git diff --name-only main...HEAD` is a subset of the declared owned-path set.
7. `uv run alembic heads` shows exactly one head.
8. The MODIFIED-block scenario diff passes (changes carrying one: `ingestion-chunking`,
   `retrieval-sql`).

---

## 11. Escalate to the human, do not decide

Stop and ask when:

- A stop-gate measurement flips a change's scope (`retrieval-sql` 1.2, `knowledge-stack` 1.2).
- A spec requirement appears wrong or unimplementable. **Never edit the spec to match the code.**
- Two changes need the same file and it is not one of the two managed resources in §4.
- A CodeRabbit finding contradicts a recorded ADR and the ADR looks wrong.
- The gates get *worse* and the cause is not obvious within one investigation.
- Anything requires touching the live database beyond what a change's tasks authorise. The standing
  ruling is permissive — the database has zero data and zero users, so `CREATE DATABASE`, seeding,
  `EXPLAIN` capture and `alembic upgrade` are all authorised — but it is a ruling about *this* database
  in *this* state, and it is not a general licence.

One standing caution about the database: a prior note records the live instance as being at alembic
`0004`, while disk carries migrations through `0017` and `0014` is literally
*"create the five phantom relations"*. **That discrepancy is unverified.** Settle it with
`alembic current` against the live instance before any change relies on the live schema — do not
propagate either number as fact.
