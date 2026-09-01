# Handoff — implementing `error-handling-foundation`

You are implementing an OpenSpec change whose planning artifacts define a complete
repository migration program. All six artifacts exist and validate. Work from
`tasks.md`; when a task explicitly corrects an artifact, update that artifact and
record the completed correction rather than preserving stale text.

**State as of 2026-09-01: sections 1–11 have landed on `main`; 97 of 141 tasks were
complete before section 12 began.** Verify the live count yourself with
`grep -c '^- \[x\]' tasks.md` before trusting this line — it is a snapshot, and the
count changes as you work. Every completed task carries a `> **DONE:**` block naming
the sites it touched; keep that convention.

**Section 9 is new** and was added after implementation had already started, when the
owner brought five previously-excluded directories into scope. It is not blocked by
anything you have done. Read design **D20** before starting it.

Repo: `Harmeet10000/AgentNexus-LangChain-FastAPI` · default branch `main` ·
change dir: `openspec/changes/error-handling-foundation/`

---

## 1. Read these first, in this order

```bash
openspec status --change error-handling-foundation --json
openspec instructions apply --change error-handling-foundation --json
```

`status` confirms 6/6 artifacts and names the schema (`spec-gated`).
`instructions apply` returns `contextFiles` (artifact → real file paths), progress
counts, the parsed task list, and two fields you must honour:

- `context` — **required** prompt-level input. Project facts and conventions. Apply it.
- `operationGuidance` — advisory. Follow entries that are applicable and don't
  conflict with a controlling input (a resolved path, a CLI contract, a user choice).

Then read, in this order: `proposal.md` (why + scope) → `adrs.md` (the six decisions
you may not relitigate) → `design.md` (D1–D20, Migration Plan) → `review.md` (five
APPROVED passes; the **Method notes** at the end of passes 2, 3, 4 and 5 are binding on
how you measure) → `tasks.md` (your work).

Skip `specs/` on first read; consult a capability's `spec.md` when you implement the
tasks that cite it.

## 2. Which workflow

Use the project's OpenSpec skills. They are in `.opencode/skills/`:

| Phase | Skill | When |
|---|---|---|
| Implement | `openspec-apply-change` | working through `tasks.md` |
| Self-check | `openspec-verify-change` | before opening each PR |
| Archive | `openspec-archive-change` | **only after the last PR merges to `main`** |

`openspec-continue-change` is for unfinished *planning* — you will not need it.
Do **not** use `openspec-new-change` or `openspec-propose`; planning is done.

There is no store in play, so never pass `--store`.

## 3. Git: split the remainder, don't ship 97 tasks in one PR

97 tasks in one PR makes bot review useless. Split on the dependency seams the task
file marks. **Never commit to `main`.**

### What has already landed

Six commits on `feature/error-handling-foundation`, none merged to `main` yet:

| Commit | Covers |
|---|---|
| `d5854a0` `fix(repositories): roll back before returning Failure…` | section 1 — 9 repositories |
| `0ca39ea`, `789b331` `chore(openspec): …` | ticks + `DONE:` blocks for 1.1–1.13 |
| `58422c1` `feat(result): shared spine + renderer` | sections 2 and 4, **plus task 3.1** |
| `853fe25` `chore(openspec): add planning artifacts…` | the six artifacts + a no-match fixture |
| `7074738` `refactor(db): replace DB_ERROR literals…` | **tasks 7.1 and 7.2 in code — still unticked** |

Two warnings from that history. `7074738` changed nine repositories and no checkbox, so
`tasks.md` understates progress: verify 7.1/7.2 against the code before redoing them, and
tick them with a `DONE:` block once you have. And sections 2, 4 and task 3.1 arrived in
one commit, so the original PR 1 / 2 / 3 split no longer maps onto the history — do not
try to retrofit it. Open **one PR for the branch as it stands** and split only what
remains:

| # | Branch | Sections | Tasks | Depends on |
|---|---|---|---|---|
| A | `feature/error-handling-foundation` (exists) | 1, 2, 4, 3.1, 7.1–7.2 | 28 | nothing — merge first |
| B | `feat/enforcement-gates` | 3.2–3.11 | 10 | A |
| C | `feat/subscriptions-exemplar` | 5 | 8 | B |
| D | `refactor/error-classification-and-docs` | 6, 7.3–7.7, 8 | 21 | A (C not required) |
| E | `chore/scope-exemptions-and-examples` | 9 | 16 | D — task 9.3 must ship in the same commit as 7.5 |

Section 10 (verification) runs on **every** PR. Section 11 (handoff notes) goes in E.

Task **9.11** is the exception to the table: it is four `# noqa: BLE001` sites in
`features/subscriptions/service.py`, so do it in C with the rest of the exemplar and
tick it there.

E is small and mostly configuration, but do not fold it into D — task 9.1 removes
8 lint suppressions and the findings that surface are the point of the PR. Mixed into a
23-task refactor, a reviewer cannot tell an uncovered defect from a new one.

**Two things about PR A specifically.** Task 3.1's regex fix to
`.ast-grep/rules/no-match-on-result.yml` is *correct* — verified both ways, see §6 — but
check that its fixture pair is actually committed; ADR-005 requires one with every rule
this change touches, and a corrected rule is the exact case that ADR was written for.
That is task 3.2. Do not revert the rule, it works.

And **re-check `tasks.md`'s section list before you trust it.** `853fe25` committed the
planning artifacts, which fixed the original problem — the branch now carries its
proposal, ADRs and spec deltas rather than bare checkboxes. But the `tasks.md` in that
commit was the **79-task, 10-section** version: section 9, the renumbering, and tasks
10.7 and 11.5 had been lost from it while the specs kept all 45 requirements. Restored
since. `grep -n '^## ' tasks.md` must show `## 9. The five later-added directories`,
`## 10. Verification`, `## 11. Handoff`, and 97 total checkboxes. If it shows
`## 9. Verification`, you are on the narrowed copy and section 9's 16 tasks are missing.

Branch prefixes match the repo's history — `fix/`, `feat/`, `refactor/`, `chore/`.
Commit messages are conventional-commit scoped, e.g.
`fix(repositories): roll back before returning Failure from a caught DB exception`.

```bash
git checkout main && git pull
git checkout -b feat/enforcement-gates      # or the next branch in the table
# ... work, commit in logical units ...
git push -u origin feat/enforcement-gates
gh pr create --base main --fill
```

`gh pr create` picks up `.github/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md`.
Fill it honestly — tick only the boxes you actually did. Two notes on that template:
it says `mypy src`, but this project uses **`ty`**, not mypy; and it predates `uv`,
so use the commands in section 5 below, not the ones in its code block.

In the PR body, always include:
- which `tasks.md` sections this PR covers, and the task numbers
- the exact `openspec validate --strict` and gate output (paste it)
- for PR A: that section 1 is a standalone correctness fix closing live
  poisoned-commit paths, reviewable without the error redesign

## 4. Waiting for review comments

Two bots auto-review every PR in this repo — **`sourcery-ai`** and
**`greptile-apps`**. They comment within a few minutes; there is no human gate you
need to wait on unless the owner asks for one.

```bash
gh pr checks <n> --watch                 # CI (see §5) — wait for this to go green
gh pr view <n> --json reviews,comments   # bot reviews + summaries
gh api repos/Harmeet10000/AgentNexus-LangChain-FastAPI/pulls/<n>/comments \
  -q '.[] | "\(.path):\(.line) [\(.user.login)] \(.body)"'   # inline line comments
```

Handling them:

1. Fix what is a real defect. Commit on the same branch and push — the bots re-review.
2. Where a comment is wrong or conflicts with an ADR, **reply saying which ADR and
   why**, and do not change the code. Bot reviewers do not know the spec; several of
   this design's rules look wrong out of context. The two most likely:
   - a bot will call the flat-sibling duplication "code duplication that should be
     extracted to a base class" — refusing that is the point of **ADR-001**
   - a bot will suggest `match` on `Success`/`Failure` — **ADR-002** forbids it, with
     measured evidence
3. Re-run section 10 locally after every fix round. Do not push a fix without it.
4. Merge only when CI is green **and** every bot thread is either resolved or answered.

```bash
gh pr merge <n> --squash --delete-branch
```

Squash, so each PR is one commit on `main`. Then `git checkout main && git pull`
before starting the next branch.

## 5. Validating after implementing

Three layers. Run all three; they do not overlap as much as they look.

### (a) OpenSpec artifact validation

```bash
openspec validate error-handling-foundation --strict
openspec status --change error-handling-foundation --json   # task progress
```

`--strict` checks scenario formatting (`#### Scenario:` — exactly four hashes),
that every requirement has ≥1 scenario, and delta-block well-formedness. Run it
after any edit to `tasks.md` (including ticking checkboxes) and before every PR.

Tick `- [x]` as you finish each task. The apply phase parses those checkboxes — an
untracked task reads as undone.

### (b) The project's own checks — always `uv run`

```bash
uv lock --check                     # FIRST. See the warning below.
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/
uv run ty check src/ tests/
uv run pytest
ast-grep scan src/
```

**`uv lock --check` first, and do not run a bare `uv sync`.** The test dependencies
sit outside `default-groups`, so a bare `uv sync` uninstalls `pytest-asyncio` and
your suite stops collecting. If `uv lock --check` passes, you need no sync at all.
If you must sync, use CI's form: `uv sync --extra dev`.

**Note the widths.** `CLAUDE.md` says `src/`; **CI checks `src/ tests/`** for ruff
and `ty`. Use `src/ tests/` locally or you will pass locally and fail in CI.

**Measure the `ty` baseline before you start**, on a clean checkout. Do not trust a
count written in any document, including this one. Fixing one shadowed import can
make the error count *go up*, because `# ty: ignore` comments that were suppressing
real errors turn dead and become unused-ignore errors.

**`pytest` baseline: 103 pass, 12 pre-existing websocket fixture-drift failures**
owned by no change here. Verify that baseline yourself on `main` before your first
commit, then hold it: your PR must not grow the failure count. If CI is already red
on `main` from those 12, say so in the PR rather than trying to fix them.

Never reach a green check by adding `# noqa` or `# ty: ignore` — task 10.5 forbids it,
and it forbids adding an entry back to `per-file-ignores` for the same purpose.

### (c) CI (`.github/workflows/test.yml`, runs on every PR to `main`)

Spins up Postgres 16, Mongo 6 and Redis, then in order:
`uv sync --extra dev` → `ruff check src/ tests/` → `ruff format --check src/ tests/`
→ `ty check src/ tests/` → `alembic upgrade head` → `alembic heads` →
**`alembic check`** → `pytest tests/ --cov` → `pytest tests/ -m integration`.

Two things to know. `alembic check` fails on **model drift** — this change touches
repositories and error types, not ORM models, so it should stay green; if it goes
red, you changed a model and should not have. And `src/alembic/env.py` imports a
feature model at runtime, so deleting or moving a model breaks *every* alembic
command while no test in the suite runs alembic — CI is the only place that catches it.

`-m integration` runs tests the default run deselects, so a green local `pytest` does
not mean everything ran.

## 6. Gate discipline — ADR-005, non-negotiable

Seven new ast-grep rules and one fix. **A rule is not trusted until you have shown it
flags the construct it forbids and spares the nearest construct it permits.** Ship a
fixture pair with each. Register them in `sgconfig.yml`.

`no-match-on-result` **was** broken and task 3.1 has fixed it. Its regex matched only
the argument-less `Success()` form (`^(Success|Failure)\(\s*\)$`), so
`case Success(value):` — the exact thing it exists to catch — passed. 3.1 widened it to
`^(Success|Failure)\(`, which was then verified both ways against a probe: it flags
`case Success(value)`, `case Failure(err)`, `case Success()` and `case Failure()`, and
spares `case SubscriptionNotFoundError():` and `case SubscriptionConflictError(code=c):`.

**Re-measured from scratch after the fix: 0 violations in `src/`, 0 in `tests/`** —
and `rg 'case\s+(Success|Failure)\s*\('` over the same trees independently returns 0,
so this zero is real, not a rule looking for something nobody writes. Section 3 has no
existing violations to clean up; its remaining ten tasks are about the *new* rules.

What 3.1 still owes is task **3.2**: a **committed** fixture pair. The verification above was a throwaway
probe, so nothing in the repo stops a future edit from narrowing the regex again — which
is precisely the failure ADR-005 exists to prevent. Its severity is also `warning`, not
`error`, unlike three of the other four rules; decide deliberately which the new rules
get.

One coverage fact you can rely on, measured the same way: `ast-grep scan` reads **411 of
the 427** `.py` files under `src/ tests/`, and the 16 it skips are exactly the 16
zero-byte `__init__.py` files. There is no hidden path exclusion in `sgconfig.yml` —
unlike ruff, whose exclusions are the subject of the next paragraph.

**ADR-005 has a second form, and it is why section 9 exists.** A rule can also report
nothing because it was *pointed away from the code*. `pyproject.toml`'s
`per-file-ignores` disables `BLE001`, `E722`, `B904`, `TRY201`, `TRY300`, `TRY301`,
`TRY400` and `S112` for `src/app/examples/*.py` — so `ruff check src/app/examples/`
says "All checks passed!" while `ast-grep`, the only gate there with no per-path
ignore, reports **4 `error`-level violations** in the same files. Before you cite any
gate's clean run, read what it was configured to skip. Task 10.7 makes that an explicit
step; task 9.1 removes the entries.

Six shapes must NOT be flagged by your new rules. Verify each explicitly:

1. `middleware/global_exception_handler.py` — zero `except` blocks; dispatches by registration
2. an `except ImportError` capability flag
3. the three `raise TooManyRequestsException` sites in `features/crawler/router.py` — a boolean policy guard, not a rendered failure
4. `raise ValueError` inside a Pydantic validator (`config/settings.py:473`, `api/strict_envelope.py:26`) and `raise AttributeError` inside a PEP 562 module `__getattr__` (`src/database/__init__.py:37`) — the framework reads those, not project code
5. a broad `except` carrying a written reason after `# noqa: BLE001` — 55 of the repo's 62 such sites already do
6. a blind `except` that ends in a bare `raise` (`middleware/server_middleware.py:100`) — nothing was survived, and `BLE001` itself spares this shape

## 7. Four measurement rules from `review.md`

Seven errors in this change's own drafting came from skipping these. They are cheap.

1. **Before a count becomes a claim about a population, enumerate the population a
   second, structurally different way and reconcile.** `rg -c` summed over files is
   one query; iterating enumerated members is the other. Where they disagree, the
   total is usually right and the attribution invented. This one fired again in the
   fourth pass: a summed count said 6 documented `# noqa: BLE001` sites in `src/tasks/`
   and the enumeration showed 2 — and the two reconciling queries then agreed on 7
   reasonless sites repo-wide, which is the number in the spec.
2. **`ls` the path a plan says it will create, and read the module it says it will
   extend, before reasoning about its content.** Two review passes missed that
   `app/shared/result/` already held the "new" vocabulary. And match the probe to the
   edge kind: `rg <string>; test $? -eq 1` cannot see symbol imports, and
   `python -c "import x"` cannot see `TYPE_CHECKING` ones.
3. **Before citing a gate's clean run, read its exclusion list.** A working rule
   pointed away from the code produces the same zero as a broken rule. `per-file-ignores`,
   `sgconfig.yml`'s `ruleDirs`, and any rule-level path filter all count.
4. **You and the branch are two moving objects.** Never state a task total from memory
   or from another document — derive it twice. `grep -c '^- \[ \] '` counts only the
   *open* boxes; the total needs `grep -cE '^- \[( |x)\] '`, and a per-section sum is the
   reconciling second query. And a `DONE` block that says "partial" is a debt nothing
   collects: `--strict` cannot see it and `openspec status` counts the task complete, so
   grep the `DONE` blocks for "partial", "deferred" and "TODO" before opening the PR that
    contains them.

5. **Resolve `tasks.md` conflicts by union, never by choosing one side.** Keep the
   superset of sections and the union of every `- [x]` task line. A checkbox is a
   claim about repository state, not branch ownership. This rule is required because
   five historical branch copies held different completed subsets and no single copy
   held the union; choosing one side previously deleted section 9 while all specs
   continued to validate.

## 8. Archiving — last step, after the last merge

```bash
git checkout main && git pull
openspec instructions archive --change error-handling-foundation --json   # advisory
openspec archive error-handling-foundation
openspec validate --specs --strict
```

`archive` folds the delta specs into `openspec/specs/` and moves the change to
`openspec/changes/archive/`. Do **not** pass `--skip-specs` (this change has real
spec deltas) or `--no-validate`.

**One archive hazard specific to this change.** It contains a `## REMOVED` block for
`pattern-matching-standard`'s `match`-on-Result requirement, and a MODIFIED block
elsewhere. On archive, a MODIFIED block replaces its requirement **wholesale** — an
omitted scenario is silently deleted and `validate --strict` cannot detect it. Before
archiving, diff each MODIFIED block against the deployed spec and confirm every
scenario you meant to keep is present in the block.

Archive in its own commit on `main`, or a small `chore/archive-...` PR.

## 9. Scope boundary

**In scope:** `src/app/{features,connections,lifecycle,middleware,shared,utils}`, plus
the five directories added after the third review pass — `src/app/api`,
`src/app/config`, `src/app/examples`, `src/database`, `src/tasks`. Section 9's tasks
are all of them; design **D20** tabulates every site and its disposition. Most of that
work is removing an exemption or writing one down, not converting code.

Read D20 before you touch any of the five. Three raises there must stay exactly as
they are (`config/settings.py:473`, `api/strict_envelope.py:26`,
`src/database/__init__.py:37`) because a framework, not project code, reads them —
converting one of those would make an invalid setting validate successfully.

**Explicitly out of scope, by the owner's decision** — do not touch, do not report as
a coverage gap: `src/mcp_core` (19 modules, 23 raises, 10 `except`). `src/lynk` is
outside by nature: 24 `.go` files, zero `.py`.

Two tasks reach outside `features/subscriptions/` on purpose and are marked in
`tasks.md`: **2.6** fixes a kindless `AppError` in `ingestion/service.py:86` because
the renderer is `kind`'s first consumer, and **section 1** touches nine features'
repositories because staging the rollback across seventeen changes would leave the
defect open for the duration. Nothing else may reach outside its section's scope.

## 10. Stop and ask

- An ADR appears wrong. Say which, and why, with the measurement.
- A task's target does not exist or has moved.
- `ty` or `pytest` baseline differs materially from §5.
- Task 2.3 fails — if `ty` does **not** reject a hand-written `ClassVar` code string,
  the whole ClassVar design is unenforceable. Stop there; do not work around it.
- A bot review and an ADR conflict and you are unsure which governs.

## 11. What comes after (not yours unless asked)

Phase 1a — `shared/services/`, **before** the `crawler` feature change, because
`crawler/service.py:18` imports `search` from `tavily.py`. Three modules: `storage`
(21 raises), `tavily` (8), `mailer` (2, no external importer). `rate_limiter.py` is
excluded — it raises nothing.

Then 14 feature changes in the corrected order in `design.md`'s Phase 2.
`shared/crawler/` lands with `crawler`, and `shared/rag/`'s provider boundary lands
with `documents`. `utils/cache/` is already complete under task 7.5.

18 features exist: `subscriptions` is the exemplar; `audit`, `crawler`, `users`,
`ingestion`, `dunning`, `profile`, `plans`, `invoices`, `payments`, `webhooks`,
`agent_saul`, `credits`, `documents`, and `auth` are the 14 conversions; `chat` and
`search` are no-ops; `health` is classify-only. That is the complete
**18 = 1 + 14 + 2 + 1** arithmetic.
