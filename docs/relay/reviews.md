# Review ledger — the five openspec changes

Each change was authored by one agent and reviewed by a **fresh** agent that did not write it. `tasks.md` is gated
behind the review: the reviewer never writes it, and the author writes it only after remediating.

Validation baseline, `openspec validate --all` (measured 2026-08-18): **21 passed, 6 failed.** The 6 are the
`mintlify-documentation` change plus specs `cognee-v1-api`, `noqa-documentation`, `pattern-matching-standard`,
`transactional-outbox`, `typed-exception-handling`. Four of the five spec failures are a missing `## Purpose`
section, which **no change delta can repair** — see `findings-openspec-baseline.md`. No change may add a 7th.

| # | Change | Review | Verdict | Blocking | Remediation | `tasks.md` |
|---|---|---|---|---|---|---|
| 0 | `cleanup-foundation` | done, 35 claims audited (**7 wrong**) | **CHANGES REQUESTED** | 5 (F1–F5) + 5 non-blocking + **F11 (orchestrator)** | **remediated** — §11 absorbed, D-6 corrected, **F11 closed** (all six scenarios verbatim; my REMOVED advice refuted). Open: **ADR-3/4/5/6 cited but absent** (A23) | **written by orchestrator, 248 lines** after 3 agent crashes |
| 1 | `ingestion-pipeline-unification` | done, 340 lines | **CHANGES REQUESTED** | 8 (A–I) + 6 nits | **remediated** — 8/8 + 6/6 nits | **written, 368 lines** |
| 2 | `documents-unified-schema` | done, 343 lines | **CHANGES REQUESTED** | 7 (F1–F7) + 8 nits | **remediated** — 7/7 fixed, 8/8 nits | **written, 16 steps** |
| 3 | `agent-tools-unification` | done, 62 claims audited (44 exact) | **CHANGES REQUESTED** | 3 (F1–F3) + 5 material | **remediated** — F1–F8 + M1–M9, **F4 refuted** | **written, 281 lines** |
| 4 | `cognee-agent-memory` | done, 443 lines | **CHANGES REQUESTED** | 6 (B1–B6) + 7 nits | **remediated** — 6/6, 6 nits, **N4 refuted** | **written, 46 tasks** |

Every review lives at `openspec/changes/<change>/review.md`. Authors append a `## Author response` section to the
bottom of that same file — one line per finding, `fixed | refuted (+evidence) | deferred (+reason)` — so the audit
trail stays in one place and nothing is silently dropped.

---

## Orchestrator adjudications

Cross-change calls a single-change author cannot make. Recorded here because both reviewers surfaced findings whose
correct owner was a *different* change.

### A1 — change 4's **B1** is CONFIRMED, and it retires a claim of my own

Verified directly against the installed package, not delegated. `RelationalConfig`
(`.venv/.../cognee/infrastructure/databases/relational/config.py:12-23`) exposes **discrete fields only** — no DSN
field exists — and `cognee_client.py:91-101` already passes them with a working password, while the
`postgres_url` at `:111` sits in a dict the `else:` branch merely returns.

Consequences, both propagated:
- `findings-database.md` §2's row for `cognee_client.py:111` is **retracted**; full retraction written up as that
  file's new **§9**.
- **§8 consequence 3 drops from three URL flavours to two.** Change 0's `infrastructure-client-access` serves
  SQLAlchemy+asyncpg and plain libpq/psycopg only. Change 0's reviewer was mid-flight holding my incorrect
  three-flavour instruction and was sent the correction, with instructions to judge the change against the
  corrected fact and to report any surviving "three" as a finding.
- The **surviving** Cognee defect is worse than the retracted one: `:96`/`:100` read `POSTGRES_HOST` /
  `POSTGRES_DB_NAME` independently of `get_database_url()`, so Cognee can succeed against a *different database
  than the application*, silently and consistently. Assigned to change 4.

**Lesson, same family as D14.3:** a finding inherited from an earlier report is not evidence. §2's row was written
from a call site read at one line rather than across the `try`/`else` split, and two changes were then planned on
it. Re-verify a load-bearing claim at the point where it becomes load-bearing.

### A2 — change 4's **B2** (lost tool-exposure half) → **change 4 owns it**

The requirement originated in `cognee-saul-memory-migration`, which change 4 supersedes, so the harvest gap is
change 4's. It restores the retrieval contract in its own `saul-agent-memory` capability and adds a Coordination
point naming change 3's `agent-tool-registry` / `agent-tool-contract` as the exposure surface. Change 4 must not
write requirements into a capability change 3 owns.

This was the one suspicion I flagged to change 4's reviewer in advance, and it was confirmed — the harvest from the
superseded change had narrowed the requirement to a *when*-prefetch clause and dropped the tool half.

### A3 — change 4's **B4** (no worker/beat runtime) → **change 1 already owns it**

`dispositions.md` item 198.4 dispositions the missing `docker-compose.yml` worker/beat services (and
`Makefile:52`'s non-existent `celery_config`) **IN change 1**. Change 4 therefore records a Coordination point plus
a stated **Non-Goal** — the consolidation beat entry it registers is **inert until change 1 lands the worker** —
rather than duplicating the work. Duplicating it would put the same fix in two changes with no owner.

### A4 — change 2's **F3/F4** are on change 1's critical path

Decision **D15** makes change 2's ADR a gate on change 1, and F3 (`id` missing from both the required-writer and
defaulted lists → `NotNullViolation` for an implementer who follows the ADR and writes by raw SQL) and F4
(`markdown` / `summary` / `thread_id` have no home in a closed column set, yet change 1's pipeline persists them at
`ingestion_kb/nodes.py:496-499`) both make the ADR actively misleading. Both were ranked first in change 2's
remediation brief for that reason.

### A5 — change 2's **F1** is delegated back, not accepted

The reviewer's refutation of the "drift gate is red on three counts" claim is a spec-internal reasoning claim, so
change 2's remediator was told to verify the count itself against the gate rule and the two migration revisions,
and to **record a refutation with evidence** if it disagrees rather than silently complying. A reviewer's
confidence is not proof — one earlier agent's confident rejection of a locked user decision turned out to be wrong
in both of its stated reasons (D14.1).

---

## Still open and blocking `tasks.md` for other changes

- **F8 — the `bm25` access-method name.** ~~Change 2's F2 depends on it and it is still open.~~
  **CLOSED 2026-08-18 — see `findings-database.md` §10.** The user authorized the one `CREATE EXTENSION` statement;
  `pg_textsearch` 1.3.0 is installed and the real names are measured. Access method **`bm25`**; opclasses
  `text_bm25_ops` / `text_array_bm25_ops`. The repo's existing BM25 SQL is **already correct** — no rewrite. Two
  residual obligations: no `bm25` index exists anywhere, and because the two-argument `to_bm25query` overload takes
  the index name as a **literal SQL argument** (pinned at `search/constants.py:15`), the index name is part of the
  **query contract**. Change 0 must create both indexes **by exact name**.
- **Queue topology** — flagged by change 1 as a question that must be answered before its `tasks.md`.

---

## Orchestrator adjudications — change 3's review

Change 3's review audited **62** file:line claims and found 44 exact — the sharpest of the five. It also
independently reproduced the `ty` baseline of **46** with the 11/5/4/3 per-file split, and `validate --all` at
21/6 with an *identical* failing set (no 7th). Two of its findings are cross-change; one was urgent.

### A13 — finding **F1** is CONFIRMED: there are **four** result envelopes, and the spec to reuse already exists

Verified directly, not delegated. `ToolOutput` (`shared/langchain_layer/agents/tools/base.py:30`) is a genuine
fourth definition — `success`/`data`/`error`/`metadata`, `ok()`/`fail()` classmethods, and a `to_agent_string()`
returning `f"ERROR: {self.error}"`, which **is** the string-as-error anti-pattern the change exists to remove — with
**13 uses in `tools/shell.py`**.

The decisive part is not the count. The **deployed** `openspec/specs/typed-exception-handling/spec.md` already
governs this exact class, naming `ToolOutput.fail()` at `:219, :223, :227, :235, :239`. So change 3's **D-12**
rejected reuse of a spec that already covers its subject, and D-12's second rationale (that nothing else edits that
spec) is independently false because change 0 does. D-12 is reversed; change 3 writes a **MODIFIED** delta there.

Two consequences beyond change 3:
- `dispositions.md` **Up#10** is corrected from three competing definitions to four.
- Any gate of the form `rg -c "^class ToolResult"` 3→1 **passes while `ToolOutput` survives**, because `ToolOutput`
  does not match the pattern. The gate must count all four envelopes. Same family as D14.3: a mechanical-looking
  proof attracts less scrutiny than a prose one, and this one was wrong in a way that reads as green.

**Standing coordination hazard — `typed-exception-handling` now has three claimants.** Change 0 **MODIFIES** its
asyncpg requirement; change 1 **ADDS** four requirements (A12/finding D); change 3 now **MODIFIES** the
`ToolOutput` requirement. openspec requires a `MODIFIED` block to reproduce the original requirement including
**every scenario title verbatim**, and one author has already failed `--strict` on exactly this. Each remediator was
given its own lane explicitly and told not to touch the other two.

### A14 — finding **F2** is CONFIRMED in substance and **routed to change 2**; change 3 becomes a consumer

Change 3's `legal-corpus-retrieval` requirement *"Statute identity attributes are addressable and efficiently
retrievable"* mandates that the corpus carry the instrument name, the section reference and the year with an
index-served point lookup and a newest-applicable-year rule — but **change 3 ships no DDL**, and change 2 owns the
retrieval schema. Neither change knew about the other: change 3 wrote a schema requirement it cannot implement, and
change 2 was about to close its column set against it.

Routed **urgently**, because D15 makes change 2's ADR a gate on change 1, so a late column addition is expensive.
Change 2 must now accommodate **both** A4/finding F4's `markdown` / `summary` / `thread_id` and F2's three identity
attributes plus the index — together, or excluded explicitly with a reason and a named alternative home. Change 3
keeps the requirement and adds a Coordination point naming change 2's ADR as the provider.

**Evidence caveat, recorded because it matters for how much weight the finding carries.** The reviewer supported F2
by grepping change 2 for `act_name` / `section_ref` and finding zero. But change 3 deliberately wrote the
requirement at the *attribute* level and never named those columns, so the missing strings are not what proves the
gap. The gap is real; that particular evidence is weaker than stated, and change 2 owns the naming. Change 2's
remediator was told both halves.

### A15 — finding **F7**: the ruff gate is cross-change and currently hides regressions

Change 3's gate reads 125 → ≤123, but **123 is the count now**: change 0's `todo_temp.py` deletion already moved the
baseline 125 → 123 (`dispositions.md`, change 0's second row). A gate whose ceiling equals the current count admits
two new errors and still reports green. Re-baselined to **≤121** after change 0, with the dependency named. This is
the second gate in change 3 that passes without proving anything (see A13), and both were arithmetic rather than
prose — worth noting as a pattern in how these plans fail.

### A16 — **Q-A** is answered by the user: relocate, do not harvest

`shared/rag/rag_agent_advanced.py` **moves to `src/app/examples/`** (`CLAUDE.md` puts examples there). The user was
offered harvest-then-delete as the recommended option and **chose relocation instead**, on a preview that stated the
losses explicitly: the `f"Search error: {e!s}"` anti-pattern **survives** (quarantined, not fixed) and the
iterative-RAG prior art **stays unused**. Both are recorded as Non-Goals rather than quietly repaired — the decision
was made with the downsides visible, so absorbing them silently would misrepresent it. No harvest task is written.


---

## Orchestrator adjudications — change 1's review

Five of change 1's eight blocking findings are cross-change ownership collisions. Settled here; the remediator was
told to follow these and not re-litigate them.

### A6 — finding **G** is CONFIRMED, and it corrects `dispositions.md`, not just change 1

Verified directly. Re-ranking is **not missing**:

- `retrieval_kb/reranker.py:19` `class CrossEncoderReranker` — a real `sentence_transformers` `CrossEncoder`
  wrapper (`BAAI/bge-reranker-v2-m3`, falling back to `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- It is wired as a **graph edge**, not an optional extra: `retrieval_kb/graph.py:49` adds the node and `:60-61`
  wires `hybrid_postgres → reranker → context_grader`.
- `graph.py:35`'s `reranker: CrossEncoderReranker | None = None` **looks** like an off-by-default feature, but
  `nodes.py:203` resolves `reranker or CrossEncoderReranker()`, so the node **self-provisions**. Nothing injects
  one anywhere, and it still runs.
- A **second, independent** call path exists at `documents/service.py:426`, which constructs a fresh
  `CrossEncoderReranker()` **per call** — loading a cross-encoder model each invocation, against the class's own
  docstring warning that it is CPU-bound.

So `dispositions.md` item 195's "**add re-ranking** (genuinely missing)" was wrong and is corrected in place. The
work is **harvest + unify + fill one gap**: the only genuine gap is `search/service.py:161 hybrid_search`, which
fuses and hydrates but never re-ranks.

Second-order correction, also applied: `dispositions.md` item 176 said drop the direct dependency "if the only use
is token counting". The re-ranker needs `sentence_transformers` for real, so that half is **settled — it stays**;
item 176 narrows to `AutoTokenizer`/`transformers` only.

**This is the third time a plan declared greenfield over working code** (BM25 and RRF were the first two, corrected
by D5.1). The pattern is now frequent enough to be a standing rule: before any requirement says "add X", grep for
X's *edge wiring*, not just its symbol — and follow one layer past an `| None = None` signature, because the
default is often resolved downstream.

### A7 — finding **C**: change 2 wins. Missing extension **fails loudly**.

Change 1's `hybrid-retrieval-ranking:71-73,100-102` requires degrade-and-continue; change 2's
`document-retrieval-schema:83` requires fail-loudly. Same code path, contradictory contracts, and neither design
recorded the other. Ruling: **fail loudly**, for two reasons.

1. Change 0 creates all four extensions explicitly (D14.4). A missing extension at runtime therefore means the
   migration did not run — a **deployment error**, not a runtime condition to be absorbed.
2. Degrade-and-continue is the exact pattern that produced this repo's invisible-failure register. The outbox has
   been silently, permanently dead behind two warning lines (`findings-database.md` §8) precisely because a broad
   handler absorbed a structural failure. Change 2's own F2 is a *further* instance of the same thing.

Change 1 deletes the degrade-and-continue requirement **and** its duplicated fusion / single-source requirements,
and references change 2's `document-retrieval-schema`. Change 2 keeps ownership and **must not weaken** its
scenario to meet change 1 halfway — its in-flight remediator was told so directly.

### A8 — finding **A**: the fail-closed read site is **change 3's**, per D17

D17 is explicit: *"`get_saul_graph` failing closed matters MORE, not less (change 3 step 1). …
`features/agent_saul/dependencies.py:45` reading `app.state.langgraph_checkpointer` unguarded is the defect to fix,
and it is now the primary justification for that step."* The reviewer is right.

Split: **change 1 owns provisioning** the checkpointer (`dispositions.md` item 138 residue a, IN 1);
**change 3 owns the read-site 503**. Change 1 drops `langgraph-checkpointing:131-142` and adds a Coordination
point.

### A9 — finding **F**: upheld, and the escape hatch is the teardown asymmetry

`langgraph-checkpointing:72-83` ("application owns the pool, closes on shutdown") forces uncommenting
`lifespan.py:294-305`, which **D17 forbids** — that is one of the two blocks D17 names as deliberately commented,
and D17 further requires proofs there to be import-level and type-level only.

Change 1 therefore scopes its checkpointer work to what is provable **by construction**: add `psycopg[binary]`
(without it `AsyncPostgresSaver` cannot even import — `findings-database.md` §5), fix `from_conn_string` being an
`@asynccontextmanager` rather than a factory, and consume change 0's psycopg-flavoured URL. Lifespan wiring becomes
a **stated Non-Goal blocked by D17**.

The one live defect it *can* fix without uncommenting anything: shutdown is **asymmetric** — `lifespan.py:317`
calls `teardown_langgraph_checkpointer` while the setup block at `:294-305` is commented out. Making teardown
tolerate the never-provisioned state is a real fix and is import/type-provable.

### A10 — finding **B**: change 0 owns the URL accessor

`langgraph-checkpointing:144-159` duplicates change 0's `infrastructure-client-access:43,61,85`, which D14 already
concedes. Change 1 drops it and becomes a **consumer**. Note the corrected flavour count from **A1**: two, not
three, and the psycopg flavour is exactly the checkpointer's need — so change 1 is the reason that flavour exists.

### A11 — finding **I**: an unexecutable proof, same family as D14.3

`document-ingestion-pipeline:155-157` asserts a terminal status that lives in the `documents` table, which **does
not exist** (`findings-database.md` §4). Change 1 must either name the dependency on change 0's migration and
re-scope the proof to run after it, or replace it with an import/type-level proof. **A proof that cannot be
executed as written is not a proof** — D14.3's lesson, and this is its second occurrence.

### A12 — findings **D** and **E**: change 1 fixes both, no cross-change question

- **D** is a genuine technical error: `raise … from exc` sets `__cause__`; it does not preserve the exception
  *type*, so the remedy fails its own scenario. The fix must stay inside change 1's own **ADDED**
  `typed-exception-handling` requirements — change 0 MODIFIES the asyncpg requirement in that same shared
  capability, and the two must not collide.
- **E** is a disclosure failure: F8 is left **open** by D14.2 while change 1's Decision 6 reads as closed. It must
  move into Open Questions. F8 is in the batched user round below.

---

## Orchestrator adjudications — change 0's review

The most damaging of the five reviews: **35 code claims audited, 7 wrong**, plus 3 stale figures and a scope-count
error — and several of the false claims sit *inside* the section change 0 added in order to prevent false claims.
I re-verified three findings personally; all three were exact.

### A17 — **F6**: the "third URL flavour" is fictional, and the error was mine

My `findings-database.md` §8 claimed three URL flavours. There are **two**. Cognee takes a **discrete-field** config
object and has no DSN field at all (A1). I sent change 0's reviewer this correction while it was mid-review and asked
it to report any surviving "three" as a finding — it did, and found the fiction encoded in **three places: ADR-3,
D-6, and a spec scenario**.

**ADR-3 is wrong on its central count and is not salvageable as written** — rewritten around two flavours or
withdrawn. The remediator was told the premise was mine and is retracted, so it does not waste effort defending it.

**This is the value of correcting a brief mid-flight rather than waiting.** Had the correction arrived after the
review, the fiction would have been ratified by a clean review and then *archived into `openspec/specs/`* as the
deployed record.

### A18 — **F1** is CONFIRMED, and the reason it is blocking is the archive semantics

Verified directly: `auth/service.py:512` does `engine = create_async_engine(get_database_url())` with
`await engine.dispose()` in a `finally` at `:524` — an engine constructed and disposed **per call**, reachable from
`auth/router.py:269`. So `outbox-helper-extraction`'s MODIFIED delta asserting the engine-per-call helper "no longer
exists" is false.

Why it is blocking rather than cosmetic: openspec **archives deltas into `openspec/specs/` on deploy**. A false
MODIFIED delta does not merely mislead a reader — it becomes the deployed spec of record, and the next change reads
it as truth. It also contradicts change 0's own Non-Goal deferring that pool to change 1, and no Migration Plan step
touches auth.

### A19 — **F4(a)** is CONFIRMED but must not be over-corrected

`design.md:386-388` says the offline render aborts. Measured: `uv run alembic upgrade heads --sql` → **exit 0, 697
lines, one `COMMIT;`**. It does not abort.

**The weaker claim survives and is the one that matters:** the same run's stderr walks `0002 → 0003 → 0004`,
confirming that offline mode renders **from base** because it has no database to read `alembic_version` from. So the
true property is "renders from base regardless of live state". Every proof in every change that depends on offline
SQL being *incremental* is invalid; change 2's `tasks.md` already encodes the prohibition three times.

### A20 — **F9** is CONFIRMED and it retires a stale figure of mine

Measured: `uv run pytest --collect-only -q` → **90 tests collected, zero collection errors.** The "thirteen
collection errors" figure — which appeared in change 0's design **and** in my own `open-questions.md` — is withdrawn.
Corrected in both places.

Consequence the remediator must act on: the justification for asserting the `401` by direct probe rather than an
automated test has evaporated, so the choice is re-opened on real grounds. One genuine obstacle survives and was
handed over with it: the suite **fails its coverage gate — 22.16% against a required 80%** — so `pytest` exits
non-zero even when every test passes, which makes any task whose proof is "pytest is green" unexecutable today.

### A21 — **F3** is CONFIRMED and converges with my own F8 closure

The reviewer found that **no requirement in change 0 names any index**, independently reaching the conclusion I had
already drawn from F8 (see the F8 entry above). It is fatal for the same reason: the two-argument `to_bm25query`
overload takes the index name as a **literal SQL argument**, so an index of the correct shape under a different name
**silently matches nothing** — no error, zero rows. A conforming implementation of change 0 as written can therefore
leave BM25 exactly as broken as it is today.

Coordination, so change 0 and change 2 do not collide: `clauses_bm25_idx` has **four** readers —
`search/repository.py:356,361,362` and `ingestion_kb/nodes.py:751` calling `bm25_force_merge('clauses_bm25_idx')` —
and **change 2 owns retargeting that literal** onto the unified store. Change 0 creates the indexes by exact name and
names the coordination; it does not also retarget.

### A22 — **F2** is decidable, and the ruling keeps the fail-loudly line consistent with A7

Change 0's ADDED "fail loudly" requirement in `transactional-outbox` mandates work the change disclaims in three
places, ships **no implementing step**, and collides with the untouched, **accepted** `typed-exception-handling`
requirement that sanctions `except Exception` in the relay.

Ruling — exactly one of two coherent branches: **(a)** keep fail-loudly, ship the implementing step, and express the
conflict as a **MODIFIED** delta against that deployed relay requirement, extending change 0's existing lane in that
capability rather than adding a contradicting requirement elsewhere; or **(b)** drop the ADDED requirement and record
the silent-absorption gap as an explicit **Non-Goal**. What is forbidden is keeping a requirement with no
implementing step that contradicts an accepted spec — that is F4(c)'s unexecutable-proof defect wearing different
clothes. `design.md` **D-9**'s claim that "no delta is added there" is false while shipping a 96-line delta, and is
fixed on either branch.

Note this is the **second** fail-loudly adjudication (A7 was change 1 vs change 2 on a missing extension). The line
is consistent: a structural failure that means "the migration did not run" is a deployment error, not a runtime
condition to absorb.

### A23 — the most valuable finding is an absence: two missing ADRs

The reviewer judged ADR-1 and ADR-2 sufficient, ADR-3 unsalvageable — and then found that **no ADR exists for the two
decisions most likely to be got wrong**: (1) **index-name-as-query-contract**, and (2) **what the relay owes when a
relation is absent**. Both outlive the change, both are invisible when wrong, and neither was anybody's finding until
a reviewer asked what was *missing* rather than what was incorrect. Both are now required deliverables.

### A24 — `typed-exception-handling` now has **three** claimants; each was given an explicit lane

Change 0 **MODIFIES** its asyncpg requirement (and possibly its relay requirement under A22 branch (a)); change 1
**ADDS** four requirements (A12/finding D); change 3 **MODIFIES** its `ToolOutput` requirement (A13). openspec
requires a `MODIFIED` block to reproduce the original requirement including **every scenario title verbatim**, so
three agents editing sibling requirements in one capability will each pass `--strict` in isolation and still
conflict. This is an orchestrator-only failure mode — no single-change reviewer can see it — and all three
remediators were given their lane boundaries in writing.

---

## Orchestrator adjudications — cross-change checks no single reviewer could run

These came from diffing the five changes against **each other** and against the deployed specs. Every one of them is
invisible from inside a single change, because the evidence lives in a file the change does not contain. `openspec
doctor` confirms the tool offers no help here — it reports `References (none declared)` and has no cross-change
conflict detection at all.

### A25 — **F11**, found by the orchestrator: change 0's MODIFIED block silently drops a deployed scenario

The deployed requirement *"Database operations SHALL catch asyncpg.exceptions.PostgresError"* carries **six**
scenarios. Change 0's delta reproduces **five** — `#### Scenario: Reconciliation fetch failure catches PostgresError`
is simply absent.

Because `MODIFIED` replaces a requirement **wholesale on archive**, archiving change 0 would **delete that scenario
from the deployed spec** with no `## REMOVED` block, no Reason and no Migration. `validate --strict` passes, because
the delta is structurally well-formed — the defect is only visible by comparing against a file outside the change.
Same class as change 0's own F1: not a misleading document but a **false record written into `openspec/specs/` as
deployed truth**.

Routed to change 0's remediator as **F11** with two acceptable fixes: restore the scenario byte-verbatim, or remove
it on purpose as a disclosed `## REMOVED` with a Reason naming the reconciliation-module deletion. It is plausibly
intentional — change 0 does delete the reconciliation path — which is exactly why it has to be **stated rather than
implied**.

For contrast, **change 3's delta on the sibling requirement in the same capability is correct**: all six original
scenario titles reproduced verbatim, one appended.

### A26 — the three-claimant `typed-exception-handling` collision is **clean**; lane assignment held

Verified by reading all three deltas:

| Change | Operation | Requirement |
|---|---|---|
| 0 | `MODIFIED` | Database operations SHALL catch `asyncpg.exceptions.PostgresError` |
| 1 | `ADDED` ×4 | embedding-failure, retry-boundary named types, one typed transient failure, no whole-node wrapping |
| 3 | `MODIFIED` | Agent tools SHALL catch OS-level and library-specific exceptions |

**No title overlap.** Three agents edited one shared capability concurrently without conflict because each was given
an explicit lane in writing. Worth recording as the thing that *worked* — the risk was real (A24) and the mitigation
was cheap.

### A27 — change 4's renamed MODIFIED scenario titles are **acceptable**; the distinguishing test is disclosure

Change 4 self-reported a tension: three of its pre-existing `MODIFIED` blocks rename scenarios wholesale, which looks
like a violation of the verbatim-reproduction rule. Verified — the renames are real:

- `Store relationships` → `Relationship summaries are no longer stored in agent memory`
- `Search episodic memory` → `Recall is scoped to the caller's memory partition`
- `Process report after store` → `A write does not trigger consolidation`

**Ruling: this is fine, and change 4's judgment was correct.** Its own argument is decisive — two of those titles
*cannot* be kept verbatim without asserting the **opposite** of their new bodies. Keeping `Store relationships` above
a body specifying that relationships are no longer stored would produce a spec that contradicts itself, which is
worse than a renamed title. The requirement **titles** are preserved, and those are what identify the requirement
for replacement.

The rule's purpose is to prevent **silent** loss, not to freeze titles whose semantics are deliberately inverted. So
the test is not "did the title change" but **"is the change recoverable by a future reader"**:

- change 4: **yes** — it added a seven-row old→new mapping table to Decision 9 recording the replacement as intended.
- change 0 (A25): **no** — a scenario vanishes with no record of who removed it or why.

Identical-looking symptom, opposite verdicts, and the mapping table is the whole difference. Change 4 also found and
fixed a stale baseline of its own the reviewer missed (`16 passed / 6 failed of 22` → 21/6 of 27) and noted correctly
that **the pass count is not an invariant — only the failure count is.**

### A28 — the silent-fail trap is clean across all five changes

`grep -rn '^### Scenario:' openspec/changes/` → **0 hits**. Three-hashtag scenario headers, which openspec drops
**without any error**, appear nowhere. Changes 1, 2 and 4 each validate `--strict`, and `openspec validate --all`
holds at **21 passed / 6 failed** with the baseline failing set.


### A29 — change 0's remediation round 1 is **partial**, and F11 survived it (verified on disk, not reported)

Its remediator terminated on an API error — *"the response stopped arriving"* — immediately after emitting
*"Now let me write `tasks.md`."* Rather than trust either that message or its silence, I measured the tree.

**Landed.** `design.md` is 734 lines and has genuinely absorbed the §11 material: `clauses` ×14, `9f4a1b7c6d2e` ×9,
`scratch` ×6, `bm25` ×14, `diskann` ×8. Its `## Decisions` block runs **D-1 … D-9**, and **D-6 now reads "two URL
flavours"** — so the fictional third flavour from **A17** is gone at its most load-bearing site.

**Did not land, all four verified by direct measurement:**

1. **`tasks.md` is absent** — the only missing artifact across all five changes, and therefore the single thing
   holding planning open.
2. **F11 survived.** The delta reproduces **five** scenarios under
   `### Requirement: Database operations SHALL catch asyncpg.exceptions.PostgresError`
   (`specs/typed-exception-handling/spec.md:25,29,33,37,41`); the deployed spec carries **six**, and
   `openspec/specs/typed-exception-handling/spec.md:149` — `#### Scenario: Reconciliation fetch failure catches
   PostgresError` — is still the one dropped. Since this change deletes the reconciliation module, an explicit
   `## REMOVED Requirements` block with a Reason and Migration is the honest fix; verbatim reproduction is the other
   acceptable one. Silence is not.
3. **No `## Author response` heading exists in `review.md`.** All four other changes carry one.
4. **The two ADRs from A23 are still absent** — `rg '^#+ *ADR' design.md` → 0 hits.

**Method note, and it is the reusable part.** A crash notification carries a `<result>` field holding the agent's
last words, and those words describe an *intention* (`"Now let me write tasks.md"`), not an outcome. Both crashed
remediators — change 3's and change 0's — died at exactly that boundary. In change 3's case the work was in fact
complete and only the report was cut off; in change 0's it was genuinely incomplete. **The two cases are
indistinguishable from the notification alone**, so the disk is the only authority, and checking it converts a
resume-from-scratch into a four-item punch list handed to the agent.

### A30 — change 0's `tasks.md` written by the orchestrator after a **third** crash at the same boundary; F11 closed, and my own recommendation on it was **wrong**

The remediator crashed three times, each time emitting a variant of *"Now writing `tasks.md`."* and dying. Three
failures at an identical boundary is a pattern, not luck, so I stopped resuming and wrote the file myself from the
change's own `design.md` Migration Plan — **248 lines, 12 groups**, which is the twelve-step plan one-to-one with
executable Proofs. All five changes now carry a `tasks.md`; **planning is artifact-complete.**

**F11 is closed, and the remediator refuted my instruction — correctly.** I had told it that
`## REMOVED Requirements` with a Reason and Migration was "probably the honest fix." **That would have been
destructive.** REMOVED operates at **requirement** granularity, so naming
*Database operations SHALL catch asyncpg.exceptions.PostgresError* there would have retired the **entire asyncpg
guarantee**, including the five scenarios that must survive. **There is no scenario-level REMOVED.** It reproduced all
six scenarios verbatim in accepted order, kept the stale one deliberately, and routed its retirement to the
spec-hygiene pass that the four `## Purpose` failures also need. Its intermediate attempt is instructive too:
disclosing the omission *in prose inside the requirement body* was honest but **still deleted the scenario on
archive**, because prose is not a mechanism.

It also **improved my own correction.** I had withdrawn the "thirteen collection errors" figure as stale. It is
sharper than that: the thirteen are **real**, but they are **setup** errors — all thirteen `fixture 'client' not
found`, in `tests/integration/test_health.py` and `test_api_deprecation.py`. Right magnitude, **wrong kind**, and the
kind was what misled: it implied a broken test tree rather than one missing fixture. That re-decided the 401 question
on measured grounds and reached the same answer (direct probe) for better reasons.

**Cross-change defect found while matching the house format, and fixed.** Change 2's `tasks.md` rule 3 carried the
claim that a from-base render *"cannot even complete, because `9f4a1b7c6d2e:103` alters the phantom `clauses`
relation."* Change 0's own D14.3 had already refuted this by measurement — `alembic upgrade heads --sql` exits **0**
and emits **697 lines** — because offline rendering emits DDL as *text* and never executes it. So change 2 reached the
**right rule on a false reason**. Corrected in place, with the refutation recorded rather than silently overwritten:
a correct rule resting on a false premise is a live hazard, because anyone who tests the premise, finds it wrong, and
discards the rule loses a constraint that genuinely binds.

**Still open, and it is the one gap in change 0:** `design.md` cites **ADR-3, ADR-4, ADR-5 and ADR-6** but contains
**zero `## ADR` headings** (`rg '^#+ *ADR' design.md` → 0). This is the **A23** finding, unresolved. It is a
cross-reference defect rather than missing thinking — the reasoning each ADR would carry is present in the body under
other headings (route decision in D-1, index naming in D14.4 and the bm25 risk bullet, spec precedence in D-9's final
bullet, flavour-set closure in D-6). The fix is either to add the four headings or to correct the four citations;
either way it must not be left as a dangling reference, because a citation to a section that does not exist reads as
authority that was never written.