> Change class: **L** — full checklist + verification matrix.
> Role: reviewer, not author. `proposal.md`, `design.md`, `adrs.md` and both `specs/**/spec.md` read in full
> (1202 lines). Every code claim below was re-derived from the tree, not from the change's own citations.

## Scope of this review

Read: all five artifacts of this change; `openspec/specs/cognee-v1-api/spec.md` (the delta target); both spec
deltas of the superseded `cognee-saul-memory-migration`; `docs/relay/decisions.md` (D1–D17, D14.1–D14.4),
`dispositions.md` § change 4, `findings-openspec-baseline.md` §1, `findings-database.md` §2/§7/§8,
`conventions-openspec-skeleton.md`; `plan-change4.md` by targeted `rg` only. Cross-checked the neighbouring
changes `cleanup-foundation` (change 0) and `agent-tools-unification` (change 3) for ownership collisions and for
the NG11 handoff.

Ran: ~35 targeted `rg`/`awk` probes against `src/`, `.venv/.../cognee/`, `docker-compose.yml`, `Makefile`,
`src/alembic/env.py` and `openspec/`. Results in § Verification matrix.

Counts confirmed: **15 requirements / 39 scenarios** (cognee-v1-api 7 req / 16 scen; saul-agent-memory 8 req /
23 scen). Every scenario header uses exactly four hashtags — the silent-drop trap (`schema.yaml:164-165`) is
clear throughout.

## Completeness — the ten checks

| # | Check | Result |
|---|---|---|
| 1 | D10 carried as explicit Non-Goal + ADR known consequence; "entirely" scoped to reconciliation removal | **PASS — strongest section of the change** |
| 2 | ADR settles the three-way in-code disagreement by partition key; report → Cognee only; `CogneeStore` deleted | **PASS** |
| 3 | Baseline stays 16/6; the two `cognee-v1-api` defects distinguished; Purpose repair tracked separately | **PASS** |
| 4 | Delta mechanics on `cognee-v1-api` | **PASS with one gap** — see B4 |
| 5 | Harvest of the superseded change's 7 requirements; archived not deleted | **FAIL on one requirement** — see B2 |
| 6 | Trap3 + the missing worker/beat stated as a dependency, not assumed | **PASS** |
| 7 | Item 152's two config bugs are requirements with scenarios, not prose | **PASS** |
| 8 | D17 read seam labelled speculative | **PASS** |
| 9 | Scope: no `check_graphiti`, no outbox/URL-accessor claim, D4 deletions land here | **PASS** (one wrong path, N1) |
| 10 | Falsifiability, spec leakage, code-claim accuracy | **FAIL** — see B1, B3, B5 |

### Check 1 — D10 (mandatory) — PASS, and it is stated as a loss, not hedged

`design.md` § Goals / Non-Goals carries NG1–NG5 as a table naming, per capability, the sole prior implementation
and the answer to "replacement anywhere in the repo?" (four × **None**, one × **partial, wrong axis**), then
states flatly: *"After this change, agent memory grows without decay, curation, or dedup. This is D10, mandatory
and non-negotiable. It is not mitigated here."* `adrs.md` § Consequences → Negative/accepted opens with the same
sentence as a **known consequence** of the accepted boundary and adds *"This ADR does not fix it; it names it."*
Both documents carry the D10 sentence verbatim in the required scoping: *"Item 155's word 'entirely' is honoured
for reconciliation removal, never for capability parity."* `proposal.md` repeats it in Non-Goals **and** in Risks
("Not mitigated — accepted and recorded"). No hedge, no "future work will address", no mitigation dressed up as a
fix. The one safeguard claimed (a size metric on consolidation) is correctly described as making growth
*observable*, not bounded.

### Check 2 — the ADR boundary — PASS

The ADR does settle rather than restate. Its Context table names all three disagreeing sites, including
`write_final_report.py:8-13` routing to **both**; the Decision splits by each library's own partition key; the
alternatives table rejects dual-write (A), Cognee-primary (B), Graphiti-primary (C) and `CogneeStore` (D) with
reasons that are not merely "D2 says so" for A and D. The two load-bearing facts are real: `remember.py:274-276`
raises `session_id is required for typed memory entries`, and every Graphiti write in `src/` is
`group_id=document_id` / `state.doc_id`. Final report → Cognee only (Decision 4); `CogneeStore` deleted not
implemented (Decision 3 + NG9). Status **Accepted**, dated.

### Check 3 — the baseline claim — PASS

`design.md` § Context isolates the two defects into a two-row mechanism table, states that **nothing in the delta
mechanism writes a `## Purpose` header**, keeps the baseline at **16 passed / 6 failed**, and explicitly corrects
two errors in `plan-change4.md` (its 17/5 claim at plan `:327`/`:438`, and its ordering constraint that the header
must be repaired before the delta). `rg` over the change's artifacts finds 17/5 only in those corrective sentences
and in the conditional "if and when it is done" row — no artifact claims this change reaches 17/5. The delta target
parses today (verified: `openspec/specs/cognee-v1-api/spec.md` opens at `### Requirement:` on line 1 and its four
requirement blocks are well-formed), so the plan's ordering constraint is indeed wrong and the correction is right.

### Check 5 — harvest ledger

| Superseded requirement | Represented in `saul-agent-memory`? |
|---|---|
| writes approved final reports to Cognee | **Yes** — Req 1 + scenario 1 |
| does not write final reports to Graphiti | **Yes** — Req 2, strengthened ("exactly one memory owner") |
| persistence gated by approved output | **Yes** — Req 1 scenario 2, strengthened (unapproved writes **nothing**, at any trust level) |
| prefetch after qna | **Yes** — Req 6 + scenario 1 |
| prefetch SHALL be Cognee-first | **Yes** — Req 6 scenario 2 |
| **deep retrieval limited to selected reasoning nodes** | **PARTIAL — see B2.** Only the negative behavioural half survives, as one scenario under the prefetch requirement. The tool-exposure half is asserted to be change 3's, and change 3 does not carry it. |
| retrieval failures fail open | **Yes** — Req 7, both directions (read and write) |

`design.md` Decision 11 states the superseded directory is **archived, never deleted**, gives the reason (its
`proposal.md:20-21` is the primary citation for D10 and must stay quotable), records `superseded-by` on both ends,
and pre-decides the fallback if `openspec archive` refuses a 0/15-task change. Verified: that sentence is at
`proposal.md:20` of the superseded change, exactly as quoted.

### Check 6 — Trap3 and the infrastructure dependency — PASS

`design.md` § Context has a dedicated subsection *"The infrastructure this change depends on and cannot create"*
which states, in the imperative: no worker and no beat service in `docker-compose.yml` at all; `ai-service-1`
declares no `command:`; `Makefile:52` starts a worker from a `celery_config` module that does not exist; therefore
this change *"registers a task and a schedule entry, and states plainly that no process exists to execute them"*,
and *"anyone reading the consolidation requirement as 'consolidation runs nightly in production' is reading it
wrong until that service exists."* Repeated in `proposal.md` Risks, `design.md` Risks and `adrs.md` Consequences.
All four infrastructure facts verified below. This is exactly the treatment the check asks for.

### Check 7 — item 152 — PASS

Both defects are `ADDED` requirements with two scenarios each: *Memory embeddings match the document embedding
dimensionality* (with "the third-party embedding default is never used") and *Memory vectors are persisted in the
application's managed database* (with "no memory data is written to local files"). Not prose. Both underlying
facts verified in the installed library.

### Check 8 — D17 — PASS

`design.md` Decision 10 is titled *"The read seam is built, and it is **speculative** (D17)"* and opens
*"Labelled explicitly, as D17 requires."* It states the unwired graph was deliberate and stays commented, that the
seam therefore *"cannot be exercised by running the product — not temporarily, but permanently under the current
decision"*, and that its proofs are import/type/unit-level only. Reinforced by NG10, a Risks bullet, and an
`adrs.md` Consequences bullet. Nothing implies verifiability by running the product.

### Check 9 — scope — PASS

- `check_graphiti` is **not** claimed. Decision 7 disclaims it explicitly and narrows the change to `check_cognee`.
  Verified: `check_graphiti` exists and is already in `ALL_PROBES`. (Wrong path cited — N1.)
- Outbox tables, the alembic head merge and the connection-URL accessor are disclaimed as change 0's in NG12/NG13.
  Verified against `cleanup-foundation`: no double ownership, and change 0's `design.md:73-76` reciprocally
  assigns the Cognee probe to change 4.
- `write_final_report.py` and `memory_pipeline.py` deletion sits here (Decision 12), not in change 0 — verified:
  `rg` over all of `cleanup-foundation/` returns **zero** mentions of either file. D4's carve-out is honoured.

## Correctness — blocking findings, most severe first

### B1 — The "authenticated database connection" requirement is premised on a false code claim, and describes an interface Cognee does not have

**Files:** `specs/cognee-v1-api/spec.md:55-71` (`### Requirement: The memory subsystem receives an authenticated
database connection`) · `design.md` Decision 5 (`:247-258`) · `adrs.md` Decision 6 (`:96-100`).

**The claim.** All three say `cognee_client.py:111` *"reads `settings.POSTGRES_URL` **raw**"*, that this hands
Cognee a credential-less URL, and that the fix is for Cognee to receive the output of change 0's single
connection-string accessor. The spec turns this into a normative requirement: *"SHALL receive a database
connection string that is usable as given — correct scheme, credentials present, and no transport parameters the
driver rejects. It MUST NOT read a raw configuration value that bypasses the application's single connection-string
accessor."*

**What the code does.** `cognee_client.py:92-102` configures Cognee's relational store with **discrete fields** —
`db_host`, `db_port`, `db_username`, `db_password=settings.POSTGRES_PASSWORD.get_secret_value()`, `db_name` — i.e.
the password **is** supplied, correctly, from the secret. Line 111 is inside the **returned metadata dict**
(`config = {... "postgres_url": settings.POSTGRES_URL}`) that becomes `app.state.cognee_config`; it is never passed
to Cognee. Confirmed downstream: `cognee/infrastructure/databases/relational/config.py` `RelationalConfig` has
**no connection-string field at all** (only `db_path/db_name/db_host/db_port/db_username/db_password/db_provider/
database_connect_args/pool_args`), and the pgvector path *builds its own* string —
`create_vector_engine.py:205-208`: `f"postgresql+asyncpg://{vector_db_username}:{vector_db_password}@{vector_db_host}:{vector_db_port}/{vector_db_name}"`.

**Why this is blocking.** The requirement cannot be satisfied as written — there is no interface into which a
connection string can be handed — so an implementer must either ignore it or, worse, satisfy it literally by
replacing today's working discrete credential config with a URL-derived one. It also spends a cross-change
dependency (NG13, and change 0's ADR-3 "third flavour … the value handed to the embedded third-party component")
on a consumer that does not consume a URL. Note the same wrong claim sits in `findings-database.md` §2 and in
change 0's `design.md:218-221`; change 4 is the change closest to this code and is the right place to catch it.

**The real gap this masks, unrecorded anywhere:** the app's Postgres is Timescale Cloud and the raw
`POSTGRES_URL` carries `sslmode=require`/`channel_binding=require` (stripped by `get_database_url()` for asyncpg).
Cognee's discrete config offers no place for those; transport security would have to go through
`database_connect_args`, which nothing sets. That is the question the precondition audit should be asking about
this connection, and it is not asked.

**Fix.** Restate the requirement in terms of what Cognee actually accepts: the memory subsystem SHALL be
configured with credentials drawn from the application's single settings source (never a placeholder default),
SHALL have transport-security parameters supplied in the form its own driver accepts, and its first database
operation SHALL authenticate. Delete the "connection string … usable as given" and "single connection-string
accessor" language from `specs/cognee-v1-api/spec.md:55-59`, and correct Decision 5 / ADR Decision 6 to say that
line 111 leaks a passwordless URL into `app.state.cognee_config` (from where `rag/graphiti/registry.py:29` passes
it as `cognee_client=`) — a real defect, but a different one. Then re-check whether NG13's dependency on change 0
is still needed at all.

### B2 — "Deep memory retrieval is limited to selected reasoning nodes" is handed to a change that does not carry it

**Files:** `design.md` NG11 (`:163-167`) and Decision 11 (`:350`) · `specs/saul-agent-memory/spec.md:137-140`.

NG11 says the tool-exposure half is *"Harvested out of the superseded change and handed to **change 3**"*, and
Decision 11 says *"three of the four requirements of its prefetch capability are harvested; the fourth (tool
exposure) goes to change 3 as NG11."*

**Verified:** `openspec/changes/agent-tools-unification/` is already authored (proposal, design, adrs, 7 spec
deltas) and `rg -ni "cognee|memory|deep(er)? retrieval"` over it returns **nothing** on this subject. Its
`agent-tool-registry/spec.md:74-92` has a generic *"every agent role receives the tools assigned to it"*
requirement whose scenarios enumerate precedent/statute and knowledge-graph/obligation-chain tools — no memory
tool, and **no negative requirement that the orchestrator does not get one**. `retrieve_from_memory` has zero hits
in `src/`. So after change 4 archives the superseded change, the only surviving trace of this cost-limiting
constraint is one scenario ("Deep retrieval is not performed for every task"), and the positive half (risk analysis
and compliance *may*) plus the orchestrator exclusion are gone from the change set entirely.

Compounding it: the scenario's own term **"deeper memory retrieval" is never defined in the capability**. The
parent requirement (`:115-119`) is about *when* prefetch runs and mentions only "agent memory" and a "bounded
supplement"; nothing in `saul-agent-memory` says what a *deeper* retrieval is or that one exists. As written the
scenario is not falsifiable by a reader of the spec alone.

**Fix.** Either (a) promote it to a first-class requirement in `specs/saul-agent-memory/spec.md` that defines
deeper memory retrieval as an operation distinct from prefetch, permits it for risk analysis and compliance, and
forbids it elsewhere — which keeps the whole constraint in the change that owns memory behaviour; or (b) keep NG11
but name the receiving artifact and the requirement text change 3 must add, and record it as an open cross-change
obligation rather than a completed handoff. Option (a) is cheaper and does not depend on a change already authored.

### B3 — The spec forbids the design's own pre-decided fallback

**Files:** `specs/cognee-v1-api/spec.md:21-36` vs `design.md` Decision 4 (`:243-245`) and `adrs.md:122-125`.

The requirement: *"Agent-memory vectors … MUST NOT be written to the local filesystem of the process that produced
them"*, with the scenario *"**THEN** no memory vector store SHALL be configured against a local filesystem path"* —
absolute, no exception.

The design's recorded fallback, if the managed instance refuses the schema or the extension: *"use a local-file
vector store **on a mounted persistent volume**, for memory recall only"* (`design.md` Decision 4, repeated in
Risks and in `adrs.md`'s amendment as `vector_db_provider="lancedb"`). A mounted volume **is** a local filesystem
path. So the branch the design pre-decided is one the delta prohibits, and the contradiction surfaces exactly when
the precondition audit fails — the moment the fallback is needed.

**Fix.** Make the requirement's normative test durability rather than storage medium: forbid a store whose data
does not survive process/container replacement, and add an explicit scenario for the durable-volume fallback
(permitted for memory recall only, never for document retrieval per D5.1). Alternatively keep the absolute
prohibition and record in `design.md` that taking the fallback requires a spec amendment — but do not ship a delta
that outlaws the documented contingency.

### B4 — A deployed requirement is invalidated by this change and no delta covers it

**Files:** `openspec/specs/cognee-v1-api/spec.md` (4th requirement, *No type ignore suppressions*) ·
`specs/cognee-v1-api/spec.md` (no delta for it) · `proposal.md:81-82`.

The deployed capability has **four** requirements. The delta touches three. The fourth's only scenario is:
*"**WHEN** `uv run ty check src/app/shared/langchain_layer/agents/memory/cognee_client.py` is run **THEN** no type
errors are reported on `cognee.remember()`, `cognee.improve()`, or `cognee.recall()` calls."* This change retires
that module (Decision 12/14: the three module-level functions and `CogneeStore` go, the package is replaced by a
service), so the requirement's only scenario names a path that will not exist — an accepted, deployed requirement
that becomes unverifiable. Meanwhile `proposal.md` § Removed from Scope asserts *"No deployed requirement is
removed."* True as to `REMOVED` deltas, misleading as to effect.

Related and worth stating because the reviewer's brief asks it directly: each of the three `MODIFIED` blocks
reproduces the original `### Requirement:` header **exactly** (verified character-for-character against the
deployed file) and *Query memory via recall* reproduces the original body verbatim, but **none of the eight
original scenarios is carried through** — each is replaced by a rewritten scenario covering the same concern
(`Store final report` → `Approved final report is stored in conversation scope`; `Store relationships` →
`Relationship summaries are no longer stored in agent memory`; `Process report after store` / `Process
relationships after store` → `A write does not trigger consolidation` / `Consolidation is invoked only on a
schedule`; `Search episodic memory` → `Recall is scoped to the caller's memory partition`; `Search returns results
as dicts` → `Recall results are fully serialisable and retain their origin`; `Search handles failures gracefully` →
kept, renamed, normative). Every original concern is accounted for and every replacement is a deliberate behaviour
change, so I do **not** treat this as accidental detail loss — but the archive will overwrite the deployed
scenarios wholesale, and that is only correct because it is intended. State that intent in `design.md` Decision 9
so a later reader does not read it as the partial-copy mistake `schema.yaml` warns about.

**Fix.** Add a fourth `MODIFIED` block for *No type ignore suppressions* that keeps the prohibition but restates
its scenario against the surviving module (or path-neutrally, "on the agent-memory call surface"), and correct
`proposal.md`'s "no deployed requirement is removed" to name this one as modified.

### B5 — The read seam's harvest rests on a misread of the code it relocates

**File:** `design.md` Decision 10 (`:328-329`).

Claim: *"`memory_pipeline.py:213,220` already branches on exactly the two task names"* — offered as the reason
relocating the logic now is cheaper than rebuilding it later.

**Verified:** `memory_pipeline.py:213` is `if task in {"risk_analysis", "obligation_chain"}:` and `:220` is
`elif task == "compliance":`, inside `_do_retrieve_graphiti_context`. So it branches on **three** task names, not
two, and it is the **Graphiti supplement** branch — not a deep *memory* retrieval branch. Two consequences at
implementation time: `obligation_chain` is silently dropped from supplement eligibility if the relocation follows
the design's reading, and the spec's "risk analysis or compliance" gate (`saul-agent-memory:137-140`) will be
implemented over the Graphiti path while presenting as a memory-retrieval constraint.

**Fix.** Correct Decision 10 to name the three task values and the function's actual purpose, and state explicitly
whether `obligation_chain` keeps supplement eligibility in the relocated prefetch node. If it does, the spec's
scenario at `:137-140` needs its third task named or a rationale for excluding it.

### B6 — The startup posture is prescribed two ways, and the one unguarded lifespan call is never mentioned

**Files:** `specs/cognee-v1-api/spec.md:13` and `:35-36` · `specs/saul-agent-memory/spec.md:165-169` ·
`design.md` § Migration Plan step 3.

`cognee-v1-api` requires *"startup SHALL fail rather than proceed if the two differ"* (embedding dimension) and,
two requirements later, *"startup SHALL report the subsystem as degraded rather than silently falling back to local
files"*; `saul-agent-memory` requires unconfigured memory to report **degraded** and *"SHALL NOT fail the
request"*. Those are three different postures for three misconfiguration classes, which may well be intentional —
but nothing in `design.md` says so, and the change never observes the relevant fact: **`lifespan.py:206`
`await setup_cognee(settings)` is unguarded**, while every optional subsystem around it (Graphiti `:211-223`,
Crawl4AI `:258`, object storage `:266`, Celery `:273`, outbox `:284`) is wrapped with graceful degradation — and
commit `1b3891f` *"make startup resilient to optional services"* rewrote 121 lines of that file without touching
it. An implementer reading "startup SHALL fail" literally will take the API down on a Cognee misconfiguration.

**Fix.** Add one paragraph to `design.md` (Decision 8 or a new decision) stating the intended posture per failure
class — hard-fail on a dimension mismatch because it corrupts stored vectors; degrade on an unreachable or absent
store — and say explicitly whether `lifespan.py:206` becomes guarded. If it stays unguarded, say that too, since it
is then the only optional subsystem that can kill boot.

## Standards

Checked against `.opencode/instructions/` and `config.yaml:6-35`'s injected conventions, to the extent artifacts
(not code) can be checked:

- **No spec leakage.** `specs/saul-agent-memory/spec.md` contains zero library names, class names, function names,
  file paths or `.py` references (`rg -ni "cognee|graphiti|celery|postgres|lancedb|pgvector|class |def |\.py"` →
  no hits). It is a clean behavioural contract. `specs/cognee-v1-api/spec.md` names `cognee.remember()` /
  `improve()` / `recall()` — inherited from the deployed capability, which *is* an API-surface contract; the ADDED
  requirements it introduces are library-neutral. Accepted, not a finding.
- **`RESULT-PATTERN` / `EXCEPTION-RULES`.** Decision 13 consolidates three coexisting failure idioms onto one, and
  keeps `e.add_note()` before re-raise as house style (verified live at `cognee_client.py:251`). No `match/case` on
  `Success`/`Failure` is proposed anywhere. Correct.
- **`SecretStr.get_secret_value()`.** Not stated as a requirement, but the code being replaced already uses it
  (`cognee_client.py:89,98`) and B1's corrected requirement should keep it.
- **Async-first / no blocking calls.** Not at issue; every seam described is `await`ed.
- **Artifact schema.** Change-class blockquote present on `proposal.md` and `design.md`; `design.md` carries all
  seven mandated sections in order; `Risks / Trade-offs` uses the literal `[Risk] → Mitigation` form; every
  Decision carries *alternatives considered*; `Open Questions` are all precondition checks with both branches
  pre-decided (schema-conformant — none would change the specs); `adrs.md` carries Status / Context / Decision /
  Rationale-Alternatives / Consequences. `.openspec.yaml` declares `schema: spec-gated`, matching
  `openspec/config.yaml:1`. `## Purpose` correctly **absent** on the `cognee-v1-api` delta and **present and real**
  (five substantive lines, not a "TBD - created by archiving" stub) on `saul-agent-memory`.

## Risk

The change's own risk register is unusually honest and I found nothing material missing from it beyond B1's SSL
question and B6's boot posture. Two residual risks I would rank differently than the author:

- **The `env.py` filter is load-bearing and is the change's only guard against a third-party library's tables being
  dropped by `--autogenerate`.** The change's belt-and-braces framing ("`include_schemas` defaults to `False`, so
  it's already invisible") is correct today — verified: `src/alembic/env.py` contains no `include_object`,
  `include_name` or `include_schemas` anywhere — but the filter is the only thing that survives someone flipping
  that flag, and it is scheduled late (step 3's neighbourhood) relative to the first write that creates those
  tables. Consider ordering the filter **before** the vector-store configuration lands.
- **`cognee.prune()` must never run against the shared Neo4j.** Noted in both `design.md` Risks and `adrs.md`
  Consequences as *"worth an explicit grep guard"*. That is the right instinct and should become a task with a
  proof (`rg -n "cognee.*prune" src/` → 0), not a remark.

## Non-blocking — accuracy corrections and nits (do not block `tasks.md`)

- **N1 — wrong path for `check_graphiti`.** `design.md` Decision 7 cites `features/health/health_check.py:83-90`.
  There is no such file. It is `src/app/middleware/health_check.py:83-90` (registered in `ALL_PROBES` at `:94+`) —
  which `plan-change4.md:273` already had right, so this is a regression introduced in the artifact. The
  substantive claim (it exists, it is registered, it is not change 4's) is **true and verified**. Fix the path in
  Decision 7. (`dispositions.md` 198.2 carries the same wrong path and could be corrected in the same pass.)
- **N2 — wrong line for `target_metadata`.** `design.md` Decision 4 and `adrs.md`'s amendment both say
  *"`src/alembic/env.py:23-30` sets `target_metadata = Base.metadata`"*. Lines 23-33 are model imports;
  `target_metadata = Base.metadata` is at **`:39`** (with `:42` `target_metadata = None` on the offline branch).
  The substance — no filter of any kind — is verified true.
- **N3 — `remember.py:895-900`** is cited in Decision 2 for *"appends to the conversation cache and never touches
  the rebuild"*. Those exact lines are the `except`/`create_task`/`return` of the detached session-improve bridge;
  the substance (session mode returns at `:900`, before `_run()` at `:915` is ever reached) is correct. The ADR's
  `~885-890` for the `asyncio.create_task` is `:898`.
- **N4 — `cognee_client.py:257`** cited as the swallow-to-empty-list; the `return []` is at `:256`.
- **N5 — "zero production call sites".** `design.md` § Context says `store_final_report`, `store_relationships`
  and `search_episodic_memory` have zero production call sites and only the package `__init__` re-exports them.
  `search_episodic_memory` is genuinely uncalled, but the first two are invoked at
  `write_final_report.py:122,146` through a `Protocol` declared at `:44,48` — a duck-typed edge `graphify affected`
  would not surface. Dead code calling dead code, and this change deletes both in one step, so nothing follows from
  it; the sentence should say "no *live* call sites" and name the protocol edge, because it is precisely the kind of
  edge the deletion ordering has to respect.
- **N6 — the `memory` field name collision.** `saul-agent-memory`'s health requirement says both surfaces SHALL
  include the agent-memory subsystem. The second surface already reports a field called `memory` —
  `features/health/service.py:69` `_check_memory()`, which is psutil RAM (`:200-213`). Whoever implements the probe
  must not reuse that key. Worth one sentence in `design.md` Decision 7.
- **N7 — the Purpose-header repair has no home yet.** `design.md` says it is *"carried as its own separately-
  tracked one-line file edit in `tasks.md`"*. `tasks.md` does not exist (correctly — it is gated on this review).
  Carry it forward; if it is descoped, the delta is unaffected, as `design.md` already says.

## Verified clean (examined and found correct — distinguish from "not examined")

- **D10 treatment** in all three of `proposal.md`, `design.md` and `adrs.md`, including the "entirely ≠ capability
  parity" scoping sentence in both required places.
- **The ADR's four library facts**, each re-derived from `.venv`: session requirement, Graphiti's document
  partition key, `remember()`'s permanent-mode `add`→`cognify`→`improve` chain, and Graphiti-only bitemporality.
- **Baseline arithmetic and mechanism** (16/6, unchanged; two defects, two mechanisms; no 17/5 claim).
- **Delta hygiene:** three `MODIFIED` headers match the deployed spec exactly; no `## Purpose` on the existing
  capability; real `## Purpose` on the new one; all 39 scenarios use four hashtags; 15 requirements / 39 scenarios
  as advertised; new capability name `saul-agent-memory` collides with none of the 20 deployed capabilities.
- **Scope boundaries:** no `check_graphiti` claim; no outbox/URL-accessor/alembic-merge claim; D4's two deferred
  files land here and nowhere else; no overlap with change 0's or change 3's authored artifacts (other than B2's
  missing handoff).
- **Trap3 mechanism:** session-mode `remember()` genuinely bypasses `cognify()`; `self_improvement=True` genuinely
  fires a detached `asyncio.create_task`; `improve(dataset, session_ids=[...])` genuinely exists.
- **Item 152 both halves:** the 3072-vs-768 model default and the `lancedb` vector default are both real, and the
  ADR's correction about `embedding_dimensions` now defaulting to `None` is accurate down to the source comment.
- **The ACL mechanism**, including the ADR's correction that the raise comes from a handler/provider mismatch and
  not from an absent-backend list — verified end to end, including that `"ladybug"` is a real key whose
  `handler_provider` is `"ladybug"`, so the mismatch branch is the one taken.
- **Infrastructure claims:** compose services are exactly `rabbitmq`, `timescale`, `caddy`, `ai-service-1` with no
  `command:` and no worker/beat; `celery_config` exists nowhere in the repo; `include` has 4 entries;
  `beat_schedule` has 4 billing entries; `tasks/__init__.py:6-9,18-20` is the reconciliation edge that gates the
  new task's registration.
- **Deletion targets and their paired re-exports:** `rag/graphiti/__init__.py:47,59` and
  `memory/__init__.py:3-9,23-39` are exactly as described; missing either would be an `ImportError` at boot.
- **`cognify` has zero call sites in `src/`** — `rg -n "cognify|cognee\.add|cognee\.search" src/` returns nothing.
  The change's central thesis holds.

**Not examined:** the wiki/graph indexes; `plan-change4.md` in full (targeted `rg` only); test suites; anything
about changes 1 and 2 beyond ownership collisions.

## Verification matrix — code claims spot-checked

35 claims probed. **31 confirmed exactly** (some ±1 line), **1 wrong in substance**, **3 wrong in citation with
correct substance**.

| Claim | Result |
|---|---|
| `cognify` zero call sites in `src/` | **confirmed** (zero hits) |
| `remember.py:274-276` raises `session_id is required for typed memory entries` | **confirmed, verbatim** |
| `remember.py:610` `self_improvement: bool = True` | **confirmed** |
| `remember.py:915-944` `add` → `cognify(run_in_background=False)` → `improve` | **confirmed** |
| `remember.py:41` `_ensure_migrations_run`; `cognee/alembic/` exists | **confirmed** |
| `improve.py:36` `improve(dataset=…, session_ids=…)` | **confirmed** |
| `forget.py:16` deletion by identifier | **confirmed** |
| `vector/config.py:30` `vector_db_provider = "lancedb"` | **confirmed** |
| `vector/embeddings/config.py:72` default model; `:73-77` the 3072-removal comment | **confirmed, verbatim** |
| `context_global_variables.py:88-92` unset → `multi_user_support_possible()`; `:34-81` raises `EnvironmentError` on handler/provider mismatch; `:96` list feeds only `is_multi_user_support_possible()` | **confirmed** |
| `supported_dataset_database_handlers.py:18-21` `neo4j_aura_dev`; `"ladybug"` present at `:31` | **confirmed** |
| `graph/config.py:45,59` provider/handler defaults; `:77-79` remaps only kuzu and postgres | **confirmed** |
| `cognee_client.py:12-15` docstring boundary claim | **confirmed** (`:12-14`) |
| `cognee_client.py:150-151` `remember()` then a second `improve()` | **confirmed** |
| `cognee_client.py:140,189,238` three bare f-string dataset names | **confirmed, all three** |
| `cognee_client.py:159` re-raise · `:251` `add_note` · `:259` `[dict(r) for r in …]` · `:304` `CogneeStore.search` | **confirmed** |
| `cognee_client.py:257` swallow-to-empty-list | citation off by one (`:256`) — **N4** |
| `cognee_client.py:111` "hands Cognee the raw connection string" | **WRONG in substance — B1** |
| `write_final_report.py:8-13` routes to both · `:110` Graphiti write · `:156-161` error-string collection | **confirmed** |
| `rag/graphiti/client.py:311-350` `write_final_report_episode`, `group_id=metadata.user_id` at `:345` | **confirmed** |
| `memory_pipeline.py:109-116` duplicate trim · `:129-157` tool filter · `:160-201` context prefix · `:258-260` fail-open | **confirmed** |
| `memory_pipeline.py:213,220` "exactly the two task names" | **WRONG — three names, Graphiti branch — B5** |
| `messages.py:40-52` duplicate trim, same counter and strategy | **confirmed** |
| `agent_saul/nodes.py:802` `COGNEE_WRITE_FAILED` on the persist-memory node (`:772-814`) | **confirmed** |
| `agent_saul/graph.py:86` `build_saul_graph`, no caller | **confirmed** |
| `lifespan.py:206` `setup_cognee` — and it is **unguarded** | **confirmed (+ B6)** |
| `settings.py:212` `EMBEDDING_DIMENSION: int = Field(default=768, gt=0)`; no `cognee` in `settings.py` | **confirmed** |
| `check_graphiti` exists at `…/health_check.py:83-90`, in the probe list | **confirmed, wrong path cited — N1** |
| `features/health/service.py` probes mongo/redis/postgres/neo4j/celery/memory/disk, neither graph nor cognee | **confirmed** |
| `src/alembic/env.py` has no `include_object`/`include_name`/`include_schemas` | **confirmed; `:23-30` citation wrong — N2** |
| `Makefile:52` `celery -A celery_config`; no `celery_config` module in the repo | **confirmed** |
| `docker-compose.yml` services, no worker/beat, no `command:` on `ai-service-1` | **confirmed** |
| `connections/celery.py:191-196` 4 includes; `:259-276` 4 billing beat entries | **confirmed** |
| `tasks/__init__.py:6-9,18-20` reconciliation import + re-export | **confirmed** |
| `memory_decay_reconciliation_tasks.py:51,64,180,186,198`; no `@celery_app.task` in the module | **confirmed** |
| `documents/service.py:544,753` · `graphiti_verifier.py:56,70` · `ingestion_kb/nodes.py:384,397` | **confirmed** |
| superseded `proposal.md:20-21` "Cognee v1.1 has no built-in curation/decay/dedup" | **confirmed, verbatim** |
| `openspec/config.yaml:1` `schema: spec-gated` | **confirmed** |

## Verdict

Six blocking findings. Five are artifact-level and cheap to fix (B2–B6 are text edits plus one added requirement);
B1 requires re-deriving one requirement from what the library actually accepts. None of them touches the change's
core reasoning: the D10 record, the partition-key boundary, the Trap3 topology, the 16/6 baseline analysis and the
D17 labelling are all sound, and the artifacts are the most carefully evidenced I have reviewed in this relay. But
B1 and B3 would each produce wrong work at implementation time, and B2 silently drops a cost-limiting constraint
from the whole five-change set, so this cannot be approved as it stands.

VERDICT: CHANGES REQUESTED

(openspec schema token: `CHANGES-REQUESTED`. Do not author `tasks.md` until B1–B6 are addressed; N1–N7 may be
folded into the same pass but do not gate it.)

## Author response

> Remediation pass, 2026-08-18. Keyed to **this review's own numbering** (B1…B6, N1…N7). Nothing above this heading
> was altered. One numbering note: the remediation brief renumbered two items — its "B3" is this review's **B4**
> (the type-ignore requirement) and its "B4" is the worker/beat runtime gap, which this review raised inside
> Check 4 and Check 6 rather than as a numbered finding. Both are answered below, the second as **B-runtime**.

- **B1** (single-connection-string requirement is unimplementable): **fixed.** Independently re-derived and
  **confirmed** — `RelationalConfig` (`…/cognee/infrastructure/databases/relational/config.py:12-23`) has seven
  discrete fields and no DSN/URL field; `cognee_client.py:91-101` already passes them with a working password at
  `:98`; `:107-112` builds a second local dict also named `config` whose `postgres_url` is only returned, never
  consumed. Requirement rewritten to the discrete-field reality (`specs/cognee-v1-api/spec.md`, now *The memory
  subsystem is configured against the application's own database*, 4 scenarios), Decision 5 rewritten with the
  retraction recorded rather than overwritten, ADR Decision 6 likewise, NG13 restated as **no longer a dependency on
  change 0**, proposal bullet rewritten. **The surviving defect is now the one specified:** `:96`/`:100` read
  `POSTGRES_HOST`/`POSTGRES_DB_NAME` independently of `get_database_url()`. Verified as instructed —
  `.env.development` sets them to the same Timescale Cloud host/port/db as `POSTGRES_URL`, so **they agree today by
  hand-maintained duplication, not by construction**, and the Pydantic defaults diverge outright
  (`settings.py:140` `postgresql://user:pass@host/db` vs `:141` `localhost`, `:145` `db`); `.env.example` sets none of
  them, so any environment that configures only `POSTGRES_URL` points memory at `localhost:5432/db` and **succeeds
  silently**. Same-instance resolution, no-placeholder-defaults, and startup failure on divergence are now normative.
  Item 152's `set_vector_db_config` claim is untouched and still a requirement, as instructed. Your SSL point is
  adopted: a transport-security risk bullet, an ADR bullet and a precondition question were added, since
  `database_connect_args` is the only place `sslmode` could go and nothing sets it.
- **B2** (deep retrieval lost its tool-exposure half): **fixed, and ownership reassigned to this change.**
  Confirmed — change 3's artifacts mention it nowhere. Option (a) taken: `saul-agent-memory` gains *Deeper memory
  retrieval is available only to designated reasoning roles* (4 scenarios) which defines deeper retrieval as an
  operation **distinct from prefetch**, permits it for risk analysis and compliance, forbids it to the orchestrating
  role, and requires refusal-with-reason rather than an empty result — closing your falsifiability point too. NG11
  restated as behaviour-owned-here / binding-owed-by-change-3, and a new **Coordination points** table in `design.md`
  § Context names the receiving artifacts (`agent-tool-registry`'s role-assignment requirement, `agent-tool-contract`'s
  *Unavailability SHALL never be reported as absence*) as an **open obligation**, not a completed handoff. No
  requirement was written into a capability change 3 owns.
- **B3** (spec forbids the design's own pre-decided fallback): **fixed.** Confirmed by reading both artifacts. The
  persistence requirement's normative test is now **durability** — "no store whose data is lost on process or
  container replacement" — and a third scenario, *A durable file-backed store is permitted only for memory recall*,
  admits the fallback, bounds it to recall, forbids it for document retrieval (D5.1) and requires the health surface
  to report the fallback rather than reporting fully configured. Decision 4 and the ADR amendment both now say the
  fallback needs **no spec amendment**.
- **B4** (deployed 4th requirement archives stale): **fixed as `MODIFIED`, not `REMOVED`.** Confirmed. A fourth
  `MODIFIED` block keeps *No type ignore suppressions*, keeps its original scenario title *Type checker passes*
  **verbatim**, restates the test path-neutrally over "the module or modules that hold the agent-memory call surface",
  and adds that retiring or relocating that surface is not a way to satisfy it. `MODIFIED` rather than `REMOVED`
  because the prohibition is still wanted — `REMOVED` would need a Reason and a Migration for a rule this change
  keeps. `proposal.md` § Removed from Scope corrected to "no deployed requirement is **deleted**, and all four are
  modified". Your related point is also addressed: Decision 9 now carries the seven-row **old → new scenario title
  mapping** and states the wholesale replacement is intended, noting that two titles (`Store relationships`,
  `Process report after store`) cannot be kept verbatim without asserting the opposite of their own bodies.
- **B-runtime** (consolidation registered with no worker/beat): **recorded, not duplicated — per instruction.** The
  runtime gap is dispositioned **in change 1** (`dispositions.md` 198.4), so no change-4 requirement was added.
  Instead: **NG14** ("after this change, consolidation never runs"), coordination point **C-B** with its consequence
  column, and amended Risk/ADR/proposal bullets that call the beat entry **inert on the day it lands**. The separate
  change-0 ordering dependency (`src/tasks/__init__.py:6-9,18-20` re-exports, without which registration cannot even
  be proven) is stated alongside it.
- **B5** ("exactly the two task names"): **fixed.** Confirmed at the real path
  `src/app/shared/rag/graphiti/memory_pipeline.py`: `:213` `if task in {"risk_analysis", "obligation_chain"}:`,
  `:220` `elif task == "compliance":`, inside `_do_retrieve_graphiti_context` (`:204-237`) — **three** values, and the
  **knowledge-graph supplement** branch, not a memory-retrieval branch. Decision 10 corrected with both consequences
  settled: **`obligation_chain` keeps supplement eligibility** (dropping it would be a silent regression smuggled in
  under a relocation), and the two constraints are now two requirements — the three-task supplement gate in the
  prefetch requirement (scenario retitled *The knowledge-graph supplement is fetched only for the tasks that need it*)
  and the two-role deeper-retrieval restriction in its own requirement.
- **B6** (startup posture prescribed three ways; `lifespan.py:206` unmentioned): **fixed with a stated posture.**
  Confirmed: `:206` `cognee_config = await setup_cognee(settings)` is bare while Graphiti `:211-223` and every other
  optional subsystem degrade. New **Decision 15** gives a per-failure-class table — **hard fail** on dimension/model
  mismatch (unrepairable vectors, no re-embedding path), **degrade** on unconfigurable or unreachable stores — and
  states explicitly that **`:206` becomes guarded**, re-raising only the mismatch class. Per D10's honesty rule the
  cost is stated as a loss, not hedged: *guarding it means a Cognee misconfiguration no longer stops a deploy, leaving
  the health probe as the only signal — which is why the probe is a requirement and not a nicety.* Mirrored in an ADR
  Consequences bullet and a proposal Risk.
- **N1** (`check_graphiti` path): **fixed** — `src/app/middleware/health_check.py:83`, registered in `ALL_PROBES`
  (`:93-99`, fifth entry at `:98`). Corrected in Decision 7 with the wrong path called out so it is not re-inherited.
  `dispositions.md` 198.2 is outside this change's edit scope and was left alone.
- **N2** (`target_metadata` line): **fixed** — `src/alembic/env.py:39` (`:42` offline `None`; `:23-33` model imports),
  in `design.md` Decision 4 and in the ADR amendment and alternative F.
- **N3** (`remember.py` citations): **fixed** — session mode now cited as returning at `:900` before `_run()` at
  `:915`, with the detached bridge at `:885-898`; the ADR's `~885-890` corrected to `:898`.
- **N4** (`return []` at `:256`): **refuted.** Verified by line-addressed read of
  `src/app/shared/langchain_layer/agents/memory/cognee_client.py`: `:256` is
  `).exception("Cognee recall failed")` and **`:257` is `return []`**. `rg -n "return \[\]"` on that file returns
  `257`, `313`, `329`. The change's citation of `:257` in Decision 13 is correct and was left unchanged.
- **N5** ("zero production call sites"): **fixed** — now "no **live** call sites", with the duck-typed edge named:
  `write_final_report.py:122,146` call through the structural `CogneeService` interface declared at `:41-50`, which
  `graphify affected` does not surface, so caller and callee must be deleted together. Added to Migration Plan step 10.
- **N6** (`memory` field-name collision): **fixed** — one paragraph in Decision 7 naming
  `features/health/service.py:69` `_check_memory()` / `:200-213` as psutil RAM and requiring a distinct field name,
  plus a pointer in Migration Plan step 5.
- **N7** (Purpose-header repair has no home): **fixed** — carried as its own task in `tasks.md`, with its own proof
  and an explicit note that it is the only step that moves the failure count and is independently descopable.
- **Self-found, not raised above — the baseline arithmetic is stale.** `design.md` said *"16 passed / 6 failed of
  22"*; measured today `openspec validate --all` reports **21 passed / 6 failed of 27**, because the sibling changes of
  this relay have since been authored and each passes. Corrected in `design.md`, with the reason spelled out: **the
  pass count is not an invariant and must never be used as an acceptance number — the invariant is the failure count,
  6.** Your Check 3 verdict is unaffected; the mechanism argument it validated is unchanged.
- **Two Risk-section recommendations, adopted as ordering and as a task rather than as remarks.** The `env.py` filter
  now lands **before** the vector-store configuration (its own Migration Plan step 3, with the reason: the store is
  what creates the tables the filter protects), and the `cognee.prune()` guard is a task with the proof you specified.

Counts after remediation: **17 requirements / 47 scenarios** (`cognee-v1-api` 8/20, `saul-agent-memory` 9/27), every
scenario header at four hashtags.
