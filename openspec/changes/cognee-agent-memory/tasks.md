> Change class: **L** (grouped sections, each task verifiable). Authored 2026-08-18, after `review.md` moved off
> `CHANGES-REQUESTED` — every blocking finding is answered in `review.md` § Author response.
> **Read before starting:** `design.md` § Migration Plan is the ordering and the reason for it; the ten groups below
> follow its eleven steps in the same order, splitting its later steps into registration (group 9) and
> harvest-then-delete plus disposal (group 10).
> **Three standing rules for every proof in this file.**
> 1. **Compare the summary line, never `$?`.** `pytest` runs `--cov-fail-under=80` against 18.38% coverage, so a
>    green suite still exits 1. Baselines to beat: **`pytest` ≥ 55 passed**, **`ruff check` ≤ 123**, **`ty check`
>    ≤ 46**, **`openspec validate --all` failures ≤ 6**.
> 2. **Never print a credential.** Probes that touch a connection print host/port/database only.
> 3. **No task may call `cognify()`.** Trap3: it is a full graph rebuild. Request-path writes are conversation-scoped
>    and enrichment happens only in the scheduled consolidation job (group 9). A per-document `cognify()` is the one
>    implementation this change exists to prevent.

## 1. Preconditions — read-only, no code, no DDL

- [x] 1.1 Determine whether the target graph database exposes APOC and GDS. Record the answer in `design.md`
      § Open Questions.
      **Proof:** `cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USERNAME" --format plain
      "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc.' OR name STARTS WITH 'gds.' RETURN count(name) AS n"`
      → record `n`. **If `n = 0`, take the pre-decided branch:** this change ships write-only, group 9's task is still
      registered, and the consolidation requirement's *refuses to run when its graph preconditions are absent*
      scenario becomes the observable behaviour rather than an edge case.
- [x] 1.2 Establish, read-only, whether the application role may create a schema and whether the vector extension is
      available — and re-confirm that no third-party memory table exists yet.
      **Proof:** `psql "$POSTGRES_URL" -Atc "select has_database_privilege(current_user, current_database(),
      'CREATE'), (select count(*) from pg_available_extensions where name = 'vector'), (select count(*) from
      pg_tables where schemaname = 'public' and tablename in
      ('entities','relationships','events','memory_versions'))"` → expect `t|1|0`. **No DDL is executed by this
      task.** A false first column takes the Decision 4 fallback (a durable file-backed store, which the delta
      permits explicitly).
- [x] 1.3 Determine whether a connection built from the **discrete** settings fields negotiates TLS at all — the
      question the retracted B1 claim was hiding (`design.md` Decision 5, § Open Questions).
      **Proof:** connect using `POSTGRES_HOST`/`POSTGRES_PORT`/`POSTGRES_USERNAME`/`POSTGRES_DB_NAME` and run
      `select ssl, version from pg_stat_ssl where pid = pg_backend_pid()`. Record `ssl`. **If `ssl = f`**, task 4.5
      must set `database_connect_args` — a configuration line, not a design change.
- [x] 1.4 Prove or disprove, executably, that the memory subsystem's discrete connection settings resolve to the same
      instance the application's own engine resolves. This is the surviving B1 defect made falsifiable.
      **Proof:** `uv run python -c "from urllib.parse import urlparse; from app.config.settings import get_settings;
      from app.connections.postgres import get_database_url; s=get_settings(); u=urlparse(get_database_url());
      print(u.hostname==s.POSTGRES_HOST, u.port==s.POSTGRES_PORT, (u.path or '/').lstrip('/')==s.POSTGRES_DB_NAME)"`
      → expect `True True True`, and note in the record that today they agree only because `.env.development` sets
      both by hand (`.env.example` sets neither, and the code defaults diverge: `settings.py:140` vs `:141`/`:145`).
- [x] 1.5 Find out whether `openspec archive` accepts a 0/15-task change, which decides the shape of task 10.4.
      **Proof:** `openspec archive cognee-saul-memory-migration --dry-run 2>&1 | tail -5` (or the CLI's nearest
      no-op flag) — record whether it refuses. **It is an archive either way, never a delete.**

## 2. Spec-side housekeeping — tracked separately from the delta on purpose

- [x] 2.1 Insert a real `## Purpose` header into `openspec/specs/cognee-v1-api/spec.md` as a **one-line direct file
      edit**, outside the delta mechanism (which structurally cannot write one — `design.md` § Context). This is
      **independently descopable**: dropping it leaves the delta correct and unaffected.
      **Proof:** `openspec validate --all 2>&1 | tail -1` → failure count moves **6 → 5**. This is the only step in
      the whole change that moves that number.
- [x] 2.2 Capture the pre-code validation baseline so any later regression is attributable.
      **Proof:** `openspec validate cognee-agent-memory --type change --strict` → `is valid`; `openspec validate --all
      2>&1 | tail -1` → record the exact totals line (**21 passed / 6 failed (27 items)** before task 2.1).

## 3. The migration filter — before any store is configured

- [x] 3.1 Add an `include_object` / `include_name` filter to `src/alembic/env.py` that excludes the memory schema
      from `--autogenerate` reflection. **Ordering is load-bearing** (Decision 4): the filter is the only protection
      that survives someone setting `include_schemas=True`, and group 4 is what causes the tables it protects to
      exist. `target_metadata = Base.metadata` is at `env.py:39`; the offline branch sets `None` at `:42`.
      **Proof:** a unit test importing the filter directly and asserting both directions — an object in the memory
      schema returns `False`, a table in `Base.metadata` returns `True`; `uv run pytest -k alembic_filter` summary
      line shows the new tests passing.
- [x] 3.2 Wire the filter into **both** `context.configure(...)` calls, not just the online one.
      **Proof:** `rg -n "include_object|include_name" src/alembic/env.py` → the definition plus **two** call sites
      (the online configure near `:77` and the offline one near `:58`).

## 4. Configuration — and the startup posture

- [x] 4.1 Add the `COGNEE_*` settings surface, which does not exist today (`rg -i cognee src/app/config/settings.py`
      currently returns nothing): schema name, dataset prefix, vector provider, and the access-control flag.
      **Proof:** `uv run python -c "from app.config.settings import get_settings; s=get_settings();
      print(s.COGNEE_DB_SCHEMA, s.COGNEE_VECTOR_PROVIDER, s.COGNEE_ACCESS_CONTROL_ENABLED)"` prints all three.
- [x] 4.2 Write the access-control setting into the process environment **explicitly, before the first memory
      configuration call** — left unset, the default branch reaches `multi_user_support_possible()` and raises
      `EnvironmentError` on this repository's handler/provider pair (Decision 6).
      **Proof:** a unit test with a faked memory module asserting `os.environ` holds the key after setup **and** that
      the assignment happens before the first `set_*_config` call (assert on the fake's recorded call order).
- [x] 4.3 Pin the embedding model to the repository's configured embedder and derive the dimension from
      `settings.EMBEDDING_DIMENSION` (768, `settings.py:212`), with a startup assertion that the two are equal.
      Setting the **model** is the fix; setting the dimension is belt-and-braces (Decision 3).
      **Proof:** two tests against a faked memory module — the captured embedding config carries the repository's
      model and `EMBEDDING_DIMENSION`; and a deliberately mismatched dimension **raises at startup** rather than
      degrading (Decision 15, hard-fail class).
- [x] 4.4 Configure the vector store explicitly so the library default is never reached — `set_vector_db_config` is
      never called today, which is why memory vectors would land in local files (item 152, defect two).
      **Proof:** `rg -n "set_vector_db_config" src/` → exactly one call site; a test asserting the captured provider
      is the configured one and is **not** the library default.
- [x] 4.5 Make the relational configuration resolve to the application's own database: derive the discrete fields
      from the single connection accessor (or assert equality against it), fail startup on divergence, and reject any
      field still holding its placeholder default. If task 1.3 found `ssl = f`, set `database_connect_args` here —
      it is the only field the library offers for transport parameters.
      **Proof:** three tests — equal settings configure successfully; a settings object whose `POSTGRES_HOST`
      disagrees with `POSTGRES_URL` raises a **named** configuration error; a field left at its placeholder default
      (`localhost` / `db`) raises rather than being used. **Closes B1's surviving defect.**
- [x] 4.6 Rename the shadowed local `config` at `cognee_client.py:107` and stop returning `postgres_url` from it —
      two variables named `config` in one function, one configuring nothing, is what made the original misread easy,
      and the returned URL reaches `app.state.cognee_config` where a reader can mistake it for configuration.
      **Proof:** `rg -n "postgres_url" src/app/shared/langchain_layer/agents/memory/` → **0 hits**.
- [x] 4.7 Replace the `dict[str, Any]` startup return with a typed configuration result, so the health probe has
      something to assert on and no credential-shaped field exists to leak.
      **Proof:** `uv run ty check src/ 2>&1 | tail -1` → ≤ 46; a test asserting the returned object is the model type
      and exposes no password or URL field.
- [x] 4.8 Guard the startup call. `lifespan.py:206` `await setup_cognee(settings)` is today the **only** unguarded
      optional-subsystem call in that file (Graphiti `:211-223`, Crawl4AI `:258`, object storage `:266`, Celery
      `:273`, outbox `:284` all degrade; commit `1b3891f` rewrote 121 lines there and skipped `:206`). Wrap it in the
      Graphiti shape, set `app.state.cognee_config = None` on failure, and **re-raise the dimension/model-mismatch
      class** — the one failure Decision 15 says must stop the boot.
      **Proof:** two lifespan tests with a faked setup — raising `ConnectionError` leaves
      `app.state.cognee_config is None` and startup completes; raising the mismatch error propagates and startup
      fails. Plus `rg -n -B2 -A6 "setup_cognee\(settings\)" src/app/lifecycle/lifespan.py` shows the `try/except`.

## 5. Observability — before any behaviour depends on it

- [x] 5.1 Add `check_cognee` to `src/app/middleware/health_check.py` and register it in `ALL_PROBES` (`:93-99`, where
      `check_graphiti` is already the fifth entry at `:98` — **not** this change's concern, Decision 7). It must
      distinguish three states: *degraded* when configuration is absent, *fail* when configured but unreachable,
      *ok* otherwise. A boolean cannot express this subsystem's failure mode.
      **Proof:** `rg -n "check_cognee" src/app/middleware/health_check.py` → definition **and** an `ALL_PROBES` entry;
      three tests, one per state, asserting the reported status string.
- [x] 5.2 Add the same probe to the second surface, `src/app/features/health/service.py`, under a field name that
      does **not** collide with its existing `memory` key — that key is `_check_memory()` (`:69`, defined
      `:200-213`), which is psutil RAM and unrelated (N6). Both surfaces must report the same state.
      **Proof:** a test asserting the response holds **both** the psutil `memory` field and the new agent-memory
      field with distinct values, and a test asserting the two surfaces return the same state for the subsystem.
- [x] 5.3 Report the graph-procedure precondition as a **named sub-field**, and do not fail the whole check when it is
      absent — that precondition is the only way item 140's silent consolidation failure is ever observed.
      **Proof:** a test with a faked graph driver reporting no APOC/GDS: the sub-field is `false` while the overall
      check does **not** return `fail`.

## 6. The memory service — the repository's first memory call site

- [x] 6.1 Replace the three bare partition-name interpolations (`cognee_client.py:140,189,238`) with **one validated
      construction helper**. With access control unavailable (NG6) this name is the *only* tenant boundary.
      **Proof:** `rg -n "legal_reports" src/` → a single construction site; a test asserting two tenants never
      produce the same identity and that an identity failing validation raises rather than defaulting.
- [x] 6.2 Implement the conversation-scoped report write: a conversation identity is required, and self-improvement is
      **disabled** so no detached `asyncio.create_task` bridge (`remember.py:898`) is started inside the caller's
      event loop.
      **Proof:** a test with a faked memory module asserting the write is called with the run's conversation identity
      **and** self-improvement disabled; a second asserting no background task is created (`asyncio.all_tasks()`
      unchanged across the call).
- [x] 6.3 Implement the typed trace / QA / feedback writes, and reject a write with no conversation identity **before
      it reaches the library** — the library raises `session_id is required for typed memory entries`
      (`remember.py:274-276`), and that must surface as a caller error, not a memory-store failure.
      **Proof:** a test asserting the rejection raises the caller-error type **and** that the faked memory module was
      never called at all.
- [x] 6.4 Implement recall with a **full model dump** and the origin field preserved — the existing
      `[dict(r) for r in results]` (`cognee_client.py:259`) is a shallow conversion that leaves nested models as
      objects, which type-checks and then fails at serialisation time.
      **Proof:** a test asserting `json.dumps(result)` round-trips every returned item and that the field
      distinguishing a conversation-cache hit from a permanent-graph hit survives.
- [x] 6.5 Implement consolidation as the **only** caller of enrichment, and report what it consolidated (conversation
      count and resulting memory size — the single safeguard against D10's unbounded growth, which makes growth
      *observable*, not bounded).
      **Proof:** `rg -n "cognify\(" src/` → **0 hits** (Trap3); `rg -n "improve\(" src/` → exactly one call site, and
      it is inside the consolidation method; a test asserting the returned report carries both counters.
- [x] 6.6 Make consolidation abort with a **named precondition failure** when the required graph procedures are
      absent — the underlying rebuild fails without raising, so silence would read as success.
      **Proof:** a test with a faked graph reporting no APOC/GDS: consolidation raises the named error and does
      **not** return a success result.
- [x] 6.7 Settle on one failure idiom across the service, keeping `e.add_note()` before re-raise (house style, live at
      `cognee_client.py:251`), replacing the three that coexist today: re-raise (`:159`), swallow-to-empty-list
      (`:257`), collect-error-strings (`write_final_report.py:156-161`).
      **Proof:** `uv run ruff check src/ 2>&1 | tail -1` ≤ 123 and `uv run ty check src/ 2>&1 | tail -1` ≤ 46; a test
      asserting a store failure surfaces as the chosen idiom rather than an empty list.
- [x] 6.8 Guard against the prune operation ever being called — the permanent memory graph and the document entity
      graph share one Neo4j instance, so a prune there destroys the other library's data (ADR § Consequences).
      **Proof:** `rg -n "prune" src/` → **0 hits**.

## 7. The write seam

- [x] 7.1 Retarget the memory-persist node onto the service, gated on human approval.
      **Proof:** two node tests — an approved run calls the service write exactly once; an unapproved run calls it
      **zero** times at any trust level and still completes successfully.
- [x] 7.2 Preserve the fail-open shape: a memory failure records `COGNEE_WRITE_FAILED` (already present at
      `agent_saul/nodes.py:802`) and never aborts a completed legal analysis.
      **Proof:** a test where the service raises: the node returns, the run's errors contain `COGNEE_WRITE_FAILED`,
      and no exception propagates out of the node.
- [x] 7.3 Stop writing the final report to the document knowledge graph — the accepted boundary expressed in code.
      **Proof:** a test asserting no knowledge-graph client method is called during memory persistence;
      `rg -n "write_final_report_episode" src/` → **0 hits** after group 10.

## 8. The read seam — speculative by construction (D17, NG10)

- [x] 8.1 Harvest the two helpers that have **no other implementation in the repository** into
      `shared/langchain_layer/messages.py` **before** anything is deleted: the tool-message filter
      (`memory_pipeline.py:129-157`) and the structured context-prefix builder (`:160-201`). Do **not** harvest the
      trim step (`:109-116`) — it duplicates `messages.py:40-52`, same counter and strategy, so deleting it is pure
      subtraction.
      **Proof:** `uv run python -c "from app.shared.langchain_layer.messages import <filter>, <prefix_builder>"`
      succeeds; `rg -c "def .*trim" src/app/shared/langchain_layer/messages.py` → 1.
- [x] 8.2 Add the prefetch node after clarification: agent memory queried first, with a **bounded** knowledge-graph
      supplement gated on **three** task values — `risk_analysis`, `obligation_chain`, `compliance`
      (`memory_pipeline.py:213,220`, inside `_do_retrieve_graphiti_context`). `obligation_chain` **keeps**
      eligibility; dropping it would be a silent regression smuggled in under a relocation (B5, Decision 10).
      **Proof:** a parametrised test over the three eligible values plus at least two ineligible ones — the eligible
      ones fetch a supplement, the ineligible ones fetch none and the run proceeds on memory alone.
- [x] 8.3 Keep the read path fail-open (the pattern already at `memory_pipeline.py:258-260`).
      **Proof:** a test where recall raises during prefetch: the node returns and the run continues with current-run
      context plus any supplement already obtained.
- [x] 8.4 Add the deeper-retrieval **service operation** with its role restriction: available to the risk-analysis and
      compliance roles only, returning the caller's own partition only, and **refusing with a named reason** rather
      than returning an empty result when called by any other role or without a partition identity.
      **Proof:** four tests — risk analysis permitted; compliance permitted; the orchestrating role refused; a missing
      partition identity refused — and in both refusal cases the result is **not** an empty list.
      **Dependency (change 3, coordination point C-A):** binding this operation to a tool name and assigning it to
      exactly those two roles belongs to change 3's `agent-tool-registry` / `agent-tool-contract` and is **not done
      here** (D6.1 — no second tool-registration path). Until change 3 lands, no reasoning node can invoke it.
- [x] 8.5 Do not make the agent graph reachable, and do not let this seam make re-enabling it harder (D17, NG10).
      **Proof:** `rg -n "build_saul_graph" src/` still shows the definition (`agent_saul/graph.py:86`) with **no
      caller**; `uv run python -c "import app.main"` succeeds. Proofs for this whole group are import-, type- and
      unit-level only — node reachability is **not** claimed.

## 9. Scheduled consolidation — registration only

- [x] 9.1 **Depends on change 0.** Confirm the reconciliation re-exports are gone from
      `src/tasks/__init__.py:6-9,18-20`. Until they are, any worker importing the task package dies at import and
      registration cannot be proven at all.
      **Proof:** `rg -n "memory_decay_reconciliation" src/tasks/__init__.py` → **0 hits**;
      `uv run python -c "import src.tasks"` succeeds.
- [x] 9.2 Add the consolidation task with a real `@celery_app.task` decorator (the reconciliation module never had
      one) and register its module in the `include` list (`connections/celery.py:191-196`, 4 entries today).
      **Proof:** `uv run python -c "from app.connections.celery import celery_app;
      print(sorted(n for n in celery_app.tasks if 'memory' in n)); print(len(celery_app.conf.include))"` lists the new
      task name and shows **5** include entries.
- [x] 9.3 Add one beat entry on a nightly schedule, named distinctly from the existing billing reconciliation entry —
      they share only the word "reconciliation" (`beat_schedule` at `:259-276`, 4 billing entries today).
      **Proof:** `uv run python -c "from app.connections.celery import celery_app;
      bs=celery_app.conf.beat_schedule; print(len(bs), sorted(bs))"` → 5 entries, and the new key is not a prefix or
      suffix of the billing one.
- [ ] 9.4 **Depends on change 1 — and this task stays open when the rest of the change is done.** There is no worker
      and no beat service in `docker-compose.yml` at all, and `Makefile:52` starts one from a `celery_config` module
      that does not exist; that runtime gap is dispositioned **in change 1** (`dispositions.md` 198.4), not here.
      Record in the change log that **the beat entry this change adds is inert on the day it lands** (NG14,
      coordination point C-B).
      **Proof:** `docker compose config --services` lists no worker and no beat service — so **no execution proof is
      claimed by this change**. Registration (9.2, 9.3) is the whole of what is provable today, and nothing in the
      consolidation requirement may be read as evidence that a consolidation has ever run.

## 10. Harvest complete, then delete — and dispose of the superseded change

- [x] 10.1 Delete, in **one commit with their paired re-export edits**: `write_final_report.py`,
      `memory_pipeline.py`, `CogneeStore`, the three legacy module-level functions, and the knowledge-graph
      final-report episode writer (`rag/graphiti/client.py:311-350`) with its result model. The re-exports are
      `rag/graphiti/__init__.py:47,59` and `memory/__init__.py:3-9,23-39`; missing either yields `ImportError`
      **at boot**, which no unit test can see.
      **Proof:** `uv run python -c "import app.main"` succeeds — this is the load-bearing proof for the whole group,
      because the reconciliation and memory modules have **zero** test coverage and a green suite proves nothing
      about a deletion.
- [x] 10.2 Delete caller and callee together. `write_final_report.py:122,146` call the memory functions through the
      structural `CogneeService` interface declared at `:41-50` — a duck-typed edge `graphify affected` does **not**
      surface (N5), so a deletion that removes one side leaves a break no graph query warned about.
      **Proof:** `rg -n "store_final_report|store_relationships|search_episodic_memory|CogneeStore" src/ tests/` →
      **0 hits**.
- [x] 10.3 Confirm no count regressed across the deletions.
      **Proof:** `uv run ruff check src/ 2>&1 | tail -1` ≤ **123**; `uv run ty check src/ 2>&1 | tail -1` ≤ **46**;
      `uv run pytest 2>&1 | tail -3` summary line ≥ **55 passed** (read the summary line — the coverage gate makes a
      green run exit 1).
- [x] 10.4 **Archive** `openspec/changes/cognee-saul-memory-migration` — **never delete it.** Its `proposal.md:20-21`
      (*"Cognee v1.1 has no built-in curation/decay/dedup"*) is the primary citation for D10's recorded gap and must
      stay quotable. Record `superseded-by` on the archived `.openspec.yaml` **and** this change's, so the link is
      discoverable from both ends. If task 1.5 found that `openspec archive` refuses a 0/15-task change, move the
      directory by hand and record that in `review.md` — do **not** tick 15 tasks that were never done.
      **Proof:** `ls openspec/changes/archive/ | rg cognee-saul-memory-migration` shows a `YYYY-MM-DD-` prefixed
      directory; `rg -n "superseded" openspec/changes/archive/*cognee-saul-memory-migration/.openspec.yaml
      openspec/changes/cognee-agent-memory/.openspec.yaml` returns a line from **both**.
- [x] 10.5 Re-validate, and confirm the archive of the superseded change did not add a seventh failure — the new
      capability carries a **real** `## Purpose`, precisely so the archive flow's stub habit does not.
      **Proof:** `openspec validate cognee-agent-memory --type change --strict` → `is valid`;
      `openspec validate --all 2>&1 | tail -1` → failures ≤ **6** (21/6 of 27 today; 22/5 if task 2.1 landed). The
      **failure count is the invariant** — the pass count moves as sibling changes are authored and is never an
      acceptance number.
- [ ] 10.6 Run the one manual round-trip against a **non-production** instance. It is the only check in this change
      that can detect the silent rebuild failure, and the only evidence that these operations have *ever* succeeded
      here — there are no call sites, no tests and no dataset artifact to compare against.
      **Proof:** the observable transition, not parity with code that never ran — (a) write one conversation-scoped
      entry; (b) recall **with** the conversation scope returns a conversation-cache hit; (c) run the consolidation
      task once, by hand, since no worker exists (9.4); (d) recall **without** the conversation scope returns a
      permanent-graph hit. Record the four outputs. If (d) returns nothing while (c) reported success, task 1.1's
      graph precondition is absent and consolidation is failing silently.


---

## Execution record — band F, 2026-08-23/24

All groups executed against the handover and measured on this tree. Deviations and answers:

* **1.1** The configured Neo4j instance does not DNS-resolve from the execution environment — unreachable
  beats absent, so the pre-decided write-only branch applies. Answers recorded in `design.md` § Open Questions.
* **1.2 / 1.3 / 1.4** `create_schema=True vector_available=1 memory_tables_present=0`; discrete-fields
  connection negotiates **TLSv1.3**, so no `database_connect_args` needed; accessor/settings agreement holds
  (`True True True`) but is now asserted at startup so hand-synced env files cannot drift.
* **1.5** `openspec archive` has no `--dry-run`; the real archive of the superseded change succeeded despite
  its 0/15 tasks — the CLI accepts it. Archived as `2026-08-24-cognee-saul-memory-migration` with
  `superseded-by:` / `supersedes:` links written from BOTH ends.
* **2.1** Making `cognee-v1-api` valid surfaced a latent delta defect: three MODIFIED blocks omitted scenarios
  the current spec still has (the §9 trap, live). Carried forward with bodies restated to this change's reality.
  Validate totals moved **21/6 → 22/5 → 23/5(28)** — the failure count only fell, never rose.
* **4.x** Typed credential-free `CogneeSetupConfig`; env key written before the first config call (test pins
  order); embedding dimension pinned to the document width with boot-stopping mismatch error; placeholder and
  divergent Postgres identities refused with named errors before any library call.
* **6.x / 8.x** Service + read seam implemented; tests pin: no background task on writes, typed-entry rejection
  before the library, JSON round-trip recall with origin preserved, consolidation refusing loudly without graph
  procedures, zero prune surface, deeper-retrieval role refusals returning errors not empty lists.
* **7.x** Persist node retargeted onto the service, approval-gated, fail-open `COGNEE_WRITE_FAILED`, and it
  never touches a knowledge-graph client (hostile-probe test).
* **9.2/9.3 amended counts** — the task text expected include=5 and beat=5, measured at authoring. Measured now:
  **include=8, beat=7** (billing/credits grew after this change was authored). Registration verified by import;
  the task also registers its payload model in the typed registry, which the original plan omitted.
* **10.2 lexical-trap noted**: the Proof pattern `store_relationships` substring-matches the unrelated local
  `_store_relationships` in `ingestion_kb/nodes.py`. Different symbol, different owner — left untouched; a
  structural check (import graph) confirms no caller of the deleted functions remains.
* **10.6 NOT executed** — needs a reachable non-production graph instance and a manual beat run; neither exists
  here (see 1.1). This box intentionally stays unchecked; nothing in this change may be read as evidence that a
  consolidation has ever succeeded.
