# Tasks — agent-tools-unification

> Change class: **L** (grouped sections, each task independently verifiable). Authored 2026-08-18, after `review.md`
> moved off `CHANGES-REQUESTED` — every blocking finding is answered in `review.md` § Author response.
> **Read before starting:** `design.md` § Migration Plan is the ordering and the reason for it. The eleven groups below
> follow its four phases: group 2 is phase 1, groups 3–5 are phase 2, groups 6–10 are phase 3, group 11 is phase 4.
>
> **Six standing rules for every proof in this file.**
> 1. **Compare the summary line, never `$?`.** `pyproject.toml:752-760` puts `--cov-fail-under=80` in `addopts`
>    against 18.38% coverage, so a green suite still exits 1. Baselines: **`pytest` ≥ 55 passed**,
>    **`ruff check` ≤ 123** (**≤121** once change 0 deletes `todo_temp.py`), **`ty check` ≤ 46**,
>    **`ast-grep scan` 4 errors at exit 0**, **`openspec validate --all` failures ≤ 6**.
> 2. **No Proof may execute the application's agent graph** (D17). Construction-path tasks prove by `ty`, by import,
>    and by a unit test against the constructor. The six runtime scenarios with no import- or type-level witness use
>    the **throwaway two-node `StateGraph` + `InMemorySaver` built inside the test** that D-10 authorises — and
>    nothing else.
> 3. **No task enables the commented lifespan wiring, and no task adds a flag that defaults on** (D17). Task 1.4
>    records that the blocks are commented; task 11.4 re-checks it. If a task appears to need the wiring live, it is
>    mis-scoped — stop and escalate.
> 4. **Never print a credential.** Any probe that touches a connection prints host/port/database only.
> 5. **No Proof may depend on an outbox event firing.** `outbox_events` and `dead_letter_events` do not exist
>    (`findings-database.md` §8, change 0 owns that). If a later task needs one, its Proof is blocked on change 0 and
>    must say so.
> 6. **Tool-error work targets the middleware seam, never `ToolNode(handle_tool_errors=…)`.** That parameter is
>    unreachable through `create_agent` at langchain 1.2.12 (Q2, closed). Writing against it produces code that cannot
>    run.
>
> **Two cross-change dependencies, both directional and both named.**
> - **Group 2 is a blocking predecessor of change 0** (`cleanup-foundation`). Change 0 must not delete
>   `src/app/shared/agents/**` until 2.4 reports zero importers, or `graphiti/registry.py:40-45`'s eager imports raise
>   `ImportError` at boot. Change 0 may cite **task 2.4** by number as its precondition.
> - **Group 11 is blocked by change 2** (`documents-unified-schema`). Its requirement is unsatisfiable until change 2's
>   retrieval-schema migration lands, so 11.1 proves at import level today and 11.2 is explicitly marked blocked. A
>   Proof asserting an index-served lookup **today** would be unexecutable — do not write one.

## 1. Preconditions — read-only, no edits

- [ ] 1.1 Record every gate baseline **as measured at task start**, not as quoted from `design.md`. Tasks that run
      before change 0 compare against these numbers, not against the post-change-0 gate.
      **Proof:** `uv run ty check src/ 2>&1 | tail -1; uv run ruff check src/ 2>&1 | tail -1;
      uv run pytest 2>&1 | tail -1; ast-grep scan src/ 2>&1 | tail -3;
      /home/harmeet/.bun/bin/openspec validate --all 2>&1 | tail -3` — paste the five lines into the PR body.
- [ ] 1.2 Confirm the defect surface still matches the design: **four** envelope definitions and **three**
      `ToolRegistry` classes.
      **Proof:** `rg -n "class Tool(Result|Output)\b" src/` prints 4 lines
      (`shared/agents/tools/idempotency.py:11`, `langchain_layer/agents/tools/base.py:30`,
      `langchain_layer/agents/tools/idempotency.py:34`, `rag/document_processing/models.py:318`);
      `rg -n "^class ToolRegistry" src/` prints 3. If either count differs, stop — the plan's arithmetic is stale.
- [ ] 1.3 Confirm the shadow package is still present, so group 2 is still a bug fix and still precedes change 0.
      **Proof:** `test -f src/app/shared/agents/tools/idempotency.py && rg -c "app\.shared\.agents\." src/` prints a
      non-zero count.
- [ ] 1.4 Record that the lifespan wiring is **commented** and that this change leaves it that way (D17 precondition).
      **Proof:** `rg -n "^\s*#.*(agent|tool).*(router|lifespan|include_router)" src/app/lifecycle/lifespan.py
      src/app/main.py` — capture the output verbatim; task 11.4 diffs against it.
- [ ] 1.5 Confirm `ToolRetryMiddleware` is already wired to the survivor factory, so group 9 installs an existing
      mechanism rather than designing one (Q2).
      **Proof:** `rg -n "ToolRetryMiddleware" src/app/shared/langchain_layer/agents/middlewares/guardrails.py` prints
      `:345` and `:369`; `rg -n "middleware=" src/app/shared/langchain_layer/agents/factory.py` prints `:188`;
      `rg -c "middleware" src/app/shared/langgraph_layer/agent_saul/factory.py` prints **0**.

## 2. Phase 1 — the cycle predecessor. **Blocking predecessor of change 0.**

Both files already call the survivor API; only their import lines point at the shadow. Zero call-site edits are
expected — if one is needed, the premise in `design.md` phase 1 is wrong and the discrepancy goes in the PR body.

- [x] 2.1 Rewrite `src/app/features/legal/.../get_obligation_chain.py:29` to import `IdempotencyGuard` / `ToolResult` /
      `MemoryScope` from the D6/D6.1 survivor modules instead of `app.shared.agents.*`.
      **Proof:** `rg -n "app\.shared\.agents\." <file>` prints nothing;
      `uv run python -c "import <module>"` exits 0; the call sites at `:67` are unedited
      (`git diff -U0 <file> | rg "^[-+]" | rg -v "^[-+]{3}"` shows import lines only).
- [x] 2.2 Rewrite `precedent_tools.py:21,22` the same way. Its scope usage already assumes the **real** `MemoryScope`
      (`scope.top_k` at `:104`, `scope=scope` into `expand_from_seeds` at `:115`, call opening `:113`) against a
      one-line `str` stub, so this is the fix, not a rename.
      **Proof:** `rg -n "app\.shared\.agents\." src/app/.../precedent_tools.py` prints nothing;
      `uv run python -c "import <module>"` exits 0; `git diff` touches only lines 21–22.
- [x] 2.3 Drive the `ty` gate. Expect **≤31** from 46 (15 diagnostics localised to these two files).
      **If diagnostics survive, the task does not fail — enumerate the residue by file and line and record the gate as
      ≤31+N with the residue named.** Do not suppress with `# type: ignore`.
      **Proof:** `uv run ty check src/ 2>&1 | tail -1` shows ≤31, or the residue list is in the PR body.
- [x] 2.4 **Report the predecessor as discharged.** This is the task change 0 waits on.

      **Verified discharged 2026-08-23** (measured at band E execution): `rg -c "app\.shared\.agents\." src/`
      prints nothing; `uv run python -c "import app.main"` exits 0; `uv run ty check src/` reports
      **All checks passed** (0 diagnostics — far under the ≤31 gate). Band A's `6525c6f` did the rewrite;
      these boxes were left unchecked only because no one had re-measured.
      **Proof:** `rg -c "app\.shared\.agents\." src/` prints **0**, and
      `uv run python -c "import app.main"` exits 0. Post both lines to change 0.

## 3. Phase 2 — registry: populate, adopt, rename

- [ ] 3.1 Add an **explicit registration entry point** that populates the registry. Import must remain side-effect
      free (D-1(c); the spec requirement is *"populated by explicit registration before any consumer resolves a
      tool"*).
      **Proof:** one command asserting both halves —
      `uv run python -c "import app.shared.langchain_layer.agents.tools as t; r=t.get_tool_registry();
      assert len(r) == 0, 'import must register nothing'; t.register_default_tools();
      assert len(r) > 0; print('empty-on-import then populated: OK')"`
- [ ] 3.2 Make registration **idempotent** and make resolution of an unregistered name **fail loudly**.
      **Proof:** `uv run python -c "...; register_default_tools(); n=len(r); register_default_tools();
      assert len(r)==n; import pytest; pytest.raises(KeyError, r.get, 'no_such_tool'); print('OK')"`
- [ ] 3.3 Tag the tools, including `web` on `web_search.py:80` and `crawl.py:114`, so `by_tags` selection is real.
      **Proof:** `uv run python -c "...; register_default_tools();
      assert {'web'} <= set(r.tags()); assert len(r.by_tags('web')) >= 2; print(sorted(r.tags()))"`
- [ ] 3.4 Adopt the registry in `agents/factory.py`, making the string branch at `:146`
      (`resolved_tools.append(get_tool_registry().get(t))`) reachable for the first time.
      **Proof:** a unit test builds a spec with a **string** tool name and asserts a resolved tool object comes back;
      `uv run pytest tests/ -k registry_adoption 2>&1 | tail -1` shows the new tests passing.
- [ ] 3.5 Rename the Graphiti bundle to **`AgentToolBundle`** per ADR 2, keeping `ToolRegistry = AgentToolBundle` as a
      deprecation alias **for one commit only**, sequenced after 3.4.
      **Proof:** `rg -c "^class ToolRegistry" src/` prints **1**;
      `uv run python -c "from app.shared.rag.graphiti.registry import AgentToolBundle, ToolRegistry;
      assert ToolRegistry is AgentToolBundle; print('alias OK')"`
- [ ] 3.6 Remove the alias and fix the misleading docstrings at `graphiti/registry.py:9,25`.
      **Proof:** `rg -c "ToolRegistry" src/app/shared/rag/graphiti/registry.py` prints **0**;
      `uv run ty check src/ 2>&1 | tail -1` shows **≤28**; `uv run python -c "import app.main"` exits 0.

## 4. Phase 2 — one envelope, four definitions to one

- [ ] 4.1 Define the survivor envelope with its success, failure and **unavailability** constructors. The unavailability
      constructor is what the honesty work in group 6 requires, so it lands first.
      **Proof:** `uv run python -c "from <survivor> import ToolResult as R;
      assert R.ok(data={}).success and not R.fail(error='x').success and R.unavailable(reason='y').unavailable;
      print('three constructors OK')"`
- [ ] 4.2 Rewrite `shell.py`'s **13** `ToolOutput` sites
      (`:4,18,68,71,106,108,111,126,129,145,155,158,216`) onto the survivor, and **delete `to_agent_string()`**
      (`base.py:46`) — the self-rendering method that turned an envelope back into `f"ERROR: {self.error}"`.
      **Proof:** `rg -c "ToolOutput|to_agent_string" src/` prints **0**;
      `uv run pytest tests/ -k shell 2>&1 | tail -1` passes.
- [ ] 4.3 Delete the remaining definitions: `base.py:30` and `document_processing/models.py:318`. **`shared/agents/tools/idempotency.py:11`
      is deleted by change 0**, not here — group 2 already removed its importers.
      **Proof:** `rg -n "class Tool(Result|Output)\b" src/` prints exactly **1** line.
- [ ] 4.4 Prove no envelope renders itself, in any name (the F1 defect was shape, not name).
      **Proof:** `ast-grep -p 'def to_agent_string($$$)' src/` prints nothing, and
      `rg -n 'return f"(ERROR|Error|Search error)' src/app/shared/langchain_layer src/app/features` prints nothing.
- [ ] 4.5 Hold the envelope gate.
      **Proof:** `rg -c "class Tool(Result|Output)\b" src/` prints **1** (was 4);
      `uv run ty check src/ 2>&1 | tail -1` shows no increase; `uv run python -c "import app.main"` exits 0.

## 5. Phase 2 — the idempotency key contract (D-4)

- [ ] 5.1 Make `make_key` (`idempotency.py:65-76`) keyword-only with an explicit `structural: dict` and optional
      `content: dict | None`. The opaque `input_data` dict is what allowed the drift.
      **Proof:** `uv run python -c "import inspect; from <mod> import IdempotencyGuard as G;
      p=inspect.signature(G.make_key).parameters;
      assert all(v.kind is v.KEYWORD_ONLY for k,v in p.items() if k!='self');
      assert 'structural' in p and 'content' in p; print(inspect.signature(G.make_key))"`
- [ ] 5.2 Move the read/search callers onto `structural=` **plus** canonicalised `content=`, and the write path
      (`graphiti/write_clause_episodes.py` — `:35` is the *import* of `IdempotencyGuard`; **locate the `make_key` call
      yourself, do not trust `:35`**) onto `content=None`.
      **Proof:** `rg -n "make_key\(" src/` shows every call using keywords;
      `rg -n "make_key\([^)]*content=None" src/app/shared/rag/graphiti/` matches the write path;
      a unit test asserts two differently-worded search queries produce **different** keys and that one write
      replayed twice produces the **same** key.
- [ ] 5.3 Bump `_REDIS_KEY_PREFIX` (`idempotency.py:31` — distinct from `_POSTGRES_TTL_DAYS` at `:30`). **One cold
      cache is accepted; no dual read.**
      **Proof:** `rg -n "_REDIS_KEY_PREFIX" src/` shows the bumped literal and exactly one definition;
      `rg -c "_REDIS_KEY_PREFIX" src/` confirms no second prefix was introduced for a fallback read.

## 6. Phase 3 — honesty: the unavailability register

This is the group that makes the rest shippable. Until it lands, a missing corpus is reported as an answer.

- [ ] 6.1 Replace the fabricating paths with the unavailability constructor from 4.1 at
      `retrieve_statute_section.py:170-172`, `search_legal_precedents.py:227-229` and `precedent_tools.py:221-240`.
      **Proof:** a unit test patches the corpus call to raise, then asserts the returned envelope is
      `unavailable` with a non-empty reason — **and asserts the result is not a `str`**:
      `uv run pytest tests/ -k unavailab 2>&1 | tail -1` passes.
- [ ] 6.2 Delete the docstring sentence at `search_legal_precedents.py:179-180` that licensed the whole failure class.
      Quote the sentence in the commit message so the deletion is reviewable.
      **Proof:** `sed -n '175,185p' src/app/.../search_legal_precedents.py` shows the sentence gone, and
      `uv run python -c "import <module>"` exits 0 (a docstring edit must not change behaviour).
- [ ] 6.3 Prove the anti-pattern is gone from production tool bodies repo-wide, with the one accepted exception
      quarantined under `src/app/examples/` by group 10.
      **Proof:** `rg -n 'f"(Search error|Error): \{e' src/app --glob '!examples/**'` prints nothing.
- [ ] 6.4 Confirm a catch site returns the envelope rather than a rendered sentence (the 7th scenario of the
      `typed-exception-handling` MODIFIED delta).
      **Proof:** `ast-grep -p 'except $_ as $E: return f"$$$"' src/app` prints nothing.

## 7. Phase 3 — prompts (D-7, D-8)

- [ ] 7.1 Add the **kinded assembly seam**: sections are distinguished by *kind* — standing instruction, output
      contract, retrieved evidence, task restatement — not by label text. Ordering lives here, not in `build()`.
      **Proof:** a unit test passes sections in scrambled order and asserts the emitted order is
      instruction → contract → evidence → task, **and** asserts two different label strings with the same kind sort
      identically: `uv run pytest tests/ -k prompt_ordering 2>&1 | tail -1` passes.
- [ ] 7.2 Split the reusable preamble from per-turn content so retrieved evidence never enters the cacheable prefix,
      and pass evidence as a **ranked sequence**.
      **Proof:** a unit test asserts the preamble is byte-identical across two calls with different evidence, and that
      evidence order is preserved: `uv run pytest tests/ -k prompt_cache 2>&1 | tail -1` passes.
- [ ] 7.3 Leave `render_prompt_sections` (`prompts.py:145`) **untouched** for its **27** callers — the recorded gap in
      `agent-prompt-assembly`, not an oversight.
      **Proof:** `git diff --stat src/app/shared/langchain_layer/prompts.py` shows no change to that function, and
      `rg -c "render_prompt_sections" src/ | ...` still totals 27 external call sites.
- [ ] 7.4 Route tabular tool payloads through `serialize_to_toon` (~11 measured call sites) at the seam only.
      **Proof:** `uv run pytest tests/ -k toon 2>&1 | tail -1` passes; `uv run ty check src/ 2>&1 | tail -1` no increase.

## 8. Phase 3 — citations and declared output schemas (D-9, Q3)

- [ ] 8.1 Add the **non-empty validator** to the citation list. `Citation` (`agent_saul/state.py:103`) **already**
      carries `claim`, `source` and bounded `confidence` — do **not** redefine the type, or it forks (M-1).
      **Proof:** `uv run python -c "from app...state import Citation;
      import inspect; f={n for n in Citation.model_fields};
      assert {'claim','source','confidence'} <= f, f; print('fields pre-exist:', sorted(f))"` — this asserts the type
      was **not** redefined — plus a test asserting `citations=[]` raises `ValidationError`:
      `uv run pytest tests/ -k citation 2>&1 | tail -1` passes.
- [ ] 8.2 Declare the output schemas, and **assert the strategy actually selected** rather than assuming the native
      path. Per Q3 the configured `gemini-3.1-flash` / `gemini-3.1-pro` are absent from the profile table, so
      `AutoStrategy` silently degrades to `ToolStrategy`.
      **Proof:** a unit test asserts the resolved strategy type explicitly and records it in the assertion message, so
      a future model change surfaces as a diff rather than a silent upgrade:
      `uv run pytest tests/ -k output_strategy 2>&1 | tail -1` passes. If the test must construct a model, it passes
      the provider explicitly — `models.py:126-138`'s default `implementation="generic"` infers `google_vertexai`,
      which is not installed and raises `ImportError`.

## 9. Phase 3 — the retry seam, tool binding, and D-5

Q2 is closed: the mechanism exists and is wired to the survivor factory. The gap is `agent_saul`.

- [ ] 9.1 Install the existing middleware stack on `agent_saul`'s factory, which today has **zero** occurrences of
      `middleware`. Use `build_default_middleware_stack` / `ToolRetryMiddleware`; **do not** write
      `ToolNode(handle_tool_errors=…)` (unreachable through `create_agent`), and **do not** use the deprecated
      `on_failure="raise"` / `"return_message"` spellings.
      **Proof:** `rg -c "middleware" src/app/shared/langgraph_layer/agent_saul/factory.py` prints non-zero and
      `rg -n "handle_tool_errors" src/` prints nothing.
- [ ] 9.2 Expose the handoff helper as `transfer_to_<role>` tools tagged `handoff`, so the orchestrator becomes
      tool-using (this is what makes 9.3's gate reachable at all).
      **Proof:** `uv run python -c "...; register_default_tools();
      names={t.name for t in r.by_tags('handoff')};
      assert any(n.startswith('transfer_to_') for n in names); print(sorted(names))"`
- [ ] 9.3 Bind tools to all three roles, closing the three `tools=[]` sites at
      `agent_saul/factory.py:116,122,128` (the `create_agent` calls open at `:114,120,126`).
      **Proof:** `rg -c "tools=\[\]" src/app/shared/langgraph_layer/agent_saul/factory.py` prints **0**.
      **Keep this file-scoped** — a fourth, out-of-scope `tools=[]` lives at `agents/registry.py:149`, so a repo-wide
      count would never reach 0.
- [ ] 9.4 Add the hydration step and collapse the version constant to one definition (D-5).
      **Proof:** `rg -c "<VERSION_CONST>" src/` prints **1**; a unit test asserts a hydrated state carries the
      constant: `uv run pytest tests/ -k hydrat 2>&1 | tail -1` passes.
- [ ] 9.5 Prove the six runtime scenarios against the **throwaway two-node `StateGraph` + `InMemorySaver` built inside
      the test** (D-10). A raising tool must not terminate the run — note the library default re-raises everything
      except `ToolInvocationError` (`tool_node.py:379-387`), so this asserts the middleware, not the default.
      **Proof:** `uv run pytest tests/ -k throwaway_graph 2>&1 | tail -1` passes, and
      `rg -n "build_saul_graph|app.main" tests/<that file>` prints nothing — the test must not reach the application
      graph.

## 10. Q1 — relocate the abandoned RAG agent. **Pure move, no body edits.**

Closed by user decision: the file is **moved**, not deleted and not harvested. The two accepted losses are Non-Goals.

- [x] 10.1 `git mv src/app/shared/rag/rag_agent_advanced.py src/app/examples/rag_agent_advanced.py`.
      **Amended 2026-08-23:** done, but **not** as a pure rename — the todo-210 D14→F handover
      instructed repointing the five `embedder.embed_query` sites to
      ``app.shared.langchain_layer.embeddings.embed_text(..., task_type=QUERY)`` in the same leg, so
      the body changed and the R100 proof below no longer describes reality. The rename is still
      visible via `git log --follow`.
- [x] 10.2 Confirm nothing imported it and nothing does now (it had **zero** importers; its entry point is
      `run_cli()`).
      **Proof passed 2026-08-23:** `rg -n "rag_agent_advanced" src/ --glob '!examples/**'` prints only
      the embedder docstring note; `uv run python -c "import app.main"` exits 0.
- [x] 10.3 Record the two accepted losses as landed Non-Goals, not as future work. The
      `f"Search error: {e!s}"` strings survive (**quarantined**, not fixed) and the iterative-RAG prior
      art stays unused — **no harvest task exists, by decision**.
      **Proof amended 2026-08-23:** `rg -c 'Search error' src/app/examples/rag_agent_advanced.py`
      prints **4** (`:168`, `:237`, `:283`, `:471`) — the "re-measured … 5" claim here was itself off
      by one, confirmed against the pre-move blob at HEAD.

## 11. Phase 4 — the floating retarget (D-11). **Blocked by change 2.**

- [ ] 11.1 Write the statute point lookup against the retrieval-schema contract, reading the attribute names from
      change 2's `document-retrieval-schema` capability rather than inventing them.
      **Proof (runs today, import-level):** `uv run python -c "import <retarget module>; print('imports clean')"`
      exits 0, and `rg -n "FROM statutes" src/` prints nothing.
- [ ] 11.2 **BLOCKED ON CHANGE 2'S MIGRATION — do not attempt before it lands.** Verify the lookup is index-served.
      This Proof is unexecutable today because the relation does not exist in the deployed database; running it early
      produces a false failure.
      **Proof (after change 2's migration only):** `EXPLAIN` the point lookup and confirm an index scan on the
      identifying attributes — using the index **change 2 names**, not a name restated here. Print host/port/database
      only; **never print a password.**
- [ ] 11.3 Hold every gate one final time.
      **Proof:** `uv run ty check src/ 2>&1 | tail -1` ≤28; `uv run ruff check src/ 2>&1 | tail -1` **≤121** if change 0
      has landed, else no increase on 1.1's number; `uv run pytest 2>&1 | tail -1` **≥75 passed** with the same
      failures; `ast-grep scan src/ 2>&1 | tail -3` still **4**; `uv run python -c "import app.main"` exits 0.
- [ ] 11.4 Confirm D17 held: the lifespan wiring is still commented and no flag defaults on.
      **Proof:** re-run 1.4's command and diff against the captured output — identical;
      `git diff HEAD~<n> -- src/app/lifecycle/lifespan.py src/app/main.py` shows no uncommenting.
- [ ] 11.5 Re-validate the spec set. **Do not add a 7th failure.**
      **Proof:** `/home/harmeet/.bun/bin/openspec validate agent-tools-unification --type change --strict` prints
      valid, and `/home/harmeet/.bun/bin/openspec validate --all 2>&1 | tail -3` still shows **21 passed / 6 failed**
      of 27. `spec/typed-exception-handling` is a **pre-existing** failure of the deployed spec and is not caused by
      this change's MODIFIED delta.
