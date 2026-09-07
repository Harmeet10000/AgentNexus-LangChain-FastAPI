# Review — `agent-tools-unification` (change 3)

**Reviewer:** fresh adversarial reviewer (not the author) · **Date:** 2026-08-18
**Artifacts reviewed:** `proposal.md` (148), `design.md` (463), `adrs.md` (77), `specs/**` (7 capabilities,
31 requirements, 82 scenarios). No `tasks.md` exists — correct, it is gated behind this review.

**Verdict: CHANGES REQUESTED.** Two blocking findings, four material, several minor. The change is unusually
well-evidenced — most of its file:line claims check out and its decision records are genuinely load-bearing —
but it has one unrecorded omission of exactly the kind the gate exists to catch (F1), and one spec requirement
it cannot satisfy because the owning change never accepted the ask (F2).

---

## Measured baselines (independently re-run)

| Check | Change claims | Measured now | Verdict |
|---|---|---|---|
| `uv run ty check src/` | **46** | `Found 46 diagnostics` | ✅ **not invented** |
| `openspec validate agent-tools-unification --type change --strict` | passes | `Change 'agent-tools-unification' is valid`, exit 0 | ✅ |
| `openspec validate --all` | "no new failures beyond 6" | **21 passed, 6 failed (27 items)** — `spec/cognee-v1-api`, `change/mintlify-documentation`, `spec/noqa-documentation`, `spec/pattern-matching-standard`, `spec/transactional-outbox`, `spec/typed-exception-handling` | ✅ **6, not 7** — the identical failing set; `change/agent-tools-unification` passes |
| `openspec/specs/` capability count | proposal says "21 capabilities" | **20** | ⚠️ minor (M-3) |
| `design.md` gate row for `validate --all` | "16 passed / 6 failed of 22" | 21/6 of 27 | ⚠️ stale (M-4) |

---

## Blocking findings

### F1 — BLOCKING. A **fourth** tool-result class exists, is live, and is never mentioned: `ToolOutput`

`src/app/shared/langchain_layer/agents/tools/base.py:30` defines `class ToolOutput(BaseModel)` with
`ok()` (`:39`), `fail()` (`:43`) and `to_agent_string()`. It has **13 use sites**, all in
`src/app/shared/langchain_layer/agents/tools/shell.py` (`:68,71,106,108,111,126,129,145,155,158,216`).

Every artifact says "three competing definitions" / "one tool-result shape" and the design's own gate is
`rg -c "^class ToolResult" src/` → 3 → 1. That gate is **satisfiable while the defect survives**: `ToolOutput`
does not match `^class ToolResult`, so the change can pass its own measurable gate and still leave two result
shapes in the *same module* — `ToolOutput` at `base.py:30` and `ToolResult` at `idempotency.py:34` — 28 lines
apart, in the package whose export surface this change is deliberately correcting.

Three reasons this is blocking rather than cosmetic:

1. **`agent-tool-contract` Requirement "One result envelope for every agent tool"** is, as written, *false*
   after the change ships. Its scenarios ("A successful tool call returns the common envelope") are not
   satisfied by the five `shell.py` tools.
2. **These are exactly the tools D-1 populates.** D-1 step 1 is "register the five decorated tools from
   `shell.py`". So the change's first shape task deliberately promotes into the registry of record the five
   tools that return the *unrecorded* envelope. The two decisions collide and neither notices.
3. **`shell.py` is a live instance of the anti-pattern the ADR is built on.** `ToolOutput.fail(str(exc)).to_agent_string()`
   at `shell.py:71,111,129,158` returns a *string* to the model — the ADR condemns precisely this ("an exception
   … gets caught and re-stringified into the model's context, which is the string-as-error anti-pattern being
   eliminated") but cites only `rag_agent_advanced.py:169,241,290,342,478`, a **zero-importer** module, while
   missing the one in the package it is unifying.

**Required:** either bring `ToolOutput` into the collapse (and change the gate to something that catches it,
e.g. `rg -c "class Tool(Result|Output)\b" src/`), or record it as an explicit Non-Goal with a reason — but the
`agent-tool-contract` requirement must then be narrowed so it is not stating something false. A silent omission
is not available: `dispositions.md` Up#10 is scoped to `ToolResult`, so nothing upstream authorises leaving a
fourth shape unaddressed.

### F2 — BLOCKING. `legal-corpus-retrieval` carries a requirement this change cannot satisfy and change 2 never accepted

`legal-corpus-retrieval` Requirement **"Statute identity attributes are addressable and efficiently
retrievable"** (spec `:29-48`) mandates that *the corpus* carry `act_name` / `section_ref` / `year` under a
documented contract and that a point lookup on the first two be **index-served**. Both are DDL.

- `proposal.md:90-91` and `design.md:373` state the change ships **no migration** and **no DDL**.
- `design.md:309-313` files these as "asks **A1** / **A2** against change 2".
- Change 2 (`openspec/changes/documents-unified-schema/`) contains **zero** occurrences of `act_name`,
  `section_ref`, `A1`, `A2`, or `agent-tools-unification` (grepped). Its two capabilities are
  `document-retrieval-schema` and `llm-injection`. The asks were never filed anywhere the owning change can see.

So change 3 asserts a normative requirement whose only implementer neither knows about nor has scheduled. This
is the "ask floats in prose" failure mode. **Required:** either move the two schema clauses into change 2's
`document-retrieval-schema` as a delta (and reduce change 3's requirement to *consuming* the contract), or add
them to change 3's Non-Goals with an explicit predecessor citation. As written, the requirement is
unimplementable inside its own change and invisible to the change that could implement it.

### F3 — BLOCKING. `agent-tool-registry` mandates the population mechanism `design.md` D-1 explicitly rejected

`specs/agent-tool-registry/spec.md:27-35`:

> **Requirement: The registry is populated deterministically at import** — "The registry SHALL contain every tool
> intended to be resolvable **as soon as the agent tools package has been imported**. Population SHALL NOT depend
> on some other module happening to be imported first."
> *Scenario:* "**WHEN** the agent tools package is imported **and nothing else** — **THEN** the registry SHALL report
> a non-empty set of registered tool names"

`design.md:108-112`, D-1 alternative (c):

> "**Populate by importing `shell.py` from the package `__init__`** — **rejected** in favour of **explicit
> registration**: importing a module nothing has ever imported to obtain a decorator side effect makes package
> import order load-bearing, and would run that module's top-level code in every process including Celery workers."

The five tools are registered *only* by `@register_tool` decorators executing at `shell.py` import
(`shell.py:41,96,114,132,174` — verified; `base.py:138` is where the decorator calls `registry.register`). Verified:
`shell.py` has **zero importers** anywhere in `src/` or `tests/`.

So the only mechanism that makes "the package imported and nothing else ⇒ non-empty" true is an import-time side
effect inside `tools/__init__.py` — the rejected alternative. Conversely, under "explicit registration" the registry
is empty until some bootstrap function runs, and the scenario fails. **The requirement title itself ("at import")
names the rejected design.** An implementer must violate one artifact to satisfy the other, and the failure mode is
the exact one D-1's whole safety argument is built to avoid.

Compounding it: D-1 promises `get_all_tools`/`get_web_tools` become "thin aliases over `registry.by_tags("web")`"
(a real method — verified at `base.py:82`), but nothing in `shell.py`'s five tags (`system`, `shell`, `filesystem`,
`read`, `write`, `list`, `search`) is `web`; `web_search`/`crawl_url` are *not* decorated and must be registered by
whatever the resolution to this contradiction turns out to be. The spec's "Web-capable tools are reachable by their
tag" therefore also depends on it.

**Required:** pick one. Either narrow the requirement to "populated before any consumer resolves a tool" (explicit
registration, order-independent, provable by a bootstrap unit test), or overturn D-1(c) and accept the import side
effect — in which case the Celery-worker top-level-code concern D-1 raises needs an answer, not silence.

---

## Material findings (fix before `tasks.md`, not necessarily re-review)

### F4 — The headline defect claim about `factory.py:146` is **reversed**, and D-1's hazard argument rests on the reversed version

`proposal.md:33-34`:

> "`agents/factory.py:146` calls `.get_tool(...)` on a class that defines only `.get` (an `AttributeError` on first use)"

`design.md:100-101`:

> "`factory.py:146` currently calls `get_tool_registry().get_tool(t)` which returns `None` on miss, and the
> survivor's `get` **raises `KeyError`** (`base.py:73`). Against an empty registry that converts a silent miss into
> a hard failure for *every* tool name."

Actual code, `src/app/shared/langchain_layer/agents/factory.py:146`:

```python
resolved_tools.append(get_tool_registry().get(t))
```

and `src/app/shared/langchain_layer/agents/tools/registry.py:9`'s class defines `get_tools`, `get_tool`,
`get_search_tool`, `get_crawl_tool` — and **no `get`**. The direction is exactly inverted: the call site uses `.get`,
the loser class provides only `.get_tool`.

The *conclusion* (a live `AttributeError` at `:146`) is correct, and `design.md:102-103` states the correct version
two lines later — the design contradicts itself inside one paragraph. But the consequence is not cosmetic:

- **There is no "silent miss" today.** `factory.py:146` cannot reach `get_tool`'s `return None`, so the "silent
  `None` → `KeyError`" framing in the proposal's Impact, the Risks section, and D-1's rationale describes a state
  that does not exist. The true state is "unconditional `AttributeError` for *any* string-named tool, today".
- That makes the populate-then-adopt ordering *more* obviously right, not less — but it also means the proposal's
  **BREAKING** note ("resolving an unknown tool name changes from returning nothing to failing loudly") is false as
  a description of current behaviour, because no reachable caller resolves names through `get_tool` at all. The
  breaking-change framing should be about the *package symbol identity*, which is real, not about miss semantics.

### F5 — `agent-prompt-assembly`'s ordering requirement is not satisfiable under D-7

The requirement (`spec.md:26-30`) says the fixed order "SHALL be a property of the assembly, **not of the call
site**", and the seam requirement (`:11-13`, `:21-24`) says "No caller assembles prompt sections independently".

D-7 (`design.md:228-229`) chooses: "**do not migrate 26 sites.** Make `render_prompt_sections` the section-assembly
primitive it already is, make `SystemPromptParts` consume it, and implement the Up#6 ordering rule **once**, inside
`build()`."

Verified against `src/app/shared/langchain_layer/prompts.py:145-156`:

```python
def render_prompt_sections(*sections: tuple[str, str | None]) -> str:
    """Render plain labeled prompt sections, skipping empty bodies."""
    ...
        rendered.append(f"{label}\n{normalized}")
    return "\n\n".join(rendered)
```

It is a **positional, label-agnostic, order-preserving** concatenator. It has no notion of section *kind*, so it
cannot reorder. Its 27 measured call sites (see M-5) each therefore continue to own their own order. Putting the
rule "once, inside `build()`" scopes it to `SystemPromptParts`, which has **2–3** construction sites
(`agents/registry.py:103`, `:150`, plus the module-level `AGENT_SYSTEM_PROMPT` in `prompts.py` itself that the
design's "2 real sites" count omits). So the ordering rule would govern 3 of ~30 prompt-construction paths while
the spec asserts it governs all of them.

Second, sharper problem: `SystemPromptParts.build()` (`prompts.py:99-124`) emits a **fixed set of named fields** —
identity, objective, context_policy, execution_policy, constraints, uncertainty_policy, examples. There is **no
evidence field and no task-restatement field**. "Retrieved evidence in the middle with highest-salience at the head
and tail of the evidence block" cannot be implemented in `build()` without adding those fields — which is
unmentioned work, and which puts retrieved evidence inside a *system* prompt, where Lost-in-the-Middle ordering is
least applicable and prompt-cache reuse is worst.

**Required:** either scope the ordering requirement to prompts assembled through the parts seam (and say so), or
schedule the section-kind migration D-7 declines. As written, one of the two documents is wrong.

### F6 — Five-plus scenarios can only be proven by executing a graph, which D-10's own proof discipline forbids, and no alternative is authorised

`design.md:412-413`: "**Proof discipline under D17:** no task's Proof may involve executing the agent graph.
Construction-path tasks prove by `ty`, by import, and by a unit test against the constructor." `:279-282` repeats it
("**never** by running the graph. No task's Proof may read 'the graph produces X'").

But these scenarios are behavioural-at-runtime and have no import- or type-level witness:

| Scenario | File |
|---|---|
| "The graph actually pauses" — "the run SHALL pause awaiting that input" | `agent-runtime-resilience` `:91-94` |
| "A pause raised inside a wrapped invocation is not retried" | `agent-runtime-resilience` `:85-89` |
| "A raising tool does not terminate the run" / "the run SHALL continue or pause" | `agent-runtime-resilience` `:102-106` |
| "A run exceeding its step budget terminates explicitly" | `agent-state-handoff` `:41-44` |
| "The version check runs before any reasoning step" / "WHEN a run resumes from persistence" | `agent-state-handoff` `:74-77` |
| "Older recognised version is upgraded … before any other step reads it" | `agent-state-handoff` `:63-66` |

Worse, `interrupt` cannot pause without a checkpointer, and the design itself records
(`design.md:274-275`, citing `findings-database.md` §5) that the checkpointer "short-circuits to `None` because
`psycopg`'s driver cannot load, and nothing assigns the graph at all". So the pause-propagation test — described in
both `proposal.md:147-148` and `design.md:357-358` as "the one test that cannot be dropped for time" — has **no
stated way to run**.

The fix is available and cheap (a throwaway two-node `StateGraph` compiled with an in-memory saver, which is not
"the agent graph" and does not touch the commented lifespan wiring), but the design must **say** it, because as
written the discipline forbids the one test it calls non-negotiable. An implementer facing that will mark the
scenario satisfied by inspection.

### F7 — The `ruff` gate baseline is wrong in the loose direction

`design.md:407` states: baseline **125**, "(→123 after change 0 deletes `todo_temp.py`)", gate **≤123, flat
acceptable**.

Measured now, with `todo_temp.py` still present:

```
uv run ruff check src/   →  Found 123 errors.
uv run ruff check src/ --statistics  →  2  invalid-syntax   (both are todo_temp.py, per D11)
```

So the current baseline is **123, not 125**, and the two `invalid-syntax` errors **are** inside the 123. After change
0 deletes `todo_temp.py` the value drops to **≤121**, not 123. A gate of "≤123" therefore permits this change to add
up to two new lint errors and still pass. Set the gate from a measurement taken after change 0 lands, or make it
"no increase against the value measured at task start".

(`ty`, `ast-grep`, and both `rg -c "^class Tool*"` gates are correct — see the verification table below. This is the
only numerically wrong gate.)

### F8 — The `tools=[]` gate demands 3→0 but the specs assign tools to only 2 of the 3 agents

`design.md:404` gate: `rg -n "tools=\[\]" .../agent_saul/factory.py` **3 → 0**.

Verified at `src/app/shared/langgraph_layer/agent_saul/factory.py:114,120,126` — three `create_agent` calls, and the
first one's comment is `# TODO: add **delegation** tools when available`. `agent-tool-registry:79-87` names tool sets
for the **compliance** and **risk** roles only. No delegation/handoff tool exists in the repo, and D-5 promises "one
construction helper for the handoff message" without saying it is exposed to the orchestrator as a tool.

Either the orchestrator's tool set is specified (most naturally the D-5 handoff tool, which is also how
`brief:ref:1473-1479`'s `transfer_to_*` convention works), or the gate must be 3→1 with the remaining `tools=[]`
justified. As it stands the gate is unreachable and the spec is silent on a third of its own subject.

---

## Minor findings and inaccuracies

- **M-1 — D-9 misdescribes existing code as work to be done.** `design.md:253-254`: "The type already exists:
  `agent_saul/state.py:103` defines `class Citation(BaseModel, frozen=True)`. **It is extended to carry claim /
  source / bounded confidence**". Verified — all three **already exist**:
  ```python
  class Citation(BaseModel, frozen=True):
      claim: str = Field(description="The specific claim being made")
      source: str = Field(description="Document section, statute, or precedent ID")
      confidence: float = Field(ge=0.0, le=1.0)
  ```
  `RiskFinding.citations: list[Citation]` (`:203+`) and `ComplianceFinding.citations: list[Citation]` (`:219+`) are
  already required fields. The **only** new work for Up#11 is the *non-empty* validator. Consequently
  `agent-structured-output`'s "Confidence is bounded" scenario already passes today, and "A cited finding is
  accepted" already passes. Fix the description, or an implementer redefines `Citation` and forks the type.
- **M-2 — "zero `Command(goto=...)` anywhere in the repo" is false.** `design.md:189-190` (D-5's third reason).
  Measured: **10 sites** in `src/app/shared/langgraph_layer/open_deep_search/graph.py:76,92,262,342,369,370,381,385,389,396`.
  D-5's conclusion still stands for `agent_saul`, but the claim as written is wrong, and it matters — `open_deep_search`
  is the repo's existing prior art for exactly the routing convention D-5 says must be invented. D7 puts it out of
  scope; that is a reason not to *edit* it, not a reason to claim it does not exist.
- **M-3 — Capability count.** `proposal.md:95` says "Checked `openspec/specs/` first (**21** capabilities)".
  There are **20** (`ls openspec/specs/`). The reuse analysis itself is sound (see checklist item 1).
- **M-4 — Stale `validate --all` baseline in the gate table.** `design.md:409` records "16 passed / 6 failed of 22",
  faithfully quoting D12 — but that was measured before the five refactor changes existed. Measured now:
  **21 passed / 6 failed of 27**. The acceptance criterion ("no new failures") is unaffected; the number will read
  as a regression to whoever runs it.
- **M-5 — Off-by-one and off-by-N citations** (all verified against source; none change a conclusion):
  | Claim | Actual |
  |---|---|
  | `idempotency.py:31` = `_POSTGRES_TTL_DAYS = 30` (D-3) | `:30`. `:31` is `_REDIS_KEY_PREFIX` — the line D-3 *also* wants bumped, so the two get conflated |
  | reads back with `model_validate_json` (`:77`) | `:83` |
  | `chunks_bm25_idx` at `a71f0d7d9c12:102` | `:97` |
  | `graphiti/registry.py:41-46` eager imports | `:40-45` (inherited verbatim from D6.1, not the author's error) |
  | `precedent_tools.py:117` `scope=scope` into `expand_from_seeds` | `:115` (call opens `:113`) |
  | `write_clause_episodes.py:35` "holds the guard" | `:35` is the *import* of `IdempotencyGuard`; the `make_key` call is elsewhere in the file |
  | `render_prompt_sections` "**26** call sites" | **27** — and the design's own itemisation (11+8+4+3+1) sums to 27 |
  | `SystemPromptParts` "2 real sites" | 2 constructions in `agents/registry.py`, plus `AGENT_SYSTEM_PROMPT` in `prompts.py` itself |
  | `ToolNode` "**zero** occurrences in `src/`" (Q2) | one, in a docstring at `open_deep_search/tools.py:50`. No construction — substantively right, literally wrong |
  | `serialize_to_toon` "16 call sites" | ~11 calls measured (18 name occurrences − 1 def − imports/`__all__`). Inherited from `decisions.md:100` |
- **M-6 — `proposal.md:76` puts a nonexistent directory in scope.** It lists `shared/langchain_layer/middlewares/`.
  The real path is **`shared/langchain_layer/agents/middlewares/`** (`guardrails.py:49,159` both verified as
  `@wrap_model_call  # type: ignore`). `design.md` uses the bare filename and is not wrong; the proposal's scope
  list would have an implementer create a second middleware package.
- **M-7 — `to_tsvector` is in a docstring, not the query.** `design.md:303-304` says the old SQL "is
  `to_tsvector('english', body)` + `ts_rank`". The executed SQL (`search_legal_precedents.py:182-200`) uses a
  pre-existing `fts_vector` **column** with `ts_rank(fts_vector, plainto_tsquery(...))`; `to_tsvector('english', body)`
  appears only in the docstring's suggested DDL. Conclusion (tsvector engine → BM25) unaffected.
- **M-8 — The BM25 "harvest" is not a copy, and the difference is unstated.** The working reference
  (`features/search/repository.py:415-419`, verified) is
  `c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')` against `search_chunks`. The target index is
  `CREATE INDEX chunks_bm25_idx ON chunks USING bm25(**search_text**) …` (`a71f0d7d9c12:97`) — a **different column
  name** on a different table. `legal-corpus-retrieval:50-53` forbids "a second ranking implementation", which is
  right, but the harvest requires a column-name/param change the design presents as a lift-and-shift.
- **M-9 — Trap2 is reinterpreted, not implemented.** `dispositions.md` Trap2 says "hash structural IDs (`clause_id`,
  `doc_id`), **never content**" and calls it "a one-line rule inside the surviving `IdempotencyGuard`". D-4 splits it
  by tool kind and makes `make_key` keyword-only with `structural`/`content` parameters. The argument (search-cache
  collisions across distinct legal questions) is **correct and better than the literal rule** — but it is a
  reinterpretation of a dispositioned item, and dispositions are the orchestrator's, so it should be ratified
  explicitly rather than absorbed. `make_key`'s current signature (`idempotency.py:65-76`, positional
  `step_id, input_data, user_id`) and `precedent_tools.py:82`'s `{"query","user_id","num_results"}` are both exactly
  as described.

---

## The ten commissioned checks — verdict on each

### 1. All 7 capabilities NEW — justified, or a failure to reuse? — **MOSTLY HOLDS, one omission**

`llm-injection` rejection: **correct, verified by reading `openspec/specs/llm-injection/spec.md` directly.** Its four
requirements are `SearchService` constructor injection of `llm: BaseChatModel`, document-function `llm` parameters,
`_build_chat_model()` called once in the dependency layer, and API back-compat. It is chat-model **dependency**
injection end to end; there is not one word about prompt assembly, sections, or rendering. Independent corroboration
that this reading is the project's own: **change 2 reuses `llm-injection` with a `MODIFIED`/`REMOVED` delta** about
exactly that subject (`openspec/changes/documents-unified-schema/specs/llm-injection/spec.md`). Rejection stands.

`session-required`: correctly (if silently) not reused — every requirement is about `OutboxRelay._publish` /
`_mark_published` / `_mark_failed`, `_MAX_RETRIES`, and `shutdown()`. Zero overlap.

Swept the other 18. `datetime-utc-cleanup`, `settings-validation`, `test-mock-isolation`, `noqa-documentation`, the
eight `mcp-*`, `cognee-v1-api`, `outbox-helper-extraction`, `transactional-outbox`, `mcp-context-di` — none touch a
tool registry, a result envelope, idempotency identity, prompt sections, handoff, or corpus retrieval. Correct to
leave alone.

**`typed-exception-handling` is the failure.** `openspec/specs/typed-exception-handling/spec.md:207-239` carries
`### Requirement: Agent tools SHALL catch OS-level and library-specific exceptions`, and **five of its six scenarios
prescribe the return value**, not just the caught type:

> "**THEN** the code catches `OSError`, adds a note …, and **returns a `ToolOutput.fail()` result**" (`:219`)
> — likewise `:223`, `:227`, `:235`, `:239`.

That falsifies D-12's stated distinction (`design.md:325-327`: "`typed-exception-handling` governs *which* exception
type is caught and annotated, whereas the requirements here govern *what the tool reports to the model*"). The
existing spec governs both, and it names **`ToolOutput`** — the class F1 shows this change leaves un-unified. So:

- If `ToolOutput` is collapsed into the single envelope (which `agent-tool-contract` requires), five scenarios in
  `typed-exception-handling` become false and a `MODIFIED` delta **is** warranted.
- If it is not collapsed, `agent-tool-contract`'s first requirement is false.

D-12's second rationale is also undercut by the sibling change: **change 0 already edits
`typed-exception-handling`** (`openspec/changes/cleanup-foundation/specs/typed-exception-handling/spec.md`, a
`MODIFIED` delta on the `asyncpg`/`PostgresError` requirement). So "touching them would confound the acceptance
signal … it would move the baseline other authors are measuring against" is a rule the refactor does not itself
follow. Empirically it does not confound anything — `validate --all` still reports the same 6 failures with change 0
present. The remaining half of D-12 (`pattern-matching-standard` cited not edited) is fine.

### 2. D6.1 boot-ordering constraint — **HOLDS, stated as hard, not as a note**

Stated four times and load-bearing each time: `proposal.md:132-134` (Impact, "**blocking predecessor** of change 0's
deletion"), `proposal.md:138-140` (Risk #1, with a paired-restore plan), `design.md:21-27` ("**This change's first
task is therefore a blocking predecessor of a task in change 0.**"), `design.md:340-344` (Risk, with the revert
sequence and the note that "**boot risk ends at that deletion**"), and `design.md:377-382` (Migration Plan phase 1,
first item).

All the underlying facts verified: `precedent_tools.py:21` → `app.shared.agents.memory.memory_scope` (**30 bytes**);
`precedent_tools.py:22` and `get_obligation_chain.py:29` → `app.shared.agents.tools.idempotency`;
`shared/rag/graphiti/registry.py:40-45` eagerly imports the four `make_*_tool` factories **from the package
`__init__`**, at module scope, outside `TYPE_CHECKING`; `subgraph.py:30` imports the real **7189-byte**
`langchain_layer` path. The `ImportError`-before-FastAPI-construction conclusion is right. `design.md:377-382` also
adds a genuinely useful observation the decisions did not have: **zero call-site edits are needed**, because both
files already call the survivor's API and `precedent_tools.py` already treats its scope as the real `MemoryScope`
(`scope.top_k` at `:104` — verified) against a one-line `str` stub. Good work.

Only gap: `design.md:402`'s gate is `rg -n "app\.shared\.agents\." src/` → **0** after phase 1. That is the right
gate, but it is listed under "Measurable gates" rather than being tied to the cross-change handshake, and change 0's
side of the handshake ("change 0's deletion task cites it **by number**") cannot be verified — this change has no
`tasks.md` yet, so there is no number to cite. Make that reciprocal citation a condition of writing `tasks.md`.

### 3. D17 / "step 18" — **HOLDS on all three tests**

D-10 (`design.md:263-292`) is the subject. Verified against D17 (`decisions.md:274-294`):

- **Not a restoration:** "**No restoration, and no flag that defaults on.** The proposal's task set does not enable
  the graph." Corroborated by `agent-runtime-resilience:37-47` — "The agent graph SHALL NOT be constructed during
  application startup as part of this change, and **no configuration default SHALL cause it to be constructed**."
- **No flag defaulting on:** explicitly withdrawn — "This change also introduces no `SAUL_GRAPH_ENABLED`-style
  toggle defaulting to `True`; an earlier draft of the plan proposed exactly that and it is **withdrawn**."
  Confirmed against the plan: `docs/relay/plan-change3.md:707` shows step 18 originally "Ships behind [a flag]".
  Alternative (a) is recorded and rejected for the right reason.
- **Proofs import-/type-level only:** stated twice (`:279-282`, `:412-413`), including "No task's Proof may read 'the
  graph produces X'", and expressed as a spec scenario (`agent-runtime-resilience:54-57`, "the import SHALL succeed
  and the construction entry point SHALL accept the arguments its callers would pass").
- D17's *"at that time"* clause is honoured explicitly (`:283-286`) with the re-enabling property asserted by the
  import/type proof.

D-10 is also **sharper than D17** on the fail-closed point, correctly: it observes that `dependencies.py:46`'s
`if checkpointer is None` check does not protect against the attribute never being **assigned**. Verified —
`get_saul_graph` (`:40-41`) returns `request.app.state.saul_graph` with no guard at all, `get_saul_checkpointer`
reads `:45` then checks `:46`, `get_redis` reads `:53` unguarded, and `api/v1.py:17` mounts the router. The 500 is
live. This is the strongest single section of the design.

Caveat: see **F6** — the proof discipline is stated but collides with six runtime scenarios.

### 4. `MessagesState` rejection (D-5) — **HOLDS, and is said out loud**

`design.md:176-199` is titled "`MessagesState` is rejected as a vehicle; **sub-todo (i)'s intent is honoured**" and
closes with "Sub-todo (j)'s… " — for (i) it states the split explicitly at `:212`-equivalent phrasing and at
`:176`. Nothing is silently dropped. The reasoning matches `dispositions.md`'s correction note verbatim, including
the `brief:ref:1341-1345` quote about `TypedDict` and Pydantic/dataclass state no longer being supported.

Verified independently: `LegalAgentState` (`state.py:317`) **is** a `TypedDict`; `messages` (`:329`) **is**
`Annotated[list[BaseMessage], add_messages]`; `:343-345` carry `Annotated[..., operator.add]` sibling channels. So
D-5's claim that adopting `MessagesState` would be "a lateral rename that **loses** the sibling channels" is
factually right, and the conclusion — build a handoff-message helper plus a router rule, set `recursion_limit`,
**convert no state class** — is the correct reading of (i)'s intent. `agent-state-handoff:9-34` encodes exactly that
(one construction path, routing reads the message, unrecognised recipient refused, history accumulates).

One factual error inside the argument: see **M-2** (`Command(goto=…)` does exist, 10 times, in `open_deep_search`).

### 5. tenacity / middleware split (D-6) — **HOLDS**

`design.md:201-217` states the boundary as `dispositions.md` item 172 requires: tenacity "**is** installed (9.1.4)
and already used correctly at I/O-client boundaries — `kb_retry.py`, `connections/redis.py`, `razorpay_client.py`.
It is **not** extended into graph nodes." Version verified: **tenacity 9.1.4**. Middleware ownership is scoped to
`@wrap_model_call` (verified live at `agents/middlewares/guardrails.py:49,159`) plus
`ToolNode(handle_tool_errors=…)`, exactly item 172's scope.

The `interrupt` hazard is encoded correctly and in the right place. The options table's Cons cell reproduces the
mechanism: "`ref:1633` forbids a bare `try/except` around `interrupt`, which pauses **by raising** — `tenacity`'s
default `retry_if_exception_type(Exception)` is exactly that catch-all, so HITL silently stops pausing", plus the
second-order point that "a node-local attempt counter is not a checkpointed channel and the retry budget silently
multiplies on every resume" (`ref:1628`). The "Both" option is rejected for multiplicative budgets. `design.md:216-217`
even pre-prices a user override ("the override needs its own task proving `interrupt` still propagates").

The spec side keeps a decorator away from `interrupt` behaviourally rather than by naming a library — right call for
a spec: `agent-runtime-resilience:59-62` ("a single designated seam"; "SHALL NOT be re-implemented inside individual
graph nodes or tool bodies") and `:80-83` ("SHALL propagate unchanged. It SHALL NOT be retried, suppressed, converted
into an error result, or counted as a failed attempt"). Also correctly notes it does not contradict change 1's
tenacity work — it *confirms* the boundary.

Caveat: the pause-propagation test has no runnable path as written — **F6**.

### 6. Coverage of the assigned backlog rows — **HOLDS for 7 of 7; both non-goals are recorded**

| Row | Represented by | Verdict |
|---|---|---|
| **Up#10** one `ToolResult`, survivor `idempotency.py:34` | `agent-tool-contract:9-13`; D-3; gate `rg -c "^class ToolResult"` 3→1 | ✅ survivor named correctly; **but see F1** — a 4th class (`ToolOutput`) is outside the collapse |
| **Up#11** citation: claim / source / confidence | `agent-structured-output:34-63`; D-9 | ✅ all three named; **M-1** — they already exist, only the non-empty validator is new |
| **Up#6** Lost-in-the-Middle ordering | `agent-prompt-assembly:26-50` (head/tail salience, order is a property of the assembly) | ✅ present; **F5** — not satisfiable under D-7 as scoped |
| **153 + Up#7** hydration after checkpointer + `schema_version` | `agent-state-handoff:51-92` (version-checked before any reasoning step; one value governs write and read) | ✅ — and verified: `state.py:322` field exists, `:9` docstring documents the intent, **no node exists**, and there are exactly **two** independent literals (`features/agent_saul/service.py:401` `"schema_version": 1`; `state.py:384` `schema_version: int = 1`) |
| **Trap2** structural IDs, never content | `agent-tool-contract:88-116`; D-4 | ✅ — reinterpreted as a split-by-tool-kind contract; **M-9**, needs ratification |
| **172** middleware | `agent-runtime-resilience:59-112` | ✅ see check 5 |
| **Up#9 cheap half** `response_format` / `include_raw` | `agent-structured-output:9-32, 65-79` | ✅ and reachable: `response_format` prior art verified at `agents/registry.py:130,177` and `agents/factory.py:109,189`; langchain **1.2.12** installed, so `ProviderStrategy.strict`'s `>=1.2` floor is met (Q3 correctly keeps the Gemini-profile question open) |

**Both recorded Non-Goals present, with reasons, not silently omitted:**

- **Item 67 (DROP)** — `design.md:61-64`: "**Item 67 — structured message bus / agent communication protocol /
  persistent shared state: DROPPED.**", with `brief:ref:1473-1479` as the reason and, correctly, **the gap named**
  ("cross-agent messaging has no transport of its own and inherits whatever the message channel provides"). Also in
  `proposal.md:86`.
- **Up#9 escalation machine (DEFER)** — `design.md:65-67`: "**Up#9's Accept / Retry / Escalate state machine:
  DEFERRED**", with the gap named ("a failed validation is refused, but nothing decides *whether to retry, repair,
  or escalate to a human*"). Also `proposal.md:87`.

Beyond the mandate, the Non-Goals also carry items 151 and 194 from `dispositions.md`'s unclassifiable table, D7,
memory construction (with the `MemoryManager` unmasking hazard spelled out — verified: `factory.py:69-74` stub,
`:113` `enable_long_term_memory: bool = True`, `:246` `inject_long_term_context`, `:256` `save_session`, neither
method defined on the stub), and schema DDL. **No unrecorded omission found in the assigned rows** — the one
unrecorded omission is F1, which sits outside them.

### 7. `legal-corpus-retrieval` vs the stubbed clause table and change 2 — **PARTLY HOLDS**

Does **not** assume a live clause table: ✅. `design.md:294-299` states `statutes`/`clauses`/`parent_documents`/
`entities`/`relationships`/`events`/`memory_versions` "**does not exist and never has**" per `findings-database.md`
§4, D15 makes the target `chunks`, "zero rows exist anywhere in this subject area, so this is a retarget with
**nothing to migrate**". `precedent_tools.py:237`'s stub is named in `design.md:34` and in the ADR (`:23`), and its
unimplemented leg gets its own scenario (`agent-tool-contract:82-86`, "SHALL record a warning naming it as not
implemented **and** SHALL NOT count that source toward the total"). Verified: `_vector_search_clauses`
(`precedent_tools.py:221-240`) is a stub whose docstring says "pgvector cosine similarity search on **clauses**
table" and whose body is `return []`. No migration in `src/alembic/versions/` creates `statutes` or `clauses` as a
target — correct.

Does **not** invent a third retrieval path: ✅. D-11 is explicit — "**A third BM25 implementation in this repo would
be a planning failure**" — and the spec forbids it normatively (`:50-53`). Verified that the harvest source is real:
`features/search/repository.py:415-419` and `features/search/fusion.py:28`. D5.1 is honoured.

Alignment with change 2 is where it breaks: change 2's capability is `document-retrieval-schema`, and change 3
**never names it** — it says "change 2's consolidated target" in prose. Combined with **F2** (the A1/A2 asks exist
nowhere change 2 can see) the coupling is by narrative only. Also **M-8**: the target index is on `chunks.search_text`,
not `chunks.content`.

### 8. The `ty` gate numbers — **HOLD. Not invented.**

```
uv run ty check src/ 2>&1 | tail -3   →  Found 46 diagnostics
```

Baseline **46**: confirmed exactly. The derivation is also confirmed, per-file, against the live checker:

| File | `baseline-tests.md:167-170` | Measured now |
|---|---|---|
| `agents/tools/precedent_tools.py` | 11 | **11** |
| `agents/middlewares/guardrails.py` | 5 | **5** |
| `agents/tools/get_obligation_chain.py` | 4 | **4** |
| `agents/factory.py` | 3 | **3** |

So 46 − (11+4) = **31** and 31 − 3 = **28** — both gates are arithmetically sound and grounded in a real measurement,
not a guess. One caveat worth writing into the task: the ≤31 gate assumes **all 15** diagnostics in those two files
are caused by the stub imports. The 15 sites are `get_obligation_chain.py:67,77,94,104` and
`precedent_tools.py:80,85,103,109,115,135,188,193,197,199,207` — all inside the tool bodies that touch
`IdempotencyGuard` / `ToolResult` / `MemoryScope`, so the causal claim is plausible but is **not proven**. If two
survive, a gate fails and blocks a task on a technicality. State it as "≤31, and if not, enumerate the residue".

(`ast-grep scan src/` → **4 errors, exit 0**: confirmed exactly, including the design's warning to compare the
printed count rather than the exit code. `rg -c "^class ToolResult" src/` → **3** and `rg -c "^class ToolRegistry"` →
**3**: both confirmed. `pyproject.toml:759` `--cov-fail-under=80`: confirmed, and the design is right that the suite
exits 1 regardless. The **ruff** number is the one wrong gate — **F7**.)

### 9. `openspec validate` — **HOLDS. 6 failures, not 7.**

```
$ openspec validate agent-tools-unification --type change --strict
Change 'agent-tools-unification' is valid
EXIT=0

$ openspec validate --all
- Validating...
✓ change/agent-tools-unification
✓ change/cleanup-foundation
✓ change/cognee-agent-memory
✓ change/cognee-saul-memory-migration
✗ spec/cognee-v1-api
✓ spec/datetime-utc-cleanup
✓ change/documents-unified-schema
✓ change/ingestion-pipeline-unification
✓ spec/llm-injection
✓ spec/mcp-context-di
✓ spec/mcp-directory-restructure
✓ spec/mcp-server-codemode
✓ spec/mcp-server-composition
✓ spec/mcp-server-pagination
✓ spec/mcp-server-prompts
✓ spec/mcp-server-resources
✓ spec/mcp-telemetry
✓ spec/mcp-testing
✗ change/mintlify-documentation
✗ spec/noqa-documentation
✓ spec/outbox-helper-extraction
✗ spec/pattern-matching-standard
✓ spec/session-required
✓ spec/settings-validation
✓ spec/test-mock-isolation
✗ spec/transactional-outbox
✗ spec/typed-exception-handling
Totals: 21 passed, 6 failed (27 items)
EXIT=0
```

**21 passed / 6 failed / 27 items.** The failing set is byte-identical to D12's six. `agent-tools-unification` passes,
strict included — **no 7th failure added.** Scenario headers use four hashtags throughout (D12's silent-failure trap),
and all seven files use `## ADDED Requirements`, correct for new capabilities.

### 10. Code-claim audit — **62 claims checked, 44 exact, 18 inaccurate (4 materially)**

Audited **62** distinct `file:line` assertions from `proposal.md`, `design.md`, and `adrs.md` against the real files,
plus 7 tool-measured baselines.

**44 verified exactly**, including every one that carries a decision: the three `tools=[]` sites; `base.py:58/73/82/99`;
`registry.py:9`'s missing `get`; `shell.py`'s five `@register_tool` sites **and its zero importers**;
`tools/__init__.py:7-12`'s export inversion (the D6.1 survivor is indeed not re-exported by its own package);
`graphiti/registry.py:56,98`; the three shadow-import lines and the 30-vs-7189-byte `memory_scope` pair;
`idempotency.py:34,65-76`; both duplicate `ToolResult` definitions and `todo_temp.py:8` as the third's sole importer;
`retrieve_statute_section.py:87-92,128-146,159,170-172`; `search_legal_precedents.py:109,110,227-229` and the
`:179-180` docstring **verbatim**; `state.py:103,203,219,239,317,322,329,343-345,384`; the two `schema_version`
literals and the absent hydration node; `graph.py:16,50,55,66,74,91`; `guardrails.py:49,159`;
`prompts.py:19,99,122,126,145`; `agents/factory.py:69-74,113,171,246,256`; `agents/registry.py:103,150`;
`dependencies.py:40-41,45,46,53`; `lifespan.py:234-247`; `api/v1.py:4,17`; `search/repository.py:415-419`;
`fusion.py:28`; `open_deep_search/utils.py:260`; `pyproject.toml:759`; and the three pinned versions
(langchain 1.2.12, tenacity 9.1.4, langchain-google-genai 4.2.1).

**18 inaccurate.** Four matter: **F4** (`factory.py:146` reversed), **F7** (ruff 125), **M-1** (`Citation` described
as needing fields it already has), **M-2** (`Command(goto=…)` "zero in the repo" — 10 sites). The other 14 are
off-by-N line refs, count drift, one nonexistent scope path, and one docstring-vs-query mix-up (M-3 → M-8), none of
which changes a conclusion. That is a **~71% exact rate on cited lines with a 94% correct-conclusion rate** — high
for a change of this size, and notably higher than the scout reports it corrects.

---

## Are the ADRs sufficient for an implementer not to get it wrong?

**Almost, and specifically not.**

`adrs.md` records **one** ADR — absent vs unavailable as a typed field — and it is genuinely good: the Context is
evidence-first (`retrieve_statute_section.py:159` and `:170-172` both returning `None`, `:87-92` turning that into
*"Section {x} of {y} not found"*, all verified), the Decision names three mutually distinguishable outcomes plus an
explicit *completeness-unknown* signal for aggregate verdicts, four alternatives are rejected with reasons that
survive scrutiny, and the Consequences carry the forward-safe/backward-unsafe `extra="forbid"` trap with its
mitigation (prefix bump, one cold cache, **no dual read**). An implementer cannot get *this* decision wrong. The
header's justification for having exactly one ADR is also defensible — the other decisions are change-scoped.

Three reservations:

1. **Status is `Proposed`, not `Accepted`.** D15 sets the precedent that a schema ADR is *accepted* before dependent
   work starts, and this ADR is the load-bearing contract for four of the seven capabilities. It should be accepted
   at review close.
2. **It says "There is exactly one envelope definition in the codebase" — which F1 shows the change does not
   deliver.** The ADR's central invariant is contradicted by its own scope. This is the single most important thing
   to fix, because the ADR outlives the change and will be read as settled.
3. **A second ADR is arguably owed.** `design.md:362-363` records the risk "[Renaming inside `shared/rag/graphiti/`
   collides with change 4, which also works there] → Land the rename before change 4 starts, **or agree the name in
   `adrs.md` first**." The `ToolRegistry` → `AgentToolBundle` rename (D-2) is a cross-change naming commitment with
   four importers (`agent_saul/factory.py:10,182`, `agent_saul/graph.py:16,91` — verified) and the design itself
   points at `adrs.md` as the venue. Right now that pointer dangles.

Where an implementer **will** get it wrong without repair: the registry population mechanism (**F3** — the spec and
the design instruct opposite things), the prompt-ordering scope (**F5**), how to prove the pause test at all
(**F6**), and whether `ToolOutput` is in or out (**F1**).

---

## What is right, on the record

Worth saying plainly, because the findings above are the exceptions rather than the tenor: the three "no scout
reported this" theses in `proposal.md:19-31` are all **independently verified true**, and each materially changes the
plan — tools never bound, the survivor registry empty at runtime, and the package exporting the loser. The export
inversion in particular corrects a prior scout report in the direction that makes the work *harder*, which is the
right direction for a reviewer to find an author moving. The four-phase ordering (fail-closed + cycle predecessor →
shape → honesty → floating retarget) is derived from real constraints rather than convenience, the floating-retarget
argument ("an honest failure is a shippable state, a fabricated legal conclusion is not") is the correct way to
decouple from change 2's slip risk, and the five Open Questions are all genuinely deferrable and each names how to
close it. Q2's instruction — "Close by reading the installed `langchain/agents/` and `langgraph/prebuilt/tool_node.py`
at langchain 1.2.12 **before** writing that task. Do not write it against the 0.2-era signature." — is exactly the
right discipline.

---

## Gate

`tasks.md` should not be written until **F1**, **F2**, and **F3** are resolved, since each changes what the tasks
are. F4–F8 are corrections to text and gate numbers and can be folded into the same pass. M-1 through M-9 should be
fixed for accuracy but block nothing.

**VERDICT: CHANGES REQUESTED**

---

## Author response

Written 2026-08-18, after the review. Nothing above this line has been altered. Every "fixed" below is a change on
disk; every number quoted was re-measured rather than copied from the review.

**Blocking**

- **F1: fixed.** `ToolOutput` is in scope as the **fourth** envelope. `design.md` gains a "The fourth envelope"
  paragraph (13 sites, `shell.py:4,18,68,71,106,108,111,126,129,145,155,158,216`, 28 lines from the survivor); D-3 is
  retitled "four → one" with a 4-row disposition table; ADR 1's Context and Decision are rewritten to four. The gate
  became `rg -c "class Tool(Result|Output)\b" src/` **4 → 1** — the old `^class ToolResult` form was satisfiable while
  the defect survived, and that is now stated as the reason. `typed-exception-handling` **is** reused: a `MODIFIED`
  delta lives at `specs/typed-exception-handling/spec.md`, reproducing the requirement and **all six scenario titles
  verbatim**, replacing the five `ToolOutput.fail()` clauses with envelope-neutral wording and adding a 7th scenario.
  D-12 is rewritten as an explicit **reversal** with a three-change ownership table (change 0 MODIFIES asyncpg,
  change 1 ADDs four, this change MODIFIES only the agent-tools requirement — lanes disjoint).
- **F2: fixed, as a consumer.** The requirement stays, at attribute level, and names no column. A **Coordination
  point** names change 2's Accepted ADR and its `document-retrieval-schema` capability as provider. Both the spec and
  D-11 now state the dependency **directionally**: unsatisfiable until change 2's migration lands, so every proof runs
  after it or is import/type-level, and a proof asserting an index-served lookup *today* is forbidden as unexecutable.
  Column names are referenced, never restated — `grep` for them across this change returns **zero** hits, so a later
  rename cannot give one contract two owners. *One narrowing:* the zero-hits-for-`act_name`/`section_ref` evidence is
  weaker than the review states, since attribute-level language was deliberate. The gap was real; that string was not
  what proved it.
- **F3: fixed.** The contradiction was mine. `agent-tool-registry` no longer mandates the mechanism D-1(c) rejects:
  the requirement is now *"populated by explicit registration before any consumer resolves a tool"*, with five
  order-independent scenarios including **"Importing the package registers nothing on its own"** and **"Resolving
  before registration fails loudly"**. D-1(c) stands unchanged; the spec moved to it.

**Material**

- **F4: fixed, and the hazard is refuted.** The claim was reversed — `factory.py:146` is
  `resolved_tools.append(get_tool_registry().get(t))`, now quoted verbatim. The "silent `None` → `KeyError`" hazard
  **does not exist**: `registry.py:24`'s `return None` has **zero reachable callers**, because no caller passes tool
  names as strings, so `:146` is unreachable today. D-1 is re-justified on real merits (adoption is what makes `:146`
  reachable; an empty registry converts `AttributeError` into `KeyError` for every name; the breaking surface is symbol
  identity, not miss semantics). `proposal.md`'s BREAKING note was rewritten on the same basis.
- **F5: fixed by changing the mechanism, not the wording.** D-7 was rewritten around two measured impossibility facts:
  `render_prompt_sections` is label-agnostic, and `build()` emits a fixed seven labels with no evidence or task field.
  Ordering now lives in a **new kinded assembly seam** (standing instruction / output contract / retrieved evidence /
  task restatement) that distinguishes sections by **kind, not label text**; `render_prompt_sections` is untouched for
  its **27** callers, and that residual is a recorded gap replacing the false "No caller assembles prompt sections
  independently". Evidence is a ranked sequence in per-turn content, which answers the prompt-cache objection.
- **F6: fixed by authorising a vehicle.** D-10 now defines "the graph", tables the six scenarios, and authorises a
  **throwaway two-node `StateGraph` compiled with `InMemorySaver`, constructed inside the test** — explicitly not the
  application's agent graph — with a "Still forbidden" list. D17 is preserved: no task enables the commented lifespan
  wiring, and no flag defaults on.
- **F7: fixed, further than requested.** Re-measured: `Found 123 errors` **with `todo_temp.py` still present**, and
  both `invalid-syntax` errors are **inside** the 123. So ≤123 would have permitted this change to add two new lint
  errors and pass. The gate is now **≤121 after change 0**, and "no increase against the value at task start" before it.
- **F8: fixed, and the gate is now reachable.** D-5 exposes the handoff helper as `transfer_to_<role>` tools tagged
  `handoff`, so the orchestrator is tool-using and `agent-tool-registry` gains **"The orchestrator role receives its
  delegation tools"**. All three roles are now tool-assigned, making **3 → 0** honest. *Two corrections found while
  verifying:* the `tools=[]` lines are `:116,122,128` (`:114,120,126` are the `create_agent` calls — the review and my
  own Context table shared this error, both now precise), and a **fourth** `tools=[]` exists at
  `agents/registry.py:149`, out of scope; the gate is file-scoped on purpose and must not be widened to `src/`.

**Minor**

- **M-1: fixed.** Confirmed — `Citation` already carries claim, source and bounded confidence. D-9 now quotes the type
  verbatim, states the **only** new work is the non-empty validator, and records that two `agent-structured-output`
  scenarios already pass today.
- **M-2: fixed, with the count corrected upward.** The "zero `Command(goto=…)`" claim was false. Measured **13**, not
  10, in `open_deep_search/graph.py`; `agent_saul` has **0**. D-5's conclusion is unchanged and now says so honestly.
- **M-3: fixed.** **20** capabilities, and the same paragraph now records that `typed-exception-handling` **is** reused,
  citing `:219,223,227,235,239` as the evidence that falsified the original claim.
- **M-4: fixed.** **21 passed / 6 failed of 27**, re-measured, with the failing set enumerated and the note that
  `spec/typed-exception-handling` is a **pre-existing** failure of the deployed spec, not caused by my delta.
- **M-5: fixed, all ten.** `idempotency.py:30` (and `:31` distinguished as the key prefix), `:83`, `a71f0d7d9c12:97`,
  `registry.py:40-45` (×2), `precedent_tools.py:115` (call opens `:113`), `write_clause_episodes.py:35` marked as the
  *import* with the implementer told to locate the `make_key` call, **27** call sites, `SystemPromptParts`' third site,
  `ToolNode`'s docstring-only occurrence, and `serialize_to_toon` at **~11** measured.
- **M-6: fixed.** `shared/langchain_layer/agents/middlewares/`.
- **M-7: fixed.** D-11 now says the executed SQL is `ts_rank(fts_vector, plainto_tsquery(...))` and that
  `to_tsvector('english', body)` is docstring-only.
- **M-8: fixed.** D-11 records the column/table difference — `chunks_bm25_idx` on `chunks.search_text` versus the
  working reference `c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')` on `search_chunks` — so the harvest
  is no longer presented as a lift-and-shift.
- **M-9: fixed as flagged, not absorbed.** D-4 now states plainly that it **deviates** from Trap2's literal wording,
  that dispositions are the orchestrator's, and that **if the literal wording was binding this is the decision to
  reverse** — at the cost of only the search-path row. No other decision depends on it.

**Open Questions**

- **Q1 — CLOSED by user decision.** `shared/rag/rag_agent_advanced.py` is **moved to `src/app/examples/`**, not deleted
  and not harvested. Both losses are recorded as **Non-Goals**: the `f"Search error: {e!s}"` anti-pattern survives
  quarantined at `:172,244,293,345,481` — re-measured, the review's numbers were off by three, and the iterative-RAG prior art stays unused, so change 1 designs item 195
  from scratch. **This change writes no harvest task**; Q1 yields one pure-move task.
- **Q2 — CLOSED by reading the installed packages** (langchain 1.2.12 / langgraph 1.1.2). `handle_tool_errors` is
  **unreachable through `create_agent`** — no such parameter and no `tool_node` parameter
  (`langchain/agents/factory.py:673-691`) — so the 0.2-era shape is not writable here. The reachable seam is
  `@wrap_tool_call`, and the mechanism is **`ToolRetryMiddleware`** (`tool_retry.py:30`), default
  `on_failure="continue"` (`:134`) → `ToolMessage(status="error")` (`:273-286`). **It is already built and wired** at
  `guardrails.py:345,:369` through `factory.py:152,188`. The real gap is `agent_saul/factory.py`, which has **zero**
  occurrences of `middleware`; the library default re-raises everything except `ToolInvocationError`
  (`tool_node.py:379-387`), so a raising tool aborts the run there today. D-6 is corrected accordingly.
- **Q3 — CLOSED, and the silent fallback is real.** The configured `gemini-3.1-flash` / `gemini-3.1-pro`
  (`settings.py:191-192`) are **absent** from the profile table — all four `gemini-3` keys in it are `-preview`
  variants. A miss returns **`{}`, not `None`**, which passes an `is not None` guard but fails
  `.get("structured_output")` in `_supports_provider_strategy` (`factory.py:509-522`), so `AutoStrategy` degrades to
  **`ToolStrategy`**. Recorded rather than fixed (the fix is a settings change, out of scope); no spec claims
  strictness, and the proof asserts the strategy actually selected.
- **Q4 — DELIBERATELY OPEN.** Unchanged. No task addresses it; the envelope must exist before a trimming policy can be
  designed against real message volume.
- **Q5 — CLOSED by measurement.** Zero code callers outside `src/app`. Registry-adoption priority is **unchanged**, and
  the second `AttributeError` this change unmasks is equally unreachable — a recorded gap, not a live regression.

**Not fixed, by intent:** the two Q1 losses (Non-Goals, per user decision) and Q4 (deferred). Everything else above is
either fixed on disk or refuted with the measurement that refutes it.
