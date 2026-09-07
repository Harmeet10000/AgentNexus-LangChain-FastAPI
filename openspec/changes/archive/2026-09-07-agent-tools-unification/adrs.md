> Two durable architectural decisions are recorded here. The first is the tool-result contract; the second fixes a
> cross-change name that `design.md` had promised to settle here and had left dangling. The change's other decisions
> (registry survivor, prompt seam direction, retry ownership, state-schema handling) are change-scoped and live in
> `design.md`; per the schema, missing non-decisions is not a defect.

## ADR: A tool result distinguishes *absent* from *unavailable*, and unavailability is a first-class field

- **Date:** 2026-08-17
- **Change:** `agent-tools-unification`

**Status:** **Accepted** — 2026-08-18, at review close.

Accepted ahead of this change's own code, on the precedent D15 sets for a schema ADR: this is the load-bearing
contract for four of the seven capabilities in the change, and the honesty work in phase 3 cannot be written against a
proposed contract. Change 4 and anywhere else that reads tool results build against it.

## Context

Every agent, node, verdict, and downstream compliance judgement in this product reads tool output. Today the tool layer
has **four** separately-defined result shapes under two different class names, and no way to say *"I could not reach the
corpus"* — so it says the only other thing it can say, which is *"the thing you asked for is not there"*.

The four: `langchain_layer/agents/tools/idempotency.py:34` (`ToolResult`, the survivor),
`shared/agents/tools/idempotency.py:11` (`ToolResult`, dies with change 0),
`shared/rag/document_processing/models.py:318` (`ToolResult`, one importer, deleted), and
`langchain_layer/agents/tools/base.py:30` (**`ToolOutput`**) — which is the worst of them and was missed by every scout
and by this ADR's first draft precisely *because* its class name differs. `ToolOutput` carries a `to_agent_string()`
(`base.py:46`) whose failure branch returns `f"ERROR: {self.error}"`, and all **13** of its use sites
(`tools/shell.py`) call it. Those five tools therefore return an error *sentence* to the model rather than a result —
the exact anti-pattern this ADR exists to eliminate, in the same package as the survivor, 28 lines away.

Concretely: `retrieve_statute_section.py:_fetch_statute_section` returns `None` for **both** "no matching row" (`:159`)
and "`SQLAlchemyError`" (`:170-172`), and `:87-92` converts that `None` into a failure whose message is
*"Section {x} of {y} not found in {z}"*. The backing table does not exist and never has
(`findings-database.md` §4), so **the model is currently told that a section of law does not exist whenever the
database is unreachable.** The same shape appears twice more: `search_legal_precedents.py:227-229` swallows a database
error into `[]`, and because `total_sources` (`:109`) then counts only the surviving evidence leg, the sufficiency
verdict `insufficient_basis` (`:110`) reports **sufficient statutory basis that was never retrieved**; and
`precedent_tools.py:221-237` returns `[]` from an unimplemented vector leg with no log line while its docstring
(`:62`) advertises the capability. A docstring at `search_legal_precedents.py:179-180` explicitly licensed this,
describing the fallback as letting you *"deploy before the statutes table is populated"*.

For a legal product this is not a bug class, it is a fabricated legal conclusion — and the decision must be settled
**before** anything wires this layer up, because wiring it is what makes it fire.

## Decision

A tool result carries **availability as an explicit, typed field**, distinct from success and distinct from the error
description. Three outcomes are representable and mutually distinguishable without parsing prose: **success**,
**absent** (the corpus was reached and holds nothing matching), and **unavailable** (the corpus, index, or graph could
not be reached, or the capability is not implemented). Any aggregate verdict computed over multiple evidence sources
must additionally expose that its completeness is **unknown** whenever any source was unavailable.

There is **exactly one** envelope definition in the codebase after this change — **one, from four** — and the
unavailable state has its own constructor rather than a convention in a metadata dictionary. "Envelope definition" is
determined by shape and role, **not by class name**: the collapse explicitly includes `ToolOutput`, and no definition
is out of scope for being named differently from the survivor. No envelope carries a method that renders itself to a
bare error sentence.

## Rationale / Alternatives

Availability is a *control* signal: it decides whether to retry, to escalate, to degrade, or to tell a lawyer that the
statute book could not be opened. Control signals must be typed, because a caller that has to grep an error string for
`"not found"` to decide whether to trust a compliance verdict is a caller that will get it wrong.

- **Carry it in the existing `metadata` dict** — technically free, since `**meta` already flows there. Rejected: an
  unkeyed convention inside a free-form dict is precisely how the present defect survived review, and no type checker
  can enforce it. The field must be visible in the JSON schema.
- **Raise an exception for unavailability** — rejected: these values are returned to a language model as tool output.
  An exception either aborts the run or gets caught and re-stringified into the model's context, which is the
  string-as-error anti-pattern being eliminated. The **in-scope** live instance is
  `langchain_layer/agents/tools/base.py:46`'s `ToolOutput.to_agent_string()` returning `f"ERROR: {self.error}"`, called
  at all 13 of its sites in `tools/shell.py` — in the very package this change unifies. (An earlier draft of this ADR
  cited only `shared/rag/rag_agent_advanced.py:172,244,293,345,481`, a zero-importer module which the user has decided
  to **relocate to `src/app/examples/`** rather than fix or harvest; its instances of the anti-pattern survive there,
  quarantined and unimported, and are a recorded Non-Goal in `design.md`.)
- **A distinct result type per outcome (a union)** — rejected: results are persisted as JSON with a 30-day TTL and read
  back through a single validator, so a union needs a discriminator field, which is the field this decision adds anyway.
- **Fix the two statute tools only, without changing the shared envelope** — rejected: the same defect appears in four
  places already and would reappear in the next tool written. The contract, not the call site, is the fix.

## Consequences

**Committed to, positively:**

- Every future tool inherits the distinction for free, and "unavailable" becomes an expressible, testable state rather
  than something a caller infers.
- Retry, escalation, and degradation logic can be written against a field instead of against prose, which is what makes
  the middleware retry seam (`design.md` D-6) implementable at all.
- The floating corpus retarget stops being a blocker: un-retargeted tools report unavailability truthfully, so an honest
  failure is a shippable state.

**Costs, accepted on the record:**

- The persisted envelope forbids unknown fields, so adding a defaulted field is forward-safe but **not** backward-safe:
  a new-schema entry read by old code raises. Mitigated by bumping the persistence key prefix in the same commit as the
  idempotency key-shape change and accepting **one cold cache**. Explicitly not mitigated by a dual read.
- Callers that today treat any non-success as absence must be updated; that is a real, if small, blast radius, and it is
  the reason the unavailability register is a single commit rather than four.
- Once the distinction exists, *not* handling `unavailable` becomes a visible omission in every consumer — which is the
  intent, but it does surface work in change 4 and anywhere else that reads these results.

---

## ADR: The Graphiti tool bundle is named `AgentToolBundle`, and `ToolRegistry` names exactly one class repo-wide

- **Date:** 2026-08-18
- **Change:** `agent-tools-unification`

**Status:** **Accepted** — 2026-08-18.

Recorded here rather than in `design.md` because `design.md`'s Risks section promised exactly this
("*Land the rename before change 4 starts, or agree the name in `adrs.md` first*") and, until now, nothing about the
name existed here — the pointer dangled. A cross-change naming commitment with importers in another change's working
directory is not change-scoped, so it belongs in the durable record.

## Context

Three classes in this repo are called `ToolRegistry`, and they are not three copies of one idea:

| Definition | What it actually is |
|---|---|
| `langchain_layer/agents/tools/base.py:58` | the real registry — `register` / `get` / `all` / `by_tags` / `by_names` / `names` / `descriptions` over a name→tool mapping. D6.1's survivor |
| `langchain_layer/agents/tools/registry.py:9` | a second, divergent registry — `get_tools` / `get_tool` / `get_search_tool` / `get_crawl_tool`, **no `get`**. Deleted by D-1 |
| `shared/rag/graphiti/registry.py:56` | **not a registry at all** — an immutable Pydantic bundle of four pre-built Graphiti tools, constructed once by `build_tool_registry` (`:98-122`) and consumed as a value object |

The name collision is why three separate scout reports disagreed about which class survives, and D6.1 forbids deleting
the third file: `:34-122` is live and the bundle is consumed by `agent_saul/graph.py:16,91` and
`agents/factory.py:182,205`. Meanwhile change 4 (memory) also works inside `shared/rag/graphiti/`, so whatever this
change calls that class, change 4 inherits.

## Decision

The Graphiti value object is renamed **`AgentToolBundle`**. `ToolRegistry` thereafter names exactly one class in the
repository — the survivor at `base.py:58` — and `rg -c "^class ToolRegistry" src/` is **1**.

Three properties are committed to:

1. **The name is fixed now, before change 4 starts**, so change 4 writes `AgentToolBundle` from its first line rather
   than renaming across two changes.
2. **The rename ships with a same-module alias** (`ToolRegistry = AgentToolBundle`) for exactly one commit, so a missed
   importer cannot fail boot at import time; the alias is removed in the following commit once the four importers are
   confirmed rewritten.
3. **It is sequenced after D-1's registry adoption**, so no window exists in which one imported name has two meanings.

## Rationale / Alternatives

`AgentToolBundle` says what the object is — a fixed, immutable set of already-constructed tools handed to an agent —
and it cannot be confused with a mutable name→tool lookup. The distinction is load-bearing rather than cosmetic:
per-agent tool assignment living inside a global registry is the design D-2 rejects.

- **Delete the file and fold the bundle into the registry** — rejected: D6.1 says it is not deletable, and a
  bundle-of-four is genuinely not a registry.
- **Rename the survivor instead and leave the bundle as `ToolRegistry`** — rejected: the survivor holds the public
  package symbol and has the larger potential importer set; renaming it moves the churn to the wrong side.
- **Leave the collision and disambiguate by import path** — rejected: that is the status quo, and the status quo cost
  three scouts and one reviewer real time to untangle.
- **`GraphitiToolBundle`** — rejected: the bundle is handed to agents and its content is not intrinsically Graphiti's;
  naming it after its current construction site would mislead the moment a fifth tool comes from elsewhere.
- **`ToolSet`** — rejected as too close to `by_tags()`'s return value, which is also a set of tools but is derived,
  not constructed.

## Consequences

**Committed to, positively:**

- One `ToolRegistry` repo-wide, so "the registry" is unambiguous in every future conversation and every future spec.
- Change 4 gets a stable name to build against with no coordination cost beyond reading this ADR.
- The alias-for-one-commit pattern makes the rename revertible by a one-line re-add rather than by a multi-file revert.

**Costs, accepted on the record:**

- Four importers change in one commit (`agent_saul/factory.py:10,182`, `agent_saul/graph.py:16,91`). If one is missed
  and the alias has already been removed, boot fails at import — which is why the alias exists and why the
  `rg -c "^class ToolRegistry" src/` → 1 gate runs in the same task.
- `shared/rag/graphiti/registry.py`'s module name still says "registry" while its class no longer does. Renaming the
  module as well would collide with change 4's working set for no behavioural gain, so the mismatch is accepted and the
  module docstring (`:9,25`) is corrected in the same commit — it currently points at the deleted-stub import path and
  at an `app.state.saul_graph` assignment that exists nowhere, making it the most misleading comment in the layer.
