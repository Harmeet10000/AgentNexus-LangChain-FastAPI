# Batched question round — what actually needs the user

Deliberately batched into **one** round rather than asked piecemeal. Every item here has a **stated default**, so
none of them stops work: if a question goes unanswered, the recorded default is what ships, and the loss is written
into the owning change's Non-Goals rather than left silent.

Status: **two of three answered.** **Q-A** (relocate `rag_agent_advanced.py`) and **Q-B** (install `pg_textsearch`,
which closed **F8**) were both put to the user and answered on 2026-08-18. Only **Q-C** (queue topology) is still
open, and change 1's remediator was told to mark the dependency rather than guess, so nothing is blocked on it. The
round is re-put to the user once all five `review.md` files carry an `## Author response`.

---

## Q-A — Disposition of `shared/rag/rag_agent_advanced.py` (~600 lines) · change 3's Q1 · **ANSWERED 2026-08-18**

> **DECISION: move to `src/app/examples/`.** The user was shown harvest-then-delete as the recommendation and chose
> relocation instead, on a preview that named the losses explicitly. So the algorithm is **not** harvested into
> change 1 and the file is **not** deleted. Both accepted losses are recorded as Non-Goals in change 3's `design.md`
> rather than quietly repaired: the `f"Search error: {e!s}"` anti-pattern **survives** (quarantined out of
> production, not fixed), and the iterative-RAG prior art **stays unused**.

The decision was made with those downsides visible, so absorbing either one silently would misrepresent it.
Relayed to change 3's remediator; recorded as adjudication **A16**.

**Line numbers corrected 2026-08-18** — my original citation was off by three at every site, found by change 3's remediator and verified directly. Its state, as established by change 3's author: it returns `f"Search error: {e!s}"` as tool output at
`:172,244,293,481` (plus a differently-worded fifth at `:345`, `Error retrieving document:`) — the exact anti-pattern change 3 removes — but it is **pydantic-ai, not langchain**, has
**zero importers**, its entry point is a CLI (`run_cli()` `:517`), it queries a `match_chunks()` function defined in
no migration and no source file, and it imports `from ingestion.embedder import create_embedder`
(`:119,198,267,373`) — **a package that does not exist in this repo**. So every tool in it `ImportError`s on first
call.

| Option | Consequence |
|---|---|
| **Harvest then delete** (author's recommendation) | `search_with_self_reflection` (`:353`, grading `:420`, refinement `:460`) and `expand_query_variations` (`:52`) are the repo's **only iterative-RAG prior art**, and `dispositions.md` routes agentic query rewriting into change 1's item 195. Copy the algorithm into change 1's design notes first, then delete. |
| **Move to `src/app/examples/`** ← **CHOSEN** | Per `CLAUDE.md`'s rule for examples. Keeps the code, stops it reading as production. The anti-pattern survives but is quarantined. |
| **Leave it** (was the default if unanswered) | Change 3 ships without touching it; the `Search error:` anti-pattern survives in a zero-importer module. |

## Q-B — Authorization to install `pg_textsearch` on the live database · **F8** · **ANSWERED 2026-08-18, F8 CLOSED**

> **DECISION: "Yes — install it now",** scoped by the user's selected preview to *"this one extension. No tables, no
> drops, nothing else."* Executed as exactly one `CREATE EXTENSION IF NOT EXISTS pg_textsearch` followed by
> read-only catalog queries; the password was never printed and the public table count is still 16.

**F8 is closed, and the answer is better than expected** — full write-up in `findings-database.md` §10. Access method
is **`bm25`**; opclasses `text_bm25_ops` / `text_array_bm25_ops`; and the repo's existing BM25 SQL is **already
correct**, using the right two-argument index-scoped overload with the right negation and ordering. No rewrite.

Two residual obligations fall out of it, both now change 0's: **no `bm25` index exists anywhere** in the database,
and because the two-argument `to_bm25query` overload takes the index name as a **literal SQL argument** — pinned at
`search/constants.py:15` — the index name is part of the **query contract**, not a naming convention. An index of the
right shape under a different name silently fails to satisfy the query, so both indexes must be created by exact
name.

**Correction on the record, and it reverses what I told you earlier.** I previously reported that a subagent's
earlier `CREATE EXTENSION` against the live instance had been transaction-wrapped and rolled back, leaving the
database clean. **That was wrong — the statement committed.** OID forensics settles it: `pg_textsearch` holds OID
**46640** while every other extension sits between 13560 and 19498, and 46640 is higher than
`max(oid) FROM pg_class` (46505), making it the newest object in the catalog. Combined with the separately-measured
fact that zero `to_bm25query` rows existed in `pg_proc` at probe time, the extension cannot have been present
before and cannot have been rolled back.

What that did and did not mean: the change was additive and benign, nothing else persisted, the table count never
moved, and it is now authorized anyway. What it cost is the reporting accuracy — I passed on a subagent's claim of a
rollback as though it were an observation. The lesson is recorded: **a rollback claimed by an agent is not evidence
of a rollback.**


## Q-C — Queue topology · change 1 · **pending**

Change 1 flagged queue topology as a question that must be answered before its `tasks.md`. Its remediator has been
instructed to write the dependent tasks and **mark the dependency** rather than guess a topology, so the change is
not blocked on it. Exact wording to be filled in from change 1's remediated Open Questions.

Related and already settled, so not a question: the missing worker/beat runtime is change 1's
(`dispositions.md` 198.4), and change 4's consolidation beat entry is **inert until change 1 lands it**
(adjudication A3).

---

## Explicitly NOT user questions

Recorded so the round stays short and so nothing gets asked that an agent should have answered itself.

- change 3 **Q2** — how tool-error handling is reached (`ToolNode` has zero occurrences in `src/`; agents use
  `create_agent`). **CLOSED 2026-08-18 by change 3's remediator, verified here.** The answer is a live defect on
  210's tools hop, and it is the fifth instance of the "declared missing, already built" pattern:
  `ToolRetryMiddleware` **is** built (`guardrails.py:345,369`) **and wired** (`factory.py:184 create_agent(` …
  `:188 middleware=middleware`) — **and tool errors are still unhandled today**, because `handle_tool_errors` is
  **unreachable through `create_agent`** at langchain 1.2.12 and the library default re-raises everything except
  `ToolInvocationError`. So a raising tool **aborts the run right now**. The seam is `@wrap_tool_call`, not a
  `ToolNode` kwarg. "Already built" and "currently broken" were both true at once.
- change 3 **Q3** — whether the configured Gemini models expose native structured output through their profile on
  `langchain-google-genai` 4.2.1. **CLOSED 2026-08-18, same source.** A profile miss returns **`{}`, not `None`** —
  which is truthy-checked as present in the wrong places, so `AutoStrategy` **silently degrades to `ToolStrategy`**.
  Functional, but not the strictness it appears to be, and the degradation emits nothing.

  The dedicated agent assigned to Q2/Q3 (`aa998a310c99846bf`) never reported and is no longer running; both
  questions were closed independently by change 3 and confirmed against the tree, so nothing is outstanding. Recorded
  because an agent that vanishes without a report is not evidence of an unanswered question — nor of an answered one.
- change 3 **Q4** — message trimming for the handoff envelope. A follow-up, not a blocker; needs the library source
  as authority.
- change 3 **Q5** — whether `get_research_agent` / `get_code_review_agent` are reached from outside `src/app`.
  Closable with one repo-root search. If yes, `factory.py:146`'s `AttributeError` is promoted from rare to
  first-request.
- change 0 — whether the `401` can be asserted by an automated test. **CORRECTED 2026-08-18 — the stated reason was
  false.** I wrote that there is no working test client fixture, "thirteen collection errors involve it". Measured
  just now: `uv run pytest --collect-only -q` → **90 tests collected, zero collection errors.** The thirteen-error
  figure was stale and is withdrawn. Found by change 0's reviewer as its F9, verified directly here. Whether the
  `401` is best asserted by a test or a direct probe is now **genuinely open** and belongs to change 0's remediation,
  because the justification for choosing the probe has evaporated — not because the answer changed, but because the
  reason given for it was never true. (Note the coverage gate does fail — 22.16% against a required 80% — which is a
  different problem and not a collection failure.)
- change 0 — whether the fresh-environment procedure becomes a committed script. Tooling preference, moot if the
  history squash lands.
