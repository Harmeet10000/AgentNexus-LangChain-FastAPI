---
name: orient
description: |
  Orient: find/explain/search codebase or look up external docs. Use when user asks where/find/how/explain/architecture, or needs external research. First tool before grep/rg.
---

# orient

**This file is the only authority on search routing.** CLAUDE.md holds a summary
of the table below; where anything else disagrees, this file wins. There is no
mandatory first tool — route on what you already hold.

## Routing

| What you hold | Call | Cost |
|---|---|---|
| exact symbol / file name | `codegraph_explore "<names>"` (MCP) | ~4–15 KB |
| a concept, no name | `graphify query "<q>" --budget 12000` → feed the names it returns to `codegraph_explore` | ~7 KB + explore |
| two known symbols, want the link | `graphify path "A" "B"` | 88 B |
| one symbol, want its edges | `graphify explain "X"` | 1.2 KB |
| "what breaks if I change X" | `graphify affected "X"` **or** `codegraph impact <symbol>` | small |
| which tests cover a change | `codegraph affected <files>` | small |
| literal string / config / comment | `rg '<text>'` → the hit gives a name → back up this table | 352 B |
| structural shape | `ast-grep -p '<pattern>' -l py` | small |

Do not start a vague question at `codegraph_explore`. Measured: on *"how does
authentication work"* it returned 2 files (`exceptions.py` + `router.py`) and
missed `security.py`, `dependencies.py`, and `service.py` — the JWT, dependency,
and login code. `graphify query` named all three for less than half the tokens.
Names → codegraph. Concepts → graphify first.

Grep is a **discoverer, not an interpreter**: a hit gives you a symbol name, and
that name goes back up the table. Never run `rg` twice for the same task.

**Stop rule:** two discovery calls. Then either answer, or state the narrowed
question. Escalating past that costs more than asking.

## Tool surface — MCP vs CLI

Only **`codegraph_explore`** is exposed as an MCP tool here. Everything else is
the `codegraph` CLI via Bash — these are real and useful, they are just not MCP
tools: `query` (symbol search), `node` (one symbol + caller trail), `callers`,
`callees`, `impact`, `affected`, `files`, `status`, `sync`, `index`.

```bash
codegraph query "<partial name>"     # does this symbol exist?
codegraph node "<symbol>"            # one symbol, source + caller/callee trail
codegraph impact "<symbol>"          # blast radius
codegraph affected src/a.py src/b.py # which tests cover these
codegraph status .                   # index size, node/edge counts, staleness
```

```bash
graphify query "<question>" --budget 12000   # BFS depth 2 — see note below
graphify path "A" "B"                        # shortest path, cheapest call here
graphify explain "X"                         # node + typed edges + rationale
graphify affected "X" --depth N              # reverse traversal
graphify god-nodes --top N                   # architectural hubs
```

`ast-grep` patterns and `firecrawl` syntax live in [`REFERENCE.md`](REFERENCE.md).

## graphify's two limits are different things

`--budget N` (default 2000) caps **display**. Depth is **hardcoded at 2** — there
is no depth flag on `query`. So raising the budget un-truncates what was found;
it does not widen the search. Measured on one question: default budget showed 59
of 77 nodes; `--budget 12000` showed all 77 — *still 77*, because depth 2 bounds
the frontier, not the budget.

**Use `--budget 12000` by default.** At the default 2000 you silently lose ~23%
of the traversal, and the cut is by BFS order, not relevance.

To see past the 2-hop horizon, hop deliberately: take a frontier symbol and
re-seed — `graphify explain "<frontier symbol>"`, `graphify path "A" "B"`, or
`graphify affected "X" --depth 4`. Chaining hops beats one wide query.

## External branch

- **Context7 MCP** — library/framework/API docs: `context7_resolve-library-id`, then `context7_query-docs`.
- **firecrawl CLI** — everything Context7 doesn't cover. Always write output to `.firecrawl/`.

## After editing code — refresh both indexes

Wired in `.claude/settings.json`, both `async` so neither blocks you:

| Hook | Runs | Cost |
|---|---|---|
| `PostToolUse` on `Edit\|Write\|MultiEdit` | `scripts/codegraph-sync-on-edit.sh` | ~0s for docs, ~2.5s for source |
| `Stop` (turn end) | `scripts/refresh-code-graphs.sh` | ~2.5s, plus 25s only if source changed |

The per-edit script gates on extension (`.py .pyi .go .ts .tsx .js .jsx .rs
.java`) — editing markdown or JSON exits immediately. The Stop script always
syncs codegraph, and runs `graphify update .` **only** when a `.py`/`.go` file is
newer than `graphify-out/graph.json`, so doc-only turns don't pay the 25s.

Run them by hand only if you edited outside the tools, or a hook failed:

```bash
codegraph sync .        # 0.7s — incremental; the daemon also watches (~1s lag)
graphify update .       # 25s  — AST re-extract + Leiden clustering, no LLM
```

- `graphify update . --force` after a refactor that **deletes** code — the plain
  update refuses to write a graph with fewer nodes than the last one.
- Never `graphify update . --no-cluster` as a "fast path": it writes raw
  extraction and drops **every** community assignment (verified: 4537 nodes, 0
  with a community). Recover with a full `graphify update .`.
- Verify with the project's own checks, not the graph:
  `uv run ruff check --fix src/` · `uv run ty check src/` · `uv run pytest` ·
  `ast-grep scan src/` (vendored rules in `.ast-grep/rules/`).

A third hook, `PreToolUse` on `Bash`, routes through
`scripts/graphify-search-guard.sh`: it forwards to `graphify hook-guard search`
only when the command actually contains `rg`/`grep`/`find`/`ast-grep`. Matchers
match the tool **name**, not the command string, so an unwrapped `Bash` matcher
annotated `git status` and `uv run pytest` too.

## Cost table — measured 2026-08-16, re-measure when either tool updates

| Call | Output | Returns |
|---|---|---|
| `graphify path A B` | 88 B | 1 typed edge |
| `rg --files src/app/features/auth/` | 352 B | 9 file names |
| `graphify explain X` | 1.2 KB | node + 10 typed edges + rationale |
| `graphify query` (budget 2000) | 6.6 KB | 59 of 77 names, **0 edges** |
| `graphify query` (budget 12000) | ~7 KB | all 77 names, **0 edges** |
| `codegraph_explore "<NL>"` | ~15 KB | verbatim source, 2 files |

Cost is the whole argument for routing. If the numbers rot, the routing is
guesswork — re-measure rather than trusting the table.

`query` returns **names, not relationships**, despite the "knowledge graph"
framing. Edges come only from `explain` and `path`. Treat `query` as a name
resolver: it is a discoverer, same role as grep, and you still owe a second call
to understand anything.

---

## Deep Internals

**1. Community names are hub-derived, never semantic, and they churn on every
update.** Leiden clusters get named after their highest-centrality member, so
`hash_password()` and `.login()` land in a community called
`log_expected_failure` — the decorator wins centrality because many service
methods reference it. `graphify update .` says so out loud: *"renamed 260
community(ies) by their hub."* Worse, the labels are unstable: across one update
`LoginRequest` moved from community `log_expected_failure` to `APIResponse` with
no code change. **Never reference a community by name** in a doc, rule, or query
— it will rot silently. `graphify label` restores LLM-written names (costs API
calls); `--exclude-hubs 99` suppresses utility hubs so domain nodes surface.

**2. graphify's truncation is BFS-ordered, not relevance-ranked.** The dropped
nodes are whatever the traversal reached last, so a 2-hop-away critical symbol
loses to a 1-hop markdown heading — in a live run `7. Backstage Developer Portal`
(a docs heading) survived while `.oauth_callback()` was nearly cut. The banner's
*"the answer may be among the N cut nodes"* is literal, not a hedge. Raising
`--budget` is the only fix that preserves the result set; narrowing the query
changes the **seed set** instead, which silently searches somewhere else.

**3. codegraph's freshness guarantee is split, and only half of it is strong.**
The index is SQLite in WAL mode behind a watcher daemon with ~1s lag.
`codegraph_explore` re-reads files from disk at call time, so the *source text*
is genuinely current — but *which symbols it selects* comes from the possibly
stale index. Immediately after an edit you can get byte-perfect source for the
wrong set of symbols. That is a different failure mode from "stale cache," and
it is invisible: nothing in the output marks the selection as stale. `codegraph
sync .` costs 0.7s and closes the window.
