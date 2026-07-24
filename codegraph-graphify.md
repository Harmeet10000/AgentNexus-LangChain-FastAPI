# CodeGraph / Graphify / ast-grep / grep — Search Strategy

## The Decision Tree (Not a Flat Stack)

The real entry point depends on **what you know**, not a rigid layer order:

```
Know a symbol name?             → CodeGraph (source + relationships)
  └─ Need to rewrite everywhere? → + ast-grep in parallel
  └─ Need architecture context?  → + Graphify in parallel

Vague concept, no symbol name?  → Graphify query (god nodes + fuzzy discovery)
  └─ Found a symbol name?        → CodeGraph for source + blast radius

String, config, generated file? → grep / rg
  └─ Found a structural pattern? → ast-grep for precision
```

---

## Layer 1: CodeGraph

**Precondition:** `.codegraph/` exists.

### What it is

Tree-sitter AST → SQLite (FTS5 + graph traversal). Exposed as an MCP tool. FileWatcher (inotify/FSEvents) keeps the index fresh in <1s.

### Tools available (not just `codegraph_explore`)

| Tool | Use case |
|---|---|
| `codegraph_explore` | Primary — returns verbatim source + call paths + blast radius in one call |
| `codegraph_search` | Quick symbol lookup — returns file + line only, no source. Use as lightweight pre-probe before a full explore |
| `codegraph_impact` | Dedicated blast radius calculation when you don't need full source |
| `codegraph_callers` / `codegraph_callees` | Trace just call relationships |

### What `codegraph_explore` returns

- Verbatim, line-numbered source (Read-equivalent — safe to Edit from)
- Call paths (including dynamic-dispatch hops: callbacks, React re-render, JSX children, Python decorators)
- Blast-radius summary
- Overloaded names: every definition's body in one call
- **Capped at 12 files** by default — narrow your query if results are truncated

### When to use

| Question | Use CodeGraph? |
|---|---|
| "What is this symbol / where is it defined?" | **Yes** — one call |
| "Who calls this function?" | **Yes** — callers returned automatically |
| "What breaks if I change this?" | **Yes** — blast radius included |
| "How does X reach Y?" | **Yes** — call path surfaced automatically |
| "I need to edit this, show me source + impact" | **Yes** |
| Before ANY edit to indexed code | **Yes** |
| "Does this symbol exist? I'm not sure of the name" | **codegraph_search** — lightweight pre-probe |

### When this layer can't answer

- **Vague concept, no symbol name** → skip to Graphify query
- **Config files, string literals, generated code** → skip to grep
- **CodeGraph not indexed** → skip to Graphify if available, else grep

### Cost scale note

Token/cost savings are **scale-dependent**. On a small project (<500 files), adopt CodeGraph for the speed and precision — the cost savings compound into real money only when the codebase (and team) gets large. The universal win is fewer tool calls and faster answers at every size.

### Anti-patterns

- **Don't grep or Read first** — codegraph_explore IS Read-equivalent.
- **Don't reconstruct flows by hand** — name the endpoints; codegraph surfaces the path including dynamic-dispatch hops.
- **Don't ignore the staleness banner** — after editing, Read only the files listed. Everything else is fresh.
- **Don't call codegraph on non-indexed projects** — if `.codegraph/` doesn't exist, stop.
- **Don't use codegraph_explore when codegraph_search suffices** — if you just need to confirm a symbol exists, search is lighter (~0 vs ~56k tokens).

---

## Layer 2: Graphify

**Precondition:** `graphify-out/graph.json` exists.

### What it is

Knowledge graph from tree-sitter AST (code) + LLM semantic extraction (docs, PDFs, images, video). God nodes, Leiden communities, cross-file links, suggested questions.

### Access methods

| Method | When |
|---|---|
| `graphify query "..."` | Ad-hoc questions via CLI |
| `graphify path "A" "B"` | Shortest path between two things |
| `graphify explain "X"` | Explain one concept |
| MCP server (`python -m graphify.serve graph.json`) | Zero-subprocess access from agent — same overhead model as CodeGraph |
| Shared HTTP server | Whole team points at one `--transport http` URL |

### When to use

| Question | Use Graphify? |
|---|---|
| "How do these subsystems relate?" | **Yes** — `graphify query "what connects X to Y?"` |
| "What connects auth to the database?" | **Yes** — `graphify path "AuthService" "DatabasePool"` |
| "Explain how the rate limiter works" | **Yes** — `graphify explain "RateLimiter"` |
| "What are the core modules?" | **Yes** — god nodes |
| "Find the email sending code" (vague, no known symbol) | **Yes** — Graphify handles fuzzy concept matching CodeGraph can't |
| "Show me the architecture overview" | **Yes** — `GRAPH_REPORT.md` for broad context |
| "How does this doc/PDF fit into the codebase?" | **Yes** — non-code asset linking |

### When this layer can't answer

- Need to find all occurrences of a code pattern → ast-grep
- Need string/config search → grep
- Need verbatim source for editing → CodeGraph (or Graphify to discover symbol name, then CodeGraph for source)

### MCP server setup

```bash
python -m graphify.serve graphify-out/graph.json
```

Registers as an MCP server. The agent can query the graph with zero subprocess overhead — same model as CodeGraph. Recommended over CLI when the agent has MCP access.

### Tips

- `graphify query "<question>"` is ~200 tokens vs `GRAPH_REPORT.md` at ~10k+
- `graphify path "A" "B"` catches cross-module connections grep can't find (no shared text)
- `graphify extract . --update` after file changes instead of full re-extract
- `--cluster-only --resolution 1.5` for finer-grained communities
- `--exclude-hubs 99` to see domain-specific gods, not utility modules

---

## Layer 3: ast-grep

**Precondition:** Language supported by tree-sitter grammar (20+ languages).

### What it is

AST-level structural pattern matching — **grep × eslint × codemod**. Patterns use meta-variables (`$VAR`, `$$$REST`). Supports rewrite (`-r`), rule files, and interactive mode.

### When to use

| Question | Use ast-grep? |
|---|---|
| "Find all functions returning Optional[T] without a None check" | **Yes** — AST pattern, grep misses variants |
| "Find all bare `except:` blocks" | **Yes** |
| "Find calls to `.send()` without `await`" | **Yes** |
| "Batch rename a method across 50 files" | **Yes** — `-r '$OBJ.newName($$$ARGS)'` |
| "Find all try/except/finally missing a finally" | **Yes** |
| "Add a team lint rule for our anti-pattern" | **Yes** — YAML rule file in `.ast-grep/rules/` |

### Beyond ad-hoc: persisted rule sets

Define team-enforced patterns in `.ast-grep/rules/*.yaml` and run them as a pre-commit lint step:

```bash
ast-grep --config .ast-grep/rules/ --check
```

This makes institutional knowledge (we don't catch bare `Exception`, we always log before raise) into checkable, version-controlled rules — the same as a linter plugin but without writing a plugin.

### Pre-edit and post-edit

- **Pre-edit**: CodeGraph finds the symbol; ast-grep finds every call site that needs updating.
- **Post-edit**: `ast-grep --config .ast-grep/rules/` as a regression guard — catches if the edit introduced a pattern violation.

Run ast-grep **in parallel** with CodeGraph when the question has both a "find this symbol" and "find all places with this pattern" component. Same question, two tools, half the wall time.

### When this layer can't answer

- String literals, comments, config files → grep
- Generated or minified files → grep
- Languages without a tree-sitter grammar → grep

### Tips

- Single-quote patterns (`'$PAT'`) to prevent shell expansion of `$`
- `$$$META` for ellipsis (zero or more nodes)
- `--interactive` for safe batch rewrites
- `--format json` for programmatic processing of results

---

## Layer 4: grep / ripgrep

**Precondition:** None — works everywhere, but use only when higher layers can't.

### When to use — and only then

| Use case | Example |
|---|---|
| String literals | `rg 'rate_limit' src/` |
| Config values | `rg 'postgres://' config/` |
| Comments | `rg '# TODO: fix' src/` |
| Non-code files | `rg 'firecrawl' .firecrawl/` |
| Generated files | `rg 'autogenerated' dist/` |
| Quick existence check | `rg '__init__' src/app/features/` |
| **Discoverer for higher layers** | grep finds a match → that gives you a symbol name → CodeGraph for source |

### When NOT to use

- Finding a symbol definition or its callers → CodeGraph
- Tracing call relationships → CodeGraph
- Understanding architecture → Graphify
- Matching by code structure → ast-grep

Grep is the **most expensive tool per unit of understanding** because every hit needs a Read call to interpret. Use it as a discoverer, not an interpreter: grep finds the anchor, then CodeGraph/Graphify builds the understanding.

---

## The Vague Question Workflow

When you don't know what symbol to start with (the most common real-world scenario):

```
1. User: "find the email sending code"
2. Graphify query "email sending in this codebase"
     → returns god nodes, communities, symbols related to email
     → you now have symbol names: "EmailService", "send_email", ...
3. CodeGraph codegraph_explore "EmailService send_email"
     → returns verbatim source, callers, blast radius
4. ast-grep -p 'send_email($$$)' (in parallel with step 3)
     → finds every call site
```

Without Graphify, step 2 would be `rg -r 'email\|send\|mail' src/` + 10+ Read calls. Graphify collapses the discovery into one query.

---

## Edits and Changes — Full Workflow

```
1. Know the symbol?  → codegraph_explore for source + blast radius
   Vague concept?    → graphify query to discover symbol names first
   Neither?          → grep to find an anchor → CodeGraph or Graphify

2. Need to rewrite occurrences? → ast-grep -p 'pattern' (parallel with step 1)

3. Edit files

4. Check CodeGraph staleness banner → Read files pending re-index

5. Verify:
   ├── graphify update .                     (graph stays current)
   ├── codegraph_affected tests              (run only affected tests)
   ├── ast-grep --config .ast-grep/rules/    (regression lint)
   └── linter + type checker                 (project tools)

6. Commit
```

---

## Why This Order

Each tool has a different **unit of analysis**. The hierarchy maximizes **information-per-tool-call**:

| Tool | Unit | Query shape | Returns |
|---|---|---|---|
| CodeGraph | **symbol** | "show me X, who calls it" | source + edges + blast radius |
| Graphify | **concept / community** | "how does X relate to Y" | subgraph nodes + paths |
| ast-grep | **syntax tree** | "find all patterns like P" | file:line matches |
| grep | **byte string** | "find this text anywhere" | file:line hits |

But CodeGraph and Graphify are **parallel entry points**, not strict layers. The branching factor is what you know:

- **Know a symbol name** → CodeGraph first (collapses discovery into one call)
- **Know a vague concept** → Graphify first (fuzzy matching, then CodeGraph for source)
- **Know a string** → grep first (as discoverer, then escalate back up to CodeGraph or Graphify)
- **Know a pattern** → ast-grep first (structural search when you don't need relationships)

---

## For the chosen ones

The real power isn't the tools — it's the **branching decision in under a second**: knowing which entry point a question maps to without thinking about tiers. "How does X work" with a known name → CodeGraph. "What connects auth to billing" with no name → Graphify. "Find the OpenAPI spec path" as a string → grep as discoverer, then codegraph_explore once you have the symbol that loads it.

The hierarchy is training wheels. Once you internalize the decision tree, you stop going through layers and start answering questions directly in the correct tool. The sign you've arrived: you feel *physical discomfort* reaching for grep on indexed code — you're paying the exploration tax for information the graph already has. One codegraph_explore call returns what 10-50 grep+Read round-trips build slowly. The tools exist so you never pay that tax a single time.

Two more signals you're doing it right:
- You never run `rg` twice for the same task. The first hit gives you a symbol name → CodeGraph takes it from there.
- You can answer "will this work?" faster than "let me search for it" — because CodeGraph gives you the blast radius before you edit, not after.