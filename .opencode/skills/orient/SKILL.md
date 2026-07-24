---
name: orient
description: |
  Orient: find/explain/search codebase or look up external docs. Use when user asks where/find/how/explain/architecture, or needs external research. First tool before grep/rg.
---

# orient

Orient before acting. Two **branches**: **codebase** (local) and **external** (outside the repo). Escalate through the rungs in order. Question answered? Stop. Not answered? Next rung. All rungs exhausted? Surface the narrowed question.

---

## Codebase branch

### codegraph_explore (MCP)

One call returns verbatim source + call paths + blast radius. Always try first.

```
query: "how does auth work"
query: "AuthService loginUser"
query: "src/app/utils/exceptions.py"
```

Thin results? Fall through.

### graphify (CLI)

Requires `graphify-out/graph.json`. If missing, run `graphify build .` first.

```
graphify query "<question>"     # BFS — architecture, structure, connections
graphify path "A" "B"           # shortest path between two symbols
graphify explain "X"            # plain-language node + neighbours
graphify affected "X"           # reverse traversal — impact analysis
graphify tree                   # interactive D3 HTML
```

Flags: `--graph <path>` for a specific `graph.json`, `--budget N` for output token cap.

### ast-grep (CLI)

AST-aware structural search. Use for precision: calls matching a specific argument shape.

```bash
ast-grep -p '<pattern>' -l <language>
ast-grep -p 'console.log($$$)' -l ts
ast-grep -p '$PROP && $PROP()' -r '$PROP?.()' -l ts --interactive
```

Full pattern syntax, flags, one-liners in [`REFERENCE.md`](REFERENCE.md).

### grep / ripgrep

Text search. Last resort.

---

## External branch

### Context7 MCP

For library/framework/API/CLI docs.

1. `context7_resolve-library-id` — resolve package name to Context7 ID
2. `context7_query-docs` — query using the resolved ID

### firecrawl (CLI)

For everything Context7 doesn't cover — web search, page scraping, site mapping.

See [`REFERENCE.md`](REFERENCE.md) for exact CLI syntax. Always write output to `.firecrawl/`.

---

## After modifying code

```bash
graphify update .
```
