---
name: graphify
description: Use when exploring codebase architecture, structure, components, finding code, understanding relationships between modules, or answering "how does X work" questions. Graphify queries a pre-built knowledge graph for scoped results. Use before reading files or grepping.
---

# graphify

Query the project's knowledge graph for architecture, structure, and code relationships. Prefer graphify over grep/read for exploration.

## Prerequisites

Graph must exist at `graphify-out/`. If missing, run `graphify build .` first.

## Commands

```bash
# Architecture / structure questions
graphify query "how does auth work"
graphify query "where is the service layer"

# Relationship between two things
graphify path "AuthService" "UserRepository"

# Focused concept explanation
graphify explain "circuit breaker pattern"

# Update graph after code changes (no API cost)
graphify update .
```

## Decision flow

1. `graphify query` for any codebase question — returns scoped subgraph (5-10 nodes), not full files
2. `graphify path` for "how does A connect to B"
3. `graphify explain` for "what is X in this codebase"
4. If graphify returns enough context → use it, skip reading files
5. If graphify lacks detail → fall back to ast-grep for structural patterns, then read specific files
6. After modifying code → `graphify update .`

## Interpreting results

- **NODE** lines: symbol name, source file, line number, community ID
- **Community ID**: groups of related symbols (same feature/module)
- **INFERRED edges**: model-reasoned, may need verification
- **God nodes** (high edge count): core abstractions, high coupling risk

## Token efficiency

graphify queries return ~200-500 tokens vs reading entire files (~2K-20K tokens). Always query first, read only when modifying.
