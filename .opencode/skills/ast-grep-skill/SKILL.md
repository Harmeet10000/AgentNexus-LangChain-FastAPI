---
name: ast-grep
description: Structural code search, linting, and rewriting using AST-level pattern matching. Use when users need to search code by syntax structure (not text), write custom lint rules, perform codemods across a codebase, replace regex grep for precise AST-aware queries, or create reusable lint rule sets. Covers CLI usage, pattern syntax, rule system (atomic/relational/composite/utility), lint rule authoring, rewrite/transform, project setup, and testing.
version: 0.1.0
ast-grep_version: ">=0.43.0"
last_updated: 2026-06-21
---

# ast-grep

## Overview

ast-grep searches and transforms code using AST patterns instead of text regex. Think `grep` × `eslint` × `codemod` — but polyglot, with a single pattern language across 20+ languages via tree-sitter.

Powered by Rust. Handles 10k+ files in seconds. Supports search, lint, rewrite, and programmatic API usage.

### Supported Languages

| Domain | Languages |
|--------|-----------|
| System | C, C++, Rust |
| Server | Go, Java, Python, C# |
| Web | JS(X), TS(X), HTML, CSS |
| Mobile | Kotlin, Swift |
| Config | JSON, YAML, HCL |
| Scripting | Lua, Nix |

## Quick Start

### Installation

```bash
# npm (macOS/Linux/Windows)
npm i -g @ast-grep/cli

# pip
pip install ast-grep-cli

# cargo
cargo install ast-grep --locked
```

### First Search

```bash
# Find all console.log calls
ast-grep -p 'console.log($$$)' -l ts

# Find with rewite
ast-grep -p '$PROP && $PROP()' -r '$PROP?.()' -l ts

# Interactive mode
ast-grep -p 'var $NAME = $VAL' -r 'const $NAME = $VAL' --interactive
```

> Always single-quote patterns to prevent shell expansion of `$`.

## Reference Files

The skill is split into focused reference files for easier navigation:

| File | Covers |
|------|--------|
| `references/cli-usage.md` | CLI commands, flags, JSON output, stdin, completions |
| `references/pattern-syntax.md` | Meta variables, capturing, anonymous patterns, matching strictness |
| `references/rules-overview.md` | Rule essentials, atomic rules (pattern, kind, regex, nthChild, range) |
| `references/relational-rules.md` | Relational rules (inside, has, follows, precedes), composite rules (all, any, not, matches), ESQuery selectors |
| `references/lint-rules.md` | Lint rule structure, severity, files/ignores, labels, inline suppression |
| `references/rewrite-transform.md` | fix, transform operators (9 types), FixConfig, rewriters |
| `references/project-setup.md` | sgconfig.yml, CI/CD, advanced config |
| `references/testing.md` | Test file structure, running tests |
| `references/common-patterns.md` | Quick one-liner patterns |
| `references/editor-integration.md` | VS Code, Neovim, Astra TUI |
| `references/catalog-index.md` | Ready-to-use rule templates index |
| `references/troubleshooting.md` | FAQ, common issues and solutions |
| `references/api-usage.md` | JS (napi), Python, Rust API usage |
| `references/rule_reference.md` | Full rule grammar reference |
| `references/transform-operators.md` | Complete transform operator reference |
| `references/catalog/` | Ready-to-use rule YAML files (7 patterns) |

