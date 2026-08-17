# orient reference

Tool details pushed down from SKILL.md to keep the main skill lean. Consult when you need exact CLI syntax.

---

## ast-grep — structural code search

AST-aware search over syntax trees. Powers through 10k+ files in seconds.

```bash
ast-grep -p '<pattern>' -l <language>
```

### Pattern syntax

| Meta-variable | Matches |
|---|---|
| `$VAR` | One AST node (uppercase) |
| `$$$` | Zero or more AST nodes |
| `$_` | Anonymous wildcard (discarded) |
| `$A == $A` | Back-reference enforces structural equality |

### Key flags

| Flag | Purpose |
|---|---|
| `-l, --lang <lang>` | Target language (auto-detected from extension) |
| `-r, --rewrite <string>` | Replacement for search-and-replace |
| `--interactive` | Confirm each replacement |
| `--json` / `--json=stream` | Machine-readable output |
| `--strictness ast\|relaxed\|signature` | Matching precision |
| `-i, --ignore-case` | Case-insensitive |
| `--stdin` | Read code from pipe |

### One-liners — this repo is Python, so patterns are Python

Hit counts measured against `src/` on 2026-08-16; they are there so you can tell
a broken pattern from a clean codebase.

```bash
# Raw HTTPException instead of a typed APIException subclass          → 4
ast-grep -p 'raise HTTPException($$$ARGS)' -l py src/

# Every injected dependency (this repo uses Annotated, not `x: T = Depends()`)  → 16
ast-grep -p 'Annotated[$T, Depends($D)]' -l py src/

# .unwrap() call sites — each one needs an isinstance/Failure guard above it   → 27
ast-grep -p '$X.unwrap()' -l py src/

# Result construction inside the feature layer                        → 37
ast-grep -p 'Success($V)' -l py src/app/features/

# Retired mapper raise (review each hit; do not batch-rewrite)        → 30
ast-grep -p 'raise app_error_to_exception($E)' -l py src/
```

Two traps that cost real time:

- `$$$ARGS` is the multi-node metavar. `$ARGS` matches exactly **one** node and
  silently misses `HTTPException(status_code=404, detail="...")`.
- **Two `$$$` metavars in one argument list do not parse.**
  `def $FN($$$A, db: Session = Depends($$$D), $$$B)` yields *"Pattern contains an
  ERROR node"* and matches nothing — while still exiting 0, so it reads as a
  clean codebase. Match the parameter shape on its own instead, or express the
  "function containing X" idea as a YAML rule with `has:`.

### When a pattern can't express it: `kind` + `regex`

Some shapes have no literal form — an *empty* capture pattern, for instance.
Match the tree-sitter node kind and constrain it with a regex instead:

```yaml
rule:
  kind: case_pattern
  regex: ^(Success|Failure)\(\s*\)$
```

That is `.ast-grep/rules/no-match-on-result.yml`. Run `ast-grep -p '<pattern>'
-l py --debug-query` to print the parse tree and learn the kind names — it is
also the fastest way to spot an ERROR node before trusting a zero-hit result.

### Project rules

Five vendored rules live in `.ast-grep/rules/`, registered by repo-relative
`sgconfig.yml`. They encode conventions from `.opencode/instructions/` that ruff
cannot express — do not add a rule ruff already covers.

```bash
ast-grep scan src/                   # all rules
ast-grep scan --json=compact src/    # machine-readable
```

They are **not** wired into `.pre-commit-config.yaml`: `no-raw-httpexception` is
severity `error` with live hits, so a blocking hook would fail every commit
today. Run the scan by hand until those are resolved.

Full lint/transform/YAML reference at
`/home/harmeet/Desktop/prompts/skills/ast-grep-skill/` (machine-local, not in
this repo).

---

## firecrawl — web search and scrape

```bash
# Search
firecrawl search "query" -o .firecrawl/search-{tag}.json --json

# Scrape page
firecrawl scrape https://example.com -o .firecrawl/{tag}.md

# Discover site URLs
firecrawl map https://example.com --limit 500 -o .firecrawl/urls.txt
```

Always write to `.firecrawl/`. Run parallel scrapes with `&` and `wait`. Full docs at the `firecrawl` skill.
