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

### One-liners

```bash
# Find type assertions
ast-grep -p '$x as $T' -l ts

# Empty catch blocks
ast-grep -p 'catch ($_) { }'

# Convert require to import
ast-grep -p 'const $NAME = require($PATH)' -r 'import $NAME from $PATH' -l js

# Nullish coalescing
ast-grep -p '$A = $A ?? $B' -r '$A ??= $B'
```

Full lint/transform/YAML reference at `/home/harmeet/Desktop/prompts/skills/ast-grep-skill/`.

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
