---
description: Run ruff format + ruff check on the entire src/ tree
---

Run `uv run ruff format src/` then `uv run ruff check src/` and report any errors found.

Display the output in a concise summary:
- Number of files formatted (if any)
- Number and type of lint errors (if any)
- A one-line verdict: "✅ Clean" or "❌ N errors remaining"
