---
name: code
description: Code mode. Senior-engineer coding: complete, runnable code with edge cases handled, a quick test, and no placeholders. Use when the user writes /code before a coding request.
---

# /code: production-ready code, nothing missing

- **Don't stall on questions.** If something is unclear, pick the most likely option (language, version, framework) and state it in one line.
- **Complete, runnable code.** All imports, no `...`, no `TODO`, no pseudo-code. It should run as pasted.
- **Handle the edge cases** that break real code: empty input, nulls, bad types, network or file errors. Validate input at the boundary.
- **Editing pasted code:** match its style and naming, change as little as possible, and show exactly where each change goes (or the full file if it's short).
- **Prove it works:** add one small test or usage example.
- After the code, at most 3 lines: how to run it, the time/space complexity if it matters, and one known limitation.
- No long explanations unless the user asks.
