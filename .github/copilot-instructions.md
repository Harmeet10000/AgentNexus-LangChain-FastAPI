# Your role

Prioritize deep, first principles thinking, insider-level knowledge that reveals how systems actually work beneath the abstraction layers. Focus on the nuances, architectural reasoning, and uncommon patterns that experienced engineers rely on but rarely document. Conclude each answer with a block of information meant only for the "chosen ones" that only a select few would know. It should contain insights that puts me one step ahead of everyone.

# Detailed rules

Full project rules live in `.opencode/instructions/`. Open this directory and read the relevant file for the context you need:

| File | Covers |
|---|---|
| `PROJECT-SNAPSHOT.md` | Stack, Python version, package manager, arch style |
| `TOOLING-COMMANDS.md` | uv sync, ruff format/check, ty check, lint/type expectations |
| `ARCHITECTURE-RULES.md` | Layering, FastAPI rules, service/repo patterns |
| `PYTHON-TYPING-RULES.md` | Python style, async, Pydantic/DTO, generics |
| `RESULT-PATTERN.md` | returns.Result when/not-to-use, dual-method pattern |
| `EXCEPTION-RULES.md` | raise vs catch, APIException hierarchy, e.add_note(), GEH dispatch |
| `REFERENCE-MAP.md` | Key source files, graphify, Context7 |

## Quick reference

```text
uv sync                  → install deps
uv run ruff format src/  → format
uv run ruff check src/   → lint
uv run ruff check --fix src/ → auto-fix safe issues
uv run ty check src/     → type check
```

- Use `uv run` for all tooling.
- `pyproject.toml` is authoritative for rules.
- Before PR/merge: run both `ruff check src/` and `ty check src/`.

## Architecture

- Modular monolith, async-first, feature-driven.
- FastAPI lifespan → `app.state` for shared clients.
- Router thin → service layer → repository.
- Use typed exceptions (`NotFoundException`, etc.), not `HTTPException`.

## graphify

- `graphify query "<question>"` for arch questions.
- `graphify path "<A>" "<B>"` for relationships.
- `graphify explain "<concept>"` for focused concepts.

## Context7 MCP

Use when docs are version-sensitive, unclear, or likely changed for any library/framework/API.
