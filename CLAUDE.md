# langchain-fastapi-production

Python 3.12 · `uv` · FastAPI · Pydantic v2 · LangChain/LangGraph · SQLAlchemy · Beanie · Redis · Celery.
Modular monolith, feature-driven, async-first.

## Search strategy

Route on **what you hold**. There is no mandatory first tool.

| What you hold | First call |
|---|---|
| exact symbol / file name | `codegraph_explore "<names>"` |
| a concept, no name | `graphify query` → feed the names it returns to `codegraph_explore` |
| two known symbols, want the link | `graphify path "A" "B"` (88 B — cheapest call in the repo) |
| one symbol, want its edges | `graphify explain "X"` |
| literal string / config | `rg` → hit gives a name → back up the ladder |
| structural shape | `ast-grep -p` |
| "what breaks if I change X" | `graphify affected "X"` — not `codegraph_affected` |

`codegraph_explore` is Read-equivalent — prefer it over Read on indexed code. Grep is a **discoverer, not an interpreter**: a hit gives you a symbol name, and that goes back up the ladder.

Two corrections to earlier guidance, both measured (2026-08-16):

- **Don't start a vague question at `codegraph_explore`.** On *"how does authentication work"* it returned 2 files and missed `security.py`, `dependencies.py`, and `service.py`. `graphify query` named all three for fewer tokens.
- **`graphify query` costs ~6.6 KB, not "~200 tokens"** — and returns names only, zero edges. Pass `--budget 12000`; the default 2000 silently drops ~23% of the traversal. Edges come from `explain` and `path`.

**Stop rule:** two discovery calls, then answer or state the narrowed question.

After modifying code both indexes refresh via hooks (`codegraph sync` per edit, `graphify update .` at turn end). Verify with the project's own checks — `uv run ruff check --fix src/`, `uv run ty check src/`, `uv run pytest`, `ast-grep scan src/`.

`.opencode/skills/orient/SKILL.md` is the **sole authority** on routing — this table is its summary. It also holds graphify's depth-2 horizon, the dated cost table, ast-grep patterns, and Context7/firecrawl for external docs.

## Commands

Always `uv run` — never bare `ruff` or `ty`.

```bash
uv run ruff format src/       # format
uv run ruff check --fix src/  # lint, safe autofix
uv run ty check src/          # types
uv run pytest                 # tests
```

`ruff` is source of truth for format and lint; `ty` for typing; `pyproject.toml` is authoritative for enabled rules. Prefer patterns that satisfy the checks over `# noqa` or `# type: ignore`.

## Detailed rules

Read the relevant file before working in its area:

| File | Covers |
|---|---|
| `PROJECT-SNAPSHOT.md` | Stack, Python version, package manager, arch style |
| `TOOLING-COMMANDS.md` | uv/ruff/ty commands, lint and type expectations |
| `ARCHITECTURE-RULES.md` | Layering, FastAPI rules, service/repo patterns |
| `PYTHON-TYPING-RULES.md` | Style, async, Pydantic/DTO, generics |
| `RESULT-PATTERN.md` | `returns.Result` when and when not, dual-method pattern |
| `EXCEPTION-RULES.md` | raise vs catch, APIException hierarchy, `e.add_note()`, GEH dispatch |
| `CODE-QUALITY-PATTERNS.md` | Quality patterns and anti-patterns |
| `REFERENCE-MAP.md` | Key source files, graphify, Context7 |

All under `.opencode/instructions/`.

## Key files

- Exceptions: `src/app/utils/exceptions.py` · handler: `src/app/middleware/global_exception_handler.py`
- Result→exception bridge: `src/app/shared/result/mappers.py`
- Response envelope: `src/app/utils/response_type.py`
- Cache: `src/app/utils/cache/redis_func.py` · Lifespan: `src/app/lifecycle/lifespan.py`
- Examples belong in `src/app/examples/`

## Relay

`/relay <task>` runs the four-leg workflow — scout, planner, verifier, anchor — with you orchestrating throughout. See `.claude/skills/relay/SKILL.md`.

## Response priority

1. **Answer with first-principles depth** — how systems work beneath the abstraction, the nuances and uncommon patterns experienced engineers rely on but rarely document.
2. **Multiple options?** Give a pros/cons table.
3. **Append "Deep Internals"** — 1–3 non-obvious, underdocumented, or counterintuitive facts about the libraries, APIs, or patterns in play.
4. **Context missing?** Ask one focused clarifying question rather than proceeding.

## Skills

Matt Pocock's skills live in `~/.agents/skills/`. Before loading one, ask which to use, and keep asking until the choice is unambiguous.

Project skills are in `.opencode/skills/` (orient, graphify, openspec-*, caveman) and `.github/skills/`. Token compression (`caveman`) applies only when explicitly requested.

`/graphify` invokes the graphify skill. Prefer `graphify query`/`path`/`explain` over reading `GRAPH_REPORT.md` (71 KB). Dirty `graphify-out/` files after hooks are expected and are not a reason to skip it.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- Route by the Search strategy table above — `graphify query` is the entry point for **concepts without a name**, not for every question. It returns names, not edges; `path` and `explain` return edges.
- Pass `--budget 12000` to `query`. Depth is hardcoded at 2 — to go deeper, re-seed from a frontier symbol with `explain`/`path`/`affected`.
- Community names are hub-derived and churn on every update — never reference one by name.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- Refresh is hooked (`codegraph sync` per edit, `graphify update .` at turn end). By hand: `graphify update .` (25s, AST-only, no API cost); add `--force` after a refactor that deletes code. Never `--no-cluster` — it drops every community.
