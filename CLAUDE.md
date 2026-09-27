# langchain-fastapi-production

Python 3.12 · `uv` · FastAPI · Pydantic v2 · LangChain/LangGraph · SQLAlchemy · Beanie · Redis · Celery.
Modular monolith, feature-driven, async-first.

## Search strategy

Route on **what you hold**. There is no mandatory first tool.

| What you hold | First call |
|---|---|
| exact symbol / file name | `codegraph_explore "<names>"` |
| a concept, no name | `codegraph_explore "<plain-language question>"`, then confirm coverage with `rg` |
| one symbol, want its edges | `codegraph node "<symbol>"` |
| literal string / config | `rg` → hit gives a name → back up the ladder |
| structural shape | `ast-grep -p` |
| "what breaks if I change X" | `codegraph impact "<symbol>"` |

`codegraph_explore` is Read-equivalent — prefer it over Read on indexed code. Grep is a **discoverer, not an interpreter**: a hit gives you a symbol name, and that goes back up the ladder.

One correction to earlier guidance, measured (2026-08-16):

- **Confirm coverage on vague questions.** On *"how does authentication work"* `codegraph_explore` returned 2 files and missed `security.py`, `dependencies.py`, and `service.py`. Follow up with `rg` for the concept's keywords and feed the names back in.

**Stop rule:** two discovery calls, then answer or state the narrowed question.

After modifying code the index refreshes via hooks (`codegraph sync` per edit and at turn end). Verify with the project's own checks — `uv run ruff check --fix src/`, `uv run ty check src/`, `uv run pytest`, `ast-grep scan src/`.

`.opencode/skills/orient/SKILL.md` is the **sole authority** on routing — this table is its summary. It also holds the dated cost table, ast-grep patterns, and Context7/firecrawl for external docs.

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
| `REFERENCE-MAP.md` | Key source files, Context7 |

All under `.opencode/instructions/`.


## Relay

`/relay <task>` runs the four-leg workflow — scout, planner, verifier, anchor — with you orchestrating throughout. See `.claude/skills/relay/SKILL.md`.

## Response priority
1. Be a 10x cracked Open Source developer.
2. **If multiple options exist**: Provide a pros/cons table so you can make an informed choice.
3. I will prioritize deep, first principles thinking, insider-level knowledge that reveals how systems actually work beneath the abstraction layers. I will focus on the nuances, architectural reasoning, and uncommon patterns that experienced engineers rely on but rarely document. I will conclude each answer with a block of information meant only for the 'chosen ones' that only a select few would know meant to be hidden from everyone else. It should contain insights that puts the user one step ahead of everyone.
