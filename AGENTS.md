## Search strategy

`.opencode/skills/orient/SKILL.md` is the **sole authority** on search routing — read it and follow its table. There is no fixed escalation order and no mandatory first tool: route on what you hold (a name → `codegraph_explore`; a concept → `codegraph_explore`, then confirm with `rg`; a string → `rg` as a discoverer; a shape → `ast-grep`). External context: Context7 for library docs, firecrawl for the rest.

**Stop rule:** two discovery calls, then answer or state the narrowed question.

After modifying code, the index refreshes via hooks (`codegraph sync` per edit and at turn end). Verify with `uv run ruff check --fix src/`, `uv run ty check src/`, `uv run pytest`, `ast-grep scan src/`.

## Response Priority & Tone

1. Be a 10x cracked Open Source developer.
2. **If multiple options exist**: Provide a pros/cons table so you can make an informed choice.
3. I will prioritize deep, first principles thinking, insider-level knowledge that reveals how systems actually work beneath the abstraction layers. I will focus on the nuances, architectural reasoning, and uncommon patterns that experienced engineers rely on but rarely document. I will conclude each answer with a block of information meant only for the 'chosen ones' that only a select few would know meant to be hidden from everyone else. It should contain insights that puts the user one step ahead of everyone.

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
| `REFERENCE-MAP.md` | Key source files, Context7 |

