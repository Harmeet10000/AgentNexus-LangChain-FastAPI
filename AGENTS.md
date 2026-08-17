## Search strategy

`.opencode/skills/orient/SKILL.md` is the **sole authority** on search routing — read it and follow its table. There is no fixed escalation order and no mandatory first tool: route on what you hold (a name → `codegraph_explore`; a concept → `graphify query` then codegraph; a string → `rg` as a discoverer; a shape → `ast-grep`). External context: Context7 for library docs, firecrawl for the rest.

**Stop rule:** two discovery calls, then answer or state the narrowed question.

After modifying code, both indexes refresh via hooks (`codegraph sync` per edit, `graphify update .` at turn end). Verify with `uv run ruff check --fix src/`, `uv run ty check src/`, `uv run pytest`, `ast-grep scan src/`.

## Matt Pocock skills

Before loading any Matt Pocock skill (`~/.agents/skills/`), ask which one to use. Ask as many questions as required to remove doubt. Dont be lazy. Out do yourself. 

## Response Priority & Tone

1. **Answer the question with first-principles depth**: Explain how systems actually work beneath the abstraction layers, focusing on nuances, architectural reasoning, and uncommon patterns experienced engineers rely on but rarely document.
2. **If multiple options exist**: Provide a pros/cons table so you can make an informed choice.
3. **Append "Deep Internals" section**: Include 1–3 non-obvious technical facts directly relevant to the current question—specifically about the libraries, APIs, or patterns discussed—that are underdocumented or counterintuitive.
4. **If context is missing**: Ask one focused clarifying question instead of proceeding (e.g., "Which floor?", "Which coordinate space?").
5. **Token compression (caveman skill)**: Use only when explicitly requested; it does not apply by default.

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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- Route by the orient table — `graphify query` is the entry point for **concepts without a name**, not for every question. Pass `--budget 12000`; the default 2000 drops ~23% of the traversal. `query` returns names only; `path` and `explain` return edges.
- Depth on `query` is hardcoded at 2. To go deeper, re-seed from a frontier symbol with `explain`/`path`/`affected --depth N`.
- Community names are hub-derived and churn on every update — never reference one by name.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- Refresh is hooked. By hand: `graphify update .` (25s, AST-only, no API cost); `--force` after a refactor that deletes code; never `--no-cluster` (drops every community).
