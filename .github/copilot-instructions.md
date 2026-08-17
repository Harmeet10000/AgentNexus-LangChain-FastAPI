## Search strategy

Use the `orient` skill (`.opencode/skills/orient/`) for all codebase and external context search — the escalation: codegraph → graphify → ast-grep → grep for local code, Context7 → firecrawl for external. Reach before grep/rg.

After modifying code, run `graphify update .` to keep the graph current.

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
