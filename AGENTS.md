## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## ast-grep

Skill at `.github/skills/ast-grep-skill/SKILL.md`. Invoke `skill: "ast-grep"` before using.

Rules:
- Use ast-grep for structural/AST-based code search (e.g., "find async functions without try-catch", "find all console.log calls")
- For simple text search, prefer grep/ripgrep
- Always verify: `ast-grep --version`
- Rule reference has transform operators, catalog patterns, and FAQ at `references/catalog/` for ready-made rule YAML
