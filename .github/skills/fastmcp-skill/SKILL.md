---
name: fastmcp-skill
description: FastMCP documentation navigator and implementation guide. Use when Agent needs to answer FastMCP questions, build or debug FastMCP servers or clients, configure authentication or authorization, choose transports or deployment patterns, integrate FastMCP with FastAPI, OpenAPI, ChatGPT, Claude, Cursor, Gemini, or inspect package-level FastMCP API reference details from the bundled docs.
---

# FastMCP Skill

Use this skill as a dispatcher over the split FastMCP documentation in `references/`.

## Workflow

1. Open `references/index.md` first.
2. Pick the smallest relevant reference file instead of reading multiple large files.
3. Prefer the guide-style references (`01` through `18`) before the package API references (`19` through `24`).
4. Use `rg` on `references/` when the user asks for a specific class, function, provider, transport, or integration.

## Reference Selection

- For apps, prefab UI, charts, forms, and custom HTML apps, use `references/01-apps-ui.md`.
- For CLI usage, server installation, run commands, and client auth helpers, use `references/02-cli-install-run-auth.md`.
- For general client usage, callbacks, prompts, resources, notifications, sampling, and background tasks, use `references/03-client-core.md`.
- For transports, tool calls, HTTP deployment, ASGI mounting, and production hosting, use `references/04-client-transports-deployment.md`.
- For setup, project config, installation, quickstart, and intro material, use `references/05-project-setup-quickstart.md`.
- For provider-specific OAuth setup, start with `references/06-auth-provider-guides-1.md` and `references/08-config-openapi-auth-guides-2.md`.
- For ChatGPT, Claude, Cursor, FastAPI, Gemini, GitHub, Google, and Goose integrations, use `references/07-client-framework-integrations.md`.
- For core auth architecture, OAuth servers and proxies, remote auth, token verification, and authorization concepts, use `references/09-auth-architecture.md`.
- For composition, context, dependency injection, icons, and lifespans, use `references/10-composition-context-lifecycle.md`.
- For middleware, prompts, providers, resources, sampling, and provider patterns, use `references/11-middleware-lifecycle.md` and `references/12-provider-patterns-resources.md`.
- For server internals, tools, code mode, transforms, visibility, telemetry, testing, and storage, use `references/13-server-core-storage-testing.md`, `references/14-tools-and-code-mode.md`, and `references/15-transforms-visibility.md`.
- For changelog, contribution workflow, and migration guidance, use `references/16-changelog-and-auth-utils.md`, `references/17-contributing-and-tests.md`, and `references/18-upgrade-guides.md`.
- For package-level API details, use the `references/19-*.md` through `references/24-*.md` files.

## Search Shortcuts

Use targeted search before opening a large reference:

```bash
rg -n "FastAPI|OpenAPI|ChatGPT|Cursor|Claude|Gemini" .github/skills/fastmcp-skill/references
rg -n "OAuthProxy|RemoteAuthProvider|TokenVerifier|MultiAuth" .github/skills/fastmcp-skill/references
rg -n "FastMCP\\(|@mcp\\.tool|@mcp\\.resource|AppConfig" .github/skills/fastmcp-skill/references
rg -n "fastmcp\\.server\\.|fastmcp\\.client\\." .github/skills/fastmcp-skill/references
```

## Notes

- The original 49k-line documentation dump was split into bounded reference files so agents do not need to load the whole source at once.
- If the question is package- or symbol-specific, search first, then open the matching API reference chunk.
- If the question is implementation- or workflow-oriented, start with the guide references and only fall back to API reference files when needed.
    