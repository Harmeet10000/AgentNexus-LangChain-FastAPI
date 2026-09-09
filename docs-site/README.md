# Docs Site

Mintlify documentation for AgentNexus.

## Adding a new page
1. Create `docs-site/<group>/<page>.mdx` with front-matter `title` + `description`.
2. Add the path (without `.mdx`) to `docs-site/mint.json` navigation.
3. Verify locally: `uv run python docs-site/scripts/validate_frontmatter.py && uv run python docs-site/scripts/check_nav_completeness.py && uv run python docs-site/scripts/verify_docs.py`

## Updating the OpenAPI spec
```bash
uv run python docs-site/scripts/extract_openapi.py
# Verify rendering: check docs-site/api-reference/overview.mdx references match openapi.json paths
```

## Deploy
Docs are hosted on Mintlify Cloud, which builds on push to `main` once the repo is connected — no CI workflow needed. (The Mintlify CLI has no static-build command, so GitHub Pages deployment is not supported; `docs-ci.yml` covers validation on PRs.)

## Verification
```bash
uv run python docs-site/scripts/verify_docs.py
uv run python docs-site/scripts/validate_frontmatter.py
uv run python docs-site/scripts/check_nav_completeness.py
```
