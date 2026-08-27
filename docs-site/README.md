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
Push to `main` triggers `.github/workflows/deploy-docs.yml` (GitHub Pages). Alternatively Mintlify Cloud builds on push.

## Verification
```bash
uv run python docs-site/scripts/verify_docs.py
uv run python docs-site/scripts/validate_frontmatter.py
uv run python docs-site/scripts/check_nav_completeness.py
```
