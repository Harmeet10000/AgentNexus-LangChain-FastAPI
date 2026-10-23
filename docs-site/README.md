# Docs Site

Mintlify documentation for AgentNexus.

## Adding a new page
1. Create `docs-site/<group>/<page>.mdx` with front-matter `title` + `description`.
2. Add the path (without `.mdx`) to `docs-site/docs.json` navigation.
3. Verify locally: `uv run python docs-site/scripts/validate_frontmatter.py && uv run python docs-site/scripts/check_nav_completeness.py && uv run python docs-site/scripts/verify_docs.py`

## Updating the OpenAPI spec and local Postman collection
```bash
uv run python docs-site/scripts/extract_openapi.py
uv run python docs-site/scripts/sync_postman_collection.py
uv run python docs-site/scripts/sync_postman_collection.py --check
```

The extractor builds the FastAPI app in-process without running its lifespan. The Postman sync preserves curated request descriptions, auth, scripts, and safety settings while reconciling the operation set and schema-owned request details. The generated collection is a local manual-testing fixture and is intentionally ignored because it may contain private environment values; keep real tokens and provider secrets in a private Postman environment. Commit `openapi.json` and the sync script/tests. CI checks the checked-in snapshot and the live app-built schema; the collection synchronizer is covered by fixture tests and can be run locally when the fixture exists.

## Deploy
Docs are hosted on Mintlify Cloud, which builds on push to `main` once the repo is connected — no CI workflow needed. (The Mintlify CLI has no static-build command, so GitHub Pages deployment is not supported; `docs-ci.yml` covers validation on PRs.)

## Verification
```bash
uv run python docs-site/scripts/verify_docs.py
uv run python docs-site/scripts/validate_frontmatter.py
uv run python docs-site/scripts/check_nav_completeness.py
```
