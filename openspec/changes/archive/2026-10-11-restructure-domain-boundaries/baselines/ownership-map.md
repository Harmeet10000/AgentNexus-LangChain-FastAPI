# Ownership map — restructure-domain-boundaries task 1.3

Produced 2026-10-10 by static import scan (`ast`, absolute imports only).
Composition roots (allowed to import features): `app.api.*` (router assembly),
`app.lifecycle.*` (lifespan/graph wiring), `app.worker.*`, `database.registry`
+ seeders, MCP tool assembly, `app.examples.*` (non-production).

## Reusable → feature edges (must be removed)

| User | Reaches into | Fix (task) |
|---|---|---|
| `app.utils.rate_limit.dependencies` | `features.auth` (token decode) | 3.3 → `app.security` |
| `app.connections.cognee` | `features.documents.model` (embedding dim) | 3.2 → retrieval embedding contract |
| `app.shared.rag.docling.chunker` | `features.documents` (policy/classification) | 3.2 chunk policy / 5.2 engines |
| `app.shared.langgraph_layer.retrieval_kb.graph` | `features.documents` services | 4.2 move graph under documents |
| `app.shared.langchain_layer.agents.tools.search_legal_precedents` | `features.documents` | 4.3 move to owning feature |

## Feature → reusable edges (allowed direction, kept)

`features.documents` → `shared.langgraph_layer` / `shared.rag/*`;
`features.*` → `shared.services.*`, `utils.cache`, `shared.result`.

## Eager package initializers (must go — task 3.1)

- `features/auth/__init__.py` — imports router, repositories, websocket impl.
- `features/billing/invoices/__init__.py` — imports router.
- Kept (pure type exports): `features/chat` (models), `features/billing/credits`
  (dto/models). All other feature `__init__` files are already empty/docstring-only.

## Explicitly not moved (assessment rejections, recorded)

- `lifecycle/signals.py` stays (Uvicorn delegation, not Celery wiring).
- `features/search/` tombstone stays until zero-import evidence beyond grep
  (task 2.5 proves removal only if unreferenced incl. dynamic imports).
- Outbox relay vs writer, object-storage lifecycle vs operations, Tavily client
  vs service adaptation: consolidate only with behavior-matrix evidence (5.1).
