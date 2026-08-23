"""Retired feature package. Everything here moved to `app.features.documents`.

This package held a second document/chunk schema — `search_documents` and `search_chunks` — beside
the unified one, with its own models, repository, router, DTOs, constants and Celery ingest path.
Step 10 of `documents-unified-schema` deleted all of it. Nothing is re-exported on purpose: a shim
here would let an import that should fail keep working, and the point of the deletion is that the
twin schema has exactly one successor rather than two spellings of the same thing.

Where things went, for anyone arriving from an old import:

| Was | Is now |
|---|---|
| `search.chunking`, `search.fusion`, `search.rag` | the same names under `app.features.documents` |
| `search.model.SearchChunk`, `.SearchDocument` | `documents.model.UnifiedChunk`, `.UnifiedDocument` |
| `search.repository.SearchRepository` | `documents.repository.DocumentRepository` |
| `search.service.SearchService` | `documents.service.DocumentQueryService` / `DocumentCommandService` |
| `search.service.ask_legal` | `documents.service.DocumentQueryService.ask_via_retrieval_graph` |
| `search.constants` | `documents.constants` |
| `tasks.search_tasks.ingest_search_document` | `tasks.document_tasks.ingest_document` |

The router this package exported was **never mounted** — neither `api/v1.py` nor `api/v2.py` ever
included it — so no HTTP route was lost with it. That is worth stating rather than leaving to be
rediscovered: the endpoints looked live in the source and were unreachable in the running app.

The directory itself survives only so that a stale `import app.features.search` fails on the
*symbol* rather than on the package, which is the clearer error message of the two.
"""

__all__: list[str] = []
