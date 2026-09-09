# File-Ownership Ledger — seven-change RAG cluster

Governed by `docs/relay/IMPLEMENTATION-PROTOCOL.md` §3. The orchestrator
refuses to schedule two changes in one wave whose owned sets intersect,
except the two managed resources in protocol §4.

## Active wave

| Wave | Change | Branch | Worktree | State |
|---|---|---|---|---|
| W0 | `rag-tree-repair` | `impl/rag-tree-repair` | `../lcfp-rag-tree-repair` | in progress |

## Declared owned-path sets (from each proposal's Impact section)

| Change | Owned paths |
|---|---|
| `rag-tree-repair` | `src/app/shared/rag/docling/**`, `src/app/shared/rag/langextract/langextract_to_graph.py`, `src/app/features/documents/**` (imports only), `src/app/shared/langgraph_layer/ingestion_kb/nodes.py`, `src/app/utils/embedding.py`, `src/app/examples/policy_examples.py`, `src/alembic/env.py`, `src/alembic/versions/0013_*`, `tests/unit/shared/rag/**`, `openspec/specs/**` |
| `rag-eval-harness` | `src/app/shared/evaluation/**` (new), `tests/unit/shared/evaluation/**`, `tests/property/**`, golden-set data file |
| `graph-lifecycle` | `src/app/lifecycle/**`, `src/app/shared/langgraph_layer/checkpointer.py`, `src/app/shared/langgraph_layer/open_deep_search/graph.py`, `src/app/shared/rag/graphiti/registry.py`, `src/app/features/documents/service.py`, Celery worker bootstrap module |
| `ingestion-chunking` | `src/app/features/documents/chunking.py`, `src/app/shared/rag/docling/**` chunker, **`model.py` (column definitions)**, **`src/alembic/versions/` (holds the token)**; §7 adds `pyproject.toml`, `uv.lock`, `retrieval_kb/reranker.py` |
| `retrieval-sql` | `src/app/features/documents/repository.py`, `fusion.py`, `constants.py`, `service.py`, `src/app/shared/langgraph_layer/retrieval_kb/nodes.py`, `src/app/examples/policy_examples.py`, **`model.py` (`__table_args__` only)**, **index-only migration** |
| `agentic-retrieval` | `src/app/shared/langgraph_layer/retrieval_kb/**`, `src/app/features/documents/rag.py` |
| `knowledge-stack` | `src/app/shared/rag/langextract/**`, `src/app/shared/rag/pageindex/**` (deleted), `src/app/shared/rag/__init__.py`, `src/app/shared/rag/graphiti/write_clause_episodes.py`, `src/app/lifecycle/lifespan.py`, `src/app/config/settings.py`, `docs-site/configuration/environment-variables.mdx`, `src/app/features/documents/repository.py` + `service.py` (navigation branch only), conditionally `model.py` + migration |

## Standing edges

- W2 token (Alembic head + `model.py`): held by `ingestion-chunking`.
  `retrieval-sql` writes no migration and defers its `model.py` edit to
  its last task post-rebase. Fallback: serial W2.
- Archive ordering: `rag-tree-repair` must be **archived** (not merely
  merged) before `ingestion-chunking` or `retrieval-sql` can be archived
  (MODIFIED targets absent from `openspec/specs/`).
- W3/W5 default: serial (retrieval attribution). Parallel only on human
  instruction.

## Amendments

- None yet.
