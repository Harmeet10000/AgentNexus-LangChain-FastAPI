# Merged Reusable RAG Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the numbered demo RAG files with one reusable application-aligned module and one merged reference document.

**Architecture:** Build a single async-first module around existing LangChain model factories and the app-scoped SQLAlchemy session factory. Strategy helpers share common chunking, embedding, retrieval, serialization, and answer-generation utilities so behavior lives in one place instead of 11 disconnected demos.

**Tech Stack:** FastAPI app state, SQLAlchemy async engine/session, pgvector SQL queries via `text(...)`, LangChain Gemini model factories, `orjson`, project logger, pytest

---

### Task 1: Add focused failing tests for pure helpers

**Files:**
- Create: `tests/unit/shared/rag/test_rag_strategies.py`
- Test: `tests/unit/shared/rag/test_rag_strategies.py`

- [ ] **Step 1: Write the failing test**

```python
from app.shared.rag.strategies import (
    LateChunk,
    deduplicate_strings,
    mean_pool_embeddings,
    parse_query_variants,
    semantic_chunk_text,
    serialize_metadata,
)


def test_parse_query_variants_keeps_original_and_strips_noise() -> None:
    variants = parse_query_variants(
        "refund policy",
        "1. refund terms\n- return policy\n\nrefund policy\ncoverage details",
        limit=3,
    )
    assert variants == [
        "refund policy",
        "refund terms",
        "return policy",
        "coverage details",
    ]


def test_semantic_chunk_text_splits_when_similarity_drops() -> None:
    embeddings = {
        "A": [1.0, 0.0],
        "B": [0.9, 0.1],
        "C": [0.0, 1.0],
    }

    chunks = semantic_chunk_text(
        "A. B. C",
        embedder=lambda text: embeddings[text],
        similarity_threshold=0.75,
    )

    assert chunks == ["A. B", "C"]


def test_mean_pool_embeddings_averages_dimensions() -> None:
    assert mean_pool_embeddings([[1.0, 3.0], [3.0, 5.0]]) == [2.0, 4.0]


def test_deduplicate_strings_preserves_first_seen_order() -> None:
    assert deduplicate_strings(["a", "b", "a", "c"]) == ["a", "b", "c"]


def test_serialize_metadata_returns_json_string() -> None:
    assert serialize_metadata({"strategy": "hierarchical"}).startswith("{")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/shared/rag/test_rag_strategies.py -v`
Expected: FAIL with import errors because `app.shared.rag.strategies` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def deduplicate_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/shared/rag/test_rag_strategies.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/shared/rag/test_rag_strategies.py src/app/shared/rag/strategies.py
git commit -m "test: add rag strategy helper coverage"
```

### Task 2: Implement consolidated reusable RAG module

**Files:**
- Create: `src/app/shared/rag/strategies.py`
- Modify: `src/app/shared/rag/__init__.py`
- Test: `tests/unit/shared/rag/test_rag_strategies.py`

- [ ] **Step 1: Write the failing test**

```python
from types import SimpleNamespace

from app.shared.rag.strategies import RAGStrategyService


def test_service_stores_app_scoped_dependencies() -> None:
    app = SimpleNamespace(state=SimpleNamespace(db_engine="engine", db_session_local="session"))
    service = RAGStrategyService.from_app(app)
    assert service.engine == "engine"
    assert service.session_local == "session"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/shared/rag/test_rag_strategies.py::test_service_stores_app_scoped_dependencies -v`
Expected: FAIL because `RAGStrategyService` is missing.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(slots=True)
class RAGStrategyService:
    engine: Any
    session_local: Any

    @classmethod
    def from_app(cls, app: Any) -> "RAGStrategyService":
        return cls(
            engine=app.state.db_engine,
            session_local=app.state.db_session_local,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/shared/rag/test_rag_strategies.py::test_service_stores_app_scoped_dependencies -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/shared/rag/strategies.py src/app/shared/rag/__init__.py tests/unit/shared/rag/test_rag_strategies.py
git commit -m "feat: add consolidated reusable rag strategies module"
```

### Task 3: Merge folder docs and remove numbered demos

**Files:**
- Modify: `src/app/shared/rag/README.md`
- Delete: `src/app/shared/rag/All kinds of RAG.md`
- Delete: `src/app/shared/rag/01_reranking.py`
- Delete: `src/app/shared/rag/02_agentic_rag.py`
- Delete: `src/app/shared/rag/03_knowledge_graphs.py`
- Delete: `src/app/shared/rag/04_contextual_retrieval.py`
- Delete: `src/app/shared/rag/05_query_expansion.py`
- Delete: `src/app/shared/rag/06_multi_query_rag.py`
- Delete: `src/app/shared/rag/07_context_aware_chunking.py`
- Delete: `src/app/shared/rag/08_late_chunking.py`
- Delete: `src/app/shared/rag/09_hierarchical_rag.py`
- Delete: `src/app/shared/rag/10_self_reflective_rag.py`
- Delete: `src/app/shared/rag/11_fine_tuned_embeddings.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_merged_rag_readme_mentions_strategies_module() -> None:
    readme = Path("src/app/shared/rag/README.md").read_text()
    assert "strategies.py" in readme
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/shared/rag/test_rag_strategies.py::test_merged_rag_readme_mentions_strategies_module -v`
Expected: FAIL because the current README still documents the numbered demos.

- [ ] **Step 3: Write minimal implementation**

```markdown
## Module Layout

- `strategies.py`: reusable ingestion, retrieval, and answer helpers for advanced RAG strategies
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/shared/rag/test_rag_strategies.py::test_merged_rag_readme_mentions_strategies_module -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/shared/rag/README.md tests/unit/shared/rag/test_rag_strategies.py
git rm src/app/shared/rag/All kinds of RAG.md src/app/shared/rag/01_reranking.py src/app/shared/rag/02_agentic_rag.py src/app/shared/rag/03_knowledge_graphs.py src/app/shared/rag/04_contextual_retrieval.py src/app/shared/rag/05_query_expansion.py src/app/shared/rag/06_multi_query_rag.py src/app/shared/rag/07_context_aware_chunking.py src/app/shared/rag/08_late_chunking.py src/app/shared/rag/09_hierarchical_rag.py src/app/shared/rag/10_self_reflective_rag.py src/app/shared/rag/11_fine_tuned_embeddings.py
git commit -m "refactor: merge rag strategy docs and remove demo files"
```
