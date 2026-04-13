# Reconciliation LangGraph Package Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the shared LangGraph reconciliation package, split its monolithic pipeline module into focused files, and expose a clean package API consistent with `ingestion_kb`.

**Architecture:** Keep runtime behavior in node factory functions, move prompt text and state schemas into dedicated modules, and isolate graph wiring in a small builder module. Preserve public construction helpers through package-level re-exports so callers import from the package instead of file internals.

**Tech Stack:** Python 3.12, LangGraph, LangChain Core, Pydantic v2, SQLAlchemy async

---

### Task 1: Rename And Split The Package

**Files:**
- Create: `src/app/shared/langgraph_layer/reconciliation/__init__.py`
- Create: `src/app/shared/langgraph_layer/reconciliation/state.py`
- Create: `src/app/shared/langgraph_layer/reconciliation/prompt.py`
- Create: `src/app/shared/langgraph_layer/reconciliation/graph.py`
- Modify: `src/app/shared/langgraph_layer/reconciliation/pipeline_node.py`
- Modify: `src/app/shared/langgraph_layer/ingestion_kb/__init__.py`
- Delete: `src/app/shared/langgraph_layer/reconsiliation/pipeline_node.py`

- [ ] **Step 1: Inspect the current package boundaries**

Run: `find src/app/shared/langgraph_layer/ingestion_kb src/app/shared/langgraph_layer/reconsiliation -maxdepth 2 -type f | sort`
Expected: the ingestion package already contains `state.py`, `prompt.py`, `graph.py`, and `pipeline_node.py`, while reconciliation is still monolithic.

- [ ] **Step 2: Create the new typed state module**

```python
class ReconciliationState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = ""
    run_id: str = ""
    lookback_hours: int = 24
```

- [ ] **Step 3: Move the reconciliation prompt to its own module**

```python
reconcile_prompt = """
You are a memory reconciliation system for a legal knowledge graph.
...
"""
```

- [ ] **Step 4: Reduce `pipeline_node.py` to node factories and helpers**

```python
def make_reconcile_node(
    reconcile_llm: ReconciliationRunnable,
) -> Callable[[ReconciliationState], Awaitable[dict[str, object]]]:
    async def reconcile_node(state: ReconciliationState) -> dict[str, object]:
        ...
```

- [ ] **Step 5: Move graph assembly into `graph.py`**

```python
def build_reconciliation_graph(
    reconcile_llm: ReconciliationRunnable,
    db_engine: AsyncEngine,
) -> CompiledStateGraph:
    graph = StateGraph(ReconciliationState)
    ...
```

- [ ] **Step 6: Export the package surface through `__init__.py` files**

```python
from .graph import build_reconciliation_graph
from .pipeline_node import make_apply_changes_node
from .state import ReconciliationState
```

- [ ] **Step 7: Run targeted validation**

Run: `python -m compileall src/app/shared/langgraph_layer/reconciliation src/app/shared/langgraph_layer/ingestion_kb`
Expected: both packages compile without syntax errors.

- [ ] **Step 8: Review changed paths**

Run: `git diff -- src/app/shared/langgraph_layer/ingestion_kb src/app/shared/langgraph_layer/reconciliation`
Expected: diff shows the new package split and `__init__.py` exports only.
