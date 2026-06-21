## Why

Agent Saul currently has memory wiring split across Cognee scaffolding, Graphiti helpers, and a stubbed `persist_memory` path. The result is inconsistent ownership: final reports, curated observations, and retrieval context do not have a single clear system of record. We need to make Cognee the primary recall layer for Saul now, while keeping Graphiti focused on knowledge-base extraction and relationship storage.

## What Changes

- Make Cognee the primary long-term memory system for Agent Saul recall.
- Change `persist_memory` so it writes approved final reports directly to Cognee after human approval.
- Remove any Saul final-report write path to Graphiti.
- Add a post-`qna` memory prefetch step that uses Cognee first and Graphiti only as a small supplement for grounding.
- Expose deeper memory retrieval only to `risk_analysis` and `compliance`.
- Route Cognee maintenance, curation, and reconciliation through Celery-backed async jobs.
- Add a separate Cognee reconciliation workflow for duplicated observations, preference promotion, and decay/cleanup.

## Capabilities

### New Capabilities
- `saul-cognee-final-report-write`: Directly persist approved final reports to Cognee from the Saul graph.
- `saul-memory-prefetch-and-retrieval`: Load Cognee-first recall after `qna` and expose deeper retrieval only to selected reasoning nodes.
- `saul-cognee-maintenance-worker`: Run post-run curation, decay, and promotion jobs in Celery.
- `saul-cognee-reconciliation`: Reconcile Cognee memory separately from the existing Graphiti/entity reconciliation workflow.

### Modified Capabilities
- None.

## Impact

- `src/app/shared/langgraph_layer/agent_saul/*`
- `src/app/shared/langchain_layer/agents/memory/*`
- `src/app/shared/rag/graphiti/*`
- `src/tasks/*`
- `src/app/lifecycle/lifespan.py`
- `src/app/features/agent_saul/service.py`
- Celery scheduling and worker registration
- Cognee memory namespace and persistence behavior
