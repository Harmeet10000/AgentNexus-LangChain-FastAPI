## Context

Agent Saul currently mixes several memory concepts: Graphiti-based knowledge helpers, Cognee scaffolding, and a `persist_memory` node that only appends refs instead of writing memory. The repository also has a separate Graphiti reconciliation workflow for entity deduplication. The target design is to make Cognee the primary memory/recall layer for Saul, while Graphiti remains the structural knowledge-base layer.

The chosen operating model is direct write plus async maintenance only:
- the graph writes the approved final report directly to Cognee after human approval
- the read path is hybrid, but Cognee-first

Celery-based maintenance (curation, decay, promotion, reconciliation) is deferred. Cognee v1.1 has no built-in dedup/decay/reconciliation, but memory drift is acceptable at v1 scale. Add maintenance workers only when actual duplication or drift is observed.

## Goals / Non-Goals

**Goals:**
- Make Cognee the authoritative memory store for Saul recall
- Persist only approved final reports into Cognee
- Keep Graphiti for KB extraction and relationship storage only
- Add a post-`qna` prefetch stage and selected deeper retrieval tools
- Remove Saul final-report write path from Graphiti

**Non-Goals:**
- Replacing Graphiti entirely
- Making memory retrieval mandatory for task success
- Exposing memory retrieval broadly to all Saul nodes
- Turning Cognee into the legal KB system of record
- Rewriting unrelated agent workflows
- Celery-based maintenance, curation, decay, promotion, or reconciliation (deferred until drift is observed)

## Decisions

1. **Direct write in `persist_memory` for approved final reports**
   - Why: the final approved report is the highest-signal artifact and should be captured while Saul still has the freshest state.
   - Alternatives considered:
     - enqueue-only write: simpler worker isolation, but loses immediacy
     - worker-only write: too indirect for the highest-value artifact

2. **Cognee-only final report persistence**
   - Why: the user-defined boundary is that Graphiti is for knowledge-base extraction and Cognee is for recall of what happened.
   - Alternatives considered:
     - dual-write to Graphiti and Cognee: violates ownership split
     - Graphiti-only final reports: conflicts with the chosen memory boundary

3. **Hybrid retrieval with Cognee-first prefetch after `qna`**
   - Why: `qna` gives enough semantic shape to retrieve useful memory without over-fetching.
   - Alternatives considered:
     - prefetch before `qna`: too noisy
     - global retrieval at graph start: too broad and expensive

4. **Deeper retrieval only in `risk_analysis` and `compliance`**
   - Why: those nodes benefit most from task-specific memory context.
   - Alternatives considered:
     - all reasoning nodes: too much retrieval authority
     - orchestrator access: blurs routing and reasoning responsibilities

5. **No Celery maintenance or reconciliation in v1**
   - Why: Cognee v1.1 has no built-in dedup/decay/reconciliation. Adding Celery workers for curation, decay, promotion, and reconciliation is the heaviest part of the original plan. Memory drift is acceptable at v1 scale. Add maintenance only when actual duplication or drift is observed.
   - Alternatives considered:
     - full Celery maintenance suite: too heavy for v1, YAGNI
     - skip entirely: acceptable — Cognee accumulates but drift is tolerable at current scale

## Risks / Trade-offs

- [Two memory systems] → Mitigate with explicit ownership: Graphiti for KB, Cognee for recall.
- [Memory drift over time] → Accepted. Cognee v1.1 has no built-in dedup/decay. Add Celery maintenance workers when drift is actually observed, not speculatively.
- [Latency from memory retrieval] → Mitigate by keeping prefetch small and fail-open.
- [Implementation churn in Saul graph] → Mitigate by making targeted node-level changes instead of broad rewrites.

## Migration Plan

1. Create a real Cognee memory service abstraction and wire it into app startup.
2. Replace the Saul `persist_memory` stub with a direct Cognee write for approved final reports.
3. Add post-`qna` memory prefetch and scoped retrieval tools.
4. Remove any Saul final-report Graphiti write path.

## Open Questions

- None.