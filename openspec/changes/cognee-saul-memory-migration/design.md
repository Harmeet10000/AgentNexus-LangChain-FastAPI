## Context

Agent Saul currently mixes several memory concepts: Graphiti-based knowledge helpers, Cognee scaffolding, and a `persist_memory` node that only appends refs instead of writing memory. The repository also has a separate Graphiti reconciliation workflow for entity deduplication. The target design is to make Cognee the primary memory/recall layer for Saul, while Graphiti remains the structural knowledge-base layer.

The chosen operating model is direct write plus async maintenance only:
- the graph writes the approved final report directly to Cognee after human approval
- Celery handles curation, decay, promotion, and reconciliation after the run
- the read path is hybrid, but Cognee-first

## Goals / Non-Goals

**Goals:**
- Make Cognee the authoritative memory store for Saul recall
- Persist only approved final reports, curated observations, and user preferences into Cognee
- Keep Graphiti for KB extraction and relationship storage only
- Add a post-`qna` prefetch stage and selected deeper retrieval tools
- Make async maintenance isolated, idempotent, and Celery-based
- Split Cognee reconciliation from Graphiti reconciliation

**Non-Goals:**
- Replacing Graphiti entirely
- Making memory retrieval mandatory for task success
- Exposing memory retrieval broadly to all Saul nodes
- Turning Cognee into the legal KB system of record
- Rewriting unrelated agent workflows

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

5. **Celery-only async maintenance**
   - Why: curation, decay, and reconciliation need retries, idempotency, and scheduling.
   - Alternatives considered:
     - FastAPI background tasks: too weak for operational maintenance
     - hybrid fallback tasks: unnecessary complexity for the target design

6. **Separate Cognee reconciliation workflow**
   - Why: Cognee memory semantics differ from Graphiti entity reconciliation.
   - Alternatives considered:
     - reuse Graphiti reconciliation: couples unrelated semantics
     - skip reconciliation: increases memory drift over time

## Risks / Trade-offs

- [Two memory systems] → Mitigate with explicit ownership: Graphiti for KB, Cognee for recall.
- [More moving parts] → Mitigate by keeping the read path simple and async maintenance isolated.
- [Memory drift] → Mitigate with write gating, curation rules, and idempotent reconciliation jobs.
- [Latency from memory retrieval] → Mitigate by keeping prefetch small and fail-open.
- [Implementation churn in Saul graph] → Mitigate by making targeted node-level changes instead of broad rewrites.

## Migration Plan

1. Create a real Cognee memory service abstraction and wire it into app startup.
2. Replace the Saul `persist_memory` stub with a direct Cognee write for approved final reports.
3. Add post-`qna` memory prefetch and scoped retrieval tools.
4. Add Celery jobs for Cognee curation, decay, and promotion.
5. Add a separate Cognee reconciliation workflow and schedule it.
6. Remove any Saul final-report Graphiti write path.

## Open Questions

- None.
