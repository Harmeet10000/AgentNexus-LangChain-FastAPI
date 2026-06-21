## 1. OpenSpec and Memory Boundary Cleanup

- [ ] 1.1 Confirm the change scope and artifact names in the new OpenSpec directory
- [ ] 1.2 Remove any Saul final-report Graphiti write path from the implementation plan
- [ ] 1.3 Document the Cognee vs Graphiti ownership split in the repo plan docs

## 2. Cognee Direct Write Path

- [ ] 2.1 Add a real Cognee memory service facade for Saul writes and reads
- [ ] 2.2 Wire `persist_memory` to write the approved final report directly to Cognee
- [ ] 2.3 Ensure `persist_memory` no longer invokes any Graphiti final-report write helper
- [ ] 2.4 Add tests proving only approved reports are persisted

## 3. Saul Memory Prefetch and Retrieval

- [ ] 3.1 Add a post-`qna` memory prefetch node
- [ ] 3.2 Make the prefetch Cognee-first with a limited Graphiti supplement
- [ ] 3.3 Expose deeper retrieval only to `risk_analysis` and `compliance`
- [ ] 3.4 Add fail-open behavior when memory retrieval is unavailable

## 4. Celery Maintenance Jobs

- [ ] 4.1 Add Celery tasks for Cognee curation, decay, and promotion
- [ ] 4.2 Make the tasks idempotent by content/run identity
- [ ] 4.3 Add scheduled sweep support for maintenance and cleanup
- [ ] 4.4 Add tests for duplicate-delivery and retry safety

## 5. Cognee Reconciliation

- [ ] 5.1 Create a separate Cognee reconciliation workflow
- [ ] 5.2 Add duplicate-observation and preference-drift handling
- [ ] 5.3 Preserve approved final reports as immutable source artifacts
- [ ] 5.4 Add idempotency-key coverage for reconciliation jobs

## 6. Graphiti KB Cleanup

- [ ] 6.1 Remove final-report persistence from the Graphiti memory route
- [ ] 6.2 Keep Graphiti writes focused on knowledge-base extraction and relationships
- [ ] 6.3 Add tests proving Saul final reports no longer write to Graphiti

## 7. Verification

- [ ] 7.1 Run targeted unit tests for Saul memory write and prefetch paths
- [ ] 7.2 Run worker and reconciliation tests for idempotency and decay
- [ ] 7.3 Run `ruff check` and `ty check`
