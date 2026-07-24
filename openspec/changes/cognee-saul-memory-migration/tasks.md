## 1. Cognee Direct Write Path

- [ ] 1.1 Add a real Cognee memory service facade for Saul writes and reads
- [ ] 1.2 Wire `persist_memory` to write the approved final report directly to Cognee
- [ ] 1.3 Ensure `persist_memory` no longer invokes any Graphiti final-report write helper
- [ ] 1.4 Add tests proving only approved reports are persisted

## 2. Saul Memory Prefetch and Retrieval

- [ ] 2.1 Add a post-`qna` memory prefetch node
- [ ] 2.2 Make the prefetch Cognee-first with a limited Graphiti supplement
- [ ] 2.3 Expose deeper retrieval only to `risk_analysis` and `compliance`
- [ ] 2.4 Add fail-open behavior when memory retrieval is unavailable

## 3. Graphiti KB Cleanup

- [ ] 3.1 Remove final-report persistence from the Graphiti memory route
- [ ] 3.2 Keep Graphiti writes focused on knowledge-base extraction and relationships
- [ ] 3.3 Add tests proving Saul final reports no longer write to Graphiti

## 4. Verification

- [ ] 4.1 Run targeted unit tests for Saul memory write and prefetch paths
- [ ] 4.2 Run `ruff check` and `ty check`